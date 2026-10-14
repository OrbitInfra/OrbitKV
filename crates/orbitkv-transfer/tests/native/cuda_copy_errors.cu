#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "tent/platform/cuda.h"

namespace {
thread_local int current_device = 7;
thread_local CUcontext current_context = reinterpret_cast<CUcontext>(0x777);
std::string failure;
std::atomic<unsigned> copies{0}, syncs{0}, switches{0};
std::array<unsigned char, 4096> source, target;
void *device_address = target.data();
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace
extern "C" CUresult CUDAAPI cuInit(unsigned int) {
  return failure == "driver-init" ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void *output,
                                                  CUpointer_attribute attribute,
                                                  CUdeviceptr ptr) {
  if (failure == "memory-type")
    return CUDA_ERROR_UNKNOWN;
  if (failure == "ordinal" && attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
    return CUDA_ERROR_INVALID_DEVICE;
  if (attribute == CU_POINTER_ATTRIBUTE_MEMORY_TYPE) {
    *static_cast<unsigned *>(output) =
        ptr == reinterpret_cast<CUdeviceptr>(device_address)
            ? CU_MEMORYTYPE_DEVICE
            : CU_MEMORYTYPE_HOST;
  } else if (attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL) {
    *static_cast<int *>(output) = 2;
  }
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
cuPointerGetAttributes(unsigned int count, CUpointer_attribute *attributes,
                       void **outputs, CUdeviceptr ptr) {
  if (failure == "memory-type")
    return CUDA_ERROR_UNKNOWN;
  if (failure == "ordinal")
    return CUDA_ERROR_INVALID_DEVICE;
  require(count == 2 && attributes[0] == CU_POINTER_ATTRIBUTE_MEMORY_TYPE &&
              attributes[1] == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
          "unexpected attributes query order");
  const bool device = ptr == reinterpret_cast<CUdeviceptr>(device_address) &&
                      failure != "host-only";
  *static_cast<unsigned *>(outputs[0]) =
      failure == "registered-host" ? CU_MEMORYTYPE_HOST
      : device ? ((failure == "managed" || failure == "managed-invalid-device")
                      ? CU_MEMORYTYPE_UNIFIED
                      : CU_MEMORYTYPE_DEVICE)
               : 0;
  *static_cast<int *>(outputs[1]) =
      (failure == "invalid-device" || failure == "managed-invalid-device") ? -1
                                                                           : 2;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context) {
  if (failure == "context-query")
    return CUDA_ERROR_INVALID_CONTEXT;
  *context = current_context;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxSetCurrent(CUcontext context) {
  if (failure == "restore" || failure == "set-device-restore")
    return CUDA_ERROR_INVALID_CONTEXT;
  current_context = context;
  current_device = context ? 7 : -1;
  return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  *device = current_device;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device) {
  ++switches;
  if (failure == "restore" && device == 7)
    return cudaErrorInvalidDevice;
  if (failure == "set-device" && device == 2)
    return cudaErrorInvalidDevice;
  current_device = device;
  current_context = reinterpret_cast<CUcontext>(device + 0x200);
  if ((failure == "set-device-mutated" || failure == "set-device-restore") &&
      device == 2)
    return cudaErrorInvalidDevice;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  *count = 8;
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void **, size_t) {
  return cudaErrorNotSupported;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void *) {
  return cudaErrorNotSupported;
}
extern "C" cudaError_t CUDARTAPI cudaGetLastError() { return cudaSuccess; }
extern "C" const char *CUDARTAPI cudaGetErrorString(cudaError_t) {
  return "controlled CUDA API error";
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                                 size_t length, cudaMemcpyKind,
                                                 cudaStream_t) {
  ++copies;
  if (failure == "copy")
    return cudaErrorInvalidValue;
  std::memcpy(dst, src, length);
  return cudaSuccess;
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t) {
  ++syncs;
  return failure == "synchronize" ? cudaErrorUnknown : cudaSuccess;
}
namespace mooncake::tent {
CUDAStreamHandle::~CUDAStreamHandle() {
  std::cout << "stream_owner_released" << std::endl;
}
Status CudaPlatform::getStreamFromPool(CUDAStreamHandle &, int device) {
  require(device == 2, "copy stream used wrong device");
  return failure == "stream" ? Status::CudaError("controlled stream error")
                             : Status::OK();
}
} // namespace mooncake::tent
int main(int argc, char **argv) {
  try {
    mooncake::tent::CudaPlatform platform(nullptr);
    if (argc > 1) {
      failure = argv[1];
      if (failure == "admission" || failure == "byte-admission" ||
          failure == "context-admission") {
        const bool byte_cap = failure == "byte-admission";
        const bool context_failure = failure == "context-admission";
        const size_t quantum = byte_cap ? 32 * 1024 * 1024 : 1;
        const unsigned calls = byte_cap ? 32 : 64;
        std::vector<unsigned char> payload(quantum), destination(quantum);
        device_address = destination.data();
        failure = context_failure ? "set-device-restore" : "copy";
        auto &entered = context_failure ? switches : copies;
        for (unsigned index = 0; index < calls; ++index) {
          std::thread([&] {
            (void)platform.copy(destination.data(), payload.data(), quantum);
          }).detach();
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(3);
          while (entered.load() <= index) {
            if (std::chrono::steady_clock::now() >= deadline)
              _exit(1);
            usleep(1000);
          }
        }
        auto result = platform.copy(destination.data(), payload.data(), 1);
        std::cout << "retained_admission=" << result.ToString()
                  << " byte_cap=" << byte_cap << std::endl;
        if (!result.IsTooManyRequests())
          _exit(1);
        if (context_failure && copies.load() != 0)
          _exit(1);
        // No DMA exists in this isolated interposed-API control. This process
        // exit is synthetic teardown, not a physical drain qualification.
        _exit(0);
      }
      if (failure == "repeated-errors") {
        failure = "stream";
        std::vector<unsigned char> payload(32 * 1024 * 1024),
            destination(payload.size());
        device_address = destination.data();
        for (int index = 0; index < 100; ++index) {
          auto result =
              platform.copy(destination.data(), payload.data(), payload.size());
          if (result.ok() || result.IsTooManyRequests())
            return 1;
        }
        failure = "normal";
        device_address = target.data();
        for (int index = 0; index < 100; ++index) {
          auto result =
              platform.copy(target.data(), source.data(), source.size());
          require(result.ok(), "pre-submit errors leaked admission");
        }
        return 0;
      }
      if (failure == "copy" || failure == "synchronize" ||
          failure == "restore" || failure == "set-device-restore") {
        struct ScratchOwner {
          ~ScratchOwner() { std::cout << "scratch_owner_dropped" << std::endl; }
        } owner;
        (void)platform.copy(target.data(), source.data(), source.size());
        std::cout << "unproven_copy_returned" << std::endl;
        return 1;
      }
      target.fill(0x5a);
      auto result = platform.copy(target.data(), source.data(), source.size());
      require(!result.ok() && copies == 0 && current_device == 7,
              "driver failure must reject before copy");
      std::cout << "driver-init status=" << result.ToString() << std::endl;
      return 0;
    }
    unsigned errors = 0;
    for (const std::string scenario :
         {"memory-type", "ordinal", "invalid-device", "context-query",
          "set-device", "set-device-mutated", "stream", "oversize", "host-only",
          "registered-host",
          "managed", "managed-invalid-device", "no-context", "normal"}) {
      failure = scenario;
      copies = syncs = switches = 0;
      current_device = 7;
      current_context = scenario == "no-context"
                            ? nullptr
                            : reinterpret_cast<CUcontext>(0x777);
      const auto original_context = current_context;
      for (size_t i = 0; i < source.size(); ++i)
        source[i] = (i * 17) % 251;
      target.fill(0x5a);
      auto result = platform.copy(target.data(), source.data(),
                                  scenario == "oversize" ? (size_t{1} << 30) + 1
                                                         : source.size());
      const bool rejected = !result.ok();
      const bool before_copy =
          scenario == "memory-type" || scenario == "ordinal" ||
          scenario == "invalid-device" ||
          scenario == "managed-invalid-device" || scenario == "context-query" ||
          scenario == "set-device" || scenario == "set-device-mutated" ||
          scenario == "stream" ||
          scenario == "oversize" || scenario == "copy";
      const bool sentinel = std::all_of(target.begin(), target.end(),
                                        [](auto c) { return c == 0x5a; });
      if ((scenario != "restore" && current_context != original_context) ||
          rejected != (scenario != "normal" && scenario != "host-only" &&
                       scenario != "registered-host" && scenario != "managed" &&
                       scenario != "no-context") ||
          (before_copy && !sentinel) ||
          ((scenario == "memory-type" || scenario == "ordinal") && copies))
        ++errors;
      std::cout << scenario << " status=" << result.ToString()
                << " device=" << current_device << " copies=" << copies
                << " sentinel=" << sentinel << std::endl;
    }
    std::cout << "CUDA_API_CONTROL_ERRORS=" << errors << std::endl;
    return errors ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
