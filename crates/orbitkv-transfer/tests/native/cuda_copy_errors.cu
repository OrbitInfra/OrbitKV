#include <algorithm>
#include <array>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include "tent/platform/cuda.h"

namespace {
int current_device = 7;
bool context_active = true;
std::string failure;
unsigned copies = 0, syncs = 0, switches = 0;
std::array<unsigned char, 4096> source, target;
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
        ptr == reinterpret_cast<CUdeviceptr>(target.data())
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
  require(count == 2, "unexpected attributes query");
  const bool device = ptr == reinterpret_cast<CUdeviceptr>(target.data()) &&
                      failure != "host-only";
  *static_cast<unsigned *>(outputs[0]) = device ? CU_MEMORYTYPE_DEVICE : 0;
  *static_cast<int *>(outputs[1]) = failure == "invalid-device" ? -1 : 2;
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext *context) {
  if (failure == "context-query")
    return CUDA_ERROR_INVALID_CONTEXT;
  *context = context_active ? reinterpret_cast<CUcontext>(1) : nullptr;
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
      if (failure == "admission") {
        failure = "copy";
        for (int index = 0; index < 64; ++index) {
          std::thread([&] {
            (void)platform.copy(target.data(), source.data(), 1);
          }).detach();
          usleep(10000);
        }
        auto result = platform.copy(target.data(), source.data(), 1);
        std::cout << "retained_admission=" << result.ToString() << std::endl;
        if (!result.IsTooManyRequests())
          _exit(1);
        // Synthetic APIs never submitted DMA; process exit stops only this
        // isolated control. It is not graceful cleanup or real drain evidence.
        _exit(0);
      }
      if (failure == "copy" || failure == "synchronize") {
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
          "set-device", "stream", "restore", "oversize", "host-only",
          "normal"}) {
      failure = scenario;
      copies = syncs = switches = 0;
      current_device = 7;
      context_active = true;
      for (size_t i = 0; i < source.size(); ++i)
        source[i] = (i * 17) % 251;
      target.fill(0x5a);
      auto result = platform.copy(target.data(), source.data(),
                                  scenario == "oversize" ? (size_t{1} << 30) + 1
                                                         : source.size());
      const bool rejected = !result.ok();
      const bool before_copy =
          scenario == "memory-type" || scenario == "ordinal" ||
          scenario == "invalid-device" || scenario == "context-query" ||
          scenario == "set-device" || scenario == "stream" ||
          scenario == "oversize" || scenario == "copy";
      const bool sentinel = std::all_of(target.begin(), target.end(),
                                        [](auto c) { return c == 0x5a; });
      if ((scenario != "restore" && current_device != 7) ||
          rejected != (scenario != "normal" && scenario != "host-only") ||
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
