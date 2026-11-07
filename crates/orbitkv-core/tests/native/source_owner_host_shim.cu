// CPU ownership control only. No CUDA registration or device operation occurs.
#include <atomic>
#include <chrono>
#include <cstdlib>
#ifdef ORBITKV_CPU_HOST_CONTROL
#include <cstddef>
// Match only the three runtime ABI functions below. This build is a CPU
// ownership control and must never be used as a CUDA qualification artifact.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
#else
#include <cuda_runtime_api.h>
#endif

static std::atomic<unsigned long long> registered{0}, unregistered{0};
static std::atomic<unsigned long long> unregister_ns{0};
extern "C" cudaError_t cudaHostRegister(void *, size_t, unsigned int) {
  ++registered;
  return cudaSuccess;
}
extern "C" cudaError_t cudaHostGetDevicePointer(void **device, void *host,
                                                unsigned int) {
  if (std::getenv("ORBITKV_TEST_IMPORT_ADDRESS_FAILURE"))
    return static_cast<cudaError_t>(1);
  *device = host;
  return cudaSuccess;
}
extern "C" cudaError_t cudaHostUnregister(void *) {
  if (std::getenv("ORBITKV_TEST_CUDART_UNLOADING"))
    return static_cast<cudaError_t>(4);
  if (std::getenv("ORBITKV_TEST_IMPORT_UNREGISTER_FAILURE"))
    return static_cast<cudaError_t>(1);
  ++unregistered;
  unregister_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
  return cudaSuccess;
}
extern "C" unsigned long long orbitkv_test_host_unregistered() {
  return unregistered.load();
}
extern "C" unsigned long long orbitkv_test_host_registered() {
  return registered.load();
}
extern "C" unsigned long long orbitkv_test_host_unregister_ns() {
  return unregister_ns.load();
}
