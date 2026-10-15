// CPU ownership control only. No CUDA registration or device operation occurs.
#include <atomic>
#include <chrono>
#include <cuda_runtime_api.h>

static std::atomic<unsigned long long> registered{0}, unregistered{0};
static std::atomic<unsigned long long> unregister_ns{0};
extern "C" cudaError_t cudaHostRegister(void *, size_t, unsigned int) {
  ++registered;
  return cudaSuccess;
}
extern "C" cudaError_t cudaHostGetDevicePointer(void **device, void *host,
                                                unsigned int) {
  *device = host;
  return cudaSuccess;
}
extern "C" cudaError_t cudaHostUnregister(void *) {
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
