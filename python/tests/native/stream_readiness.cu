#include <cuda.h>
#include <stdint.h>

extern "C" int orbitkv_test_stream_gate_version() { return 1; }

extern "C" int orbitkv_test_stream_wait(uint64_t stream, uint64_t flag) {
  return static_cast<int>(cuStreamWaitValue32(
      reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(flag), 1,
      CU_STREAM_WAIT_VALUE_EQ));
}

extern "C" int orbitkv_test_stream_release(uint64_t stream, uint64_t flag) {
  return static_cast<int>(cuStreamWriteValue32(
      reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(flag), 1, 0));
}
