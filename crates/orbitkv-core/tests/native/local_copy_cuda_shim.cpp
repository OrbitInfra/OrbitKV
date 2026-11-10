// CPU control for the production OrbiTE local-copy owner. Never a GPU artifact.
// Only the documented Driver ABI used by this control is implemented. Unsupported
// IPC calls fail explicitly. Queued operations run when an event is progressed.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

using Address = std::uint64_t;
struct Stream { std::deque<std::function<void()>> work; };
struct Event { Stream *stream = nullptr; };
struct Copy2D {
  size_t srcX, srcY; int srcType; const void *srcHost; Address srcDevice;
  void *srcArray; size_t srcPitch;
  size_t dstX, dstY; int dstType; void *dstHost; Address dstDevice;
  void *dstArray; size_t dstPitch; size_t width, height;
};
static std::mutex mutex;
static std::map<Address, size_t> allocations;
static thread_local void *current = nullptr;
static thread_local std::vector<void *> contexts;
static bool held = false;
static int query_error = 0, query_delay = 0, fail_after = -1;
static unsigned long counters[6] = {};
static void *primary = reinterpret_cast<void *>(0x1234);
static int progress(Stream *stream) {
  if (held) return 600;
  while (stream && !stream->work.empty()) {
    auto work = std::move(stream->work.front()); stream->work.pop_front(); work();
  }
  return 0;
}
static int enqueue(Stream *stream, std::function<void()> work) {
  if (fail_after == 0) { fail_after = -1; return 1; }
  if (fail_after > 0) --fail_after;
  if (!stream || stream->work.size() >= 65536) return 1;
  stream->work.push_back(std::move(work)); return 0;
}
extern "C" {
void orbitkv_test_cuda_hold(int value) { held = value; }
void orbitkv_test_cuda_query_error(int value) { query_error = value; }
void orbitkv_test_cuda_query_delay(int value) { query_delay = value; }
void orbitkv_test_cuda_fail_after(int value) { fail_after = value; }
unsigned long orbitkv_test_cuda_count(unsigned index) { return index < 6 ? counters[index] : 0; }
int cuInit(unsigned) { return 0; }
// NVRTC probes driver-owned cache tables; the CPU shim has no such tables.
int cuGetExportTable(const void **table, const void *) { *table = nullptr; return 801; }
int cuDriverGetVersion(int *version) { *version = 13000; return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return ordinal == 0 ? 0 : 101; }
int cuDeviceGetCount(int *count) { *count = 1; return 0; }
// Stable fake identity for Manager ownership tests; no hardware is enumerated.
int cuDeviceGetUuid_v2(void *uuid, int device) {
  if (device != 0) return 101;
  std::memset(uuid, 0x42, 16); return 0;
}
int cuDeviceGetAttribute(int *value, int attr, int) {
  *value = attr == 11 ? 2147483647 : (attr == 16 ? 8 : 0); return 0;
}
int cuDeviceGetPCIBusId(char *out, int size, int) { std::snprintf(out, size, "0000:00:00.0"); return 0; }
int cuDevicePrimaryCtxRetain(void **ctx, int) { *ctx = primary; return 0; }
int cuDevicePrimaryCtxRelease_v2(int) { return 0; }
int cuCtxGetCurrent(void **ctx) { *ctx = current; return 0; }
int cuCtxSetCurrent(void *ctx) { current = ctx; return 0; }
int cuCtxPushCurrent_v2(void *ctx) { contexts.push_back(current); current = ctx; return 0; }
int cuCtxPopCurrent_v2(void **ctx) {
  if (contexts.empty()) return 1;
  *ctx = current; current = contexts.back(); contexts.pop_back(); return 0;
}
int cuCtxSynchronize() { return held ? 600 : 0; }
int cuCtxGetDevice(int *device) { *device = 0; return 0; }
int cuMemAlloc_v2(Address *out, size_t size) {
  void *p = std::calloc(1, size); if (!p) return 2;
  *out = reinterpret_cast<Address>(p);
  std::lock_guard<std::mutex> lock(mutex); allocations[*out] = size; return 0;
}
int cuMemFree_v2(Address ptr) {
  std::lock_guard<std::mutex> lock(mutex); allocations.erase(ptr);
  std::free(reinterpret_cast<void *>(ptr)); ++counters[5]; return 0;
}
int cuMemsetD8_v2(Address ptr, unsigned char value, size_t size) { std::memset(reinterpret_cast<void *>(ptr), value, size); return 0; }
int cuMemsetD8Async(Address ptr, unsigned char value, size_t size, Stream *stream) {
  return enqueue(stream, [=] { std::memset(reinterpret_cast<void *>(ptr), value, size); });
}
int cuMemGetAddressRange_v2(Address *base, size_t *size, Address address) {
  std::lock_guard<std::mutex> lock(mutex);
  for (auto [start, length] : allocations) if (address >= start && address - start < length) { *base = start; *size = length; return 0; }
  return 1;
}
int cuPointerGetAttribute(void *out, int attr, Address address) {
  Address base; size_t size;
  if (cuMemGetAddressRange_v2(&base, &size, address)) return 1;
  switch (attr) {
    case 1: *static_cast<void **>(out) = primary; break;
    case 2: *static_cast<int *>(out) = 2; break;
    case 8: case 9: *static_cast<int *>(out) = 0; break;
    default: return 801;
  }
  return 0;
}
int cuPointerGetAttributes(unsigned count, int *attrs, void **outputs, Address address) {
  for (unsigned i = 0; i < count; ++i) { int r = cuPointerGetAttribute(outputs[i], attrs[i], address); if (r) return r; }
  return 0;
}
int cuPointerSetAttribute(const void *, int, Address) { return 0; }
int cuFlushGPUDirectRDMAWrites(int, int) { return 801; }
int cuStreamCreate(Stream **stream, unsigned) { *stream = new Stream; return 0; }
int cuStreamCreateWithPriority(Stream **stream, unsigned flags, int) { return cuStreamCreate(stream, flags); }
int cuStreamDestroy_v2(Stream *stream) { if (progress(stream)) return 600; delete stream; return 0; }
int cuStreamGetCtx(Stream *, void **ctx) { ++counters[0]; *ctx = primary; return 0; }
int cuStreamIsCapturing(Stream *, int *status) { *status = 0; return 0; }
int cuStreamSynchronize(Stream *stream) { return progress(stream); }
int cuStreamQuery(Stream *stream) { return progress(stream); }
int cuEventCreate(Event **event, unsigned) { *event = new Event; return 0; }
int cuEventDestroy_v2(Event *event) { delete event; return 0; }
int cuEventRecord(Event *event, Stream *stream) { event->stream = stream; ++counters[1]; return 0; }
int cuEventQuery(Event *event) {
  if (query_error) return query_error;
  if (query_delay > 0) { --query_delay; return 600; }
  return progress(event->stream);
}
int cuEventSynchronize(Event *event) { return cuEventQuery(event); }
int cuStreamWaitEvent(Stream *stream, Event *event, unsigned) {
  return enqueue(stream, [event] { progress(event->stream); });
}
int cuMemcpyHtoDAsync_v2(Address dst, const void *src, size_t size, Stream *stream) {
  ++counters[2]; return enqueue(stream, [=] { std::memcpy(reinterpret_cast<void *>(dst), src, size); });
}
int cuMemcpyDtoHAsync_v2(void *dst, Address src, size_t size, Stream *stream) {
  ++counters[2]; return enqueue(stream, [=] { std::memcpy(dst, reinterpret_cast<void *>(src), size); });
}
int cuMemcpyAsync(Address dst, Address src, size_t size, Stream *stream) { return cuMemcpyHtoDAsync_v2(dst, reinterpret_cast<void *>(src), size, stream); }
int cuMemcpy(Address dst, Address src, size_t size) { std::memmove(reinterpret_cast<void *>(dst), reinterpret_cast<void *>(src), size); return 0; }
int cuMemcpy2DAsync_v2(const Copy2D *desc, Stream *stream) {
  Copy2D d = *desc; ++counters[3];
  return enqueue(stream, [d] {
    auto src = d.srcType == 1 ? reinterpret_cast<Address>(d.srcHost) : d.srcDevice;
    auto dst = d.dstType == 1 ? reinterpret_cast<Address>(d.dstHost) : d.dstDevice;
    for (size_t row = 0; row < d.height; ++row)
      std::memcpy(reinterpret_cast<void *>(dst + (row + d.dstY) * d.dstPitch + d.dstX), reinterpret_cast<void *>(src + (row + d.srcY) * d.srcPitch + d.srcX), d.width);
  });
}
int cuMemcpyBatchAsync_v2(Address *dst, Address *src, size_t *sizes, size_t count, void *, size_t *, size_t, Stream *stream) {
  std::vector<Address> d(dst, dst + count), s(src, src + count);
  std::vector<size_t> n(sizes, sizes + count); ++counters[2];
  return enqueue(stream, [d, s, n] { for (size_t i = 0; i < n.size(); ++i) std::memcpy(reinterpret_cast<void *>(d[i]), reinterpret_cast<void *>(s[i]), n[i]); });
}
int cuMemHostAlloc(void **out, size_t size, unsigned) { *out = std::calloc(1, size); return *out ? 0 : 2; }
int cuMemFreeHost(void *ptr) { std::free(ptr); return 0; }
int cuMemHostRegister_v2(void *, size_t, unsigned) { return 0; }
int cuMemHostUnregister(void *) { return std::getenv("ORBITKV_TEST_IMPORT_UNREGISTER_FAILURE") ? 1 : 0; }
int cuMemHostGetDevicePointer_v2(Address *out, void *host, unsigned) {
  if (std::getenv("ORBITKV_TEST_IMPORT_ADDRESS_FAILURE")) return 1;
  *out = reinterpret_cast<Address>(host); return 0;
}
int cuModuleLoadData(void **module, const void *) { *module = reinterpret_cast<void *>(0x4567); return 0; }
int cuModuleGetFunction(void **function, void *, const char *name) {
  if (!std::strcmp(name, "orbite_batch_copy")) { *function = reinterpret_cast<void *>(0x5678); return 0; }
  const char *codec[] = {"fp8_encode_batch", "fp8_decode_batch", "turbo_encode_batch",
                        "turbo_decode_batch", "copy_batch", "crc32_parts", "crc32_finish"};
  for (size_t i = 0; i < 7; ++i) if (!std::strcmp(name, codec[i])) {
    *function = reinterpret_cast<void *>(0x6000 + i); return 0;
  }
  return 500;
}
int cuModuleUnload(void *) { return 0; }
int cuLaunchKernel(void *function, unsigned grid_x, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Stream *stream, void **args, void **) {
  // CRC fixture checks descriptor/receipt plumbing, not CUDA kernel execution.
  if (function == reinterpret_cast<void *>(0x6005)) return enqueue(stream, [] {});
  if (function == reinterpret_cast<void *>(0x6006)) {
    Address desc = *static_cast<Address *>(args[0]), out = *static_cast<Address *>(args[3]);
    int encode = *static_cast<int *>(args[4]);
    if (encode) return 801;
    return enqueue(stream, [=] {
      for (unsigned i = 0; i < grid_x; ++i) {
        const auto *words = reinterpret_cast<const Address *>(desc + i * 80);
        const auto *bytes = reinterpret_cast<const unsigned char *>(words[0]);
        std::uint32_t crc = 0xffffffff;
        for (Address j = 0; j < words[3]; ++j) {
          crc ^= bytes[j];
          for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
        }
        reinterpret_cast<std::uint32_t *>(out)[i] = crc ^ 0xffffffff;
      }
    });
  }
  if (function != reinterpret_cast<void *>(0x5678)) return 801;
  Address desc = *static_cast<Address *>(args[0]); int count = *static_cast<int *>(args[1]); ++counters[4];
  return enqueue(stream, [=] {
    auto p = reinterpret_cast<const Address *>(desc);
    for (int i = 0; i < count; ++i) std::memcpy(reinterpret_cast<void *>(p[3*i]), reinterpret_cast<const void *>(p[3*i+1]), p[3*i+2]);
  });
}
int cuIpcGetMemHandle(void *, Address) { return 801; }
int cuIpcOpenMemHandle_v2(Address *, void *, unsigned) { return 801; }
int cuIpcCloseMemHandle(Address) { return 801; }
int cuGetErrorName(int, const char **name) { *name = "CPU_CUDA_SHIM_ERROR"; return 0; }
int cuGetErrorString(int, const char **name) { *name = "CPU CUDA shim injected error"; return 0; }
}
