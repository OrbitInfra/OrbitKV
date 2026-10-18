// Benchmark-only interposition for Mooncake 71973589 on Linux x86-64/libstdc++.
// No transport setting, descriptor, return value or ownership is changed.
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "tent/common/status.h"

namespace mooncake::tent {
struct BootstrapDesc;
class RdmaContext;
struct EndPointParams;
}  // namespace mooncake::tent

using mooncake::tent::BootstrapDesc;
using mooncake::tent::EndPointParams;
using mooncake::tent::RdmaContext;
using mooncake::tent::Status;

namespace {
constexpr size_t kLimit = 4096;
struct Event {
    const char* stage;
    uint64_t begin_ns;
    uint64_t end_ns;
    long tid;
    int status;
};
Event events[kLimit];
std::atomic<size_t> count{0};
std::atomic<bool> published[kLimit]{};
constexpr uint64_t kSealed = uint64_t{1} << 63;
std::atomic<uint64_t> active_spans{0};

pid_t trace_owner() {
    static const pid_t owner = [] {
        const char* path = std::getenv("TENT_NATIVE_TRACE_PATH");
        if (!path) return pid_t{0};
        const char* text = std::getenv("TENT_NATIVE_TRACE_PID");
        if (!*path || !text || *text < '1' || *text > '9') {
            dprintf(STDERR_FILENO, "TENT trace requires its owner PID before exec\n");
            _exit(2);
        }
        pid_t value = 0;
        for (; *text; ++text) {
            if (*text < '0' || *text > '9' ||
                value > (std::numeric_limits<pid_t>::max() - (*text - '0')) / 10) {
                dprintf(STDERR_FILENO, "TENT trace has an invalid owner PID\n");
                _exit(2);
            }
            value = value * 10 + (*text - '0');
        }
        return value;
    }();
    return owner;
}

__attribute__((constructor)) void validate_owner() { (void)trace_owner(); }

bool owns_trace() { return trace_owner() != 0 && getpid() == trace_owner(); }

uint64_t now() {
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000 + value.tv_nsec;
}

template <typename Function>
Function original(const char* symbol) {
    auto address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        dprintf(STDERR_FILENO, "TENT trace missing native symbol: %s\n", symbol);
        std::abort();
    }
    return reinterpret_cast<Function>(address);
}

class Span {
   public:
    explicit Span(const char* stage) : stage_(stage) {
        if (!owns_trace()) return;
        enabled_ = true;
        begin_ = now();
        tid_ = syscall(SYS_gettid);
        if (active_spans.fetch_add(1, std::memory_order_acq_rel) & kSealed) _exit(2);
    }
    ~Span() {
        if (!enabled_ || !owns_trace()) return;
        const uint64_t end = now();
        const size_t index = count.fetch_add(1, std::memory_order_relaxed);
        if (index < kLimit) {
            events[index] = Event{stage_, begin_, end, tid_, status_};
            published[index].store(true, std::memory_order_release);
        }
        active_spans.fetch_sub(1, std::memory_order_release);
    }
    void finish(int status) { status_ = status; }

   private:
    const char* stage_;
    uint64_t begin_ = 0;
    long tid_ = 0;
    int status_ = -1;
    bool enabled_ = false;
};

__attribute__((destructor)) void write_events() {
    if (!owns_trace()) return;
    const char* path = std::getenv("TENT_NATIVE_TRACE_PATH");
    if (!path || !*path) _exit(2);
    if (active_spans.exchange(kSealed, std::memory_order_acq_rel) != 0) _exit(2);
    char boot_id[37]{};
    FILE* identity = std::fopen("/proc/sys/kernel/random/boot_id", "r");
    if (!identity) _exit(2);
    const int boot_fields = std::fscanf(identity, "%36[0-9a-f-]", boot_id);
    if (std::fclose(identity) != 0 || boot_fields != 1 || std::strlen(boot_id) != 36) _exit(2);
    char stat[4096]{};
    identity = std::fopen("/proc/self/stat", "r");
    if (!identity) _exit(2);
    const bool read_stat = std::fgets(stat, sizeof(stat), identity) != nullptr;
    if (std::fclose(identity) != 0 || !read_stat) _exit(2);
    char* field = std::strrchr(stat, ')');
    if (!field || field[1] != ' ') _exit(2);
    field += 2;
    for (size_t i = 0; i < 19; ++i) {
        field = std::strchr(field, ' ');
        if (!field) _exit(2);
        ++field;
    }
    char* end = nullptr;
    errno = 0;
    const auto start_ticks = std::strtoull(field, &end, 10);
    if (errno || !start_ticks || end == field || *end != ' ') _exit(2);
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        perror("TENT trace output");
        _exit(2);
    }
    FILE* output = fdopen(fd, "w");
    if (!output) {
        close(fd);
        _exit(2);
    }
    const size_t seen = count.load(std::memory_order_acquire);
    const size_t retained = seen < kLimit ? seen : kLimit;
    std::fprintf(output,
                 "{\"schema\":\"tent.native-trace.v1\",\"pid\":%ld,"
                 "\"boot_id\":\"%s\",\"start_ticks\":%llu,"
                 "\"clock\":\"CLOCK_MONOTONIC\",\"seen\":%zu,"
                 "\"limit\":%zu,\"overflow\":%s,\"events\":[",
                 static_cast<long>(getpid()), boot_id, start_ticks, seen, kLimit,
                 seen > kLimit ? "true" : "false");
    for (size_t i = 0; i < retained; ++i) {
        if (!published[i].load(std::memory_order_acquire)) _exit(2);
        const auto& event = events[i];
        std::fprintf(output,
                     "%s{\"stage\":\"%s\",\"begin_mono_ns\":%llu,"
                     "\"end_mono_ns\":%llu,\"tid\":%ld,\"status\":%d}",
                     i ? "," : "", event.stage, static_cast<unsigned long long>(event.begin_ns),
                     static_cast<unsigned long long>(event.end_ns), event.tid, event.status);
    }
    std::fprintf(output, "]}\n");
    const bool failed = std::ferror(output);
    if (std::fclose(output) != 0 || failed) _exit(2);
}
}  // namespace

// These symbol names and nontrivial Status return ABI are pinned, not a public
// Mooncake API. Use its exact status header and check exports before deployment.
extern "C" Status trace_metadata(const std::string&, std::string&) asm(
    "_ZN8mooncake4tent13ControlClient14getSegmentDescERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEERS7_");
extern "C" Status trace_metadata(const std::string& address, std::string& response) {
    using Function = Status (*)(const std::string&, std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent13ControlClient14getSegmentDescERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEERS7_");
    Span span("metadata_rpc");
    auto status = function(address, response);
    span.finish(static_cast<int>(status.code()));
    return status;
}

extern "C" Status trace_bootstrap(const std::string&, const BootstrapDesc&, BootstrapDesc&) asm(
    "_ZN8mooncake4tent13ControlClient9bootstrapERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEERKNS0_13BootstrapDescERSA_");
extern "C" Status trace_bootstrap(const std::string& address, const BootstrapDesc& request,
                                  BootstrapDesc& response) {
    using Function = Status (*)(const std::string&, const BootstrapDesc&, BootstrapDesc&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent13ControlClient9bootstrapERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEERKNS0_13BootstrapDescERSA_");
    Span span("bootstrap_rpc");
    auto status = function(address, request, response);
    span.finish(static_cast<int>(status.code()));
    return status;
}

extern "C" int trace_construct(void*, RdmaContext*, EndPointParams*, const std::string&) asm(
    "_ZN8mooncake4tent12RdmaEndPoint9constructEPNS0_11RdmaContextEPNS0_14EndPointParamsERKNSt7__"
    "cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
extern "C" int trace_construct(void* self, RdmaContext* context, EndPointParams* params,
                               const std::string& name) {
    using Function = int (*)(void*, RdmaContext*, EndPointParams*, const std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent12RdmaEndPoint9constructEPNS0_11RdmaContextEPNS0_14EndPointParamsERKNSt7_"
        "_cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    Span span("endpoint_construct");
    const int status = function(self, context, params, name);
    span.finish(status);
    return status;
}

extern "C" Status
trace_connect(void*, const std::string&, const std::string&, const std::string&) asm(
    "_ZN8mooncake4tent12RdmaEndPoint7connectERKNSt7__cxx1112basic_stringIcSt11char_"
    "traitsIcESaIcEEES9_S9_");
extern "C" Status trace_connect(void* self, const std::string& server, const std::string& nic,
                                const std::string& rpc) {
    using Function = Status (*)(void*, const std::string&, const std::string&, const std::string&);
    static const auto function = original<Function>(
        "_ZN8mooncake4tent12RdmaEndPoint7connectERKNSt7__cxx1112basic_stringIcSt11char_"
        "traitsIcESaIcEEES9_S9_");
    Span span("endpoint_connect");
    auto status = function(self, server, nic, rpc);
    span.finish(static_cast<int>(status.code()));
    return status;
}
