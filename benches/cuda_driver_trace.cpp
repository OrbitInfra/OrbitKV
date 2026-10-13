// Diagnostic-only CUPTI callbacks; no CUDA arguments, results or owners change.
#include <cupti.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr size_t kLimit = 262144;
struct Event {
    const char* name;
    uint64_t begin_ns;
    uint64_t end_ns;
    long tid;
    uint32_t context;
    uint32_t correlation;
    int result;
};
Event events[kLimit]{};
std::atomic<bool> published[kLimit]{};
std::atomic<size_t> count{0};
std::atomic<size_t> active{0};
std::atomic<bool> sealed{false};
std::atomic<bool> invalid{false};
pid_t owner_pid = 0;
CUpti_SubscriberHandle subscriber{};
bool enabled = false;

uint64_t monotonic_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        invalid.store(true, std::memory_order_relaxed);
        return 0;
    }
    return uint64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId,
                       const void* raw) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API || getpid() != owner_pid) return;
    const auto* data = static_cast<const CUpti_CallbackData*>(raw);
    if (!data || !data->correlationData) {
        invalid.store(true, std::memory_order_relaxed);
        return;
    }
    if (data->callbackSite == CUPTI_API_ENTER) {
        *data->correlationData = 0;
        if (sealed.load(std::memory_order_acquire)) return;
        active.fetch_add(1, std::memory_order_acq_rel);
        if (sealed.load(std::memory_order_acquire)) {
            active.fetch_sub(1, std::memory_order_release);
            return;
        }
        const size_t index = count.fetch_add(1, std::memory_order_relaxed);
        if (index >= kLimit || !data->functionName) {
            invalid.store(true, std::memory_order_relaxed);
            active.fetch_sub(1, std::memory_order_release);
            return;
        }
        events[index] = Event{data->functionName, monotonic_ns(), 0,
                              syscall(SYS_gettid), data->contextUid,
                              data->correlationId, -1};
        *data->correlationData = index + 1;
    } else if (data->callbackSite == CUPTI_API_EXIT) {
        const uint64_t token = *data->correlationData;
        if (!token) return;
        if (token > kLimit) {
            invalid.store(true, std::memory_order_relaxed);
            return;
        }
        const size_t index = token - 1;
        Event& event = events[index];
        event.end_ns = monotonic_ns();
        if (data->functionReturnValue) {
            event.result = *static_cast<const CUresult*>(data->functionReturnValue);
        } else {
            invalid.store(true, std::memory_order_relaxed);
        }
        if (event.end_ns < event.begin_ns || event.correlation != data->correlationId ||
            event.tid != syscall(SYS_gettid) || published[index].load()) {
            invalid.store(true, std::memory_order_relaxed);
        }
        *data->correlationData = 0;
        published[index].store(true, std::memory_order_release);
        active.fetch_sub(1, std::memory_order_release);
    } else {
        invalid.store(true, std::memory_order_relaxed);
    }
}

void json_string(FILE* output, const char* text) {
    fputc('"', output);
    for (const auto* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
        if (*p == '"' || *p == '\\') fputc('\\', output);
        if (*p < 32) fprintf(output, "\\u%04x", unsigned(*p));
        else fputc(*p, output);
    }
    fputc('"', output);
}

void write_events() {
    if (!enabled || getpid() != owner_pid) return;
    sealed.store(true, std::memory_order_release);
    const size_t total = count.load(std::memory_order_acquire);
    bool complete = active.load(std::memory_order_acquire) == 0 && total <= kLimit &&
                    !invalid.load(std::memory_order_acquire);
    // Function names are CUPTI constants. Serialize before unsubscribing.
    char path[4096];
    const int length = snprintf(path, sizeof(path), "%s/cuda-driver-%d.jsonl",
                                std::getenv("ORBITKV_CUDA_DRIVER_TRACE_DIR"), owner_pid);
    if (length < 0 || size_t(length) >= sizeof(path)) _exit(2);
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { perror("CUDA driver trace output"); _exit(2); }
    FILE* output = fdopen(fd, "w");
    if (!output) { close(fd); _exit(2); }
    fprintf(output, "{\"stage\":\"cuda_driver_open\",\"schema\":1,\"pid\":%d,"
                    "\"capacity\":%zu,\"clock\":\"CLOCK_MONOTONIC\","
                    "\"scope\":\"Process-scoped DRIVER_API callbacks; fork inheritance unsupported; diagnostic wall time\"}\n",
            owner_pid, kLimit);
    const size_t stored = total < kLimit ? total : kLimit;
    for (size_t index = 0; index < stored; ++index) {
        if (!published[index].load(std::memory_order_acquire)) {
            complete = false;
            continue;
        }
        const Event& event = events[index];
        fprintf(output, "{\"stage\":\"cuda_driver_api\",\"id\":%zu,\"name\":", index);
        json_string(output, event.name);
        fprintf(output, ",\"begin_ns\":%llu,\"end_ns\":%llu,\"tid\":%ld,"
                        "\"context\":%u,\"correlation\":%u,\"result\":%d}\n",
                (unsigned long long)event.begin_ns, (unsigned long long)event.end_ns,
                event.tid, event.context, event.correlation, event.result);
    }
    if (cuptiUnsubscribe(subscriber) != CUPTI_SUCCESS) complete = false;
    fprintf(output, "{\"stage\":\"cuda_driver_complete\",\"pid\":%d,"
                    "\"calls\":%zu,\"complete\":%s}\n", owner_pid, total,
            complete ? "true" : "false");
    const bool io_ok = !ferror(output) && fflush(output) == 0;
    const bool close_ok = fclose(output) == 0;
    if (!complete || !io_ok || !close_ok) _exit(2);
}

#ifndef ORBITKV_CUDA_TRACE_TEST
__attribute__((constructor)) void start() {
    const char* directory = std::getenv("ORBITKV_CUDA_DRIVER_TRACE_DIR");
    if (!directory || !*directory) return;
    owner_pid = getpid();
    if (cuptiSubscribe(&subscriber, callback, nullptr) != CUPTI_SUCCESS ||
        cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API) != CUPTI_SUCCESS) {
        fputs("CUDA driver trace subscription failed\n", stderr);
        _exit(2);
    }
    enabled = true;
}
__attribute__((destructor)) void stop() { write_events(); }
#endif
}  // namespace
