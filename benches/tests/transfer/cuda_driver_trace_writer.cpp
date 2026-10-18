// CPU-only callback/output controls; no CUDA initialization or payload submission.
#define ORBITKV_CUDA_TRACE_TEST
#include "../../transfer/native/cuda_driver_trace.cpp"

#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2 || !std::getenv("ORBITKV_CUDA_DRIVER_TRACE_DIR")) return 3;
    owner_pid = getpid();
    enabled = true;
    const CUresult result = CUDA_SUCCESS;
    uint64_t token = 0;
    CUpti_CallbackData data{};
    data.functionName = "cuMemcpyHtoDAsync_v2";
    data.correlationData = &token;
    data.correlationId = 1;
    data.functionReturnValue = const_cast<CUresult*>(&result);
    data.callbackSite = CUPTI_API_ENTER;
    callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    if (strcmp(argv[1], "unpaired") != 0) {
        data.callbackSite = CUPTI_API_EXIT;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    }
    if (strcmp(argv[1], "overflow") == 0) count.store(kLimit + 1);
    if (strcmp(argv[1], "duplicate") == 0) {
        token = 1;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    }
    if (strcmp(argv[1], "natural-duplicate") == 0 || strcmp(argv[1], "orphan-exit") == 0) {
        token = 0;
        if (strcmp(argv[1], "orphan-exit") == 0) data.correlationId = 2;
        data.callbackSite = CUPTI_API_EXIT;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    }
    if (strcmp(argv[1], "sealed-pair") == 0) {
        active.fetch_or(kSealed);
        data.callbackSite = CUPTI_API_ENTER;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
        data.callbackSite = CUPTI_API_EXIT;
        callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    }
    // The CPU control has no subscriber; success/failure is supplied by a stub.
    write_events();
    return 0;
}

extern "C" CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle) {
    return CUPTI_SUCCESS;
}
