// CPU-only loader/exec controls; the actual collector uses stub CUPTI entry points.
#include "../../transfer/native/cuda_driver_trace.cpp"

extern "C" CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle* handle,
                                               CUpti_CallbackFunc, void*) {
    *handle = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t{1});
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t, CUpti_SubscriberHandle,
                                                CUpti_CallbackDomain) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle) {
    return CUPTI_SUCCESS;
}
extern "C" void cpu_test_copy() {
    const CUresult result = CUDA_SUCCESS;
    uint64_t token = 0;
    CUpti_CallbackData data{};
    data.functionName = "cuMemcpyHtoDAsync_v2";
    data.correlationData = &token;
    data.correlationId = 1;
    data.functionReturnValue = const_cast<CUresult*>(&result);
    data.callbackSite = CUPTI_API_ENTER;
    callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
    data.callbackSite = CUPTI_API_EXIT;
    callback(nullptr, CUPTI_CB_DOMAIN_DRIVER_API, 0, &data);
}
