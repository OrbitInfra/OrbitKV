#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr unsigned kThreads = 256;
constexpr unsigned kBlocks = 1024;

__global__ void fill(std::uint32_t* data, std::size_t words, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += gridDim.x * blockDim.x) {
        const auto index = static_cast<std::uint32_t>(i);
        data[i] = (index * 0x9e3779b9U) ^ (index >> 9) ^ seed;
    }
}

__host__ __device__ bool selected(std::size_t word, std::size_t payload,
                                std::size_t chunk, std::size_t stride) {
    return word / stride < payload / chunk && word % stride < chunk;
}

__global__ void verify_device(const std::uint32_t* data, std::size_t words,
                              std::uint32_t seed, std::size_t payload,
                              std::size_t chunk, std::size_t stride,
                              unsigned long long* result) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += gridDim.x * blockDim.x) {
        const auto index = static_cast<std::uint32_t>(i);
        const auto expected = selected(i, payload, chunk, stride)
                                  ? (index * 0x9e3779b9U) ^ (index >> 9) ^ seed
                                  : 0xa5a5a5a5U;
        if (data[i] != expected) {
            atomicAdd(result, 1ULL);
            atomicMin(result + 1, static_cast<unsigned long long>(i));
        }
    }
}

__global__ void corrupt(std::uint32_t* word) { *word ^= 1U; }
}  // namespace

extern "C" int tent_gpu_fill(void* data, std::size_t bytes, std::uint32_t seed) {
    if (!data || !bytes || bytes % 4) return cudaErrorInvalidValue;
    fill<<<kBlocks, kThreads>>>(static_cast<std::uint32_t*>(data), bytes / 4, seed);
    auto status = cudaGetLastError();
    return status == cudaSuccess ? cudaDeviceSynchronize() : status;
}

extern "C" int tent_gpu_corrupt(void* data, std::size_t word) {
    if (!data) return cudaErrorInvalidValue;
    corrupt<<<1, 1>>>(static_cast<std::uint32_t*>(data) + word);
    auto status = cudaGetLastError();
    return status == cudaSuccess ? cudaDeviceSynchronize() : status;
}

extern "C" int tent_gpu_pointer(void* data, int* device, int* kind) {
    if (!data || !device || !kind) return cudaErrorInvalidValue;
    cudaPointerAttributes info{};
    auto status = cudaPointerGetAttributes(&info, data);
    if (status == cudaSuccess) {
        *device = info.device;
        *kind = static_cast<int>(info.type);
    }
    return status;
}

extern "C" int tent_gpu_uuid(int device, void* output) {
    if (!output) return cudaErrorInvalidValue;
    cudaDeviceProp properties{};
    auto status = cudaGetDeviceProperties(&properties, device);
    if (status == cudaSuccess)
        std::memcpy(output, properties.uuid.bytes, sizeof(properties.uuid.bytes));
    return status;
}

// Both complete CPU comparison and actual GPU reads are outside transfer clocks.
extern "C" int tent_gpu_verify(void* data, std::size_t bytes, std::uint32_t seed,
                                std::size_t payload, std::size_t chunk,
                                std::size_t stride, void* scratch,
                                std::uint64_t* cpu_result,
                                std::uint64_t* gpu_result) {
    if (!data || !scratch || !cpu_result || !gpu_result || !bytes || bytes % 4 ||
        !payload || payload % 4 || !chunk || chunk % 4 || stride % 4 ||
        stride < chunk || payload % chunk || chunk > bytes ||
        (payload / chunk - 1) > (bytes - chunk) / stride)
        return cudaErrorInvalidValue;
    const unsigned long long initial[] = {0, std::numeric_limits<unsigned long long>::max()};
    auto status = cudaMemcpy(scratch, initial, sizeof(initial), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    verify_device<<<kBlocks, kThreads>>>(static_cast<std::uint32_t*>(data), bytes / 4,
                                       seed, payload / 4, chunk / 4, stride / 4,
                                       static_cast<unsigned long long*>(scratch));
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    status = cudaMemcpy(gpu_result, scratch, sizeof(initial), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    cpu_result[0] = 0;
    cpu_result[1] = std::numeric_limits<std::uint64_t>::max();
    try {
        std::vector<std::uint32_t> actual(bytes / 4);
        status = cudaMemcpy(actual.data(), data, bytes, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status;
        const auto record = [&](std::size_t i, std::uint32_t expected) {
            if (actual[i] == expected) return;
            if (!cpu_result[0]) cpu_result[1] = i;
            ++cpu_result[0];
        };
        for (std::size_t base = 0, row = 0; base < actual.size();
             base += stride / 4, ++row) {
            const auto selected_end = row < payload / chunk ? base + chunk / 4 : base;
            for (std::size_t i = base; i < selected_end; ++i) {
                const auto index = static_cast<std::uint32_t>(i);
                record(i, (index * 2654435769U) ^ (index / 512U) ^ seed);
            }
            const auto end = std::min(actual.size(), base + stride / 4);
            for (std::size_t i = selected_end; i < end; ++i) record(i, 0xa5a5a5a5U);
        }
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}
