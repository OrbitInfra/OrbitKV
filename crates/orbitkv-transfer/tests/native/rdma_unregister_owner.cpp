#include <array>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <string>

#include "tent/transport/rdma/buffers.h"
#include "tent/transport/rdma/context.h"
#include "tent/transport/rdma/rdma_transport.h"

using namespace mooncake::tent;

namespace {
std::array<ibv_mr*, 2> regions{};
std::array<bool, 2> owned{};
std::array<unsigned, 2> calls{};
ibv_mr* refused = nullptr;

int deregister(ibv_mr* region) {
    for (size_t index = 0; index < regions.size(); ++index) {
        if (region != regions[index]) continue;
        ++calls[index];
        if (!owned[index]) return EINVAL;
        if (region == refused) return EBUSY;
        owned[index] = false;
        return 0;
    }
    return EINVAL;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string scene = argv[1];
    if (scene != "normal" && scene != "context-refusal" &&
        scene != "buffer-refusal")
        return 2;
    // Private access is disabled only by the diagnostic compiler flag. No
    // production API, object layout or shared-library byte is changed.
    RdmaTransport transport;
    RdmaContext first(transport), second(transport);
    auto& symbols = const_cast<IbvSymbols&>(first.verbs_);
    const auto original = symbols.ibv_dereg_mr;
    symbols.ibv_dereg_mr = deregister;
    std::array<unsigned char, 4096> backing{};
    std::array<ibv_mr, 2> memory{};
    for (size_t index = 0; index < memory.size(); ++index) {
        memory[index].addr = backing.data();
        memory[index].length = backing.size();
        regions[index] = &memory[index];
        owned[index] = index == 0 || scene != "context-refusal";
    }
    first.status_ = second.status_ = RdmaContext::DEVICE_ENABLED;
    first.device_name_ = "cpu-provider-0";
    second.device_name_ = "cpu-provider-1";
    first.mr_set_.insert(regions[0]);
    if (owned[1]) second.mr_set_.insert(regions[1]);
    bool passed = false;
    unsigned test_only_reconciliations = 0;
    {
        LocalBufferManager manager;
        const AddressRange range(backing.data(), backing.size());
        if (scene != "context-refusal") {
            auto& entry = manager.buffer_list_[range];
            entry.mem_reg_map.emplace(&first, regions[0]);
            entry.mem_reg_map.emplace(&second, regions[1]);
        }
        BufferDesc descriptor{};
        descriptor.addr = reinterpret_cast<uint64_t>(backing.data());
        descriptor.length = backing.size();
        descriptor.lkey = {11, 12};
        descriptor.rkey = {21, 22};
        if (scene == "context-refusal") {
            refused = regions[0];
            const int status = first.unregisterMemReg(regions[0]);
            const bool retained = first.mr_set_.count(regions[0]) == 1;
            refused = nullptr;
            const int retry = first.unregisterMemReg(regions[0]);
            passed = status != 0 && retained && retry == 0 && !owned[0] &&
                     first.mr_set_.empty() && calls[0] == 2;
            std::cout << "status=" << status << " retained=" << retained
                      << " retry=" << retry << " provider_calls=" << calls[0]
                      << std::endl;
        } else {
            refused = scene == "buffer-refusal" ? regions[0] : nullptr;
            const auto status = manager.removeBuffer(descriptor);
            const auto found = manager.buffer_list_.find(range);
            const bool retained = found != manager.buffer_list_.end();
            const size_t pending =
                retained ? found->second.mem_reg_map.size() : 0;
            const bool failed_context_kept =
                first.mr_set_.count(regions[0]) == 1;
            const bool successful_context_empty = second.mr_set_.empty();
            const bool original_keys_kept =
                descriptor.lkey.size() == 2 && descriptor.rkey.size() == 2;
            refused = nullptr;
            const auto retry = manager.removeBuffer(descriptor);
            if (scene == "normal") {
                passed = status.ok() && retry.ok() && !retained && !owned[0] &&
                         !owned[1] && calls[0] == 1 && calls[1] == 1 &&
                         descriptor.rkey.empty();
            } else {
                passed = !status.ok() && retained && pending == 1 &&
                         failed_context_kept && successful_context_empty &&
                         original_keys_kept && retry.ok() &&
                         manager.buffer_list_.empty() && !owned[0] &&
                         !owned[1] && calls[0] == 2 && calls[1] == 1 &&
                         descriptor.lkey.empty() && descriptor.rkey.empty();
            }
            std::cout << "status_ok=" << status.ok() << " retained=" << retained
                      << " pending=" << pending << " retry_ok=" << retry.ok()
                      << " provider_calls=" << calls[0] << "," << calls[1]
                      << " owned=" << owned[0] << "," << owned[1] << std::endl;
        }
        std::ifstream mappings("/proc/self/maps");
        std::ofstream archived(argv[2]);
        archived << mappings.rdbuf();
        archived.close();
        if (!archived) passed = false;
        // All MRs above are test structs with fake provider flags. Reconcile
        // those flags directly if the historical code lost them; no driver MR
        // exists.
        for (auto& owner : owned) {
            if (owner) ++test_only_reconciliations;
            owner = false;
        }
        manager.buffer_list_.clear();
        first.mr_set_.clear();
        second.mr_set_.clear();
    }
    first.status_ = second.status_ = RdmaContext::DEVICE_UNINIT;
    symbols.ibv_dereg_mr = original;
    std::cout << "scene=" << scene << " strict_contract_passed=" << passed
              << " test_only_reconciliations=" << test_only_reconciliations
              << " device_registration_calls=0 payload_DMA=false" << std::endl;
    return passed ? 0 : 1;
}
