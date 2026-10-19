// Actual TENT engine bookkeeping with a test-only transport refusal. No payload
// transfer or device registration occurs; these stable host bytes stay alive.
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "tent/platform/cpu.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/segment_manager.h"
#include "tent/runtime/segment_registry.h"
#include "tent/runtime/transfer_engine_impl.h"
#include "tent/transfer_engine.h"

namespace mooncake::tent {
#ifndef ORBITKV_UNREGISTER_RUST_CONTROL
Platform& Platform::getLoader(std::shared_ptr<Config>) {
    static CpuPlatform platform(std::make_shared<Config>());
    return platform;
}
#endif

class RefusingTransport final : public Transport {
   public:
    bool refuse = false;
    bool owns = false;
    unsigned registrations = 0;
    unsigned removals = 0;
    std::unordered_set<uint64_t> addresses;

    Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions&) override {
        ++registrations;
        if (!addresses.insert(desc.addr).second)
            return Status::RdmaError("test transport still owns this address");
        owns = true;
        desc.transports.push_back(RDMA);
        return Status::OK();
    }

    Status removeMemoryBuffer(BufferDesc& desc) override {
        ++removals;
        if (refuse) return Status::RdmaError("controlled unregister refusal");
        if (!addresses.erase(desc.addr))
            return Status::RdmaError(
                "test transport already removed this address");
        owns = !addresses.empty();
        return Status::OK();
    }

    const char* getName() const override { return "unregister-owner-control"; }
};
}  // namespace mooncake::tent

int historicalControl(int argc, char** argv) {
    using namespace mooncake::tent;
    if (argc != 4) return 2;
    const std::string scenario = argv[1], expectation = argv[2];
    if ((scenario != "normal" && scenario != "refused") ||
        (expectation != "strict" && expectation != "reproduce"))
        return 2;
    auto config = std::make_shared<Config>();
    config->set("metadata_type", "p2p");
    config->set("metadata_servers", "");
    config->set("rpc_server_hostname", "127.0.0.1");
    config->set("rpc_server_port", "0");
    config->set("log_level", "warning");
    for (const auto* name : {"tcp", "shm", "rdma", "io_uring", "nvlink",
                             "mnnvl", "gds", "ascend_direct"})
        config->set(std::string("transports/") + name + "/enable", false);
    bool accepted = false;
    auto transport = std::make_shared<RefusingTransport>();
    std::array<unsigned char, 4096> bytes{};
    {
        TransferEngineImpl engine(config);
        if (!engine.available())
            throw std::runtime_error("CPU engine unavailable");
        engine.swapTransportForTest(RDMA, transport);
        std::ifstream mappings("/proc/self/maps");
        std::ofstream archived(argv[3]);
        archived << mappings.rdbuf();
        archived.close();
        if (!archived)
            throw std::runtime_error("actual mapping evidence failed");
        auto registered =
            engine.registerLocalMemory(bytes.data(), bytes.size());
        if (!registered.ok() || !transport->owns)
            throw std::runtime_error(
                "registration control did not establish ownership");
        transport->refuse = scenario == "refused";
        auto first = engine.unregisterLocalMemory(bytes.data(), bytes.size());
        const bool retained = transport->owns;
        transport->refuse = false;
        auto retry = engine.unregisterLocalMemory(bytes.data(), bytes.size());
        const bool retained_after_retry = transport->owns;
        const bool lost = first.ok() && retained && retry.ok() &&
                          retained_after_retry && transport->removals == 1;
        bool stale_owner_denies_reregister = false;
        bool reregister_ok = false;
        unsigned reregister_attempts = 0;
        if (scenario == "refused") {
            const unsigned prior = transport->registrations;
            auto registration_retry =
                engine.registerLocalMemory(bytes.data(), bytes.size());
            reregister_ok = registration_retry.ok();
            reregister_attempts = transport->registrations - prior;
            stale_owner_denies_reregister =
                transport->owns && reregister_attempts == 1;
        }
        accepted = scenario == "normal" ? first.ok() && !retained &&
                                              retry.ok() && !transport->owns
                   : expectation == "reproduce"
                       ? lost && stale_owner_denies_reregister && reregister_ok
                       : !first.ok() && retained && retry.ok() &&
                             !retained_after_retry;
        std::cout << "scenario=" << scenario << " unregister_ok=" << first.ok()
                  << " owner_after_first=" << retained
                  << " retry_ok=" << retry.ok()
                  << " owner_after_retry=" << retained_after_retry
                  << " remove_calls=" << transport->removals
                  << " lost_unregister_ownership=" << lost
                  << " live_transport_denies_reregister="
                  << stale_owner_denies_reregister
                  << " reregister_reported_ok=" << reregister_ok
                  << " reregister_calls=" << reregister_attempts << std::endl;
        // Reconcile only the injected transport's test owner. This bypass is
        // not a production drain receipt and does not qualify physical
        // termination.
        if (transport->owns) {
            BufferDesc descriptor{};
            descriptor.addr = reinterpret_cast<uint64_t>(bytes.data());
            auto status = transport->removeMemoryBuffer(descriptor);
            if (!status.ok() || transport->owns)
                throw std::runtime_error("test owner could not be reconciled");
        }
    }
    std::cout << "test_owner_reconciled=" << !transport->owns
              << " payload_DMA=false engine_scope_ended=true" << std::endl;
    return accepted ? 0 : 1;
}

namespace {
using namespace mooncake::tent;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

class PublicationRegistry final : public SegmentRegistry {
   public:
    bool refuse = false;
    unsigned puts = 0;
    Status getSegmentDesc(SegmentDescRef&, const std::string&) override {
        return Status::NotImplemented(
            "control does not resolve remote segments");
    }
    Status putSegmentDesc(SegmentDescRef&) override {
        ++puts;
        return refuse ? Status::InternalError("controlled publication refusal")
                      : Status::OK();
    }
    Status deleteSegmentDesc(const std::string&) override {
        return Status::OK();
    }
};

// Successful removal deliberately scrubs the argument. The next transport must
// still receive its original descriptor, including keys, on every retry.
class ScrubbingTransport final : public Transport {
   public:
    bool refuse = false;
    bool owns = false;
    unsigned removals = 0;
    Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions&) override {
        owns = true;
        desc.transports.push_back(TCP);
        desc.lkey = {11, 12};
        desc.rkey = {21, 22};
        return Status::OK();
    }
    Status removeMemoryBuffer(BufferDesc& desc) override {
        ++removals;
        if (!owns) return Status::RdmaError("completed transport called again");
        if (desc.lkey != std::vector<uint32_t>{11, 12} ||
            desc.rkey != std::vector<uint32_t>{21, 22})
            return Status::RdmaError("original transport keys were discarded");
        if (refuse)
            return Status::RdmaError("controlled second transport refusal");
        owns = false;
        desc.lkey.clear();
        desc.rkey.clear();
        return Status::OK();
    }
    const char* getName() const override { return "scrubbing-owner-control"; }
};

int retryControl(const std::string& scene, const char* maps) {
    auto config = std::make_shared<Config>();
    config->set("metadata_type", "p2p");
    config->set("metadata_servers", "");
    config->set("rpc_server_hostname", "127.0.0.1");
    config->set("rpc_server_port", "0");
    config->set("log_level", "warning");
    for (const auto* name : {"tcp", "shm", "rdma", "io_uring", "nvlink",
                             "mnnvl", "gds", "ascend_direct"})
        config->set(std::string("transports/") + name + "/enable", false);
    std::array<unsigned char, 8192> bytes{};
    TransferEngineImpl engine(config);
    require(engine.available(), "CPU engine is unavailable");
    auto first = std::make_shared<RefusingTransport>();
    engine.swapTransportForTest(RDMA, first);
    auto& manager = engine.metadata_->segmentManager();
    auto registry = std::make_unique<PublicationRegistry>();
    auto* publication = registry.get();
    manager.registry_ = std::move(registry);
    std::ifstream mappings("/proc/self/maps");
    std::ofstream archived(maps);
    archived << mappings.rdbuf();
    archived.close();
    require(bool(archived), "mapping evidence is unavailable");
    const auto handle = reinterpret_cast<tent_engine_t>(&engine);
    void* address = bytes.data();
    void* other = bytes.data() + 4096;
    if (scene == "retry-partial") {
        auto second = std::make_shared<ScrubbingTransport>();
        engine.swapTransportForTest(TCP, second);
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        second->refuse = true;
        require(tent_unregister_memory(handle, address, 4096) != 0,
                "C ABI swallowed partial removal failure");
        require(!first->owns && second->owns,
                "partial transport ownership changed");
        require(tent_register_memory(handle, address, 4096) != 0,
                "new generation admitted before old cleanup");
        second->refuse = false;
        require(tent_unregister_memory(handle, address, 4096) == 0,
                "retry failed");
        require(first->removals == 1 && second->removals == 2 && !second->owns,
                "retry repeated completed transport or lost original keys");
    } else if (scene == "retry-publication") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        publication->refuse = true;
        require(tent_unregister_memory(handle, address, 4096) != 0,
                "publication failure was swallowed");
        require(!first->owns && first->removals == 1,
                "physical removal failed");
        require(tent_register_memory(handle, address, 4096) != 0,
                "publication tombstone did not prevent reuse");
        publication->refuse = false;
        require(tent_unregister_memory(handle, address, 0) == 0,
                "wildcard publication retry failed");
        require(first->removals == 1,
                "publication retry repeated physical removal");
        require(tent_register_memory(handle, address, 4096) == 0,
                "new generation rejected after cleanup");
        require(tent_unregister_memory(handle, address, 4096) == 0,
                "new generation cleanup failed");
    } else if (scene == "retry-references") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        require(tent_register_memory(handle, address, 4096) == 0,
                "second ref failed");
        require(tent_unregister_memory(handle, address, 0) == 0,
                "nonfinal ref failed");
        require(
            first->owns && first->removals == 0 && first->registrations == 1,
            "nonfinal reference removed the MR");
        first->refuse = true;
        require(tent_unregister_memory(handle, address, 0) != 0,
                "final failure lost");
        require(tent_register_memory(handle, other, 4096) == 0,
                "retired address blocked an unrelated address");
        first->refuse = false;
        require(tent_unregister_memory(handle, address, 0) == 0,
                "final retry failed");
        require(tent_unregister_memory(handle, other, 0) == 0,
                "other cleanup failed");
        require(first->removals == 3 && !first->owns,
                "final owner was not cleaned");
    } else if (scene == "retry-batch-publication") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        require(tent_register_memory(handle, other, 4096) == 0,
                "register other failed");
        require(tent_register_memory(handle, other, 4096) == 0,
                "other second ref failed");
        void* addresses[] = {address, other};
        size_t sizes[] = {4096, 4096};
        publication->refuse = true;
        require(tent_unregister_memory_batch(handle, addresses, sizes, 2) != 0,
                "batch publication failure was swallowed");
        publication->refuse = false;
        unsigned puts = publication->puts;
        require(tent_unregister_memory_batch(handle, addresses, sizes, 2) == 0,
                "batch publication retry failed");
        require(publication->puts == puts + 1 && first->removals == 1 &&
                    first->owns,
                "batch retry consumed a later reference or repeated physical "
                "cleanup");
        require(tent_unregister_memory(handle, other, 4096) == 0,
                "remaining reference was lost");
        require(first->removals == 2 && !first->owns,
                "batch cleanup was incomplete");
    } else if (scene == "retry-batch-normal") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        require(tent_register_memory(handle, other, 4096) == 0,
                "register other failed");
        void* addresses[] = {address, other};
        size_t sizes[] = {4096, 4096};
        unsigned puts = publication->puts;
        require(tent_unregister_memory_batch(handle, addresses, sizes, 2) == 0,
                "normal batch cleanup failed");
        require(publication->puts == puts + 1 && first->removals == 2 &&
                    !first->owns,
                "normal batch did not keep one publication");
    } else {
        throw std::runtime_error("unknown retry scene");
    }
    std::cout << "scene=" << scene << " strict_contract_passed=true"
              << " test_transport_owns=" << first->owns
              << " remove_calls=" << first->removals
              << " metadata_publications=" << publication->puts
              << " payload_DMA=false" << std::endl;
    return 0;
}
}  // namespace

#ifndef ORBITKV_UNREGISTER_RUST_CONTROL
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        if (std::string(argv[1]).starts_with("retry-"))
            return retryControl(argv[1], argv[3]);
        return historicalControl(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
#else
extern "C" void orbitkv_test_install_unregister_control(void* native) {
    auto* engine = static_cast<mooncake::tent::TransferEngineImpl*>(native);
    engine->swapTransportForTest(
        mooncake::tent::RDMA,
        std::make_shared<mooncake::tent::RefusingTransport>());
}

extern "C" void orbitkv_test_set_unregister_refusal(void* native, int refused) {
    auto* engine = static_cast<mooncake::tent::TransferEngineImpl*>(native);
    auto* owner = static_cast<mooncake::tent::RefusingTransport*>(
        engine->transport_list_[mooncake::tent::RDMA].get());
    owner->refuse = refused != 0;
}

extern "C" int orbitkv_test_unregister_state(void* native,
                                             unsigned* registrations,
                                             unsigned* removals) {
    auto* engine = static_cast<mooncake::tent::TransferEngineImpl*>(native);
    auto* owner = static_cast<mooncake::tent::RefusingTransport*>(
        engine->transport_list_[mooncake::tent::RDMA].get());
    *registrations = owner->registrations;
    *removals = owner->removals;
    return owner->owns;
}
#endif
