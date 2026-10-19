// Actual TENT engine bookkeeping with a test-only transport refusal. No payload
// transfer or device registration occurs; these stable host bytes stay alive.
#include <array>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

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
    bool refuse_registration = false;
    bool partial_registration = false;
    bool skip_registration = false;
    bool missing_removal_is_ok = false;
    bool complete_batch = false;
    TransportType type = RDMA;
    std::set<std::pair<uint64_t, uint64_t>> rejected;
    bool owns = false;
    unsigned registrations = 0;
    unsigned removals = 0;
    std::set<std::pair<uint64_t, uint64_t>> addresses;
    std::function<void()> after_remove;

    Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions&) override {
        ++registrations;
        if (skip_registration) return Status::OK();
        if (refuse_registration || rejected.count({desc.addr, desc.length}))
            return Status::RdmaError("controlled registration refusal");
        if (!addresses.emplace(desc.addr, desc.length).second)
            return Status::RdmaError("test transport still owns this address");
        owns = true;
        if (partial_registration)
            return Status::RdmaError("controlled partially owned registration");
        desc.transports.push_back(type);
        return Status::OK();
    }

    Status addMemoryBuffer(std::vector<BufferDesc>& descriptors,
                           const MemoryOptions& options) override {
        if (!complete_batch) return Transport::addMemoryBuffer(descriptors, options);
        Status first_error;
        for (auto& descriptor : descriptors) {
            auto status = addMemoryBuffer(descriptor, options);
            if (!status.ok() && first_error.ok()) first_error = status;
        }
        return first_error;
    }

    bool tracksLocalBuffer(const BufferDesc& desc) override {
        return addresses.count({desc.addr, desc.length}) != 0;
    }

    Status removeMemoryBuffer(BufferDesc& desc) override {
        ++removals;
        if (missing_removal_is_ok && !addresses.count({desc.addr, desc.length}))
            return Status::OK();
        if (refuse) return Status::RdmaError("controlled unregister refusal");
        if (!addresses.erase({desc.addr, desc.length}))
            return Status::RdmaError(
                "test transport already removed this address");
        owns = !addresses.empty();
        desc.lkey.clear();
        desc.rkey.clear();
        if (after_remove) after_remove();
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
            descriptor.length = bytes.size();
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
    std::function<Status()> before_put;
    Status getSegmentDesc(SegmentDescRef&, const std::string&) override {
        return Status::NotImplemented(
            "control does not resolve remote segments");
    }
    Status putSegmentDesc(SegmentDescRef&) override {
        ++puts;
        if (before_put) return before_put();
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
    } else if (scene == "retry-batch-prefix") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        require(tent_register_memory(handle, other, 4096) == 0,
                "register other failed");
        void* addresses[] = {address, other};
        size_t sizes[] = {4096, 4096};
        publication->refuse = true;
        require(tent_unregister_memory_batch(handle, addresses, sizes, 2) != 0,
                "batch publication failure was swallowed");
        require(tent_register_memory(handle, address, 4096) != 0 &&
                    first->registrations == 2,
                "unpublished batch prefix re-entered its provider");
        publication->refuse = false;
        require(tent_unregister_memory(handle, other, 4096) == 0,
                "suffix publication retry failed");
        require(tent_register_memory(handle, address, 4096) == 0,
                "published prefix could not be registered again");
        require(tent_unregister_memory(handle, address, 4096) == 0,
                "new generation cleanup failed");
        require(first->removals == 3 && !first->owns,
                "batch suffix retry repeated physical removal");
    } else if (scene == "retry-batch-wildcard") {
        for (bool refuse : {false, true}) {
            require(tent_register_memory(handle, address, 4096) == 0,
                    "small range registration failed");
            require(tent_register_memory(handle, address, 8192) == 0,
                    "large range registration failed");
            void* addresses[] = {address, address};
            const auto puts = publication->puts;
            const auto removals = first->removals;
            publication->refuse = refuse;
            const auto status =
                tent_unregister_memory_batch(handle, addresses, nullptr, 2);
            require((status != 0) == refuse &&
                        first->removals == removals + 2 && !first->owns &&
                        publication->puts == puts + 1,
                    "completed prefix shadowed the next wildcard range");
            if (refuse) {
                publication->refuse = false;
                require(tent_unregister_memory(handle, address, 0) == 0,
                        "wildcard suffix selected the completed prefix");
                require(tent_register_memory(handle, address, 8192) == 0,
                        "wildcard suffix left an unfinished tombstone");
                require(tent_unregister_memory(handle, address, 8192) == 0,
                        "wildcard new generation cleanup failed");
                require(first->removals == removals + 3 && !first->owns,
                        "wildcard suffix repeated physical removal");
            }
        }
#ifdef ORBITKV_UNREGISTER_PUBLICATION_REVISION
    } else if (scene == "retry-publication-revision") {
        require(tent_register_memory(handle, address, 4096) == 0,
                "register failed");
        require(tent_register_memory(handle, other, 4096) == 0,
                "register other failed");
        const auto before = manager.localRevision();
        std::mutex mutex;
        std::condition_variable changed;
        bool entered = false, release = false, removed = false;
        unsigned calls = 0;
        publication->before_put = [&] {
            std::unique_lock<std::mutex> lock(mutex);
            if (calls++ == 0) {
                entered = true;
                changed.notify_all();
                if (!changed.wait_for(lock, std::chrono::seconds(4),
                                      [&] { return release; }))
                    return Status::InternalError("old put release timed out");
                return Status::OK();
            }
            return Status::InternalError("new publication refused");
        };
        first->after_remove = [&] {
            if (first->removals == 2) {
                std::lock_guard<std::mutex> lock(mutex);
                removed = true;
                changed.notify_all();
            }
        };
        Status old_status;
        std::thread old_put([&] { old_status = manager.synchronizeLocal(); });
        bool old_ready;
        {
            std::unique_lock<std::mutex> lock(mutex);
            old_ready = changed.wait_for(lock, std::chrono::seconds(4),
                                         [&] { return entered; });
            if (!old_ready) {
                release = true;
                changed.notify_all();
            }
        }
        if (!old_ready) {
            old_put.join();
            require(false, "old snapshot did not reach the registry");
        }
        int batch_status = 0;
        std::thread batch([&] {
            void* addresses[] = {address, other};
            size_t sizes[] = {4096, 4096};
            batch_status =
                tent_unregister_memory_batch(handle, addresses, sizes, 2);
        });
        bool withdrew;
        {
            std::unique_lock<std::mutex> lock(mutex);
            withdrew = changed.wait_for(lock, std::chrono::seconds(4),
                                        [&] { return removed; });
            release = true;
            changed.notify_all();
        }
        old_put.join();
        batch.join();
        publication->before_put = {};
        first->after_remove = {};
        require(withdrew && old_status.ok() && batch_status != 0,
                "controlled old/new publication ordering was not reached");
        require(manager.localRevision() > before &&
                    manager.publishedRevision() == before,
                "old put incorrectly acknowledged a later withdrawal");
        require(tent_register_memory(handle, address, 4096) != 0 &&
                    first->registrations == 2,
                "old publication released a newer batch prefix");
        require(tent_unregister_memory(handle, other, 4096) == 0,
                "new publication retry failed");
        require(tent_register_memory(handle, address, 4096) == 0,
                "new publication did not release its completed prefix");
        require(tent_unregister_memory(handle, address, 4096) == 0,
                "new generation cleanup failed");
#endif
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

int registerControl(const std::string& scene, const char* maps) {
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
    require(engine.available(), "CPU engine unavailable");
    auto first = std::make_shared<RefusingTransport>();
    first->missing_removal_is_ok = first->complete_batch = true;
    engine.swapTransportForTest(RDMA, first);
    auto& manager = engine.metadata_->segmentManager();
    auto registry = std::make_unique<PublicationRegistry>();
    auto* publication = registry.get();
    manager.registry_ = std::move(registry);
    std::ifstream mappings("/proc/self/maps");
    std::ofstream archived(maps); archived << mappings.rdbuf(); archived.close();
    require(bool(archived), "mapping evidence unavailable");
    const auto handle = reinterpret_cast<tent_engine_t>(&engine);
    void* address = bytes.data(); void* other = bytes.data() + 4096;
    const auto key = [](void* value) {
        return std::make_pair(reinterpret_cast<uint64_t>(value), uint64_t{4096});
    };
    const auto buffers = [&] {
        return std::get<MemorySegmentDesc>(manager.getLocal()->detail).buffers;
    };
    const auto register_one = [&](void* ptr) {
        return tent_register_memory(handle, ptr, 4096);
    };
    const auto remove_one = [&](void* ptr) {
        return tent_unregister_memory(handle, ptr, 4096);
    };
    if (scene == "register-refusal" || scene == "register-empty-success") {
        first->refuse_registration = scene == "register-refusal";
        first->skip_registration = scene == "register-empty-success";
        require(register_one(address) != 0, "unusable range reported success");
        require(buffers().empty() && !first->owns, "unusable range was advertised");
        first->refuse_registration = first->skip_registration = false;
        require(register_one(address) == 0 && remove_one(address) == 0,
                "healthy retry after refusal failed");
    } else if (scene == "register-partial") {
        first->partial_registration = first->refuse = true;
        require(register_one(address) != 0, "partial registration reported success");
        require(buffers().empty() && first->owns &&
                    engine.retired_registrations_.at(key(address)).registration_failed,
                "unpublished physical owner was discarded");
        require(register_one(address) != 0 && first->registrations == 1,
                "new registration entered before pending cleanup");
        first->partial_registration = first->refuse = false;
        require(register_one(address) == 0 && first->registrations == 2,
                "cleanup did not permit a new registration");
        require(remove_one(address) == 0, "new registration cleanup failed");
    } else if (scene == "register-batch-rollback") {
        require(register_one(address) == 0, "prior owner failed");
        first->rejected.insert(key(other));
        void* addresses[] = {address, address, other};
        size_t sizes[] = {4096,4096,4096};
        require(tent_register_memory_batch(handle, addresses, sizes, 3, kGlobalReadWrite) != 0,
                "failed batch reported success");
        const auto rows = buffers();
        require(rows.size() == 1 && rows[0].ref_count == 1 &&
                    first->addresses.count(key(address)) == 1 &&
                    !first->addresses.count(key(other)),
                "failed attempt changed prior references or advertised suffix");
        require(remove_one(address) == 0, "prior owner cleanup failed");
    } else if (scene == "register-new-duplicates") {
        void* addresses[] = {address,address}; size_t sizes[] = {4096,4096};
        require(tent_register_memory_batch(handle, addresses, sizes, 2, kGlobalReadWrite) == 0,
                "new repeated-range batch failed");
        require(first->registrations == 1 && buffers().size() == 1 &&
                    buffers()[0].ref_count == 2,
                "same range registered twice instead of counted once");
        require(remove_one(address) == 0 && first->owns && first->removals == 0,
                "nonfinal reference removed physical owner");
        require(remove_one(address) == 0 && !first->owns && first->removals == 1,
                "last duplicate did not remove exactly once");
    } else if (scene == "register-options") {
        auto second = std::make_shared<RefusingTransport>();
        second->type = TCP; second->missing_removal_is_ok = true;
        engine.swapTransportForTest(TCP, second);
        MemoryOptions options; options.type = RDMA;
        require(engine.registerLocalMemory({address},{4096},options).ok(),
                "explicit initial registration failed");
        options.type = TCP;
        require(!engine.registerLocalMemory({address},{4096},options).ok() &&
                    buffers()[0].ref_count == 1 && second->registrations == 0,
                "incompatible required transport accepted as duplicate");
        require(remove_one(address) == 0, "prior option owner cleanup failed");
    } else if (scene == "register-publication") {
        require(register_one(address) == 0, "prior owner failed");
        publication->refuse = true;
        void* addresses[] = {address,other}; size_t sizes[] = {4096,4096};
        require(tent_register_memory_batch(handle, addresses, sizes, 2, kGlobalReadWrite) != 0,
                "failed publication granted references");
        require(buffers().size() == 1 && buffers()[0].ref_count == 1 &&
                    !first->addresses.count(key(other)) &&
                    engine.retired_registrations_.count(key(other)) == 1,
                "publication failure lost withdrawal or prior owner");
        const auto attempts = first->registrations;
        require(register_one(other) != 0 && first->registrations == attempts,
                "unpublished withdrawal allowed same-base reuse");
        publication->refuse = false;
        require(register_one(other) == 0 && remove_one(other) == 0 &&
                    remove_one(address) == 0, "publication retry failed");
    } else if (scene == "register-per-range-fallback" ||
               scene == "register-all-refusal") {
        auto second = std::make_shared<RefusingTransport>();
        second->type = TCP;
        second->missing_removal_is_ok = second->complete_batch = true;
        engine.swapTransportForTest(TCP, second);
        first->rejected.insert(key(other)); second->rejected.insert(key(address));
        if (scene == "register-all-refusal") {
            first->rejected.insert(key(address)); second->rejected.insert(key(other));
        }
        void* addresses[] = {address,other}; size_t sizes[] = {4096,4096};
        const int status = tent_register_memory_batch(handle, addresses, sizes, 2, kGlobalReadWrite);
        if (scene == "register-all-refusal") {
            require(status != 0 && buffers().empty() && !first->owns && !second->owns,
                    "all-provider refusal accepted or advertised");
        } else {
            const auto rows = buffers();
            require(status == 0 && rows.size() == 2 &&
                        rows[0].transports == std::vector<TransportType>{RDMA} &&
                        rows[1].transports == std::vector<TransportType>{TCP},
                    "per-range fallback did not preserve only usable providers");
            require(remove_one(address) == 0 && remove_one(other) == 0 &&
                        !second->owns, "fallback owners did not clean up");
        }
    } else {
        throw std::runtime_error("unknown registration scene");
    }
    require(buffers().empty() && !first->owns && engine.retired_registrations_.empty(),
            "registration control left a test owner");
    std::cout << "scene=" << scene << " strict_contract_passed=true"
              << " registration_calls=" << first->registrations
              << " remove_calls=" << first->removals
              << " payload_DMA=false" << std::endl;
    return 0;
}
}  // namespace

#ifndef ORBITKV_UNREGISTER_RUST_CONTROL
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        if (std::string(argv[1]).starts_with("register-"))
            return registerControl(argv[1], argv[3]);
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

extern "C" void orbitkv_test_set_registration_refusal(void* native, int partial) {
    auto* engine = static_cast<mooncake::tent::TransferEngineImpl*>(native);
    if (partial == 1) engine->swapTransportForTest(mooncake::tent::TCP, nullptr);
    auto* owner = static_cast<mooncake::tent::RefusingTransport*>(
        engine->transport_list_[mooncake::tent::RDMA].get());
    owner->missing_removal_is_ok = true;
    owner->refuse_registration = partial == 1;
    owner->partial_registration = owner->refuse = partial == 2;
}

extern "C" int orbitkv_test_partial_registration_pending(void* native) {
    auto* engine = static_cast<mooncake::tent::TransferEngineImpl*>(native);
    auto* owner = static_cast<mooncake::tent::RefusingTransport*>(
        engine->transport_list_[mooncake::tent::RDMA].get());
    if (!owner->owns || engine->retired_registrations_.size() != 1) return 0;
    const auto& retired = engine->retired_registrations_.begin()->second;
    return retired.registration_failed && retired.withdrawn &&
           retired.pending == std::vector<mooncake::tent::TransportType>{
                                  mooncake::tent::RDMA};
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
