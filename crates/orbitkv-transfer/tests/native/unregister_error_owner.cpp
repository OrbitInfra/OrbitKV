// Actual TENT engine bookkeeping with a test-only transport refusal. No payload
// transfer or device registration occurs; these stable host bytes stay alive.
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "tent/platform/cpu.h"
#include "tent/runtime/transfer_engine_impl.h"

namespace mooncake::tent {
Platform& Platform::getLoader(std::shared_ptr<Config>) {
  static CpuPlatform platform(std::make_shared<Config>());
  return platform;
}

class RefusingTransport final : public Transport {
 public:
  bool refuse = false;
  bool owns = false;
  unsigned registrations = 0;
  unsigned removals = 0;

  Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions&) override {
    ++registrations;
    if (owns) return Status::RdmaError("test transport still owns this address");
    owns = true;
    desc.transports.push_back(RDMA);
    return Status::OK();
  }

  Status removeMemoryBuffer(BufferDesc&) override {
    ++removals;
    if (refuse) return Status::RdmaError("controlled unregister refusal");
    owns = false;
    return Status::OK();
  }

  const char* getName() const override { return "unregister-owner-control"; }
};
}  // namespace mooncake::tent

int main(int argc, char** argv) {
  using namespace mooncake::tent;
  if (argc != 4) return 2;
  const std::string scenario = argv[1], expectation = argv[2];
  if ((scenario != "normal" && scenario != "refused") ||
      (expectation != "strict" && expectation != "reproduce")) return 2;
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
    if (!engine.available()) throw std::runtime_error("CPU engine unavailable");
    engine.swapTransportForTest(RDMA, transport);
    std::ifstream mappings("/proc/self/maps");
    std::ofstream archived(argv[3]);
    archived << mappings.rdbuf();
    archived.close();
    if (!archived) throw std::runtime_error("actual mapping evidence failed");
    auto registered = engine.registerLocalMemory(bytes.data(), bytes.size());
    if (!registered.ok() || !transport->owns)
      throw std::runtime_error("registration control did not establish ownership");
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
      auto registration_retry = engine.registerLocalMemory(bytes.data(), bytes.size());
      reregister_ok = registration_retry.ok();
      reregister_attempts = transport->registrations - prior;
      stale_owner_denies_reregister = transport->owns && reregister_attempts == 1;
    }
    accepted = scenario == "normal"
                   ? first.ok() && !retained && retry.ok() && !transport->owns
                   : expectation == "reproduce"
                         ? lost && stale_owner_denies_reregister
                               && reregister_ok
                         : !first.ok() && retained && retry.ok() &&
                               !retained_after_retry;
    std::cout << "scenario=" << scenario << " unregister_ok=" << first.ok()
              << " owner_after_first=" << retained << " retry_ok=" << retry.ok()
              << " owner_after_retry=" << retained_after_retry
              << " remove_calls=" << transport->removals
              << " lost_unregister_ownership=" << lost
              << " live_transport_denies_reregister=" << stale_owner_denies_reregister
              << " reregister_reported_ok=" << reregister_ok
              << " reregister_calls=" << reregister_attempts
              << std::endl;
    // Reconcile only the injected transport's test owner. This bypass is not a
    // production drain receipt and does not qualify physical termination.
    if (transport->owns) {
      BufferDesc descriptor{};
      auto status = transport->removeMemoryBuffer(descriptor);
      if (!status.ok() || transport->owns)
        throw std::runtime_error("test owner could not be reconciled");
    }
  }
  std::cout << "test_owner_reconciled=" << !transport->owns
            << " payload_DMA=false engine_scope_ended=true" << std::endl;
  return accepted ? 0 : 1;
}
