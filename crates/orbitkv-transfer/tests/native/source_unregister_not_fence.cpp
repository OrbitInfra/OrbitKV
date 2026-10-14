// Real TENT TCP/C ABI, valid persistent CPU buffers, and a test-only copy
// barrier. This negative control deliberately mutates an allocation after
// logical unregister.
#include "tent/platform/cpu.h"
#include "tent/transfer_engine.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
namespace {
std::array<unsigned char, 8192> source{}, target{}, old_source{};
std::atomic<bool> pause_copy{true}, inside_copy{false};
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace
namespace mooncake::tent {
class ControlledPlatform final : public CpuPlatform {
public:
  ControlledPlatform() : CpuPlatform(std::make_shared<Config>()) {}
  MemoryType getMemoryType(void *pointer) override {
    return pointer == source.data() ? MTYPE_CUDA : MTYPE_CPU;
  }
  const std::string type() const override { return "cuda"; }
  Status copy(void *dst, void *src, size_t bytes) override {
    if (src == source.data()) {
      inside_copy.store(true, std::memory_order_release);
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(15);
      while (pause_copy.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline)
          return Status::InternalError("source copy control timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    std::memcpy(dst, src, bytes);
    if (src == source.data())
      inside_copy.store(false, std::memory_order_release);
    return Status::OK();
  }
};
Platform &Platform::getLoader(std::shared_ptr<Config>) {
  static ControlledPlatform platform;
  return platform;
}
} // namespace mooncake::tent
namespace {
tent_engine_t make_engine() {
  for (auto [key, value] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"metadata_type", "p2p"},
           {"metadata_servers", ""},
           {"rpc_server_hostname", "127.0.0.1"},
           {"rpc_server_port", "0"},
           {"local_segment_name", ""},
           {"transports/tcp/enable", "true"},
           {"transports/rdma/enable", "false"},
           {"transports/shm/enable", "false"},
           {"transports/nvlink/enable", "false"},
           {"enable_auto_failover_on_poll", "false"},
           {"enable_runtime_queue", "false"},
           {"transports/tcp/max_retry_count", "0"},
           {"transports/tcp/max_concurrent_tasks", "1"}})
    tent_set_config(key, value);
  auto engine = tent_create_engine();
  require(engine && tent_available(engine) == 1, "engine unavailable");
  return engine;
}
} // namespace
int main() {
  tent_engine_t owner = nullptr, requester = nullptr;
  try {
    for (size_t i = 0; i < source.size(); ++i)
      source[i] = (i * 17 + 31) % 251;
    old_source = source;
    target.fill(0x5a);
    owner = make_engine();
    requester = make_engine();
    require(tent_register_memory(owner, source.data(), source.size()) == 0,
            "source register");
    require(tent_register_memory(requester, target.data(), target.size()) == 0,
            "target register");
    char endpoint[256]{};
    require(tent_segment_name(owner, endpoint, sizeof(endpoint)) == 0,
            "source endpoint");
    tent_segment_id_t segment = 0;
    require(tent_open_segment(requester, &segment, endpoint) == 0,
            "open source");
    auto batch = tent_allocate_batch(requester, 1);
    require(batch != 0, "batch allocation");
    tent_request_t request{};
    request.opcode = OPCODE_READ;
    request.source = target.data();
    request.target_id = segment;
    request.target_offset = reinterpret_cast<uint64_t>(source.data());
    request.length = source.size();
    request.transport_hint = TRANSPORT_TCP;
    require(tent_submit(requester, batch, &request, 1) == 0, "submit");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!inside_copy.load(std::memory_order_acquire)) {
      require(std::chrono::steady_clock::now() < deadline,
              "source handler not reached");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    auto unregistered = std::async(std::launch::async, [&] {
      return tent_unregister_memory(owner, source.data(), source.size());
    });
    const bool returned = unregistered.wait_for(std::chrono::seconds(2)) ==
                          std::future_status::ready;
    int unregister_rc = returned ? unregistered.get() : -999;
    const bool still_active = inside_copy.load(std::memory_order_acquire);
    if (returned && unregister_rc == 0 && still_active) {
      for (size_t i = 0; i < source.size(); ++i)
        source[i] = (i * 17 + 79) % 251;
    }
    pause_copy.store(false, std::memory_order_release);
    if (!returned)
      unregister_rc = unregistered.get();
    tent_status_t status{};
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      require(tent_task_status(requester, batch, 0, &status) == 0, "status");
      if (status.status == STATUS_COMPLETED || status.status == STATUS_FAILED)
        break;
      require(std::chrono::steady_clock::now() < deadline,
              "completion timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (true);
    const bool reproduced = returned && unregister_rc == 0 && still_active &&
                            status.status == STATUS_COMPLETED &&
                            status.transferred_bytes == source.size() &&
                            target == source && target != old_source;
    std::cout << "unregister_returned_before_copy=" << returned
              << " unregister_rc=" << unregister_rc
              << " source_handler_active=" << still_active
              << " native_status=" << status.status
              << " bytes=" << status.transferred_bytes
              << " target_is_new_generation="
              << (target == source && target != old_source)
              << " reproduced=" << reproduced << std::endl;
    require(!inside_copy.load(), "source CPU copy did not return");
    require(tent_free_batch(requester, batch) == 0,
            "completed batch free request");
    tent_unregister_memory(requester, target.data(), target.size());
    // Buffers remained valid through the completed CPU-copy control.
    // Destruction here is teardown after that evidence, not a physical-fence
    // assertion.
    tent_destroy_engine(requester);
    requester = nullptr;
    tent_destroy_engine(owner);
    owner = nullptr;
    return reproduced ? 0 : 1;
  } catch (const std::exception &error) {
    pause_copy.store(false, std::memory_order_release);
    std::cerr << error.what() << std::endl;
    if (requester)
      tent_destroy_engine(requester);
    if (owner)
      tent_destroy_engine(owner);
    return 2;
  }
}
