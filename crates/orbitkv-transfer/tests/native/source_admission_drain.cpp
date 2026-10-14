// Real CPU/TCP source handlers, with a deterministic synchronous copy barrier.
// The CUDA type tag selects the copy hook; this executable uses only CPU
// memory.
#include "tent/platform/cpu.h"
#include "tent/runtime/control_plane.h"
#include "tent/transfer_engine.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/random.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace {
constexpr size_t bytes = 8192;
std::unique_ptr<unsigned char[]> source(new unsigned char[bytes]);
std::array<unsigned char, bytes> target{}, expected{};
std::atomic<bool> pause_copy{true}, inside_copy{false};
std::atomic<bool> classify_source_for_barrier{true};
std::atomic<bool> fail_entropy{false};
bool inject_partial_submit = false;
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void await_copy(bool value) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (inside_copy.load(std::memory_order_acquire) != value) {
    require(std::chrono::steady_clock::now() < deadline,
            "copy barrier timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
tent_engine_t make_engine() {
  for (auto [key, value] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"metadata_type", "p2p"},
           {"metadata_servers", ""},
           {"rpc_server_hostname", "127.0.0.1"},
           {"rpc_server_port", "0"},
           {"rpc_server_threads", "4"},
           {"local_segment_name", ""},
           {"transports/tcp/enable", "true"},
           {"transports/rdma/enable", "false"},
           {"transports/shm/enable", "false"},
           {"transports/nvlink/enable", "false"},
           {"transports/mnnvl/enable", "false"},
           {"enable_auto_failover_on_poll", "false"},
           {"enable_runtime_queue", "false"},
           {"transports/tcp/max_retry_count", "0"},
           {"transports/tcp/max_concurrent_tasks", "1"}})
    tent_set_config(key, value);
  auto engine = tent_create_engine();
  require(engine && tent_available(engine) == 1, "engine unavailable");
  return engine;
}
std::string endpoint(tent_engine_t engine) {
  char value[256]{};
  require(tent_segment_name(engine, value, sizeof(value)) == 0,
          "endpoint unavailable");
  return value;
}
tent_status_t complete(tent_engine_t engine, tent_batch_id_t batch,
                       size_t task) {
  tent_status_t status{};
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    require(tent_task_status(engine, batch, task, &status) == 0,
            "status unavailable");
    if (status.status == STATUS_COMPLETED || status.status == STATUS_FAILED ||
        status.status == STATUS_CANCELED)
      return status;
    require(std::chrono::steady_clock::now() < deadline, "status timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (true);
}
} // namespace

ssize_t getrandom(void *buffer, size_t length, unsigned int flags) {
  if (fail_entropy.load() && length == 16)
    return 0;
  return syscall(SYS_getrandom, buffer, length, flags);
}

// Deterministic C-ABI fault: the native engine admits the first real request,
// then this test-only boundary returns failure before the second submission.
int tent_submit(tent_engine_t engine, tent_batch_id_t batch,
                tent_request_t *requests, size_t count) {
  using Submit =
      int (*)(tent_engine_t, tent_batch_id_t, tent_request_t *, size_t);
  static auto native_submit =
      reinterpret_cast<Submit>(dlsym(RTLD_NEXT, "tent_submit"));
  if (!native_submit)
    return -1;
  if (inject_partial_submit && count == 2) {
    auto result = native_submit(engine, batch, requests, 1);
    return result == 0 ? -1 : result;
  }
  return native_submit(engine, batch, requests, count);
}

namespace mooncake::tent {
class ControlledPlatform final : public CpuPlatform {
public:
  ControlledPlatform() : CpuPlatform(std::make_shared<Config>()) {}
  MemoryType getMemoryType(void *pointer) override {
    return pointer == source.get() && classify_source_for_barrier.load()
               ? MTYPE_CUDA
               : MTYPE_CPU;
  }
  const std::string type() const override { return "cpu"; }
  Status copy(void *dst, void *src, size_t length) override {
    if (src == source.get()) {
      inside_copy.store(true, std::memory_order_release);
      while (pause_copy.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::memcpy(dst, src, length);
    if (src == source.get())
      inside_copy.store(false, std::memory_order_release);
    return Status::OK();
  }
};
Platform &Platform::getLoader(std::shared_ptr<Config>) {
  static ControlledPlatform platform;
  return platform;
}
} // namespace mooncake::tent

int main(int argc, char **argv) {
  tent_engine_t owner = nullptr, requester = nullptr;
  try {
    require(argc == 2, "expected scenario");
    std::string scenario = argv[1];
    require(scenario == "delayed" || scenario == "cancel" ||
                scenario == "partial" || scenario == "restart" ||
                scenario == "delegate" || scenario == "entropy" ||
                scenario == "staging",
            "unknown scenario");
    for (size_t i = 0; i < bytes; ++i)
      source[i] = expected[i] = (i * 17 + 31) % 251;
    target.fill(0x5a);
    owner = make_engine();
    if (scenario == "entropy") {
      fail_entropy.store(true);
      auto unavailable = tent_create_engine();
      fail_entropy.store(false);
      require(unavailable && tent_available(unavailable) == 0,
              "entropy control did not create unavailable native object");
      char epoch[33];
      std::memset(epoch, 0x5a, sizeof(epoch));
      uint64_t active = UINT64_MAX;
      auto close = tent_close_source(unavailable, 0, epoch, &active);
      require(close == -1 && active == UINT64_MAX &&
                  std::all_of(std::begin(epoch), std::end(epoch),
                              [](auto c) { return c == 0x5a; }),
              "unavailable engine produced or modified a source receipt");
      // Neither engine registered any allocation or submitted any transfer.
      tent_destroy_engine(unavailable);
      tent_destroy_engine(owner);
      owner = nullptr;
      std::cout << "scenario=entropy native_close=-1 receipt_untouched=1 "
                   "no_registration_or_submit=1 result=PASS\n";
      return 0;
    }
    requester = make_engine();
    require(tent_register_memory(owner, source.get(), bytes) == 0,
            "source register");
    require(tent_register_memory(requester, target.data(), bytes) == 0,
            "target register");
    auto owner_endpoint = endpoint(owner);
    if (scenario == "staging") {
      uint64_t unissued = UINT64_MAX;
      auto pin = mooncake::tent::ControlClient::pinStageBuffer(
          owner_endpoint, "invalid-control-location", unissued);
      require(!pin.ok() && unissued == UINT64_MAX,
              "CPU receipt profile admitted Pin before closure");
      auto unpin = mooncake::tent::ControlClient::unpinStageBuffer(
          owner_endpoint, UINT64_MAX);
      require(!unpin.ok(), "CPU receipt profile admitted Unpin before closure");
    }
    if (scenario == "delegate") {
      tent_segment_id_t receiver_segment{};
      auto receiver_endpoint = endpoint(requester);
      require(tent_open_segment(owner, &receiver_segment,
                                receiver_endpoint.c_str()) == 0,
              "open delegate receiver");
      mooncake::tent::Request delegated{};
      delegated.opcode = mooncake::tent::Request::WRITE;
      delegated.source = source.get();
      delegated.target_id = receiver_segment;
      delegated.target_offset = reinterpret_cast<uint64_t>(target.data());
      delegated.length = bytes;
      pause_copy.store(false, std::memory_order_release);
      classify_source_for_barrier.store(false);
      auto denied =
          mooncake::tent::ControlClient::delegate(owner_endpoint, delegated);
      require(!denied.ok() && std::all_of(target.begin(), target.end(),
                                          [](auto c) { return c == 0x5a; }),
              "unfenced delegate touched source/destination");
      classify_source_for_barrier.store(true);
      pause_copy.store(true, std::memory_order_release);
      require(tent_close_segment(owner, receiver_segment) == 0,
              "close delegate receiver");
    }
    tent_segment_id_t segment{};
    require(tent_open_segment(requester, &segment, owner_endpoint.c_str()) == 0,
            "open source");
    auto batch = tent_allocate_batch(requester, 2);
    require(batch != 0, "batch allocate");
    tent_request_t request{};
    request.opcode = OPCODE_READ;
    request.source = target.data();
    request.target_id = segment;
    request.target_offset = reinterpret_cast<uint64_t>(source.get());
    request.length = bytes;
    request.transport_hint = TRANSPORT_TCP;
    int cancel_rc = -999, partial_rc = -999;
    if (scenario == "partial") {
      std::array<tent_request_t, 2> requests{request, request};
      inject_partial_submit = true;
      partial_rc =
          tent_submit(requester, batch, requests.data(), requests.size());
      inject_partial_submit = false;
      require(partial_rc == -1,
              "partial boundary did not fail after first admission");
    } else {
      require(tent_submit(requester, batch, &request, 1) == 0, "submit first");
    }
    await_copy(true);
    if (scenario == "cancel")
      cancel_rc = tent_cancel_task(requester, batch, 0);
    char epoch[33]{};
    uint64_t active = UINT64_MAX;
    int first_close = tent_close_source(owner, 10, epoch, &active);
    require(first_close == 1 && active == 1 && inside_copy.load(),
            "close must retain admitted physical copy");
    if (scenario == "staging") {
      uint64_t unissued = UINT64_MAX;
      require(!mooncake::tent::ControlClient::pinStageBuffer(
                   owner_endpoint, "invalid-control-location", unissued)
                      .ok() &&
                  unissued == UINT64_MAX,
              "post-close Pin reached native owner");
      require(!mooncake::tent::ControlClient::unpinStageBuffer(owner_endpoint,
                                                               UINT64_MAX)
                   .ok(),
              "post-close Unpin reached native owner");
    }
    std::array<unsigned char, 45> denied;
    denied.fill(0x5a);
    auto denied_status = mooncake::tent::ControlClient::recvData(
        owner_endpoint, reinterpret_cast<uint64_t>(source.get()), denied.data(),
        denied.size(), epoch);
    require(!denied_status.ok(), "new READ must be denied");
    require(std::all_of(denied.begin(), denied.end(),
                        [](auto c) { return c == 0x5a; }),
            "denied READ modified destination");
    // Cancellation is an observation/request, never the drain receipt.
    require(inside_copy.load(),
            "cancel or invalid submit ended the paused copy");
    pause_copy.store(false, std::memory_order_release);
    auto status = complete(requester, batch, 0);
    require(status.status == STATUS_COMPLETED &&
                status.transferred_bytes == bytes && target == expected,
            "complete byte oracle");
    await_copy(false);
    char final_epoch[33]{};
    int final_close = tent_close_source(owner, 1000, final_epoch, &active);
    require(final_close == 0 && active == 0 &&
                std::strcmp(epoch, final_epoch) == 0,
            "drain must bind the exact closed engine epoch");
    require(tent_register_memory(owner, source.get(), bytes) != 0,
            "closed admission reopened");
    require(tent_unregister_memory(owner, source.get(), bytes) == 0,
            "unregister after drain");
    require(tent_free_batch(requester, batch) == 0, "free completed batch");
    require(tent_close_segment(requester, segment) == 0, "close old segment");
    tent_destroy_engine(owner);
    owner = nullptr;
    if (scenario == "restart") {
      // Same live address, new native source epoch. No allocator-generation
      // claim.
      owner = make_engine();
      require(tent_register_memory(owner, source.get(), bytes) == 0,
              "restart register");
      auto restarted_endpoint = endpoint(owner);
      denied.fill(0x5a);
      auto stale = mooncake::tent::ControlClient::recvData(
          restarted_endpoint, reinterpret_cast<uint64_t>(source.get()),
          denied.data(), denied.size(), epoch);
      require(!stale.ok() && std::all_of(denied.begin(), denied.end(),
                                         [](auto c) { return c == 0x5a; }),
              "old epoch reached restarted source");
      require(tent_open_segment(requester, &segment,
                                restarted_endpoint.c_str()) == 0,
              "open restart");
      batch = tent_allocate_batch(requester, 1);
      request.target_id = segment;
      target.fill(0x5a);
      require(tent_submit(requester, batch, &request, 1) == 0,
              "restart submit");
      status = complete(requester, batch, 0);
      require(status.status == STATUS_COMPLETED && target == expected,
              "new epoch byte oracle");
      require(tent_free_batch(requester, batch) == 0, "restart free batch");
      require(tent_close_segment(requester, segment) == 0,
              "restart close segment");
      require(tent_close_source(owner, 1000, final_epoch, &active) == 0 &&
                  std::strcmp(epoch, final_epoch) != 0,
              "restart native epoch was reused");
      require(tent_unregister_memory(owner, source.get(), bytes) == 0,
              "restart unregister");
      tent_destroy_engine(owner);
      owner = nullptr;
    }
    // The source heap is actually released only after its native receipt and
    // unregister.
    source.reset();
    require(tent_close_source(requester, 1000, final_epoch, &active) == 0,
            "requester source drain");
    require(tent_unregister_memory(requester, target.data(), bytes) == 0,
            "target unregister");
    tent_destroy_engine(requester);
    requester = nullptr;
    std::cout
        << "scenario=" << scenario << " first_close=" << first_close
        << " final_close=" << final_close << " cancel_rc=" << cancel_rc
        << " partial_submit_rc=" << partial_rc << " payload_bytes=" << bytes
        << " physical_source_released=1 normal_engine_teardown=1 result=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    pause_copy.store(false, std::memory_order_release);
    std::cerr << error.what()
              << "\nNo engine destruction or memory reclamation claimed on "
                 "failure.\n";
    // Failed control owners survive to process isolation; no unsafe destructor
    // retry.
    source.release();
    return 2;
  }
}
