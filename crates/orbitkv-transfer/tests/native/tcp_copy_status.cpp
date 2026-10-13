// Controlled copy failures on valid host buffers exercise the real TENT TCP
// handlers and C ABI. This helper is linked only into the test executable.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "tent/platform/cpu.h"
#include "tent/runtime/control_plane.h"
#include "tent/transfer_engine.h"

namespace {
std::atomic<void *> failed_source{nullptr};
std::atomic<void *> failed_destination{nullptr};
std::atomic<void *> device_classification{nullptr};
std::atomic<unsigned> copy_failures{0};
constexpr size_t capacity = 8192;
std::array<unsigned char, capacity> source;
std::array<unsigned char, capacity> destination;
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace

namespace mooncake::tent {
class ControlledPlatform final : public CpuPlatform {
public:
  ControlledPlatform() : CpuPlatform(std::make_shared<Config>()) {}
  Status copy(void *dst, void *src, size_t length) override {
    if (src == failed_source.load() || dst == failed_destination.load()) {
      ++copy_failures;
      return Status::InternalError("controlled copy failure before submission");
    }
    std::memcpy(dst, src, length);
    return Status::OK();
  }
  MemoryType getMemoryType(void *addr) override {
    return addr == device_classification.load() ? MTYPE_CUDA : MTYPE_CPU;
  }
  const std::string type() const override { return "cuda"; }
};
Platform &Platform::getLoader(std::shared_ptr<Config>) {
  static ControlledPlatform platform;
  return platform;
}
} // namespace mooncake::tent

namespace {
void config(const char *key, const char *value) { tent_set_config(key, value); }
tent_engine_t make_engine() {
  config("metadata_type", "p2p");
  config("metadata_servers", "");
  config("rpc_server_hostname", "127.0.0.1");
  config("rpc_server_port", "0");
  config("local_segment_name", "");
  config("transports/tcp/enable", "true");
  config("transports/rdma/enable", "false");
  config("transports/shm/enable", "false");
  config("transports/nvlink/enable", "false");
  config("enable_auto_failover_on_poll", "false");
  config("enable_runtime_queue", "false");
  config("transports/tcp/max_retry_count", "0");
  config("transports/tcp/max_concurrent_tasks", "1");
  auto engine = tent_create_engine();
  require(engine && tent_available(engine) == 1, "native engine unavailable");
  return engine;
}
std::string endpoint(tent_engine_t engine) {
  char name[256]{};
  require(tent_segment_name(engine, name, sizeof(name)) == 0, "segment name");
  return name;
}
void disarm() {
  failed_source = nullptr;
  failed_destination = nullptr;
  device_classification = nullptr;
}
void arm(const std::string &site) {
  copy_failures = 0;
  if (site == "client-send" || site == "server-recv") {
    failed_source = source.data();
    device_classification = source.data();
  } else {
    failed_destination = destination.data();
  }
}
void reset(unsigned round) {
  for (size_t i = 0; i < capacity; ++i)
    source[i] = (i * 17 + round * 31) % 251;
  destination.fill(0x5a);
}
bool sentinel() {
  return std::all_of(destination.begin(), destination.end(),
                     [](auto value) { return value == 0x5a; });
}
int native_transfer(tent_engine_t engine, tent_segment_id_t segment, int opcode,
                    bool expect_failure, size_t length = capacity) {
  tent_request_t request{};
  request.opcode = opcode;
  request.source = opcode == OPCODE_WRITE ? source.data() : destination.data();
  request.target_id = segment;
  request.target_offset = reinterpret_cast<uint64_t>(
      opcode == OPCODE_WRITE ? destination.data() : source.data());
  request.length = length;
  request.transport_hint = TRANSPORT_TCP;
  auto batch = tent_allocate_batch(engine, 1);
  require(batch != 0, "batch allocation");
  require(tent_submit(engine, batch, &request, 1) == 0,
          "native submit rejected");
  tent_status_t status{};
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    require(tent_task_status(engine, batch, 0, &status) == 0, "task status");
    if (status.status == STATUS_COMPLETED || status.status == STATUS_FAILED)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  const int observed = status.status;
  std::cout << "native status=" << observed
            << " bytes=" << status.transferred_bytes
            << " expected_failure=" << expect_failure << std::endl;
  // Retain the failed batch until engine teardown joins its CPU workers. This
  // test does not use free_batch success as a physical termination fence.
  if (expect_failure && observed == STATUS_FAILED) {
    require(status.transferred_bytes == 0,
            "failed copy reports completed bytes");
  }
  if (!expect_failure) {
    require(observed == STATUS_COMPLETED && status.transferred_bytes == length,
            "normal completion/byte count");
    require(tent_free_batch(engine, batch) == 0, "normal free request");
  }
  return observed;
}
} // namespace

int main() {
  tent_engine_t sender = nullptr, receiver = nullptr;
  try {
    sender = make_engine();
    receiver = make_engine();
    require(tent_register_memory(sender, source.data(), capacity) == 0,
            "source registration");
    require(tent_register_memory(receiver, destination.data(), capacity) == 0,
            "destination registration");
    const auto sender_name = endpoint(sender),
               receiver_name = endpoint(receiver);
    tent_segment_id_t sender_segment{}, receiver_segment{};
    require(tent_open_segment(sender, &receiver_segment,
                              receiver_name.c_str()) == 0,
            "receiver open");
    require(tent_open_segment(receiver, &sender_segment, sender_name.c_str()) ==
                0,
            "sender open");
    unsigned errors = 0, round = 0;
    for (const std::string site :
         {"client-send", "server-send", "server-recv", "client-recv"}) {
      reset(++round);
      arm(site);
      auto status =
          site.find("send") != std::string::npos
              ? mooncake::tent::ControlClient::sendData(
                    receiver_name,
                    reinterpret_cast<uint64_t>(destination.data()),
                    source.data(), capacity)
              : mooncake::tent::ControlClient::recvData(
                    sender_name, reinterpret_cast<uint64_t>(source.data()),
                    destination.data(), capacity);
      const bool direct_rejected =
          !status.ok() && sentinel() && copy_failures > 0;
      std::cout << "site=" << site << " direct=" << status.ToString()
                << " sentinel=" << sentinel() << " injected=" << copy_failures
                << std::endl;
      reset(++round);
      arm(site);
      const bool write = site.find("send") != std::string::npos;
      const int observed = native_transfer(
          write ? sender : receiver, write ? receiver_segment : sender_segment,
          write ? OPCODE_WRITE : OPCODE_READ, true);
      if (!direct_rejected || observed != STATUS_FAILED || !sentinel() ||
          !copy_failures)
        ++errors;
      disarm();
      reset(++round);
      native_transfer(write ? sender : receiver,
                      write ? receiver_segment : sender_segment,
                      write ? OPCODE_WRITE : OPCODE_READ, false);
      require(source == destination, "normal recovery payload/sentinel oracle");
    }
    // Error text with the requested payload length must never be consumed
    // as valid bytes; this is a separate server-READ negative control.
    arm("server-recv");
    auto failed = mooncake::tent::Status::InternalError(
        "controlled copy failure before submission");
    const size_t collision =
        std::string("RecvData failed: copy: " + failed.ToString()).size();
    reset(++round);
    auto status = mooncake::tent::ControlClient::recvData(
        sender_name, reinterpret_cast<uint64_t>(source.data()),
        destination.data(), collision);
    if (status.ok() || !sentinel())
      ++errors;
    std::cout << "error_length_collision=" << collision
              << " status=" << status.ToString() << " sentinel=" << sentinel()
              << std::endl;
    disarm();
    // Destroy engines first: CPU transport joins every worker before these
    // test-owned static buffers leave scope. Failed batches are retained.
    tent_destroy_engine(sender);
    sender = nullptr;
    tent_destroy_engine(receiver);
    receiver = nullptr;
    std::cout << "CPU_NATIVE_ENGINES_JOINED failures=" << errors << std::endl;
    return errors ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    disarm();
    if (sender)
      tent_destroy_engine(sender);
    if (receiver)
      tent_destroy_engine(receiver);
    return 2;
  }
}
