// Test-owned valid buffers and a controlled Platform exercise the real TCP
// worker. No helper override is linked into the production native library.
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "tent/platform/cpu.h"
#include "tent/runtime/control_plane.h"
#include "tent/transport/tcp/tcp_transport.h"

namespace {
std::atomic<bool> throw_copy{false};
std::atomic<bool> fail_copy{false};
std::atomic<unsigned> throw_count{0};
std::array<unsigned char, 8192> buffers;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
} // namespace
namespace mooncake::tent {
class ControlledPlatform final : public CpuPlatform {
public:
  ControlledPlatform() : CpuPlatform(std::make_shared<Config>()) {}
  MemoryType getMemoryType(void *ptr) override {
    return ptr == buffers.data() ? MTYPE_CUDA : MTYPE_CPU;
  }
  Status copy(void *dst, void *src, size_t length) override {
    if (throw_copy.exchange(false)) {
      ++throw_count;
      throw std::runtime_error("controlled worker copy exception");
    }
    if (fail_copy.load())
      return Status::InternalError("controlled worker copy status");
    std::memcpy(dst, src, length);
    return Status::OK();
  }
  const std::string type() const override { return "cuda"; }
};
Platform &Platform::getLoader(std::shared_ptr<Config>) {
  static ControlledPlatform platform;
  return platform;
}
} // namespace mooncake::tent

int main(int argc, char **argv) {
  using namespace mooncake::tent;
  using namespace std::chrono_literals;
  try {
    auto config = std::make_shared<Config>();
    config->set("transports/tcp/max_retry_count", 0);
    config->set("transports/tcp/max_concurrent_tasks", 1);
    auto metadata = std::make_shared<ControlService>("p2p", "", nullptr);
    uint16_t port = 0;
    require(metadata->start(port, false, 2).ok(), "metadata start");
    std::string endpoint = "127.0.0.1:" + std::to_string(port);
    require(metadata->segmentManager()
                .updateLocal([&](SegmentDesc &desc) {
                  desc.type = SegmentType::Memory;
                  desc.rpc_server_addr = endpoint;
                  BufferDesc buffer;
                  buffer.addr = reinterpret_cast<uint64_t>(buffers.data());
                  buffer.length = buffers.size();
                  std::get<MemorySegmentDesc>(desc.detail)
                      .buffers.push_back(buffer);
                  return Status::OK();
                })
                .ok(),
            "registered local metadata");
    unsigned failures = 0;
    // A callback paused after terminal publication must remain owned even
    // if the caller immediately frees the batch that contained the task.
    for (bool failed : std::vector<bool>{false, true}) {
      if (argc > 1)
        break;
      TcpTransport transport;
      require(transport.install(endpoint, metadata, nullptr, config).ok(),
              "transport install");
      Transport::SubBatchRef batch = nullptr;
      require(transport.allocateSubBatch(batch, 1).ok(), "allocate batch");
      std::promise<void> entered, resume;
      auto entered_future = entered.get_future();
      auto wait = resume.get_future().share();
      std::weak_ptr<int> lifetime;
      {
        auto marker = std::make_shared<int>(7);
        lifetime = marker;
        batch->notify_progress = [marker, wait, &entered](BatchID) {
          // Locals remain safe even in the original failed control.
          auto local_wait = wait;
          entered.set_value();
          local_wait.wait();
        };
      }
      Request request{};
      request.opcode = Request::WRITE;
      request.source = buffers.data();
      request.target_id = LOCAL_SEGMENT_ID;
      request.target_offset = reinterpret_cast<uint64_t>(buffers.data() + 4096);
      request.length = 4096;
      request.transport_hint = TCP;
      for (size_t index = 0; index < 4096; ++index)
        buffers[index] = (index * 17) % 251;
      std::fill(buffers.begin() + 4096, buffers.end(), 0x5a);
      fail_copy = failed;
      require(transport.submitTransferTasks(batch, {request}).ok(),
              "submit callback control");
      require(entered_future.wait_for(5s) == std::future_status::ready,
              "callback not reached");
      TransferStatus status{};
      require(transport.getTransferStatus(batch, 0, status).ok(),
              "callback status");
      if (status.s != (failed ? FAILED : COMPLETED))
        ++failures;
      const bool payload = std::equal(buffers.begin(), buffers.begin() + 4096,
                                      buffers.begin() + 4096);
      const bool sentinel =
          std::all_of(buffers.begin() + 4096, buffers.end(),
                      [](auto value) { return value == 0x5a; });
      if ((!failed && !payload) || (failed && !sentinel))
        ++failures;
      require(transport.freeSubBatch(batch).ok(), "free completed CPU batch");
      const bool owned = !lifetime.expired();
      std::cout << "terminal=" << status.s
                << " callback_owned_after_batch_free=" << owned << std::endl;
      if (!owned)
        ++failures;
      resume.set_value();
      require(transport.uninstall().ok(), "join callback worker");
      require(lifetime.expired(),
              "callback owner retained after worker joined");
    }
    fail_copy = false;
    // Throwing in client staging is a real worker exception, not a synthetic
    // task-status assignment. After worker join, inspect without racing it.
    TcpTransport transport;
    require(transport.install(endpoint, metadata, nullptr, config).ok(),
            "exception install");
    Transport::SubBatchRef batch = nullptr;
    require(transport.allocateSubBatch(batch, 1).ok(), "exception batch");
    Request request{};
    request.opcode = Request::WRITE;
    request.source = buffers.data();
    request.target_id = LOCAL_SEGMENT_ID;
    request.target_offset = reinterpret_cast<uint64_t>(buffers.data() + 4096);
    request.length = 4096;
    request.transport_hint = TCP;
    throw_copy = true;
    require(transport.submitTransferTasks(batch, {request}).ok(),
            "exception submit");
    const auto exception_deadline = std::chrono::steady_clock::now() + 5s;
    while (!throw_count.load()) {
      require(std::chrono::steady_clock::now() < exception_deadline,
              "worker exception injection not exercised");
      std::this_thread::sleep_for(1ms);
    }
    require(transport.uninstall().ok(), "join exception worker");
    TransferStatus status{};
    require(transport.getTransferStatus(batch, 0, status).ok(),
            "exception task status");
    std::cout << "worker_exception_status=" << status.s
              << " bytes=" << status.transferred_bytes << std::endl;
    if (status.s != FAILED || status.transferred_bytes != 0)
      ++failures;
    require(transport.freeSubBatch(batch).ok(),
            "post-join exception batch free");
    std::cout << "TCP_WORKER_CONTROL_FAILURES=" << failures << std::endl;
    return failures ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
