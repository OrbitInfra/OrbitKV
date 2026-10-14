// Actual TENT CPU/TCP with one worker; buffers remain valid until process exit.
#include "tent/platform/cpu.h"
#include "tent/runtime/control_plane.h"
#include "tent/transfer_engine.h"
#ifdef BASELINE_WORKER_CACHE
#include "tent/transport/tcp/tcp_transport.h"
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace {
std::array<unsigned char, 8192> source{}, target{}, original{};
std::atomic<unsigned> dropped_pushes{0};
void require(bool condition, const char *reason) {
  if (!condition) {
    std::cerr << reason << "\nCPU isolation; no teardown proof.\n";
    std::_Exit(2);
  }
}
tent_engine_t engine(uint16_t port = 0) {
  const auto p = std::to_string(port);
  for (auto [k, v] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"metadata_type", "p2p"},
           {"rpc_server_hostname", "127.0.0.1"},
           {"local_segment_name", ""},
           {"transports/tcp/enable", "true"},
           {"transports/rdma/enable", "false"},
           {"transports/nvlink/enable", "false"},
           {"transports/shm/enable", "false"},
           {"transports/mnnvl/enable", "false"},
           {"enable_runtime_queue", "false"},
           {"enable_auto_failover_on_poll", "false"},
           {"transports/tcp/max_concurrent_tasks", "1"},
           {"transports/tcp/max_retry_count", "0"},
           {"rpc_server_threads", "2"}})
    tent_set_config(k, v);
  tent_set_config("rpc_server_port", p.c_str());
  auto e = tent_create_engine();
  require(e && tent_available(e) == 1, "native engine unavailable");
  return e;
}
tent_status_t settled(tent_engine_t e, tent_batch_id_t b, size_t index = 0) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  tent_status_t s{};
  do {
    require(tent_task_status(e, b, index, &s) == 0, "status unavailable");
    if (s.status == STATUS_COMPLETED || s.status == STATUS_FAILED)
      return s;
    require(std::chrono::steady_clock::now() < end, "native status timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (true);
}
} // namespace
namespace mooncake::tent {
#ifdef BASELINE_WORKER_CACHE
// Counterfactual: the prior worker lookup ignores the grant expectation.
Status TcpTransport::findRemoteSegment(uint64_t address, uint64_t length,
                                       uint64_t id, std::string &server,
                                       std::string &epoch,
                                       const std::string &) {
  return metadata_->segmentManager().withCachedSegment(
      id, [&](SegmentDesc *segment) {
        auto *buffer = segment->findBuffer(address, length);
        if (!buffer)
          return Status::NeedsRefreshCache("Missing source buffer");
        server = segment->rpc_server_addr;
        epoch = buffer->source_epoch;
        return Status::OK();
      });
}
#endif
// A test-only lost-push control. Transport metadata refresh must recover
// without it.
void ControlClient::notifySegmentUpdatedAsync(
    const std::string &, const std::string &,
    const onNotifySegmentUpdateFailure &) {
  ++dropped_pushes;
}
} // namespace mooncake::tent
int main(int argc, char **argv) {
  require(argc == 2, "expected scenario");
  const std::string scenario = argv[1];
  require(scenario == "warm" || scenario == "lost-push" ||
              scenario == "preflight-race" || scenario == "abi-mixed",
          "unknown scenario");
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = (i * 17 + 31) % 251;
  original = source;
  target.fill(0x5a);
  auto owner = engine(), requester = engine();
  require(tent_register_memory(owner, source.data(), source.size()) == 0,
          "source register");
  require(tent_register_memory(requester, target.data(), target.size()) == 0,
          "destination register");
  char endpoint[256]{}, old_epoch[33]{}, new_epoch[33]{};
  require(tent_segment_name(owner, endpoint, sizeof(endpoint)) == 0,
          "endpoint");
  const auto port = static_cast<uint16_t>(std::stoi(
      std::string(endpoint).substr(std::string(endpoint).rfind(':') + 1)));
  require(tent_source_epoch(owner, old_epoch, sizeof(old_epoch)) == 0,
          "source owner epoch");
  tent_segment_id_t segment{};
  require(tent_open_segment(requester, &segment, endpoint) == 0, "open source");
  tent_request_t request{};
  request.opcode = OPCODE_READ;
  request.source = target.data();
  request.target_id = segment;
  request.target_offset = reinterpret_cast<uint64_t>(source.data());
  request.length = source.size();
  request.transport_hint = TRANSPORT_TCP;
  std::memcpy(request.expected_source_epoch, old_epoch, 33);
  auto warm = tent_allocate_batch(requester, 1);
  require(tent_submit(requester, warm, &request, 1) == 0, "warm submit");
  auto s = settled(requester, warm);
  require(s.status == STATUS_COMPLETED && target == original,
          "warm old byte oracle");
  require(tent_free_batch(requester, warm) == 0, "free completed warm batch");
  require(tent_validate_remote_epoch(requester, segment, old_epoch) == 0,
          "successful preflight before restart");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (scenario == "abi-mixed") {
    target.fill(0x5a);
    std::array<tent_request_t, 2> entries{request, request};
    entries[0].length = 4096;
    entries[1].length = 4096;
    entries[1].source = target.data() + 4096;
    entries[1].target_offset += 4096;
    std::memset(entries[1].expected_source_epoch, 'f', 32);
    entries[1].expected_source_epoch[32] = 0;
    auto batch = tent_allocate_batch(requester, 2);
    require(tent_submit(requester, batch, entries.data(), 2) == 0,
            "mixed ABI submit");
    auto a = settled(requester, batch, 0), b = settled(requester, batch, 1);
    require(a.status == STATUS_COMPLETED && b.status == STATUS_FAILED,
            "different epochs merged or wrong C ABI stride");
    require(std::equal(target.begin(), target.begin() + 4096, original.begin()),
            "valid mixed bytes");
    require(std::all_of(target.begin() + 4096, target.end(),
                        [](auto c) { return c == 0x5a; }),
            "invalid mixed destination modified");
    std::cout << "scenario=abi-mixed request_size=" << sizeof(tent_request_t)
              << " epoch_offset="
              << offsetof(tent_request_t, expected_source_epoch)
              << " valid_bytes=4096 invalid_target_sentinel=1 "
                 "failed_batch_retained=1 result=PASS\n";
    return 0;
  }
  uint64_t active{};
  char closed[33]{};
  require(tent_close_source(owner, 1000, closed, &active) == 0,
          "source real CPU drain");
  require(tent_unregister_memory(owner, source.data(), source.size()) == 0,
          "unregister old source");
  tent_destroy_engine(owner);
  owner = engine(port);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = (i * 29 + 73) % 251;
  require(tent_register_memory(owner, source.data(), source.size()) == 0,
          "replacement register");
  char same_endpoint[256]{};
  tent_segment_name(owner, same_endpoint, sizeof(same_endpoint));
  require(std::strcmp(endpoint, same_endpoint) == 0, "endpoint changed");
  require(tent_source_epoch(owner, new_epoch, sizeof(new_epoch)) == 0 &&
              std::strcmp(new_epoch, old_epoch) != 0,
          "new source identity");
  target.fill(0x5a);
  if (scenario == "preflight-race") {
    auto batch = tent_allocate_batch(requester, 1);
    require(tent_submit(requester, batch, &request, 1) == 0, "race submit");
    s = settled(requester, batch);
    require(s.status == STATUS_FAILED,
            "old expectation upgraded after successful preflight");
    require(std::all_of(target.begin(), target.end(),
                        [](auto c) { return c == 0x5a; }),
            "race accessed replacement payload");
    std::cout
        << "scenario=preflight-race successful_preflight_before_restart=1 "
           "target_sentinel=1 failed_batch_retained=1 result=PASS\n";
    return 0;
  }
  require(tent_validate_remote_epoch(requester, segment, new_epoch) == 0,
          "fresh grant preflight");
  std::memcpy(request.expected_source_epoch, new_epoch, 33);
  auto batch = tent_allocate_batch(requester, 1);
  require(tent_submit(requester, batch, &request, 1) == 0,
          "fresh warmed-worker submit");
  s = settled(requester, batch);
  require(s.status == STATUS_COMPLETED && target == source,
          "worker retained stale epoch after fresh grant");
  require(dropped_pushes.load() > 0, "lost update push control not exercised");
  require(tent_free_batch(requester, batch) == 0, "free fresh completed batch");
  require(tent_close_source(owner, 1000, closed, &active) == 0 &&
              tent_close_source(requester, 1000, closed, &active) == 0,
          "normal source drain");
  require(tent_unregister_memory(owner, source.data(), source.size()) == 0 &&
              tent_unregister_memory(requester, target.data(), target.size()) ==
                  0,
          "normal unregister");
  tent_destroy_engine(owner);
  tent_destroy_engine(requester);
  std::cout << "scenario=" << scenario
            << " one_warmed_worker=1 same_endpoint_address=1 dropped_pushes="
            << dropped_pushes.load() << " exact_bytes=" << source.size()
            << " normal_teardown=1 result=PASS\n";
}
