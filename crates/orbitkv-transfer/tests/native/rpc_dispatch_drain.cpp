// Real RPC server/global executor ownership. One isolated CPU process per case.
#include "tent/rpc/rpc.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
using mooncake::tent::CoroRpcAgent;
namespace {
void require(bool value, const char *message) {
  if (!value) {
    std::cerr << message << "\nCPU control isolation; no teardown proof.\n";
    std::_Exit(2);
  }
}
void wait_flag(const std::atomic<bool> &flag) {
  auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!flag.load()) {
    require(std::chrono::steady_clock::now() < deadline,
            "control barrier timeout");
    std::this_thread::sleep_for(1ms);
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected queued/running/throw scenario");
    std::string scenario = argv[1];
    require(scenario == "queued" || scenario == "running" ||
                scenario == "throw",
            "unknown scenario");
    // Configure only this test process's otherwise uninitialized global pool.
    auto *executor = coro_io::get_global_block_executor(1);
    std::atomic<bool> blocker_entered{false}, release{false}, entered{false},
        finished{false};
    if (scenario == "queued") {
      asio::post(executor->get_asio_executor(), [&] {
        blocker_entered = true;
        while (!release.load())
          std::this_thread::sleep_for(1ms);
      });
      wait_flag(blocker_entered);
    }
    CoroRpcAgent server, client;
    server.registerFunction(
        31337,
        [&](std::string_view request, std::string &response) {
          entered = true;
          if (scenario != "queued")
            while (!release.load())
              std::this_thread::sleep_for(1ms);
          require(request == "owned-request",
                  "offloaded request lifetime changed");
          response = "owned-response";
          finished = true;
          if (scenario == "throw")
            throw std::runtime_error("controlled callback exception");
        },
        true);
    uint16_t port = 0;
    require(server.start(port, false, 2).ok(), "server start");
    auto call = std::async(std::launch::async, [&] {
      std::string response;
      return client.call("127.0.0.1:" + std::to_string(port), 31337,
                         "owned-request", response);
    });
    if (scenario == "queued") {
      std::this_thread::sleep_for(100ms);
      require(!entered.load(), "queued control ran before executor release");
    } else
      wait_flag(entered);
    auto active = server.closeDispatch(0ms);
    require(active == 1, "dispatch ownership was not acquired before offload");
    auto stopped =
        std::async(std::launch::async, [&] { return server.stop(); });
    require(stopped.wait_for(100ms) == std::future_status::timeout,
            "stop returned before actual offload completion");
    release = true;
    require(stopped.wait_for(5s) == std::future_status::ready &&
                stopped.get().ok(),
            "actual dispatch drain");
    require(finished.load(), "callback was destroyed before completion");
    require(call.wait_for(5s) == std::future_status::ready,
            "client actual wait");
    auto result = call.get();
    if (scenario == "throw")
      require(!result.ok(), "callback exception reported as success");
    std::promise<void> still_shared;
    auto shared = still_shared.get_future();
    asio::post(executor->get_asio_executor(),
               [&] { still_shared.set_value(); });
    require(shared.wait_for(5s) == std::future_status::ready,
            "shared executor was stopped");
    std::cout << "scenario=" << scenario
              << " dispatch_before_offload=" << active
              << " callback_finished=1 source_owner_waited=1 "
                 "shared_executor_live=1 result=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what()
              << "\nCPU control isolation; no native teardown proof claimed.\n";
    // A failed control may still own queued closures. Don't destroy their
    // stack.
    std::_Exit(2);
  }
}
