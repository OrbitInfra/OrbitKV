// Withhold a result at the actual C ABI boundary; all other native work is
// real.
#include "tent/transfer_engine.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static std::atomic<unsigned long long> registers{0}, unregisters{0},
    destroys{0}, source_closes{0};
static std::atomic<unsigned long long> unregister_ns{0};
static bool scenario(const char *value) {
  auto name = std::getenv("ORBITKV_TEST_SOURCE_OWNER_CASE");
  return name && std::strcmp(name, value) == 0;
}
static void* registration_control = nullptr;
static tent_engine_t failed_registration_engine = nullptr;
template <typename Function> Function native(const char *name) {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}
int tent_register_memory_ex(tent_engine_t engine, void *address, size_t length,
                            tent_memory_options_t *options) {
  auto call = ++registers;
  if (scenario("native-register-partial")) {
    auto path = std::getenv("ORBITKV_REGISTER_CONTROL_LIBRARY");
    if (!path) return -1;
    registration_control = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!registration_control) return -1;
    using Install = void (*)(void*);
    using Refuse = void (*)(void*, int);
    auto install = reinterpret_cast<Install>(
        dlsym(registration_control, "orbitkv_test_install_unregister_control"));
    auto refuse = reinterpret_cast<Refuse>(
        dlsym(registration_control, "orbitkv_test_set_registration_refusal"));
    if (!install || !refuse) return -1;
    install(engine);
    refuse(engine, 2);
    failed_registration_engine = engine;
  }
  if (scenario("partial") && call == 2)
    return -1;
  auto fn =
      native<int (*)(tent_engine_t, void *, size_t, tent_memory_options_t *)>(
          "tent_register_memory_ex");
  return fn ? fn(engine, address, length, options) : -1;
}
extern "C" int tent_close_source(tent_engine_t engine, uint64_t timeout,
                                 char *epoch, uint64_t *active) {
  ++source_closes;
  auto fn = native<int (*)(tent_engine_t, uint64_t, char *, uint64_t *)>(
      "tent_close_source");
  if (!fn)
    return -1;
  auto status = fn(engine, timeout, epoch, active);
  if (status != 0)
    return status;
  if (scenario("unsupported") || scenario("partial"))
    return 2;
  if (scenario("still-owned"))
    return 1;
  return status;
}
int tent_unregister_memory(tent_engine_t engine, void *address, size_t length) {
  auto call = ++unregisters;
  if (scenario("unregister-error") ||
      (scenario("unregister-after-first") && call == 2))
    return -1;
  auto fn =
      native<int (*)(tent_engine_t, void *, size_t)>("tent_unregister_memory");
  auto result = fn ? fn(engine, address, length) : -1;
  if (result == 0)
    unregister_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  return result;
}
void tent_destroy_engine(tent_engine_t engine) {
  ++destroys;
  auto fn = native<void (*)(tent_engine_t)>("tent_destroy_engine");
  if (fn)
    fn(engine);
}
extern "C" unsigned long long orbitkv_test_registers() {
  return registers.load();
}
extern "C" unsigned long long orbitkv_test_unregisters() {
  return unregisters.load();
}
extern "C" unsigned long long orbitkv_test_destroys() {
  return destroys.load();
}
extern "C" unsigned long long orbitkv_test_source_closes() {
  return source_closes.load();
}
extern "C" unsigned long long orbitkv_test_unregister_ns() {
  return unregister_ns.load();
}

extern "C" int orbitkv_test_pending_registration() {
  if (!registration_control || !failed_registration_engine) return 0;
  using Pending = int (*)(void*);
  auto pending = reinterpret_cast<Pending>(dlsym(
      registration_control, "orbitkv_test_partial_registration_pending"));
  return pending ? pending(failed_registration_engine) : 0;
}
