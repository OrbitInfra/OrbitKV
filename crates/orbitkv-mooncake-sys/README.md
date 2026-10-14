# OrbitKV Mooncake Sys

`orbitkv-mooncake-sys` is the native artifact and raw ABI boundary for Mooncake
TENT. It does not build or load the legacy Transfer Engine backend.

The crate has three responsibilities:

1. build the pinned stable Mooncake release from `third-party/mooncake`;
2. build the `tent_shared` target and stage `libtent_shared.so`,
   `libmooncake_common.so`, and `libasio.so` in
   a relocatable runtime directory;
3. load and expose the C ABI required by `orbitkv-transfer`.

It deliberately contains no OrbitKV cache identity, lease, retry, timeout, or
transfer-plan policy. Those belong to `orbitkv-transfer` and `orbitkv-core`.

## Upstream Rust bindings

Mooncake's older Rust wrapper targets the compatibility Transfer Engine API and
is not published on crates.io in the pinned release. OrbitKV binds the stable
TENT C ABI dynamically so cancellation, explicit memory options, notification
and rail-load semantics remain available without linking C++ into every binary.

OrbitKV therefore keeps this small sys crate for reproducible native builds and
relocatable shared-library packaging. The higher-level API remains isolated in
`orbitkv-transfer`, so the raw bindings can be replaced by the official crate
later without changing cache or framework code.

## Native source

- release: `v0.3.13.post1`
- commit: `719735896c86b56fabec6cf3e825fb2ea640597a`

The private build copy applies the checked TCP copy/context and worker-ownership
backports plus the unreleased `tcp-source-admission-drain.patch` candidate.
`tent_close_source` closes a whole engine permanently: status 0 is CPU/TCP
handler drain, 1 remains owned, 2 has an unsupported physical path, and negative
status is an error. It never drains requester batches or fences one RDMA ticket.
CUDA builds refuse this physical receipt. Matching TCP peers consume the native
engine epoch and READ success envelope; there is no old-wire fallback. Loading
a prebuilt runtime without the new symbol fails before creating an engine.
The grant-epoch candidate additionally requires `tent_source_epoch` and
`tent_validate_remote_epoch`, and appends the expected source epoch to each
`tent_request_t`. The source reply supplies the expectation; TCP never replaces
it with a restarted source's value. Rust preflight refreshes TENT metadata before
batch allocation. The wire/ABI requires matching bundles and is unreleased.

To use compatible prebuilt libraries instead of invoking CMake, set
`ORBITKV_MOONCAKE_LIB_DIR` to a directory containing the three TENT runtime
shared objects. A directory containing only `libtransfer_engine.so` is rejected.
