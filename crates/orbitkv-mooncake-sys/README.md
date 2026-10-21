# OrbitKV Mooncake Sys

`orbitkv-mooncake-sys` is the native artifact and raw ABI boundary for Mooncake
TENT. It does not build or load the legacy Transfer Engine backend.

The crate has three responsibilities:

1. build the pinned Mooncake integration commit from `third-party/mooncake`;
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

- repository: [OrbitInfra/Mooncake](https://github.com/OrbitInfra/Mooncake)
- branch: `orbitkv-integration`
- pinned commit: `12cbf549b6351c800de0f94a480276d216b9a43b`
- upstream base: `v0.3.13.post1`, `719735896c86b56fabec6cf3e825fb2ea640597a`

The gitlink selects an immutable integration commit; builds never resolve a moving
branch head. The fork carries the previously consumed TCP copy/context,
worker ownership, source drain/epoch and registration ownership fixes as ordinary
commits. Cargo copies this source into its private build directory and invokes
CMake directly. It does not apply patches or rewrite the submodule.
See the [integration branch guide](https://github.com/OrbitInfra/Mooncake/blob/12cbf549b6351c800de0f94a480276d216b9a43b/ORBITKV_INTEGRATION.md)
for upstream origins and the update policy. The initial native source is identical
to the old patched build; this migration grants no new hardware qualification.

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
Record the integration commit and artifact hashes for an override; a compatible
symbol set alone does not establish its source provenance or runtime qualification.
