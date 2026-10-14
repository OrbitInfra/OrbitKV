# OrbitKV Transfer

`orbitkv-transfer` is OrbitKV's single remote byte-movement implementation. It
wraps the pinned upstream Mooncake TENT engine and contains no private verbs
stack or legacy Transfer Engine fallback.

OrbitKV owns state identity, recovery completeness, replica choice, leases, and
generation validation. Mooncake owns segment discovery, memory registration,
RDMA/TCP selection, multi-rail routing, transfer completion, and notification.

```text
OrbitKV authorized transfer plan
        │
        ▼
Mooncake TENT segment + remote address + batch operation
        │
        ├── RDMA / GPUDirect when available
        └── TCP fallback or forced TCP for validation
```

The public Rust API is deliberately small:

- `TransferEngine::new` creates one upstream Mooncake TENT engine.
- `register_memory_owned` returns an RAII token for stable host or device memory.
- `submit_and_wait` executes a READ or WRITE batch.
- `close_source` permanently closes Rust admission and the native source gate;
  only a CPU/TCP handler receipt and empty requester admission permit success.
  Other paths keep their registrations and backing owners.
- `submit_and_notify` couples a batch to a peer notification.
- generation-scoped notification waits retain messages for concurrent requests
  and wake cancelled/replaced waiters; `send_notification` emits completion and
  failure signals.

The upstream source is pinned to stable release `v0.3.13.post1` by the
`third-party/mooncake` submodule. `orbitkv-mooncake-sys` builds and stages
`libtent_shared.so`,
`libmooncake_common.so`, and `libasio.so`; the libraries use an `$ORIGIN`
runpath so the three files can be bundled together in a wheel or container.
The sys crate is the only raw C ABI boundary. This crate contains the
OrbitKV-facing safe types, segment cache, batch draining, timeout semantics,
and notification API.

## Build and smoke test

Initialize the nested source dependencies once:

```bash
git submodule update --init --recursive third-party/mooncake
```

Then run:

```bash
cargo test -p orbitkv-transfer --no-default-features --features cuda-13
```

The host-safe smoke test forces Mooncake TCP and verifies registration, segment
opening, WRITE completion, and byte equality. RDMA/GPUDirect qualification is a
separate hardware gate.

For the source-shutdown candidate, use the private patched headers from the
Cargo native `OUT_DIR`, its matching CPU libraries, and the actual fetched YLT
source. Keep each output directory outside the checkout and new:

```bash
python crates/orbitkv-transfer/tests/native/run_source_drain.py \
  --source /path/to/out/mooncake-source --runtime /path/to/frozen/runtime-cpu \
  --ylt /path/to/out/native/_deps/yalantinglibs-src --output /var/tmp/source-drain-001
```

This gate executes delayed CPU copy, partial C ABI submission, cancellation
request, restart/epoch rejection, unversioned Delegate rejection and failed
entropy creation. TCP cancellation remains unsupported; the original admitted
copy must still finish. The copy barrier and partial-submit fault are test-only.
Pin/Unpin rejection before and after closure is also checked. `--gate dispatch`
runs actual queued/running/throwing handlers on the global YLT executor and
verifies owned callback drain and that the shared executor remains usable.
Each process records actual mapped libraries, identity and wait/reap. A timeout
is forced CPU control cleanup and fails the gate.

The actual Rust pool-owner control is
`crates/orbitkv-core/tests/native/run_source_owner.py`. Pass a built core unit
binary, matching patched source/CPU runtime, CUDA include directory and a new
external output directory. It uses real memfd mappings and the existing
`MooncakeTransport`, with test-only host-registration and withheld-C ABI results
for normal, unsupported, still-owned, unregister-error and partial construction.
It proves retention/ordering at those owner boundaries; it performs no CUDA
registration, GPU operation or hardware reclamation qualification.
