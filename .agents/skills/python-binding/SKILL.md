---
name: python-binding
description: Modify OrbitKV PyO3 APIs, native client ownership, Python type stubs or wheel packaging. Use for native binding and installed-artifact changes in this repository.
---

# OrbitKV Python bindings and adapters

Read `AGENTS.md` from the Git root. Keep Python responsible for framework callbacks,
layout inspection, GPU allocations and ownership handoff; shared cache state
machines, batching, waiting and execution belong to the existing Rust owners.

- Public bindings: `python/src/lib.rs`, `python/src/client.rs` and
  `python/orbitkv/orbitkv.pyi`. Update stubs when an exposed API changes. Verify
  actual call sites rather than restoring removed aliases or client facades.
- Common client ownership: `crates/orbitkv-channel/src/cache_client.rs`.
- `publish_progress()` is a local, immediate activity observation, independent
  of blocked descriptor/lifecycle locks. Contention is an unavailable error.
  Its active interval, overdue bit, retained count and pidfd liveness are never
  completion receipts. Keep engine pages held while synchronous `save` runs;
  callers must not turn diagnostic failure into save completion.
- Engine callback changes: use `.agents/skills/engine-integration/SKILL.md` and
  `docs/adapters.md`; release targets and current qualification are distinct.
- Model/layout identity and hybrid recovery are shared contracts. Do not equate
  compatible API shapes with cross-engine byte compatibility.
- `register_context_batch` returns an opaque native `GpuRegistration` and takes
  client-local CUDA ordinals and retained tensors. Publish calls
  `save(registration, saves)`; adapters capture that object at producer handoff,
  not when a delayed queue worker executes. Retired or foreign-client authority
  is rejected. The object retains exporters but is neither a page-slot
  generation nor a device completion receipt.
  Resolve their full physical GPU UUIDs; the native owner retains the Manager's
  resolved ordinal for Publish/Restore. Do not reuse the client's ordinal on the
  Manager or convert a UUID into an ordinal with physical `nvidia-smi` ordering.
  Lifecycle protocol 9 requires matching installed client and Manager artifacts.
  The selected Torch 2.13 cudaMalloc IPC storage handle has a version/type
  envelope around the native CUDA handle. Keep the validated 66-byte version-3
  cudaMalloc export unchanged through the importer; bare legacy and expandable
  handles are outside this owner contract. Real installed export/import controls
  remain necessary even when synthetic metadata validation passes.
  A normal return from Torch's best-effort counter release is not repayment
  proof. The Rust attempt owner must retain the original counter file and verify
  identity plus the exclusive slot's one-to-zero transition; missing, replaced,
  silent or failed repayment remains charged without retry. This counter proof
  does not establish CUDA/RDMA termination.
  Installed registration gates persist the immutable `ipc_credit_evidence`
  handoff snapshot, or `retained_credit_evidence(instance_id)` on failure. The
  latter reads original held counter descriptors and distinguishes current-path
  identity, missing paths and read errors. Neither API authorizes repayment or
  physical reuse; preserve the original failure and quarantine on uncertainty.

Use the gate table in `AGENTS.md`: default Python tests must remain source-only;
native lifecycle changes need process integration, and consumed adapter changes
need the relevant engine correctness/restart gate. Run `benches/tests` when
changing benchmark helpers. Put runtime outputs outside the checkout.

Use `scripts/build-wheel.sh` for a complete installed artifact; `maturin develop`
is a development build. See `docs/releases.md` for CUDA variants, bundled native
dependencies and installed-package checks. Freeze native artifacts before GPU
tests: Cargo can restage shared libraries used by live Managers.
