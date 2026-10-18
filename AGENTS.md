# OrbitKV Agent Guide

This file provides guidance for agents working in the OrbitKV repository.

## Current execution and review

Use [the completion plan](docs/completion-plan.md) for the next delivery sequence.
It is the only execution queue; architecture documents explain contracts,
not independent completion claims. Do not recreate a root TODO or parallel roadmap. Check the current code before reviving an old
unchecked item. Mark implemented-but-unqualified work separately from missing code.

For the current handoff, the implementation agent delivers one stage or coherent
substage at a time; the reviewing agent independently inspects its diff, reruns
the relevant gates and checks the evidence before accepting dependent work.
Return the commit, changed contracts, tests, external artifact locations and
remaining limits. A missing hardware gate stays unqualified, not passed.

## Documentation, layout and experiment evidence

- Follow [LMCache's project organization](https://github.com/LMCache/LMCache) and
  [documentation structure](https://docs.lmcache.ai/) for the README, quickstart,
  deployment recipes, compatibility matrix, benchmark methodology and contributor
  navigation. Adapt these to OrbitKV's Rust ownership boundaries; do not copy
  LMCache's implementation layers or claim its deployment support as our own.
- Preserve OrbitKV's website typography, colors, components and visual identity.
  Maintain technical content in `docs/` and render it on the website. Diagrams
  must distinguish implemented paths, experimental paths and future work.
- Keep experiment results outside the source checkout and Git history: raw
  JSON/CSV, logs, traces, run manifests, generated plots and per-run reports belong
  in an explicitly selected external artifact directory or CI/release artifacts.
  Do not force-add ignored output. Keep failed runs and controls with their evidence.
- Keep benchmark programs, reusable workload definitions, small deterministic
  test fixtures and reproduction instructions in the repo. Public README/docs
  may contain concise reviewed conclusions and stable evidence links; generated
  result collections and duplicate report tables are not source code. Publish
  measured charts from versioned external artifacts. Record revisions, hardware,
  budgets, correctness and uncertainty before claiming a performance advantage.
- Migrate existing tracked result collections only after verifying their archive
  and updating every documentation, website and reproduction reference. Do not
  delete the sole copy of evidence or rewrite Git history to hide old results.

## Repository Codex skills

Project skills live in `.agents/skills/` and are versioned with the code. Launch
Codex inside this repository so repository-scoped discovery applies. Read the
relevant `SKILL.md` for Python bindings, Manager operations, version changes or
released-engine integration and lifecycle diagnosis; resolve repository paths from the Git root. Keep one
maintained skill per workflow and update it when the consumed API changes.

## Project Overview

OrbitKV is a framework-neutral state cache with compiled recovery requirements
for LLM inference; general physical planning remains future work. Engine pins are vLLM `0.31.0`
and SGLang `0.5.21`; see S5.1 for the 0.31.0 qualification scope and historical
0.29.0 evidence. The SGLang UnifiedRadixCache linker is implemented, while
public lifecycle integration and broader deployment qualification remain open.

- Single-node KV cache offloading between GPU and host memory
- Cross-node KV cache sharing via Mooncake Transfer Engine (RDMA/TCP)
- Prefix cache reuse for repeated requests
- Python bindings and connectors for inference frameworks
- Framework-neutral state identity and recovery contracts

## Repository Layout

```text
orbitkv/
├── crates/
│   ├── orbitkv-state/           # Framework-neutral state and recovery contracts
│   ├── orbitkv-channel/         # iceoryx2/UDS process transport
│   ├── orbitkv-common/           # Process logging and peer-connection defaults
│   ├── orbitkv-core/             # Storage owners, planning, costs and execution
│   ├── orbitkv-proto/            # Protobuf and gRPC definitions
│   ├── orbitkv-server/           # Cache Manager orchestration and protocol adapters
│   ├── orbitkv-catalog/          # Complete local global index
│   ├── orbitkv-mooncake-sys/     # Pinned native build and dynamic C ABI
│   └── orbitkv-transfer/         # Mooncake transfer wrapper
├── python/                       # PyO3 package and framework adapters
├── third-party/                  # Pinned Mooncake, vLLM, and SGLang sources
├── examples/                     # Runnable usage examples
├── benches/                       # Benchmark code, workloads and reproduction
├── docs/                         # Architecture and completion plan
├── scripts/                      # Project helper scripts
├── .agents/skills/                # Repository-scoped Codex workflows
└── prek.toml                     # Local check configuration
```

## Where To Change Code

| Target | Location |
|--------|----------|
| State identity and recovery contracts | `crates/orbitkv-state/` |
| Inference-to-Cache-Manager process channel | `crates/orbitkv-channel/` |
| Process logging and peer-connection defaults | `crates/orbitkv-common/` |
| NUMA topology and affinity | `crates/orbitkv-core/src/memory/numa.rs` |
| Cache reuse statistics | `crates/orbitkv-server/src/metric/hll.rs` |
| Core engine and storage path | `crates/orbitkv-core/` |
| Batch replica evidence, completion targets and SSD/peer source plans | `crates/orbitkv-core/src/planning/` |
| GPU storage codecs, CPU SIMD fallback and encoding metadata | `crates/orbitkv-core/src/codec/` |
| gRPC protocol changes | `crates/orbitkv-proto/` |
| Cache Manager cache operations and process endpoint | `crates/orbitkv-server/src/cache/`, `endpoint/` |
| Local global index | `crates/orbitkv-catalog/` |
| etcd member registration, renewal and Watch | `crates/orbitkv-server/src/cluster/` |
| Cached membership and remote admission | `crates/orbitkv-catalog/src/membership.rs` |
| Mooncake remote transfer path | `crates/orbitkv-transfer/` |
| PyO3 bindings | `python/src/` |
| Python package and helpers | `python/orbitkv/` |
| vLLM connector | `python/orbitkv/vllm/` |
| SGLang adapter | `python/orbitkv/sglang/` |
| Cache client ownership | `crates/orbitkv-channel/src/cache_client.rs` |
| Python connection configuration | `python/orbitkv/client/` |

## Key Entry Points

- `crates/orbitkv-state/src/lib.rs`: shared state and recovery contract
- `crates/orbitkv-channel/src/lib.rs`: versioned iceoryx2 process channel API
- `crates/orbitkv-core/src/lib.rs`: public Rust API
- `crates/orbitkv-core/src/engine/`: registration, query, Publish and restore orchestration
- `crates/orbitkv-core/src/memory/`: NUMA, allocations and pools
- `crates/orbitkv-core/src/transfer/`: GPU layouts, copy backends and worker ownership
- `crates/orbitkv-core/src/codec/`: GPU ANS/FP8/TurboQuant and bounded representation validation
- `crates/orbitkv-core/src/storage/ssd/`: SSD index, io_uring and optional cuFile read/write
- `crates/orbitkv-core/src/storage/mod.rs`: residency assembly and pinned-memory reclamation
- `crates/orbitkv-core/src/storage/dram/`: DRAM residency, policy and inventory
- `crates/orbitkv-core/src/storage/publish.rs`: queued sealing and DRAM/SSD publication
- `crates/orbitkv-core/src/cost/`: bounded observations, estimates and shadow comparisons
- `crates/orbitkv-core/src/planning/`: batch candidates/targets, SSD route eligibility/acquisition and peer source segmentation
- `crates/orbitkv-core/src/query/`: byte admission, shared-read coordination, query leases and ownership handoff
- `crates/orbitkv-core/src/peer/`: local index discovery, authoritative source exports, remote READ and completion ownership
- `crates/orbitkv-server/src/peer.rs`: peer transfer control service
- `crates/orbitkv-server/src/cache/`: cache operations, lifecycle, and pending queries
- `crates/orbitkv-server/src/endpoint/`: two-process iceoryx2 endpoint and authenticated UDS lifecycle channel
- `crates/orbitkv-server/src/wire.rs`: protobuf-to-cache registration conversion
- `crates/orbitkv-server/src/http_server.rs`: HTTP health and metrics
- `crates/orbitkv-catalog/src/`: local global-index implementation
- `crates/orbitkv-transfer/src/`: transfer engine implementation
- `python/src/lib.rs`, `python/src/client.rs`: PyO3 module and cache client bindings
- `python/orbitkv/vllm/scheduler.py`: vLLM scheduler-side connector
- `python/orbitkv/vllm/worker.py`: vLLM worker-side connector
- `python/orbitkv/vllm/connector.py`: vLLM connector entry point
- `python/orbitkv/sglang/linker.py`: direct SGLang GPU-page linker
- `crates/orbitkv-channel/src/cache_client.rs`: query tickets, publish connection and restore lifetime
- `python/orbitkv/client/connection.py`: Cache Manager socket selection for adapters
- `python/orbitkv/identity.py`: shared model-artifact and computation fingerprinting
- `python/orbitkv/orbitkv.pyi`: Python type stubs

## Build, Check, Test

The selected engine baselines are the pinned `third-party/vllm` v0.31.0 and
`third-party/sglang` v0.5.21 tags. Keep Python optional dependency pins and
the source submodules aligned when updating a release. Native builds and wheel
CI only initialize `third-party/mooncake`; initialize engine submodules for
source inspection or release upgrades. The latest-release reference is vLLM
0.31.0 and SGLang 0.5.21 as checked on 2026-10-06; S5.1 records local upgrade
qualification and independent acceptance separately. Check official release tags again when starting
an upgrade. Use main only to locate fixes and prepare upstream contributions.
See `docs/adapters.md` and the `engine-integration` skill for ownership and
removal gates; do not delete a correctness Hook before its replacement is consumed.

The official vLLM cache profile requires `VLLM_USE_V2_MODEL_RUNNER=0` and one
attention cache group; V2/recurrent serving is rejected pending safe released
lifecycle contracts. P/D uses official native NIXL/MultiConnector or SGLang
native disaggregation, with independent cache adapters. Do not reintroduce
fork-only factories/callbacks. Manager shared-cache traffic stays on TENT.

### Rust

```bash
cargo build
cargo build --release
cargo test
```

On CUDA 13 dev machines, pass `--no-default-features --features cuda-13,mooncake` to `cargo test`/`cargo clippy` (default `cuda-12` can fail with missing `libcudart` symbols).

Run Cargo builds/checks/Clippy and Mooncake runtime tests sequentially in one
checkout. Native builds restage `.orbitkv/mooncake` shared libraries; overwriting
them while a source-built Manager or test has them mapped can crash that process.

### Python Bindings

```bash
cd python
maturin develop
maturin develop --release
../scripts/build-wheel.sh --release  # complete installable wheel
```

### Local Checks

```bash
prek run
```

Notes:
- `prek` is the local check entrypoint.
- `prek run` will fail on `master` or `main` because of the `no-commit-to-branch` hook in `prek.toml`.

### Python Test Gates

| Gate | When to run | Command | Notes |
|------|-------------|---------|-------|
| Default unit | Every Python PR before review | `cd python && uv run --group test pytest` | Must not start vLLM, `orbitkv-cache-manager`, or GPU runtime. Collection still imports deselected files, so top-level imports must be in the `test` dependency group or moved behind fixtures. |
| Source-only default | CI and dependency-boundary checks | `cd python && uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest` | Proves default gate does not need torch, vLLM, CUDA, native extension build, or a running server. |
| Integration | Server/native/client/session lifecycle changes | `cd python && uv run --group test pytest -m integration` | Requires built native extension, server binary, and GPU where the test uses CUDA IPC. |
| vLLM correctness E2E | Python test gates, vLLM connector, connector-visible cache semantics, save/load, query planning, or release-confidence changes | `cd python && ../.venv/vllm-release/bin/python -m pytest -m e2e tests/e2e/test_vllm_e2e_correctness.py --model /path/to/model --max-model-len 4096` | Use the vLLM `0.31.0` release environment described in `python/README.md`; reviewer reruns the gate on the GPU machine. |
| SGLang direct GPU E2E | SGLang linker, CUDA IPC layout, or plugin changes | `cd python && ../.venv/sglang-release/bin/python -m pytest -m e2e tests/e2e/test_sglang_direct_e2e.py --model /path/to/model` | Checks actual GPU load bytes after SGLang process restart against a cold-control namespace. |
| Shared-cache serving E2E | Peer transfers, index recovery, source ownership or shared-cache adapter changes | See `docs/shared-cache-qualification.md`; run `tests/e2e/test_shared_cache.py` separately in both engine environments | Requires etcd and a prebuilt Manager; one GPU, two TP=1 replicas, same-host TCP only. |
| Stress | Warm-hit pressure, lease cleanup, scheduler/cache concurrency | `cd python && uv run --group test pytest -m stress tests/stress/test_vllm_warm_hit_stress.py --model /data/models/Qwen3-4B --max-model-len 2048` | Targeted single-GPU evidence, not default PR feedback. |
| Release smoke | Published wheel/image, loader path, installed console script, CUDA runtime | See `python/tests/README.md` | Validates final installed artifact, not the source checkout. |

Do not default to running all of `python/tests`. Current project taste is `uv` + pytest markers for Python and Cargo/CI for Rust; do not add an `xtask` wrapper until the gate contract is stable and repeated execution is the real bottleneck.

### Benchmarks and Examples

```bash
uv run python examples/basic_vllm.py --model /path/to/immutable-model
.venv/vllm-release/bin/python -m benches.serving.single_node --engine vllm --backend orbitkv --model /path/to/model --output /var/tmp/orbitkv-bench/run-001
```

## Run Services

### Server

```bash
cargo run -r --bin orbitkv-cache-manager -- --addr 127.0.0.1:50055 --pool-size 30gb
```

### Distributed cache

Run the same Manager with `--etcd-endpoints`, `--node-id`, and matching
`--cluster-name` on every host. etcd stores protocol identity and leased members; bounded owner inventory
streams maintain explicitly covered local indexes. Peer gRPC serves source grants and
release; there is no standalone directory binary. See `docs/p2p.md`.

## Code Style

### General

- Use English in comments
- Use `.venv` for the Python virtual environment
- Keep changes scoped and aligned with the existing module structure
- Keep Python limited to engine callbacks, layout inspection, GPU allocation and handoff. Put shared cache state machines, planning, batching and waiting in Rust; release the GIL around native work and prepare immutable native hash batches once per lookup instead of converting per-page inputs on repeated polls. Do not add a Python forwarding facade over the native client.
- Prefer established KV-cache ownership, prefetch, retention and transfer patterns over speculative policy machinery. Record the upstream release/commit and distinguish implemented behavior from open proposals; extend OrbitKV's existing owners and validate with matched workloads.
- Update affected documentation, README capability claims, and website content with each behavior or deployment change. The website renders `docs/` directly; keep one source for technical documentation.
- Before 1.0, remove obsolete APIs and compatibility code instead of adding aliases or fallback paths. Keep boundaries that own behavior; remove classes and functions that only forward calls without a separate responsibility.
- Organize modules by the behavior and resources they own. A new type or trait must have a concrete responsibility; avoid temporary context wrappers, configuration-only wrappers for a few constructor arguments, and speculative abstraction layers. Call the behavior owner directly when another function would only forward the call.
- Code should be self-documenting. If a comment seems necessary, first try refactoring so the code explains itself.

### Rust

- Prefer minimal visibility: `fn` > `pub(crate)` > `pub`
- Prefer explicit errors over `unwrap` and `expect`
- Keep `use` ordering consistent: std, external crates, local crate
- Prefer `NonNull` over raw pointers where practical in unsafe code

### Python

- Target Python 3.10+
- Use native generics such as `list` and `dict`
- Use PEP 604 unions such as `X | None`
- Use `%s` formatting in logging calls
- Keep imports grouped: standard library, third-party, local

### PyO3

- Keep Python-facing APIs thin and delegate core logic to Rust crates
- Convert Rust errors to `PyErr` cleanly
- When modifying `python/src/lib.rs`, update `python/orbitkv/orbitkv.pyi`

## Testing Principles

- Keep test implementations and fixtures under `tests/`, separate from production source. Rust private unit tests live in each crate's `tests/unit/` tree, mirroring `src/`, and are loaded with `#[cfg(test)]` plus `#[path = "..."] mod tests;` from the owning module. Preserve private access and test names; do not expose production internals just to move tests.
- Rust integration tests remain in each crate's `tests/` root with shared fixtures under `tests/common/`. Python tests and support code live in `python/tests/`. Performance code stays in `benches/`; generated evidence goes outside the checkout.
- Keep test-only helper implementations in the test modules too. Do not duplicate business logic or add forwarding production APIs for tests.
- Do not add tests just for the sake of adding tests.
- Before adding or keeping a test, answer: would skipping this test materially reduce confidence to merge a PR in its trigger area? If not, delete it or keep it out of routine gates.
- Merge tests that protect the same contract; prefer table-driven cases with clear ids over copy-pasted methods.
- Delete tests with no clear consumer. A test that no dev, reviewer, CI job, release gate, or scheduled job uses is noise.
- Default Python tests must be dev friendly: fast, stable, no GPU, no vLLM, no native extension build, and failure messages that point to a Python contract.
- Heavy tests must declare their trigger: integration, e2e, stress, or release smoke. A heavy test without a trigger should not live in the main pytest surface.
- Stub and mock tests are allowed only for local contracts and must not shadow real runtime modules during integration/e2e collection.
- Prefer integration tests when they prove a real boundary that units cannot. Prefer units when they give faster, clearer feedback for local connector state machines.

## Git Workflow

- Use `feichai0017 <songguocheng348@gmail.com>` for commits in this repository.
- Do not commit directly to `master`
- Create a `feat/`, `fix/`, `chore/`, `refactor/`, `style/`, or `ci/` branch first
- We use Commitizen commit message format
- Use `cz c` when creating commits interactively
- Keep PR descriptions concise. Link external experiment artifacts and maintained methodology instead of committing result files. Preserve historical evidence through verified archives or immutable Git links when retiring old datasets.
