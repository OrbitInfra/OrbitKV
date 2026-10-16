# Python Test Gates

`python/tests` is organized around developer workflow, not around how much code has accumulated. A test belongs here only when skipping it would materially reduce confidence to merge a PR for its trigger area. The default pytest invocation is intentionally small: it runs unit/helper contracts and excludes tests that start `orbitkv-cache-manager`, require CUDA, run vLLM, or create pressure workloads.

## Layout

- `unit/`: source-only contracts, including `unit/pd/`.
- `integration/`: Cache Manager, native channel, and CUDA page transfers.
- `e2e/`: actual inference-engine correctness and restart recovery.
- `stress/`: concurrent or pressure workloads with explicit resource requirements.
- `support/`: process helpers, import stubs, paths, and immutable test data.
- `conftest.py`: shared options and the Cache Manager fixture used by both engine
  E2E and channel integration tests; `integration/conftest.py` owns GPU fixtures.

Performance workloads and their results live in [`benches/`](../../benches/README.md),
with their own CPU-only harness tests. They are not correctness-test fixtures.

Queued-warming changes require both engine GPU gates with
`ORBITKV_QUEUE_WARMUP=1` (automatic warming is otherwise disabled). The default client and
vLLM scheduler tests cover bounded hint tickets and revalidated admission;
`integration/test_sglang_admission.py` exercises the pinned upstream queue/key
contract. `integration/test_sglang_direct_transfer.py` also proves unpolled SSD
warmups release reservations and supply DRAM pages for a later leased GPU restore.
A released query lease does not count as use, last-owner cleanup settles unused
bytes, and a fresh read credits its footprint once after successful H2D across
all layers. These checks cover both stored page layouts.

Consumer-owned preparation uses `ORBITKV_PREPARE_REQUESTS=1` instead. Native
fault tests cover ready-result expiry without polling, claim-to-GPU ownership,
shared-read cancellation, bounded batches and deadline fallback. The Qwen3
serving fault gate below runs both ordinary and prepared paths with the
corresponding environment setting. Automatic hybrid preparation stays disabled;
hybrid recovery E2Es continue to qualify ordinary selected-range reads.

## What To Run

| Change area | Gate | Command | Failure boundary |
| --- | --- | --- | --- |
| Connector helper math, scheduler state, worker load failure handling, GPU registration | Default unit | `uv run --group test pytest` | Python contract or local connector state-machine regression |
| Clean source-only Python changes, docs touching test layout, CI test dependency changes | Source-only default | `uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest` | Default test accidentally depends on torch, vLLM, CUDA, or native extension |
| vLLM upgrade or MultiConnector delivery semantics | Native connector contract | `python -m pytest -m integration tests/integration/test_vllm_connector_contract.py` | Use the selected native P/D engine revision; ordinary cache must not weaken reliable P/D delivery. |
| Server client, native extension, CUDA IPC registration, session lifecycle | Integration | `uv run --group test pytest -m integration` | Server/native/GPU lifecycle regression |
| CUDA-visible device identity, receiver route, NUMA allocator | Installed GPU identity | `python -B -m pytest -m integration tests/integration/test_gpu_identity.py` | Requires an installed wheel and two physical CUDA devices; see the installed identity gate below |
| Distributed startup, placement, embedded catalog protocol and packaged Manager | Distributed process gate | `ETCD_BIN=/path/to/etcd pytest -m integration tests/integration/test_distributed_cache.py` | Starts two Managers and real etcd, checks remote Mooncake/GPU bytes and local recovery after coordinator loss; requires built native artifacts and CUDA. |
| Peer transfer, source ownership, catalog recovery or either engine adapter | Shared-replica serving gate | `ETCD_BIN=/path/to/etcd ORBITKV_CACHE_MANAGER_BINARY=/path/to/manager pytest -m e2e tests/e2e/test_shared_cache.py -k vllm --model /path/to/qwen3-8b` | Repeat in SGLang's environment with `-k sglang`; checks remote bytes, output, catalog replay, source restart and reservation drain. Same-host TCP only. |
| vLLM connector correctness, cache semantics, save/load/hit behavior, release candidate confidence | vLLM correctness E2E | `../.venv/vllm-release/bin/python -m pytest -m e2e tests/e2e/test_vllm_e2e_correctness.py --model /path/to/model` | Native prefix-cache control follows the same prompt plan; `long_warm` must load saved KV after vLLM restart. |
| SGLang registration reference retirement | Installed pool/linker lifecycle | `ORBITKV_TEST_STRICT_SHUTDOWN=1 python -B -m pytest -m 'integration and gpu' tests/integration/test_sglang_registration_lifetime.py --model /path/to/model` | Actual released GPU pool, queued offload, normal close and independent exporter hold; requires a matching installed wheel and Manager. No cancellation/fault qualification. |
| SGLang direct GPU linker, CUDA IPC layout, or plugin registration | SGLang direct E2E | `../.venv/sglang-release/bin/python -m pytest -m e2e tests/e2e/test_sglang_direct_e2e.py --model /path/to/model` | Restores after HBM flush and engine restart, with DRAM/forced-SSD byte counters. Output IDs and finite log probabilities match native HBM reuse; a changed identity matches cold computation. |
| Native P/D, cache composition or request lifetime | vLLM native P/D model gate | `python -m pytest -m e2e tests/e2e/test_vllm_native_pd_e2e.py --model /path/to/dense-model --basetemp /var/tmp/orbitkv-pd/run-001` | Official vLLM/NIXL and upstream router from `docs/pd.md`; standalone P/D, both cache orders, restart, real pressure preemption and strict output. TP=1/PP=1 eager NIXL/UCX. |
| SGLang native P/D or cache composition | SGLang P/D model gate | `python -m pytest -m e2e tests/e2e/test_sglang_pd_e2e.py --model /path/to/model --basetemp /var/tmp/orbitkv-pd/sglang-001` | Official engine, built Manager and `sglang-router==0.3.2`; one GPU for two model copies or two GPUs. Native retraction, restart/cache bytes and strict monolithic output; delayed ACK/partial-write faults need separate release gates. |
| Warm-hit pressure, pending lease release, scheduler/cache concurrency | Stress | `uv run --group test pytest -m stress tests/stress/test_vllm_warm_hit_stress.py --model /data/models/Qwen3-4B --max-model-len 2048` | Real vLLM cache pressure regression |
| Wheel, loader path, installed console script, target CUDA runtime, published package | Release smoke | See Release Smoke | Packaging, loader, final artifact, or runtime contract regression |

A heavy test without a clear trigger should not be promoted into a routine gate. A generated fuzz workload used to live here, but it had no stable owner, cadence, data contract, or debugging path; it was removed from the main pytest surface instead of pretending to be a regular gate.

## Default Unit Gate

```bash
cd python
uv run --group test pytest
```

The test group includes `tomli` on Python 3.10 for the TOML-based engine pin
checker; Python 3.11+ uses the standard-library parser.

This command must not start vLLM, `orbitkv-cache-manager`, or any GPU runtime. It still
imports every test module during pytest collection, so any top-level import used
by deselected heavy tests must be present in the `test` dependency group or moved behind a
fixture/helper boundary.

CI uses the source-only variant below so the unit gate does not compile the native extension or require CUDA:

```bash
cd python
uv run --isolated --no-project --with pytest --with numpy --with 'requests>=2.26.0' pytest
```

Runs:
- connector arithmetic and scheduler state-machine contracts (`test_combine_hashes.py`)
- connector load fault-tolerance unit tests with fake transport (`test_connector_fault_tolerance.py`)
- GPU registration layout with fake torch objects (`test_gpu_registration.py`)
- import-stub safety checks for default unit tests (`test_unit_stubs.py`)

This gate must collect and run without torch, vLLM, CUDA, external models, or a running OrbitKV server. Stub modules are allowed only inside tests that explicitly mock the connector boundary, and they must not shadow a real runtime during integration or E2E collection.

`test_sglang_shutdown.py` uses real CPU parent/worker processes and an inherited
file lock to protect the direct SGLang E2E cleanup contract. The E2E sends normal
SIGTERM to the service leader, waits for the owned group and fails on nonzero
exit, inspection errors or forced cleanup of residual workers. Signal/wait errors
and unknown group ownership are recorded as failures, and cleanup preserves any
original workload failure. These controls do not establish GPU
worker or physical CUDA IPC lifetime qualification.

The vLLM service helper selects the official release's `--shutdown-timeout 30`
so normal SIGTERM requests native drain instead of the default immediate abort.
Its 90-second process wait remains a separate bound; the grace period is not a
DMA fence. Installed-wheel runs must set `ORBITKV_CACHE_MANAGER_BINARY` to the
wheel's frozen Manager executable. Preserve actual service exits, complete
native shutdown logs and any forced-worker diagnostics before accepting a run.
The 2026-10-14 official 0.31.0 model assertions pass, but the separate strict
drain profile remains rejected by the upstream intentional-shutdown race
[vLLM #48745](https://github.com/vllm-project/vllm/issues/48745). A successful
pytest exit does not override that native-log failure or establish S3 drain.
The subsequent independently accepted `c5681616`/`eefcf8f6` healthy installed
cohort passes the strict native log/wait checks for six dense vLLM cases. Keep
that nominal result separate from the original rejected run and fault/reuse
qualification; see [PR #246](https://github.com/feichai0017/orbitkv/pull/246).

Shared client ownership is tested in Rust under
`crates/orbitkv-channel/tests/unit/cache_client.rs`: query revisions, immutable
hash snapshots and shared views, warming limits/expiry, separate publish traffic, lost restore
notifications, deadlines, and client-bound handles. These replace the old mocked
Python facade tests. The GPU process-boundary gate below verifies the PyO3 API.

The Rust completion-owner controls in `python/tests/unit/local_restore.rs` run
separately from pytest. For native Restore result or retention changes, build
`cargo test -p orbitkv-channel --test completion_process --no-run` first, then
set `ORBITKV_SHARED_COMPLETION_TEST_BINARY` to that test executable. Run
`cargo test -p orbitkv-py --lib --no-default-features --features cuda-13,mooncake,test-hooks
local_restore::tests -- --include-ignored --nocapture` from the repository root.
The benchmark/harness migration also runs
`pytest benches/tests/test_registration_cleanup.py`: CPU weak-reference controls
execute the actual workload cleanup bodies, require release after normal
unregister/close, and retain failed cleanup plus independent queued holds.
The optional `ORBITKV_CLEANUP_SOURCE_ROOT` selects an archived source for the
negative control; it never changes production lookup or admission.

The installed registration campaign controller is
`tests/support/registration_campaign.py`. Its CPU gate
`pytest tests/unit/test_registration_campaign.py` checks real READY lock ownership
and rejects missing/forced/unreaped/native-error evidence. It also checks
adopted exit codes, expected logs, observed owner identities, selected lock
replacement, READY/controller grant replay, loader/cache environment leakage,
shared remaining deadlines and final-error retention. Runtime manifest controls
use the consumed input check to reject payload changes, extra installed files,
symlink target drift, changed manifests, duplicate/traversal entries and
nonregular input. Each frozen root has a complete file census, file sizes,
resolved targets and SHA256 values; manifests have a 64 MiB/200,000-file bound.
Startup verification has one 180-second deadline. Cell admission/postflight can
select only declared roots consumed by that cell; startup, grant and final input
checks retain the complete manifest set. Freeze and review each cell's consumer
root selection as part of launch acceptance, with the unchanged cleanup budget.
The original controller
and its counterexamples stay in the external artifact archive. It separates precheck
from formal cells, preserves locks on unknown ownership and requires a distinct
independent grant matching the immutable contract. A source/CPU controller pass
is not GPU launch approval; freeze and review the actual runtime/resources before
execution. Runtime outputs and contracts belong outside this checkout.

For explicit CUDA registration/credit-owner changes, run
`registration::tests` and the actual duplicate-entry control in `client::tests`
with the authenticated channel fixture below. CPU tests use a declared local
Torch storage ABI stub to count exports and repayments; they never import a
real CUDA mapping. Partial/lost-return controls preserve the owner, and repayment
is not retried after an unknown return. Separately run the server lifecycle
unit gate: actual UDS rejects bad registration with a generation-bound
NeverImported disposition; this uses the explicit host-registration test shim
and read-only driver discovery, not CUDA byte/DMA qualification. Actual installed
Torch export/import/repayment and both engine GPU gates remain required.

For producer-bound Publish registration changes, run `client::tests` with the
same fixture and embedding environment. These controls pause a real native
worker-lock lookup, replace the registration, and check stale/foreign-client
rejection through both save entry points. The idle worker owns no CUDA executor;
its positive result validates the retained Manager route, not GPU bytes.
The exporter weak-reference control checks that dropping a worker's hold cannot
release tensors still referenced by a registration authority. vLLM's source-only
save-lifecycle gate checks request/boundary queue capture and mixed-owner refusal;
SGLang's corresponding released-API control is
`pytest -m integration tests/integration/test_sglang_admission.py -k offload_keeps`.
It mocks CUDA readiness and native save, so it remains a queue/response contract,
not a device-copy or engine lifetime gate.

Choose an embedding-capable `PYO3_PYTHON`; when testing the extension crate on
Debian Python 3.11, use
`RUSTFLAGS='-C link-arg=/usr/lib/x86_64-linux-gnu/libpython3.11.so'` for that
invocation. This link setting belongs only to the CPU test executable.
The actual native `advance` path is exercised with an authenticated subprocess
and shared completion records; its test executor owns no CUDA mapping and
submits no device work. A pending diagnostic cannot consume the result or
release the job. The test-only thread-local claim pause stops after the state
snapshot and before CAS, allowing another real claimant to win deterministically.
A losing claimant cannot publish drain or a terminal result for that owner. Actual GPU tensor ownership, managed DMA termination and
adapter page reuse remain separate installed hardware gates.

## Server Integration Gate

### Installed GPU identity gate

Run `test_gpu_identity.py` for registration routing or NUMA allocation changes
with a complete installed wheel and working CUDA on two physical devices. Use a
harness containing `python/tests/` and `python/pyproject.toml`, without a source
`python/orbitkv/` package. Put the installed wheel directory first in
`PYTHONPATH`, and set `ORBITKV_CACHE_MANAGER_BINARY` to its packaged Manager.

```bash
export ORBITKV_GPU_IDENTITY_DEVICES=7,0
export ORBITKV_GPU_IDENTITY_OUTPUT=/var/tmp/orbitkv-gpu-identity/run-001
export ORBITKV_TEST_GPU_LOCK_DIR=/var/tmp/orbitkv-resource-locks
python -B -m pytest -q -m integration --maxfail=1 tests/integration/test_gpu_identity.py
```

The twelve positive cells use real Torch CUDA IPC for DRAM and forced io_uring
SSD recovery under numeric, UUID, reordered and partial visibility. They check
full GPU bytes, native receiver routing, worker affinity, resident memfd NUMA
pages and normal ownership drain. The three rejection controls use unissued
metadata-only sentinels to check guards before import. They do not qualify real
issued-ticket rejection or partial-import rollback lifetime. Keep failed runs
and frozen inputs alongside the final external evidence.

### Other integration gates

```bash
cd python
uv run --group test pytest -m integration
```

Runs tests that start or require a local `orbitkv-cache-manager` but do not run vLLM:

- `test_channel.py` proves Python-to-Cache Manager iceoryx2 ping, epoch fencing,
  UDS/memfd bootstrap, lifecycle registration/health with gRPC disabled,
  local publish, cold and warm `QueryBundle`, asynchronous
  restore with GPU byte verification, local lease release, and shutdown across
  a real process boundary;
- `test_tent_transfer.py` checks native GPU READ/WRITE bytes and exact batch
  accounting when TENT merges adjacent descriptors. Run it for native transfer
  completion changes with a built wheel and working CUDA driver.
- `test_session_watcher.py`
- `test_state_demand.py` checks the native public demand API, absolute ranges,
  identity/alignment rejection and the distinction between required state and
  available state. Run this for recovery-contract or binding changes; it needs
  the extension but no engine, Manager or CUDA runtime.
- `test_sglang_direct_transfer.py` writes SGLang-shaped GPU pages through CUDA
  IPC, clears their source slots, and verifies a byte-exact restore into new
  slots. Run this when changing page registration or GPU transfer layout.
- `test_sglang_recovery.py` joins sparse auxiliary state with a full prefix,
  restores poisoned Full + SWA and Full + recurrent/conv buffers from DRAM and
  SSD, checks copied-plus-retained coverage, and verifies auxiliary state beyond
  the attention prefix is not fetched. It rejects incomplete plans and completes
  published destinations before releasing an aborted request. Run this for
  recovery-rule or hybrid-layout changes.
- `test_vllm_recovery.py` checks vLLM's absolute-span evidence against the real
  Rust validator and restores full/window/conv/temporal buffers through the real
  scheduler/worker adapters from DRAM and forced SSD. Run this for vLLM hybrid
  query, boundary selection or lease-handoff changes, followed by the vLLM E2E.
- `test_sglang_combined_recovery.py` registers Full + SWA + checkpoint pools
  together, including same-layer convolution and an empty temporal tensor.
  It rejects incomplete windows and verifies exact GPU/SSD bytes.

Hybrid serving changes also require Qwen3.5 and the native Full + SWA fixture.
Use the generated Inkling fixture with `--sglang-load-format dummy` for the
combined SGLang path. See [the fixtures and commands](../../docs/hybrid-recovery.md#reproducible-gates).

Requirements:
- built Python extension, for example `uv run maturin develop -r`
- a discoverable `orbitkv-cache-manager` binary from the installed package or Cargo target
- CUDA-capable GPU for tests that register real CUDA IPC tensors
- loader paths for the active Python and CUDA runtime, when the environment does not provide them globally


Set `ORBITKV_CACHE_MANAGER_BINARY` to an absolute path to test a specific build
(for example `target/debug/orbitkv-cache-manager`) instead of a previously
installed or release binary. Both integration and vLLM helpers honor this override.

Automatically selected listener ports exclude Linux's outgoing ephemeral range.
GPU and pinned-pool initialization can delay binding long enough for an outgoing
connection to claim a released `bind(0)` port. The SGLang restart gate also passes
an explicit rendezvous port. Other test jobs can still race for a selected port;
run GPU gates sequentially on the same host.

## vLLM Correctness E2E Gate

Use the vLLM `0.31.0` release environment described in
[`python/README.md`](../README.md). The SGLang `0.5.21` environment is separate.
The default `uv run --group test` environment intentionally has no GPU framework.

```bash
cd python
../.venv/vllm-release/bin/python -m pytest -m e2e tests/e2e/test_vllm_e2e_correctness.py \
  --model /path/to/model \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --max-model-len 4096
```

For release upgrades, use a new external cache directory for each run and
record it with the command:

```bash
export ORBITKV_E2E_CACHE_ROOT=/var/tmp/orbitkv-e2e/run-001/cache
export TORCHINDUCTOR_CACHE_DIR="$ORBITKV_E2E_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$ORBITKV_E2E_CACHE_ROOT/triton"
export VLLM_CACHE_ROOT="$ORBITKV_E2E_CACHE_ROOT/vllm"
export TORCH_EXTENSIONS_DIR="$ORBITKV_E2E_CACHE_ROOT/torch-extensions"
export PATH="$PWD/../.venv/vllm-release/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
```

Preserve shared-cache failures before retrying with these isolated directories.
A successful isolated run does not repair or qualify an older compiler cache.
The installed-wheel gate already assigns independent cache paths per cell.
Select the SGLang environment's `bin` directory for its gate. Invoking an
environment's Python alone does not put its Ninja executable on `PATH`.
Benchmark launch and the SGLang direct helper also select that executable
directory and use a private Torch extension cache.

This is the main correctness E2E. It runs the same ordered prompt plan with
vLLM native prefix caching and with OrbitKV, then requires exact completion
equality at each plan step. The native control stays in one process to retain
its HBM cache; OrbitKV restarts vLLM between cold saves and warm loads. The
gate verifies that native `long_warm` had a prefix-cache hit, checks OrbitKV
save/hit/load activity, and requires OrbitKV `long_warm` to load KV bytes after
that restart.

Use `--vllm-cudagraph-mode FULL` to qualify restored prefixes entering full-graph
replay, or `NONE` / `PIECEWISE` for the other consumption boundaries. The option
applies identically to the native and OrbitKV engines; omitting it retains vLLM's
default graph selection. Add `--vllm-multi-connector` to exercise cache recovery
as the second child of the pinned engine's `MultiConnector`; this checks child
metadata mapping and ordinary cache recovery independently of the
separate two-worker P/D payload gate.

Add `--vllm-cache-tier ssd` to enable an 8 GiB SSD cache, wait for writes to
drain and evict Manager DRAM before the engine restart. The same output and
GPU-load controls then also require new SSD reads. DRAM remains the default.
See [model qualification](../../docs/models.md) for pinned larger checkpoints
and the Qwen3.8 FP8 kernel setting used on H20.

This gate is required before merging PRs that change Python test gates, the
vLLM connector, cache semantics visible to the connector, save/load behavior,
query planning, or release confidence. The code author runs it before requesting
merge, and review reruns it independently on the GPU machine.

SGLang direct-linker changes require the SGLang direct E2E in the release
environment. It checks exact generated text, a nonzero external prefix hit
after `/flush_cache`, and an actual Cache Manager GPU load after the SGLang
process restarts while the Cache Manager remains alive. A separate namespace
provides a true cold inference control for the restarted process.

The SGLang helper sends SIGTERM to the service leader and waits for the native
worker lifecycle before inspecting process-group clearance. A timeout records
forced cleanup. The release outcome must also inspect real cleanup receipts and
ordered native zero-request/ASGI completion for each leader PID. Repeated
child-only cleanup messages alone do not imply a failed exit. Preserve actual
producer-lifetime warnings and internally signalled child exits: passing model
assertions and zero service-leader exits do not qualify physical CUDA IPC
lifetime or S3 crash reclamation.

SGLang P/D payload-engine or cache-composition changes additionally require
`tests/e2e/test_sglang_pd_e2e.py`. Use the official engine in `docs/pd.md`, a built
Cache Manager and one GPU with room for two model copies (or two GPUs). The gate starts prefill, decode and router, publishes state,
restarts the P/D workers, and requires a continuation to recover beyond the
original prefill-only page boundary. It then compares both generations with a
monolithic control. The local gate forces TCP; cross-host RDMA/GPUDirect claims
require a separate run with `MC_FORCE_TCP` unset, explicit NIC selection, and
external NIC/device evidence.

Use `--cache-protected-percent 80` to qualify demand protection in either E2E.
For the SGLang DRAM/SSD gate, add `--ssd-write-policy reuse` to exercise
selective writes: its foreground restore before engine restart admits the
SSD replica. The Rust SSD integration gate also covers first-publication
skips, warming exclusion and recomputation followed by republication.

Requirements:
- vLLM installed in the active environment
- local model path, not an implicit network download
- GPU runtime compatible with the installed wheel variant
- enough free GPU memory for the model and configured context length

## Stress Gate

```bash
cd python
uv run --group test pytest -m stress tests/stress/test_vllm_warm_hit_stress.py \
  --model /data/models/Qwen3-4B \
  --max-model-len 2048
```

This is a targeted single-GPU vLLM scenario for warm-hit pressure. The checked profile uses `/data/models/Qwen3-4B`, `max_model_len=2048`, `gpu_memory_utilization=0.82`, `max_num_seqs=16`, and 12 concurrent repeated prompts; it has been validated on a 16GB GPU. Run it for cache warm-hit, pending lease release, scheduler/cache concurrency, or pressure-profile changes. It is not a default PR gate.

## Model-serving fault stress

The dense Qwen3 serving fault gate runs each engine in its own pinned environment:

```bash
ORBITKV_FAULT_TESTS=1 ORBITKV_PREPARE_REQUESTS=1 \
ORBITKV_CACHE_MANAGER_BINARY=/path/to/test-hooks/orbitkv-cache-manager-py \
../.venv/vllm-release/bin/python -m pytest -m stress \
  tests/stress/test_recovery_faults.py -k vllm --model /workspace/models/qwen3-8b \
  --basetemp=/var/tmp/orbitkv-bench/serving-fault-vllm
```

Use `ORBITKV_PREPARE_REQUESTS=0` for ordinary demand, and the SGLang environment
with `-k sglang` for the other engine. Build the
Manager with `orbitkv-server/test-hooks`, retain that binary separately, then
complete the ordinary native build before starting any GPU processes. Never
restage Mooncake libraries while a Manager or engine is running. The gate uses
private barriers, exact generated-output controls, real SSD reads, cancellation,
lost completion signals and engine/Manager restart. Independent requests must
progress while a cancelled read retains its buffers; reservations and submitted
I/O must subsequently drain. Logs are retained separately for every process
incarnation together with `fault-results.json`. This deterministic fault profile
is separate from the ordinary performance benchmark.

## Release Smoke

Release smoke validates the final installed package, not the source checkout. It should use a clean non-editable environment and record Python libdir, `PYTHONHOME`, `PYTHONPATH`, CUDA runtime path, package name/version, GPU, model path, and metrics excerpt.

After installing the candidate wheel in the pinned engine environment, run:

```bash
mkdir -p /var/tmp/orbitkv-bench
.venv/vllm-release/bin/python -m pytest -m release_smoke \
  python/tests/release/test_installed_wheel.py -k vllm \
  --model /workspace/models/qwen3-8b \
  --basetemp=/var/tmp/orbitkv-bench/wheel-smoke-vllm
```

Run from the repository root; substitute the SGLang environment and `-k sglang`
for the other adapter, with its own `--basetemp` directory. The test starts
subprocesses outside the source package, rejects editable/source imports,
checks the pinned engine version, verifies installed engine and OrbitKV files
against the wheel RECORD manifests before and after execution, rejects unrecorded
package data/code and conflicting plugin metadata, and proves TENT libraries load from the
installed wheel. Only generated `__pycache__/*.pyc` files are exempt from the
unrecorded-file check. Each engine runs DRAM and forced io_uring SSD cells. Use
`-k "vllm and dram"` to select a single cell.

Each cell compares cold, native HBM, full external recovery after engine restart,
and partial recovery against an engine-native control. HBM hits must increase
native reuse without external query or H2D work. The partial case first resets
HBM and requires positive external GPU-load bytes; SSD cases also require
positive io_uring reads after DRAM eviction. Output text/tokens, final query/I/O
ownership, process exit status and installed-file integrity are checked.
`responses.json`, metrics, before/after manifests and per-service cleanup records
stay under the selected external `--basetemp`. A forced cleanup fails the cell;
Manager SIGTERM must drain and exit 0 through the installed console entry.
Run the engines sequentially on one GPU. This dense TP=1/PP=1 eager gate does
not qualify CUDA graphs, P/D, multi-GPU or cancellation during transfer. See
[release preparation](../../docs/releases.md) for the wheel build matrix.

### Two engines sharing one Manager

This gate includes overlapping external restores after both engine processes
restart, and a separate normal Manager restart after both engines have drained.
The replacement Manager rebuilds cold; the test does not claim persistence or
crash reclamation. For kernel qualification, add
`--orbitkv-transfer-backend kernel` with the newly frozen complete wheel. Each
engine's worker backend is checked in the Manager log, alongside exact model
outputs, real restore bytes and normal process/resource drain.

Run the installed-package gate below when changing Manager multi-client lifetime,
engine restart, registration isolation or shared DRAM/SSD ownership. Keep vLLM
and SGLang in separate non-editable environments with the same complete OrbitKV
wheel. Run it on an otherwise idle GPU with enough memory for two model copies;
the fixed dense profile uses TP=1/PP=1, eager execution and 0.4 GPU-memory fraction
per engine.

```bash
.venv/vllm-release/bin/python -m pytest -m release_smoke \
  python/tests/release/test_shared_manager.py \
  --vllm-release-python "$PWD/.venv/vllm-release/bin/python" \
  --sglang-release-python "$PWD/.venv/sglang-release/bin/python" \
  --model /workspace/models/qwen3-8b \
  --basetemp=/var/tmp/orbitkv-bench/shared-manager-001
```

Both DRAM and io_uring SSD cells use a separate native output control for each
engine. The gate requires overlapping HTTP requests, zero external work for
native HBM reuse, and positive restore bytes after each engine restarts while
the other engine keeps serving through the same Manager PID. The SSD cell
clears Manager DRAM before each recovery and requires physical io_uring reads.
It checks both installed environments before/after, compares the packaged
Manager and TENT hashes, requires normal exits, zero resource ownership and no
remaining UDS. The run controller must match installed package members to the
frozen wheel contract as well as RECORD; RECORD alone does not identify a wheel.

The selected external `--basetemp` retains response timing/output, metrics,
package probes and process-cleanup records. These are two instances sharing a
service and budget, not interchangeable cross-engine KV bytes. This gate does
not qualify crash reclamation, concurrent-load fairness, capacity pressure,
P/D, containers, CUDA graphs or cross-host operation. A new immutable artifact
cohort must retain any failure; do not reuse `--basetemp` to overwrite it.

For a targeted SGLang normal-exit IPC diagnosis, replace `python -I -m
sglang.launch_server` with `python -I /absolute/path/to/python/tests/support/trace_sglang_ipc.py`
and retain the same engine arguments and installed wheel. Set
`ORBITKV_CLOSE_TRACE_DIRECTORY` to a new external directory before startup.
The test helper wraps only OrbitKV export/close in memory; it does not edit
installed files and is separate from the ordinary installed-package gate.
Read the `linker_close_return` and `python_atexit` rows: require nonempty exports,
no read errors and zero credits after normal unregister. A missing slot or file
must not be counted as zero. Calibrate the 64-byte counter payload offset against
the installed PyTorch revision using a known live and then drained IPC export;
keep the revision and calibration with the trace. The file header and PyTorch
exit warning cannot substitute for those credits. This does not prove crash or
in-flight cancellation safety.

Minimum checks:
- `orbitkv-cache-manager --help`
- minimal installed `orbitkv-cache-manager` startup and `/health` 200
- vLLM + `OrbitKVConnector` with `/v1/models`, one completion, one repeated long prompt, and non-zero save/load/hit/HLL metrics

The SGLang release environment runs `tests/integration/test_sglang_admission.py`
with real release `Req` objects and controlled backing completion. It checks
nonblocking admission, other-request progress, changed keys, deadline fallback,
and shared rank decisions. `test_sglang_direct_transfer.py -k ssd` checks real
SSD cancellation/disconnect cleanup and exact GPU restoration. These are
separate from the serving E2E: neither a fake completion nor test-side polling
alone qualifies SGLang serving recovery.

## Storage representations

The vLLM correctness and SGLang direct E2E gates accept `--kv-cache-dtype`
(default `auto`) and `--storage-codec none|ans|fp8|turboquant-4|turboquant-3`. They keep output controls at
the same KV precision and require actual SSD recovery after eviction/restart.
The text-equality assertion remains strict with FP8/TurboQuant storage and can fail because
the codec is lossy. Report those differences as quality evidence, not a passing
exact-recovery result. Use `--storage-codec none` for the exact regression gate.
See [format qualification](../../docs/storage-formats.md) for commands, the
storage-quantization scope and the distinction from native GDS qualification.

## Official native P/D candidates

Use the release environments and native transport dependencies in [P/D setup](../../docs/pd.md).
GPU gates are explicitly marked and deselected by the source-only suite.

- `test_vllm_native_pd_e2e.py`: NIXL and upstream router; standalone transfer,
  cache orders, restart, observed pressure preemption and strict model output.
- `test_sglang_pd_e2e.py`: native backend and external cache; observed retraction,
  restarted recovery of D-produced state and strict monolithic control.
- `test_vllm_connector_contract.py`: real released MultiConnector consumption of
  best-effort cache versus reliable native P/D delivery.

Partial writes, cancellation during DMA, delayed ACKs and page reuse require
independent native lifetime evidence before opening support. Retired fork-only
fault gates and results remain at immutable commit `9aee895e` and its external
handoff. Do not count those as official-release passes. Preserve failures,
controls, wheel/engine hashes and raw runs outside the checkout.


### Static local LoRA installed model gate

`release/test_static_lora.py` runs separately in each official engine environment:

```bash
cd python
python -m pytest -m release_smoke tests/release/test_static_lora.py -k vllm \
  --model /path/to/Qwen3-8B --basetemp /external/artifacts/static-lora-vllm
# Use -k sglang from the SGLang 0.5.21 environment for its cell.
```

The gate requires an installed complete wheel and the dense Qwen3 fixture family.
It creates deterministic nonzero local PEFT adapters outside the checkout, checks
base/A/B against native model controls, preserves native HBM hits, restores every
selection through actual io_uring SSD after engine restart, and checks that
same-name/path changed adapter bytes miss old state. Selected output token IDs
and text must match exactly; selected-token log probabilities allow absolute
0.005 with zero relative tolerance. A/B must each change native probabilities
from the base by more than 0.001. Package RECORD hashes and owned process cleanup
are checked. This qualifies fixed artifacts in one process deployment, not
runtime LoRA loading, sessions, hybrid models, graph or cross-host serving.

### Explicit released decode-graph recovery

The installed-wheel gate accepts `--release-cuda-graph`. It selects official
vLLM V1 `FULL` mode with capture sizes 1/2/4 or SGLang native decode `full`
with prefill disabled. Each eight-token response must add at least seven released
one-token FULL runtime observations (vLLM) or decode-graph passes (SGLang).
vLLM does not label those rows as prefill/decode: eligible one-token prefill can
contribute, so this is not a claim of seven pure decode replays or Graph on every
decode iteration. New snapshots identify the engine-specific observation kind.
Capture logs and eager forwards cannot satisfy this gate. Results include every
raw output, graph observation and native/cache byte boundary. Graph qualification
and remaining profile limits are tracked in S5.3 of the completion plan.

```bash
cd python
/path/to/official-engine/bin/python -m pytest -q -m release_smoke \
  tests/release/test_installed_wheel.py -k 'vllm and ssd' \
  --release-cuda-graph --orbitkv-transfer-backend kernel \
  --model /path/to/immutable-dense-model --basetemp /external/graph-kernel-cell
```

Run SGLang in its own official environment with `-k 'sglang and ssd'`; run direct
and kernel separately. Long prefills can stay eager; this gate does not qualify
full prefill capture, hybrid or native P/D fault lifetimes. It uses an installed
wheel and retains the normal release gate's RECORD, restart, HBM and SSD checks.
