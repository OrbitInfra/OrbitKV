# Fault behavior and qualification

> Current OrbiTE GPU/RDMA/GDS and installed profiles are unqualified.
> The gates below describe required behavior, not new passing results.
> CPU/TCP and io_uring controls remain CI evidence.

## Engine conformance matrix

The [test registry](../python/tests/support/engine_registry.py) assigns every
engine suite (and mixed-suite exceptions) an evidence level, independently of its
filename: **source**, **installed-CPU** or **installed-GPU**. Source contracts may
use stubs; installed profiles verify the exact official release, non-editable
package origins and real native extension before collection, and reject engine
module stubs before and after each test. CPU controls use released classes with
narrow CUDA-event/Manager-reply seams; they never claim to have drained DMA.

`ORBITKV_REQUIRE_ENGINE_TESTS=1`, `--engine=vllm|sglang` and
`--engine-evidence=source|installed-cpu|installed-gpu` select a required run.
Missing prerequisites, an empty selection, collection skips and unreviewed test
skips fail. No skip exceptions are approved in the registry. A future exception
needs an exact node/profile, reason and review link, not merely a `skip` marker.
Negative controls remove prerequisites, force skips, empty the selection and
inject a stub in the last test. JSON/JUnit inventory and package freezes go to
external/CI artifacts. A passing inventory is not hardware qualification.

The following controls run against **installed vLLM 0.33.0 / SGLang 0.5.23**
without a model, GPU or Manager. Test names below refer to these maintained suites:
VQ = [pending/identity](../python/tests/integration/test_engine_fault_contracts.py),
VL = [vLLM load](../python/tests/unit/test_connector_fault_tolerance.py),
VS = [vLLM save](../python/tests/unit/test_connector_save_lifecycle.py),
VA = [vLLM admission](../python/tests/unit/test_restore_admission.py),
SA = [SGLang admission/save](../python/tests/integration/test_sglang_admission.py),
SL = [SGLang load](../python/tests/integration/test_sglang_direct_transfer.py).
The selected V* unit assertions are explicitly replayed with installed classes;
their ordinary source-only run remains a different evidence level.

| Case, both engines | Logical outcome and owners until a physical receipt | CPU-run control / installed-GPU queue |
| --- | --- | --- |
| Abort with pending query | Stop demand; never invent a hit or release a lease not received. Manager read reservations and shared readers outlive submitted I/O. | VQ pending abort; SA native `finish(ABORT)`. GPU: SSD query-cancel barrier and model streaming cancellation. |
| Abort with load in flight | Fail/abort consumption; retain destination pages, tensor registration, source leases/grants and completion handles until drain. | VL submission/enqueue/drain failures; SL cancellation at each boundary. GPU: restore timeout/lost-notification and partial-enqueue barriers. |
| Abort with save in flight | Request finish cannot finish Save. Hold source pages, producer registration and save tasks/tree locks. | VA pending-save abort; SA native finish with pending offload; VS/SA unknown-wait controls. GPU: Publish-stall controls; real scheduler abort under Save remains unqualified. |
| Queue-wait timeout / cancellation | Recompute or cancel demand; timeout is not a release receipt. Never-submitted work may cancel without DMA. | VQ miss/cancel replies; SA real admission timeout/expiry; SL pre-forward cancel. GPU: SSD relative-deadline/best-effort controls. VQ injects the native deadline result, not a real Manager timer. |
| Preemption with Save in flight | Defer source reuse until every Save drains; unknown completion retains all holds. | VS preemption waits for all batches; SA retained FIFO offloads block reset/close. SGLang retraction scheduling is not exercised by that CPU control. GPU: actual pressure/preemption remains unqualified. |
| Free and reuse a page slot | A stale generation must never identify new bytes. Payload registration IDs are not allocator-slot generations. | VS reuse-before-last-batch negative control and SA captured-registration rejection. GPU: registration replacement/partial enqueue bytes only; full allocator-generation proof is blocked on the issues below. |
| Engine worker restart, live Manager | New registration may reuse compatible cached state; the dead worker's unfinished grants remain held/quarantined until a real receipt. | CPU tests cover unknown completion, not worker death. GPU-written: model engine restart and native session-watcher/engine-death controls. |
| Manager restart | Reject old handles/leases; cold miss is valid until M2, stale hit is not. Reconnect with a new incarnation. Process disappearance alone does not revoke submitted GPU work. | VQ/SA connection-loss retention, not process evidence. GPU-written: old-handle/lease rejection, explicit old-key miss and model cold-restart byte counter. |
| Same token hash, different identity | Revision, layout, computation, representation and scope must produce different namespaces; no cross-identity hit. | VQ for both engines plus content-fingerprint controls. GPU model/layout variants remain unrun. |
| Partial load: layer / Full-SWA pool fails | No complete-boundary hit or successful completion; keep all submitted destinations and source leases through terminal drain. | VL known/unknown failure; SL layer-ready versus drain; VQ Full/SWA partial failure. GPU: partitioned-restore and partial-enqueue byte controls. |
| Partial save | Unfinished state stays invisible. Release only terminal batches, never an unknown head or a later FIFO completion in its place. | VS multi-batch and unknown-wait controls; SA FIFO and terminal failed-publication consumer. GPU: Publish/write fault controls; partial serving remains unqualified. |

Every row's physical execution is **unrun and unqualified** for these pins.
The CPU seams cannot verify remote grant reclamation, CUDA ordering or worker
termination. The native GPU fixtures and model-serving gates are complementary;
neither alone proves the whole matrix. Allocator-generation issues
[vLLM #62810](https://github.com/vllm-project/vllm/issues/62810) and
[SGLang #45493](https://github.com/sgl-project/sglang/issues/45493) are open as
checked November 9, 2026. Do not add a private runner, shadow tree, synthetic
generation or a passing placeholder for that missing proof.

### V2 controls

The [V2 suite](../python/tests/integration/test_vllm_v2_contract.py) uses installed
vLLM 0.33.0 `Scheduler`, `KVCacheManager`, `Request`, `ActiveKVConnector` and the
OrbitKV adapter. CPU seams control only GPU events and Manager replies. It proves
the [admitted ordering](adapters.md#released-v2-lifetime-contract), not CUDA behavior.

| Case | Installed-CPU control | Written installed-GPU gate |
| --- | --- | --- |
| Preempt during Save | Reassign the actual block ID; V2's preemption barrier must prevent the next load until Save completion. | Pause native Publish; V2 dispatch waits; reallocate/overwrite after drain and restore the original saved bytes. |
| Abort during Load | Deferred allocator refs survive abort and a failed Restore wait; only a completed step releases them. | Pause after the first native CUDA enqueue; abort in the released scheduler; check retained refs and bytes before reallocation. |
| Finish before Save completion | `request_finished` delays free; no `finished_sending` until the native receipt. | Pause Publish, finish, verify held block; drain, reallocate and check saved bytes. |
| Freed/reallocated slot | The actual allocator reuses the same object/ID; the public preemption barrier fences writes, not ID reassignment. | The three cases above reuse the physical slot after their receipt and compare sentinels. No allocator-generation claim. |

All GPU entries remain **unrun and unqualified**. The native fixture drives
released scheduler/connector methods with real CUDA and a Manager, but no model
forward; the separate model-output/restart gate does not prove injected model
preemption. End-to-end model fault qualification remains in the M1 queue.

## Native fault boundaries

The deterministic process gate runs a real Cache Manager, retained engine
tensors, independently imported shared payload arenas, CUDA IPC physical routes,
and SSD reads. It uses a separate `test-hooks` build; default release
binaries contain no fault barriers. Each test owns its Manager and a private
barrier directory. Ordinary completion and model-output gates run separately.

| Fault | Required behavior |
| --- | --- |
| SSD completion paused after submission; query cancelled or replaced | Submitted buffers and reservations stay owned until I/O drains; old results cannot attach to the new query; unrelated queries progress; reservations return to zero. |
| One-page read batches with cancellation, a relative deadline or best-effort stopping | No second batch is submitted. A deadline lets demand recompute while the first batch drains with its budget retained. Unread pages are not classified as HLL misses. |
| One owner cancels a shared preparation read | Another demand owner completes from the shared read; cancellation cannot revoke its buffers. |
| Prepared result expires without another poll | The Manager releases the undelivered lease; a matching claim before expiry keeps its bytes owned through GPU completion. |
| Restore grant/result publication delayed and eventfd notification dropped | A wait deadline returns no ownership of destination pages. Polling the same handle discovers terminal completion; restored bytes match, and reservations drain. |
| Restore submission ACK corrupted after claim, with delayed/lost completion notification | The native client retains the pre-reserved handle despite closing descriptor admission. Its native result reports the real outcome. Cancellation during raw preparation is terminal only after source cleanup and must leave GPU pages untouched; Managed work still waits for its actual drain. |
| cuFile worker paused before reading, query cancelled and notification dropped | The SSD extent remains pinned through restore completion. A concurrent DRAM restore completes; polling recovers completion, bytes match and ownership counters drain. |
| cuFile write paused, completion failed or Manager killed | Unfinished objects stay invisible; GPU pages remain owned; DRAM restores progress; failed reservations can be retried and staging is released on unregister. |
| GDS hot-copy completion held while SSD work continues | Publish and unregister keep engine mappings; unrelated SSD demand restores complete with exact GPU bytes. |
| FP8 read canceled, mixed raw/encoded prefix or corrupted file | Scratch survives submitted I/O, FP8-representable inputs and raw fallbacks restore correctly, damaged decodes become misses and scratch returns to zero. |
| Publish delayed beyond the call deadline | The publisher retains source pages while other query sessions progress. Releasing the barrier completes the save. Killing the Manager terminates the wait safely. |
| Publish acknowledgement malformed | The session is poisoned and the publisher remains fenced until Manager death. Descriptor corruption cannot be mistaken for DMA completion. |
| Manager restart with old clients, leases and a pending restore | A fresh service incarnation starts behind the same UDS address; old handles/leases are rejected. A newly registered engine can publish and restore. Both default and configured service prefixes are exercised. |
| Engine process killed with registered CUDA IPC buffers | Session watching drains Manager-owned work before dropping its registration. Any claimed engine-local grant without drain evidence retains its source owners and query credits in quarantine. |

Publish logs a warning after the configured ordinary call deadline, then at
most once per minute. It never frees sources merely because a timer expired.
A permanently stuck live Manager requires operational restart; this gate does
not install an automatic process killer.

Channel tests additionally drop the submission ACK entirely, race cancellation
against claim in separate mappings, and prove a delayed request cannot execute
after cancellation. They distinguish UDS closure from actual peer process exit
and cover shared-record capacity, generation reuse, and duplicate claim rejection.

## Required installed-GPU runs

Use separate non-editable release environments with matching complete OrbitKV
wheels. Build a matching test-hooks Manager for the fault commands below. Make
a tests/benches-only projection so source `orbitkv` cannot shadow the wheel:

```bash
# From the repository root; choose a fresh external directory for each run.
HARNESS=/var/tmp/orbitkv-conformance/harness
mkdir -p "$HARNESS/python"
cp -R python/tests "$HARNESS/python/"
cp python/pyproject.toml "$HARNESS/python/"
cp -R benches "$HARNESS/"
# Set ENGINE, PYTHON, MODEL and MANAGER to this run's installed artifacts.
ENGINE=vllm
PYTHON=/absolute/path/to/vllm-release/bin/python
MODEL=/absolute/path/to/immutable-model
MANAGER=/absolute/path/to/test-hooks-manager
export ORBITKV_REQUIRE_ENGINE_TESTS=1 ORBITKV_FAULT_TESTS=1
export ORBITKV_CACHE_MANAGER_BINARY="$MANAGER" VLLM_USE_V2_MODEL_RUNNER=0
cd "$HARNESS/python"

# Real-model faults: repeat with ENGINE=sglang and its interpreter, dense Qwen3.
"$PYTHON" -m pytest -m '' --engine="$ENGINE" --engine-evidence=installed-gpu \
  tests/stress/test_recovery_faults.py --model "$MODEL" \
  --basetemp=/var/tmp/orbitkv-conformance/model-faults \
  --engine-report=/var/tmp/orbitkv-conformance/model-faults.json

# Native physical boundaries, in each engine environment; no model execution.
"$PYTHON" -m pytest -m '' --engine="$ENGINE" --engine-evidence=installed-gpu \
  tests/integration/test_cache_faults.py --model "$MODEL" \
  -k 'ssd_cancellation_revisions or stopping_submits or restore_timeout_and_lost_notification or publish_stall or manager_restart or local_partial_enqueue or partitioned_restore or engine_death_after_claim' \
  --basetemp=/var/tmp/orbitkv-conformance/native-faults \
  --engine-report=/var/tmp/orbitkv-conformance/native-faults.json
```

Select only the admitted profile; explicit selection is not permission to ignore
a skip inside it. Run these normal-build recovery gates separately, with the
same required flag, `--model`, external report and evidence-directory arguments:

```bash
# vLLM V1 / one attention group; HMA is explicitly outside this profile.
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  tests/e2e/test_vllm_e2e_correctness.py -k 'not same_process_hma' \
  --model "$MODEL" --max-model-len 4096 --vllm-cache-tier ssd

# SGLang: run the model gate for an admitted Full model and a Full/SWA model.
# Use the SGLang interpreter and each model's immutable path.
"$PYTHON" -m pytest -m '' --engine=sglang --engine-evidence=installed-gpu \
  tests/e2e/test_sglang_direct_e2e.py --model "$MODEL"
"$PYTHON" -m pytest -m '' --engine=sglang --engine-evidence=installed-gpu \
  tests/integration/test_sglang_recovery.py -k window --model "$MODEL"
```

Repeat vLLM recovery with `--vllm-cache-tier dram`; SGLang parameterizes DRAM/SSD.
Its `window` control uses declared Full/SWA pool fixtures, not model-generated
KV. Recurrent fixtures, V2, multi-group/HMA, Graph and P/D are separate profiles.
The harness checks CUDA, local model config/weights and an executable Manager;
missing test-hooks barriers fail by deadline rather than being skipped. These
commands do not manufacture the still-missing allocator-generation receipt.

### V2 opt-in gates

Keep ordinary commands above on V1. In the same tests-only projection, use the
vLLM 0.33.0 interpreter, an immutable dense Full-attention model and fresh output
directories. `--vllm-runner=v2` admits only `vllm_v2`-marked GPU tests; an unmarked
or empty selection fails. The connector opt-in is added by the model fixture.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1 ORBITKV_REQUIRE_ENGINE_TESTS=1
# Normal matching wheel + Manager: repeat with --vllm-cache-tier ssd.
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  --vllm-runner=v2 tests/e2e/test_vllm_e2e_correctness.py \
  -k 'not same_process_hma' --model "$MODEL" --max-model-len 4096 \
  --vllm-cudagraph-mode NONE --vllm-cache-tier dram \
  --basetemp=/var/tmp/orbitkv-conformance/v2-model-dram \
  --engine-report=/var/tmp/orbitkv-conformance/v2-model-dram.json

# Matching test-hooks wheel AND Manager: actual native transfers and sentinels.
export ORBITKV_FAULT_TESTS=1 ORBITKV_CACHE_MANAGER_BINARY="$MANAGER"
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  --vllm-runner=v2 tests/integration/test_vllm_v2_contract.py \
  -k native_v2_page_reuse_bytes --model "$MODEL" \
  --basetemp=/var/tmp/orbitkv-conformance/v2-native-reuse \
  --engine-report=/var/tmp/orbitkv-conformance/v2-native-reuse.json
```

These commands are **written only**, not qualification. The normal gate compares
model outputs and requires restart restore bytes; the fault gate compares exact
fixture bytes. Repeat failing runs as failures, never omit a case to make a gate
green. Neither gate resolves #62810 or authorizes release on an unknown receipt.

### Full+SWA gates

The [HMA suite](../python/tests/integration/test_vllm_hma_contract.py) is registered
as installed-CPU, except its explicitly registered native GPU byte test. CPU
controls use official 0.33.0 specs, requests, allocator, scheduler retirement and
V1 worker finalization; Manager replies and CUDA events are narrow seams.

| Case | CPU-run control | Written-only GPU evidence |
| --- | --- | --- |
| Atomic boundary | Full + window hit; missing either group, missing materialization or different boundaries miss entirely; changed allocation boundary rejects. One Restore owns all leases. | Per-group DRAM/SSD bytes and untouched out-of-window destinations. |
| Window retirement / preemption / abort during Save | Real `build_connector_meta` pins all groups; retire through the allocator, exhaust the free queue, and reallocate only after all-worker drain. Outstanding jobs reject at the pool-sized cap. | Pause native Publish, retire/preempt, retain the source, drain, overwrite its reused slot and restore the original bytes for both groups. |
| Abort / preemption during Load | Processed-token window retirement and deferred frees keep required destinations. A failed physical wait cannot return a V1 step; preemption waits for that step. | Pause CUDA restore, abort/refuse premature preemption, verify retained owners, drain and compare both groups before reuse. |
| Layout / startup exclusions | Changed order, window, block size, head shape or layer names change identity. Unknown specs, MLA, Mamba, zeroing, speculation/draft, host groups, producer-only, PP and V2 multi-group reject. | Model/layout variants remain unrun; recurrent is not an admitted GPU profile. |

Use the same tests-only installed projection and prerequisite variables described
above. Set `PYTHON` to the official vLLM 0.33.0 environment and `MODEL` to a local
immutable **Gemma 3 Full/SWA** checkpoint with its revision recorded. The model
gate requires admitted Full/SWA groups in both engine processes, so a dense model
cannot substitute. Start eager TP=1/PP=1 with raw KV and no speculation.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0 ORBITKV_REQUIRE_ENGINE_TESTS=1
export ORBITKV_REQUIRE_VLLM_FULL_SWA=1
# Matching normal wheel + Manager; repeat with --vllm-cache-tier ssd.
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  --vllm-runner=v1 tests/e2e/test_vllm_e2e_correctness.py \
  -k 'not same_process_hma' --model "$MODEL" --max-model-len 4096 \
  --vllm-cudagraph-mode NONE --vllm-cache-tier dram \
  --basetemp=/var/tmp/orbitkv-conformance/hma-model-dram \
  --engine-report=/var/tmp/orbitkv-conformance/hma-model-dram.json

# Exact Full/SWA fixture bytes in DRAM and io_uring SSD. Checkpoint cases excluded.
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  tests/integration/test_vllm_recovery.py \
  -k 'window_and_checkpoint_restore and False' --model "$MODEL" \
  --basetemp=/var/tmp/orbitkv-conformance/hma-bytes \
  --engine-report=/var/tmp/orbitkv-conformance/hma-bytes.json

# Matching test-hooks wheel AND Manager; no engine or CUDA module stubs.
export ORBITKV_FAULT_TESTS=1 ORBITKV_CACHE_MANAGER_BINARY="$MANAGER"
"$PYTHON" -m pytest -m '' --engine=vllm --engine-evidence=installed-gpu \
  tests/integration/test_vllm_hma_contract.py -k native_hma_page_reuse_bytes \
  --model "$MODEL" --basetemp=/var/tmp/orbitkv-conformance/hma-native-reuse \
  --engine-report=/var/tmp/orbitkv-conformance/hma-native-reuse.json
```

All three are **unrun and unqualified**. The excluded `same_process_hma` assertion
belongs to the old recurrent fixture, which is not admitted here; restart hits
and output comparisons remain required. Native fixtures prove different facts
from a serving run; neither supplies the missing allocator-generation contract.
Real model-level pressure/preemption and abort remain required hardware follow-up.

### Build the native fault runtime

Build the matching OrbiTE artifacts before starting any native/GPU tests.
These commands target the current path; historical results do not qualify it.

```bash
PYO3_PYTHON=$PWD/.venv/sglang-release/bin/python \
  cargo build --release -p orbitkv-server -p orbitkv-py \
  --no-default-features --features cuda-13,orbite,orbitkv-server/test-hooks
# Install the matching extension using the development/build instructions.
cd python
ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
ORBITKV_FAULT_TESTS=1 ../.venv/sglang-release/bin/python -m pytest -m integration \
  tests/integration/test_cache_faults.py tests/integration/test_session_watcher.py
```

The test fixture alone sets `ORBITKV_TEST_FAULTS` for its private process. Do not
ship `test-hooks` binaries. Run the [hybrid recovery gates](hybrid-recovery.md#reproducible-gates)
with normal builds as well. A container file test cannot establish physical NVMe or RDMA behavior.

The cuFile cases additionally require `--ssd-backend cufile` and a writable
`--basetemp` on a cuFile-compatible mount. Use the environment settings in
[GPU storage recovery](gds.md) and record whether compatibility mode was enabled.
Without this option, the ordinary fault gate skips the cuFile-only cases.

## Concurrent Qwen3 serving

The explicit stress gate uses Qwen3-8B with deterministic inference in each
pinned engine. It restarts the engine while retaining the Manager, pauses SSD
reads and abandons a streaming request while an unrelated request progresses,
drops completion notifications, and kills the Manager during a GPU restore.
It checks reference outputs, positive restore bytes, cancellation observations
and final ownership counters. A restarted Manager starts cold; this is not an
SSD index persistence test.

```bash
cd python
ORBITKV_FAULT_TESTS=1 ORBITKV_PREPARE_REQUESTS=1 \
ORBITKV_CACHE_MANAGER_BINARY=/absolute/path/to/test-hooks-manager \
../.venv/vllm-release/bin/python -m pytest -m stress \
  tests/stress/test_recovery_faults.py -k vllm --model /workspace/models/qwen3-8b \
  --basetemp=/var/tmp/orbitkv-bench/serving-fault-vllm
```

Use the SGLang interpreter and `-k sglang` for its gate. Set
`ORBITKV_PREPARE_REQUESTS=0` for ordinary demand. Keep the two GPU runs sequential.
The workspace test directory retains every engine/Manager incarnation log and
`fault-results.json`. Manager sockets use short temporary paths independently
of the evidence directory.

Multi-rank serving, long-running injected-fault traffic, hardware hangs and remote
failover have separate qualification gates. Logical framework page-generation
evidence remains separate from payload-allocation IDs and process/session fencing.

## Shared-cache gates

Use matching immutable models, engine releases, layouts, identities and
independent Manager sockets/pools. Start with [P2P](p2p.md) and fresh namespaces;
keep optional preparation disabled. Run byte checks before model comparisons.
For forced peer SSD, evict the source DRAM only after saves complete, then require
positive source SSD reads plus requester OrbiTE READ and GPU restore bytes.
An HTTP success or an HBM-only hit is not peer recovery evidence.

### Byte-exact GPU recovery

Use `benches.distributed.shared_cache_bytes` before model comparisons on different GPU
architectures. It checks four split-K/V layers, source pages spaced by one gap,
permuted destination pages and untouched sentinel bytes. After an 8 MiB remote
Restore, the driver evicts the original source's DRAM and requires the received
replica to serve the same bytes back. Both directions require 8 MiB of OrbiTE
READ and GPU Restore, source-release acknowledgements, matching SHA-256 digests
and drained resource counters. This checks the full GPU-to-Manager-to-peer-to-GPU
path; it does not claim direct GPU-to-GPU RDMA or model-output equivalence.

Start fresh DRAM-only test Managers, with no unrelated traffic, using the
P2P setup. Run one fixture worker beside each Manager from a checkout
with a built or installed OrbitKV extension and the engine's Torch environment.
The workers expose only fixed test operations; use the trusted test network:

```bash
# Host A; on host B use 10.0.0.2 and --instance byte-target.
python -m benches.distributed.shared_cache_bytes worker \
  --host 10.0.0.1 --port 8002 --manager-socket /tmp/orbitkv-50055.sock \
  --instance byte-source --namespace byte-qualification

# Controller; leave both workers running during this command.
python -m benches.distributed.shared_cache_bytes run \
  --source-url http://10.0.0.1:8002 --target-url http://10.0.0.2:8002 \
  --source-manager http://10.0.0.1:9091 --target-manager http://10.0.0.2:9091 \
  --output /var/tmp/orbitkv-bench/gpu-byte-roundtrip.json
```

The source DRAM cleanup is intentional. Stop both workers after the gate; use
fresh test Managers and a matching namespace on both workers for each rerun.

### Model requests

Create `prompts.json` containing fresh token-ID arrays of at least 128 tokens,
tokenized with the deployed model's tokenizer. Prefer distinct first pages and
several prompt lengths. Each prompt is first computed by the source, then sent
to a consumer that has never seen it.

```bash
python -m benches.distributed.shared_cache \
  --engine vllm --model /path/to/immutable-model \
  --source-url http://10.0.0.1:8000 --target-url http://10.0.0.2:8000 \
  --source-manager http://10.0.0.1:9091 --target-manager http://10.0.0.2:9091 \
  --prompts /path/to/prompts.json --deployment two-host-tcp \
  --output /var/tmp/orbitkv-bench/shared-cache-vllm.json
```

Use `--engine sglang` for its native serving endpoint. The driver requires only
the benchmark HTTP dependencies, not an installed engine or CUDA runtime.

The source must expose new bytes. `POST /cache/sync` waits for already submitted
saves and returns an `inventory_fence`; the driver sends it to the consumer's
bounded `/cache/metadata/await` endpoint with the exact scope digest. Each
consumer request must increase both OrbiTE READ and GPU
restore bytes, match the cold source output, and drain query, source-transfer
and I/O reservations, including requester completion records awaiting a source
acknowledgement. A response without these counters does not pass as a
remote hit. This gate proves recovery, not throughput superiority.

### Restart and ownership

The repository's model-serving test starts etcd, two Managers and two replicas
on one GPU. It checks ordinary sharing, a complete index rebuild after restarting
the consumer, and a clean recomputation after the source restarts without its payload.
The gate requires each explicitly restarted Manager to exit successfully within
10 seconds without the cleanup helper's forced-kill fallback. It then observes
old membership deletion and verifies an increased epoch and changed incarnation
before testing the new process. This exercises inventory-stream shutdown as well
as restart fencing. A crash can leave its old registration until lease expiry.
It runs separately in the pinned vLLM and SGLang environments:

```bash
cd python
ETCD_BIN=/path/to/etcd \
ORBITKV_CACHE_MANAGER_BINARY=/path/to/orbitkv-cache-manager \
  ../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_shared_cache.py -k vllm --model /workspace/models/qwen3-8b
```

Repeat with `../.venv/sglang-release/bin/python` and `-k sglang`. The fixture uses
the installed OrbitKV wheel and explicit eager mode. It resolves input paths
before starting child processes in a private temporary directory. Its common vLLM launcher
sets the official `--shutdown-timeout 30`; the consumed shutdown validator
requires native drain/teardown messages and rejects abort or force-kill logs.
Service-leader exit zero alone cannot pass this check. Preserve SGLang's native
child exit statuses and CUDA IPC warnings separately; healthy output and final
GPU zero do not establish physical lifetime or fault reclamation.

Build the Manager before starting these processes. Raw logs belong in external artifact directories or CI
artifacts; retain only the final summary and reproduction commands in a PR.

Rust tests separately verify stale owner/residency rejection, source budget
exhaustion, retained source allocations after timeout, cancellation during a
blocking transfer, bounded retry of lost release replies, lost authorization
replies after pinning, close-before-authorize races, stale slot generations and
idle-window eviction. Connection recovery verifies that an old requester
completion cannot release a new runtime’s hold. These tests cannot prove
transport revocation after a permanently lost requester. Such source pins remain
charged until safe release or coordinated Manager teardown. Real partitions,
multi-rank replicas and metadata HA remain separate gates; historical results do not qualify the current OrbiTE path.


Current required gates must not treat absent prerequisites or unexpected skips
as success. M1 completes real-engine conformance; M2 adds persistent SSD crash
recovery. The [historical shared-cache record](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/shared-cache-qualification.md)
retains earlier native/TENT revisions, forced-SSD recipes and failed controls.
