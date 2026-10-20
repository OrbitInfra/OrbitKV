# Single-node fault qualification

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

## Completion-evidence requalification

The September 28 completion-evidence increment uses bootstrap 7, channel ABI 11,
cache schema 8 and lifecycle 4. Its matching test-hooks build passes **12 selected
H20 process/GPU fault cases in 37.22 seconds**. They cover the local Restore
faults below plus lost completion notification/ACK, restart with stale clients,
and `test_local_completion_evidence_excludes_retirement_and_trains_once`.
That test pauses Manager retirement after native drain: the engine consumes
correct GPU bytes while Manager source reservations remain held. On release,
the report trains once with the native duration, excluding the delayed reap.
Channel tests also reject stale/recycled or duplicate reports and discard
invalid optional timings without blocking authoritative completion.

The final production build, including source/backend cost-key isolation
(`c1d8d43c`), passes **7** native channel/client cases in 13.02 seconds. On
Qwen3-8B with direct, unencoded DRAM, vLLM passes **6 / 1 skipped** in 279.98
seconds and SGLang passes **1 / 1 deselected** in 204.96 seconds. The vLLM skip
is the dense model's inapplicable recurrent-state case; the SGLang SSD case was
not selected. Both use their existing deterministic restart/output contracts.
This is single-GPU serving correctness, not multi-rank/P-D/RDMA qualification.

| Final production artifact | SHA-256 |
| --- | --- |
| Manager | `85ec9d8fa351259970ee24d06ec504f39d2651ee362a0666e41565eb1dcc4b34` |
| Native extension | `5cfef266efa208318cc0e8595dd7f06f335aa179a86ab06a8bdd573e5f39ecce` |

Logs and artifact identifiers are retained in
`.orbitkv/completion-evidence-20260928/`. The diagnostic performance cohorts in
[communication measurements](https://github.com/feichai0017/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/communication-performance.md#completion-evidence-and-copy-path-diagnosis)
use the preceding Manager build; their hashes are recorded separately.

## Engine-local raw Restore gates

The frozen `compact-restore-final-fault` bundle passed the selected real
Manager/native-client integration and fault suite: **38 passed, 30 skipped in
97.05 seconds**. The skipped cases require the cuFile configuration, which was
not selected. All selected single-GPU local Restore lifecycle cases listed below
passed with bootstrap 6/channel ABI 9/lifecycle 4.

| Frozen artifact | SHA-256 |
| --- | --- |
| Cache Manager | `c60b7a780bc24a8eaa9e95f4f7d8aabd9c6498afed6cc9ff64d772d78c0ffbbc` |
| Python native extension | `fee45f97cfda2a8980b2cdb33f76782c434ff6dc89511950c024efc86e9a2806` |

The cases are implemented in
[`test_cache_faults.py`](../python/tests/integration/test_cache_faults.py).
Earlier Manager-owned Restore results remain separate from this evidence.

| Passed test | Observed contract |
| --- | --- |
| `test_local_restore_survives_manager_death_after_claim` | Kill the Manager after claim and after the first copy enqueue. The engine must remain pending while its local barrier is held, then drain and verify exact GPU bytes through its own imported registration. |
| `test_local_partial_enqueue_failure_drains_before_page_reuse` | Inject failure after an accepted copy; the failed result permits page reuse only after stream drain, and Manager source reservations eventually retire. |
| `test_engine_death_after_claim_quarantines_source_reservation` | Kill a separate engine process after claim; the live Manager retains its charged sources despite UDS loss. |
| `test_local_restore_retains_tensor_when_caller_drops_handle_and_tensor` | Dropping Python tensor references and the handle cannot free the native binding during a pending copy; unregister releases it after drain. |
| `test_local_restore_fences_previous_use_on_nondefault_stream` | Two successive restores use different busy nondefault engine streams. Each previous write finishes before Restore overwrites the destination; exact bytes prove that the reused event does not substitute an earlier completion. |

The CUDA unit test
`caller_context_survives_registration_and_readiness_success_and_failure` in
[`transfer/local.rs` tests](../crates/orbitkv-core/tests/unit/transfer/local.rs)
also passed, covering restoration of the caller's CUDA context after both
successful and rejected registration/readiness paths.

The final release workspace gate passed **488 tests / 38 ignored**, excluding
nested child-helper invocations from the pass count. The context test above was
also run explicitly outside that default gate. Python unit tests passed **374**
cases and benchmark-tool units passed **199**. The matching production Manager
and extension passed all **7** ordinary channel/client GPU integration cases.
The same-host Mooncake TCP raw and encoded peer round trips also passed. These
remote checks qualify data correctness through the new local executor, not
cross-host RDMA performance.

The compaction unit gates cover permuted dense/sparse destinations, split K/V,
adjacent distinct allocation IDs, destination overlap, a 32,768-page dense plan,
and lease preservation for oversized fragmented plans. The existing 32,768-block
GPU benchmark passes Criterion `--test` mode; it checks submission/drain, while
the matched communication runs separately verify GPU bytes and copy counters.

The `local_restore_dma` barrier is reached after the first enqueue call. It
proves that Manager death does not manufacture local completion, and that
retained imports support subsequent drain and correct bytes. It does **not**
prove that the copy was physically in flight at the exact instant of SIGKILL.
The engine-death case pauses after claim; it verifies conservative source
quarantine, not engine death during proven active hardware DMA.

The native worker captures `ready_stream` and owns tensors independently of
Python waiters. Its local terminal result precedes asynchronous Manager source
reaping. The channel suite separately covers claim/revoke races, plan-bank
pressure, record generations, and retirement acknowledgement. Full serving,
multiple-GPU, huge-page, prolonged allocator-pressure, and graph replay gates
remain separate from these single-GPU process tests. See
[engine-local qualification](engine-local-restore.md#qualification-gates).

The matching production bundle subsequently passed Qwen3-8B DRAM serving
correctness on H20: vLLM **6 passed / 1 skipped** (the skipped recurrent-state
case does not apply to this dense model), and SGLang **1 passed / 1 deselected**
(DRAM selected, SSD excluded). Both exercise engine restart with a retained
Manager. These are normal serving gates, not additional injected-fault tests;
the [exact configuration](https://github.com/feichai0017/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/single-node-performance.md#engine-local-restore-serving-qualification)
does not qualify arbitrary graph modes, hybrid models, or multiple GPUs.

## Reproduce

Build before starting any native/GPU tests: Mooncake shared libraries are
restaged by Cargo and must not be replaced under running processes.

```bash
PYO3_PYTHON=$PWD/.venv/sglang-release/bin/python \
  cargo build --release -p orbitkv-server -p orbitkv-py \
  --no-default-features --features cuda-13,mooncake,orbitkv-server/test-hooks
# Install the matching extension using the development/build instructions.
cd python
ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
ORBITKV_FAULT_TESTS=1 ../.venv/sglang-release/bin/python -m pytest -m integration \
  tests/integration/test_cache_faults.py tests/integration/test_session_watcher.py
```

The test fixture alone sets `ORBITKV_TEST_FAULTS` for its private process. Do not
ship `test-hooks` binaries. Run the [hybrid recovery gates](hybrid-recovery.md#reproducible-gates)
with normal builds as well. The H20 container verifies functional SSD I/O on its
mounted filesystem; it does not measure physical NVMe or RDMA behavior.

The cuFile cases additionally requires `--ssd-backend cufile` and a writable
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
