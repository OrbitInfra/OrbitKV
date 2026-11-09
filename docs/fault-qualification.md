# Fault behavior and qualification

> Current OrbiTE GPU/RDMA/GDS and installed profiles are unqualified.
> The gates below describe required behavior, not new passing results.
> CPU/TCP and io_uring controls remain CI evidence.

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

## Reproduce

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
