# Shared-cache qualification

The stage021 three-host etcd process-only result is independently accepted for
normal leader SIGTERM/rejoin, quorum stop/recovery and complete final keys.
The leader stop performs graceful leadership transfer. No Manager or payload
consumer runs in that cohort, so it does not qualify abrupt crash detection,
cache/source authorization, Watch recovery, GPU lifetime or independent physical
failure domains. The completion plan retains the remaining S2/S3 gates.

All recorded serving results below predate the S2.8 owner inventory-stream
cutover. They remain historical controls and do not qualify the current
candidate until rerun with frozen S2.8 artifacts. Their directory-RPC,
publication/Watch and catalog-restart terminology describes the corresponding
older revisions.

This gate covers independent replicas of the same model, engine and TP=1 storage
layout. Every engine uses its host's Cache Manager. Rust owns candidate lookup,
source validation, memory budgets and Mooncake transfers; Python only drives
serving requests and checks the exported results.

The driver accepts already running replicas, so the same requests can run on one
machine or on two hosts. A deployment label records the operator's setup; it is
not automatic proof of distinct physical hosts or an RDMA transport. Keep these
three results separate: same-host TCP, two-host TCP and two-host RDMA.

## Recorded result

### Historical etcd-block global-index cutover, 2026-09-29

The new `/orbitkv/v2` path uses fenced etcd publication and fixed-revision
snapshot/Watch into each Manager's complete global index. The tests build native
artifacts first and run with Cargo stopped. Both pinned engines pass the same-host
H20 Qwen3-8B serving/restart gate: three remote restores per engine, 288 MiB TENT
READ and H2D, exact output matching, acknowledged releases and drained resources.
Restarting the consumer rebuilds its index; restarting the empty source causes
correct recomputation with zero remote/H2D bytes.

Physical two-host TCP runs on the H20 and A100 pass with the same Qwen3-8B model,
TP=1, 64-token pages and deterministic eager settings used by the historical
natural-text gate. Both engines restore the requested cabinet keys exactly.

| New metadata-path gate | vLLM 0.29.0 | SGLang 0.5.20 |
| --- | --- | --- |
| DRAM sharing and consumer index restart | Passed; 288 MiB READ/H2D | Passed; 288 MiB READ/H2D |
| Empty source restart | Passed; recomputation, zero remote/H2D | Passed; recomputation, zero remote/H2D |
| Forced source SSD after DRAM eviction | Passed; 216 MiB SSD reads, READ and H2D | Passed; 216 MiB SSD reads, READ and H2D |
| Output equality and resource drain | Passed in all six cases | Passed in all six cases |

The cluster group passes seven cases, including six real-etcd/GPU gates: duplicate identity and
incarnation restart, compaction, three-member leader loss and quorum-loss fencing,
lost publication replies and delayed retry rejection, independent DRAM/SSD
entries, a paginated snapshot exceeding the default 4 MiB gRPC decoder, owner
reconciliation, 260-block raw GPU recovery and four encoded formats. Logical
index budget failure and per-medium candidate bounds are covered in unit tests.
Three etcd processes on one host qualify process faults, not host-failure domains.

The release Manager and extension used on both physical hosts have matching
SHA-256 values:

- Manager: `38937c1104a659436cf81e38f719d52f055ecf37c34a83bb7d2c87ce1acd6964`.
- Extension: `b87a1f828119cc6e25f9a32449804cfc1f2f1b0da0c6bc2a0f7538e8482376b6`.

This is source-build correctness evidence, not a throughput comparison, installed
wheel/container qualification, RDMA validation or permanent-requester revocation
proof. Raw runs are retained under `/workspace/orbitkv-index-runs/`.

### Two-host TCP, 2026-09-28

Native source `d2ef3a60` runs on a local H20 and a remote A100 over IPv6 TCP,
with identical Manager/extension artifacts and Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`. Both Managers use separate 2 GiB
DRAM pools, TP=1 and 64-token pages; preparation is disabled. vLLM 0.29.0 uses
explicit FlashAttention 2 on both hosts and `VLLM_BATCH_INVARIANT=1`. SGLang
0.5.20 uses its deterministic flag. Both use eager execution.

| Gate | vLLM | SGLang |
| --- | --- | --- |
| Fresh 513/1025-token remote Restore | Passed; 72 + 144 MiB READ/H2D | Passed; 72 + 144 MiB READ/H2D |
| Sole catalog host restart, inventory replay and Restore | Passed; 72 MiB READ/H2D | Passed; 72 MiB READ/H2D |
| Source restart without payload | Passed; correct recomputation, zero remote/H2D bytes | Passed; correct recomputation, zero remote/H2D bytes |
| Output controls and source-release/resource drain | Passed in every recorded case | Passed in every recorded case |

The natural-text prompts require retrieving a specific cabinet key from the
cached first page among distractors. Complete eight-token outputs match the
source controls and contain the correct key. Each engine transfers and restores
288 MiB in total. Raw launches, request outputs and counter deltas are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/two-host-natural-20260928/`.

A separate [byte-exact gate](#byte-exact-gpu-recovery) passes 8 MiB in each
direction, including re-serving the received replica after original-source
eviction. Every K/V byte, destination-page permutation and sentinel gap matches,
with source-release acknowledgements and drained counters. It found and now
regresses a real placement error: H20's NUMA 0 had been used for allocations
on the A100 host, whose GPU-local pool is NUMA 1. The fix derives destination
placement from the receiving instance's registered slots; host NUMA identifiers
never define cross-host storage identity. The final normal release with bounded
Restore partitioning repeats this bidirectional byte gate successfully; its
outputs and matching native hashes are retained under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/partitioned-restore-20260928/`.

**Numerical scope:** the initial random-token model suite did not pass strict
cross-GPU output equality. Native monolithic H20/A100 controls also differ for
some of those prompts; forcing FA2 on both hosts alone does not close the
remote-output failure. A100 native HBM prefix hits match their own cold controls. The same random
513/1025-token requests also pass through two vLLM/FA2 replicas on one A100,
with 72/144 MiB remote/GPU restores; those diagnostic logs are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/same-host-a100-20260928/`.
Keep the rejected cross-GPU runs and native controls under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/two-host-20260928/`; do not describe the natural-text or
byte gate as proving arbitrary cross-GPU token equality.

The model-serving rows above are source-build correctness results. A later
[installed-wheel gate](releases.md#cuda-13-candidate-qualification-2026-09-28)
repeats the byte-exact roundtrip on both hosts. Neither container exposes RDMA
hardware, and the hosts have different GPUs. These results do not qualify RDMA,
GPUDirect, distributed throughput, P/D or mid-transfer crash/partition revocation.

### Earlier same-host gates

The [2026-09-25 ownership-layout recheck](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#ownership-layout-final-evidence)
passed on native source `046b16f5` with a matching query-body-v6 client and frozen
Manager. Both engines completed three remote GPU restores, catalog restart
replay and correct recomputation after source payload loss. Each transferred
and restored 288 MiB; outputs matched and checked resource counters drained.
This remains single-H20, same-host TCP evidence.

The [2026-09-23 Qwen3-8B gate](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results/20260923-shared-cache/README.md)
passed on one H20 with vLLM 0.29.0 and SGLang 0.5.20, tested separately. Each
engine completed three remote GPU restores, including catalog restart recovery,
and one correct recomputation after source payload loss. Outputs matched and
the checked resource counters drained in every case. Each engine transferred
288 MiB remotely and restored the same amount to HBM. The final rerun includes
reusable transfer windows and slot generations for lost authorization replies;
513/1025-token requests used 3/6 discovery RPCs on each engine, retaining the
catalog-host batching reduction from 8/14 on vLLM and 6/16 on SGLang in the
preceding gate. All 12 source-release
acknowledgements per engine were observed, and requester completion slots drained.

This is a same-host TCP correctness result. Short-prompt restoration was not
consistently faster than recomputation. Physical two-host/RDMA deployment and
performance comparisons still require separate measurements.

## Start matching replicas

Follow [distributed startup](p2p.md) for etcd and two Managers, then the
[single-node instructions](single-node.md) for the selected engine on each host.
Use the same immutable model, engine version, KV dtype, block/page size, TP=1
and `PYTHONHASHSEED=0`. Use separate Manager pools, instance IDs, sockets and
ports. Do not serve unrelated traffic during the gate.

For deterministic Qwen3 controls, set `VLLM_BATCH_INVARIANT=1` for vLLM or
`--enable-deterministic-inference` for SGLang. Keep preparation disabled. Start
with DRAM-only Managers for the sharing/restart baseline. Qualify peer SSD in a
separate run: force source DRAM eviction and record SSD read/staging evidence.
The cutover results above cover this ordinary two-host SSD route; mixed-load
selection and cancellation still need separate qualification. A two-host TCP
run sets `MC_FORCE_TCP=1` on both Managers. For RDMA, expose the devices and
select the appropriate `--nics`; record the actual Mooncake transport and NIC
counters with the result.

The Manager's HTTP endpoint must be reachable by the qualification driver on a
trusted test network. It includes administrative operations and is not a public
inference endpoint. The engine still uses UDS/iceoryx2, regardless of the driver
location.

## Byte-exact GPU recovery

Use `benches.shared_cache_bytes` before model comparisons on different GPU
architectures. It checks four split-K/V layers, source pages spaced by one gap,
permuted destination pages and untouched sentinel bytes. After an 8 MiB remote
Restore, the driver evicts the original source's DRAM and requires the received
replica to serve the same bytes back. Both directions require 8 MiB of Mooncake
READ and GPU Restore, source-release acknowledgements, matching SHA-256 digests
and drained resource counters. This checks the full GPU-to-Manager-to-peer-to-GPU
path; it does not claim direct GPU-to-GPU RDMA or model-output equivalence.

Start fresh DRAM-only test Managers, with no unrelated traffic, using the
cluster setup above. Run one fixture worker beside each Manager from a checkout
with a built or installed OrbitKV extension and the engine's Torch environment.
The workers expose only fixed test operations; use the trusted test network:

```bash
# Host A; on host B use 10.0.0.2 and --instance byte-target.
python -m benches.shared_cache_bytes worker \
  --host 10.0.0.1 --port 8002 --manager-socket /tmp/orbitkv-50055.sock \
  --instance byte-source --namespace byte-qualification

# Controller; leave both workers running during this command.
python -m benches.shared_cache_bytes run \
  --source-url http://10.0.0.1:8002 --target-url http://10.0.0.2:8002 \
  --source-manager http://10.0.0.1:9091 --target-manager http://10.0.0.2:9091 \
  --output /var/tmp/orbitkv-bench/gpu-byte-roundtrip.json
```

The source DRAM cleanup is intentional. Stop both workers after the gate; use
fresh test Managers and a matching namespace on both workers for each rerun.

## Run requests

Create `prompts.json` containing fresh token-ID arrays of at least 128 tokens,
tokenized with the deployed model's tokenizer. Prefer distinct first pages and
several prompt lengths. Each prompt is first computed by the source, then sent
to a consumer that has never seen it.

```bash
python -m benches.shared_cache \
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
consumer request must increase both Mooncake READ and GPU
restore bytes, match the cold source output, and drain query, source-transfer
and I/O reservations, including requester completion records awaiting a source
acknowledgement. A response without these counters does not pass as a
remote hit. This gate proves recovery, not throughput superiority.

## Forced source-SSD gate

The 2026-09-28 two-host H20→A100 TCP run passes on vLLM 0.29.0 and SGLang
0.5.20 with Qwen3-8B and the same natural-text controls described above. After
source DRAM eviction, each engine recovers 513/1025-token requests with
72/144 MiB of source io_uring reads, remote TENT READs and destination GPU
restores. Complete outputs match the source controls, source staging succeeds,
and both sides' checked reservations drain. Raw launches, output controls and
counter snapshots are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/partitioned-restore-20260928/{vllm,sglang}-shared-ssd/`.
This qualifies ordinary source-SSD recovery over TCP; it does not cover
cancellation during SSD staging/READ, RDMA or a throughput advantage.

Run this as a separate result from the DRAM baseline. Configure SSD on the
source Manager with write policy `all`; the target may use its ordinary local
configuration. Add `--source-medium ssd` to the command above; the driver
performs the synchronization, DRAM-only cleanup and source/target evidence
checks below. For each prompt:

1. Execute it on the source, call `POST /cache/sync`, and verify the SSD write
   completed before changing residency.
2. Call `POST /cache/memory/cleanup` on the source. Require positive evicted
   blocks, zero `still_referenced_blocks`, and surviving global-index evidence with
   medium `ssd`; do not clear the SSD ring or restart the source.
3. Send the prompt to a target replica with a cold local namespace. Require an
   increase in source `orbitkv_ssd_prefetch_bytes_total`, target
   `orbitkv_remote_fetch_bytes_total`, and target GPU restore bytes. The output
   must match the deterministic cold control.
4. After completion, require source `orbitkv_ssd_read_pinned_bytes`,
   `orbitkv_transfer_reserved_bytes` and `orbitkv_transfer_lock_active` plus
   target query and `orbitkv_transfer_completion_outstanding` reservations to
   return to baseline. Repeat cancellation
   while source I/O is queued and while Mooncake READ is active.

Run first with `MC_FORCE_TCP=1`, then on two physical hosts with the intended
RDMA rails and `--nics`. Record the actual Mooncake transport, NIC counters,
source SSD device/mount, byte amplification, authorization/staging/READ
latencies and peak sender/receiver memory. A successful same-host or TCP run
does not qualify RDMA, and a remote byte increase without a source SSD-read
increase does not qualify the peer-SSD route.

For cost evidence, first set only `ORBITKV_COST_OBSERVATIONS=1`. Keep
`ORBITKV_COST_SELECTION` unset so the fixed route remains the control. Record
`local_ssd_host_ready`, `peer_dram_host_ready`, `peer_ssd_host_ready` and shadow
decisions for identical stored-byte/block shapes. Multi-owner or different-
coverage rows are not valid cross-medium comparisons.

Only after that control passes, run a separately labelled experiment with all
three variables set to `1`: `ORBITKV_COST_OBSERVATIONS`,
`ORBITKV_COST_SELECTION`, and `ORBITKV_CROSS_MEDIUM_SELECTION`. Require positive
`orbitkv_cost_route_decisions_total{decision="selected",scope="cross_medium"}`
evidence, the same
output/byte/drain checks, and a matched fixed-route control. Do not carry the
third flag into ordinary serving until both engines pass TCP and RDMA cells.

For the later GPU-memory gate, register each granted region with the exact
Mooncake `cuda:N` location and keep the engine page-generation owner beside the
registration token until completion. Exercise explicit unregister, cancellation,
engine-handle release and instance teardown. Do not treat successful host-memory
RDMA or a configured `--nics` value as GPUDirect evidence; record GPU/NIC counters
from the external H20 hosts.

The artifact under test must contain `libtent_shared.so` and must not contain
`libtransfer_engine.so`. Capture TENT startup logs showing the installed
transports. A CUDA-enabled build alone is not evidence that a batch selected
RDMA, NVLink or GPUDirect.

## Restart and ownership gates

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

Build the Manager before starting these processes. Native builds restage
Mooncake libraries. Raw logs belong in external artifact directories or CI
artifacts; retain only the final summary and reproduction commands in a PR.

Rust tests separately verify stale owner/residency rejection, source budget
exhaustion, retained source allocations after timeout, cancellation during a
blocking transfer, bounded retry of lost release replies, lost authorization
replies after pinning, close-before-authorize races, stale slot generations and
idle-window eviction. Connection recovery verifies that an old requester
completion cannot release a new runtime’s hold. These tests cannot prove
transport revocation after a permanently lost requester. Such source pins remain
charged until safe release or coordinated Manager teardown. Real partitions,
multi-rank replicas and metadata HA remain separate gates; scoped two-host TCP
serving results are recorded above.

## Full-Manager remote DRAM path timing

Start two dedicated Managers and the maintained GPU byte fixture beside each.
For this descriptive profile, run the fixture worker with `--profile` and choose
`--segment-bytes 4096`, `65536` or `262144` for 512 KiB, 8 MiB or 32 MiB of
payload. Both workers must use the same segment size, namespace and layout.
The production wheel and native libraries are frozen separately from the test
harness. Payload timing always runs through the actual Rust Manager.

```bash
python -m benches.shared_cache_profile \
  --source-url http://source-worker:9100 \
  --target-url http://consumer-worker:9100 \
  --source-manager http://source-manager:9091 \
  --target-manager http://consumer-manager:9091 \
  --payload-bytes 8388608 --warmup 5 --samples 30 \
  --output /var/tmp/cache-profile/run-001.json
```

The first read is kept as a cold sample. Five additional warm-up operations
are retained separately from the thirty measured operations. Consumer DRAM is
evicted before every later read, so a reused prefix cannot become a local cache
hit. Every sample requires exact remote-fetch and H2D byte deltas, one native
authorization/allocation/READ/rebuild/release observation, full destination
tensor and sentinel equality, and completed query/source ownership drain.

Manager stage durations come from before/after metrics on the consumer's clock;
worker query/restore durations use its own monotonic clock. They are not
cross-host timestamp subtractions. Restore timing covers native submit/wait and
verified completion, not an isolated CUDA DMA duration. The query timing also
includes the fixture's explicit 10 ms pending-query polling; that is not model
scheduler latency. GPU clearing, byte/hash checks, HTTP transport, eviction,
barriers and quiescence waits are separately recorded outside those clocks.
The requests are serialized with eviction, barriers and three idle observations
between samples; this is a paced latency profile. Subtracting polling sleep from
the query clock does not produce an event-driven engine latency measurement.

Compare both directions with a fixed single-NIC or four-NIC allowlist. Record
actual physical NIC counter deltas; a four-NIC configuration does not prove all
four rails carry payload. Report descriptive median, nearest-rank p95 and max;
thirty samples do not establish stable p99, saturation throughput, a hardware
advantage or S2/S5 performance qualification. Raw per-request timing, metrics,
logs, hashes, workload order and failure/cleanup evidence stay outside Git.

## Official-engine cross-host serving restores

Use [the serving restore driver](../benches/README.md#first-and-repeated-cross-host-serving-restores)
for two dedicated official-engine replicas, each connected to its local
Manager. Freeze matching checkpoint assets, dtype, page layout and computation
identity. The topology controller records the installed wheel, engine and native
library hashes, actual mapped TENT libraries, payload transport policy and
physical NIC counters. Source DRAM and source io_uring SSD are separate cells;
SSD requires source DRAM eviction and an exact completed io_uring SSD read-byte delta.

Every request compares complete generated text and native input/output counts
with a fresh source control, requires exact remote/H2D bytes and acknowledged
release, and drains the exported ownership gauges. Clear consumer HBM and DRAM
between repetitions while preserving external identity. Keep reset, eviction,
inventory fences and startup outside request timing, retaining first requests
and failed responses. Client TTFT includes HTTP transport and ends at the first
nonempty streamed text; it is not isolated RDMA or DMA latency. Correlated
repetitions are descriptive evidence, not a formal matched performance campaign.

Test consumer Manager restart with the source still alive: observe the old member
disappear, require an advanced epoch and new incarnation, reinstall complete
inventory coverage and repeat remote recovery. Separately stop the source and
restart it without surviving payload. After removing consumer HBM/DRAM copies,
require zero source inventory, zero remote/H2D byte deltas and matching output
from recomputation. This controlled empty-source restart is distinct from crash
reclamation, host failure or etcd HA.

Persist literal service-stop results before validating them. Require a running
engine before requested idle SIGTERM, leader exit zero, complete reap, empty
owned process groups and no harness forced cleanup. SGLang's released normal
path also records SIGTERM, zero remaining requests and child-only native process
termination; retain child exit statuses separately. A native leader exit -9 is
not evidence of healthy shutdown. Manager exit zero, source/receiver drain and
final ports, sockets, GPU memory and installed-file verification remain separate
checks. A PyTorch CUDA IPC producer warning must stay in the handoff; final GPU
zero and successful outputs do not prove the producer/importer lifetime contract.

The separately frozen 2026-10-10 remediation cohort has scoped independent
acceptance for both official releases on two physical H20 hosts: dense Qwen3-8B,
BF16, TP=1/PP=1 eager, serial 513/1,025-token prompts and eight output tokens.
Each of the four engine/medium cells passes 20 profile restores, a consumer
restart restore and an empty-source recomputation. Remote/H2D bytes agree, SSD
read bytes are exact, selected physical NIC counters advance and the exported
ownership gauges drain. Inputs are unchanged and final services, GPU memory,
ports and sockets are clean without harness forced cleanup. The original
controller's SGLang stop-assertion failure remains an immutable invalid cohort.

The accepted payload path is source DRAM, or source SSD through bounded
O_DIRECT/io_uring staging, then TENT RDMA to consumer DRAM and a local GPU copy.
The source SSD filesystem is ext4 on an exposed NVMe partition; native GDS is not
used. A single-member etcd fixture supplies membership, so this is not etcd HA.
Eight SGLang CUDA IPC producer warnings remain recorded. Output and native token
count equality do not establish raw GPU page-value equality, GPU-buffer network
RDMA, crash reclamation or stable tail performance. The independently reviewed
inputs, original invalid run, fresh cohort and full storage archives are at
`/root/orbitkv-artifacts/s5-rdma-serving-20261010/`; full S2/S5 remain Partial.
