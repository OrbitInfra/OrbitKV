# GPU storage

OrbitKV defaults to automatic SSD backend selection when SSD caching is configured.
Complete state groups can be written
from registered engine GPU pages to SSD, and selected SSD state can be restored
to those pages. The cuFile routes use bounded, registered GPU staging in both directions.
vLLM and SGLang share this Rust implementation and their existing cache API.
Normal deployment only configures the SSD path and budget; leave
`--ssd-backend` unset. Its default `auto` tries native cuFile and uses io_uring
when unavailable. cuFile is a dynamically loaded Manager library, not an
additional service or an inference-engine option.

Calling cuFile does **not** prove native GPUDirect Storage: its compatibility
mode performs host staging internally. Native-mode probes in this H20 container
have failed driver initialization (5001) or file registration (5027); automatic
selection falls back to io_uring. An A100 native-only initialization probe on
2026-10-07 also returned driver error 5001 on its `/tmp` ext4 NVMe mount; `auto`
selected io_uring and the Manager exited normally. Its surviving capability
record is retained in the SSD host-batching archive, separate from native I/O
qualification. Native GDS throughput remains unqualified.

## Automatic selection

The Manager owns this decision; engine adapters use the same cache API.

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Try cuFile on ext4/XFS with sufficient alignment capacity; disable both allow/force compatibility through NVIDIA's parameter API before driver initialization. Missing API/library, driver, mount or file-registration support selects io_uring. |
| `uring` | Use host staging; never load cuFile. Useful as a reproducible control. |
| `cufile` | Require cuFile initialization and follow NVIDIA configuration, including explicitly enabled CPU compatibility for functional development. Operation failures remain errors. |

Auto selection is currently cache-wide: every configured shard must register;
otherwise the whole SSD cache uses io_uring. It requires cuFile's parameter API
to enforce the native-only initialization policy. An already initialized
compatibility-capable driver is not accepted as native evidence. Other filesystem
types need separate qualification and an explicit `cufile` selection.

On a GPU storage buffer or I/O failure in `auto`, the Manager stops admitting
new cuFile work. With no read-route override, subsequent operations use io_uring
until restart; an explicit cuFile read route does not silently switch routes.
When several engine instances share one Manager, one pool owns each CUDA
device's persistent cuFile/codec staging until its lanes drain. Another
instance's automatically selected read may use io_uring over the same extent;
an explicit cuFile read fails instead so qualification cannot pass on fallback.
Submitted work retains its file/extent/page ownership and completes or reports
its error; changing the backend does not revoke DMA or silently replay an
already submitted restore. The failing operation retains normal error semantics.
Logs identify the selection and fallback reason; `orbitkv_ssd_backend_fallbacks_total`
counts the transition. Per-mount/per-device isolation, retries after a cooldown,
and online latency-based selection remain future work.

This is capability and failure adaptation. Successful initialization does not
establish that cuFile beats io_uring for a workload, or replace native-path
statistics. An SSD path is required to enable disk caching. Capacity defaults
to `512gb`; set an explicit budget that fits the storage filesystem.

## Independent demand read routes

Immutable SSD extent leasing is now separate from GPU-route eligibility.
io_uring and cuFile are independently executable routes over the same stored
representation and source generation. The store keeps its io_uring engine even
when cuFile is healthy. `--ssd-backend auto|uring|cufile` controls capability
initialization and existing write behavior; it is not a measured route ranking.

For controlled demand-read comparisons, use `--ssd-read-path uring` or
`--ssd-read-path cufile`. With a cuFile-capable backend, the former still uses
io_uring and pinned DRAM; the latter uses registered GPU staging. The override
does not change write policy or prepare/warmup, which still targets DRAM.
Omitting it preserves existing source priority and capability-based behavior.
If the explicit SSD route is unavailable, it is not replaced by a host-prefetch
route; a compatible peer source can still satisfy ordinary remote recovery.

The explicit io_uring route carries the same extent lease through the bounded
host-reader queue. A separate SSD host-restore lane waits for these reads, then
uses existing copy/codec completion before releasing engine destinations. It
deduplicates repeated layer references and submits at most 16 unique state keys
per existing SSD-store reader batch; same-key physical generations stay in
separate batches. All batches drain and all requested results are validated
before GPU sources change or an error returns. This reduces submission/completion
work, without merging physical I/O or adding earlier readiness. It does not block
the ordinary DRAM worker or reacquire a potentially different generation by key.
Both routes share the SSD store and extent ownership;
separate queues do not imply separate hardware bandwidth.

Opt-in cost observations compare enqueue-to-GPU-terminal route totals in shadow,
including host allocation/read/H2D or GPU staging/scatter/decode. They do not
select execution or probe an alternative. Dynamic selection and common
device-pressure scheduling remain future work. See the
[candidate route contract](state-planning.md#candidate-routes-and-eligibility).

## File capacity

After every cuFile shard registers, the Manager reserves its configured physical
space with Linux `fallocate` before starting workers. This applies to both
`auto` selecting cuFile and explicit `cufile`. Insufficient space, quota errors
or unsupported allocation fail startup with the shard and requested size;
already reserved startup files are truncated to release space, with cleanup
errors logged. Capacity failures do not silently select a sparse io_uring cache.
Choose a capacity that fits the available storage. Explicit io_uring and
capability fallback retain logical file sizing without upfront reservation.

Preallocation does not prove native GDS. First writes into unwritten extents
can still need filesystem metadata work; qualify first writes and overwrites
separately with compatibility disabled and cuFile path statistics. See NVIDIA's
[write-allocation guidance](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/#block-allocation-for-writes).

## Data flow and ownership

| Operation | Path |
| --- | --- |
| DRAM hit | Pinned DRAM → engine GPU pages |
| SSD demand with cuFile read route | SSD → registered GPU staging → engine GPU pages |
| SSD demand with explicit io_uring read route | Same leased SSD generation → pinned DRAM → engine GPU pages |
| Speculative prepare/warmup | SSD → pinned DRAM; GPU restore after consumption |
| Complete state group in one Publish | Engine GPU → registered GPU staging → SSD; also retain a hot DRAM copy |
| Fragmented/multi-writer Publish | Engine GPU → pinned DRAM; seal complete group → io_uring → SSD |
| Shared-cache remote recovery | Mooncake → pinned DRAM → engine GPU pages |

Candidate discovery remains metadata-only. A selected demand using cuFile or an
explicit SSD read route acquires an extent lease before reading bytes. Default
io_uring queries retain their existing host-prefetch behavior. The compiled
recovery requirements still select the necessary windows and checkpoints. The
lease pins the source against ring overwrite until all query/GPU interests are
released. Cancellation, expiry and session teardown release unconsumed interests;
submitted restores keep their interests until completion.

Rust validates source segment sizes, slot offsets and GPU destinations before
issuing I/O. On the cuFile route, adjacent ranges from different source leases in the same file are
merged into reads of at most **4 MiB** with **4 KiB** aligned boundaries. Different
files and unrequested aligned gaps stay separate. The worker retains the entire
restore task, keeping every source lease alive until GPU completion; it never
uses one representative lease to protect other extents. Existing 512-byte storage segments can cause edge
overread; only actual component bytes are scattered into engine pages. Large
recurrent checkpoints are split across the bounded staging buffer.

Each instance/device lazily creates one SSD worker with **two registered 4 MiB
slots**, each with its own CUDA stream and completion event. It submits
`cuFileReadAsync` / `cuFileWriteAsync` and polls completion without waiting for a
whole storage job. At most two batches are in flight, including at most one write,
leaving a slot for demand reads. Jobs rotate after each batch; after four read
submissions a waiting write gets the next available opportunity. This bounds
starvation by submissions, not elapsed time, and does not preempt DMA. Ordinary
DRAM transfers retain their separate workers; hardware bandwidth remains shared.
Reserve the total **8 MiB per instance/device** in the engine's memory budget;
it is outside `--pool-size`. Slots are reused until the worker drains. Query
reservations still charge logical source bytes under the existing query budget,
even when those bytes remain on SSD. Queued reads use those existing byte
budgets; there is no additional read-job count limit.

At most **eight GPU write jobs** can be queued or active per instance/device.
Admission exhaustion rolls back unsubmitted GPU extents and uses ordinary D2H
publication with bounded io_uring writeback. A permit stays owned until the
GPU job terminates. Mixed-load H2D and hot-copy D2H preparation enqueue on the
worker's stream and retain a per-job CUDA event. SSD submission and completion
polling continue while those copies run. Each job keeps its pages until both
its host-copy event and storage work complete; a later job's copy does not extend
an earlier job's completion fence.

Before GPU writeback, Rust reserves an unpublished, 4 KiB aligned SSD extent.
Each chunk gathers only valid source bytes and zeroes padding, then completes
asynchronous cuFile I/O before reusing staging. The complete object becomes queryable only
after all chunks finish. Failed or short writes abort the reservation; other
completed objects remain valid. Publish retains the source pages until D2H and
SSD work finish, including after a caller wait deadline. This extends page hold
time versus asynchronous DRAM writeback.

Direct writes require all slots of a storage group in the same Publish.
Fragmented layers and multi-writer TP/PP groups assemble in DRAM and keep
io_uring writeback. Both forms can subsequently restore through cuFile.
The existing `all`/`reuse` admission policy still applies. Optional
[GPU storage encoding](storage-formats.md) also supports direct cuFile payloads.
Complete groups that fit one codec batch write encoded GPU bytes through a
reusable 4 MiB registered slot. An independent bounded writeback lane retains
the codec arena until disk completion; demand reads and host publication can
continue. Fragmented groups, CPU encoding and oversized batches use io_uring
writeback. A compressed pinned-DRAM copy remains available for hot reuse.

Encoded demand reads use the two existing cuFile slots to assemble a bounded
GPU input. A decoder validates CRC32 on GPU before decompressing into engine
pages. Its retained input plus codec scratch share the configured codec budget.
The I/O queue continues polling other requests while the decoder runs. Mixed
raw/encoded prefixes use the same extent leases; a failed encoded generation
cannot be reused or overwritten while another lease still owns it. Small codec
budgets that cannot stage a segment select the host reader. Speculative warming
continues through DRAM. Compression remains disabled by default.

Each slot retains address-stable size/offset/result storage and a registered
file reference until its completion event. Submission success alone never
accepts data: Rust checks the completed byte count before scatter or publication.
A second event proves read scatter has completed before reusing staging or
reporting completion. Short/failed reads fail the restore. Partial submission
errors drain their stream; other submitted batches retain ownership until they
complete. A closed completion consumer stops unsubmitted work, then drains
submitted batches. Canceling a query after restore submission does not revoke
that restore's ownership or completion. Unregister waits for every worker and
deregisters streams/buffers before releasing engine mappings.

A ring reservation that would overwrite an
active read or write tries the other shards, then drops the write if none has
space. It does not block the directory. SSD remains a best-effort cache recreated
at Manager startup.

## Copy avoidance and remote storage

Optimize request completion time and retained resources alongside copy count.
The native cuFile path avoids host staging for SSD demand reads. OrbitKV still
uses a registered GPU staging buffer and D2D scatter/gather to handle engine
layouts and alignment. Direct I/O into registered engine pages is a separate,
unimplemented optimization: it needs compatible offsets/sizes, reusable memory
registration, engine-page lifetime protection and a comparison against coalesced
staging. Many small direct operations can cost more than a larger staged read.

| Source and destination | Relevant mechanism | OrbitKV status |
| --- | --- | --- |
| Local SSD ↔ GPU | GDS/cuFile | Bounded GPU staging implemented; direct engine-page I/O pending |
| Remote shared-cache memory ↔ GPU | GPUDirect RDMA through a transport such as Mooncake TE | Manager recovery lands in requester DRAM before GPU restore; direct engine destinations are pending |
| Prefill GPU → decode GPU | Mooncake TE memory transport; GPUDirect RDMA on supported hardware | Experimental vLLM P/D connector writes engine GPU pages; GPU and cross-host qualification pending |
| Remote storage filesystem or mounted NVMe-oF ↔ GPU | GDS with a supported filesystem/network/storage stack | Separate deployment and qualification; not enabled by the current peer-memory path |

The pinned Mooncake TE includes an NVMe-oF transport that registers buffers with
cuFile and submits cuFile batch I/O. This is distinct from its RDMA memory
transport. The OrbitKV Manager registers its pinned DRAM pool and fetches remote
state into that pool; using Mooncake does not implicitly enable its storage
transport. Remote GDS requires an accessible storage namespace and a qualified
filesystem, network and GPU path. See the pinned
[NVMe-oF implementation](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp).

Direct placement in engine HBM can remove an extra GPU staging allocation and
D2D scatter when source and destination layouts match. The KV state still
occupies HBM, and destination pages must remain reserved until completion,
including after cancellation. CUDA IPC only shares access to an allocation;
sharing a handle does not itself move data. The
[native P/D path](pd.md) uses each official engine's transport and ownership;
its direct GPU transfer does not qualify shared-cache direct placement. Direct placement in that shared-cache path is a follow-up to
qualify independently from the host-staged peer SSD route.

GDS does not schedule requests or select reusable model state. Recovery planning
selects the required ranges; the storage/transfer owners choose a viable data
path and keep leases until it completes. A future request-level selector should
measure queue delay, registration cost, I/O count, bytes/alignment amplification,
GPU scatter cost, source-page hold time and deadline. Use bounded concurrency,
read priority and hysteresis before changing paths based on latency. Those
policies require matched native hardware measurements; they are not inferred
from successful driver loading.

See [NVIDIA's buffering guidance](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html)
and [Mooncake's transport design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md).

## Review against LMCache

Reviewed on 2026-09-24 against LMCache **v0.5.5**, commit
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`. Its
[legacy GDS backend documentation](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/kv_cache/storage_backends/gds.rst)
marks in-process mode deprecated and recommends MP mode. The MP implementation
is the relevant reference for the Manager architecture.

LMCache MP also registers GPU staging; GDS does not imply direct I/O into every
engine page. Its
[GDS context](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/gds_context.py)
keeps file/buffer registrations, splits transfers at registered-region boundaries,
and retains asynchronous submissions behind per-stream GPU events. Its
[cuFile binding](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/_cufile_async.py)
uses `cuFileStreamRegister`, `cuFileReadAsync` and `cuFileWriteAsync`.

| Concern | LMCache MP reference | Current OrbitKV |
| --- | --- | --- |
| GPU registration | Reusable staging, registered in regions of at most 16 MiB | Two reusable registered 4 MiB slots per instance/device |
| File allocation | Preallocates its slab with `posix_fallocate` | Native `fallocate` reserves all GPU-storage shards before admission; allocation errors fail startup and release partial reservations |
| Submission/completion | Stream-ordered asynchronous I/O; event-scoped submission lifetime | Rust stream-ordered asynchronous I/O with stable arguments/results, completion byte checks and retained leases |
| Batching | GPU context provides four chunk slots | Two slots, at most one write; adjacent ranges merge by file across source leases without reading unrequested aligned gaps |
| Tier policy | GDS L1 replaces pinned-DRAM L1 in that configuration | Complete-group GPU writeback also creates a DRAM copy; Publish waits for SSD completion |

The four-slot geometry comes from LMCache's
[CUDA cache context](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/platform/cuda/cache_context.py);
its [MP configuration](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/configuration.rst)
describes the L1 medium switch. Neither its slot count nor OrbitKV's 8 MiB size
establishes an optimum for another model or storage device.

The next implementation gates, in order, are:

1. **Native measurements.** Qualify first writes, overwrites and mixed demand on
   supported hardware. Compare the two-slot stream path with batch I/O for small
   scattered ranges; tune slot size/count and polling from request latency, GPU
   budgets and CPU use rather than copying an upstream default.
2. **Placement and source-page hold time.** Measure the cost of producing both
   DRAM and SSD copies. Evaluate selective hot-DRAM admission and releasing engine
   pages after their final copy into owned staging; the staging and SSD reservation
   must survive until disk completion. Large objects still need bounded chunking.
   Publish visibility must never precede complete successful writes.

Physical allocation, cross-lease read coalescing, bounded asynchronous submission
and read/write scheduling are implemented. They do not establish a measured
native-GDS performance advantage. Benchmark both
engines with matched DRAM/SSD capacities, HBM budgets, working sets and native-I/O
statistics. Shared-cache direct engine-page I/O remains a separate later step.

## Enable and qualify

Install NVIDIA's GDS user-space library on the Manager and provide a supported
storage mount and compatible host driver/kernel configuration. Ordinary io_uring
deployments do not load cuFile. Explicit `cufile` selection fails startup on
library/file registration errors; GPU buffer registration errors fail the
operation. Default `auto` instead logs capability fallback and disables new
cuFile admission after an operation failure as described above. GPU-storage
capacity reservation errors fail startup in both modes.

For native qualification, explicitly disallow compatibility mode:

```bash
CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false \
orbitkv-cache-manager --pool-size 8gb \
  --ssd-cache-path /mnt/nvme/orbitkv/cache.bin \
  --ssd-cache-capacity 100gb --ssd-backend cufile
```

Check the target mount with NVIDIA's `gdscheck.py -p` and `gdsio`, then inspect cuFile
statistics for actual reads. Library installation, file registration or an
OrbitKV cuFile byte counter alone is not native-path evidence. Local NVMe GDS
does not need a network RDMA adapter. Containers require suitable host mounts,
devices and kernel support; newer NVMe PCI P2PDMA configurations can operate
without `nvidia_fs`.

Run the explicit GPU gate with no serving process active during Cargo builds:

```bash
mkdir -p /mnt/nvme/orbitkv-tests
TMPDIR=/mnt/nvme/orbitkv-tests \
CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false \
cargo test --release -p orbitkv-core --no-default-features \
  --features cuda-13,mooncake --test cufile -- --ignored --test-threads=1
```

For functional development, set both cuFile environment values to `true` and
choose a writable mount on which cuFile can register files. Compatibility mode
does not make every mount usable: this container's `/tmp` ext4 mount fails file
registration, while `/workspace` overlay works in forced compatibility mode.
Set `TMPDIR` accordingly; for pytest, set `--basetemp` on that same mount. Record
this as **cuFile compatibility-mode correctness**, not native GDS performance.
The gate covers split/page-first layouts, unaligned components, checkpoints
larger than staging, ring pinning, cross-shard progress, cancellation and
short-read failure recovery. It also covers fragmented publication sealing
and restoring its io_uring-written payload through cuFile.
The Manager process gate additionally checks actual cuFile call counts, exact
selected bytes and lease retention during canceled coalesced reads, including
separate files and unrequested gaps. Rust file tests verify physical allocation
and rollback after a later shard fails.

The development serving gates use **cuFile 1.16.1 from CUDA 13.1**. Their Torch
2.13/cu130 environments preload cuFile 1.15.1.6, which fails to register this
overlay mount even in compatibility mode. The commands below explicitly preload
the verified library. Qualify the library and mount together on another host;
do not infer native support from a successful compatibility run.

Both serving gates accept `--ssd-backend cufile` and require positive cuFile
reads after writes drain, DRAM eviction and an engine restart, plus positive
cuFile write bytes:

```bash
cd python
ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
LD_PRELOAD=/usr/local/cuda/lib64/libcufile.so.0 \
CUFILE_ALLOW_COMPAT_MODE=true CUFILE_FORCE_COMPAT_MODE=true \
../.venv/vllm-release/bin/python -m pytest -m e2e \
  tests/e2e/test_vllm_e2e_correctness.py --model /workspace/models/qwen3-8b \
  --vllm-cache-tier ssd --ssd-backend cufile \
  --max-model-len 4096 --orbitkv-pool-size 1gb \
  --basetemp=/var/tmp/orbitkv-bench/cufile-vllm

ORBITKV_CACHE_MANAGER_BINARY=../target/release/orbitkv-cache-manager-py \
LD_PRELOAD=/usr/local/cuda/lib64/libcufile.so.0 \
CUFILE_ALLOW_COMPAT_MODE=true CUFILE_FORCE_COMPAT_MODE=true \
../.venv/sglang-release/bin/python -m pytest -m e2e \
  tests/e2e/test_sglang_direct_e2e.py -k ssd --model /workspace/models/qwen3-8b \
  --ssd-backend cufile \
  --basetemp=/var/tmp/orbitkv-bench/cufile-sglang
```

Use a separate `--basetemp` for each run and run GPU gates sequentially. Change
both cuFile environment values to `false` for native-path qualification.

## Recorded functional qualification

Final checks on 2026-09-24 used one H20, cuFile 1.16.1, vLLM 0.29.0 and
SGLang 0.5.20. Explicit cuFile runs **forced CPU compatibility mode** on this
container's overlay mount. Baseline `auto` runs selected io_uring after cuFile
file registration failed. These results establish
correctness, not native GDS throughput.

| Gate | Final result |
| --- | --- |
| Rust GPU layouts, encoded payloads, checkpoints, pinning, failure recovery and auto fallback | 8 passed |
| Manager process faults, concurrent GPU I/O, encoded storage and cancellation ownership | 51 passed |
| vLLM / Qwen3-8B / SSD | 6 passed; 1 recurrent-only check skipped, with both explicit `cufile` and default `auto` |
| SGLang / Qwen3-8B / SSD | Passed with both explicit `cufile` and default `auto` |
| vLLM / Qwen3.8-27B-FP8 / SSD | 7 passed |
| SGLang / Qwen3.8-27B-FP8 / SSD | Passed |

The final Manager gate covers SSD reads during a held write completion,
GPU write admission saturation with host fallback, cancellation/unregister
with both read slots occupied, held host-copy completion, corrupted encoded
objects, GPU/CPU SIMD FP8 conversion against Torch, and ANS/TurboQuant recovery
after DRAM eviction and re-registration. Each checks actual GPU bytes and final
resource release. The exact-storage serving results are retained from the
[GDS baseline](https://github.com/feichai0017/orbitkv/pull/176).

The serving results above use exact storage. GPU codec capacity and model-quality
qualification are recorded separately in [storage formats](storage-formats.md#qualification).

Both models' explicit cuFile checks require cuFile writes and new reads after
DRAM eviction and engine restart. Default-auto checks require new SSD reads
through the selected backend after the same eviction and restart sequence.
Qwen3.8 also exercises full-attention and GDN
conv/recurrent state in the same request. Reproduce it with the
[Qwen3.8 engine settings](models.md#qwen38-on-h20) and `--ssd-backend cufile`.

The Manager gate verifies four adjacent 64 KiB pages restore with one 256 KiB
cuFile read. Splitting them across two files requires two reads; selecting only
alternating pages reads 128 KiB in two operations. It checks actual GPU bytes,
physical file allocation, per-operation metrics and cancellation ownership.
These are transfer-shape and correctness results, not native throughput claims.

## Bare-metal acceptance script

[`benches.storage.gds`](../benches/storage/gds.py) runs the native qualification sequence. It
rejects containers, verifies the selected ext4/XFS mount is NVMe-backed,
disables CPU compatibility, and enables per-process cuFile statistics through
a private config overlay. It does not change system configuration. A private
subdirectory under `--ssd-dir` holds test data and results; it never opens a raw
disk or overwrites an existing cache file.

Build the ordinary Manager, a separate `test-hooks` Manager, the matching Python
extension in **both** engine environments, and the Rust `cufile` test executable
before starting. Use the CUDA feature matching the machine. Obtain the test
executable path from `cargo test --release -p orbitkv-core --no-default-features
--features cuda-13,mooncake --test cufile --no-run`. Copy each Manager before
building the other variant. The script accepts these prebuilt artifacts and
never invokes Cargo while services are running.

```bash
.venv/vllm-release/bin/python -m benches.storage.gds \
  --ssd-dir /mnt/nvme/qualification --model /models/Qwen3-8B \
  --manager /opt/orbitkv/manager \
  --fault-manager /opt/orbitkv/manager-faults \
  --core-test /opt/orbitkv/cufile-test \
  --gds-tools /usr/local/cuda/gds/tools \
  --cufile-library /usr/local/cuda/lib64/libcufile.so.0
```

Set `--gds-tools` to the directory containing NVIDIA's `gdscheck.py`, `gdsio` and
`gds_stats`. The sequence covers:

1. GPU/storage topology, `gdscheck`, and native `gdsio` writes and reads.
2. Exact GPU-byte recovery, pinning, fragmented saves and large checkpoints.
3. Real-process faults including stalled/failed cuFile writes and Manager death.
4. vLLM and SGLang Qwen3 correctness across engine restart, DRAM and SSD recovery,
   for each requested storage codec.
5. Matched DRAM and io_uring/auto/cuFile SSD workloads across codecs, including
   serial and sustained mixed-prefix requests with equal seeds and capacities.

The default matrix includes `none`, `ans`, `fp8`, `turboquant-4` and
`turboquant-3`. Use `--storage-codecs none ans` for lossless-only qualification.
Strict model-output failures remain failures even when the script continues to
collect performance evidence; a smaller payload does not waive the quality gate.

Every auto/cuFile benchmark captures `gds_stats -p <manager-pid> -l 3` while the
Manager is alive. Success requires positive reads **and** writes and zero
POSIX, unaligned, sparse/inline or failed operations. Missing/unknown statistics
fail the gate, as does any automatic backend fallback. Ordinary io_uring work is not counted as native I/O. The final
`qualification.json` records all stage exits; per-run summaries retain TTFT,
throughput, storage counters and whole-workload Manager CPU usage. Linux process
I/O accounting is recorded separately from cuFile GPU I/O bytes. Raw logs remain
in this private run directory.

This script has source-level checks in `benches/tests/storage/test_gds.py`. Its full
native sequence is pending an accessible bare-metal GPU/NVMe host; container
functional passes do not satisfy it.

## Metrics and next gates

- `orbitkv_ssd_cufile_read_bytes_total` / `orbitkv_ssd_cufile_write_bytes_total`:
  physical bytes, including alignment.
- `orbitkv_ssd_cufile_read_seconds` / `orbitkv_ssd_cufile_write_seconds`: submission-to-completion latency, including GPU gather/scatter and polling; excludes time waiting for a slot and is not pure device I/O time.
- `orbitkv_ssd_cufile_inflight_batches`: slots occupied through I/O and scatter completion; at most two per active instance/device.
- `orbitkv_ssd_gpu_write_fallbacks_total`: write jobs sent through host publication when eight GPU writes are already admitted.
- `orbitkv_ssd_cufile_read_failures_total` / `orbitkv_ssd_cufile_write_failures_total`: failed or short I/O.
- `orbitkv_ssd_backend_fallbacks_total`: transitions from automatic cuFile admission to io_uring.
- `orbitkv_ssd_read_pinned_bytes`: SSD bytes held by restore leases.
- `orbitkv_ssd_gpu_staging_bytes`: allocated registered GPU staging memory.
- `orbitkv_ssd_pinned_write_skips_total`: reservations rejected to protect reads
  or in-flight writes.

For cuFile or explicit io_uring demand routes, the timeline's `source_ready`
event means the source lease is ready. Disk reading occurs during GPU restoration, not query
preparation; compare end-to-end TTFT and storage counters rather than treating
preparation time alone as an improvement.

With `ORBITKV_COST_OBSERVATIONS=1`, the `ssd_uring_restore` and
`ssd_cufile_restore` cost paths measure complete restore totals. Child
`local_ssd_host_ready`, `ssd_read` and cuFile/copy/codec observations overlap
these totals;
do not add them or count their physical bytes twice. Observations remain off by
default while the earlier SGLang ANS overhead gate is open. The route changes have
[separate validation](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#ssd-sourcepath-separation-final-evidence);
the earlier serving and observation matrices are not reruns of this implementation.

Before claiming a performance improvement, compare io_uring and **verified native** GDS on the same
NVMe mount, model and working set larger than DRAM. Measure TTFT, throughput, CPU
use, physical/read amplification, staging budgets and lease drain. Keep
speculative preparation off in the first comparison. Automatic cost-based path
selection and tuning concurrency/placement remain follow-up work. Complete-group direct writeback is
implemented; multi-writer GPU assembly remains separate work.

## Design references

- [FlexKV GDS implementation](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/csrc/gds/gds_manager.cpp): file registration and GPU staging with layout transforms.
- [LMCache GDS L1](https://docs.lmcache.ai/mp/configuration.html#gds-l1-tier): explicit storage-medium selection. OrbitKV retains DRAM for hot data and speculative preparation.
- [NVIDIA installation and verification](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/) and [benchmarking guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/): native configuration and per-process path evidence.
- [NVIDIA cuFile API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) and [best practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/): registration lifetime, completion, alignment and compatibility mode.
