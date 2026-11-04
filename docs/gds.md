# GPU storage

All SSD byte movement belongs to the selected transfer engine. OrbitKV owns
placement, extents, index publication, DRAM policy, layout descriptors and codec
compute. TENT uses a separate file engine so file transports cannot weaken its
peer source-drain contract. The native OrbitKV io_uring/cuFile fallback and
`--ssd-io-engine` option are removed. See [SSD file transfers](transport.md#ssd-file-transfers)
for the contracts, bounds and exact GPU gate commands.

GPU staging and native GDS at this revision are implemented but unqualified.
Historical measurements below apply to their original native SSD implementation.
CPU file bytes and ownership controls cannot qualify CUDA or GDS.

## Selection and capacity

Normal deployment sets the SSD path and capacity. Both inference adapters use
the same cache API; no separate cuFile service is needed.

| Mode | Selected engine behavior |
| --- | --- |
| `auto` (default) | GPU layout slots plus engine-owned CUDA staging/io_uring; host materialization remains available for DRAM policy and unsupported layout shapes. |
| `uring` | Host materialization through engine io_uring. |
| `cufile` | Native-only GDS. OrbitTE requires `orbitte-gds`; TENT requires a CUDA runtime built with the cuFile SDK. Missing support is an error. |

The default wheel toolchain does not install the GDS SDK for TENT. Native GDS
requires the matching source-built runtime, supported filesystem and installed
route evidence. Neither library presence nor successful bytes prove native GDS.
A failed submitted operation is never replayed through another route.

GPU-capable shards reserve physical capacity with `fallocate` before serving;
allocation errors fail startup and roll back prior reservations. Host-only
shards use logical sizing. The default capacity is 512 GiB; select a budget that
fits the target filesystem. Every file is recreated at Manager startup.

## Demand routes and ownership

`--ssd-read-path uring` requests host materialization; `cufile` requests the GPU
layout route. These spellings select destinations, not a claim about native GDS.
The selected engine owns the actual file transport. Prepare/warmup still fills
the DRAM cache tier. Explicit unavailable GPU routes do not silently fall back;
ordinary recovery may still select a compatible peer source.

| Operation | Destination/path |
| --- | --- |
| SSD GPU demand | Engine file read → bounded GPU layout slot → scatter/decode into inference pages |
| SSD host demand or prepare | Engine file read → leased pinned DRAM; restore follows cache policy |
| Complete GPU publication | Gather/encode → engine file write; retain a hot DRAM copy |
| Fragmented publication | Seal complete state in DRAM → engine file write |
| Peer GPU restore | Engine remote READ directly into GPU pages or bounded codec inputs |

The temporary `storage/ssd/gpu.rs` layout owner has two 4 MiB slots per
instance/device. A2 replaces gather/scatter with engine vectored requests.
The engine owns storage staging, native registrations, queues and completions.
OrbitKV retains layout, codec and extent owners through those completions and
through the final CUDA scatter event. Local GPU↔DRAM migration is a separate step.

Each store admits 32 file operations, including retained work, with at most
64 MiB and 16,384 descriptors per operation. Host offsets preserve 512-byte
alignment; GPU layouts use 4 KiB alignment. A failed or short READ never becomes
a cache hit. Only a complete successful WRITE publishes metadata. Unknown drain
keeps the fd, actual backing, extent lease and admission charge until process
exit; failed cleanup is never retried. Deadline, cancellation and dropped
receivers do not authorize reuse.

Layout coalescing preserves each source generation and excludes unrequested
aligned gaps. GPU publication remains bounded to eight queued/active jobs and
one active write per worker. Saturation uses ordinary host publication for
unsubmitted work. Explicit GPU restore errors retain their error semantics.
Opt-in route costs remain observational; they do not change execution.

## Qualification

Run the two explicit `ssd_engine` GPU byte gates from
[the transport guide](transport.md#ssd-file-transfers) with
`ORBITKV_SSD_TEST_ENGINE=tent` or `orbitte`, the matching build features and a
test-owned directory on the qualification SSD. The broader `cufile` integration
tests cover layouts, codec inputs, cancellation and extent lifetime under native
GDS; compatibility-mode qualification of the removed implementation does not
qualify these engines. Do not run Cargo native builds while a serving process
has the staged libraries mapped.

Preserve native-route evidence from the same installed run: cuFile logs or
nvidia-fs/PCI P2PDMA counters, successful byte checks, device topology, budgets,
first-write/overwrite behavior and drain results. Compare with engine io_uring
using equal layouts and cache policy before claiming a performance advantage.
GPU, RDMA and GDS profiles remain unqualified without that installed evidence.

## Historical review against LMCache

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

| Concern | LMCache MP reference | OrbitKV at the historical review |
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
statistics. Direct peer GPU placement is implemented separately and remains hardware-unqualified.

## Historical native SSD qualification

The following commands and results describe the removed native SSD implementation
and remain as evidence references. Current engine-file gates are linked above;
compatibility-mode runs below do not apply to the native-only engine GDS profile.

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
[GDS baseline](https://github.com/OrbitInfra/OrbitKV/pull/176).

The serving results above use exact storage. GPU codec capacity and model-quality
qualification are recorded separately in [storage formats](storage-formats.md#qualification).

Both models' explicit cuFile checks require cuFile writes and new reads after
DRAM eviction and engine restart. Default-auto checks require new SSD reads
through the selected backend after the same eviction and restart sequence.
Qwen3.8 also exercises full-attention and GDN
conv/recurrent state in the same request. Reproduce it with the
[Qwen3.8 engine settings](https://github.com/OrbitInfra/OrbitKV/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md#qwen38-on-h20) and `--ssd-backend cufile`.

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
[separate validation](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#ssd-sourcepath-separation-final-evidence);
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
