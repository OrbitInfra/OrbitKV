# GPU storage

All SSD byte movement belongs to the OrbiTE engine. OrbitKV owns
placement, extents, index publication, DRAM policy, layout descriptors and codec
compute. The native OrbitKV io_uring/cuFile fallback and
`--ssd-io-engine` option are removed. See [SSD file transfers](transport.md#ssd-file-transfers)
for the contracts, bounds and exact GPU gate commands.

GPU staging and native GDS are implemented but unqualified.
[Historical native SSD measurements](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/gds.md) apply only to that implementation.
CPU file bytes and ownership controls cannot qualify CUDA or GDS.

## Selection and capacity

Normal deployment sets the SSD path and capacity. Both inference adapters use
the same cache API; no separate cuFile service is needed.

| Mode | Selected engine behavior |
| --- | --- |
| `auto` (default) | Direct layout vectors plus engine-owned CUDA staging/io_uring; host materialization remains available for DRAM policy and unsupported layout shapes. |
| `uring` | Host materialization through engine io_uring. |
| `cufile` | Native-only GDS. Requires `orbite-gds` and the matching runtime cuFile SDK. Missing support is an error. |

Native GDS requires a matching cuFile runtime, supported filesystem and installed
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
| SSD GPU demand | Engine vectored file read → inference pages or bounded codec inputs |
| SSD host demand or prepare | Engine file read → leased pinned DRAM; restore follows cache policy |
| Complete GPU publication | Layout/encode → engine vectored file write; retain a hot DRAM copy |
| Fragmented publication | Seal complete state in DRAM → engine file write |
| Peer GPU restore | Engine remote READ directly into GPU pages or bounded codec inputs |

OrbiTE consumes one vectored request per extent. Gaps and 4 KiB tails are
padding elements, written as zeros or discarded on READ by the engine. The old
GPU layout slots and zero buffers are deleted. Imported tensors and codec inputs
retain persistent registrations, while encoded writes retain their actual codec
arena. Unknown drain keeps caller completion/admission and fences the worker.
Local GPU↔DRAM also uses OrbiTE; see [local copies](transport.md#local-gpudram-copies).

OrbiTE rejects explicit GDS padding without fallback. Its `gds_enabled` stays
false unless `cufile` is configured, even when GDS is compiled.

Each store admits 32 file operations, including retained work, with at most
64 MiB and 1,024 elements including padding per operation. Whole host extents
preserve 512-byte alignment; GPU extents use 4 KiB alignment. A failed or short
READ never becomes a cache hit. Only a complete successful WRITE publishes
metadata. Unknown drain retains backing, extent leases and admission until
process exit; failed cleanup is never retried.

Layout vectors preserve each source generation and express unrequested gaps as
padding. Fan-out of the same file bytes requires separate vectors. GPU publication
remains bounded to eight queued/active jobs and one active write per worker;
there are two in-flight file receipts. Saturation uses ordinary host publication
for unsubmitted work.

## Qualification

Run the two explicit `ssd_engine` GPU byte gates from
[the transport guide](transport.md#ssd-file-transfers) with
the `orbite`/`orbite-gds` build features and a test-owned directory on the
qualification SSD. The broader `cufile` integration
tests cover layouts, codec inputs, cancellation and extent lifetime under native
GDS; compatibility-mode qualification of the removed implementation does not
qualify OrbiTE. Build the artifacts before starting the serving processes.

Preserve native-route evidence from the same installed run: cuFile logs or
nvidia-fs/PCI P2PDMA counters, successful byte checks, device topology, budgets,
first-write/overwrite behavior and drain results. Compare with engine io_uring
using equal layouts and cache policy before claiming a performance advantage.
GPU, RDMA and GDS profiles remain unqualified without that installed evidence.

## Bare-metal acceptance script

[`benches.storage.gds`](https://github.com/OrbitInfra/OrbitKV/blob/main/benches/storage/gds.py) runs the native qualification sequence. It
rejects containers, verifies the selected ext4/XFS mount is NVMe-backed,
disables CPU compatibility, and enables per-process cuFile statistics through
a private config overlay. It does not change system configuration. A private
subdirectory under `--ssd-dir` holds test data and results; it never opens a raw
disk or overwrites an existing cache file.

Build the ordinary Manager, a separate `test-hooks` Manager, the matching Python
extension in **both** engine environments, and the Rust `cufile` test executable
before starting. Use the CUDA feature matching the machine. Obtain the test
executable path from `cargo test --release -p orbitkv-core --no-default-features
--features cuda-13,orbite-gds --test cufile --no-run`. Copy each Manager before
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
- `orbitkv_ssd_cufile_read_seconds` / `orbitkv_ssd_cufile_write_seconds`: submission-to-completion latency, including engine staging and polling; excludes admission waiting and is not pure device I/O time.
- `orbitkv_ssd_cufile_inflight_batches`: file receipts held through engine completion; at most two per active instance/device.
- `orbitkv_ssd_gpu_write_fallbacks_total`: write jobs sent through host publication when eight GPU writes are already admitted.
- `orbitkv_ssd_cufile_read_failures_total` / `orbitkv_ssd_cufile_write_failures_total`: failed or short I/O.
- `orbitkv_ssd_backend_fallbacks_total`: transitions from automatic cuFile admission to io_uring.
- `orbitkv_ssd_read_pinned_bytes`: SSD bytes held by restore leases.
- `orbitkv_ssd_pinned_write_skips_total`: reservations rejected to protect reads
  or in-flight writes.

For cuFile or explicit io_uring demand routes, the timeline's `source_ready`
event means the source lease is ready. Disk reading occurs during GPU restoration, not query
preparation; compare end-to-end TTFT and storage counters rather than treating
preparation time alone as an improvement.

Before claiming a performance improvement, compare io_uring and **verified native** GDS on the same
NVMe mount, model and working set larger than DRAM. Measure TTFT, throughput, CPU
use, physical/read amplification, staging budgets and lease drain. Keep
speculative preparation off in the first comparison. Replica selection follows the [fixed policy](architecture.md#planning). Complete-group direct writeback is
implemented; multi-writer GPU assembly remains separate work.

## Design references

- [FlexKV GDS implementation](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/csrc/gds/gds_manager.cpp): file registration and GPU staging with layout transforms.
- [LMCache GDS L1](https://docs.lmcache.ai/mp/configuration.html#gds-l1-tier): explicit storage-medium selection. OrbitKV retains DRAM for hot data and speculative preparation.
- [NVIDIA installation and verification](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/) and [benchmarking guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/): native configuration and per-process path evidence.
- [NVIDIA cuFile API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) and [best practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/): registration lifetime, completion, alignment and compatibility mode.
