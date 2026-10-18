# SSD and codec benchmarks

Run commands from the repository root. [Benchmark navigation](../README.md).

## CPU codec benchmark

`cpu_codec.rs` measures the production CPU E4M3FN encode/decode paths for BF16
and FP16 at 4 KiB, 256 KiB and 16 MiB of logical 16-bit data. It includes the
actual `codec/cpu.rs` inside its benchmark module to access private forced
backends; it adds no production API. `auto` uses the normal runtime dispatcher.
Unsupported AVX2/AVX-512 paths are recorded as `unsupported`, with no timing.

Run builds separately from live Managers or GPU benchmarks. Once the machine
is idle, use an available physical CPU for a repeatable affinity, and keep raw
CSV and logs outside tracked results:

```bash
cargo bench -p orbitkv-core --bench cpu_codec --no-default-features \
  --features cuda-13 --no-run
# Use the executable path printed by Cargo; choose a CPU in your allowed affinity.
taskset -c "$BENCH_CPU" /path/to/cpu_codec-executable --check
taskset -c "$BENCH_CPU" /path/to/cpu_codec-executable --seconds 0.25 --samples 3 \
  > /tmp/orbitkv-cpu-codec.csv 2> /tmp/orbitkv-cpu-codec.log
```

Before any timing, an independent nearest-value/ties-to-even oracle checks all
65,536 BF16/FP16 input patterns, all 256 FP8 decode codes, rejected nonfinite
and out-of-range inputs, SIMD tails, unaligned slices and every timed buffer
against each available backend. A mismatch fails the process. The deterministic
timed corpus samples finite in-range 16-bit patterns, including signed zero and
subnormals; it is synthetic data, not captured model activations.

Each CSV row contains the backend requested and selected, operation, sample,
actual iteration count, elapsed seconds, input/output bytes per iteration,
nanoseconds per iteration and logical GiB/s. The throughput numerator is the
uncompressed 16-bit byte count for both operations: decode reads half that
many encoded bytes. Tables and buffers are reused; allocation, oracle checks
and three warmup calls are outside timing. Backend order rotates between
samples. Keep the individual samples and report their spread; these measurements
describe CPU conversion only and do not measure GPU kernels, SSD transfer,
end-to-end cache recovery, TTFT or serving throughput.


## SSD restoration

Choose `--ssd-backend uring`, `auto`, or `cufile` for both engines. This benchmark
keeps `uring` as its reproducible default; the Manager itself defaults to `auto`.
The cuFile backend performs complete-group GPU writes and demand restores through bounded
GPU staging; fragmented writes and preparation still use DRAM. See
[GPU storage](../../docs/gds.md) for configuration, ownership and hardware limits.
`python -m benches.storage.gds --help` describes the complete bare-metal acceptance
sequence, including native-path statistics and matched pressure workloads.

Add `--ssd-gib 32` with `--backend orbitkv` for either engine. This creates a
dedicated cache file inside the empty result directory, verifies `O_DIRECT` on
the Manager's open descriptor, and records the filesystem and device inventory
in `storage.json`. The owned payload file is removed after services stop; all
measurement evidence remains. An overlay mount does not identify its physical
backing device, and these results must not be presented as bare-device bandwidth.
To keep raw output elsewhere, pass `--ssd-dir /mnt/nvme/orbitkv-bench` pointing
to an existing writable directory. Only a new private subdirectory there holds
the cache; it is removed after services stop. DRAM runs create no SSD payload.

The workload adds `after_host_eviction` after the original three phases. It
applies fresh GPU pressure, waits for observed SSD writes to become idle, and
evicts Manager DRAM through its HTTP administration endpoint while preserving
SSD copies. Eviction responses and counters are stored with each sample. These
preparation operations are outside request latency. This is a controlled tier
experiment, not a natural host-memory-pressure workload or a durability test.

SSD read bytes alone do not prove an inference cache hit. Reports retain GPU
load bytes, engine cache-hit evidence, and `ssd_reads_without_gpu_restore` so
a late prefetch followed by recomputation stays visible. `ssd_prefetch_p50_ms`
includes allocation, queueing, reads and reconstruction; `load_task_p50_ms`
includes H2D task construction and synchronization. Histogram sums describe
instrumented operations and are not an additive decomposition of client TTFT.
For cuFile restores, load duration also includes SSD reads and GPU scatter;
the explicitly selected io_uring host lane includes SSD reads and host
materialization before H2D or decoding. These timers overlap the SSD read timers.
Reports count both io_uring and cuFile read bytes; sustained/burst results retain
the individual counters. `manager-usage.json` records process CPU seconds and
Linux I/O accounting for the complete workload, including warmup and pressure.
Linux process I/O accounting is not a measurement of GPU DMA bytes.

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output /var/tmp/orbitkv-bench/qwen3-8b-ssd-vllm
```

For an independent read-path control, keep cuFile initialized and direct GPU
writes enabled while restoring through io_uring host staging:

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 16 --host-gib 1 --gpu-tokens 8192 --prefill-tokens 4096 \
  --lengths 1024 4096 --ssd-backend cufile --ssd-read-path uring \
  --output /var/tmp/orbitkv-bench/vllm-cufile-uring-reads
```

Use the SGLang environment and `--engine sglang` for its matching control.
`--ssd-read-path` defaults to unset; `uring` and `cufile` override only restores.
Explicit-path measurements require `--queue-warmup off --prepare-requests off`,
because aggregate read counters cannot separate demand from preparation reads.
The manifest records `ssd_backend` and `ssd_read_path` separately from the
`O_DIRECT` file flag. Verify positive cuFile write bytes and io_uring read bytes
alongside GPU restores; cuFile initialization does not prove the read route.
Use the [documented compatibility environment](../../docs/gds.md#enable-and-qualify)
for this container. A forced io_uring read is not native GDS read qualification
and cannot use `--gds-stats`. These are reproduction commands, not new performance
results; the existing qualification matrix is unchanged.


## Storage codec comparisons

Both engines accept `--storage-codec none|ans|fp8|turboquant-4|turboquant-3`
and `--storage-codec-budget 64mb` for OrbitKV DRAM and SSD. The budget is per
transfer worker, accepts bytes or binary `kb`/`mb`/`gb`, and does not change the
engine KV capacity. Use matching prebuilt Managers/extensions and set
`ORBITKV_NVCOMP_LIBRARY` for ANS where needed. The manifest records the selected
Manager's SHA-256, codec library environment, configured HBM/DRAM/SSD budgets,
and logical working-set bytes so an old binary is distinguishable from the
current source checkout. Build artifacts before starting any serving run.

This matched matrix runs real streaming inference with synthetic token prompts.
Start each revision in a new output root; keep the engine environment, model,
capacities, prefill limit and sampling controls unchanged. Choose an existing
SSD mount for `--ssd-dir`; raw logs stay in the temporary output directory.

```bash
export ORBITKV_CACHE_MANAGER_BINARY=/opt/orbitkv/manager
export ORBITKV_NVCOMP_LIBRARY=/path/to/nvidia/libnvcomp/lib64/libnvcomp.so.5
run_root=$(mktemp -d /tmp/orbitkv-codecs.XXXXXX)
for engine in vllm sglang; do
  for ssd_gib in 0 8; do
    if [ "$ssd_gib" -eq 0 ]; then host_gib=8; else host_gib=1; fi
    for codec in none ans fp8 turboquant-4 turboquant-3; do
      .venv/"$engine"-release/bin/python -m benches.serving.single_node \
        --engine "$engine" --backend orbitkv --model /workspace/models/qwen3-8b \
        --storage-codec "$codec" --storage-codec-budget 64mb \
        --host-gib "$host_gib" --ssd-gib "$ssd_gib" --ssd-backend uring \
        --ssd-dir /mnt/nvme/orbitkv-bench \
        --gpu-tokens 8192 --prefill-tokens 4096 --output-tokens 16 \
        --workload sustained --lengths 4096 --working-set 12 \
        --concurrencies 4 --duration-seconds 3600 --max-requests 64 \
        --reuse-ratio 0.75 --seed 20260920 \
        --output "$run_root/$engine-$ssd_gib-$codec"
    done
  done
done
```

For Qwen3-8B BF16, these 49,152 prefix tokens represent 6.75 GiB of logical KV.
The TQ4 payload estimate, including four exact layers, is about 2.32 GiB;
TQ3 is about 1.95 GiB before additional alignment and fallback. These estimates
exceed the 1 GiB host budget; actual SSD reads and GPU restores must still be
verified in each run. The engine retains at most 8,192 KV tokens (1.125 GiB).
Configured compression does not prove cache hits, complete working-set
residency, or faster inference.

The DRAM-only configuration has an 8 GiB host pool. This fits the prepared
prefixes, but the fixed cohort also introduces 15 cold prompts: total distinct
logical KV reaches about 15.19 GiB. DRAM results therefore include eviction and
the capacity effects of compression.

Require all 64 requests, `stop_reason=request_limit`, and 64/64 matching prompt
hashes against the same engine/tier's `none` control. Throughput uses actual wall
time, including completion of admitted requests. A duration-limited alternative
such as `--duration-seconds 20 --max-requests 512` can finish different request
counts; report comparison coverage instead of treating those cohorts as identical.
Repeat comparisons and reverse codec/revision order before drawing general
performance conclusions.

Summaries retain labelled logical/stored publication bytes, D2H/H2D payload
bytes, encode/decode durations, skip reasons and decode failures. The
`encoded_publication_stored_fraction` applies only to slots that contain
encoded segments, including aligned raw siblings; it excludes raw-only slots
and is not whole-cache compression. Codec durations include transfers and
fallback work, so they are not isolated kernel times or additive TTFT stages.
Workspace allocations, encode/decode batch counts and segment histogram
sums/counts expose batching and scratch reuse. Sampled workspace peaks include
idle retained arenas; active reservation bytes must drain, retained arenas may
remain. Missing metrics from older binaries are absent/null, never invented
zero measurements. `manager-usage.json` also retains whole-workload counter
changes, including preparation, separately from measured serving windows.

Cold/prepared output differences remain in every summary. For a matched
`none` control, compare measured prompt hashes and exact generated text:

```bash
python -m benches.common.report "$run_root/vllm-8-ans" "$run_root/vllm-8-turboquant-4" \
  --reference-run "$run_root/vllm-8-none" --output "$run_root/vllm-ssd-report"
```

Use separate reports for each engine/tier. Reports retain mismatch counts and
uncompared requests; older runs without prompt hashes cannot establish this
comparison. Greedy text agreement is a workload diagnostic, not application
accuracy or general model-quality qualification. Concurrent batching can also
change greedy outputs. TTFT means client time to first nonempty streamed text;
throughput counts observed completion tokens over measured wall time. Serial
TTFT results do not imply sustained throughput.

For native GDS, use the existing prerequisite verifier and prebuilt test
artifacts on a bare-metal NVMe host:

```bash
.venv/vllm-release/bin/python -m benches.storage.gds \
  --ssd-dir /mnt/nvme/qualification --model /workspace/models/qwen3-8b \
  --manager /opt/orbitkv/manager --fault-manager /opt/orbitkv/manager-faults \
  --core-test /opt/orbitkv/cufile-test --gds-tools /usr/local/cuda/gds/tools \
  --cufile-library /usr/local/cuda/lib64/libcufile.so.0 \
  --storage-codecs none ans fp8 turboquant-4 turboquant-3 \
  --storage-codec-budget 64mb --workloads serial sustained \
  --host-gib 1 --ssd-gib 8 --gpu-tokens 8192 --prefill-tokens 4096 \
  --length 4096 --working-set 12 --concurrencies 4 \
  --duration-seconds 20 --max-requests 512 --output-tokens 16 --seed 20260920
```

This covers both engines, both tiers and all selected codecs, with SSD controls
for io_uring/auto/cuFile. It retains bare-metal/NVMe prerequisites, disables
compatibility mode, requires per-process native reads and writes, and rejects
fallback. Whole-process native statistics do not identify the representation
of each transferred byte; the report preserves that attribution limit.
Strict DRAM/SSD correctness gates run for each selected codec. A failed gate
remains a qualification failure while the serving matrix continues to collect
diagnostics. The codec budget option controls the benchmarks; the existing
correctness fixtures retain their default 64 MiB budget. Keep raw artifacts in
the private run directory and publish only reviewed final summaries.
