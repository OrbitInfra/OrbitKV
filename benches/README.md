# OrbitKV benchmarks

Run performance experiments from the repository root. Benchmark programs, reusable workloads and harness
tests live here. Generated measurements and reports belong outside the checkout. Runtime Python
code belongs in `python/orbitkv/`; correctness gates belong in `python/tests/`.

## Layout

| Path | Responsibility |
| --- | --- |
| `client.py` | Admitted-query polling overhead with a held byte budget; no storage or model compute in the timed loop |
| `communication.py` | Real Query miss/hit, Publish D2H, Restore submit-to-ready and empty-restore controls using matching external artifacts |
| `tent_stage.py` | Frozen native C ABI CPU-buffer READ stages; explicit source/consumer control and terminal batch drain |
| `tent_compare.py` | Two-host frozen-library A/B READ comparison with independent pairs, notification-slot reuse and physical RDMA counters |
| `tent_native_trace.cpp` / `tent_trace.py` | Bounded pinned-native metadata/QP/bootstrap interposition and offline per-host trace validation |
| `cpu_codec.rs` | Production scalar/AVX2/AVX-512/auto CPU FP8 conversion with an independent oracle before timing |
| `cost_observations.py` | Same-binary off/on observation overhead, three reversed-order pairs on both engines |
| `metadata.py` | Explicit full-Manager 0/2/5 ms metadata coalescing comparison using frozen binaries |
| `live_store_capacity.py` | S2.10 1/4/16-owner ordinary install visibility plus separately named serial/concurrent barrier diagnostics |
| `live_store_isolation.py` | Matched full-Manager local save/query/restore under quiet or metadata-only inventory pressure |
| `observer_polling.py` | Bounded fixed-phase owner-status polling shared by the in-process and helper-process observer conditions |
| `analyze_observer_isolation.py` | Independent-pair A/B summaries for the diagnostic observer process-isolation crossover |
| `summarize_live_store_isolation.py` | Independent-run-pair ratios, absolute deltas and paired-run confidence intervals |
| `summarize_live_store_diagnostics.py` | Same-host channel/Manager/insert/SSD stage decomposition for bounded S2.10a runs |
| `single_node.py` | Fixed-capacity cold, HBM-hit, and post-pressure experiment |
| `shared_manager_pressure.py` | Two installed official engines sharing one SSD Manager: bounded sustained mixed traffic, native output controls, per-instance progress and an isolated sampler |
| `shared_cache.py` | Independent-replica serving requests with remote-byte, GPU-copy, output and reservation-drain evidence |
| `shared_cache_serving.py` | First/repeated serving recovery, retained warmup, source SSD bytes and terminal ownership |
| `shared_cache_compare.py` | Frozen independent-pair serial read-batch comparisons with consumed batch counts and no automatic promotion |
| `shutdown.py` | Native engine shutdown protocol and ordered engine/Manager stop evidence checks |
| `launch.py` | Engine/backend commands and matched memory budgets |
| `runtime.py` | Owned process groups, readiness, teardown, and launch manifest |
| `workload.py` | Token-exact requests, streaming timings, and pressure traffic |
| `concurrent.py` | Closed-loop bursts, shared/mixed prefixes, batch counters and sampled memory peaks |
| `sustained.py` | Bounded continuous traffic mixing reusable prefixes with cold requests |
| `metrics.py` | Cache-source evidence and statistical summaries |
| `timeline.py` | Request-linked preparation/restore intervals and explicit completion measurement coverage |
| `report.py` | Offline CSV/JSON reports from complete raw runs |
| `serving.sh` | vLLM serving measurements against an already running endpoint |
| `sharegpt.py` | Multi-turn workload using the pinned vLLM benchmark scripts |
| `tests/` | CPU-only checks for measurement and report correctness |
| `artifacts.py` | Validate external output locations, including symlink resolution |
| `reproduce_preparation.sh` | Repeated preparation controls with an explicit external output root |

### Native TENT first READ diagnostic

Run `python -m benches.tent_stage --bind HOST_ADDRESS --native-lib /frozen/native-lib
--role source` on the source and the same command with `--role consumer` on a
different host. Freeze libraries, the RDMA-only TENT configuration, NIC allowlist,
program and workload before launch; capture each endpoint's stdout/stderr in an
external artifact directory. A controller must keep both stdin pipes open and
parse the JSON `ready` record containing the registered CPU address and endpoint.
Send the consumer a JSON line with `operation: "read"`, a unique `id`, the source
`endpoint` and `address`, and `iterations` between 1 and 1000. Every READ clears
the destination and verifies the entire deterministic payload SHA-256.

The returned stages cover open, batch allocation, submit, terminal-status wait
and accepted free. Buffer clearing and SHA validation are outside `native_read_ms`.
Polling spins for the first millisecond and then waits one millisecond between
polls; these observations include that polling delay. Metadata, QP bootstrap and
worker scheduling can occur lazily after submit and are not separately identified
by this C ABI boundary. Use a fresh consumer process for each first-READ sample.

Send `operation: "stop"` to the consumer after `read_complete`, require `stopped`
and a reaped zero exit, then stop the source after every reader has drained.
Timeout/cancellation never releases an uncompleted batch. Control-pipe loss
retains registered buffers; source protocol errors await explicit source stop.
Do not force-kill an isolated owner and infer DMA completion from that signal.
Check recorded PIDs, RPC listeners, NIC counters and native-library hashes before
accepting cleanup. This benchmark excludes Manager grants/indexes, pinned pools,
GPU destination writes, serving, GDS and S3 crash-reclamation qualification.

For finer diagnosis, `tent_native_trace.cpp` interposes four exported C++ symbols
from Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a` on Linux x86-64 with
libstdc++. Build against that revision's exact `tent/common/status.h`; its
nontrivial `Status` return ABI and mangled symbols are not a public API:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -I third-party/mooncake/mooncake-transfer-engine/tent/include \
  benches/tent_native_trace.cpp -ldl -o /external/native-trace/libtent_native_trace.so
```

Verify all four interposed symbols exist in the frozen `libtent_shared.so` before
launch. Set `LD_PRELOAD` to the diagnostic library and `TENT_NATIVE_TRACE_PATH` to
a unique external JSON path on each endpoint. The fixed 4,096-event array records
metadata RPC, endpoint construction, connect and bootstrap RPC; it writes only
after normal process teardown. Per-slot publication uses release/acquire;
teardown seals span admission and rejects unfinished calls or unpublished slots.
An absent symbol, output failure, nonzero exit,
overflow, unsuccessful call or missing required stage invalidates the cell.
Run matched trace-off/on warm-READ overhead controls before deeper diagnosis,
with fresh endpoints and the same frozen payload, order, configuration and native
libraries. Preserve failed controls; do not use instrumented timing as a new
production performance qualification.

After both endpoints have drained and exited zero, reconstruct one consumer:

```bash
python -m benches.tent_trace \
  --trace /external/native-trace/consumer.trace.json \
  --endpoint-stdout /external/native-trace/consumer.stdout \
  --endpoint-exit-code 0 --role consumer \
  --output /external/native-trace/consumer-summary.json
```

Pass the actual exit code returned by the controller's `wait`, not a value
inferred from the stop record. Use `--role source` with the source's own files to summarize its construction
calls. The analyzer checks the process identity, retained event count, statuses,
clock boundaries and required stages. Its consumer window comes from that
endpoint's first READ. Stage calls overlap: the analyzer reports their interval
union rather than adding durations. Source and consumer monotonic epochs cannot
be compared across hosts. The controller must separately prove reaped zero exits
and absent PIDs; a parsed stop record alone cannot establish that cleanup.

Existing `rpc_server_threads` and `transports.rdma.num_lanes` settings are
experimental candidates, not OrbitKV defaults. Change one knob per frozen
cohort, keep first-READ and warmed samples separate, alternate matched run order,
and report independent-pair uncertainty and CPU/thread costs. Reducing lanes also
reduces worker/QP/CQ parallelism, so it needs throughput and consumed Manager
recovery controls. A private-buffer improvement alone cannot promote either
configuration or establish model latency.

When changing the C++ recorder, compile and run the CPU writer controls against
the same pinned header. These controls require a C++ compiler and are separate
from the default Python gate:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread \
  -I third-party/mooncake/mooncake-transfer-engine/tent/include \
  benches/tests/tent_native_trace_writer.cpp -ldl -o /external/native-trace/writer-control
```

Run modes `complete`, `active` and `unpublished` with separate
`TENT_NATIVE_TRACE_PATH` files. `complete` must exit zero with 1,024 fully published
events; `active` and `unpublished` must exit 2 and cannot produce acceptable
evidence. These synthetic writer controls do not measure transport performance.

### Native TENT library comparison

`benches.tent_compare` consumes an external frozen JSON contract. It records two
hosts and their addresses, an SSH config path, RDMA NICs/GID, remote probe/config
paths, the baseline/candidate/original library directories and SHA256 sets,
controller/probe/config hashes, payload and sample counts, pair ordering,
bootstrap seed/draw count and explicit performance guards. Copy the probe and
configuration to **both** hosts, verify their hashes and ELF dependencies, then
freeze the contract before launching endpoints. Keep libraries in separate
directories; never rebuild or replace a mapped runtime.

```bash
python -m benches.tent_compare \
  --contract /external/native-comparison/FORMAL-CONTRACT.json \
  --output /external/native-comparison/formal
```

Each cell creates a fresh source and consumer, performs one first READ, separate
warmups and measured READs, and clears/checks the entire destination each time.
Timing covers open, allocation, submit, terminal observation and accepted batch
free; clearing and SHA256 are outside that timer. Notifications run **after**
the READ samples. Send 512 unique names and deterministic 2,048-byte payloads to
wrap the pinned 256-slot ring; receive the complete exact set, reject duplicates
or corrupt messages, free every native result allocation, and poll for 25 ms
after the set is complete. This bounded quiet window is a diagnostic check, not
a proof against arbitrarily delayed duplicates.

The consumer must normally drain and exit zero before source stop. Record actual
native mappings, CPU ticks, physical port data/packet counters (data counters
use four-byte units), zero exits and absent PIDs. Bootstrap matched independent
endpoint pairs, never individual requests within a process. Report first READ,
warm median and descriptive p99 separately, including absolute times and
uncertainty. A readiness smoke is not performance qualification. An invalid
prelaunch input starts no endpoint; preserve its evidence and repair/verify
inputs before a separately frozen launch. The first invalid native cell stops
the cohort without retry or replacement. A completed cohort that misses any
frozen performance guard stays failed.

The probe deliberately holds an exported source on control loss or uncertain
native drain, has no reconnect channel, and ignores SIGTERM/SIGINT. A controller
timeout is not normal cleanup or an authorization to reuse its memory. Private
CPU-buffer results do not qualify Manager grants, registered engine pages,
GPUDirect RDMA, serving latency or requester-crash reclamation. Applying an
already merged upstream repair in this isolated comparison does not change the
production dependency pin or create a supported patched runtime.

### S2.10 metadata performance qualification

Build and freeze the Manager, wheel, server-test pressure source, extension and
TENT libraries before starting these gates. Build the metadata-only pressure
fixture with `cargo test --release --no-run -p orbitkv-server --no-default-features
--features cuda-13,mooncake,test-hooks` on the CUDA 13 profile. Freeze that test
executable separately; debug fixture CPU/RSS is not the formal production-profile
reference. The capacity harness selects one visibility endpoint per run:

```bash
python -m benches.live_store_capacity \
  --owners 16 --seed capacity-01 --samples 1000 --warmup-cycles 50 \
  --visibility-mode ordinary --enforce-thresholds \
  --output /external/s2-10/capacity-16-ordinary

python -m benches.live_store_capacity \
  --owners 16 --seed capacity-01 --duration-seconds 60 \
  --visibility-mode concurrent-barrier --barrier-concurrency 4 \
  --output /external/s2-10/capacity-16-concurrent
```

`ordinary` never calls the synchronization endpoints in its measured path.
Serial and concurrent barrier runs are diagnostics and cannot pass the ordinary
50 ms gate on its behalf.

Isolation uses one full foreground Manager and a test-owned Rust inventory source
from the frozen `orbitkv-server` test binary. Quiet and pressure runs have the
same topology, scope, foreground bytes, medium, cadence and GPU synchronization.
The pressure source changes metadata through the production inventory stream
without foreground GPU copies or SSD I/O:

```bash
python -m benches.live_store_isolation \
  --condition quiet --medium dram --seed pair-01 --warmup-rounds 50 \
  --samples 1000 --cadence-ms 1000 --order-index 0 \
  --output /external/s2-10/isolation/dram-pair-01-quiet

python -m benches.live_store_isolation \
  --condition pressure --medium dram --seed pair-01 --warmup-rounds 50 \
  --samples 1000 --cadence-ms 1000 --order-index 1 \
  --output /external/s2-10/isolation/dram-pair-01-pressure
```

Set `ETCD_BIN`, `ORBITKV_CACHE_MANAGER_BINARY`, `ORBITKV_MOONCAKE_LIB_DIR`
and `ORBITKV_SERVER_TEST_BINARY` to one frozen candidate. Reverse condition order
for alternating pairs. Use `benches.summarize_live_store_isolation` only after all
five pairs finish; request samples within a run are not independent repetitions.

S2.10a diagnostics reuse the same harness without changing the foreground path.
Pass a positive bounded event limit to use the explicit diagnostic client calls
and Manager/storage timeline. The pilot contract is non-qualifying:

```bash
python -m benches.live_store_isolation \
  --condition quiet --medium ssd --seed diagnostic-pilot-on \
  --warmup-rounds 10 --samples 120 --cadence-ms 250 --order-index 1 \
  --diagnostic-timeline-limit 16384 \
  --output /external/s2-10a/pilot/ssd-quiet-diagnostic-on

python -m benches.summarize_live_store_diagnostics \
  --run /external/s2-10a/pilot/ssd-quiet-diagnostic-on \
  --output /external/s2-10a/pilot/ssd-quiet-diagnostic-on/stage-summary.json
```

The limit requires transfer tracing and is set by the harness before importing
the native extension or starting the Manager. Zero leaves the new stages off.
`samples.jsonl` owns client submit/return observations; `diagnostic-events.jsonl`
owns bounded Manager/insert/SSD events keyed by session epoch/token and channel
request ID. Thread resource snapshots include schedstat when Linux exposes it.
Reaching `diagnostic_timeline_limit`, missing a required stage or comparing clocks
outside the recorded host domain invalidates the run. SSD queue/durability stages
remain after save return and are never folded into `manager_save_submit_ms`.

The S2.10b observer follow-up changes only where the same owner-status HTTP
poll, JSON parse and bounded JSONL record execute. `in-process` uses a Python
thread in the foreground harness; `helper-process` uses a child that imports no
CUDA, Torch or OrbitKV module. Both modes use one future monotonic epoch, fixed
observer/foreground phase offsets, every scheduled 25 ms slot without skipping,
and the same finite poll count. The ready/start/complete/stop/exit/error records
are mandatory: a helper error, digest mismatch or missing sample invalidates the
cell. A short tool smoke uses the formal cadence and pressure but is not a
performance result:

```bash
python -m benches.live_store_isolation \
  --condition pressure --medium ssd --seed observer-smoke-a \
  --warmup-rounds 10 --samples 20 --cadence-ms 1000 --order-index 0 \
  --observer-mode in-process --observer-sample-ms 25 \
  --observer-phase-offset-ms 0 --foreground-phase-offset-ms 12.5 \
  --diagnostic-timeline-limit 32768 \
  --output /external/s2-10b-observer/smoke-a

python -m benches.live_store_isolation \
  --condition pressure --medium ssd --seed observer-smoke-b \
  --warmup-rounds 10 --samples 20 --cadence-ms 1000 --order-index 1 \
  --observer-mode helper-process --observer-sample-ms 25 \
  --observer-phase-offset-ms 0 --foreground-phase-offset-ms 12.5 \
  --diagnostic-timeline-limit 32768 \
  --output /external/s2-10b-observer/smoke-b
```

The formal A/B run is a separately reviewed, pressure-only SSD cohort. Do not
pool request samples or use a smoke result for causality. Run
`benches.analyze_observer_isolation` only against the frozen eight-cell settings
and complete raw cells; its paired bootstrap unit is the independent A/B pair.
A positive result can justify a benchmark-harness change only and cannot
reclassify the failed isolation qualification.

Installed-artifact qualification runs the harness from a tests-only projection:
copy `benches/` to the projection root, copy `python/tests/` as `tests/`, and
retain the relevant pytest configuration without copying `python/orbitkv/`.
Put the installed wheel's site-packages before that projection in the driver
`PYTHONPATH`. `live_store_isolation` passes that installed site-packages path
explicitly to every Manager, including the namespace probe, so embedded Python
can import `orbitkv.client.gpu` from the wheel while the driver still imports
test support from the projection. A source checkout remains the default for
normal integration tests; installed qualification must record the client,
extension, Manager and helper paths and reject a source package that shadows the
wheel.

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

## Shared-cache qualification

For shared replicas, use the [shared-cache qualification guide](../docs/shared-cache-qualification.md).
Its HTTP driver works with either engine on existing deployments. The model-serving
restart gate lives in `python/tests/e2e/test_shared_cache.py`; same-host results do
not qualify physical two-host or RDMA deployment.

## Single-node comparison

Use separate installed environments for the selected official vLLM 0.31.0 and
SGLang 0.5.21 releases. Earlier 0.29.0/0.5.20 measurements remain historical
evidence for those revisions. Follow the [deployment guide](../docs/single-node.md)
to build/install the release wheel. The harness needs `requests`; the selected
engine provides its tokenizer and GPU runtime. LMCache and FlexKV are comparison
dependencies, not OrbitKV package dependencies. The LMCache comparison target is
the official 0.5.5 release with its MP connector; the target still needs startup
and output validation in each selected engine environment.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --installed-artifact \
  --output /var/tmp/orbitkv-bench/vllm-dram

.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --installed-artifact \
  --output /var/tmp/orbitkv-bench/sglang-dram
```

Choose an explicit empty directory with `--output /var/tmp/orbitkv-bench/<name>`.
The harness rejects output inside this checkout, including symlinks back into it.
See [benchmark evidence](../docs/benchmark-evidence.md) for historical archives,
hashes, retention and CI artifact handling. The
harness owns the engine and the OrbitKV/LMCache process; do not start other GPU
workloads during measurement. Use `--installed-artifact` for release comparisons:
child processes import installed packages without prepending this checkout's
`python/`, and OrbitKV selects the wheel's bundled Manager. A missing or
source-shadowed OrbitKV wheel is rejected before services start. The installed
SGLang plugin registration comes from the wheel. The manifest retains this mode,
the package path and the selected Manager hash. Omitting the flag is the developer
source mode, which uses the staged Manager and adapters in `python/orbitkv/`.
`ORBITKV_CACHE_MANAGER_BINARY` overrides either selection; record and freeze that
binary explicitly when using it.
The extension and Manager must come from the same source revision and protocol.
Listener ports, including SGLang's rendezvous port, are chosen outside Linux's
outgoing ephemeral range to reduce startup conflicts during GPU initialization.

Use `--backend native`, `cpu`, `orbitkv`, `lmcache`, or `flexkv`. Compare within
one engine, using identical model, GPU token capacity, host capacity, request
sequence, and dependencies. Supported model scope is dense Qwen3 in BF16, TP=1.
`native` measures resident engine HBM caching with no host cache. `cpu` uses
vLLM's OffloadingConnector or SGLang's HiCache and is the equal-host-budget
reference for OrbitKV and LMCache. Report the HBM reference separately from
external restore work. The [serving comparison method](../docs/benchmark-evidence.md#serving-comparisons)
defines correctness, capacity, repetitions and the separate distributed controls.
FlexKV compatibility failures observed on these releases are documented in the
[measurement report](../docs/single-node-performance.md); accepting a backend
option does not mean its current integration can start successfully.

For either engine with `--backend orbitkv`, use
`--orbitkv-transfer-backend direct` or `kernel` to fix the GPU H2D/D2H copy
backend at registration. Omitting the option preserves vLLM's model-based
choice and SGLang's `direct` default. The harness clears an inherited
`ORBITKV_TRANSFER_BACKEND`; for SGLang it sets this variable only from the
explicit benchmark option. Manifests and CSV reports retain the requested
value; Manager worker logs and transfer counters provide execution evidence.
This switch affects saves and restores, and does not enable dynamic selection.
For a fixed-mode comparison, keep other controls and artifacts identical and
run at least three pairs in direct/kernel, kernel/direct, direct/kernel order.
An observation-overhead comparison requires one matched transfer mode in each
off/on pair.

All outputs and cache-source evidence are retained, including mixed hits,
misses, and generated-text differences. A startup/request failure writes
`failure.json` and is not a latency result. The serial workload measures TTFT;
it does not establish concurrent goodput or production tail latency.

## SSD restoration

Choose `--ssd-backend uring`, `auto`, or `cufile` for both engines. This benchmark
keeps `uring` as its reproducible default; the Manager itself defaults to `auto`.
The cuFile backend performs complete-group GPU writes and demand restores through bounded
GPU staging; fragmented writes and preparation still use DRAM. See
[GPU storage](../docs/gds.md) for configuration, ownership and hardware limits.
`python -m benches.gds --help` describes the complete bare-metal acceptance
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
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --ssd-gib 32 --output /var/tmp/orbitkv-bench/qwen3-8b-ssd-vllm
```

For an independent read-path control, keep cuFile initialized and direct GPU
writes enabled while restoring through io_uring host staging:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
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
Use the [documented compatibility environment](../docs/gds.md#enable-and-qualify)
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
      .venv/"$engine"-release/bin/python -m benches.single_node \
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
python -m benches.report "$run_root/vllm-8-ans" "$run_root/vllm-8-turboquant-4" \
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
.venv/vllm-release/bin/python -m benches.gds \
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

## Sustained mixed traffic

For [retention and SSD admission](../docs/cache-policies.md), compare
`--cache-protected-percent 0|80` and `--ssd-write-policy all|reuse`
independently. Both flags configure the Rust Manager; `reuse` requires SSD.
Keep warming and preparation off for this comparison. The report records
protected byte peaks, promotions/demotions and admission skip reasons separately
from full write-queue drops. Use the same capacities and request sequence,
reverse run order, and repeat before choosing defaults. A lower SSD write count
alone does not establish faster serving.

For sustained mixed traffic, use `--workload sustained`. Each concurrency level
prepares an independent working set, then continuously replaces finished client
requests until the admission duration or request cap is reached. The reusable
prefix fraction is probabilistic; other requests have new token sequences and
exercise saves alongside restores. No explicit cache cleanup or pressure request
runs inside the measured window. A working set larger than HBM creates natural
GPU pressure; a smaller host pool with SSD enabled can exercise disk reads.
Verify the counters: a working-set configuration alone does not prove a tier hit.

Automatic queued warming is experimental and disabled by default. For comparisons, run the same OrbitKV workload with
`--queue-warmup on` and `--queue-warmup off`. Add `--trace-transfers` to record
request-correlated `cache_timeline` JSON in both service logs. Keep tracing equal
between controls. Hints exceeding a quarter of the query budget are skipped;
choose and report the budget explicitly. See [the lifecycle and current
limits](../docs/queued-warming.md). Streaming samples also retain the engine's
request ID for correlating measured traffic with the logs.
`timeline.jsonl` extracts only measured requests; `timeline-summary.json` reports
stage coverage and preparation/restore/queue intervals. Durations use one
process's monotonic clock or Manager-local elapsed time, never a subtraction
of clocks on different hosts. Missing restore/completion intervals have
`count: 0` and null quantiles. `completion_coverage` counts linked restore batches,
observed worker/notification events and client restore intervals;
missing stages are not counted as zero latency.
The [initial Qwen3-8B pressure controls](../docs/queued-warming.md#initial-pressure-controls)
increased SSD bytes per request without a throughput gain. These results also
retain a native HBM control for SGLang's prepared-reference output differences.
The [page-use/reclamation controls](../docs/queued-warming.md#page-use-and-reclamation-controls)
retain complete starting/ending counters and show why warming stays opt-in:
vLLM admits few hints, while SGLang releases most prepared pages unused.
Reports also retain warmup prepared/restored/unused byte deltas, completed
byte-seconds by outcome, and pending bytes before/after the window and at the
sampled peak. These count physical page footprints, not exact layer-copy bytes.
Carry-in and live pending pages prevent treating a window ratio as a completed
cohort hit rate. Enabling warming or transfer tracing for another backend is
rejected rather than recorded as an ineffective control.

Consumer-owned preparation has a separate `--prepare-requests on|off` control;
it cannot be combined with `--queue-warmup on`. The Manager read controls are
`--read-batch-mib`, `--read-timeout-ms` and `--read-max-batches`, all disabled by
default for ordinary demand. Prepared reads always use at most 32 MiB per
batch, or one oversized page. Retained prepared pages remain budgeted.
See [ownership and stopping](../docs/request-preparation.md) and the
[ordinary recovery profile](../docs/recovery-performance.md).
The [paired reproduction script](reproduce_preparation.sh) takes an external
output root as its first argument, for example
`bash benches/reproduce_preparation.sh /var/tmp/orbitkv-bench/preparation-001`.
It compares preparation on/off three times, reverses the middle pair's order,
and separates unbounded, deadline and one-batch controls. Fixed request caps
preserve the request sequence; duration-only runs can sample different traffic.
Additional DRAM-only runs record their larger host capacity explicitly.

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --concurrencies 1 4 8 --duration-seconds 60 \
  --working-set 12 --reuse-ratio 0.75 --max-requests 10000 \
  --host-gib 4 --ssd-gib 16 --query-budget-gib 2 \
  --output /var/tmp/orbitkv-bench/sustained-vllm-ssd
```

Use the SGLang environment and `--engine sglang` for its gate. For DRAM coverage,
omit `--ssd-gib` and choose a host budget that fits the working set. Run a separate
`--backend native` control with the same workload and GPU budget. Each backend
executes the same deterministic request-generation recipe, but duration-based
closed-loop runs may complete different numbers of requests.

For a larger cache working set, use `--working-set 32 --lengths 4096 8192
--gpu-tokens 65536 --host-gib 4 --ssd-gib 64 --query-budget-gib 3`.
Qwen3-8B BF16 needs about 27 GiB for these prepared prefixes, exceeding the
9 GiB GPU KV budget plus 4 GiB Manager DRAM. Sustained runs size the engine's
per-request context limit from the largest prompt plus 64 output tokens;
total GPU cache capacity can exceed one request's model context limit.
Use `--prefill-tokens` (default 8192) to compare compute scheduling quanta
separately from cache changes; it sets vLLM's `--max-num-batched-tokens` and
SGLang's `--chunked-prefill-size`. Keep it fixed in revision comparisons.
Keep `--max-requests` high enough to reach the configured duration. Compare
identical budgets, request seeds and read controls across revisions, and retain
both read and write counters along with resource-drain evidence.

`prefixes.jsonl` retains prepared token inputs and serial reference outputs.
`samples.jsonl` retains every measured response, client admission/start/finish
times, cached tokens and reference comparison. Cold requests can be regenerated
from the recorded seed, concurrency and request index. `windows.jsonl` stores
each complete window's counters, sampled memory peaks, stop reason and post-run
resource state. The cap bounds retained samples; capped windows are explicitly
marked and must not be presented as completing the configured duration.

Throughput divides completions by wall time including the final admitted
requests' completion, excluding preparation and the subsequent cache-drain check.
The drain check waits up to 30 seconds after the settling interval for query,
copy and SSD work to become idle; it fails a run that retains query ownership.
Cache counters include this drain, so they are window-level evidence, not
per-request tier attribution. Memory peaks are sampled every 25 ms and are
lower bounds. `decode_ms_per_token_p50` is a response-level estimate from text
TTFT and completion counts, not a measured inter-token arrival distribution.
Output differences from serial preparation are retained as diagnostics: greedy
sampling does not establish batch-invariant correctness. Use the engine E2E
gates and native/deterministic controls to investigate differences.
Separate cold/reused-prefix TTFT P95 values expose scheduling tradeoffs. A
reused-prefix choice may still miss the cache; these fields are not per-tier
hit latencies.
Reports retain pool-allocation failures, SSD I/O failures and dropped write
queue submissions. The total and speculative query counters are independent
of phase diagnostics; sustained validation rejects observed budget overruns.

The [sustained report](../docs/sustained-performance.md) records fresh-service
native/DRAM/SSD controls for both engines and the vLLM admission-stall regression.

## Concurrent bursts

For concurrent qualification, use `--workload concurrent --concurrencies 1 4 8`.
Each burst uses shared prefixes or independent mixed-length prompts; cold,
post-GPU-pressure, and optional post-host-eviction phases have fresh pressure
preparation. `samples.jsonl` contains individual responses and
`batches.jsonl` contains each burst's aggregate tier counters and wall time.
Counters are never attributed to individual overlapping requests. Engine cached
token reports alone do not distinguish HBM, DRAM, and SSD in these summaries.

```bash
.venv/sglang-release/bin/python -m benches.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload concurrent --concurrencies 1 4 8 --repeats 3 \
  --ssd-gib 32 --query-budget-gib 2 \
  --output /var/tmp/orbitkv-bench/query-budgets-sglang
```

Run vLLM with its release environment and `--engine vllm`. The query budget can
be smaller than aggregate demand while every individual prefix still fits.
Reports include TTFT p50/p95/p99, burst throughput, client decode milliseconds
per output token, budget waits/bypasses, coalesced reads, and memory peaks sampled
every 25 ms. These peaks are lower bounds on the true peak; lifecycle gates
separately check hard accounting limits. The fixed number of bursts is not a
steady-state load, natural memory-pressure experiment, or tail-latency SLO.
The [recorded concurrent baseline](../docs/concurrent-performance.md) includes
the discovered SGLang admission regression and native/deterministic controls
for output differences; ordinary greedy output is not assumed batch invariant.

Add `--trace-transfers` to either engine's OrbitKV run for request-correlated
discovery, host-read, restore and completion observations in `timeline.jsonl`
and `timeline-summary.json`. Manager restore time includes dispatch and worker
queueing; the load histogram separately measures the H2D worker task including
stream synchronization. `completion_signal_ms` starts at the GPU worker's terminal
timestamp and records the notification attempt, including the completion record
publication. Shared-memory completion consumes no Manager poll RPC, so the
report has no isolated consumer-delivery timer. `completion_coverage` counts
worker outcomes, notification attempts, missing observations and client restore
intervals. Regenerate reports with the code revision that produced their logs;
the current parser has no decoder for retired completion events.

For current revision comparisons, use `restore_ms`: the engine's same-process
restore-submit to GPU-ready observation. It includes submission, restore work,
waiting and consumer scheduling, and does not isolate completion delivery. Compare
matched workloads with tracing enabled on both revisions; retain interval counts,
TTFT and Manager restore/copy observations. Do not subtract their percentiles to
estimate delivery latency. Shared Manager restore batches count once, using the
Manager epoch and operation ID; engine intervals count per request and process.
Dense ordinary queries combine candidate discovery and reading; missing separate
discovery samples do not mean discovery takes zero time. No subtraction between
process clocks is used, and overlapping intervals must not be added to obtain TTFT.

## Report existing runs

```bash
python -m benches.report \
  /var/tmp/orbitkv-bench/<cpu-run> /var/tmp/orbitkv-bench/<orbitkv-run> \
  /var/tmp/orbitkv-bench/<lmcache-run> \
  --reference-run /var/tmp/orbitkv-bench/<native-run> \
  --output /var/tmp/orbitkv-bench/<report-name>
```

This needs only `requests`, not torch, either inference engine, or the native
extension. It writes `summary.csv` and `summary.json` without mixing samples
across runs. Incomplete workloads, duplicate samples, and failed runs are
rejected. `--reference-run` compares prompt hashes and generated text against
an unencoded run, including a different cache backend. It requires the same
workload, capacity arguments, model revision, GPU, launch CPU affinity, Python,
engine, PyTorch and Transformers versions. Backend-only packages may differ. Missing samples and
text differences remain explicit; exact text comparisons are diagnostic, not
a replacement for deterministic correctness gates. Native HBM-only runs record
zero configured host-cache bytes, while CPU/OrbitKV/LMCache runs record their
configured host pools. Manifests also retain the launch CPU affinity.
Retain reviewed exports in versioned external storage or CI artifacts. Publish
concise conclusions and stable evidence links in the repository. For ShareGPT,
pass `--output-dir /var/tmp/orbitkv-bench/sharegpt` to keep logs and dataset
downloads outside the checkout.

## Additional workloads and harness checks

```bash
RESULT_DIR=/var/tmp/orbitkv-bench/serving-001 \
  BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model LABEL=orbitkv \
  bash benches/serving.sh

git submodule update --init third-party/vllm
.venv/vllm-release/bin/python -m benches.sharegpt \
  --model /path/to/model --dataset-path /path/to/sharegpt.json \
  --output-dir /var/tmp/orbitkv-bench/sharegpt

uv run --isolated --no-project --with pytest --with numpy --with requests pytest benches/tests
```

The ShareGPT workload requires the dependencies listed by the pinned vLLM
`benchmarks/multi_turn/requirements.txt`. These endpoint workloads have their
own workload definitions; do not combine their numbers with the fixed-capacity
single-node comparison.

## Rust microbenchmarks

Set `CARGO_TARGET_DIR=/var/tmp/orbitkv-bench/cargo` before running a Criterion
benchmark so its generated reports stay outside the checkout. The old Catalog
cleanup benchmark was removed with the directory RPC implementation; metadata
capacity and recovery are qualified through completion stage S2.

## Native client polling

```bash
PYTHONPATH=python .venv/sglang-release/bin/python -m benches.client \
  --label rust-client --output /var/tmp/orbitkv-bench/client-poll
```

Requires built source artifacts in `target/release`, the native extension, and
one CUDA GPU for registration. It publishes 1024 pages, leases them to occupy
the 64 MiB instance budget, then measures admitted pending queries with
64/256/1024 hashes. A native `BlockHashes` batch is constructed outside the
timed loop and reused across polls. The script owns its Manager process and
records three batches of 1000 calls per size, wall percentiles and caller thread
CPU time. This isolates the client/control path; it is not a TTFT, SSD throughput,
or production concurrency measurement. Do not run it alongside Cargo builds or
other GPU workloads.

See the [controlled client measurements](../docs/client-performance.md) for the
baseline, final path and old-client/new-Manager control, with a link to the
historical per-batch evidence.

## Local communication and GPU-copy microbenchmark

See the [recorded comparison](../docs/communication-performance.md) for the
request-event and encoding changes, including latency ranges and CPU tradeoffs.

Use one frozen harness checkout and independently built matching Manager and
Python-package snapshots. The caller imports the selected package through
`PYTHONPATH`; the Manager inherits that same package and the caller's Python
runtime. The harness never substitutes this checkout's `python/` directory or
builds native artifacts. `ORBITKV_CACHE_MANAGER_BINARY` can replace `--manager`.

```bash
PYTHONPATH=/artifacts/baseline/python /path/to/cuda-python/bin/python \
  -m benches.communication \
  --manager /artifacts/baseline/orbitkv-cache-manager \
  --label baseline --output /var/tmp/orbitkv-bench/communication-baseline \
  --iterations 100 --warmup 20 --repeats 3 \
  --payload-bytes 4096 262144 4194304 --idle-ms 0 1

PYTHONPATH=/artifacts/candidate/python /path/to/cuda-python/bin/python \
  -m benches.communication \
  --manager /artifacts/candidate/orbitkv-cache-manager \
  --label candidate --output /var/tmp/orbitkv-bench/communication-candidate \
  --iterations 100 --warmup 20 --repeats 3 \
  --payload-bytes 4096 262144 4194304 --idle-ms 0 1
```

Requires a CUDA-enabled torch environment, `requests`, a release native
extension and Manager, and one GPU. Preserve the same CUDA library paths, MPS
configuration, CPU affinity, and artifact build mode. Run exclusively after
builds finish; the harness refuses competing Cargo/Manager/serving processes
and owns its Manager process group. It retains GPU tensors through Manager
teardown on errors and rejects incomplete or unsuccessful operations.

The default matrix has 1/64/1024 blocks of 4 KiB in one contiguous uint8 layer.
`--payload-bytes` specifies bytes **per layer**; output `payload_bytes` counts all
layers. To exercise multi-message Publish encoding with 36 layers and 128 pages,
use `--layers 36 --payload-bytes 524288`. Increase `--pool-mib` when the largest
payload across all layers exceeds half the pool. The 1024-block limit keeps
Query and Restore descriptors bounded; Publish can span several channel
messages. Every Publish uses fresh hashes to force D2H instead of measuring
resident-key deduplication.

Before timing, every payload size passes a real poison/restore/GPU-byte check
against retained source bytes, with different data in each layer. Restore
cohorts also check their final GPU bytes; GPU correctness checks are outside
the timed loop. Measured H2D and D2H byte-counter deltas must equal the requested
bytes. `restore_empty` uses the registered layers with no loads and measures
the no-payload submission/completion control.

Query latency includes all `query_prefetch` calls until `QueryReady`, with API
call counts retained; lease release is outside that interval. Restore latency
starts after lease acquisition and ends at native `wait_restore` readiness,
with submission and remaining wait recorded separately. Publish latency covers
`save`, including chunking and D2H; sealing synchronization and cache cleanup
run outside each sample to prevent capacity/eviction drift. Thus zero prescribed
idle for Publish still includes those maintenance calls between samples. The
1 ms interval is inserted before the measured operation, after lease acquisition
for Restore. These are client-observed durations, not CUDA-event timings or
isolated transport/notification latency.

The default `--idle-seconds 2` also records a quiet Manager CPU window after GPU
validation, without harness calls during the sleep. Its CPU seconds divided by
wall seconds reports occupied CPU cores. Each measured cohort records caller
thread/process CPU and Manager `/proc` CPU totals. Manager totals include query
preparation, release, Publish cleanup, and prescribed idle; they exclude warmup
and byte validation. The manifest records `/proc` clock-tick resolution: short
cohorts cannot establish precise per-RPC CPU costs. Use larger iteration counts
when assessing small CPU differences.

Outputs are `manifest.json`, `samples.jsonl`, `results.json`, `status.json`, and
`manager.log`. Only a complete status is a valid run. Raw samples exclude
warmup; p50/p95/p99 use linear interpolation separately for each repeat and
case. Do not pool repetitions or subtract percentiles to infer delivery delay.
The manifest identifies binary/extension/helper SHA-256, the harness, device,
Python, effective paths, affinities and MPS environment. Alternate baseline and
candidate launch order across independent runs, then compare matched cohorts.
This microbenchmark does not establish serving TTFT/ITL or multi-GPU performance.

## Cost observation overhead

Build and test native artifacts before running any serving process. The harness
never invokes Cargo and refuses to start alongside a build or another Manager.
It runs three off/on pairs for vLLM and SGLang in DRAM-only and pressured
io_uring-SSD modes, plus an ANS SSD control. Every second pair reverses order;
request seed, trace, capacity, engine release and Manager binary remain matched.
The process startup switch is `ORBITKV_COST_OBSERVATIONS=0|1`; observations are
off by default, and neither setting changes backend or recovery decisions.
The default cohort has 128 requests, concurrency
four and 1024/4096-token prefixes. DRAM capacity is 16 GiB, SSD-mode DRAM 1 GiB,
SSD capacity 16 GiB and engine KV capacity 8192 tokens.

```bash
export ORBITKV_NVCOMP_LIBRARY=/path/to/libnvcomp.so.5
.venv/vllm-release/bin/python -m benches.cost_observations \
  --model /path/to/qwen3-8b \
  --manager /absolute/path/to/prebuilt/orbitkv-cache-manager \
  --ssd-dir /path/to/ssd-test-directory \
  --output /var/tmp/orbitkv-bench/cost-observations
```

Predeclared budgets are 3% throughput loss, 3% TTFT p50 growth and 5% TTFT
p95/p99 growth, assessed as medians of paired ratios. TTFT p95 must be at most
2000 ms and response-average decode p95 at most 100 ms/token. Engine-reported
ITL histograms also require at least 95% of samples within the exact 100 ms
bucket boundary. Their p50/p95/p99 are bucket-interpolated estimates with bounds:
vLLM measures engine-core token intervals; SGLang measures tokenizer receipt
intervals, averaged over and weighted by newly received tokens. Neither uses
HTTP chunk spacing as token timing. Aggregate ITL cannot be associated with
individual requests, so request goodput uses TTFT and response-average decode.
A missing GPU restore, O_DIRECT SSD read/write, required encoded work, executed
prediction sample or raw-copy shadow sample fails the evidence gate. Exact
concurrent output differences are reported; independent correctness gates are
required because engine batching can affect generated text.

The output directory contains the predeclared `plan.json`, raw traces under
`runs/`, and only final aggregate JSON/CSV under `final/`. Use `--report-only`
with identical arguments to regenerate the summary. Keep raw data untracked.
The existing sampler scrapes Manager metrics every 25 ms. The paired delta
includes the additional cost-series export and sampling at that frequency;
it does not isolate observer hot-path cost. These finite cohorts measure
instrumentation overhead on the recorded host; they do not establish a throughput ceiling, dynamic-path benefit, native GDS
performance or distributed-cache qualification. See the final evidence in the
[implementation handoff](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#p41-final-evidence).

## Fixed DMA/kernel comparison

The same driver supports a separate `--comparison transfer-backends` contract.
It fixes `--orbitkv-transfer-backend direct|kernel` for both engines, with
observations explicitly disabled on both sides. The Manager log must confirm
the requested worker backend; a requested flag alone is insufficient. Only the
backend may differ within each pair. Raw storage is required, and the default
observation comparison continues to reject mismatched backends.

```bash
.venv/vllm-release/bin/python -m benches.cost_observations \
  --comparison transfer-backends --tiers dram --storage-codecs none \
  --pairs 3 --requests 128 --seed 20260925 \
  --model /path/to/qwen3-8b \
  --manager /absolute/path/to/prebuilt/orbitkv-cache-manager \
  --ssd-dir /path/to/existing-test-directory \
  --output /var/tmp/orbitkv-bench/transfer-backends
```

This produces 12 sessions: three direct/kernel pairs per engine, reversing the
middle pair. The fixed cohort, capacities, SLOs and regression budgets are the
same as above. `--tiers dram` performs no SSD I/O; include `ssd` for a separate
io_uring pressure control. The SSD directory must exist for preflight; only SSD
sessions create payload there. Both save D2H and restore H2D use the selected
backend, so this comparison is not a restore-only microbenchmark. It retains
completed logical GPU save/restore bytes, TTFT/ITL, throughput/goodput, Manager
CPU usage and exact-text diagnostics. These byte counters do not measure PCIe
traffic. Manager CPU includes preparation and drain; engine CPU and GPU
contention are not measured.

Each session starts a fresh Manager. These runs do not populate two candidate
estimates in a production process or qualify an automatic selector. Keep
registration defaults unchanged until useful gains, shared-device admission and
switching margins have their own evidence. `--report-only` must use the same
comparison, tiers and other predeclared arguments.

The S2.10 formal capacity command requires `--samples 1000 --warmup-cycles 50`
with ordinary mode, 16 owners and real restores before `--enforce-thresholds`
can run. Duration-only smoke runs cannot qualify the threshold. Isolation uses
`--pressure-cadence-ms 17 --observer-sample-ms 25` independently of the 1000 ms
foreground cadence; both quiet and pressure include the same observer sampler.
`exposure.json` records offered phase coverage, observed installation overlap,
source/sampler gaps and rate checks. It preserves unknown missed observations.
All fields and minimum exposure checks must be frozen before a qualification run.

The metadata pressure fixture prevents catch-up bursts by respecting both the
original schedule and half a source period after the prior iteration completes.
For a controlled pacing regression only, set `ORBITKV_PRESSURE_STALL_ROUND=200`
and `ORBITKV_PRESSURE_STALL_MS=38`; every event must remain present and the
8.5 ms minimum interval still applies to the 17 ms profile. A 200 ms stall is an
intentional invalid-exposure control: it must fail the 100 ms upper bound while
retaining explicit drain and graceful cleanup evidence. These fault variables
are not part of the ordinary formal workload.

### First and repeated cross-host serving restores

Use `benches.shared_cache_serving` against two dedicated, already-started official
engine replicas and their node-local Managers. Configure matching immutable
model artifacts, dtype, cache pages and computation identity, plus a separately
verified RDMA-only TENT profile. This driver does not launch services or detect
RDMA. The topology owner must freeze inputs, record physical NIC counters and
mapped native libraries, and clean up both hosts after success or failure.

```bash
python -m benches.shared_cache_serving \
  --engine vllm --model /path/to/immutable-model \
  --source-url http://source:8000 --target-url http://consumer:8000 \
  --source-manager http://source:50056 --target-manager http://consumer:50056 \
  --prompts /var/tmp/serving-inputs/fresh-prompts.json \
  --block-tokens 64 --bytes-per-token 147456 --remote-repeats 10 --warmup-repeats 0 \
  --source-medium dram --output /var/tmp/serving-results/profile.json
```

The byte geometry above is for BF16 Qwen3-8B, TP=1/PP=1 and one dense attention
cache group. Derive it from the selected model and registered layout for other
profiles. With 513/1025-token prompts this gate requires exactly 72/144 MiB of
remote fetch and local H2D restoration for every response, complete cold-text
and input/completion-count equality, acknowledged release and exported ownership
gauges at zero. SSD mode additionally evicts source DRAM and requires the exact
physical SSD prefetch byte delta.

Repeated reads use an idle HBM-only reset and consumer DRAM eviction, preserving
the external prefix identity. The official vLLM test server needs
`VLLM_SERVER_DEV_MODE=1` for `/reset_prefix_cache`; `reset_external=true` changes
identity and must not be used. SGLang uses `/flush_cache`. These preparation and
metadata-fence costs are recorded outside the request timer. Failed responses
and their metrics boundaries remain in the output; an existing result is never
replaced. TTFT means client time to the first nonempty streamed text. This is a
correctness gate with descriptive timings; one first read and correlated repeated
reads do not establish throughput, a tail SLO or a matched backend advantage.

### Independent SSD read-batch comparisons

`benches.shared_cache_compare` evaluates a complete, separately frozen serial
cohort. The topology owner must retain `FROZEN-CONTRACT.json`, `CAMPAIGN.json`
and every `<cell>-profile.json` together in an external artifact directory.
The comparator rejects incomplete cells, changed profile hashes, mismatched
model/token IDs or output, missing rows, incorrect bytes and a batch setting
that does not change the consumed authorization/READ counts. Compare complete
generated text and native prompt/completion counts; retain process-specific
SGLang usage metadata without requiring UUIDs or timing fields to match.

```bash
python -m benches.shared_cache_compare \
  --root /var/tmp/ssd-peer-batch/cohort \
  --contract /var/tmp/ssd-peer-batch/cohort/FROZEN-CONTRACT.json \
  --output /var/tmp/ssd-peer-batch/analysis.json
```

The 2026-10-10 component contract freezes five independent, alternating-order
process pairs per engine. Each cell retains five warmup and 20 measured restores
for each 513/1,025-token prefix; warmup exclusion is declared before launch.
Source and consumer use the same two physical H20 hosts, official releases,
immutable Qwen3-8B model, layouts and byte budgets in both variants. Only the
consumer's existing ordinary `--query-read-batch` changes from 32 to 128 MiB.
The source uses io_uring SSD and a 32 MiB read batch throughout. TENT reads into
consumer DRAM, followed by local H2D; this does not measure remote GPU-buffer
RDMA or native GDS. No timer sampler or tracer is enabled; reset, metadata fences,
drain and metrics collection remain outside request timing.

The comparator independently validates each cell's `stops` records; a PASS label
and correct output are insufficient. The frozen `source_host`/`consumer_host`
indices and service names must match. Stop and reap the consumer engine, source
engine, consumer Manager and source Manager in that order. Both Managers remain
available for context unregister. Each record must establish a requested SIGTERM
to a running service, zero leader exit, no forced cleanup, no remaining process
group and released ownership. Engine shutdown logs must be complete and contain
the selected release's completion markers without tracebacks, `EngineDeadError`
or native bootstrap/ignored-exception/SIGQUIT failures. Preserve literal service
exit codes, native child termination and cleanup failures separately. The first
invalid runtime, correctness or cleanup cell stops dependent launches; retain
its profile rows and never analyze the completed subset as a qualified cohort.
Valid performance failures remain in the predeclared comparison.

Compute one measured TTFT/E2E median per process cell and bootstrap the matched
pair ratios, never the correlated requests. The frozen component guard requires
at least a 5% geometric-mean TTFT reduction, a 95% pair-bootstrap upper bound
below parity, and no pair's TTFT or E2E median ratio above 1.05. Report the
intervals and every pair, including performance failures. A zero analysis-process
exit means the analysis completed: inspect its `state` for the component result.
Even `VALID_COMPONENT_PASS` leaves `production_qualified=false`; matched pressure
and cancellation/drain controls precede a production recommendation. This cohort
does not qualify p99, throughput or an advantage over another cache backend.

`benches.single_node` consumes the same native log check. Its service owner sends
normal SIGTERM to the leader so the engine can drain its own workers, then waits
for the leader and live members of its process group. It rejects a mismatched
`/proc` PID namespace before launching any service. A required group SIGKILL,
an unreaped zombie, an early/nonzero exit, an inspection error, a missing/truncated
shutdown log or a native failure prevents `summary.json` from being written.
The Linux owner adopts orphaned workers and reaps only its registered process
group. If inspection fails, it retains the caller's ownership while checking
kernel group existence and terminating/reaping that group; it never treats an
inspection error as an empty group. Cleanup records retain adopted child exits,
inspection failures and
the shutdown byte range and failure; a workload failure stays the primary error.
These Linux process checks do not establish physical CUDA/RDMA lifetime proof.

The retained vLLM consumer shutdown error matches the ordering described in
upstream [issue #48745](https://github.com/vllm-project/vllm/issues/48745): engine
teardown can wake the output handler before its cancellation. The related
[PR #49000](https://github.com/vllm-project/vllm/pull/49000) is open as checked on
2026-10-10. Official vLLM 0.31.0 remains unchanged. Do not suppress this error in
the benchmark or treat a CPU protocol check as a repaired hardware lifecycle.
