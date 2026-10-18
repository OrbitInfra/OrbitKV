# Local transfer and kernel benchmarks

Run commands from the repository root. [Benchmark navigation](../README.md).

### Actual SGLang pool byte and kernel-pair checks

`sglang_kv_oracle` allocates the released SGLang 0.5.21 Qwen3-32B NHD pool:
64 logical layers, 128 BF16 K/V planes, 257 pages of 64 tokens. It uses the
installed OrbitKV layout compiler, save/query/restore and real layer events.
Each plane/page/generation receives distinct deterministic bytes. After the
whole pool is poisoned, native recovery writes reversed destinations; full GPU
readback checks every payload and unselected byte against a CPU reference.
Three generations include partial `None` destinations and deliberately corrupted
last-payload/sentinel controls. This proves the copy fixture's bytes, rather than
dumping the computed model KV or qualifying native IPC/crash lifetime.

An owned Manager and exact installed runtime must be ready before this workload:

```bash
python -m benches.transfer.sglang_kv_oracle \
  --model-config /path/to/immutable-Qwen3-32B/config.json \
  --length 8192 --backend kernel --gpu-uuid GPU-REVIEWED-UUID \
  --manager-socket /tmp/owned-manager.sock \
  --manager-url http://127.0.0.1:50056 \
  --output /external/byte-control-001
```

For the bounded independent-pair evaluation, freeze four byte cells covering
both backends and 4K/8K, followed by a fresh native output control and four
freshly restarted DMA/kernel model pairs with balanced AB/BA/AB/BA order.
Use `single_node` with 4K-before-8K, five repeats, 16 output tokens, eager
deterministic inference, 16,384 GPU tokens, 16 GiB host cache and boundary
observations. Keep profiler and diagnostic tracing off. Stop admission on the
first invalid cell and preserve its actual waits, controls and diagnostics.
Unknown native completion retains live allocations and inherited resource locks,
including when a receipt cannot be written.

`kernel_pairs` consumes the frozen contract, complete cohort and actual report
validators. It rejects incomplete byte controls, wrong input/output or coverage,
profiling, changed budgets and inconsistent service stop order. It reports each
pair ratio and a geometric-mean bootstrap interval using independent pairs as
units; requests within a run are not independent repetitions. Its fixed 8K
direction guard requires all four pair medians to improve and the interval upper
bound to be at most 0.97. The 4K pressure and both lengths' cold/HBM interval upper
bounds must be at most 1.05. Valid slow results remain valid failed guards.

```bash
python -m benches.transfer.kernel_pairs --contract /external/freeze/CONTRACT.json \
  --runs /external/cohort --output /external/pair-analysis
```

Neither a passing fixture nor these serial medians select a universal default,
establish p99/throughput, prove all native children graceful, or qualify broader
model, Graph, topology, P/D and S3 profiles. The completion plan records the
implemented tools and independently accepted hardware scope separately.
The accepted 13-cell H20 cohort is retained in the
[kernel-pair archive](../../docs/benchmark-evidence.md#sglang-independent-kernel-pair-archive);
[performance](../../docs/performance.md#independent-sglang-kernel-pairs) reports its
four-pair statistics and remaining lifetime limits.


## Rust microbenchmarks

Set `CARGO_TARGET_DIR=/var/tmp/orbitkv-bench/cargo` before running a Criterion
benchmark so its generated reports stay outside the checkout. The old Catalog
cleanup benchmark was removed with the directory RPC implementation; metadata
capacity and recovery are qualified through completion stage S2.


## Native client polling

```bash
PYTHONPATH=python .venv/sglang-release/bin/python -m benches.transfer.client \
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

See the [controlled client measurements](../../docs/client-performance.md) for the
baseline, final path and old-client/new-Manager control, with a link to the
historical per-batch evidence.


## Local communication and GPU-copy microbenchmark

See the [recorded comparison](../../docs/communication-performance.md) for the
request-event and encoding changes, including latency ranges and CPU tradeoffs.

Use one frozen harness checkout and independently built matching Manager and
Python-package snapshots. The caller imports the selected package through
`PYTHONPATH`; the Manager inherits that same package and the caller's Python
runtime. The harness never substitutes this checkout's `python/` directory or
builds native artifacts. `ORBITKV_CACHE_MANAGER_BINARY` can replace `--manager`.

```bash
PYTHONPATH=/artifacts/baseline/python /path/to/cuda-python/bin/python \
  -m benches.transfer.communication \
  --manager /artifacts/baseline/orbitkv-cache-manager \
  --label baseline --output /var/tmp/orbitkv-bench/communication-baseline \
  --iterations 100 --warmup 20 --repeats 3 \
  --payload-bytes 4096 262144 4194304 --idle-ms 0 1

PYTHONPATH=/artifacts/candidate/python /path/to/cuda-python/bin/python \
  -m benches.transfer.communication \
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


## Fixed DMA/kernel comparison

The same driver supports a separate `--comparison transfer-backends` contract.
It fixes `--orbitkv-transfer-backend direct|kernel` for both engines, with
observations explicitly disabled on both sides. The Manager log must confirm
the requested worker backend; a requested flag alone is insufficient. Only the
backend may differ within each pair. Raw storage is required, and the default
observation comparison continues to reject mismatched backends.

```bash
.venv/vllm-release/bin/python -m benches.serving.cost_observations \
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


### Native trace CPU profile

This ignored Rust test replays a complete schema-2 or schema-3 native capture through the
actual private trace owner. It never dereferences captured addresses or calls
CUDA. It does not measure DMA or engine restoration. Use a reviewed capture,
record its digest and CPU topology/affinity, and select fresh external paths:

```bash
ORBITKV_TRACE_PROFILE_INPUT=/external/previous-cell/oracle.stderr \
ORBITKV_TRACE_PROFILE_OUTPUT=/external/profile/staged.json \
ORBITKV_TRACE_PROFILE_MODE=staged \
RUST_LOG=info \
cargo test -p orbitkv-core --release --no-default-features \
  --features cuda-13,mooncake --lib \
  transfer::local::trace::profile::captured_geometry_profile \
  -- --exact --ignored --nocapture > /external/profile/staged.stdout \
  2> /external/profile/staged.stderr
```

Run the already-built test executable directly for matched profiles; compilation
is outside the measurement. Alternate independent `staged`/`whole` processes on
the same CPU. Each process uses four warm-up and eight measured sequences of all
input parts, with a new executor budget per sequence. `staged` measures geometry
insertion, batch capture, drained record assembly/encoding, per-record log publication and synchronous flush. Every record is encoded,
published and released in its normal order. Nested encoding/publication timers
include a separate clock control; report their overhead and the separate
unmodified `whole` path. `finish_ns` includes drained record assembly, encoding
and publication; its nested fields are not additional exclusive stages.
The external controller must compare every stable output field, record count
and cumulative cap against the input after each process exits, symmetrically
for both modes. No JSON parsing or validation runs between timed parts. Reconstruction follows submission order,
which need not equal original plan construction order. Replay-log
`cpu_submit_ns` and `host_drain_ns` retain original input metadata, while the
profile JSON contains the newly measured CPU intervals. These staged CPU costs
cannot be added to infer GPU restore overhead or promote a transfer backend.
See the [profile result and limits](../../docs/kernel-optimization.md#native-trace-cpu-profile).


Set `ORBITKV_TRACE_PROFILE_PROBE=thread` for opt-in calling-thread CPU diagnostics;
omission or `off` disables the per-part probes, and unknown values fail. Sample
schema 2 includes the mode, profile TID and CPU-clock resolution. Six CPU-clock
readings surround open, geometry, batch capture, finish and flush, each bracketed
by wall reads. Raw stamps and signed wall-minus-CPU interval bounds are retained;
these windows include timer bookkeeping and are not identical to the historical
stage timers. In whole mode, synchronous publication flush remains in `finish`;
the separate explicit flush window does almost no work.

`total_ns` includes six clock probes. `observed_ns` also encloses two
`RUSAGE_THREAD` snapshots and preserves raw envelope timestamps. Both exclude
JSON result construction, trace construction/destruction and input preparation.
The coarse snapshots report enclosing context switches and faults; they do not
count allocations or identify the cause of a stage's latency. Off/thread
calibration uses the same newly built executable, validates every raw bracket,
usage delta, canonical geometry row and production-compiler result, and guards
both timing metrics independently. Never subtract these CPU windows from
unmatched original stage timers or clamp a negative residual to zero. See the
[diagnostic contract](../../docs/kernel-optimization.md#thread-cpu-decomposition)
before using observations to choose another candidate.


Schema-4 engine-local captures encode all fifteen unsigned geometry values in
column order as canonical positive run-length/unsigned-value LEB128 pairs,
then Base64, with the `uvarint-rle-base64-v1` tag and exact row count. Each
column reconstructs exactly that many rows in the original submission order.
Every opening record still declares the exact `copy_fields` order.
`benches.transfer.local_restore_shapes` validates framing, count and pre-decode allocation
bounds, positive run sums and canonical adjacent runs before restoring the named
geometry consumed by the production compiler. The offline compiler buffers JSON
file input; this validation is outside native profile timing. Archived schema-2
named and schema-3 array captures remain readable; mixing
formats within an executor is invalid. The ignored Rust CPU replay above accepts
schema-2/3 input and exercises the current writer; decoding and exact comparison
run after the measured process exits. Capture budgets, submission/drain clocks
and synchronous publication/flush are unchanged. Use the same complete byte,
drain and observation-overhead gates; smaller output alone does not qualify
application performance.


For prospective within-process CPU trace calibration, freeze the benchmark-only
native test ELF, captured input, accepted canonical encoding, production layout
validator, maintained controller/readers, runtime dependencies and a fixed
contract outside the checkout. Obtain independent method/source and launch
review before collecting data. `benches.transfer.native_trace_calibration` owns the
16-process order, complete validation, actual child waits and independent
process bootstrap; it requires the reviewed input-index hash and one-use
launch admission. See [the interleaved measurement contract](../../docs/kernel-optimization.md#prospective-interleaved-calibration).

The native profile's opt-in variables are
`ORBITKV_TRACE_PROFILE_CONTRAST=aa|ab` and
`ORBITKV_TRACE_PROFILE_ORDER=ABBABAAB|BAABABBA`, together with fixed probe `off`
and mode `whole`. Both settings are required together; invalid or mixed fixed
probe settings are rejected. They select schema 3 and preserve the original
schema 2 behavior when absent. Do not run ad hoc samples before freezing the
cohort or treat these CPU times as GPU/serving performance.

Run the lightweight contract and direct-child cleanup tests with:

```bash
python -m pytest benches/tests/transfer/test_native_trace_calibration.py
```
