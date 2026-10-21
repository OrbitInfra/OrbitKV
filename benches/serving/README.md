# Model serving benchmarks

Run commands from the repository root. [Benchmark navigation](../README.md).

## Single-node comparison

Use separate installed environments for the official releases in the
[support matrix](../../docs/compatibility.md). Historical comparisons retain
their frozen engine, OrbitKV wheel and harness versions; upgrading a pin does
not requalify their performance. Follow the [deployment guide](../../docs/single-node.md)
to build/install the release wheel. The harness needs `requests`; the selected
engine provides its tokenizer and GPU runtime. LMCache and FlexKV are comparison
dependencies, not OrbitKV package dependencies. The LMCache comparison target is
the official 0.5.5 release with its MP connector; the target still needs startup
and output validation in each selected engine environment.

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --installed-artifact \
  --output /var/tmp/orbitkv-bench/vllm-dram

.venv/sglang-release/bin/python -m benches.serving.single_node \
  --engine sglang --backend orbitkv --model /workspace/models/qwen3-8b \
  --installed-artifact \
  --output /var/tmp/orbitkv-bench/sglang-dram
```

Choose an explicit empty directory with `--output /var/tmp/orbitkv-bench/<name>`.
The harness rejects output inside this checkout, including symlinks back into it.
See [benchmark evidence](../../docs/benchmark-evidence.md) for historical archives,
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
external restore work. The [serving comparison method](../../docs/benchmark-evidence.md#serving-comparisons)
defines correctness, capacity, repetitions and the separate distributed controls.
FlexKV compatibility failures observed on these releases are documented in the
[measurement report](../../docs/performance.md); accepting a backend
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

For an SGLang admission comparison, `--sglang-priority-scheduling` passes the
released `--enable-priority-scheduling --default-priority-value 0` options.
The workload gives every request the same default priority. This also enables
native preemption plumbing and priority sorting; measure the whole configuration
and retain the actual engine command in the manifest. It changes no engine files
or adapter defaults and does not qualify mixed-priority preemption or P/D.
The [four-pair comparison](../../docs/performance.md#sglang-concurrent-admission)
improves shared-prefix recovery but fails mixed-prefix regression guards. Keep
it an opt-in experiment control, not a recommended serving default.

Use `--observation-mode boundary` for request-timing controls without an
in-process metrics sampler during requests. Engine/Manager snapshots before and
after the interval, quiet/drain checks and service cleanup still run. The default
`sampled` mode retains the 25 ms Manager observer for memory diagnostics; its
Python and Manager work can affect request timing. Sampling does not occur when
there is no Manager. Each manifest and measured interval records the mode,
Manager presence, backend and sample count. OrbitKV snapshots and every drain
poll require its eight always-exported query/transfer/SSD ownership gauges to be
finite and nonnegative; all required and present optional owners must end at zero.
Sampled OrbitKV intervals also require real query/inflight peak observations.
Live measurements and offline reports reject missing or invalid ownership
evidence. LMCache and no-Manager profiles retain snapshots and service cleanup
without claiming validation of OrbitKV's resource owners.
JSON sampled peaks are `null` when unavailable;
boundary snapshots cannot be labeled as sampled peaks or prove transient memory
capacity. Codec boundary maxima and Manager-owned reservation peaks retain their
separate scopes.

Reports reject missing or inconsistent mode metadata, and output references
reject different modes. Keep historical cells unchanged and reproduce their
reports with their frozen source; do not backfill observation metadata. Run
capacity diagnostics separately from timing controls, and predeclare a matched
observation-overhead pilot before formal tail or performance-advantage claims.
Selecting boundary mode alone does not qualify observation overhead or a faster
cache backend.

All outputs and cache-source evidence are retained, including mixed hits,
misses, and generated-text differences. A startup/request failure writes
`failure.json` and is not a latency result. The serial workload measures TTFT;
it does not establish concurrent goodput or production tail latency.

For SGLang restore diagnosis, add `--sglang-profile` to capture CPU/CUDA traces
only around each serial `after_pressure` request through the official
`/start_profile` and `/stop_profile` endpoints. The profiler uses no engine-file
patch. Every profiled sample records its mode, HTTP start/stop receipts and
compressed trace hashes under the external output directory. Missing traces or
traces without GPU kernels invalidate the cell. Stop is attempted even after
an ambiguous start failure; normal engine/Manager cleanup remains mandatory.
Keep profiler-enabled cells separate from unprofiled timing controls: profiler
activity can perturb CUDA submission and TTFT. JSON/CSV summaries and stdout
retain the profiling mode; mixed modes within a phase and inconsistent sample/
manifest declarations are rejected. Compare actual restored tokens,
output, budgets and native drain before attributing any restore-stage gap.


## Sustained mixed traffic

For [retention and SSD admission](../../docs/cache-policies.md), compare
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
limits](../../docs/queued-warming.md). Streaming samples also retain the engine's
request ID for correlating measured traffic with the logs.
`timeline.jsonl` extracts only measured requests; `timeline-summary.json` reports
stage coverage and preparation/restore/queue intervals. Durations use one
process's monotonic clock or Manager-local elapsed time, never a subtraction
of clocks on different hosts. Missing restore/completion intervals have
`count: 0` and null quantiles. `completion_coverage` counts linked restore batches,
observed worker/notification events and client restore intervals;
missing stages are not counted as zero latency.
The [initial Qwen3-8B pressure controls](../../docs/queued-warming.md#initial-pressure-controls)
increased SSD bytes per request without a throughput gain. These results also
retain a native HBM control for SGLang's prepared-reference output differences.
The [page-use/reclamation controls](../../docs/queued-warming.md#page-use-and-reclamation-controls)
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
See [ownership and stopping](../../docs/request-preparation.md) and the
[ordinary recovery profile](../../docs/performance.md).
The [paired reproduction script](reproduce_preparation.sh) takes an external
output root as its first argument, for example
`bash benches/serving/reproduce_preparation.sh /var/tmp/orbitkv-bench/preparation-001`.
It compares preparation on/off three times, reverses the middle pair's order,
and separates unbounded, deadline and one-batch controls. Fixed request caps
preserve the request sequence; duration-only runs can sample different traffic.
Additional DRAM-only runs record their larger host capacity explicitly.

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
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

The [sustained report](../../docs/performance.md) records fresh-service
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
.venv/sglang-release/bin/python -m benches.serving.single_node \
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
The [recorded concurrent baseline](../../docs/performance.md) includes
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
python -m benches.common.report \
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
  bash benches/serving/serving.sh

git submodule update --init third-party/vllm
.venv/vllm-release/bin/python -m benches.serving.sharegpt \
  --model /path/to/model --dataset-path /path/to/sharegpt.json \
  --output-dir /var/tmp/orbitkv-bench/sharegpt

uv run --isolated --no-project --with pytest --with numpy --with requests pytest benches/tests
```

The ShareGPT workload requires the dependencies listed by the pinned vLLM
`benchmarks/multi_turn/requirements.txt`. These endpoint workloads have their
own workload definitions; do not combine their numbers with the fixed-capacity
single-node comparison.


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
.venv/vllm-release/bin/python -m benches.serving.cost_observations \
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
[implementation handoff](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#p41-final-evidence).
