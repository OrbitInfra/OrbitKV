# Performance and benchmarks

OrbitKV has reviewed recovery, copy and peer-transfer measurements. A component
speedup is not an end-to-end serving advantage. The official-release Qwen3-32B
local comparison below is a reviewed descriptive collection. The separate
SGLang kernel pairs qualify a bounded serial recovery improvement; sustained
contention and tail qualification remain separate gates. Qwen3.8 native output
acceptance does not supply an external-cache comparison.

## SGLang concurrent admission

The [official-release admission comparison](https://github.com/OrbitInfra/orbitkv/pull/294)
uses SGLang 0.5.22, Qwen3-8B revision `b968826d`, one H20, BF16/raw, TP1/PP1,
eager deterministic inference, 16,384 GPU KV tokens and 16 GiB host cache.
Four fresh baseline/priority process pairs alternate AB/BA/AB/BA, with 640
measured requests across concurrency 1/4, shared/mixed prefixes and 4K/8K inputs.
The candidate uses the official equal-priority scheduling flags; neither the
installed engine nor OrbitKV runtime is patched.

For concurrency-four shared 8K recovery after HBM pressure, the mean of process
median TTFTs falls from **881.11 ms to 218.53 ms**. The priority/baseline geometric
ratio is 0.2480, with a 95% process-pair bootstrap interval of 0.2407–0.2556.
Complete shared-prefix bursts across the declared 4K/8K sequence take 43.1% less
time, with equal restored and saved work between each matched pair.

**The full regression guard fails.** Complete mixed-prefix pressure bursts take
8.1% longer, with a 95% ratio interval of 1.0351–1.1290. Mixed cold and 8K pressure
request groups also fail their declared guards. The first shared 8K request's
forward-to-finish interval grows while later requests enter earlier; this is a
scheduler tradeoff, not a faster copy path. All prompts/outputs match, no request
retractions occur, installed/model hashes agree and final owned resources drain.
Native child `-9` and IPC shutdown warnings remain explicit lifetime limits.

Keep `--sglang-priority-scheduling` an opt-in benchmark control. These results
do not recommend a default, qualify stable tails, prove GPU execution overlap or
compare against native/LMCache/HiCache. Upstream already has a related
[scheduler proposal](https://github.com/sgl-project/sglang/pull/43901);
its unreleased changes are not consumed here. Full controls and results remain
in [the evidence archive](benchmark-evidence.md#sglang-concurrent-admission-archive).

## Current dense local comparison

The reviewed [Qwen3-32B comparison](https://github.com/OrbitInfra/orbitkv/pull/221)
uses one H20, official vLLM 0.31.0/SGLang 0.5.21, LMCache 0.5.5, BF16/raw,
TP1/PP1, eager execution and 4 GiB GPU KV. Two opposite-order independent runs
contain five samples per length/phase, with output and restore-byte controls.
For 8K recovery after HBM pressure:

| Engine | OrbitKV TTFT | Official host offload TTFT | Released LMCache MP TTFT |
| --- | --- | --- | --- |
| vLLM | 168.5 ms | 170.0 ms | 175.8 ms |
| SGLang | 202.6 ms | 178.2 ms | Startup blocked |

Values are geometric means of run medians. OrbitKV requests 16 GiB host cache;
historical HiCache requests 16 decimal GB before page rounding, so its budget
is different. Native HBM-eviction recomputation takes about 4.3/4.4 seconds.
The vLLM resident control includes a 16 MiB restore; SGLang restores 8,128 tokens
and recomputes the final 64-token page. TTFT ends at first nonempty SSE text.
These observations establish neither stable tails nor general superiority.
The [frozen report](https://github.com/OrbitInfra/orbitkv/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/performance.md#current-dense-local-comparison)
retains revisions, native termination diagnostics and complete scope.

## SGLang restore diagnosis

The [follow-up diagnosis](https://github.com/OrbitInfra/orbitkv/pull/224)
reproduces the serial 8K gap and identifies fragmented submission and admission
as investigation targets. Profiler-off TTFT is 202.09 ms for DMA, 181.53 ms for
the existing kernel route and 175.90 ms for historical HiCache. Traces do not
assign a causal fraction of the unprofiled gap or establish first-layer readiness.
KDA evaluation supports the independent kernel comparison below.
The [frozen report](https://github.com/OrbitInfra/orbitkv/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/performance.md#sglang-restore-diagnosis)
preserves different preparation orders, failed controls and native child/IPC
shutdown limits. Full traces and reviews remain in [the archive](benchmark-evidence.md).

## Independent SGLang kernel pairs

The [independent follow-up](https://github.com/OrbitInfra/orbitkv/pull/225)
at harness `4b6c7eba` uses the unchanged `37b48c50` native build, official
SGLang 0.5.21 and immutable Qwen3-32B revision `9216db57`. One H20 runs BF16/raw,
TP=1/PP=1, eager deterministic inference with 16,384 GPU KV tokens and 16 GiB
host cache. Four freshly restarted DMA/kernel pairs alternate AB/BA/AB/BA;
a fresh native output control brings the model collection to nine cells and
270 measured requests. Profiling and diagnostic tracing stay off.

Pressure TTFT below is the mean of the four process-pair medians, not a pooled
request median. Uncertainty bootstraps the four independent pair log-ratios.

| Input tokens | Direct DMA TTFT ms | Kernel TTFT ms | Kernel/DMA geometric ratio | 95% pair-bootstrap interval |
| --- | --- | --- | --- | --- |
| 4,096 | 176.71 | 176.78 | 1.0004 | 0.9941–1.0094 |
| 8,192 | 202.17 | 181.05 | 0.8955 | 0.8892–0.9031 |

All four 8K pressure pairs improve. The geometric mean reduction is 10.45%,
with a 95% interval of 9.69%–11.08%, passing the frozen 3% benefit guard.
The 4K pressure result is effectively unchanged; both lengths' cold and HBM
controls also pass their frozen 5% interval guards. These measurements compare
the existing explicit kernel route with DMA. HiCache and LMCache timings from
earlier collections are not matched controls for this cohort.

Before model timing, four byte-control cells exercise both backends and lengths
on the actual released SGLang pool. Twelve restore generations check all
4,311,744,512 registered bytes per generation, with 1,536 plane digests,
partial destinations and deliberate payload/sentinel corruption controls.
This verifies deterministic fixture bytes, separately from model-computed KV;
all model outputs match the fresh native control and preserved canonical output.
Independent review reconstructs byte controls, matched statistics, consumed
inputs, literal waits and unchanged installed files. Physical postflight is empty.

Nine native child exits `-9`, eight CUDA IPC producer warnings, logged native
client PIDs without individual wait statuses and historical missing waits remain
explicit lifetime limits. This acceptance qualifies the
retained kernel direction in this serial profile; it does not select a default
or establish p99, throughput, concurrent inference, Graph, broader models/ranks,
P/D, HA or S3 reclamation. Complete inputs, failed preparations and reviews remain
in [the external archive](benchmark-evidence.md#sglang-independent-kernel-pair-archive).

## Two-host RDMA component measurements

The separate two-H20 profile used the unchanged `37b48c50` runtime and pinned
Mooncake `71973589`. Its complete 72-cell GPU-pointer matrix measured READ/WRITE,
both payload directions, three restarted runs and one versus four allowed NICs.
For a packed 32 MiB descriptor, the median of the six cell medians is
41.3–44.9 GB/s. Splitting the same payload into 8,192 4 KiB descriptors with real
4 KiB gaps gives 2.38–2.42 GB/s. Every transfer verifies the full registered
64 MiB source and target with CPU/GPU byte oracles and sentinel gaps.

These are warm binding-to-terminal measurements, including Python conversion,
Rust polling and native work. They do not isolate fabric bandwidth, identify a
TENT defect or establish four-NIC striping. Packed first-transfer samples are
64.8–70.3 ms with the near-NIC allowlist and 157.8–194.8 ms with four allowed
NICs, after context initialization and registration. Descriptor overhead and
lazy setup require separate attribution before changing a production path.

The earlier 108-cell campaign stopped after 85 cells because the next Manager
failed a gRPC bind before payload admission. Thirteen Manager byte-path cells
completed, but that campaign's planned Manager comparison and historical platform
zombie wait/reap remain incomplete; the original cohort stays invalid.
An independently reviewed physical-resource readmission enabled a new full
Manager-only cohort, without replacing any old sample or accepting old cleanup.

The new `00c4a041` harness uses the unchanged `37b48c50` runtime for 36 cells:
three separately restarted runs, both directions, three payload sizes and near-NIC
versus four-NIC allowlists. It completes 1,080 measured restores and 1,224 restores
including cold/warmup samples. Remote DRAM → RDMA → consumer DRAM → HBM bytes total
17,326,669,824 for both READ and H2D; full tensors and sentinel gaps match an
independent CPU reconstruction. Independent review accepts this descriptive
outcome and normal post-drain cleanup.

Warm values below are medians of six cell medians, representing three independent
runs in both directions. They are descriptive component measurements:

| Payload | NIC allowlist | TENT READ GB/s | Manager fetch ms | GPU restore ms |
| --- | --- | --- | --- | --- |
| 512 KiB | Near NIC | 1.42 | 1.05 | 0.61 |
| 512 KiB | Four NICs allowed | 1.45 | 1.02 | 0.60 |
| 8 MiB | Near NIC | 16.48 | 1.19 | 0.81 |
| 8 MiB | Four NICs allowed | 19.13 | 1.12 | 0.80 |
| 32 MiB | Near NIC | 29.13 | 1.84 | 1.28 |
| 32 MiB | Four NICs allowed | 41.12 | 1.51 | 1.29 |

The six independent 32 MiB cell medians span 28.95–29.23 GB/s for the near NIC
and 38.60–41.82 GB/s with four allowed NICs. In those 32 MiB four-NIC cells,
observed source-TX counter deltas are approximately equal on `mlx5_3` and `mlx5_4`.
These physical counters include background traffic; they do not establish
process-exclusive accounting or four-way striping. First READs span 63.7–71.0 ms
for the near NIC and
158.2–199.5 ms with four allowed NICs across all payload sizes. Setup and warm
throughput therefore need separate comparisons. Local restoration changes little.

The fixture's separate 10 ms query polling sleep is not transport latency or
serving TTFT. Current-stage processes are actually waited and native resources
drain, but 72 raw CUDA IPC shutdown warnings remain a lifecycle limit; this does
not qualify cancellation, page reuse or crash reclamation. There is no stable
p99, inference contention, LMCache/model comparison or support expansion in this
cohort. Full failures, per-cell clocks and provenance remain in
[the evidence archive](benchmark-evidence.md).

## Read results by scope

| Question | Evidence and limits |
| --- | --- |
| Does external reuse beat recomputation? | The [historical matched vLLM comparison](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md#matched-vllm-end-to-end-comparison) records an advantage over HBM-eviction recomputation; native CPU offload remained faster in that workload. It does not establish a universal win over native KV caching or LMCache |
| What changed in the copy/control path? | [Communication measurements](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md) retain both the original regression and later idle-stream/plan-compaction improvements; [kernel work](kernel-optimization.md) separates correctness, measured shapes and production selection |
| How do SSD and query budgets behave? | [SSD methodology](ssd-performance.md), [concurrent queries](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/concurrent-performance.md) and [recovery profiling](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/recovery-performance.md) describe bounded storage and restoration workloads |
| What is proven across hosts? | [Shared-cache qualification](shared-cache-qualification.md) records scoped TCP/RDMA recovery and restart correctness; HA, broad topology and serving performance remain separate |
| Is metadata pressure isolated? | [S2 status](completion-plan.md#s2--metadata-reliability-and-measured-capacity): sixteen-owner ordinary visibility passes, but formal DRAM/SSD isolation fails; support stays four owners |

Historical reports below retain their measured engine/build versions and link to
immutable full reports. Use [compatibility](compatibility.md) for current support,
and [benchmark evidence](benchmark-evidence.md) for archives and provenance.

## Reproduce a comparison

Use the programs and workload definitions in
[the benchmark harness](https://github.com/OrbitInfra/orbitkv/blob/main/benches/README.md).
Write output to an explicit directory outside the checkout:

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /path/to/immutable-model \
  --output /var/tmp/orbitkv-bench/run-001
```

Freeze the model revision, installed wheel, official engine and complete runtime.
Match actual input/output tokens, concurrency, HBM/DRAM/SSD budgets, reuse pattern,
KV precision, eager/Graph execution and NUMA placement. Record correctness and
actual restore bytes before interpreting latency or throughput. Compare cold,
resident-HBM, evicted/reused and restarted-engine cases separately.

`--host-gib` requests binary GiB; SGLang's integer `--hicache-size` consumes
decimal GB. The harness uses `--hicache-size 0` and a host/device byte ratio to
preserve the requested budget, before native page rounding. Earlier comparisons
passed the same numeric value without conversion: their nominal 16 GiB HiCache setting
requested 16 GB, before page rounding. Retain those measured latencies, but do
not interpret their host budgets as equal to OrbitKV's 16 GiB. Verify actual
allocated tokens and bytes in engine logs as well as configured budgets.

Report TTFT, inter-token latency, request latency and throughput with independent
run uncertainty. For transfer components, include payload size, batching,
registration/setup cost and completion semantics. Keep slow samples and failed
runs. A request timed outside native execution may include observer/GIL or
scheduling interference; do not silently subtract it or change a frozen metric.

## Evidence and instrumentation

Raw JSON/CSV, traces, logs, manifests and generated plots live in external artifact
directories or CI artifacts. Source contains reusable programs, small fixtures
and reproduction instructions. Publish reviewed conclusions with revision,
hardware, budgets and uncertainty; do not treat one smoke run as qualification.

[Metrics](metrics.md) are useful for operation. Diagnostic timeline and observer
overhead require their own control before use in a performance experiment.
The [completion plan](completion-plan.md) is the sole queue for optimization and
qualification work.
