# Performance and benchmarks

OrbitKV has reviewed recovery, copy and peer-transfer measurements. A component
speedup is not an end-to-end serving advantage. The official-release Qwen3-32B
local comparison below is a reviewed descriptive collection; sustained matched
gains and tail qualification remain separate gates. Qwen3.8 native output
acceptance does not supply an external-cache comparison.

## Current dense local comparison

The [Qwen3-32B cohort](https://github.com/feichai0017/orbitkv/pull/221) contains
14 cells and 420 measured requests on one H20, using official vLLM 0.31.0,
SGLang 0.5.21 and LMCache 0.5.5. Model revision `9216db57`, harness `e28fe829`,
accepted native build `37b48c50`, BF16/raw, TP=1/PP=1, eager execution, 4 GiB
GPU KV and 16 GiB host offload are frozen. Each length/phase has five samples in
each of two independently started, opposite-order runs; outputs, restored bytes,
installed-file hashes and final GPU/process cleanup pass independent review.

For 8K pressure recovery, the geometric mean of the two run-median client TTFTs
is 168.5 ms for vLLM OrbitKV, 170.0 ms for official CPU offload and 175.8 ms for
this released LMCache MP profile. SGLang OrbitKV is 202.6 ms versus 178.2 ms for
HiCache, with identical aligned host-hit coverage. This SGLang gap needs profiling;
the collection does not identify a kernel or transport cause. Native HBM-eviction
recomputation takes about 4.3/4.4 seconds in the respective engine controls.

Client TTFT ends at the first nonempty SSE text, not a CUDA copy event. The vLLM
resident phase includes a 16 MiB tail restore and is mixed; SGLang recovery
restores 8,128 host tokens and recomputes the final 64-token page. These serial
observations are not stable p99, confidence intervals, throughput, pure HBM-hit
comparisons or performance promotion. Full logs retain native worker termination
diagnostics. RDMA, contention, broader ranks and S3 reclamation are separate gates.

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

Thirteen Manager cells completed the remote DRAM → RDMA → consumer DRAM → HBM
path. The first complete run's 32 MiB READ medians are 28.9–41.8 GB/s; restoration
medians are 1.25–1.32 ms. The fixture's separate 10 ms query polling sleep is
not transport latency or serving TTFT. The full 108-cell campaign stopped after
85 cells because the next Manager failed a gRPC bind before payload admission.
Its Manager comparison is incomplete, and platform zombie reap remains blocked
after native resources were released. This stage has no LMCache/model comparison
or support expansion. Full failures, per-cell clocks and provenance are retained
in [the evidence archive](benchmark-evidence.md).

## Read results by scope

| Question | Evidence and limits |
| --- | --- |
| Does external reuse beat recomputation? | The [historical matched vLLM comparison](single-node-performance.md#matched-vllm-end-to-end-comparison) records an advantage over HBM-eviction recomputation; native CPU offload remained faster in that workload. It does not establish a universal win over native KV caching or LMCache |
| What changed in the copy/control path? | [Communication measurements](communication-performance.md) retain both the original regression and later idle-stream/plan-compaction improvements; [kernel work](kernel-optimization.md) separates correctness, measured shapes and production selection |
| How do SSD and query budgets behave? | [SSD methodology](ssd-performance.md), [concurrent queries](concurrent-performance.md) and [recovery profiling](recovery-performance.md) describe bounded storage and restoration workloads |
| What is proven across hosts? | [Shared-cache qualification](shared-cache-qualification.md) records scoped TCP/RDMA recovery and restart correctness; HA, broad topology and serving performance remain separate |
| Is metadata pressure isolated? | [S2 status](completion-plan.md#s210--sustained-live-store-and-independent-domain-qualification): sixteen-owner ordinary visibility passes, but formal DRAM/SSD isolation fails; support stays four owners |

Historical reports below retain their measured engine/build versions and link to
immutable full reports. Use [compatibility](compatibility.md) for current support,
and [benchmark evidence](benchmark-evidence.md) for archives and provenance.

## Reproduce a comparison

Use the programs and workload definitions in
[the benchmark harness](https://github.com/feichai0017/orbitkv/blob/main/benches/README.md).
Write output to an explicit directory outside the checkout:

```bash
.venv/vllm-release/bin/python -m benches.single_node \
  --engine vllm --backend orbitkv --model /path/to/immutable-model \
  --output /var/tmp/orbitkv-bench/run-001
```

Freeze the model revision, installed wheel, official engine and complete runtime.
Match actual input/output tokens, concurrency, HBM/DRAM/SSD budgets, reuse pattern,
KV precision, eager/Graph execution and NUMA placement. Record correctness and
actual restore bytes before interpreting latency or throughput. Compare cold,
resident-HBM, evicted/reused and restarted-engine cases separately.

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
