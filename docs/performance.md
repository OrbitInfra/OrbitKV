# Performance and benchmarks

OrbitKV has reviewed recovery, copy and peer-transfer measurements. A component
speedup is not an end-to-end serving advantage. The current official-release
native/OrbitKV/LMCache comparison on larger models remains a separate gate;
Qwen3.8 native output acceptance does not supply that comparison.

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
