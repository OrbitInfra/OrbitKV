# Performance and benchmarks

Current OrbiTE GPU, RDMA, GDS and installed profiles are **unqualified**.
CPU/TCP, io_uring and CUDA-shim controls are CI evidence. A faster component
or a successful byte fixture is not an end-to-end serving advantage. Use
[compatibility](compatibility.md) for current boundaries and the
[completion plan](completion-plan.md) for the product qualification sequence.

## Qualification limits

The following existing verdicts remain unchanged. Evidence is kept at its
original revision, not copied into another result table:

- Metadata DRAM/SSD foreground isolation is **FAIL**; sixteen owners remain
  **UNQUALIFIED**, although ordinary visibility passed. The scoped limit remains
  four owners. Three-host etcd controls were process-only, not independent
  failure domains or cache HA. [Frozen metadata record](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/completion-plan.md#s2--metadata-reliability-and-measured-capacity).
- SGLang priority scheduling improved the historical shared-prefix case but
  **failed mixed-prefix regression guards**. It remains opt-in.
  [Historical TENT admission comparison](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/performance.md#sglang-concurrent-admission).
- Native transfer and engine-local observers **failed overhead/tail guards**;
  diagnostic clocks are not free. The cost observation and selection subsystem has been removed.
  [Frozen observation requirements and failures](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).
- Physical termination, permanent-loss reclamation, broad rank/Graph support,
  native GDS and complete P/D faults are unqualified. Healthy outputs and
  logical cleanup do not prove hardware drain. [Fault requirements](fault-qualification.md).

## Historical evidence

These measurements used the **historical TENT backend** at the revisions in each
record. TENT has been removed. They do not qualify OrbiTE or justify current
performance/default claims:

- [Dense local comparison](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/performance.md#current-dense-local-comparison): official engine/native-offload/LMCache comparisons, budget differences and shutdown limits.
- [SGLang restore diagnosis and kernel pairs](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/performance.md#sglang-restore-diagnosis): the scoped serial kernel benefit is retained; it is not a throughput, default or current-engine claim.
- [Two-host RDMA components](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/performance.md#two-host-rdma-component-measurements): descriptive byte-checked cohorts, incomplete predecessor and resource/clock limits; not serving superiority or HA.
- [Shared-cache recovery](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/shared-cache-qualification.md): exact revisions, controlled restarts, failed startup controls and original scope.
- [SSD measurements](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/ssd-performance.md), [kernel measurements](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/kernel-optimization.md) and [codec quality/bytes](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/storage-formats.md#recorded-checks): original implementations, quality failures and native-GDS exclusions remain explicit.

Earlier OrbiTE pilots also predate today's data path and require re-qualification:
[experimental TCP/RDMA and installed controls](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/transport.md#experimental-orbite-peer-tcp-profile).
The [immutable archive index](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/benchmark-evidence.md) records external artifact
locations, hashes, failed runs and review limits. Those directories are unchanged;
some are private host archives, not public downloads. Git history preserves full
reports without keeping delivery diaries in the operating guides.

## Integrity follow-up (P11)

[Reviewed CPU VM path pairs (#344)](https://github.com/OrbitInfra/OrbitKV/pull/344)
retain these 16 MiB regressions: SSD→host **+41.9%**, ephemeral SSD write
**+32.4%**, TCP receive **+10.3%**. Persistent writes improved **13–30%** after
removing read-back. The remaining CRC pass follows O_DIRECT DMA, which has no
host copy to fuse with. A later optimization may pipeline CRC of each completed
chunk while the next chunk's I/O runs, for reads and writes. It is not implemented
or qualified; these CPU results are not GPU/serving evidence.

## Reproduce a comparison

Use the programs and workload definitions in
[the benchmark harness](https://github.com/OrbitInfra/OrbitKV/blob/main/benches/README.md).
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
The [completion plan](completion-plan.md) is the sole delivery and qualification queue.
