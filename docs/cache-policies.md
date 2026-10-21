# Retention and SSD write admission

For artifact locations and verification limits, see [benchmark evidence](benchmark-evidence.md).
The [pre-migration report](https://github.com/OrbitInfra/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/cache-policies.md) retains full tables and historical run details.

OrbitKV provides two optional Rust policies for workloads larger than GPU and
host memory: protect demand-reused pages from scans, and avoid writing every
new page to SSD. They use the existing cache and backing-store owners; Python
adapters do not maintain another policy queue. Defaults remain
`--cache-protected-percent 0 --ssd-write-policy all` until matched serving
measurements justify a change.

## Protect reused pages

Set `--cache-protected-percent 80` to give the protected replacement segment a
maximum footprint of 80% of the pinned pool. This is a replacement preference,
not an additional memory reservation. The cache can still use the full pool.

With protection enabled, newly saved and ordinarily restored pages enter the
probationary segment. A foreground prefix lookup promotes matching pages into
the protected segment. Promotion demotes its oldest pages first when necessary
to stay within the byte limit. An oversized probationary page is not promoted.
Speculative warming and consumer preparation do not count as foreground demand.

Pressure eviction visits reclaimable replicas, probationary pages, then
protected pages, using recency within each class. It skips pages held by query
leases or transfers. Protection does not prevent eventual pressure eviction or
override GPU completion ownership. Promotion validates the resident allocation;
a stale reference cannot protect a replacement allocation for the same key.
Pressure eviction and memory cleanup update the protected count.

The existing `--enable-lfu-admission` option is independent. Leave it disabled
when measuring this policy so admission and replacement effects are separable.

## Admit SSD writes selectively

`--ssd-write-policy all` writes newly saved pages asynchronously as before.
`--ssd-write-policy reuse` admits a page when either:

- a foreground query returns that page; or
- a new copy of the same state key is saved within the bounded reuse history.

The second condition permits a recomputed page to reach SSD after its original
DRAM replica was evicted. Duplicate saves that already hit DRAM are filtered
before this policy and do not count as a second publication. The history holds
16,384 exact state keys, including their namespace, and owns no payload memory.
History eviction can make an old key cold again. Repeated keys within one batch
do not manufacture reuse.

Both policies skip already resident SSD keys and pending writes. Queued writes
hold weak source references; the writer acquires strong references when it
prepares a batch and retains them through I/O completion. Queue rejection and
dead source references leave later writes retryable. The existing writer still
holds the prepared batch: its staging footprint is not a new per-I/O byte cap.

Foreground lookup is evidence of interest, not proof that the engine eventually
consumed the page. Pure HBM hits and prepared-lease handoff do not provide an
additional reuse signal. The first reuse after DRAM eviction may need
recomputation under `reuse`, because no SSD replica was written on the first
publication. Selective writes can therefore reduce write traffic while making
latency or throughput worse. With io_uring, save completion confirms GPU-to-DRAM
copy completion, not an SSD replica. Complete-group cuFile writes also hold
Publish until their SSD writes complete. Neither path provides durability
across Manager restart; see [GPU storage ownership](gds.md#data-flow-and-ownership).

## Configure and measure

For a capacity-pressure experiment:

```bash
orbitkv-cache-manager \
  --pool-size 4gb --query-budget 3gb \
  --ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 64gb \
  --cache-protected-percent 80 --ssd-write-policy reuse
```

Keep model revision, engine release, GPU/host/SSD capacities, prompt sequence,
concurrency, prefill batching and tracing fixed. Compare these configurations
with warming and consumer preparation disabled:

| Control | Protected percent | SSD write policy |
| --- | ---: | --- |
| Existing replacement and writes | 0 | all |
| Retention only | 80 | all |
| Write admission only | 0 | reuse |
| Combined policies | 80 | reuse |

Use a working set larger than GPU plus host capacity. Repeat windows with order
reversal before recommending a default. Record TTFT, output-token throughput,
decode timing, SSD read/write bytes per request, allocation failures, write
drops and resource drain. Intentional admission skips are distinct from full
write-queue drops. The benchmark accepts both Manager flags and checks sampled
protected bytes against the configured cap.

The [metrics reference](metrics.md) includes protected bytes, promotions,
demotions and SSD admission skips. The [benchmark guide](../benches/README.md)
describes the serving harness; the [SSD capacity comparison](ssd-performance.md)
provides the preceding allocation and queue controls.

## Measured capacity pressure (2026-09-23)

The [final aggregates and reproduction instructions](https://github.com/OrbitInfra/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results/20260923-cache-policies/README.md)
use one H20, BF16 Qwen3-8B, TP=1, vLLM 0.29.0 and SGLang 0.5.20. A 27 GiB
prefix working set competes for 9 GiB GPU KV and 4 GiB Manager DRAM; SSD is
64 GiB, query admission is 3 GiB and concurrency is eight. Warming and consumer
preparation are off, with the same tracing and 8K prefill limit in every arm.

Each short window contains the same 192 requests, including 142 reuse choices
and 50 cold requests. There are three repetitions per configuration, with the
middle repetition's order reversed. Values below are means of per-window
measurements; brackets show their min/max, not confidence intervals.

[Full historical measurement table](https://github.com/OrbitInfra/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/cache-policies.md)

Protection alone changes throughput by +1.3% in vLLM and +0.3% in SGLang;
there is no consistent paired tail-latency improvement. Committed SSD writes
fall about 36% and 44%, while read traffic rises slightly. This is not proof
of better write admission: queued weak sources can expire before the writer
acquires them, and that loss is not counted by the admission-skip counter.

Selective admission reduces short-window throughput by about 30% in both
engines. Its initial missing SSD replicas lead to recomputation on first
reuse. Combining it with protection performs worse here and produces more
partial-prefix hits. Page protection does not reserve an entire restorable
prefix or bundle; compiled recovery still validates the actual available state.

A longer 768-request comparison tests continued cold traffic and SSD turnover.
These are single windows per arm, not repeated evidence for changing defaults:

| Engine | Policy | Output token/s | P95 TTFT (ms) | SSD writes (GiB/window) |
| --- | --- | ---: | ---: | ---: |
| vLLM | 0 / all | 50.94 | 3098 | 231.31 |
| vLLM | 0 / reuse | 59.00 | 2769 | 27.00 |
| SGLang | 0 / all | 47.63 | 4073 | 205.41 |
| SGLang | 0 / reuse | 54.15 | 3857 | 27.00 |

In this extended vLLM pair, selective admission improves throughput by 15.8%
and P95 by 10.6%. Across successive 192-request segments, chosen reuse requests
with no cache hit number `1, 32, 22, 26` under `all`, versus `31, 1, 0, 0` under
`reuse`. That is consistent with an initial population cost followed by less
SSD churn. It does not establish the best policy for a different reuse
distribution, device, duration or memory budget.

The extended SGLang pair improves throughput by 13.7% and P95 by 5.3%.
Window write traffic falls 88.3% in vLLM and 86.9% in SGLang. SGLang still
reports 1,510 dropped write-queue blocks under `all` and 107 under `reuse`;
these are best-effort replica losses, not failed inference requests. Neither
extended pair reports an allocation or SSD read failure, and both drain their
query, copy and SSD work. Repeat with order reversal and production traces
before recommending selective admission for a deployment.

Reference-prefix preparation is outside latency timing. It writes about 27 GiB
before `all` windows and none before `reuse` windows; the retained CSV includes
both starting and ending write counters. The storage mount is overlay with
verified direct I/O, so these numbers do not qualify physical NVMe bandwidth.
Concurrent serving output differences are retained separately from exact-byte
and deterministic-output correctness gates.

All 24 short windows finish without allocation or SSD read failures, within
the query and protection budgets, and with query/copy/SSD work drained. SGLang
still drops some full write-queue batches; the CSV counts dropped blocks
separately from intentional admission skips. Defaults stay `0 / all`.
Use `reuse` as a measured write-traffic tradeoff, not a universal acceleration
switch. Next steps are engine HBM-use and prepared-consumption evidence,
restorable-prefix/bundle retention, and bounded writer staging. Cost-based
decisions require those signals and measured restore/recompute costs.

Qualification also passes 14 Rust SSD round trips, 12 adapter GPU checks and
12 deterministic fault cases. With protection enabled, the Qwen3-8B vLLM
output gate passes six checks (one hybrid-only check is inapplicable), and
SGLang passes both DRAM and SSD output/restart checks with selective writes.
See the [correctness gate commands](../python/tests/README.md) and
[fault contracts](fault-qualification.md). These are TP=1 qualifications.

## Reference designs and scope

The probationary/protected split follows the established segmented-LRU idea
described in [FlexKV's replacement policies at `738ddc14`](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/docs/eviction_policy/README_en.md).
OrbitKV bounds its protected segment by bytes and preserves existing transfer
leases and replica classes.

[SGLang v0.5.20's HiCache controller](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/managers/cache_controller.py)
separates write-through, selective write-through and write-back. OrbitKV applies
selective admission to its DRAM-to-SSD path; the tiers and ownership contract
are different. This is not an implementation of HiCache's entire policy stack.

[LMCache v0.5.5's prefetch controller](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_controller.py)
distinguishes demand-held results from unowned warming. OrbitKV keeps that
distinction in its existing query/lease lifecycle and does not count warming
as demand for these policies.

These are reuse heuristics, not model execution prediction. Compiled recovery
requirements still determine which pages, windows and checkpoints are legal
to restore. Cost-based restore-versus-recompute decisions, execution-time
forecasting and finer copy/compute overlap remain separate work in the
[completion plan](completion-plan.md).
