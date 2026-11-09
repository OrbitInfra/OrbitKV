# Retention and SSD write admission

For artifact locations and verification limits, see [benchmark evidence](performance.md#evidence-and-instrumentation).
The [pre-migration report](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/cache-policies.md) retains full tables and historical run details.

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
across Manager restart; see [GPU storage ownership](gds.md#demand-routes-and-ownership).

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
demotions and SSD admission skips. The [benchmark guide](https://github.com/OrbitInfra/OrbitKV/blob/main/benches/README.md)
describes the serving harness; the [SSD capacity comparison](gds.md)
provides the preceding allocation and queue controls.


## Validation

Defaults remain unchanged. Protected retention and selective SSD admission need matched
workloads; they are not automatic performance wins. The [historical capacity controls](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/cache-policies.md#measured-capacity-pressure-2026-09-23) retain original results and limits. Current OrbiTE hardware use is unqualified.
