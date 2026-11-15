# Cache retention, watermarks and SSD admission

For artifact locations and verification limits, see [benchmark evidence](performance.md#evidence-and-instrumentation).
The [pre-migration report](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/cache-policies.md) retains full tables and historical run details.

OrbitKV provides optional Rust policies for workloads larger than GPU and
host memory: protect demand-reused pages from scans, reclaim DRAM between
configured watermarks, and avoid writing every new page to SSD. They use the existing cache and backing-store owners; Python
adapters do not maintain another policy queue. Defaults remain
`--cache-protected-percent 0 --ssd-write-policy all` until matched serving
measurements justify a change.

## DRAM high/low watermarks

Set both `--cache-high-watermark-percent 80` and
`--cache-low-watermark-percent 60` to start a reclaim cycle at 80% of actual
pinned-pool capacity and stop at or below 60%. Both flags are optional together;
valid values satisfy `0 <= low < high <= 100`. Omitting them preserves
allocation-pressure eviction. These are soft maintenance targets, not admission
limits or reserved free memory. Allocation-pressure reclaim still handles
fragmentation and target-NUMA shortages.

The Manager's existing maintenance task checks once per second. Each pass
examines at most 512 candidates in each of the three replacement classes and
removes at most 512 residents in total. It follows reclaimable, probationary,
then protected order. Held candidates rotate within their class; a large held
class cannot prevent scanning the others. Partial progress keeps the cycle
active below the high watermark until actual usage reaches the low target.
The bounded scan can reach a later class before an unexamined candidate in an
earlier class; allocation-pressure reclaim still scans the full class.
No-progress passes wait for the next tick. Embedded Rust users schedule
`OrbitKVEngine::maintain_memory_cache()` themselves; the method serializes
concurrent passes without waiting for another maintenance caller.

Usage includes **all allocations charged to Storage's pinned pool**, including
pending Publish, query/transfer owners and staging allocated from that pool.
OrbiTE's separate native pools and other process memory are outside this budget.
Only cache-owned DRAM residents are selected
for removal. Removing an index entry does not prove its allocation was freed;
shared allocation owners remain charged until release. Concurrent allocations
and live holders can prevent reaching the low target. Watermarks never expire a
lease, cancel a transfer, or authorize memory reuse. Existing inventory removals
fence later source grants, and existing SSD copies are preserved. Queued weak
SSD writes can lose their source under eviction, as with allocation pressure.

This substep does not add data TTL, SSD watermarks or persistent expiry. It
changes no peer/channel protocol or SSD format. The CPU host-registration shim
exercises the real Rust allocation, replacement and inventory owners; it does
not qualify GPU or RDMA transfers. No serving-performance gain is claimed.

The reference is LMCache's
[L1 eviction controller at `bcafb031`](https://github.com/LMCache/LMCache/blob/bcafb031401d404a5dabae4853f5e23bf9e5685c/lmcache/v1/distributed/storage_controllers/eviction_controller.py):
periodic physical-usage checks and eligibility-filtered eviction. OrbitKV uses
its existing maintenance task and replacement classes, with explicit high/low
hysteresis and bounded passes instead of an eviction ratio or another controller.

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
Publish until their SSD writes complete. With `--ssd-persistent`, an SSD replica
is published only after data and journal sync. Ordinary host Publish still may
precede asynchronous write-back. Default SSD state is ephemeral; opt-in recovery
verifies every extent before exposing it. Reservations durably fence ring reuse,
and readers/native transfers retain their extents through drain. See
[persistence and GC](storage-persistence.md) and
[GPU storage ownership](gds.md#demand-routes-and-ownership).

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
demotions, the `watermark` residence-removal reason and SSD admission skips.
Keep watermarks disabled in the retention/admission comparison above; measure
them separately with the same capacities, workload and live-owner pressure.
The [benchmark guide](https://github.com/OrbitInfra/OrbitKV/blob/main/benches/README.md)
describes the serving harness; the [SSD capacity comparison](gds.md)
provides the preceding allocation and queue controls.


## Validation

Defaults remain unchanged. Protected retention and selective SSD admission need matched
workloads; they are not automatic performance wins. The [historical capacity controls](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/cache-policies.md#measured-capacity-pressure-2026-09-23) retain original results and limits. Current OrbiTE hardware use is unqualified.
