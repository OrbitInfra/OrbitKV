# OrbitKV Metrics Guide

This guide explains how to collect, export, and visualize metrics from OrbitKV.

## Overview

OrbitKV supports two methods for exposing metrics:

### Method 1: Direct Prometheus (Recommended)

```
Cache Manager → Prometheus → Grafana
   (/metrics)      (scrape)    (visualize)
```

- Simpler deployment (2 components)
- OrbitKV exposes `/metrics` endpoint directly
- Use `examples/metric-prometheus/`

### Method 2: OTLP via OpenTelemetry Collector (Optional)

```
Cache Manager → OpenTelemetry Collector → Prometheus → Grafana
   (OTLP/gRPC)         (HTTP scrape)       (HTTP queries)
```

- More flexible (supports multiple backends)
- Useful if you already have OTel infrastructure
- Use `examples/metric/`

## Available Metrics

OrbitKV exposes the following metrics for monitoring KV cache operations:

The baseline readiness gauges for query reservations, publish inflight bytes,
peer-transfer ownership and ordinary SSD operations export an explicit zero when
the Manager initializes its core metrics, before the first request. This does
not initialize speculative-prefetch, GDS, codec or phase-specific series.
`orbitkv_load_bytes_total` also starts at zero. Initialization adds zero
once; subsequent owner increments and releases remain cumulative, and obtaining
the metrics again does not reset active ownership. Readiness and drain checks
must still reject missing required gauges; an absent series is not evidence of
zero outstanding work.

### Bounded cost observations

See the [P4.1 boundaries](state-planning.md#p41-observation-contract) before
combining measurements. All labels below come from fixed enums; device, peer,
request and state identities are kept out of the exported series.

| Metric family | Labels / meaning |
| --- | --- |
| `orbitkv_cost_operations_total` | `path`, `outcome`: completed, failed, cancelled, timed_out or abandoned; externally reported handoffs also carry bounded `admission` |
| `orbitkv_cost_stage_seconds` | `path`, `outcome`, `stage`: queue, admission, completed service or inclusive total; externally reported handoffs also carry bounded `admission`; microsecond-to-second buckets |
| `orbitkv_cost_logical_bytes_total` / `orbitkv_cost_logical_unknown_total` | Known attempted logical bytes versus unavailable unpadded sizes; do not sum nested owners |
| `orbitkv_cost_io_bytes_total` / `orbitkv_cost_io_unknown_total` | Known physical bytes, including short I/O, versus unknown counts; same path/outcome labels |
| `orbitkv_cost_prediction_absolute_error_seconds` | Successful observation error against the prediction captured before submission: service for individual operations, enqueue-to-terminal total for complete SSD restore and HostReady routes |
| `orbitkv_cost_shadow_candidates_total` / `orbitkv_cost_shadow_prediction_seconds` | Feasible raw-copy or SSD restore-route candidates, labelled by path and known/unknown evidence |
| `orbitkv_cost_shadow_decisions_total` | agree, different, within_margin, incomparable or unknown; execution never follows this result |
| `orbitkv_cost_route_decisions_total` | `decision` is `default`, `selected`, `within_margin`, `unknown`, `incomparable` or `contention`; `scope` distinguishes `peer_owner` from `cross_medium` |
| `orbitkv_cost_estimate_samples`, `orbitkv_cost_estimate_age_seconds`, `orbitkv_cost_estimate_error_seconds` | Count, sample age and EWMA absolute error supporting known shadow predictions |
| `orbitkv_cost_estimate_evictions_total` / `orbitkv_cost_estimate_dropped_total` | Fixed-capacity eviction and skipped updates on estimator contention |

`path="ssd_uring_restore"` and `path="ssd_cufile_restore"` identify complete
SSD restore routes to the same GPU completion target. Their estimates include
queueing and host materialization/H2D or GPU staging/scatter/decode. Other paths
continue to estimate service from submission to observed completion; stage
histograms retain their queue/admission/service/total meanings. Neither failed
nor cancelled, timed-out or abandoned work trains successful estimates.
Peer DRAM authorization uses `path="remote_authorization"`; peer SSD uses
`path="remote_ssd_authorization"`, whose service interval also includes
source-local io_uring staging. The following `remote_read` remains the Mooncake
TE payload interval for either source medium.
`peer_dram_host_ready` and `peer_ssd_host_ready` are non-additive composite
samples from authorization start through destination block reconstruction.
They alone feed the narrow peer-owner execution selector.
`local_ssd_host_ready` spans queue admission, host allocation, io_uring reads,
validation and host-block reconstruction. It uses stored bytes and block count,
matching the peer HostReady shape. Equal-coverage, single-owner alternatives
enter cross-medium shadow. They affect execution only under the separate
three-flag experiment gate.

`prefill_to_decode_handoff` is a decode-owned, enqueue-to-TENT-terminal
observation for the registered destination GPU. It records raw logical and wire
bytes plus target-layout fragment shape after successful completion; failed,
cancelled and timed-out reports keep physical bytes unknown. The source
endpoint is hashed into the internal execution-resource key; endpoint text,
request identity, cache keys
and transfer generation are not metric labels. The generation is required
as freshness evidence but is not an estimator dimension. Only
`admission="admitted", outcome="completed"` trains the estimate.
`engine_local_restore` records native caller-to-GPU-drain duration, including
readiness, dispatch, native queue and grant wait. The record is accepted once by
the source owner for the same session/operation generation. Only a successful
submitted drain trains its bounded estimate; failure durations are diagnostic.
Its start boundary differs from `cache_restore` and P/D, so it is excluded from
cross-route comparison. Cache restore resources separate the source domain,
destination GPU and registered copy backend, so direct and kernel executions
do not train the same estimate. Enable `ORBITKV_COST_OBSERVATIONS=1` in both the engine
and Manager to collect it. `ORBITKV_TRACE_TRANSFERS=1` additionally exposes the
stage decomposition and native result-consumption delay; tracing alone does not
enable cost selection.

S2.10a publish/query stage diagnosis additionally requires a positive
`ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT`. It is disabled at zero and capped at 65,536
events per process. Events bind the authenticated session epoch/token and channel
request ID, use same-host `CLOCK_MONOTONIC`, and cover client, Manager, insert and
SSD-writer boundaries. A `diagnostic_timeline_limit` event invalidates the run.
Client return, insert completion and SSD completion are different endpoints; the
diagnostic never redefines save submission as SSD durability.

With a positive diagnostic limit, existing transfer records and the additional
stage records share a bounded asynchronous writer instead of synchronously
formatting output on request/storage threads. The total accepted records, not
only the additional stage records, consume the process limit. Normal Manager
shutdown drains request and storage owners before flushing the writer. A missing
required stage, a limit event or an abnormal process exit remains invalid; the
buffer does not manufacture completion evidence.
Required stage and foreground query-path events enter the writer as fixed-size
records; their JSON representation is built only by the writer. Less common
legacy transfer records retain their existing structured fields but consume the
same total bound.

`publish_manager_response_publish` is recorded immediately before the response
is made visible; `response_publish_to_client_ms` therefore includes response
publication and notification/wakeup. It is not an after-send timestamp. Normal
Manager shutdown stops the channel, drains GPU owners, then flushes deferred
Publish continuations, insert and SSD work before flushing the diagnostic writer.
Writer-start failure
emits a limit event and invalidates the run instead of falling back to unbounded
logging. SSD completion accounting retains every intervening write batch even
when only one batch carries diagnostic correlation.

`cache_restore` supplies the GPU-completed boundary for
Manager-executed restores, from preparation after framework page allocation
through terminal GPU completion. Its plan retains `RestoreTargetShape`
(destination device, bytes and fragment count), source-set identity and source
geometry. The shape is metadata; a separate permit owns device admission.
Engine-local raw restores do not train this estimator until their actual
drain evidence is connected. The observation does not enable direct/P-D
selection: one execution owner must first acquire both source and handoff
authority for the alternatives under consideration.

The completion report carries bounded resource values: admitted decode bytes,
queue depth, queue parallelism, TENT inflight bytes and TENT bandwidth. They
live in a 128-entry process-local freshness cache for at most two seconds and
are not exported as labels. Decode-route comparison accounts for queued waves
and RDMA rail pressure only when every candidate has fresh, shape-compatible
evidence; TCP legitimately reports an empty TENT NIC rail.

Raw GPU-copy keys retain separate logarithmic buckets for input descriptors and
DMA-coalesced ranges. Actual execution samples and shadow candidates use the
same shape from the validated copy list; the executor's merge iterator supplies
the range count. Refining that key does not restart queue/admission timing.
These fields are bounded estimator dimensions, not additional metric labels.

Shadow compares the existing load DMA/kernel pair, save DMA/kernel pair,
complete io_uring/cuFile restore pair, and equal-coverage HostReady alternatives.
The first three require matching resource identity. HostReady shadow permits
different SSD-store/peer resources but still requires known representation and
identical byte/block shape. Multi-owner peer prefixes have no single resource
identity and are excluded. Matching buckets never authorize a source or prove
resource admission. SSD GPU-restore keys additionally retain destination GPU,
copy backend, source-store set and mixed DRAM presence.

`different` requires the alternative's mean plus empirical error to beat the
current mean minus its error by more than 5% of the current mean. A faster mean
that fails this guard yields `within_margin`; a fastest or tied current mean
yields `agree`. Missing samples yield `unknown`, and incompatible comparison
families/resources/shapes or unknown representation yield `incomparable` when
all estimates are present. The 5% margin is a declared shadow threshold, not a
qualified execution policy or a statistical confidence bound.

The bounded SSD-route key includes buckets for the complete stored source image's
bytes/fragments and the SSD-derived portion of target bytes/fragments, alongside
the full task shape, representation and shared SSD/device identity. These fields
are not metric labels. Stored source sizes do not turn padded raw bytes into
known logical data. Composite route totals and nested operation timers overlap;
physical bytes are reported only by their existing I/O owners. Do not sum either
parent/child timings or logical bytes across those paths.

This family and its shadow work are disabled by default. Enable them with
`ORBITKV_COST_OBSERVATIONS=1` before Manager startup. Existing operational
metrics remain enabled independently. See the
[overhead gate](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#p41-final-evidence) before enabling cost
observations in serving.
Execution selection additionally requires `ORBITKV_COST_SELECTION=1`. It is
currently limited to equal-coverage owners of one peer medium; it does not use
nested authorization/READ timers as additive costs.
Cross-medium execution additionally requires
`ORBITKV_CROSS_MEDIUM_SELECTION=1`. The same `orbitkv_cost_route_decisions_total`
reports its guarded decision; deployment labels and external transport/NIC
evidence remain necessary to interpret the result.

### Storage encoding

See [codec qualification](storage-formats.md#qualification). `orbitkv_storage_codec_bytes_total`
labels `logical` and `stored` count encoded publications; stored includes per-segment
alignment and mixed raw segments. `orbitkv_storage_codec_transfer_bytes_total`
labels `d2h`/`h2d` count actual codec-path payload transfers, including CPU fallback.
`orbitkv_storage_codec_reserved_bytes` holds active codec operation reservations
through completion. `orbitkv_storage_codec_workspace_bytes` measures retained
GPU arenas, including idle workers and SSD decoder inputs; they are released at
worker teardown. `orbitkv_storage_codec_workspace_allocations_total` counts arena
allocations and growth, while `orbitkv_storage_codec_batches_total` and the
`orbitkv_storage_codec_batch_segments` histogram report actual batch sizes.
Budgets apply per worker, not across the whole Manager. Duration and decode-failure metrics
track complete codec transfers, corrupt objects rejected before admission and
failed GPU codec restores.
SSD prefetch byte counters now count stored physical segment bytes, including
encoded payloads and padding. GPU load counters count reconstructed logical bytes.
Codec transfer counters cover saves through the codec worker and restores of
encoded slots, including their raw sibling segments; raw-only restores use the
ordinary copy backend and are excluded from the codec counter.
GDS counters apply only to cuFile operations and do not prove native GDS.

### Query ownership and preparation

- **orbitkv_query_reserved_bytes** tracks total conservative per-owner bytes.
  Admission and release update this unlabelled counter under the budget lock;
  use it to check the configured query budget. Shared physical pages may be
  counted for several owners. Use the pool metric for actual allocator occupancy.
- **orbitkv_query_speculative_reserved_bytes** tracks the speculative share under
  the same lock; compare it with one quarter of the query budget.
- **orbitkv_query_reserved_peak_bytes** retains the maximum concurrent conservative
  reservation since this Manager's budget was created, including warm-up.
- **orbitkv_query_instance_reserved_peak_bytes** retains the largest concurrent
  reservation of any single instance over that same lifetime. It is neither the
  sum of per-instance historical peaks nor a per-engine labelled series. Both
  peaks are updated under the admission owner's existing lock, survive release
  and unregister, and reset only with a new budget. Scrapes read atomic peaks
  without acquiring the admission lock. Recording uses atomics, without emitting
  request events or retaining instance labels.
  They measure ownership charges, not physical allocator/HBM usage. The pressure
  gate rejects missing/reset peaks and checks both configured byte limits.
- **orbitkv_query_reserved_bytes_by_phase** separates `warming`, `preloading`,
  `prepared`, `preparing`, `ready`, and `restoring`. Phase samples are diagnostic:
  collection can overlap a transition, so their sum is not an atomic budget
  snapshot. Owned preparation moves from `preloading` to `prepared`, then
  into foreground ownership on claim without releasing its total reservation.
  Unowned warming retains no ready lease; its bytes return to zero after
  preparation even without polling.
- **orbitkv_query_budget_waits_total** and **orbitkv_query_budget_bypasses_total**
  include warmup admission attempts. A skipped warmup does not imply the later
  demand query will bypass restoration.
- **orbitkv_query_coalesced_reads_total** counts owners joining an identical
  backing-read plan. It is not a byte-savings counter.
- **orbitkv_warmup_prepared_bytes_total**, **orbitkv_warmup_restored_bytes_total**,
  **orbitkv_warmup_unused_bytes_total**, and **orbitkv_warmup_pending_bytes**
  follow each speculative-origin page through successful local H2D or last-owner
  release. A query hit is not use; restored bytes count unique page footprints,
  not individual layer copies or exact H2D traffic. Both unowned warming and
  owned preparation use these physical-page counters; compare them in separate runs.
- **orbitkv_warmup_wait_byte_seconds_total** records byte-weighted time from
  preparation to first successful H2D or unused release, labelled by `outcome`.
  Unresolved live intervals are excluded. See the
  [accounting contract](queued-warming.md#measuring-whether-preparation-was-useful).
- **orbitkv_warmup_foreground_skips_total** counts unowned warming hints skipped because
  foreground preparation, leases or GPU transfers already own query bytes.
  Owned preparation can overlap foreground work within its speculative share
  and total byte limits. See [request preparation](request-preparation.md).

Optional [request timelines](queued-warming.md#observing-the-path) correlate
enqueue, preparation, restore and engine consumption. Cache-tier probe counts
include warmups and demand; they are not end-user request hit rates.

### Pool Metrics (Pinned Memory)
- **orbitkv_pool_used_bytes** (Gauge)
  - Current pinned memory pool usage in bytes
  - Use case: Monitor memory pressure

- **orbitkv_pool_capacity_bytes** (Gauge)
  - Total pinned memory pool capacity in bytes
  - Use case: Derive pool utilization

- **orbitkv_pool_largest_free_bytes** (Gauge)
  - Largest contiguous free region in pinned pool (fragmentation signal)
  - Use case: Distinguish true exhaustion vs fragmentation (largest_free << free_bytes)

- **orbitkv_pool_alloc_failures_total** (Counter)
  - Total allocation failures after eviction retries
  - Use case: Detect memory exhaustion issues

### Cache Metrics (Block-level)
- **orbitkv_cache_candidate_hits_total**, **orbitkv_cache_candidate_misses_total** (Counters)
  - Metadata-only recovery discovery across attention and auxiliary groups
  - Attention counts the available contiguous prefix; auxiliary groups count
    independently available positions. The remaining positions count as misses.
  - Candidates hold no leases and can become stale. These counters describe
    planning availability, not completed reads, transferred bytes, or GPU reuse.

- **orbitkv_cache_block_hits_total** (Counter)
  - Blocks returned by terminal prefix reads, including warmup reads
  - Excludes metadata-only candidate discovery; does not prove GPU transfer

- **orbitkv_cache_block_misses_total** (Counter)
  - Missing suffix blocks reported by terminal prefix reads
  - Use candidate-miss counters for recovery plans rejected before payload reads

- **orbitkv_cache_tier_block_requests_total** (Counter)
  - Per-decision `query_prefetch` block attribution by `tier`
    (`ram`, `remote`, `ssd`, or `miss`)
  - Use case: Calculate overall hit ratio and each cache tier's contribution
    from one consistent denominator
  - Invariant: for each attributed decision,
    `ram + remote + ssd + miss == block_hashes.len()`
  - Semantics: this is decision attribution. `tier="remote"` and `tier="ssd"`
    mean the block was selected to be satisfied by that backing tier; they do
    not guarantee the later backing operation succeeded.

- **orbitkv_cache_block_insertions_total** (Counter)
  - New blocks inserted into cache
  - Use case: Track cache growth

- **orbitkv_cache_block_evictions_total** (Counter)
  - Blocks evicted from cache due to memory pressure
  - Use case: Monitor eviction frequency, tune pool size

- **orbitkv_cache_block_evictions_by_class_total** (Counter)
  - Blocks evicted from cache due to memory pressure, labelled by replacement class (`reclaimable`, `probationary` or `retained`)
  - Use case: Verify remote-fetched replicas are reclaimed before locally produced blocks

- **orbitkv_cache_block_evictions_still_referenced_total** (Counter)
  - Evicted blocks that still had external references (eviction did not immediately reclaim pinned memory)
  - Use case: Explain "evictions spike but pool_used_bytes doesn't drop"

- **orbitkv_cache_eviction_reclaimed_bytes_total** (Counter)
  - Estimated bytes actually reclaimed in pinned allocator after cache eviction
  - Use case: Measure effectiveness of eviction under real reference patterns

- **orbitkv_cache_resident_blocks** (Gauge)
  - Current number of sealed blocks resident in cache, labelled by replacement class (`reclaimable`, `probationary` or `retained`)
  - Use case: Track cache size and source-based replacement pressure in blocks

- **orbitkv_cache_resident_bytes** (Gauge)
  - Current sealed block bytes resident in cache (sum of footprints)
  - Use case: Attribute pinned pool usage to cache residency

- **orbitkv_cache_protected_bytes** (Gauge)
  - Bytes in the protected segment when `--cache-protected-percent` is enabled;
    bounded by that percentage of pool capacity. Zero with protection disabled.
    These bytes are part of residency, not an additional pool reservation.

- **orbitkv_cache_policy_promotions_total**, **orbitkv_cache_policy_demotions_total** (Counters)
  - Foreground promotions into the protected segment and capacity demotions
    back to probation. Speculative peeks do not promote. Pressure eviction and
    cleanup affect protected bytes but are not policy-capacity demotions.

- **orbitkv_ssd_write_admission_skips_total** (Counter)
  - Pages skipped before the SSD write queue, by `reason`: `cold`, `resident`,
    `pending` or `duplicate`. Intentional policy decisions, separate from
    `orbitkv_ssd_write_queue_full_total` pressure drops. See [policies](cache-policies.md).

- **orbitkv_cache_residence_duration_seconds** (Histogram)
  - RAM resident block lifetime from its first successful cache insertion to
    removal, measured in seconds
  - Labels: `reason` (`pressure` or `cleanup`)
  - `pressure`: allocator pressure removed the block through LRU reclaim
  - `cleanup`: the memory-cache cleanup endpoint removed the block
  - Buckets: `1s`, `5s`, `10s`, `30s`, `1m`, `2m`, `5m`, `10m`, `30m`,
    `1h`, `2h`, `6h`, `12h`, `24h`, and `+Inf`
  - Use case: Track typical and tail cache residence time and visualize the
    eviction-age distribution
  - Scope: each RAM residence is a separate lifetime. A block reinserted after
    eviction starts a new lifetime; cache hits, duplicate inserts, and
    replacement-class changes do not reset the original insertion time.
  - The lifetime ends when the block leaves the resident cache, even if an
    outstanding `Arc` keeps its pinned memory allocated. Correlate
    `orbitkv_cache_block_evictions_still_referenced_total` and
    `orbitkv_cache_eviction_reclaimed_bytes_total` to diagnose delayed memory
    reclamation.
  - SSD ring-cache overwrite and blocks still resident at server shutdown are
    not observed.

- **orbitkv_pinned_for_load_entries** (Gauge)
  - Current number of pinned_for_load entries (instance_id, block_key)
  - Use case: Diagnose load-path pins keeping evicted blocks alive

- **orbitkv_pinned_for_load_refs** (Gauge)
  - Current outstanding pinned_for_load consumer refcount (sum of per-entry counts)
  - Use case: Detect stuck consumers / missing release on load path

- **orbitkv_pinned_for_load_unique_blocks** (Gauge)
  - Current number of unique blocks referenced by pinned_for_load
  - Use case: Understand how many distinct blocks are being kept alive by pins

- **orbitkv_pinned_for_load_unique_bytes** (Gauge)
  - Current bytes referenced by pinned_for_load (unique blocks; sum of footprints)
  - Use case: Attribute pinned pool usage to load-path pins

### HLL Reuse Metrics
- **orbitkv_hll_cardinality** (Gauge)
  - Estimated distinct `(namespace, block hash)` objects classified as misses in a configured sliding window
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Derive approximate prefix reuse over longer windows without
    storing every block hash

- **orbitkv_hll_total_requests** (Gauge)
  - Total queried blocks in the same configured sliding window, including ready blocks and duplicates
  - Hybrid recovery records attention-prefix discovery once. Materializing the
  selected ranges does not count the same lookup again; auxiliary groups and
    speculative preparation do not enter this denominator. A dense prepared
    claim counts the ordinary lookup once. Queries stopped before finishing
    their evidence, including finite-batch prepared claims, do not enter this
    estimate: an unread page is unknown, not a demonstrated miss.
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Denominator for HLL-based reference reuse rate

- **orbitkv_hll_estimated_hit_rate** (Gauge)
  - Server-computed miss-based infinite-cache reuse reference from the same
    HLL snapshot as the two gauges above
  - Labels: `window`
  - Value is clamped to `[0, 1]`

The existing cardinality and total metrics are retained. Existing PromQL
continues to work, but new dashboards should prefer the direct gauge because
it applies the same cardinality clamp as the tracker:

```promql
1 - (
  orbitkv_hll_cardinality{window="1h"}
  /
  clamp_min(orbitkv_hll_total_requests{window="1h"}, 1)
)
```

```promql
orbitkv_hll_estimated_hit_rate{window="1h"}
```

This is a metrics semantic update, not an `/metrics` protocol breaking change:
metric names, types, existing labels, and the HTTP endpoint are unchanged;
the new gauge is additive. The default HLL size changes from 16,384 registers
(`bucket_bits=14`, about 0.8% standard error) to 65,536 registers
(`bucket_bits=16`, about 0.4%). Three default windows use about 192 KiB of
register storage; sliding slots make the live tracker a few MiB per server.
The setting remains configurable with `--metric-hll-bucket-bits`.

### Save Metrics (GPU → CPU)
- **orbitkv_save_bytes_total** (Counter)
  - Logical bytes published from GPU pages, before storage encoding
  - Use case: Monitor save throughput

- **orbitkv_save_duration_seconds** (Histogram)
  - Save operation latency distribution
  - Use case: Track save performance (p50, p99)

### Load Metrics (cache → GPU)
- **orbitkv_load_bytes_total** (Counter)
  - Logical bytes restored from DRAM or SSD into engine GPU pages
  - Use case: Monitor load throughput

- **orbitkv_load_duration_seconds** (Histogram)
  - Manager-worker restore duration, including SSD reads on the selected cuFile or explicit io_uring restore lane; engine-local caller-to-drain samples are included only when timing is enabled in the engine. Manager retirement time is never used for local samples.
  - Use case: Track load performance (p50, p99)

- **orbitkv_load_failures_total** (Counter)
  - Load operation failures (e.g., transfer errors)
  - Use case: Detect data transfer issues

### SSD Cache Metrics

- **orbitkv_ssd_backend_fallbacks_total** (Counter) - Automatic cuFile admission
  disabled after initialization or operation failure. The default policy uses
  io_uring thereafter; an explicit cuFile read route does not silently change.
  Logs record the reason.
- **orbitkv_ssd_cufile_write_bytes_total** (Counter) - Physical cuFile write bytes, including padding.
- **orbitkv_ssd_cufile_write_seconds** (Histogram) - Async submission-to-completion latency, including GPU gather and polling; excludes waiting for a slot.
- **orbitkv_ssd_cufile_write_failures_total** (Counter) - Failed or short GPU-backed writes.
- **orbitkv_ssd_cufile_read_bytes_total** (Counter) - Physical cuFile read bytes,
  including aligned edges. Does not distinguish native GDS from CPU compatibility.
- **orbitkv_ssd_cufile_read_seconds** (Histogram) - Async submission-to-completion latency, including GPU scatter and polling; excludes waiting for a slot.
- **orbitkv_ssd_cufile_read_failures_total** (Counter) - Failed or short cuFile reads.
- **orbitkv_ssd_read_pinned_bytes** (Gauge) - SSD extents owned by restore leases.
- **orbitkv_ssd_gpu_staging_bytes** (Gauge) - Registered GPU storage staging memory.
- **orbitkv_ssd_cufile_inflight_batches** (Gauge) - Occupied staging slots until I/O and scatter completion, at most two per instance/device.
- **orbitkv_ssd_gpu_write_fallbacks_total** (Counter) - Write jobs using host publication after the eight-job GPU write admission limit is reached.
- **orbitkv_ssd_gpu_read_fallbacks_total** (Counter) - Automatically selected cuFile restores changed to io_uring before submission because another instance owns this CUDA device's persistent staging. Explicit cuFile routes do not increment this counter or switch paths.
- **orbitkv_ssd_pinned_write_skips_total** (Counter) - Reservations rejected to
  protect an active SSD read or write. See [GPU storage recovery](gds.md).
- **orbitkv_ssd_write_bytes_total** (Counter) - Bytes written to SSD cache
- **orbitkv_ssd_write_duration_seconds** (Histogram) - Per-block io_uring write submission
  and completion latency; cuFile writes use their own duration histogram. Concurrent block writes overlap; summing these
  durations does not measure wall-clock flush time.
- **orbitkv_ssd_prefetch_bytes_total** (Counter) - Successfully read and
  validated SSD bytes, including reads an engine may not subsequently consume.
- **orbitkv_ssd_uring_read_operations_total** (Counter) - READV SQEs successfully
  queued by the existing io_uring shards. Counts software I/O requests, including
  ones that later fail; it does not count logical blocks, reader batches or NVMe
  device commands. cuFile reads are separate. Compare with stored read bytes,
  failures and GPU load bytes before interpreting a reduction.
- **orbitkv_ssd_prefetch_success_total** (Counter) - Successful SSD prefetches
- **orbitkv_ssd_prefetch_failures_total** (Counter) - Failed SSD prefetches
- **orbitkv_ssd_prefetch_duration_seconds** (Histogram) - Prefix prefetch or
  explicitly leased host-read latency, including queueing, pinned-memory allocation,
  reads, and block reconstruction. Excludes H2D and does not prove a successful
  GPU restore; correlate with read bytes, failures, and load bytes.

### Tier Attribution Semantics

`orbitkv_cache_tier_block_requests_total{tier}` is the canonical metric for
explaining how each cache tier contributes to prefix-query hit ratio. It is
emitted once for each `query_prefetch` decision and uses exactly one label:
`tier`.

Tier values:

- `ram`: blocks already present in the resident RAM cache at the decision point
- `remote`: blocks selected to be satisfied by Mooncake remote fetch
- `ssd`: blocks selected to be satisfied by SSD prefetch
- `miss`: blocks no tier selected for that decision, including SSD prefetch
  backpressure and residual blocks after Mooncake partial availability

This metric intentionally records decisions, not completed service outcomes.
For backing failure correlation, use:

- `orbitkv_candidate_cache_lookups{result="hit|miss"}` counts key checks before
  lookup coalescing; `orbitkv_candidate_lookup_rpcs{result="ok|error|timeout"}` counts
  actual batched directory RPCs (an OK RPC can still contain a miss).
- `orbitkv_remote_fetch_total{status="rejected"}` counts source authorization
  rejection before payload submission; `status="error"` counts other fetch failures.
- `orbitkv_remote_fetch_plan_segments` includes attempted alternative-source
  segments; `orbitkv_remote_fetch_plan_completed_segments` counts completed ones.
- `orbitkv_tent_nic_inflight_bytes` and `orbitkv_tent_nic_bandwidth_bytes_per_second`
  sample TENT's live RDMA rail state after successful peer batches, labelled by
  NIC. They are resource-pressure evidence, not proof that the sampled batch
  used RDMA.
- `orbitkv_ssd_prefetch_failures_total` for SSD prefetch failures
- `orbitkv_remote_stage_duration_seconds{stage,status}` separates
  `authorization`, `allocation`, `read`, `rebuild` and `release`. Allocation/read/rebuild
  describe completed successful transfers.
  Authorization includes first-use window setup and its bounded single-flight wait.
  Release measures time from native completion/abandonment to acknowledgement,
  including retry delays.
- `orbitkv_transfer_reserved_bytes` counts whole source allocations once per
  session, including overdue holds. `orbitkv_transfer_expired_sessions` counts
  overdue sessions still retaining memory; `orbitkv_transfer_lock_timeouts_total`
  counts the transition once. Timeout is not a release condition.
- `orbitkv_transfer_completion_outstanding` counts requester slots from window setup
  through source release acknowledgement (1024 total, at most 64 per source incarnation).
  `orbitkv_transfer_completion_retries_total` counts failed release attempts;
  `orbitkv_transfer_completion_rejections_total` counts new authorizations skipped
  at capacity. Outstanding records must drain as well as source pins.
- `orbitkv_transfer_lock_rejections_total{reason="bytes|sessions"}` counts bounded
  source admission failures. Source pins and query reservations must both drain
  after successful remote restoration.

`orbitkv_cache_block_hits_total` and `orbitkv_cache_block_misses_total` count
terminal prefix reads. Metadata discovery has separate candidate counters, so a
cold hybrid lookup can record a candidate miss without issuing a payload read.
Use `orbitkv_cache_tier_block_requests_total{tier}` for read-tier decisions;
do not add candidate counters to that denominator. Verify actual GPU reuse with
`orbitkv_load_bytes_total`, not candidate or query hits alone.

## Configuration

### Cache Manager Parameters

**Metrics Parameters:**

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`)
  - Always enabled for health check at `/health`
  - `/metrics` is enabled by default

- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
  - When enabled, metrics are available at `http://<http-addr>/metrics`
  - Health check is always available at `http://<http-addr>/health`

- `--metrics-otel-endpoint`: OTLP gRPC endpoint for metrics export (optional)
  - Example: `http://127.0.0.1:4321`
  - Leave unset to disable OTLP export

- `--metrics-period-secs`: Metric export interval in seconds (default: `10`)
  - Only used when `--metrics-otel-endpoint` is set

- `--metric-hll-windows`: Comma-separated HLL sliding windows for estimated
  prefix reuse (default: `15m,1h,1d`)
  - Supported units: `s`, `m`, `h`, `d`
  - Each configured duration becomes a canonical `window` label. For example,
    the default config exports `window="15m"`, `window="1h"`, and `window="1d"`.
  - Empty entries such as `15m,,1h` and duplicate durations such as `1h,60m`
    are rejected at startup.

- `--metric-hll-bucket-bits`: HLL bucket index bits (default: `16`)
  - `2^16 = 65,536` registers per window and about 0.4% standard error.
  - Higher values use more memory and lower estimation error; `18` remains
    the supported maximum.

**Example: Prometheus Metrics**
```bash
cargo run -r --bin orbitkv-cache-manager -- \
  --addr 127.0.0.1:50055 \
  --devices 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

For an existing OpenTelemetry deployment, add
`--metrics-otel-endpoint http://127.0.0.1:4321` with the collector's configured
endpoint. Direct Prometheus remains available.

### Distributed metadata progress

`GET /cache/metadata` reports the local index's `membership_revision`, explicit
`coverage`, active/staging/accounted bytes, expected/installed owner views, view
generation, `scope_kind` and exact namespace count. Registration validity is
separate: an interrupted inventory stream can retain positive hints with
`partial_hints`, while invalid membership or a scope-outside lookup makes
coverage `unavailable`. A complete empty owner view differs from a missing view.

The same response includes the local inventory sequence, resident and retained-
journal record counts, current/peak journal bytes, capacity and history gaps.
`inventory_last_change_mono_ns` is the source process's `CLOCK_MONOTONIC`
timestamp for the latest committed residency sequence. It is a same-host
qualification diagnostic, not a cross-host clock or a publication barrier.
Completed stream intervals report input/coalesced records, encoded bytes, delta
frames and elapsed coalescing wait. `inventory_flush_through_sequence` is the
latest requested source-fence target; it is not a received-frame or installed-
view watermark. `inventory_scope_filter_input_records`,
`inventory_scope_filter_output_records` and `inventory_scope_filter_micros`
measure source-side filtering across snapshot/replay/delta work; lower output
counts alone are not a serving-performance result.

The `stream` object reports protocol/cluster/source identity, canonical scope
digest/kind/namespace count, membership revision, current/peak source and receiver
sessions, frames and encoded bytes sent/received, current/peak aggregate outbound
queue bytes and reset count. These are process-lifetime counters. Snapshot bytes
share the frame counters; etcd network metrics now represent
membership/configuration traffic, not block traffic. Namespace filtering does
not reduce the session count. The pre-cutover Watch payload counter is removed
with its sole producer.

Current session counts release through an abort-safe owner when membership
replacement cancels an old follower. After a normal receiver restart, current
source/receiver sessions must return to the actual live topology; peak retains
the process high-water mark. A growing current count with stable membership is a
session-lifecycle fault, not harmless telemetry drift.

`GET /cache/metadata/owners?after=<uuid>&limit=<1..128>` reports bounded,
UUID-ordered installed owner views with their committed sequence, freshness,
receipt age, record count and `installed_mono_ns`. The latter is the observer's
`CLOCK_MONOTONIC` timestamp at atomic snapshot commit or delta apply; an exact
progress confirmation at unchanged sequence preserves it. Compare it with a
source timestamp only when both
processes run in the same host clock domain and the sample also matches owner,
epoch, incarnation, scope digest and sequence. HTTP response time remains a
separate harness-observation delay. This endpoint is an operational/qualification
surface, not a request-path directory lookup.

Standalone Managers return JSON `null`. `POST /cache/sync` returns an
`inventory_fence`; `POST /cache/metadata/await` succeeds only after the matching
source incarnation/epoch and exact scope are installed through that sequence. A
wrong scope fails; an empty filtered interval advances only covered progress and
does not prove a scope-outside key absent. These are synchronization boundaries,
independent of source authorization and native payload completion.

The old per-shard Catalog gauges, inventory-RPC counters and candidate-lookup
metrics were removed with their runtime. Transfer, source budget, remote-stage
and release metrics remain. Repeated reset/reconnect warnings or incomplete
coverage indicate a stream, membership or index-budget fault; inspect peer
connectivity and etcd membership health. See [metadata recovery](distributed-cache.md).

### Environment Variables

- `RUST_LOG`: Control logging verbosity (e.g., `info,orbitkv_core=debug`)
- `ORBITKV_TRACE_TRANSFERS=1`: Enable opt-in timeline records.
- `ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT=<1..65536>`: Bound S2.10a channel/storage
  stage records; zero or omission disables those additional events.

## Quick Start: Direct Prometheus (Recommended)

The `examples/metric-prometheus/` directory provides a simple monitoring stack.

### 1. Start Cache Manager

```bash
# From repository root
cargo run -r --bin orbitkv-cache-manager -- \
  --addr 127.0.0.1:50055 \
  --devices 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

### 2. Start the Monitoring Stack

```bash
cd examples/metric-prometheus

docker compose up -d
# To stop: docker compose down
```

This starts two services:
- **Prometheus** (port: 9090) - Scrapes metrics from OrbitKV
- **Grafana** (port: 3000) - Visualizes metrics

### 3. Access Grafana Dashboard

1. Open browser: http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to **Dashboards** → **OrbitKV Metrics**

### 4. Test Metrics Endpoint

```bash
curl http://localhost:9091/metrics
```

## Architecture details

### Direct Prometheus

```
┌─────────────────┐
│ Cache Manager   │
│ :9091 /metrics  │
└────────┬────────┘
         │ Prometheus scrape
         ▼
┌─────────────────┐
│   Prometheus    │
│     :9090       │
└────────┬────────┘
         │ PromQL queries
         ▼
┌─────────────────┐
│    Grafana      │
│     :3000       │
└─────────────────┘
```

### Port Reference

| Service | Port or path | Protocol | Purpose |
| --- | --- | --- | --- |
| Cache Manager | `/tmp/orbitkv-<addr-port>.sock` | UDS and iceoryx2 | Inference process connection |
| Cache Manager | 50055 | gRPC | Peer source authorization and release in distributed mode |
| Cache Manager | 9091 | HTTP | Health and Prometheus metrics |
| OTel Collector | configured endpoint | gRPC | Optional OTLP receiver |
| Prometheus | 9090 | HTTP | Query API and Web UI |
| Grafana | 3000 | HTTP | Dashboard UI |

## PromQL Query Examples

```promql
# Overall cache hit ratio from decision attribution (last 5 minutes)
sum(rate(orbitkv_cache_tier_block_requests_total{tier!="miss"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# RAM contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="ram"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Remote contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="remote"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# SSD contribution to total requested blocks
sum(rate(orbitkv_cache_tier_block_requests_total{tier="ssd"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Miss ratio from the same denominator
sum(rate(orbitkv_cache_tier_block_requests_total{tier="miss"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Average save latency (p50)
histogram_quantile(0.5, rate(orbitkv_save_duration_seconds_bucket[5m]))

# Average load latency (p99)
histogram_quantile(0.99, rate(orbitkv_load_duration_seconds_bucket[5m]))

# Save throughput (MB/s)
rate(orbitkv_save_bytes_total[1m]) / 1e6

# Pool memory utilization
orbitkv_pool_used_bytes / orbitkv_pool_capacity_bytes

# RAM cache residence-time quantiles for pressure evictions
histogram_quantile(
  0.50,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.95,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.99,
  sum by (le) (
    rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

# RAM cache residence-time buckets for a Grafana heatmap
sum by (le) (
  rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
)

# HLL estimated hit rate for the 1h window (preferred)
orbitkv_hll_estimated_hit_rate{window="1h"}

# Backward-compatible derivation from the retained gauges
1 - (
  orbitkv_hll_cardinality{window="1h"}
  /
  clamp_min(orbitkv_hll_total_requests{window="1h"}, 1)
)

# HLL estimated hit rate for every configured window
1 - (
  orbitkv_hll_cardinality
  /
  clamp_min(orbitkv_hll_total_requests, 1)
)
```

## Troubleshooting

### Metrics not appearing (Direct Prometheus)

1. Check OrbitKV is exposing metrics:
   ```bash
   curl http://localhost:9091/metrics
   ```

2. Check Prometheus targets:
   - Open http://localhost:9090/targets
   - Verify `orbitkv` target is UP

3. If Docker cannot reach host, ensure `extra_hosts` is configured:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

## Best Practices

1. **Monitor tier-attributed hit ratio** using `orbitkv_cache_tier_block_requests_total`
   - Low hit rate → consider increasing `--pool-size`

2. **Watch eviction rate**: High evictions indicate memory pressure
   - Use `rate(orbitkv_cache_block_evictions_total[5m])`

3. **Track allocation failures**: Any failures indicate critical issues
   - Alert on `orbitkv_pool_alloc_failures_total > 0`

4. **Analyze latency distributions**: Use histogram quantiles
   - p50: Typical case performance
   - p99: Worst-case user experience

## References

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

The inventory publication/install clocks are same-host diagnostics for the latest
exact sequence. A progress confirmation at unchanged sequence preserves the
original install time; snapshot/delta commit updates it after the index becomes
complete under its write lock. A benchmark must reject superseded targets and
verify the host clock domain; these timestamps do not give a cross-host latency.
