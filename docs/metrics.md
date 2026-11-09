# Metrics and observability

## Overview

Scrape the Manager's `/metrics` with Prometheus; use
[`examples/metric-prometheus`](https://github.com/OrbitInfra/OrbitKV/blob/main/examples/metric-prometheus/) for Grafana.
Existing OpenTelemetry deployments can use
[`examples/metric`](https://github.com/OrbitInfra/OrbitKV/blob/main/examples/metric/) and `--metrics-otel-endpoint` instead.
Neither export path participates in cache decisions or proves physical drain.

## Available Metrics

Query, Publish, peer-ownership, ordinary SSD and load-byte readiness instruments
initialize to zero before first use. Optional codec/GDS/speculation series may
remain absent. Missing required gauges are not measured zeros; creating another
owner adds zero without resetting live values. All six query-phase labels are
initialized. Phase scrapes may race transitions and are not an atomic budget sum.

Save gauges `orbitkv_save_active_operations`, `orbitkv_save_active_metadata_bytes`,
`orbitkv_save_active_logical_bytes` and `orbitkv_save_retained_operations` track
the 1,024-operation / 32 MiB metadata admission owner. Completion or proven
non-submission releases charges; reply loss/unknown drain retains them. Logical
bytes include overlapping requested ranges and are not unique payload residency.
`orbitkv_save_admission_rejections_total` counts pre-worker budget refusals.

The vLLM exporter reports `vllm:orbitkv_save_dropped_total{reason="job_budget"}`
when the hybrid GPU Save-job cap drops an optional write before pinning pages.
The scheduler step continues; existing jobs retain their pins until drain.
The associated warning is limited to once per minute.

`orbitkv_peer_retained_source_pools_total` and
`orbitkv_peer_retained_source_pool_bytes_total` accumulate source pools retained
after failed drain, including partial registration. Full backing remains charged
against 1,024 pools / 1 TiB. These counters do not authorize reclamation.

### Bounded cost observations

Read [cost boundaries](architecture.md#planning-and-cost-observations) before combining
samples. Fixed path/outcome/stage labels exclude device, peer, request and state
identities. Internal resource keys retain those distinctions without metric
cardinality growth.

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

Composite SSD Restore and HostReady totals overlap their child copy/codec/file
operations. **Do not sum parent/child timers or bytes.** Logical sizes exclude
padding only when known; physical bytes are counted at the I/O owner, including
short I/O. Failed/unknown work cannot train a success estimate.

| Path | Start/end and interpretation |
| --- | --- |
| `ssd_uring_restore`, `ssd_cufile_restore` | Enqueue to GPU-terminal Restore, including queues and materialization/decode |
| `remote_authorization`, `remote_ssd_authorization` | Source grant; SSD includes source preparation |
| `remote_read` | OrbiTE payload READ owner |
| `peer_dram_host_ready`, `peer_ssd_host_ready`, `local_ssd_host_ready` | Whole host preparation through reconstruction; comparable only with equal coverage/shape |
| `cache_restore` | Manager preparation after page allocation to GPU completion |
| `engine_local_restore` | Native caller to local drain, accepted once per operation; not interchangeable with Manager or P/D starts |
| `prefill_to_decode_handoff` | Decode-owned enqueue to physical drain; only admitted completed reports train |

Engine-local collection needs `ORBITKV_COST_OBSERVATIONS=1` in both processes.
Complete-route shapes include destination GPU, representation, source stores and
mixed DRAM presence. Resource snapshots occupy a separate 128-entry/two-second
cache; they are not labels or reservations. OrbiTE owns coalescing, so
`dma_ranges=0` means unavailable. Historical TENT counts are not comparable.

Observation/shadow is off by default. `ORBITKV_COST_SELECTION=1` additionally
allows guarded same-medium peer choices; `ORBITKV_CROSS_MEDIUM_SELECTION=1`
requires both preceding flags for experimental cross-medium HostReady choice.
The [original overhead failure](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#p41-final-evidence)
remains unchanged. No prediction authorizes a source or selects an unobserved
route. The 5% switching margin is empirical, not statistical confidence.

`ORBITKV_TRACE_TRANSFERS=1` emits completion and consumer-observation records.
A positive `ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT` additionally enables channel/storage
stages, bounded to 65,536 total records per process. All records share a bounded
asynchronous writer; required stages are formatted off request threads. Match
session epoch/token and request ID on one host's monotonic clock.

A missing required stage, limit event, writer-start failure or abnormal exit
invalidates diagnostic linkage. Normal shutdown drains accepted work and storage
before flushing the writer. `publish_manager_response_publish` is before response
visibility; its client interval includes publication/wakeup. Save response means
copy/encode complete and insertion queued, not SSD durability. Full diagnostic
and overhead guards remain [frozen here](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/distributed-design.md#s210a-same-host-isolation-diagnostic-contract).

### Storage encoding

| Instrument | Meaning |
| --- | --- |
| `orbitkv_storage_codec_bytes_total` | `logical` / `stored` publication bytes; stored includes alignment and raw siblings |
| `orbitkv_storage_codec_transfer_bytes_total` | `d2h` / `h2d` payload, including CPU fallback and raw siblings in encoded jobs |
| `orbitkv_storage_codec_reserved_bytes` | Active operation reservations through completion |
| `orbitkv_storage_codec_workspace_bytes` | Retained GPU arenas, including idle workers and decoder inputs |
| `orbitkv_storage_codec_workspace_allocations_total` | Arena allocations/growth |
| `orbitkv_storage_codec_batches_total`, `orbitkv_storage_codec_batch_segments` | Actual batch counts/sizes |

Budgets are per worker. Duration/failure instruments cover complete codec work,
corrupt-input rejection and failed GPU restores. SSD reads count stored physical
bytes; GPU load counts reconstructed logical bytes. Raw-only copies are outside
codec transfer counters. GDS-named counters do not prove native GDS. See
[codec qualification](storage-formats.md#qualification).

### Query ownership and preparation

| Instrument | Meaning |
| --- | --- |
| `orbitkv_query_reserved_bytes` | Conservative per-owner total under the budget lock; shared pages may be charged several times |
| `orbitkv_query_speculative_reserved_bytes` | Speculative quarter-budget share |
| `orbitkv_query_reserved_peak_bytes` | Peak total since budget creation, including warm-up |
| `orbitkv_query_instance_reserved_peak_bytes` | Largest simultaneous charge of any single instance, not sum of historical peaks |
| `orbitkv_query_reserved_bytes_by_phase` | `warming`, `preloading`, `prepared`, `preparing`, `ready`, `restoring`; diagnostic phases |
| `orbitkv_query_budget_waits_total`, `orbitkv_query_budget_bypasses_total` | Admission pressure, including warming; not proof of a backing miss |
| `orbitkv_query_coalesced_reads_total` | Owners joining identical physical plans, not saved bytes |
| `orbitkv_warmup_prepared_bytes_total`, `orbitkv_warmup_restored_bytes_total`, `orbitkv_warmup_unused_bytes_total`, `orbitkv_warmup_pending_bytes` | Unique speculative-origin page footprints through successful use or last-owner release |
| `orbitkv_warmup_wait_byte_seconds_total` | Resolved byte-weighted preparation-to-use/release time by outcome; excludes live intervals |
| `orbitkv_warmup_foreground_skips_total` | Unowned hints skipped while foreground preparation/leases/transfers own query bytes |

Peaks survive release/unregister and reset only with a new budget. Atomics avoid
scrape locking and per-instance labels. Prepared claims move phases without
releasing total ownership; warming has no ready lease. A hit is not GPU use.
Warmup footprints are not individual layer-copy bytes; compare owned preparation
and unowned warming in separate runs. See [accounting](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/queued-warming.md#measuring-whether-preparation-was-useful)
and [request preparation](request-preparation.md).

### Pool Metrics (Pinned Memory)

- **orbitkv_pool_used_bytes** (Gauge): Current pinned memory pool usage in bytes

- **orbitkv_pool_capacity_bytes** (Gauge): Total pinned memory pool capacity in bytes

- **orbitkv_pool_largest_free_bytes** (Gauge): Largest contiguous free region in pinned pool (fragmentation signal)

- **orbitkv_pool_alloc_failures_total** (Counter): Total allocation failures after eviction retries

- **orbitkv_cache_candidate_hits_total**, **orbitkv_cache_candidate_misses_total** (Counters): Metadata-only recovery discovery across attention and auxiliary groups Attention counts the available contiguous prefix; auxiliary groups count independently available positions. The remaining positions count as misses. Candidates hold no leases and can become stale. These counters describe planning availability, not completed reads, transferred bytes, or GPU reuse.

- **orbitkv_cache_block_hits_total** (Counter): Blocks returned by terminal prefix reads, including warmup reads Excludes metadata-only candidate discovery; does not prove GPU transfer

- **orbitkv_cache_block_misses_total** (Counter): Missing suffix blocks reported by terminal prefix reads Use candidate-miss counters for recovery plans rejected before payload reads

- **orbitkv_cache_tier_block_requests_total** (Counter): Per-decision `query_prefetch` block attribution by `tier` (`ram`, `remote`, `ssd`, or `miss`) Invariant: for each attributed decision, `ram + remote + ssd + miss == block_hashes.len()` Semantics: this is decision attribution. `tier="remote"` and `tier="ssd"` mean the block was selected to be satisfied by that backing tier; they do not guarantee the later backing operation succeeded.

- **orbitkv_cache_block_insertions_total** (Counter): New blocks inserted into cache

- **orbitkv_cache_block_evictions_total** (Counter): Blocks evicted from cache due to memory pressure

- **orbitkv_cache_block_evictions_by_class_total** (Counter): Blocks evicted from cache due to memory pressure, labelled by replacement class (`reclaimable`, `probationary` or `retained`)

- **orbitkv_cache_block_evictions_still_referenced_total** (Counter): Evicted blocks that still had external references (eviction did not immediately reclaim pinned memory)

- **orbitkv_cache_eviction_reclaimed_bytes_total** (Counter): Estimated bytes actually reclaimed in pinned allocator after cache eviction

- **orbitkv_cache_resident_blocks** (Gauge): Current number of sealed blocks resident in cache, labelled by replacement class (`reclaimable`, `probationary` or `retained`)

- **orbitkv_cache_resident_bytes** (Gauge): Current sealed block bytes resident in cache (sum of footprints)

- **orbitkv_cache_protected_bytes** (Gauge): Bytes in the protected segment when `--cache-protected-percent` is enabled; bounded by that percentage of pool capacity. Zero with protection disabled. These bytes are part of residency, not an additional pool reservation.

- **orbitkv_cache_policy_promotions_total**, **orbitkv_cache_policy_demotions_total** (Counters): Foreground promotions into the protected segment and capacity demotions back to probation. Speculative peeks do not promote. Pressure eviction and cleanup affect protected bytes but are not policy-capacity demotions.

- **orbitkv_ssd_write_admission_skips_total** (Counter): Pages skipped before the SSD write queue, by `reason`: `cold`, `resident`, `pending` or `duplicate`. Intentional policy decisions, separate from `orbitkv_ssd_write_queue_full_total` pressure drops. See [policies](cache-policies.md).

- **orbitkv_cache_residence_duration_seconds** (Histogram): RAM resident block lifetime from its first successful cache insertion to removal, measured in seconds Labels: `reason` (`pressure` or `cleanup`) `pressure`: allocator pressure removed the block through LRU reclaim `cleanup`: the memory-cache cleanup endpoint removed the block Buckets: `1s`, `5s`, `10s`, `30s`, `1m`, `2m`, `5m`, `10m`, `30m`, `1h`, `2h`, `6h`, `12h`, `24h`, and `+Inf` Scope: each RAM residence is a separate lifetime. A block reinserted after eviction starts a new lifetime; cache hits, duplicate inserts, and replacement-class changes do not reset the original insertion time. The lifetime ends when the block leaves the resident cache, even if an outstanding `Arc` keeps its pinned memory allocated. Correlate `orbitkv_cache_block_evictions_still_referenced_total` and `orbitkv_cache_eviction_reclaimed_bytes_total` to diagnose delayed memory reclamation. SSD ring-cache overwrite and blocks still resident at server shutdown are not observed.

- **orbitkv_pinned_for_load_entries** (Gauge): Current number of pinned_for_load entries (instance_id, block_key)

- **orbitkv_pinned_for_load_refs** (Gauge): Current outstanding pinned_for_load consumer refcount (sum of per-entry counts)

- **orbitkv_pinned_for_load_unique_blocks** (Gauge): Current number of unique blocks referenced by pinned_for_load

- **orbitkv_pinned_for_load_unique_bytes** (Gauge): Current bytes referenced by pinned_for_load (unique blocks; sum of footprints)

### HLL Reuse Metrics

| Gauge (`window` label) | Meaning |
| --- | --- |
| `orbitkv_hll_cardinality` | Estimated distinct namespace/hash misses |
| `orbitkv_hll_total_requests` | Queried blocks, including duplicates and ready blocks |
| `orbitkv_hll_estimated_hit_rate` | Same-snapshot miss-based infinite-cache reuse reference, clamped to [0,1] |

Hybrid attention discovery counts once; auxiliary groups and speculative reads
do not enter this denominator. Unread/unfinished evidence is unknown, not a miss.
Prefer the direct hit-rate gauge. Defaults are `15m,1h,1d` and 65,536 registers
(`bucket_bits=16`, about 0.4% error); sliding slots use a few MiB. Configure with
`--metric-hll-windows` / `--metric-hll-bucket-bits` (maximum 18). Empty or duplicate
windows reject startup. This estimate is not observed finite-cache service success.

### Save Metrics (GPU → CPU)

- `orbitkv_save_bytes_total`: logical GPU publication bytes before encoding.
- `orbitkv_save_duration_seconds`: Save latency histogram.

### Load Metrics (cache → GPU)

- `orbitkv_load_bytes_total`: logical restored bytes.
- `orbitkv_load_duration_seconds`: Manager-worker duration; engine-local
  caller-to-drain samples enter only with native timing enabled. Reaping is not
  local completion time.
- `orbitkv_load_failures_total`: failed restores.

### SSD Cache Metrics

- **orbitkv_ssd_backend_fallbacks_total** (Counter) - Automatic GPU-layout admission
  disabled after an operation failure. The default policy uses
  engine host materialization thereafter; an explicit cuFile read route does not silently change.
  Logs record the reason.
- **orbitkv_ssd_cufile_write_bytes_total** (Counter) - Completed GPU file-route bytes, including padding; the historical name does not imply native cuFile/GDS.
- **orbitkv_ssd_cufile_write_seconds** (Histogram) - Async submission-to-completion latency, including engine-owned gather and polling; excludes waiting for a slot.
- **orbitkv_ssd_cufile_write_failures_total** (Counter) - Failed or short GPU-backed writes.
- **orbitkv_ssd_cufile_read_bytes_total** (Counter) - Completed GPU file-route bytes,
  including aligned edges. The historical name covers engine CUDA staging and GDS.
- **orbitkv_ssd_cufile_read_seconds** (Histogram) - Async submission-to-completion latency, including engine-owned scatter and polling; excludes waiting for a slot.
- **orbitkv_ssd_cufile_read_failures_total** (Counter) - Failed or short GPU file-route reads.
- **orbitkv_ssd_read_pinned_bytes** (Gauge) - SSD extents owned by restore leases.
- **orbitkv_ssd_cufile_inflight_batches** (Gauge) - Outstanding file receipts through I/O completion, at most two per instance/device.
- **orbitkv_ssd_gpu_write_fallbacks_total** (Counter) - Write jobs using host publication after the eight-job GPU write admission limit is reached.
- **orbitkv_ssd_gpu_read_fallbacks_total** (Counter) - Automatically selected cuFile restores changed to io_uring before submission because another instance owns this CUDA device's persistent staging. Explicit cuFile routes do not increment this counter or switch paths.
- **orbitkv_ssd_pinned_write_skips_total** (Counter) - Reservations rejected to
  protect an active SSD read or write. See [GPU storage recovery](gds.md).
- **orbitkv_ssd_engine_retained_operations_total** (Counter) - File operations retaining registrations, backing, extents and admission after uncertain drain or cleanup. These owners remain charged until process exit.
- **orbitkv_ssd_write_bytes_total** (Counter) - Bytes written to SSD cache
- **orbitkv_ssd_write_duration_seconds** (Histogram) - Per-block io_uring write submission
  and completion latency; cuFile writes use their own duration histogram. Concurrent block writes overlap; summing these
  durations does not measure wall-clock flush time.
- **orbitkv_ssd_prefetch_bytes_total** (Counter) - Successfully read and
  validated SSD bytes, including reads an engine may not subsequently consume.
- **orbitkv_ssd_engine_read_batches_total** (Counter) - Successful, physically
  drained engine file READ batches with exact byte counts. Replaces the removed
  OrbitKV `ssd_uring_read_operations` SQE counter. It does not measure engine
  slicing/merging, kernel SQEs or NVMe commands; historical counts are not comparable.
- **orbitkv_ssd_prefetch_success_total** (Counter) - Successful SSD prefetches
- **orbitkv_ssd_prefetch_failures_total** (Counter) - Failed SSD prefetches
- **orbitkv_ssd_prefetch_duration_seconds** (Histogram) - Prefix prefetch or
  explicitly leased host-read latency, including queueing, pinned-memory allocation,
  reads, and block reconstruction. Excludes H2D and does not prove a successful
  GPU restore; correlate with read bytes, failures, and load bytes.

### Tier Attribution Semantics

`orbitkv_cache_tier_block_requests_total{tier}` counts one decision per requested
block: `ram`, `remote`, `ssd` or `miss`, summing to the query's hash count. It
attributes selected sources, not successful backing I/O. Candidate discovery
counters hold no leases; terminal read hits still do not prove GPU reuse.
Correlate load bytes and failures rather than adding these denominators.

| Instrument | Interpretation |
| --- | --- |
| `orbitkv_remote_fetch_total{status}` | `rejected` is source denial before submission; `error` is other fetch failure |
| `orbitkv_remote_fetch_plan_segments`, `orbitkv_remote_fetch_plan_completed_segments` | Attempted alternatives versus completed segments |
| `orbitkv_remote_stage_duration_seconds{stage,status}` | Authorization (including setup), allocation, read, rebuild and release-to-ACK/retry latency |
| `orbitkv_orbite_nic_inflight_bytes`, `orbitkv_orbite_nic_bandwidth_bytes_per_second` | Engine RDMA load by NIC after successful batches; not proof of that batch's transport |
| `orbitkv_transfer_reserved_bytes` | Whole source allocations charged once per session, including overdue holds |
| `orbitkv_transfer_expired_sessions`, `orbitkv_transfer_lock_timeouts_total` | Retained overdue sessions and their once-only timeout transitions |
| `orbitkv_transfer_completion_outstanding` | Requester slots through source release ACK: 1,024 total / 64 per source incarnation |
| `orbitkv_transfer_completion_retries_total`, `orbitkv_transfer_completion_rejections_total` | Failed releases / new authorizations skipped at capacity |
| `orbitkv_transfer_lock_rejections_total{reason="bytes|sessions"}` | Source-budget refusal |

Timeout never releases memory. Requester completion records and source pins must
both drain. Removed Catalog RPC and TENT native-batch counters have no current
producer; see the [historical metric reference](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/metrics.md#tier-attribution-semantics)
for old captures.

## Configuration

### Cache Manager Parameters

| Flag | Default / purpose |
| --- | --- |
| `--http-addr` | `0.0.0.0:9091`; health always available |
| `--enable-prometheus` | `true`; exposes `/metrics` |
| `--metrics-otel-endpoint` | Unset; optional OTLP gRPC endpoint |
| `--metrics-period-secs` | 10; OTLP export interval only |
| `--metric-hll-windows` | `15m,1h,1d`; durations use s/m/h/d |
| `--metric-hll-bucket-bits` | 16; maximum 18 |

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

- `RUST_LOG`: logging verbosity, for example `info,orbitkv_core=debug`.
- `ORBITKV_TRACE_TRANSFERS=1`: opt-in transfer timeline.
- `ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT=<1..65536>`: bounded additional stages;
  zero/omission disables them. Cost flags are described above.

## Quick Start: Direct Prometheus (Recommended)

```bash
orbitkv-cache-manager --pool-size 30gb --http-addr 0.0.0.0:9091
curl --fail http://localhost:9091/metrics
cd examples/metric-prometheus
docker compose up -d
```

Prometheus runs on 9090 and Grafana on 3000. The example Grafana login is
`admin` / `admin`, dashboard **OrbitKV Metrics**. Stop the example with
`docker compose down`. OTLP users point the Manager at their configured collector;
Prometheus can remain enabled. See [Manager options](server.md).

## Architecture details

Prometheus scrapes the Manager directly, or an OTLP collector exports to it;
Grafana queries Prometheus. These are observation paths only.

### Port Reference

| Service | Port or path | Purpose |
| --- | --- | --- |
| Cache client | `/tmp/orbitkv-<addr-port>.sock` | UDS/iceoryx2 lifecycle and operations |
| Manager peer | 50055 | Distributed inventory, source grants and release |
| Manager HTTP | 9091 | Health and metrics |
| OTLP | Configured | Optional gRPC receiver |
| Prometheus / Grafana | 9090 / 3000 | Queries / dashboard |

## PromQL Query Examples

```promql
# Decision hit ratio; use tier="ram", "remote" or "ssd" for one contribution
sum(rate(orbitkv_cache_tier_block_requests_total{tier!="miss"}[5m])) /
sum(rate(orbitkv_cache_tier_block_requests_total[5m]))

# Save p50 and load p99 (quantiles, not averages)
histogram_quantile(0.5, sum by (le) (rate(orbitkv_save_duration_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(orbitkv_load_duration_seconds_bucket[5m])))

# Save MB/s and pool utilization
rate(orbitkv_save_bytes_total[1m]) / 1e6
orbitkv_pool_used_bytes / orbitkv_pool_capacity_bytes

# Pressure eviction residence p95; use the bucket expression for a heatmap
histogram_quantile(0.95, sum by (le) (
  rate(orbitkv_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
))

orbitkv_hll_estimated_hit_rate{window="1h"}
```

## Troubleshooting

Check `/metrics`, then Prometheus `/targets`. Docker examples may need
`extra_hosts: ["host.docker.internal:host-gateway"]` to reach the host.

Pool allocation failures or high evictions need capacity/fragmentation diagnosis;
evictions without reclaimed bytes can reflect retained consumers. Use query/source
ownership gauges to distinguish charged work from resident cache. Histograms
report distributions; p99 is not a worst-case bound. Inspect retained operations
and actual drain rather than resetting gauges to conceal them.

## References

- [Prometheus query language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
- [OpenTelemetry](https://opentelemetry.io/docs/)
