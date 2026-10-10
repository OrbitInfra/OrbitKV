use opentelemetry::{
    KeyValue, global,
    metrics::{Counter, Histogram, Meter, UpDownCounter},
};
use std::sync::{LazyLock, OnceLock};

// ---------------------------------------------------------------------------
// Tier-attribution label sets for `cache_tier_block_requests`.
//
// Stored as `LazyLock<[KeyValue; 1]>` so the hot path passes a `&[KeyValue]`
// slice without rebuilding the attribute on every counter add.
// ---------------------------------------------------------------------------

static TIER_RAM: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "ram")]);
static TIER_REMOTE: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "remote")]);
static TIER_SSD: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "ssd")]);
static TIER_MISS: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "miss")]);
pub(crate) static CACHE_CLASS_RECLAIMABLE: LazyLock<[KeyValue; 1]> =
    LazyLock::new(|| [KeyValue::new("class", "reclaimable")]);
pub(crate) static CACHE_CLASS_RETAINED: LazyLock<[KeyValue; 1]> =
    LazyLock::new(|| [KeyValue::new("class", "retained")]);
pub(crate) static CACHE_CLASS_PROBATIONARY: LazyLock<[KeyValue; 1]> =
    LazyLock::new(|| [KeyValue::new("class", "probationary")]);
pub(crate) static CACHE_RESIDENCE_REASON_PRESSURE: LazyLock<[KeyValue; 1]> =
    LazyLock::new(|| [KeyValue::new("reason", "pressure")]);
pub(crate) static CACHE_RESIDENCE_REASON_CLEANUP: LazyLock<[KeyValue; 1]> =
    LazyLock::new(|| [KeyValue::new("reason", "cleanup")]);

pub(crate) struct CoreMetrics {
    pub cost_operations: Counter<u64>,
    pub cost_logical_bytes: Counter<u64>,
    pub cost_logical_unknown: Counter<u64>,
    pub cost_io_bytes: Counter<u64>,
    pub cost_io_unknown: Counter<u64>,
    pub cost_stage_seconds: Histogram<f64>,
    pub cost_prediction_absolute_error_seconds: Histogram<f64>,
    pub cost_estimate_evictions: Counter<u64>,
    pub cost_estimate_dropped: Counter<u64>,
    pub cost_shadow_candidates: Counter<u64>,
    pub cost_shadow_prediction_seconds: Histogram<f64>,
    pub cost_shadow_decisions: Counter<u64>,
    #[cfg(feature = "mooncake")]
    pub cost_route_decisions: Counter<u64>,
    pub cost_estimate_samples: Histogram<u64>,
    pub cost_estimate_age_seconds: Histogram<f64>,
    pub cost_estimate_error_seconds: Histogram<f64>,

    // Pinned pool (allocator-level)
    pub pool_capacity_bytes: UpDownCounter<i64>,
    pub pool_used_bytes: UpDownCounter<i64>,
    pub pool_alloc_failures: Counter<u64>,

    pub query_reserved_bytes: UpDownCounter<i64>,
    pub query_reserved_bytes_by_phase: UpDownCounter<i64>,
    pub query_speculative_reserved_bytes: UpDownCounter<i64>,
    pub query_budget_waits: Counter<u64>,
    pub query_budget_bypasses: Counter<u64>,
    pub query_coalesced_reads: Counter<u64>,
    pub warmup_prepared_bytes: Counter<u64>,
    pub warmup_restored_bytes: Counter<u64>,
    pub warmup_unused_bytes: Counter<u64>,
    pub warmup_pending_bytes: UpDownCounter<i64>,
    pub warmup_wait_byte_seconds: Counter<f64>,
    pub warmup_foreground_skips: Counter<u64>,

    // Inflight (write path safety/health)
    pub inflight_bytes: UpDownCounter<i64>,
    pub inflight_gc_cleaned: Counter<u64>,
    // Cache (sealed blocks in memory)
    pub cache_resident_bytes: UpDownCounter<i64>,
    pub cache_block_hits: Counter<u64>,
    pub cache_block_misses: Counter<u64>,
    pub cache_candidate_hits: Counter<u64>,
    pub cache_candidate_misses: Counter<u64>,
    pub cache_candidate_unknown: Counter<u64>,
    /// Per-decision block attribution for `query_prefetch`. Labelled by `tier`
    /// (`ram` | `remote` | `ssd` | `miss`). Each `query_prefetch` decision adds
    /// at most four times (one per non-zero tier) and the sum across tiers
    /// equals the request's `block_hashes.len()`.
    pub cache_tier_block_requests: Counter<u64>,
    pub cache_block_insertions: Counter<u64>,
    pub cache_block_admission_rejections: Counter<u64>,
    pub cache_block_evictions: Counter<u64>,
    pub cache_resident_blocks: UpDownCounter<i64>,
    pub cache_protected_bytes: UpDownCounter<i64>,
    pub cache_policy_promotions: Counter<u64>,
    pub cache_policy_demotions: Counter<u64>,
    pub cache_block_evictions_by_class: Counter<u64>,
    pub cache_block_evictions_still_referenced: Counter<u64>,
    pub cache_eviction_reclaimed_bytes: Counter<u64>,
    pub cache_residence_duration: Histogram<f64>,

    // GPU <-> CPU transfer
    pub save_bytes: Counter<u64>,
    pub save_duration_seconds: Histogram<f64>,

    pub load_bytes: Counter<u64>,
    pub load_duration_seconds: Histogram<f64>,
    pub load_failures: Counter<u64>,

    pub storage_codec_reserved_bytes: UpDownCounter<i64>,
    pub storage_codec_workspace_bytes: UpDownCounter<i64>,
    pub storage_codec_workspace_allocations: Counter<u64>,
    pub storage_codec_batches: Counter<u64>,
    pub storage_codec_batch_segments: Histogram<u64>,
    pub storage_codec_bytes: Counter<u64>,
    pub storage_codec_transfer_bytes: Counter<u64>,
    pub storage_codec_skips: Counter<u64>,
    pub storage_codec_decode_failures: Counter<u64>,
    pub storage_codec_seconds: Histogram<f64>,

    // SSD cache
    pub ssd_backend_fallbacks: Counter<u64>,
    pub ssd_read_pinned_bytes: UpDownCounter<i64>,
    pub ssd_gpu_staging_bytes: UpDownCounter<i64>,
    pub ssd_cufile_inflight_batches: UpDownCounter<i64>,
    pub ssd_gpu_read_fallbacks: Counter<u64>,
    pub ssd_gpu_write_fallbacks: Counter<u64>,
    pub ssd_pinned_write_skips: Counter<u64>,
    pub ssd_cufile_read_bytes: Counter<u64>,
    pub ssd_cufile_read_failures: Counter<u64>,
    pub ssd_cufile_read_seconds: Histogram<f64>,
    pub ssd_cufile_write_bytes: Counter<u64>,
    pub ssd_cufile_write_failures: Counter<u64>,
    pub ssd_cufile_write_seconds: Histogram<f64>,
    pub ssd_write_bytes: Counter<u64>,
    pub ssd_write_duration_seconds: Histogram<f64>,
    pub ssd_write_failures: Counter<u64>,
    pub ssd_write_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_write_queue_pending: UpDownCounter<i64>,
    pub ssd_write_queue_full: Counter<u64>,
    pub ssd_write_admission_skips: Counter<u64>,
    pub ssd_write_inflight: UpDownCounter<i64>,

    pub ssd_uring_read_operations: Counter<u64>,
    pub ssd_prefetch_bytes: Counter<u64>,
    pub ssd_prefetch_duration_seconds: Histogram<f64>,
    pub ssd_prefetch_success: Counter<u64>,
    pub ssd_prefetch_failures: Counter<u64>,
    pub ssd_prefetch_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_prefetch_inflight: UpDownCounter<i64>,
    pub ssd_prefetch_queue_closed: Counter<u64>,

    // Cross-node transfer lock (serving side)
    pub transfer_lock_active: UpDownCounter<i64>,
    pub transfer_lock_timeouts_total: Counter<u64>,
    pub transfer_reserved_bytes: UpDownCounter<i64>,
    pub transfer_expired_sessions: UpDownCounter<i64>,
    pub transfer_lock_rejections: Counter<u64>,

    // Mooncake remote fetch (client side)
    #[cfg(feature = "mooncake")]
    pub transfer_completion_outstanding: UpDownCounter<i64>,
    #[cfg(feature = "mooncake")]
    pub transfer_completion_retries: Counter<u64>,
    #[cfg(feature = "mooncake")]
    pub transfer_completion_rejections: Counter<u64>,
    #[cfg(feature = "mooncake")]
    pub remote_fetch_total: Counter<u64>,
    #[cfg(feature = "mooncake")]
    pub remote_fetch_duration_seconds: Histogram<f64>,
    #[cfg(feature = "mooncake")]
    pub remote_stage_duration_seconds: Histogram<f64>,
    #[cfg(feature = "mooncake")]
    pub remote_fetch_bytes: Counter<u64>,
    #[cfg(feature = "mooncake")]
    pub tent_nic_inflight_bytes: Histogram<u64>,
    #[cfg(feature = "mooncake")]
    pub tent_nic_bandwidth_bytes_per_second: Histogram<f64>,
    #[cfg(feature = "mooncake")]
    pub remote_fetch_plan_segments: Histogram<u64>,
    #[cfg(feature = "mooncake")]
    pub remote_fetch_plan_completed_segments: Histogram<u64>,
}

fn init_meter() -> Meter {
    global::meter("orbitkv-core")
}

/// Custom histogram boundaries for SSD throughput in bytes/s (1 to 40 GB/s, step 1 GB/s)
fn ssd_throughput_boundaries() -> Vec<f64> {
    // 1.0e9, 2.0e9, 3.0e9, ..., 40.0e9 (40 buckets in bytes/s)
    (1..=40).map(|i| i as f64 * 1.0e9).collect()
}

/// Histogram boundaries for Mooncake remote fetch (authorization + Mooncake READ).
#[cfg(feature = "mooncake")]
fn remote_fetch_duration_boundaries() -> Vec<f64> {
    vec![
        0.01, // 10ms
        0.02, // 20ms
        0.05, // 50ms
        0.1,  // 100ms
        0.2,  // 200ms
        0.5,  // 500ms
        1.0,  // 1s
        2.0,  // 2s
    ]
}

/// Histogram boundaries for the number of segments in one Mooncake fetch plan.
#[cfg(feature = "mooncake")]
fn remote_fetch_plan_segment_boundaries() -> Vec<f64> {
    vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0,
    ]
}

/// Tail-focused transfer duration boundaries in seconds.
///
/// Load/save debugging cares more about sustained tail regressions than small
/// sub-10ms jitter, so buckets stay coarse at the low end and distinguish
/// transfer stalls up to one minute.
fn duration_seconds_boundaries() -> Vec<f64> {
    vec![
        0.01,  // 10ms
        0.025, // 25ms
        0.05,  // 50ms
        0.1,   // 100ms
        0.25,  // 250ms
        0.5,   // 500ms
        1.0,   // 1s
        1.5,   // 1.5s
        2.0,   // 2s
        3.0,   // 3s
        5.0,   // 5s
        7.5,   // 7.5s
        10.0,  // 10s
        15.0,  // 15s
        30.0,  // 30s
        60.0,  // 60s
    ]
}

fn cache_residence_duration_seconds_boundaries() -> Vec<f64> {
    vec![
        1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1_800.0, 3_600.0, 7_200.0, 21_600.0,
        43_200.0, 86_400.0,
    ]
}

pub(crate) fn record_cache_tier_block_requests(ram: usize, remote: usize, ssd: usize, miss: usize) {
    let metrics = core_metrics();
    if ram > 0 {
        metrics
            .cache_tier_block_requests
            .add(ram as u64, &*TIER_RAM);
    }
    if remote > 0 {
        metrics
            .cache_tier_block_requests
            .add(remote as u64, &*TIER_REMOTE);
    }
    if ssd > 0 {
        metrics
            .cache_tier_block_requests
            .add(ssd as u64, &*TIER_SSD);
    }
    if miss > 0 {
        metrics
            .cache_tier_block_requests
            .add(miss as u64, &*TIER_MISS);
    }
}

fn cost_seconds_boundaries() -> Vec<f64> {
    vec![
        0.000_001, 0.000_005, 0.000_01, 0.000_025, 0.000_05, 0.000_1, 0.000_25, 0.000_5, 0.001,
        0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
    ]
}

pub(crate) fn core_metrics() -> &'static CoreMetrics {
    static METRICS: OnceLock<CoreMetrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let meter = init_meter();

        let metrics = CoreMetrics {
            cost_operations: meter.u64_counter("orbitkv_cost_operations")
                .with_description("Batch outcomes; abandoned means no terminal service evidence")
                .build(),
            cost_logical_bytes: meter.u64_counter("orbitkv_cost_logical_bytes")
                .with_description("Logical bytes attempted within each path boundary; nested paths must not be summed")
                .with_unit("bytes")
                .build(),
            cost_logical_unknown: meter.u64_counter("orbitkv_cost_logical_unknown")
                .with_description("Operations whose unpadded logical byte count is unavailable at this owner")
                .build(),
            cost_io_bytes: meter.u64_counter("orbitkv_cost_io_bytes")
                .with_description("Known physical bytes completed at the I/O owner, including short I/O")
                .with_unit("bytes")
                .build(),
            cost_io_unknown: meter.u64_counter("orbitkv_cost_io_unknown")
                .with_description("Operations whose physical byte count is unknown or belongs to a child owner")
                .build(),
            cost_stage_seconds: meter.f64_histogram("orbitkv_cost_stage")
                .with_description("Host-observed queue, admission, completed service and inclusive total; never device time")
                .with_unit("s")
                .with_boundaries(cost_seconds_boundaries())
                .build(),
            cost_prediction_absolute_error_seconds: meter.f64_histogram("orbitkv_cost_prediction_absolute_error")
                .with_description("Absolute prediction error: operation service or complete restore-route latency")
                .with_unit("s")
                .with_boundaries(cost_seconds_boundaries())
                .build(),
            cost_estimate_evictions: meter.u64_counter("orbitkv_cost_estimate_evictions")
                .with_description("Cost entries evicted at the fixed 512-entry limit")
                .build(),
            cost_estimate_dropped: meter.u64_counter("orbitkv_cost_estimate_dropped")
                .with_description("Estimation or shadow updates skipped on contention")
                .build(),
            cost_shadow_candidates: meter.u64_counter("orbitkv_cost_shadow_candidates")
                .with_description("Feasible metadata-backed candidate predictions; no alternative execution")
                .build(),
            cost_shadow_prediction_seconds: meter.f64_histogram("orbitkv_cost_shadow_prediction")
                .with_description("Predicted host-observed operation service or complete restore-route latency")
                .with_unit("s")
                .with_boundaries(cost_seconds_boundaries())
                .build(),
            cost_shadow_decisions: meter.u64_counter("orbitkv_cost_shadow_decisions")
                .with_description("Shadow agreement with compatibility, empirical-error and gain-margin checks; never selects execution")
                .build(),
            #[cfg(feature = "mooncake")]
            cost_route_decisions: meter.u64_counter("orbitkv_cost_route_decisions")
                .with_description("Opt-in execution choices among fresh compatible complete-route estimates")
                .build(),
            cost_estimate_samples: meter.u64_histogram("orbitkv_cost_estimate_samples")
                .with_description("Completed samples behind shadow predictions")
                .with_boundaries(vec![1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0])
                .build(),
            cost_estimate_age_seconds: meter.f64_histogram("orbitkv_cost_estimate_age")
                .with_description("Age of the last completed sample behind a shadow prediction")
                .with_unit("s")
                .with_boundaries(cost_seconds_boundaries())
                .build(),
            cost_estimate_error_seconds: meter.f64_histogram("orbitkv_cost_estimate_error")
                .with_description("EWMA absolute fitting error behind a shadow prediction")
                .with_unit("s")
                .with_boundaries(cost_seconds_boundaries())
                .build(),

            query_reserved_bytes: meter
                .i64_up_down_counter("orbitkv_query_reserved_bytes")
                .with_unit("bytes")
                .with_description("Total query-owned bytes, updated under the admission lock; shared pages count per owner")
                .build(),
            query_speculative_reserved_bytes: meter
                .i64_up_down_counter("orbitkv_query_speculative_reserved_bytes")
                .with_unit("bytes")
                .with_description("Speculative query-owned bytes, updated under the admission lock")
                .build(),
            query_reserved_bytes_by_phase: meter
                .i64_up_down_counter("orbitkv_query_reserved_bytes_by_phase")
                .with_unit("bytes")
                .with_description("Query-owned bytes by warming, preloading, prepared, preparing, ready, or restoring phase; shared pages count per owner")
                .build(),
            query_budget_waits: meter.u64_counter("orbitkv_query_budget_waits").build(),
            query_budget_bypasses: meter.u64_counter("orbitkv_query_budget_bypasses").build(),
            query_coalesced_reads: meter.u64_counter("orbitkv_query_coalesced_reads").build(),
            warmup_prepared_bytes: meter.u64_counter("orbitkv_warmup_prepared_bytes")
                .with_description("Unique page footprints read by a speculative initializer; DRAM hits and joined demand reads excluded")
                .build(),
            warmup_restored_bytes: meter.u64_counter("orbitkv_warmup_restored_bytes")
                .with_description("Warmup page footprints contributing to at least one successful local H2D, counted once per physical read")
                .build(),
            warmup_unused_bytes: meter.u64_counter("orbitkv_warmup_unused_bytes")
                .with_description("Warmup page footprints released by their last owner without a successful local H2D")
                .build(),
            warmup_pending_bytes: meter.i64_up_down_counter("orbitkv_warmup_pending_bytes")
                .with_description("Live warmup page footprints not yet restored locally; includes read cache and external owners")
                .build(),
            warmup_wait_byte_seconds: meter.f64_counter("orbitkv_warmup_wait_byte_seconds")
                .with_description("Page bytes times time from warmup readiness to first local H2D or final unused release, by outcome; live intervals excluded")
                .build(),
            warmup_foreground_skips: meter.u64_counter("orbitkv_warmup_foreground_skips")
                .with_description("Warmup hints skipped while foreground query ownership is active")
                .build(),
            // Pool
            pool_capacity_bytes: meter
                .i64_up_down_counter("orbitkv_pool_capacity_bytes")
                .with_unit("bytes")
                .with_description("Total pinned pool capacity in bytes")
                .build(),
            pool_used_bytes: meter
                .i64_up_down_counter("orbitkv_pool_used_bytes")
                .with_unit("bytes")
                .with_description("Current pinned pool usage in bytes")
                .build(),
            pool_alloc_failures: meter
                .u64_counter("orbitkv_pool_alloc_failures")
                .with_description("Pinned pool allocation failures after eviction retries")
                .build(),

            // Inflight
            inflight_bytes: meter
                .i64_up_down_counter("orbitkv_inflight_bytes")
                .with_unit("bytes")
                .with_description("Current bytes in inflight blocks (memory allocated but not yet sealed)")
                .build(),
            inflight_gc_cleaned: meter
                .u64_counter("orbitkv_inflight_gc_cleaned")
                .with_description("Stale inflight blocks cleaned by background GC")
                .build(),

            // Cache
            cache_resident_bytes: meter
                .i64_up_down_counter("orbitkv_cache_resident_bytes")
                .with_unit("bytes")
                .with_description("Current sealed block bytes resident in cache (sum of footprints)")
                .build(),
            cache_block_hits: meter
                .u64_counter("orbitkv_cache_block_hits")
                .with_description("Complete blocks found in cache (cache hit)")
                .build(),
            cache_block_misses: meter
                .u64_counter("orbitkv_cache_block_misses")
                .with_description("Complete blocks not found in cache (cache miss)")
                .build(),
            cache_candidate_hits: meter
                .u64_counter("orbitkv_cache_candidate_hits")
                .with_description("Candidate block positions available during metadata discovery; not leased reads")
                .build(),
            cache_candidate_misses: meter
                .u64_counter("orbitkv_cache_candidate_misses")
                .with_description("Block positions absent under complete metadata coverage")
                .build(),
            cache_candidate_unknown: meter
                .u64_counter("orbitkv_cache_candidate_unknown")
                .with_description(
                    "Block positions unavailable while remote metadata coverage is partial or unavailable",
                )
                .build(),
            cache_tier_block_requests: meter
                .u64_counter("orbitkv_cache_tier_block_requests")
                .with_description(
                    "Per-decision query_prefetch block attribution by storage tier \
                     (tier=ram|remote|ssd|miss). The sum across tiers equals the \
                     request's block count for that decision. This is decision \
                     attribution, not service attribution; backing failures must be \
                     inspected via orbitkv_remote_fetch_total{status=\"error\"} \
                     and orbitkv_ssd_prefetch_failures_total.",
                )
                .build(),
            cache_block_insertions: meter
                .u64_counter("orbitkv_cache_block_insertions")
                .with_description("New blocks inserted into cache")
                .build(),
            cache_block_admission_rejections: meter
                .u64_counter("orbitkv_cache_block_admission_rejections")
                .with_description("Blocks rejected by cache admission policy")
                .build(),
            cache_block_evictions: meter
                .u64_counter("orbitkv_cache_block_evictions")
                .with_description("Blocks evicted from cache due to memory pressure")
                .build(),
            cache_protected_bytes: meter
                .i64_up_down_counter("orbitkv_cache_protected_bytes")
                .with_unit("bytes")
                .with_description("Resident bytes protected after foreground demand; zero when protection is disabled")
                .build(),
            cache_policy_promotions: meter
                .u64_counter("orbitkv_cache_policy_promotions")
                .with_description("Demand promotions into the byte-bounded protected segment")
                .build(),
            cache_policy_demotions: meter
                .u64_counter("orbitkv_cache_policy_demotions")
                .with_description("Protected pages demoted to probation to make room for newer demand")
                .build(),
            cache_resident_blocks: meter
                .i64_up_down_counter("orbitkv_cache_resident_blocks")
                .with_description("Current cache blocks by replacement class")
                .build(),
            cache_block_evictions_by_class: meter
                .u64_counter("orbitkv_cache_block_evictions_by_class")
                .with_description("Cache block evictions by replacement class")
                .build(),
            cache_block_evictions_still_referenced: meter
                .u64_counter("orbitkv_cache_block_evictions_still_referenced")
                .with_description("Evicted cache blocks that still had external references (eviction did not immediately reclaim memory)")
                .build(),
            cache_eviction_reclaimed_bytes: meter
                .u64_counter("orbitkv_cache_eviction_reclaimed_bytes")
                .with_unit("bytes")
                .with_description("Estimated bytes actually reclaimed in pinned allocator after cache eviction")
                .build(),
            cache_residence_duration: meter
                .f64_histogram("orbitkv_cache_residence_duration")
                .with_unit("s")
                .with_description(
                    "RAM cache block residence duration from first insertion to removal",
                )
                .with_boundaries(cache_residence_duration_seconds_boundaries())
                .build(),

            // Transfer
            save_bytes: meter
                .u64_counter("orbitkv_save_bytes")
                .with_unit("bytes")
                .with_description("Total bytes saved from GPU to CPU storage")
                .build(),
            save_duration_seconds: meter
                .f64_histogram("orbitkv_save_duration")
                .with_unit("s")
                .with_description("Save operation latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),

            load_bytes: meter
                .u64_counter("orbitkv_load_bytes")
                .with_unit("bytes")
                .with_description("Total bytes loaded from CPU storage to GPU")
                .build(),
            load_duration_seconds: meter
                .f64_histogram("orbitkv_load_duration")
                .with_unit("s")
                .with_description("Load operation latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            load_failures: meter
                .u64_counter("orbitkv_load_failures")
                .with_description("Load operation failures (e.g., transfer errors)")
                .build(),

            // SSD
            storage_codec_reserved_bytes: meter.i64_up_down_counter("orbitkv_storage_codec_reserved_bytes")
                .with_description("Reserved GPU codec workspace until transfer completion").with_unit("bytes").build(),
            storage_codec_workspace_bytes: meter.i64_up_down_counter("orbitkv_storage_codec_workspace_bytes")
                .with_description("Retained reusable GPU codec arenas, including idle workers").with_unit("bytes").build(),
            storage_codec_workspace_allocations: meter.u64_counter("orbitkv_storage_codec_workspace_allocations")
                .with_description("GPU codec arena allocations and growth operations").build(),
            storage_codec_batches: meter.u64_counter("orbitkv_storage_codec_batches")
                .with_description("GPU codec batches by operation").build(),
            storage_codec_batch_segments: meter.u64_histogram("orbitkv_storage_codec_batch_segments")
                .with_description("Segments processed per GPU codec batch")
                .with_boundaries(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]).build(),
            storage_codec_transfer_bytes: meter.u64_counter("orbitkv_storage_codec_transfer_bytes").with_description("Actual codec-path D2H/H2D payload bytes, including raw/CPU fallbacks").with_unit("bytes").build(),
            storage_codec_bytes: meter.u64_counter("orbitkv_storage_codec_bytes")
                .with_description("Encoded publications, logical and aligned resident bytes").with_unit("bytes").build(),
            storage_codec_skips: meter.u64_counter("orbitkv_storage_codec_skips")
                .with_description("Encoding declined by representation, value range, ratio or budget").build(),
            storage_codec_decode_failures: meter.u64_counter("orbitkv_storage_codec_decode_failures")
                .with_description("Invalid encoded objects and failed codec restores").build(),
            storage_codec_seconds: meter.f64_histogram("orbitkv_storage_codec_duration")
                .with_description("GPU codec and memory transfer duration, including CPU fallback").with_unit("s")
                .with_boundaries(duration_seconds_boundaries()).build(),
            ssd_backend_fallbacks: meter.u64_counter("orbitkv_ssd_backend_fallbacks").with_description("Automatic transitions from cuFile to io_uring; startup or runtime failures").build(),
            ssd_cufile_write_bytes: meter.u64_counter("orbitkv_ssd_cufile_write_bytes").with_description("Bytes written by cuFile including alignment; does not prove native GDS").build(),
            ssd_cufile_write_failures: meter.u64_counter("orbitkv_ssd_cufile_write_failures").with_description("Failed or short cuFile writes").build(),
            ssd_cufile_write_seconds: meter.f64_histogram("orbitkv_ssd_cufile_write").with_unit("s").with_description("Asynchronous cuFile write completion latency including GPU gather and polling").build(),
            ssd_read_pinned_bytes: meter.i64_up_down_counter("orbitkv_ssd_read_pinned_bytes").with_description("SSD bytes pinned by restore source leases").build(),
            ssd_cufile_inflight_batches: meter.i64_up_down_counter("orbitkv_ssd_cufile_inflight_batches").with_description("GPU storage batches owning a staging slot until I/O and scatter complete").build(),
            ssd_gpu_read_fallbacks: meter.u64_counter("orbitkv_ssd_gpu_read_fallbacks").with_description("Automatic cuFile restores falling back to io_uring when another instance owns device staging").build(),
            ssd_gpu_write_fallbacks: meter.u64_counter("orbitkv_ssd_gpu_write_fallbacks").with_description("GPU write batches falling back to host publication after bounded admission fills").build(),
            ssd_gpu_staging_bytes: meter.i64_up_down_counter("orbitkv_ssd_gpu_staging_bytes").with_description("Registered GPU staging bytes owned by cuFile workers").build(),
            ssd_pinned_write_skips: meter.u64_counter("orbitkv_ssd_pinned_write_skips").with_description("SSD reservations rejected to protect readers or in-flight writes").build(),
            ssd_cufile_read_bytes: meter.u64_counter("orbitkv_ssd_cufile_read_bytes").with_description("Bytes read by cuFile including alignment; does not prove native GDS").build(),
            ssd_cufile_read_failures: meter.u64_counter("orbitkv_ssd_cufile_read_failures").with_description("Failed or short cuFile reads").build(),
            ssd_cufile_read_seconds: meter.f64_histogram("orbitkv_ssd_cufile_read").with_unit("s").with_description("Asynchronous cuFile read completion latency including GPU scatter and polling").build(),
            ssd_write_bytes: meter
                .u64_counter("orbitkv_ssd_write_bytes")
                .with_unit("bytes")
                .with_description("Bytes written to SSD cache")
                .build(),
            ssd_write_duration_seconds: meter
                .f64_histogram("orbitkv_ssd_write_duration")
                .with_unit("s")
                .with_description("Per-block SSD write submission and completion latency; concurrent operations overlap")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            ssd_write_failures: meter
                .u64_counter("orbitkv_ssd_write_failures")
                .with_description("SSD write failures")
                .build(),
            ssd_write_throughput_bytes_per_second: meter
                .f64_histogram("orbitkv_ssd_write_throughput")
                .with_unit("bytes/s")
                .with_description("SSD write throughput per block in bytes/s")
                .with_boundaries(ssd_throughput_boundaries())
                .build(),
            ssd_write_queue_pending: meter
                .i64_up_down_counter("orbitkv_ssd_write_queue_pending")
                .with_description("Current pending blocks in SSD write queue")
                .build(),
            ssd_write_admission_skips: meter
                .u64_counter("orbitkv_ssd_write_admission_skips")
                .with_description("SSD write candidates skipped by reason: cold, resident, pending, or duplicate")
                .build(),
            ssd_write_queue_full: meter
                .u64_counter("orbitkv_ssd_write_queue_full")
                .with_description("Write requests dropped due to full queue")
                .build(),
            ssd_write_inflight: meter
                .i64_up_down_counter("orbitkv_ssd_write_inflight")
                .with_description("Current in-flight SSD write operations")
                .build(),

            ssd_uring_read_operations: meter
                .u64_counter("orbitkv_ssd_uring_read_operations")
                .with_description("Successfully queued io_uring READV SQEs; not physical NVMe commands")
                .build(),
            ssd_prefetch_bytes: meter
                .u64_counter("orbitkv_ssd_prefetch_bytes")
                .with_unit("bytes")
                .with_description("Bytes prefetched from SSD cache")
                .build(),
            ssd_prefetch_duration_seconds: meter
                .f64_histogram("orbitkv_ssd_prefetch_duration")
                .with_unit("s")
                .with_description("SSD prefix prefetch latency including queueing, pinned allocation, reads, and reconstruction; excludes H2D")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            ssd_prefetch_success: meter
                .u64_counter("orbitkv_ssd_prefetch_success")
                .with_description("Blocks successfully prefetched from SSD cache")
                .build(),
            ssd_prefetch_failures: meter
                .u64_counter("orbitkv_ssd_prefetch_failures")
                .with_description("SSD prefetch failures (short read, rebuild error, stale)")
                .build(),
            ssd_prefetch_throughput_bytes_per_second: meter
                .f64_histogram("orbitkv_ssd_prefetch_throughput")
                .with_unit("bytes/s")
                .with_description("SSD prefetch throughput per block in bytes/s")
                .with_boundaries(ssd_throughput_boundaries())
                .build(),
            ssd_prefetch_inflight: meter
                .i64_up_down_counter("orbitkv_ssd_prefetch_inflight")
                .with_description("Current in-flight SSD prefetch operations")
                .build(),
            ssd_prefetch_queue_closed: meter
                .u64_counter("orbitkv_ssd_prefetch_queue_closed")
                .with_description("Prefetch requests dropped due to full queue")
                .build(),

            // Transfer lock
            transfer_lock_active: meter
                .i64_up_down_counter("orbitkv_transfer_lock_active")
                .with_description("Currently locked blocks for cross-node transfer")
                .build(),
            transfer_lock_timeouts_total: meter
                .u64_counter("orbitkv_transfer_lock_timeouts_total")
                .with_description("Overdue source sessions; expiry never releases memory")
                .build(),
            transfer_reserved_bytes: meter.i64_up_down_counter("orbitkv_transfer_reserved_bytes")
                .with_description("Source allocations reserved by transfers, including overdue sessions; shared slabs counted once per session")
                .build(),
            transfer_expired_sessions: meter.i64_up_down_counter("orbitkv_transfer_expired_sessions")
                .with_description("Overdue sessions still retaining source memory")
                .build(),
            transfer_lock_rejections: meter.u64_counter("orbitkv_transfer_lock_rejections")
                .with_description("Source authorizations rejected by byte or session limits")
                .build(),

            // Mooncake remote fetch (client side)
            #[cfg(feature = "mooncake")]
            transfer_completion_outstanding: meter.i64_up_down_counter("orbitkv_transfer_completion_outstanding")
                .with_description("Reserved requester completion slots, including active authorizations and READs")
                .build(),
            #[cfg(feature = "mooncake")]
            transfer_completion_retries: meter.u64_counter("orbitkv_transfer_completion_retries")
                .with_description("Unacknowledged release attempts; completion retained for retry")
                .build(),
            #[cfg(feature = "mooncake")]
            transfer_completion_rejections: meter.u64_counter("orbitkv_transfer_completion_rejections")
                .with_description("Authorizations skipped because requester completion capacity is exhausted")
                .build(),
            #[cfg(feature = "mooncake")]
            remote_fetch_total: meter
                .u64_counter("orbitkv_remote_fetch_total")
                .with_description("Mooncake remote fetch attempts (status=ok|error)")
                .build(),
            #[cfg(feature = "mooncake")]
            remote_fetch_duration_seconds: meter
                .f64_histogram("orbitkv_remote_fetch_duration")
                .with_unit("s")
                .with_description("End-to-end Mooncake fetch latency (authorization + READ)")
                .with_boundaries(remote_fetch_duration_boundaries())
                .build(),
            #[cfg(feature = "mooncake")]
            remote_stage_duration_seconds: meter.f64_histogram("orbitkv_remote_stage_duration")
                .with_unit("s")
                .with_description("Remote discovery RPC, authorization and completed transfer stages")
                .with_boundaries(remote_fetch_duration_boundaries())
                .build(),
            #[cfg(feature = "mooncake")]
            remote_fetch_bytes: meter
                .u64_counter("orbitkv_remote_fetch_bytes")
                .with_unit("bytes")
                .with_description("Total bytes fetched via Mooncake from remote nodes")
                .build(),
            #[cfg(feature = "mooncake")]
            tent_nic_inflight_bytes: meter
                .u64_histogram("orbitkv_tent_nic_inflight_bytes")
                .with_unit("bytes")
                .with_description("TENT live RDMA rail inflight-byte snapshot after a peer batch")
                .build(),
            #[cfg(feature = "mooncake")]
            tent_nic_bandwidth_bytes_per_second: meter
                .f64_histogram("orbitkv_tent_nic_bandwidth_bytes_per_second")
                .with_unit("By/s")
                .with_description("TENT RDMA rail EWMA bandwidth snapshot after a peer batch")
                .build(),
            #[cfg(feature = "mooncake")]
            remote_fetch_plan_segments: meter
                .u64_histogram("orbitkv_remote_fetch_plan_segments")
                .with_description("Number of segments attempted per Mooncake fetch plan, including stale-candidate retries")
                .with_boundaries(remote_fetch_plan_segment_boundaries())
                .build(),
            #[cfg(feature = "mooncake")]
            remote_fetch_plan_completed_segments: meter
                .u64_histogram("orbitkv_remote_fetch_plan_completed_segments")
                .with_description(
                    "Number of segments completed before a Mooncake fetch plan stopped",
                )
                .with_boundaries(remote_fetch_plan_segment_boundaries())
                .build(),
        };
        for counter in [
            &metrics.query_reserved_bytes,
            &metrics.inflight_bytes,
            &metrics.transfer_lock_active,
            &metrics.transfer_reserved_bytes,
            &metrics.ssd_prefetch_inflight,
            &metrics.ssd_read_pinned_bytes,
            &metrics.ssd_write_queue_pending,
            &metrics.ssd_write_inflight,
        ] {
            counter.add(0, &[]);
        }
        #[cfg(feature = "mooncake")]
        metrics.transfer_completion_outstanding.add(0, &[]);
        metrics.load_bytes.add(0, &[]);
        metrics
    })
}

#[cfg(test)]
#[path = "../tests/unit/metrics.rs"]
mod tests;
