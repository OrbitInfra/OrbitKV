/// The tier a block was attributed to for a single `query_prefetch` decision.
///
/// Only backing tiers (`Remote`, `Ssd`) are represented here; the local RAM
/// prefix hit is conveyed through the `hit` argument to `classify` rather
/// than as a separate `AttributionSource` variant. Keeping the enum closed
/// over the two backing sources makes the call sites in `read.rs` total
/// without a dead RAM branch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum AttributionSource {
    /// Mooncake remote-fetch was selected for at least one remaining block.
    Remote,
    /// SSD prefetch was selected for at least one remaining block.
    Ssd,
}

/// Per-decision block counts. Invariant: `ram + remote + ssd + miss == total`.
#[derive(Copy, Clone, Debug)]
pub(super) struct TierAttribution {
    ram: usize,
    remote: usize,
    ssd: usize,
    miss: usize,
}

impl TierAttribution {
    /// Build the per-decision attribution from the values the query coordinator
    /// already knows. `loading_source` is `Some` iff a backing tier was
    /// chosen for the remaining prefix.
    ///
    /// Rejects inconsistent block counts.
    pub(super) fn classify(
        total: usize,
        hit: usize,
        loading: usize,
        loading_source: Option<AttributionSource>,
    ) -> Result<Self, crate::EngineError> {
        let miss = total
            .checked_sub(hit)
            .and_then(|remaining| remaining.checked_sub(loading))
            .ok_or_else(|| {
                crate::EngineError::InvalidArgument(
                    "tier hits and loads exceed total blocks".into(),
                )
            })?;
        let (remote, ssd) = match loading_source {
            Some(AttributionSource::Remote) => (loading, 0),
            Some(AttributionSource::Ssd) => (0, loading),
            // `loading == 0` is the only valid shape when no backing tier was
            // selected; if a caller hands us `loading > 0` without a source it
            // is a programmer error, surfaced under debug builds.
            None => {
                if loading != 0 {
                    return Err(crate::EngineError::InvalidArgument(
                        "loading blocks have no backing source".into(),
                    ));
                }
                (0, 0)
            }
        };
        let attribution = Self {
            ram: hit,
            remote,
            ssd,
            miss,
        };
        debug_assert_eq!(
            attribution.sum(),
            total,
            "tier attribution must sum to total"
        );
        Ok(attribution)
    }

    fn sum(&self) -> usize {
        self.ram + self.remote + self.ssd + self.miss
    }
}

/// Record a tier attribution to the OTel counter. Skips zero-count tiers so
/// each decision adds at most four points (typically one to three).
///
/// The `total` parameter is used only for the debug invariant; release builds
/// strip it.
pub(super) fn record_cache_tier_block_requests(total: usize, attribution: TierAttribution) {
    debug_assert_eq!(
        attribution.sum(),
        total,
        "tier attribution must sum to total"
    );
    crate::metrics::record_cache_tier_block_requests(
        attribution.ram,
        attribution.remote,
        attribution.ssd,
        attribution.miss,
    );
}

#[cfg(test)]
#[path = "../../tests/unit/query/tier_attribution.rs"]
mod tests;
