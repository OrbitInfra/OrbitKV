use super::*;

#[test]
fn partial_backing_decisions_keep_selected_tier_and_residual_contract() {
    let cases = [
        // Done branch: local RAM prefix plus residual miss when no backing
        // tier is selected for this decision.
        (
            TierAttribution::classify(7, 3, 0, None).unwrap(),
            (3, 0, 0, 4, 7),
        ),
        // OrbiTE found only part of the non-RAM prefix.
        (
            TierAttribution::classify(5, 1, 3, Some(AttributionSource::Remote)).unwrap(),
            (1, 3, 0, 1, 5),
        ),
        // SSD prefetch accepted only part of the non-RAM prefix, for
        // example after backpressure trimming.
        (
            TierAttribution::classify(6, 1, 2, Some(AttributionSource::Ssd)).unwrap(),
            (1, 0, 2, 3, 6),
        ),
    ];

    for (attribution, expected) in cases {
        assert_eq!(
            (
                attribution.ram,
                attribution.remote,
                attribution.ssd,
                attribution.miss,
                attribution.sum()
            ),
            expected
        );
    }
}

#[test]
fn invalid_tier_counts_return_errors() {
    for (total, hit, loading, source) in [
        (4, 1, 1, None),
        (3, 2, 2, Some(AttributionSource::Ssd)),
        (usize::MAX, usize::MAX, 1, Some(AttributionSource::Remote)),
    ] {
        assert!(matches!(
            TierAttribution::classify(total, hit, loading, source),
            Err(crate::EngineError::InvalidArgument(_))
        ));
    }
}
