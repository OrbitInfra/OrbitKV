use super::*;
use crate::cost::{CostObservationKind, ExecutionResource, Representation};
use std::time::Duration;

fn key(path: CostObservationKind, peer: u64) -> CostEstimateKey {
    CostEstimateKey::new(
        path,
        ExecutionResource::Peer(peer),
        Representation::Raw,
        4096,
        2,
    )
}

fn estimate(seconds: f64, error: f64) -> Estimate {
    Estimate {
        count: 8,
        seconds,
        absolute_error: error,
        updated: Instant::now(),
    }
}

#[test]
fn selection_requires_complete_compatible_fresh_evidence() {
    let candidates = [
        key(CostObservationKind::PeerDramHostReady, 1),
        key(CostObservationKind::PeerDramHostReady, 2),
    ];
    assert_eq!(
        choose(&candidates, &[Some(estimate(0.02, 0.0)), None], 0),
        (0, "unknown")
    );
    assert_eq!(
        choose(
            &[
                candidates[0],
                CostEstimateKey::new(
                    CostObservationKind::PeerSsdHostReady,
                    ExecutionResource::Peer(2),
                    Representation::Ans,
                    4096,
                    2,
                ),
            ],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0)),],
            0,
        ),
        (0, "incomparable")
    );
    assert_eq!(
        choose(
            &[
                candidates[0],
                CostEstimateKey::new(
                    CostObservationKind::LocalSsdHostReady,
                    ExecutionResource::SsdStore(9),
                    Representation::Raw,
                    4096,
                    2,
                ),
            ],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0)),],
            0,
        ),
        (1, "selected"),
        "different resources and media remain comparable for the same HostReady demand"
    );
}

#[test]
fn selection_requires_gain_beyond_error_and_margin() {
    let candidates = [
        key(CostObservationKind::PeerDramHostReady, 1),
        key(CostObservationKind::PeerDramHostReady, 2),
        key(CostObservationKind::PeerDramHostReady, 3),
    ];
    assert_eq!(
        choose(
            &candidates,
            &[
                Some(estimate(0.02, 0.001)),
                Some(estimate(0.01, 0.001)),
                Some(estimate(0.015, 0.001))
            ],
            0,
        ),
        (1, "selected")
    );
    assert_eq!(
        choose(
            &candidates[..2],
            &[Some(estimate(0.02, 0.006)), Some(estimate(0.01, 0.005)),],
            0,
        ),
        (0, "within_margin")
    );
    assert_eq!(
        choose(
            &candidates[..2],
            &[Some(estimate(0.02, 0.0)), Some(estimate(0.0195, 0.0)),],
            0,
        ),
        (0, "within_margin")
    );
}

#[test]
fn engine_ready_routes_require_the_same_destination_device() {
    let direct = CostEstimateKey::new(
        CostObservationKind::GpuLoadDirect,
        ExecutionResource::Gpu(7),
        Representation::Raw,
        4096,
        2,
    );
    let ssd = CostEstimateKey::new(
        CostObservationKind::SsdUringRestore,
        ExecutionResource::SsdRestore {
            device: 7,
            copy_backend: 0,
            stores: 11,
            has_memory: false,
        },
        Representation::Raw,
        4096,
        2,
    );
    let predictions = [Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0))];
    assert_eq!(choose(&[direct, ssd], &predictions, 0), (1, "selected"));

    let other_device = ssd.with_observation_kind_and_resource(
        CostObservationKind::SsdUringRestore,
        ExecutionResource::SsdRestore {
            device: 8,
            copy_backend: 0,
            stores: 11,
            has_memory: false,
        },
    );
    assert_eq!(
        choose(&[direct, other_device], &predictions, 0),
        (0, "incomparable")
    );
}

#[test]
fn decode_ready_routes_compare_only_for_the_same_target_and_shape() {
    let direct = CostEstimateKey::new(
        CostObservationKind::CacheRestore,
        ExecutionResource::CacheRestore {
            source_set_hash: 9,
            destination_device: 7,
            copy_backend: 0,
        },
        Representation::Raw,
        4096,
        2,
    )
    .with_source_shape(4096, 2)
    .with_wire_bytes(4096);
    let handoff = CostEstimateKey::new(
        CostObservationKind::PrefillToDecodeHandoff,
        ExecutionResource::PrefillToDecodeHandoff {
            source_endpoint_hash: 11,
            destination_device: 7,
        },
        Representation::Raw,
        4096,
        2,
    )
    .with_source_shape(4096, 2)
    .with_wire_bytes(4096);
    let predictions = [Some(estimate(0.02, 0.0)), Some(estimate(0.01, 0.0))];
    assert_eq!(choose(&[direct, handoff], &predictions, 0), (1, "selected"));

    let other_device = handoff.with_observation_kind_and_resource(
        CostObservationKind::PrefillToDecodeHandoff,
        ExecutionResource::PrefillToDecodeHandoff {
            source_endpoint_hash: 11,
            destination_device: 8,
        },
    );
    assert_eq!(
        choose(&[direct, other_device], &predictions, 0),
        (0, "incomparable")
    );
}

#[test]
fn decode_ready_decision_uses_fresh_queue_and_transfer_pressure() {
    let direct = CostEstimateKey::new(
        CostObservationKind::CacheRestore,
        ExecutionResource::CacheRestore {
            source_set_hash: 31,
            destination_device: 7,
            copy_backend: 0,
        },
        Representation::Raw,
        4096,
        2,
    )
    .with_source_shape(4096, 2)
    .with_wire_bytes(4096);
    let handoff = CostEstimateKey::new(
        CostObservationKind::PrefillToDecodeHandoff,
        ExecutionResource::PrefillToDecodeHandoff {
            source_endpoint_hash: 32,
            destination_device: 7,
        },
        Representation::Raw,
        4096,
        2,
    )
    .with_source_shape(4096, 2)
    .with_wire_bytes(4096);
    crate::cost::record_resource_evidence(
        direct.resource,
        crate::CompletionResourceEvidence {
            decode_page_bytes: 4096,
            queue_depth: 1,
            queue_parallelism: 1,
            transfer_inflight_bytes: 0,
            transfer_bandwidth_bytes_per_second: 0,
        },
        Duration::ZERO,
    );
    crate::cost::record_resource_evidence(
        handoff.resource,
        crate::CompletionResourceEvidence {
            decode_page_bytes: 4096,
            queue_depth: 17,
            queue_parallelism: 16,
            transfer_inflight_bytes: 1000,
            transfer_bandwidth_bytes_per_second: 10_000,
        },
        Duration::ZERO,
    );
    let mut predictions = [Some(estimate(0.2, 0.0)), Some(estimate(0.1, 0.0))];
    apply_decode_ready_pressure(&[direct, handoff], &mut predictions, Instant::now()).unwrap();
    assert_eq!(predictions[0].unwrap().seconds, 0.2);
    assert!((predictions[1].unwrap().seconds - 0.3).abs() < 1e-10);

    let unknown = handoff.with_observation_kind_and_resource(
        CostObservationKind::PrefillToDecodeHandoff,
        ExecutionResource::PrefillToDecodeHandoff {
            source_endpoint_hash: 33,
            destination_device: 7,
        },
    );
    assert_eq!(
        apply_decode_ready_pressure(
            &[direct, unknown],
            &mut [Some(estimate(0.2, 0.0)), Some(estimate(0.1, 0.0))],
            Instant::now(),
        ),
        Err("resource_unknown")
    );
}

#[test]
fn execution_selection_requires_both_explicit_switches() {
    const CHILD: &str = "ORBITKV_TEST_ROUTE_SELECTION_CHILD";
    if let Ok(expected) = std::env::var(CHILD) {
        let candidates = [
            key(CostObservationKind::PeerDramHostReady, 101),
            key(CostObservationKind::PeerDramHostReady, 102),
        ];
        let now = Instant::now();
        for _ in 0..super::super::MIN_SAMPLES {
            ESTIMATES.lock().observe(candidates[0], 0.02, now);
            ESTIMATES.lock().observe(candidates[1], 0.01, now);
        }
        assert_eq!(
            select_route(&candidates, 0, SelectionScope::PeerOwner),
            expected.parse::<usize>().unwrap()
        );
        return;
    }

    for (observations, selection, expected) in [
        (None, None, 0),
        (Some("1"), None, 0),
        (None, Some("1"), 0),
        (Some("1"), Some("1"), 1),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "cost::decision::tests::execution_selection_requires_both_explicit_switches",
                "--nocapture",
            ])
            .env(CHILD, expected.to_string());
        if let Some(value) = observations {
            child.env("ORBITKV_COST_OBSERVATIONS", value);
        } else {
            child.env_remove("ORBITKV_COST_OBSERVATIONS");
        }
        if let Some(value) = selection {
            child.env("ORBITKV_COST_SELECTION", value);
        } else {
            child.env_remove("ORBITKV_COST_SELECTION");
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "observations={observations:?} selection={selection:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[test]
fn cross_medium_selection_requires_three_explicit_switches() {
    const CHILD: &str = "ORBITKV_TEST_CROSS_MEDIUM_SELECTION_CHILD";
    if let Ok(expected) = std::env::var(CHILD) {
        assert_eq!(
            super::super::cross_medium_selection_enabled(),
            expected == "1"
        );
        return;
    }

    for (observations, selection, cross_medium, expected) in [
        (None, None, None, false),
        (Some("1"), Some("1"), None, false),
        (Some("1"), None, Some("1"), false),
        (None, Some("1"), Some("1"), false),
        (Some("1"), Some("1"), Some("1"), true),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "cost::decision::tests::cross_medium_selection_requires_three_explicit_switches",
                "--nocapture",
            ])
            .env(CHILD, if expected { "1" } else { "0" });
        for (name, value) in [
            ("ORBITKV_COST_OBSERVATIONS", observations),
            ("ORBITKV_COST_SELECTION", selection),
            ("ORBITKV_CROSS_MEDIUM_SELECTION", cross_medium),
        ] {
            if let Some(value) = value {
                child.env(name, value);
            } else {
                child.env_remove(name);
            }
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "observations={observations:?} selection={selection:?} cross={cross_medium:?}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
