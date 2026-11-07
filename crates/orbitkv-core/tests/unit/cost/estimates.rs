use super::*;
use crate::cost::bucket;
use crate::cost::{CostObservationKind, ExecutionResource, Representation};
use std::time::Duration;

fn key(resource: u64) -> CostEstimateKey {
    CostEstimateKey::new(
        CostObservationKind::GpuLoadDirect,
        ExecutionResource::Gpu(resource),
        Representation::Raw,
        65536,
        4,
    )
}

#[test]
fn estimates_are_bounded_and_isolate_resource_representation_and_shape() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    for resource in 0..(CAPACITY as u64 + 20) {
        estimates.observe(key(resource), 0.01, start + Duration::from_millis(resource));
        assert!(estimates.entries.len() <= CAPACITY);
    }
    assert!(!estimates.entries.contains_key(&key(0)));
    let retained = key(CAPACITY as u64)
        .with_ssd_shape(131072, 8, 65536, 4)
        .with_dma_ranges(1);
    for _ in 0..MIN_SAMPLES {
        estimates.observe(retained, 0.01, start + Duration::from_secs(1));
    }
    assert!(
        estimates
            .predict(retained, start + Duration::from_secs(1))
            .is_some()
    );
    for different in [
        retained.with_observation_kind_and_resource(retained.kind, ExecutionResource::Gpu(10000)),
        CostEstimateKey {
            representation: Representation::Ans,
            ..retained
        },
        CostEstimateKey {
            size: retained.size + 1,
            ..retained
        },
        CostEstimateKey {
            fragments: retained.fragments + 1,
            ..retained
        },
        // Whole-extent reads and requested SSD ranges are independent of the
        // total restore size, especially with mixed DRAM hits and TP slots.
        retained.with_ssd_shape(262144, 8, 65536, 4),
        retained.with_ssd_shape(131072, 16, 65536, 4),
        retained.with_ssd_shape(131072, 8, 32768, 4),
        retained.with_ssd_shape(131072, 8, 65536, 2),
        retained.with_dma_ranges(4),
        retained.with_observation_kind(CostObservationKind::GpuLoadKernel),
    ] {
        assert!(estimates.predict(different, start).is_none());
    }
    assert_eq!(bucket(0), 0);
    assert_eq!(bucket(u64::MAX), 64);
}

#[test]
fn replay_requires_recent_samples_and_reports_preupdate_error() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    for i in 0..MIN_SAMPLES {
        assert!(estimates.predict(key(1), start).is_none());
        estimates.observe(key(1), 0.01, start + Duration::from_millis(i));
    }
    let predicted = estimates
        .predict(key(1), start + Duration::from_secs(1))
        .unwrap();
    assert_eq!(predicted.seconds, 0.01);
    estimates.observe(key(1), 0.02, start + Duration::from_secs(1));
    let updated = estimates
        .predict(key(1), start + Duration::from_secs(1))
        .unwrap();
    assert!((updated.seconds - 0.012).abs() < 1e-10);
    assert!((updated.absolute_error - 0.002).abs() < 1e-10);
    let stale = start + MAX_AGE + Duration::from_secs(2);
    assert!(estimates.predict(key(1), stale).is_none());
    estimates.observe(key(1), 0.5, stale);
    assert!(estimates.predict(key(1), stale).is_none());
    assert_eq!(estimates.entries[&key(1)].count, 1);
    assert_eq!(estimates.entries[&key(1)].seconds, 0.5);
}

#[test]
fn invalid_samples_do_not_replace_evidence_and_future_samples_are_not_fresh() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    for _ in 0..MIN_SAMPLES {
        estimates.observe(key(1), 0.01, start);
    }
    for sample in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        assert!(!estimates.observe(key(1), sample, start));
        assert!(!estimates.observe(key(2), sample, start));
    }
    assert_eq!(estimates.entries.len(), 1);
    let retained = estimates.predict(key(1), start).unwrap();
    assert_eq!(retained.seconds, 0.01);
    assert_eq!(retained.count, MIN_SAMPLES);
    assert!(
        estimates
            .predict(key(1), start - Duration::from_millis(1))
            .is_none()
    );
}

#[test]
fn resource_domains_and_peer_incarnations_never_share_samples() {
    let start = Instant::now();
    let mut estimates = Estimates::default();
    let resources = [
        ExecutionResource::Gpu(1),
        ExecutionResource::SsdStore(1),
        ExecutionResource::SsdFile(1),
    ];
    for (index, resource) in resources.into_iter().enumerate() {
        for _ in 0..MIN_SAMPLES {
            estimates.observe(
                key(1).with_observation_kind_and_resource(
                    CostObservationKind::GpuLoadDirect,
                    resource,
                ),
                index as f64,
                start,
            );
        }
    }
    for (index, resource) in resources.into_iter().enumerate() {
        assert_eq!(
            estimates
                .predict(
                    key(1).with_observation_kind_and_resource(
                        CostObservationKind::GpuLoadDirect,
                        resource
                    ),
                    start,
                )
                .unwrap()
                .seconds,
            index as f64
        );
    }

    {
        use crate::cost::resource_id;
        let owner = orbitkv_state::CacheOwner {
            endpoint: "same-address".into(),
            incarnation: uuid::Uuid::from_u128(1),
        };
        let old = key(1).with_observation_kind_and_resource(
            CostObservationKind::RemoteRead,
            ExecutionResource::Peer(resource_id(&owner)),
        );
        for _ in 0..MIN_SAMPLES {
            estimates.observe(old, 0.1, start);
        }
        let replacement = orbitkv_state::CacheOwner {
            incarnation: uuid::Uuid::from_u128(2),
            ..owner
        };
        let new = old.with_observation_kind_and_resource(
            CostObservationKind::RemoteRead,
            ExecutionResource::Peer(resource_id(&replacement)),
        );
        assert!(estimates.predict(old, start).is_some());
        assert!(estimates.predict(new, start).is_none());
    }
}

#[test]
fn caller_to_drain_estimates_cannot_select_preparation_or_service_routes() {
    let local = key(9).with_observation_kind(CostObservationKind::EngineLocalRestore);
    let manager = local.with_observation_kind_and_resource(
        CostObservationKind::CacheRestore,
        ExecutionResource::CacheRestore {
            source_set_hash: 1,
            destination_device: 9,
            copy_backend: 0,
        },
    );
    let now = Instant::now();
    let mut estimates = Estimates::default();
    for _ in 0..MIN_SAMPLES {
        estimates.observe(local, 0.003, now);
    }
    assert_eq!(estimates.predict(local, now).unwrap().seconds, 0.003);
    assert!(estimates.predict(manager, now).is_none());
    assert!(!local.comparable(key(9)));
    assert!(!local.comparable(manager));
    {
        assert!(!local.route_comparable(manager));
        assert!(!manager.route_comparable(local));
    }
}

#[test]
fn local_restore_estimates_separate_copy_backend_and_source_domain() {
    let base = key(9).with_observation_kind_and_resource(
        CostObservationKind::EngineLocalRestore,
        ExecutionResource::CacheRestore {
            source_set_hash: 3,
            destination_device: 9,
            copy_backend: 0,
        },
    );
    let now = Instant::now();
    let mut estimates = Estimates::default();
    for _ in 0..MIN_SAMPLES {
        estimates.observe(base, 0.02, now);
    }
    assert!(estimates.predict(base, now).is_some());
    for resource in [
        ExecutionResource::CacheRestore {
            source_set_hash: 3,
            destination_device: 9,
            copy_backend: 1,
        },
        ExecutionResource::CacheRestore {
            source_set_hash: 4,
            destination_device: 9,
            copy_backend: 0,
        },
    ] {
        assert!(
            estimates
                .predict(
                    base.with_observation_kind_and_resource(base.kind, resource),
                    now
                )
                .is_none()
        );
    }
}
