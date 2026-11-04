use super::*;
use orbitkv_state::{ReplicaLocation, ReplicaRepresentation, StateKey};

fn metadata(medium: orbitkv_state::ReplicaMedium) -> orbitkv_state::ReplicaMetadata {
    orbitkv_state::ReplicaMetadata {
        medium,
        representation: orbitkv_state::ReplicaRepresentation::Raw,
        stored_bytes: Some(4096),
    }
}

fn row(hash: u8, owners: &[&str]) -> ReplicaSet {
    let mut row = ReplicaSet::new(StateKey::new("ns".into(), vec![hash]));
    row.set_peers(
        owners
            .iter()
            .map(|owner| ReplicaLocation {
                owner: CacheOwner {
                    endpoint: (*owner).into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence: u64::from(hash),
                metadata: metadata(orbitkv_state::ReplicaMedium::Dram),
            })
            .collect(),
    );
    row
}

#[test]
fn planner_selects_longest_cover_then_stable_owner_and_stops_at_gap() {
    let mut rows = vec![
        row(1, &["c", "b", "a"]),
        row(2, &["c", "b"]),
        row(3, &["d"]),
        row(4, &[]),
        row(5, &["a"]),
    ];
    let plan = FetchPlan::new(&mut rows, 1, PeerSource::Dram).unwrap();
    assert_eq!(plan.block_count(), 3);
    let first = plan.next_segment(0).unwrap();
    assert_eq!(first.owner.endpoint, "b");
    assert_eq!(first.stored_bytes, Some(8192));
    assert_eq!(first.representation, ReplicaRepresentation::Raw);
    assert_eq!(
        first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(plan.next_segment(2).unwrap().owner.endpoint, "d");
    assert!(
        plan.complete_cost_estimate_key().is_none(),
        "a multi-owner prefix has no single complete-route resource identity"
    );
    let mut rows = (0..=DISCOVERY_MAX_KEYS)
        .map(|_| row(1, &["a"]))
        .collect::<Vec<_>>();
    assert_eq!(
        FetchPlan::new(&mut rows, 1, PeerSource::Dram)
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        DISCOVERY_MAX_KEYS
    );
    rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES - 2];
    assert_eq!(
        FetchPlan::new(&mut rows, 1, PeerSource::Dram)
            .unwrap()
            .next_segment(0)
            .unwrap()
            .records
            .len(),
        1
    );

    rows[0].key.hash = vec![1; DISCOVERY_MAX_BYTES];
    assert!(
        FetchPlan::new(&mut rows, 1, PeerSource::Dram)
            .unwrap()
            .next_segment(0)
            .is_none()
    );
}

#[test]
fn unknown_peer_shape_stays_unknown_instead_of_becoming_zero() {
    let mut row = row(1, &["a"]);
    let location = row
        .peer(orbitkv_state::ReplicaMedium::Dram)
        .next()
        .unwrap()
        .clone();
    row.set_peers(vec![ReplicaLocation {
        metadata: orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Dram,
            representation: ReplicaRepresentation::Unknown,
            stored_bytes: None,
        },
        ..location
    }]);
    let mut rows = vec![row];
    let segment = FetchPlan::new(&mut rows, 1, PeerSource::Dram)
        .unwrap()
        .next_segment(0)
        .unwrap();
    assert_eq!(segment.stored_bytes, None);
    assert_eq!(segment.representation, ReplicaRepresentation::Unknown);
}

#[test]
fn selected_source_records_keep_its_versions_and_rejection_preserves_alternatives() {
    let mut rows = vec![row(1, &["a", "b"]), row(2, &["a", "b"])];
    rows[0].set_peers(vec![
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "a".into(),
                incarnation: uuid::Uuid::from_u128(1),
            },
            sequence: 11,
            metadata: metadata(orbitkv_state::ReplicaMedium::Dram),
        },
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "b".into(),
                incarnation: uuid::Uuid::from_u128(2),
            },
            sequence: 21,
            metadata: metadata(orbitkv_state::ReplicaMedium::Dram),
        },
    ]);
    rows[1].set_peers(vec![
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "a".into(),
                incarnation: uuid::Uuid::from_u128(1),
            },
            sequence: 12,
            metadata: metadata(orbitkv_state::ReplicaMedium::Dram),
        },
        ReplicaLocation {
            owner: CacheOwner {
                endpoint: "b".into(),
                incarnation: uuid::Uuid::from_u128(2),
            },
            sequence: 22,
            metadata: metadata(orbitkv_state::ReplicaMedium::Dram),
        },
    ]);
    let mut plan = FetchPlan::new(&mut rows, 2, PeerSource::Dram).unwrap();
    let first = plan.next_segment(0).unwrap();
    assert_eq!(
        first.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [11, 12]
    );
    plan.reject(0, &first);
    let next = plan.next_segment(0).unwrap();
    assert_eq!(next.owner.endpoint, "b");
    assert_eq!(next.owner.incarnation, uuid::Uuid::from_u128(2));
    assert_eq!(
        next.records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [21, 22]
    );
}

#[test]
fn peer_ssd_plan_is_explicit_and_never_mixes_source_media() {
    let owner = |endpoint: &str, medium| ReplicaLocation {
        owner: CacheOwner {
            endpoint: endpoint.into(),
            incarnation: uuid::Uuid::from_u128(endpoint.as_bytes()[0] as u128),
        },
        sequence: 7,
        metadata: metadata(medium),
    };
    let mut rows: Vec<_> = [1, 2]
        .into_iter()
        .map(|hash| {
            let mut row = ReplicaSet::new(StateKey::new("ns".into(), vec![hash]));
            row.set_peers(vec![
                owner("dram", orbitkv_state::ReplicaMedium::Dram),
                owner("ssd", orbitkv_state::ReplicaMedium::Ssd),
            ]);
            row
        })
        .collect();

    {
        let dram = FetchPlan::new(&mut rows, 2, PeerSource::Dram).unwrap();
        let segment = dram.next_segment(0).unwrap();
        assert_eq!(segment.source, PeerSource::Dram);
        assert!(segment.records.iter().all(|record| {
            record.metadata.unwrap().medium == orbitkv_state::ReplicaMedium::Dram
        }));
    }

    let ssd = FetchPlan::new(&mut rows, 2, PeerSource::Ssd).unwrap();
    let segment = ssd.next_segment(0).unwrap();
    assert_eq!(segment.source, PeerSource::Ssd);
    assert!(
        segment
            .records
            .iter()
            .all(|record| { record.metadata.unwrap().medium == orbitkv_state::ReplicaMedium::Ssd })
    );
}

#[test]
fn peer_owner_selection_is_opt_in_and_uses_matching_complete_route_evidence() {
    const CHILD: &str = "ORBITKV_TEST_PEER_OWNER_SELECTION_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let owners = ["a-stable", "z-measured"];
        let mut rows = vec![row(1, &owners), row(2, &owners)];
        let now = std::time::Instant::now();
        for (owner, seconds) in [("a-stable", 0.02), ("z-measured", 0.01)] {
            let location = rows[0]
                .peer(orbitkv_state::ReplicaMedium::Dram)
                .find(|replica| replica.owner.endpoint == owner)
                .unwrap();
            let key = crate::cost::CostEstimateKey::new(
                crate::cost::CostObservationKind::PeerDramHostReady,
                crate::cost::ExecutionResource::Peer(crate::cost::resource_id(&location.owner)),
                ReplicaRepresentation::Raw,
                8192,
                2,
            );
            for _ in 0..4 {
                crate::cost::observe_for_test(key, seconds, now);
            }
        }
        let plan = FetchPlan::new(&mut rows, 2, PeerSource::Dram).unwrap();
        assert_eq!(plan.next_segment(0).unwrap().owner.endpoint, "z-measured");
        let direct = plan.for_gpu_restore();
        assert_eq!(
            direct.next_segment(0).unwrap().owner.endpoint,
            "a-stable",
            "HostReady costs cannot rank a GPU restore"
        );
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "planning::peer::tests::peer_owner_selection_is_opt_in_and_uses_matching_complete_route_evidence",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("ORBITKV_COST_OBSERVATIONS", "1")
        .env("ORBITKV_COST_SELECTION", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
