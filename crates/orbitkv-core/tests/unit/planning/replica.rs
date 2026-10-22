use std::sync::Arc;

use super::*;

#[test]
fn replacing_memory_evidence_never_pins_either_generation() {
    let mut replicas = ReplicaSet::new(StateKey::new("ns".into(), vec![1]));
    let old = Arc::new(SealedBlock::from_slots(Vec::new()));
    let new = Arc::new(SealedBlock::from_slots(Vec::new()));
    replicas.set_memory(Arc::downgrade(&old));
    replicas.set_memory(Arc::downgrade(&new));
    assert_eq!(replicas.replicas.len(), 1);
    assert_eq!(Arc::strong_count(&old), 1);
    assert_eq!(Arc::strong_count(&new), 1);
    assert!(replicas.is_available());
    drop(new);
    assert!(
        !replicas.is_available(),
        "an older live generation is not the selected evidence"
    );
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[test]
fn peer_evidence_is_bounded_by_runtime_identity_and_preserves_local_memory() {
    use orbitkv_state::{CacheOwner, ReplicaMedium};

    let mut replicas = ReplicaSet::new(StateKey::new("ns".into(), vec![1]));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    replicas.set_memory(Arc::downgrade(&block));
    let peers: Vec<_> = (0..DISCOVERY_MAX_REPLICAS_PER_MEDIUM + 1)
        .map(|i| ReplicaLocation {
            owner: CacheOwner {
                endpoint: "same-address".into(),
                incarnation: uuid::Uuid::from_u128(i as u128 + 1),
            },
            sequence: i as u64 + 1,
            metadata: orbitkv_state::ReplicaMetadata {
                medium: orbitkv_state::ReplicaMedium::Dram,
                representation: orbitkv_state::ReplicaRepresentation::Raw,
                stored_bytes: Some(4096),
            },
        })
        .collect();
    replicas.set_remote(DiscoveryCoverage::PartialHints, peers.clone());
    assert_eq!(
        replicas.remote_coverage(),
        Some(DiscoveryCoverage::PartialHints)
    );
    assert_eq!(
        replicas.peer(ReplicaMedium::Dram).count(),
        DISCOVERY_MAX_REPLICAS_PER_MEDIUM
    );
    assert_eq!(
        replicas.replicas.len(),
        DISCOVERY_MAX_REPLICAS_PER_MEDIUM + 1
    );
    assert_eq!(Arc::strong_count(&block), 1);
    replicas.reject_peer(&peers[0].owner);
    assert_eq!(
        replicas.peer(ReplicaMedium::Dram).count(),
        DISCOVERY_MAX_REPLICAS_PER_MEDIUM - 1
    );
    assert!(
        replicas
            .peer(ReplicaMedium::Dram)
            .all(|p| p.owner != peers[0].owner)
    );
    replicas.set_peers(vec![peers[1].clone(), peers[1].clone()]);
    assert_eq!(replicas.peer(ReplicaMedium::Dram).count(), 1);
    replicas.set_peers(vec![ReplicaLocation {
        metadata: orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Ssd,
            representation: orbitkv_state::ReplicaRepresentation::Raw,
            stored_bytes: Some(4096),
        },
        ..peers[1].clone()
    }]);
    assert_eq!(replicas.peer(ReplicaMedium::Dram).count(), 0);
    assert_eq!(replicas.peer(ReplicaMedium::Ssd).count(), 1);
    replicas.set_peers(vec![ReplicaLocation {
        metadata: orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Hbm,
            representation: orbitkv_state::ReplicaRepresentation::Raw,
            stored_bytes: Some(4096),
        },
        ..peers[1].clone()
    }]);
    assert_eq!(replicas.peer(ReplicaMedium::Dram).count(), 0);
    assert_eq!(replicas.peer(ReplicaMedium::Ssd).count(), 0);
    replicas.set_peers(Vec::new());
    assert!(
        replicas.is_available(),
        "peer refresh must preserve local evidence"
    );
    assert_eq!(replicas.replicas.len(), 1);
    drop(block);
    assert!(!replicas.is_available());
}
