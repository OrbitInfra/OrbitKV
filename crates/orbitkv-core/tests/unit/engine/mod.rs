use super::*;

#[cfg(feature = "mooncake")]
#[tokio::test]
async fn mooncake_initialization_failure_is_returned_to_caller() {
    let membership = test_membership();
    let config = crate::EngineConfig {
        mooncake_nic_names: vec!["definitely-not-a-real-nic".to_string()],
        membership: Some(membership.clone()),
        global_index: Some(std::sync::Arc::new(orbitkv_catalog::GlobalIndex::new(
            membership,
            1 << 20,
            std::sync::Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
        ))),
        inventory: Some(std::sync::Arc::new(crate::ResidencyInventory::new(
            16 << 10,
        ))),
        ..crate::EngineConfig::default()
    };

    let err = match OrbitKVEngine::new_with_config(1 << 20, false, config) {
        Ok(_) => panic!("engine startup must fail when Mooncake NIC init fails"),
        Err(err) => err.to_string(),
    };

    assert!(err.contains("Failed to initialise Mooncake TENT"), "{err}");
    assert!(err.contains("definitely-not-a-real-nic"), "{err}");
}

#[cfg(not(feature = "mooncake"))]
#[tokio::test]
async fn remote_transfer_config_is_ignored_without_feature() {
    let membership = test_membership();
    let config = crate::EngineConfig {
        mooncake_nic_names: vec!["mlx5_0".to_string()],
        membership: Some(membership.clone()),
        global_index: Some(std::sync::Arc::new(orbitkv_catalog::GlobalIndex::new(
            membership,
            1 << 20,
            std::sync::Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
        ))),
        inventory: Some(std::sync::Arc::new(crate::ResidencyInventory::new(
            16 << 10,
        ))),
        ..crate::EngineConfig::default()
    };

    let engine = OrbitKVEngine::new_with_config(1 << 20, false, config)
        .expect("a build without Mooncake should ignore remote transfer config");

    assert!(!engine.has_remote_transport());
}

fn test_membership() -> std::sync::Arc<orbitkv_catalog::MembershipView> {
    std::sync::Arc::new(orbitkv_catalog::MembershipView::new(
        orbitkv_state::CacheOwner {
            endpoint: "127.0.0.1:50055".into(),
            incarnation: uuid::Uuid::new_v4(),
        },
    ))
}

#[tokio::test]
async fn publish_requires_the_captured_registration_generation_before_binding_pages() {
    let engine = OrbitKVEngine::new_with_config(
        1 << 20,
        false,
        crate::EngineConfig {
            enable_numa_affinity: false,
            ..Default::default()
        },
    )
    .unwrap();
    let first = engine
        .get_or_create_instance("same-name", "ns", 1, 1, false)
        .unwrap();
    let old = first.generation();
    engine
        .unregister_instance_and_wait("same-name")
        .await
        .unwrap();
    let replacement = engine
        .get_or_create_instance("same-name", "ns", 1, 1, false)
        .unwrap();
    let fresh = replacement.generation();
    assert_ne!(old, fresh);
    for stale in [old, [0; 16]] {
        let error = engine
            .batch_save_kv_blocks_from_ipc("same-name", stale, 0, 0, 0, Vec::new())
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stale GPU registration generation"),
            "{error}"
        );
    }
    let allowed = engine
        .batch_save_kv_blocks_from_ipc("same-name", fresh, 0, 0, 0, Vec::new())
        .await
        .unwrap_err();
    assert!(
        !allowed
            .to_string()
            .contains("stale GPU registration generation")
    );
    assert!(
        allowed.to_string().contains("registration incomplete"),
        "fresh owner reaches normal topology validation: {allowed}"
    );
}
