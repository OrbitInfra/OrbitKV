use super::*;

#[cfg(not(feature = "orbite"))]
#[tokio::test]
async fn uncompiled_remote_transfer_config_is_rejected() {
    let membership = test_membership();
    let config = crate::EngineConfig {
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

    let error = OrbitKVEngine::new_with_config(1 << 20, false, config)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("orbite build feature"), "{error}");
}

#[cfg(not(feature = "orbite"))]
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
