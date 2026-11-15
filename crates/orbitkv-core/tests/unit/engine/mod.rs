use super::*;

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
