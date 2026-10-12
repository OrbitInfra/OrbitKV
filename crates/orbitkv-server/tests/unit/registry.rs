use super::*;

#[test]
fn drop_context_removes_only_that_context() {
    let mut registry = CudaTensorRegistry::empty();
    registry
        .contexts
        .insert("instance-a:tp0:pp0:dev0".to_string(), ContextState::new(0));
    registry
        .contexts
        .insert("instance-a:tp0:pp1:dev1".to_string(), ContextState::new(1));

    assert_eq!(registry.drop_context("instance-a:tp0:pp0:dev0"), 0);

    assert!(!registry.contexts.contains_key("instance-a:tp0:pp0:dev0"));
    assert!(registry.contexts.contains_key("instance-a:tp0:pp1:dev1"));
}

#[test]
fn register_layers_rejects_existing_context_before_materializing() {
    let mut registry = CudaTensorRegistry::empty();
    registry
        .contexts
        .insert("instance-a:tp0:pp0:dev0".to_string(), ContextState::new(7));

    let err = registry
        .register_layers("instance-a:tp0:pp0:dev0", "GPU-existing", Vec::new())
        .expect_err("existing context must be rejected");

    let message = Python::attach(|py| err.value(py).to_string());
    assert!(message.contains("already registered"));
    assert_eq!(
        registry
            .contexts
            .get("instance-a:tp0:pp0:dev0")
            .expect("existing context remains")
            .device_id,
        7
    );
}
