use super::*;

fn key(instance: &str, tp_rank: u32, pp_rank: u32, device: &str) -> ContextKey {
    ContextKey {
        instance_id: instance.into(),
        tp_rank,
        pp_rank,
        device_uuid: device.into(),
    }
}

#[test]
fn drop_context_removes_only_that_context() {
    let mut registry = CudaTensorRegistry::empty();
    let first = key("a", 0, 0, "GPU-0");
    let other = key("a", 0, 1, "GPU-1");
    registry
        .contexts
        .insert(first.clone(), ContextState::new(0));
    registry
        .contexts
        .insert(other.clone(), ContextState::new(1));
    assert_eq!(registry.drop_context(&first), 0);
    assert!(!registry.contexts.contains_key(&first));
    assert!(registry.contexts.contains_key(&other));
}

#[test]
fn register_layers_rejects_existing_context_before_materializing() {
    let mut registry = CudaTensorRegistry::empty();
    let context = key("a", 0, 0, "GPU-existing");
    registry
        .contexts
        .insert(context.clone(), ContextState::new(7));
    let err = registry
        .register_layers(&context, Vec::new())
        .expect_err("existing context must be rejected");
    let message = Python::attach(|py| err.value(py).to_string());
    assert!(message.contains("already registered"));
    assert_eq!(registry.contexts.get(&context).unwrap().device_id, 7);
}

#[test]
fn drop_instance_does_not_select_a_nested_instance_name() {
    let mut registry = CudaTensorRegistry::empty();
    let contexts = [
        key("a", 0, 0, "GPU-0"),
        key("a", 1, 0, "GPU-1"),
        key("a:b", 0, 0, "GPU-0"),
        key("a:b:c", 0, 0, "GPU-0"),
        key("aa", 0, 0, "GPU-0"),
    ];
    for context in &contexts {
        registry
            .contexts
            .insert(context.clone(), ContextState::new(0));
    }
    assert_eq!(registry.drop_instance("a"), 0);
    assert!(!registry.contexts.contains_key(&contexts[0]));
    assert!(!registry.contexts.contains_key(&contexts[1]));
    for context in &contexts[2..] {
        assert!(registry.contexts.contains_key(context));
    }
    assert_eq!(registry.drop_instance("a:b"), 0);
    assert!(!registry.contexts.contains_key(&contexts[2]));
    assert!(registry.contexts.contains_key(&contexts[3]));
    assert!(registry.contexts.contains_key(&contexts[4]));
    let replacement = key("a", 0, 0, "GPU-0");
    registry
        .contexts
        .insert(replacement.clone(), ContextState::new(9));
    assert_eq!(registry.contexts.get(&replacement).unwrap().device_id, 9);
    assert_eq!(registry.clear_and_count(), 0);
    assert!(registry.contexts.is_empty());
}

#[tokio::test]
async fn actor_cleanup_preserves_the_other_instances_registration() {
    let mut registry = CudaTensorRegistry::empty();
    let first = key("a", 0, 0, "GPU-0");
    let other = key("a:b", 0, 0, "GPU-0");
    registry.contexts.insert(first, ContextState::new(0));
    registry
        .contexts
        .insert(other.clone(), ContextState::new(0));
    let actor = RegistryHandle::spawn(registry);
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        assert_eq!(actor.drop_instance("a".into()).await, 0);
        let message = actor.register_layers(other, Vec::new()).await.unwrap_err();
        assert!(message.contains("already registered"), "{message}");
        assert_eq!(actor.clear().await, 0);
    })
    .await
    .expect("registry actor control is bounded");
}
