use super::*;

#[test]
#[ignore = "requires CUDA and Mooncake TCP; run after all native builds"]
fn registration_preserves_pool_without_cuda_source_termination() {
    let pool = Arc::new(PinnedAllocator::new_global(1 << 20, 1, false, None));
    let memory = Arc::downgrade(&pool);
    let transport = MooncakeTransport::new(&[], pool.clone(), "127.0.0.1:0").unwrap();
    drop(pool);
    assert!(
        memory.upgrade().is_some(),
        "registered memory must outlive its original owner"
    );
    drop(transport);
    assert!(
        memory.upgrade().is_some(),
        "unsupported CUDA source termination must retain the pool"
    );
}

#[test]
#[ignore = "CPU owner integration: task-owned host-registration and TENT C ABI shims; no GPU qualification"]
fn actual_source_owner_preserves_partial_and_unproven_resources() {
    let scenario =
        std::env::var("ORBITKV_TEST_SOURCE_OWNER_CASE").expect("explicit CPU control case");
    assert!(matches!(
        scenario.as_str(),
        "normal" | "unsupported" | "still-owned" | "unregister-error" | "partial"
    ));
    let host_library =
        std::env::var("ORBITKV_TEST_HOST_SHIM").expect("explicit host-registration shim");
    let native_dir = std::env::var("ORBITKV_MOONCAKE_LIB_DIR").expect("explicit TENT shim bundle");
    let host = unsafe { libloading::Library::new(host_library).unwrap() };
    let native = unsafe {
        libloading::Library::new(Path::new(&native_dir).join("libtent_shared.so")).unwrap()
    };
    let counter = |library: &libloading::Library, name: &[u8]| -> u64 {
        let function = unsafe { library.get::<unsafe extern "C" fn() -> u64>(name).unwrap() };
        unsafe { function() }
    };
    let shards = if scenario == "partial" { 2 } else { 1 };
    let pool = Arc::new(PinnedAllocator::new_global(1 << 20, shards, false, None));
    let memory = Arc::downgrade(&pool);
    let result = MooncakeTransport::new(&[], pool.clone(), "127.0.0.1:0");
    let engine = result
        .as_ref()
        .ok()
        .map(|transport| Arc::downgrade(&transport.engine));
    let maps_path = std::env::var("ORBITKV_TEST_OWNER_MAPS").expect("external control maps path");
    std::fs::write(
        maps_path,
        std::fs::read_to_string("/proc/self/maps").unwrap(),
    )
    .unwrap();
    drop(pool);
    assert!(memory.upgrade().is_some());
    if scenario == "partial" {
        assert!(result.is_err(), "second actual register must fail");
    } else {
        assert!(result.is_ok());
    }
    drop(result);
    let retained = scenario != "normal";
    assert_eq!(memory.upgrade().is_some(), retained);
    if let Some(engine) = engine {
        assert_eq!(engine.upgrade().is_some(), retained);
    }
    let held = RETAINED_SOURCES.lock().unwrap();
    let budget = REGISTERED_POOLS.lock().unwrap();
    assert_eq!(held.len(), usize::from(retained));
    assert_eq!(budget.0, usize::from(retained));
    assert_eq!(budget.1 != 0, retained);
    assert_eq!(
        counter(&host, b"orbitkv_test_host_unregistered\0"),
        if retained { 0 } else { 1 }
    );
    assert_eq!(
        counter(&native, b"orbitkv_test_unregisters\0"),
        if scenario == "normal" || scenario == "unregister-error" {
            1
        } else {
            0
        }
    );
    assert_eq!(
        counter(&native, b"orbitkv_test_destroys\0"),
        if retained { 0 } else { 1 }
    );
    assert_eq!(counter(&native, b"orbitkv_test_registers\0"), shards as u64);
    if !retained {
        let native_done = counter(&native, b"orbitkv_test_unregister_ns\0");
        let backing_release = counter(&host, b"orbitkv_test_host_unregister_ns\0");
        assert!(
            native_done > 0 && backing_release >= native_done,
            "backing release preceded actual native unregister return"
        );
    }
    println!(
        "source_owner_case={scenario} held_pools={} charged_bytes={} CPU_host_registration_shim=true GPU_qualification=false",
        held.len(),
        budget.1
    );
}
