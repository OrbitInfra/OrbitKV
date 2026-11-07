use super::*;
#[cfg(not(feature = "cuda-13"))]
use crate::P2P_METADATA;
use crate::types::{STATUS_CANCELED, STATUS_FAILED, STATUS_INVALID};

#[test]
#[cfg(not(feature = "cuda-13"))]
#[ignore = "requires a frozen CPU TENT build and its matching native unregister control library"]
fn failed_native_unregister_preserves_the_rust_registration_until_retry() {
    struct Library(NonNull<c_void>);
    impl Drop for Library {
        fn drop(&mut self) {
            // SAFETY: the fixture symbols and all native engines are gone before this sole
            // dlclose of the owned handle.
            unsafe { libc::dlclose(self.0.as_ptr()) };
        }
    }
    let path = CString::new(
        std::env::var("ORBITKV_UNREGISTER_CONTROL_LIBRARY")
            .expect("set the matching native control library path"),
    )
    .unwrap();
    let library = Library(
        // SAFETY: the explicitly selected test fixture is a trusted library with the
        // audited control ABI; Library owns the successful handle.
        NonNull::new(unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) })
            .expect("load the matching native control library"),
    );
    let symbol = |name: &std::ffi::CStr| {
        // SAFETY: library is still loaded and name is a static terminated symbol;
        // the non-null function address is checked before conversion.
        let pointer = unsafe { libc::dlsym(library.0.as_ptr(), name.as_ptr()) };
        assert!(!pointer.is_null(), "missing native test symbol {name:?}");
        pointer
    };
    let install: unsafe extern "C" fn(native::NativeEngine) =
        // SAFETY: the fixture declares this symbol as extern C with the exact
        // annotated function signature; library outlives every use.
        unsafe { std::mem::transmute(symbol(c"orbitkv_test_install_unregister_control")) };
    let refuse: unsafe extern "C" fn(native::NativeEngine, libc::c_int) =
        // SAFETY: this fixture symbol has the annotated C signature and its
        // Library remains owned through each call.
        unsafe { std::mem::transmute(symbol(c"orbitkv_test_set_unregister_refusal")) };
    let refuse_registration: unsafe extern "C" fn(native::NativeEngine, libc::c_int) =
        // SAFETY: this fixture symbol has the annotated C signature and its
        // Library remains owned through each call.
        unsafe { std::mem::transmute(symbol(c"orbitkv_test_set_registration_refusal")) };
    let state: unsafe extern "C" fn(native::NativeEngine, *mut u32, *mut u32) -> libc::c_int =
        // SAFETY: the fixture signature matches both writable count outputs;
        // its defining library remains loaded.
        unsafe { std::mem::transmute(symbol(c"orbitkv_test_unregister_state")) };
    let mut backing = vec![0xa5; 4096];
    let engine =
        Arc::new(TransferEngine::new(P2P_METADATA, "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let native = engine.native;
    // SAFETY: native is a live fixture-backed engine, and the typed control
    // symbols only mutate its test transport state.
    unsafe {
        install(native);
        refuse_registration(native, 1);
    }
    let address = NonNull::new(backing.as_mut_ptr()).unwrap();
    assert!(
        // SAFETY: backing owns the full range throughout this rejected registration;
        // no transfer or conflicting host access is in flight.
        unsafe { engine.register_memory_owned(address, backing.len(), "cpu:0") }.is_err(),
        "actual C ABI provider refusal must not return a Rust registration token"
    );
    let (mut registrations, mut removals) = (0, 0);
    assert_eq!(
        // SAFETY: native and its fixture library remain owned; both output counters
        // are exclusive writable Rust variables.
        unsafe { state(native, &mut registrations, &mut removals) },
        0
    );
    assert_eq!(registrations, 1);
    // SAFETY: native is a live fixture engine and this typed control only
    // changes its registration-refusal state.
    unsafe { refuse_registration(native, 0) };
    let removals_before_success = removals;
    // SAFETY: backing is retained until explicit unregister completes; the
    // registration token retains its actual engine owner.
    let mut registration = unsafe {
        engine
            .register_memory_owned(
                NonNull::new(backing.as_mut_ptr()).unwrap(),
                backing.len(),
                "cpu:0",
            )
            .unwrap()
    };
    let retained_engine = Arc::downgrade(&engine);
    // SAFETY: the live typed fixture engine accepts this test-only refusal flag;
    // no pointer or allocation ownership is transferred.
    unsafe { refuse(native, 1) };
    assert!(registration.try_unregister().is_err());
    assert!(registration.active);
    assert_eq!(
        // SAFETY: the fixture engine is live and both output counters are writable.
        unsafe { state(native, &mut registrations, &mut removals) },
        1
    );
    assert_eq!((registrations, removals), (2, removals_before_success + 1));
    drop(engine);
    assert_eq!(
        retained_engine.strong_count(),
        1,
        "token must retain its native engine"
    );
    // SAFETY: the typed fixture function only updates the live engine's test flag.
    unsafe { refuse(native, 0) };
    registration.try_unregister().unwrap();
    assert!(!registration.active);
    assert_eq!(
        // SAFETY: the fixture engine is retained by registration and both counters
        // are valid writable output.
        unsafe { state(native, &mut registrations, &mut removals) },
        0
    );
    assert_eq!((registrations, removals), (2, removals_before_success + 2));
    registration.try_unregister().unwrap();
    assert_eq!(
        // SAFETY: registration still owns the engine; the typed fixture function
        // only reads its state into the two writable counters.
        unsafe { state(native, &mut registrations, &mut removals) },
        0
    );
    assert_eq!(
        removals,
        removals_before_success + 2,
        "completed token must not unregister again"
    );
    assert!(backing.iter().all(|&byte| byte == 0xa5));
    drop(registration);
    assert!(retained_engine.upgrade().is_none());
    drop(library);
}

#[test]
fn fixed_tent_strings_are_bounded_even_without_a_terminator() {
    assert_eq!(fixed_c_string(&[b'a' as c_char, 0, b'b' as c_char]), "a");
    assert_eq!(fixed_c_string(&[b'a' as c_char, b'b' as c_char]), "ab");
}

#[test]
fn tcp_peers_move_merged_slices_and_count_each_byte_once() {
    if !in_tcp_process("engine::tests::tcp_peers_move_merged_slices_and_count_each_byte_once") {
        return;
    }
    let engine = Arc::new(
        TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[])
            .expect("create Mooncake TENT"),
    );
    let peer = Arc::new(
        TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[])
            .expect("create peer TENT"),
    );
    let segment = peer.local_segment_name().expect("peer segment");
    let mut memory = vec![0u8; 8192];
    memory[..4096].fill(0xa5);
    let base = NonNull::new(memory.as_mut_ptr()).expect("memory pointer");
    // SAFETY: memory owns the entire registered range through both engines'
    // transfers and the explicit unregister below.
    let mut registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory")
    };
    assert_eq!(registration.address(), base);
    // SAFETY: the same backing remains live; test transfers use disjoint source
    // and destination halves and registrations are released only after drain.
    let mut peer_registration = unsafe {
        peer.register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register peer memory")
    };
    let notification_generation = peer
        .open_notification_scope("loopback")
        .expect("open notification scope");
    // SAFETY: memory has 8192 bytes; offset 4096 is within its live allocation.
    let destination = unsafe { base.byte_add(4096) };
    let mut slices: Vec<_> = (0..4)
        .map(|index| TransferSlice {
            priority: crate::TransferPriority::High,
            // SAFETY: index is below four, keeping each 1024-byte source inside the
            // first half of the retained 8192-byte allocation.
            local: unsafe { base.byte_add(index * 1024) },
            // SAFETY: these offsets stay in the disjoint destination half of the same
            // retained allocation; no host access occurs while transfer runs.
            remote_address: unsafe { destination.byte_add(index * 1024) }.as_ptr() as u64,
            length: 1024,
        })
        .collect();
    // SAFETY: both registrations retain the disjoint memory halves and the
    // fixture reads or reuses bytes only after the blocking drain returns.
    let transferred = unsafe {
        engine.submit_and_notify(
            TransferOp::Write,
            &segment,
            &slices,
            Duration::from_secs(5),
            &Notification {
                name: "loopback".to_string(),
                message: "done".to_string(),
            },
            None,
        )
    }
    .expect("peer write");
    assert_eq!(
        transferred, 4096,
        "merged task counters must not be added twice"
    );
    assert_eq!(&memory[..4096], &memory[4096..]);
    assert_eq!(
        peer.wait_for_notification(
            "loopback",
            notification_generation,
            &[("done".to_string(), 1)],
            Duration::from_secs(5),
        )
        .expect("wait for notification"),
        Some("done".to_string())
    );
    peer.close_notification_scope("loopback", notification_generation);
    let cancelled_generation = engine
        .open_notification_scope("cancelled")
        .expect("open cancelled notification scope");
    let waiter_engine = Arc::clone(&engine);
    let waiter = std::thread::spawn(move || {
        waiter_engine.wait_for_notification(
            "cancelled",
            cancelled_generation,
            &[("done".to_string(), 1)],
            Duration::from_secs(5),
        )
    });
    std::thread::sleep(Duration::from_millis(10));
    engine.close_notification_scope("cancelled", cancelled_generation);
    assert_eq!(waiter.join().expect("join waiter").unwrap(), None);
    for priority in [crate::TransferPriority::Low, crate::TransferPriority::High] {
        engine.invalidate_segment(&segment).unwrap();
        engine.invalidate_segment(&segment).unwrap();
        assert!(!engine.segments.lock().unwrap().contains_key(&segment));
        for slice in &mut slices {
            slice.priority = priority;
        }
        memory[..4096].fill(0);
        assert_eq!(
            // SAFETY: both registered halves remain owned and disjoint during this
            // blocking read; host inspection follows its physical drain.
            unsafe {
                engine.submit_and_wait_cancellable(
                    TransferOp::Read,
                    &segment,
                    &slices,
                    Duration::from_secs(5),
                    None,
                    &std::sync::atomic::AtomicBool::new(false),
                )
            }
            .unwrap(),
            4096
        );
        assert_eq!(&memory[..4096], &memory[4096..]);
        assert!(
            engine.segments.lock().unwrap()[&segment].is_some(),
            "TENT close remains reversible"
        );
    }
    let _ = engine.nic_load_stats().expect("query TENT NIC load stats");
    peer_registration
        .try_unregister()
        .expect("unregister peer memory");
    registration.try_unregister().expect("unregister memory");
    // SAFETY: memory remains allocated and unused by native work until this
    // registration is explicitly released below.
    let mut registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory again")
    };
    #[cfg(not(feature = "cuda-13"))]
    {
        engine
            .close_source(Duration::from_secs(1))
            .expect("physical CPU/TCP source receipt");
        // SAFETY: the live backing is retained even though closed-source admission
        // rejects this new registration before it can escape.
        assert!(unsafe { engine.register_memory_owned(base, memory.len(), "cpu:0") }.is_err());
        assert!(matches!(
            // SAFETY: the original backing/registration remain live while closed-source
            // admission rejects the batch; the test does not treat cancel as drain.
            unsafe {
                engine.submit_and_wait(
                    TransferOp::Read,
                    &segment,
                    &slices,
                    Duration::from_millis(1),
                    None,
                )
            },
            Err(MooncakeError::BatchAdmissionExhausted)
        ));
        engine
            .close_source(Duration::ZERO)
            .expect("idempotent closed source receipt");
    }
    registration.try_unregister().expect("unregister memory");
    drop(engine);
    drop(registration);
}

// Set process inputs before the test harness/native workers start, not via
// setenv from a running multithreaded test process.
fn in_tcp_process(name: &str) -> bool {
    if std::env::var("ORBITKV_TEST_TCP_PROCESS").as_deref() == Ok(name) {
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture"])
        .env("ORBITKV_TEST_TCP_PROCESS", name)
        .env("MC_FORCE_TCP", "1")
        .env("MC_TE_FILTERS", "original")
        .env_remove("MC_TENT_CONF")
        .status()
        .unwrap();
    assert!(status.success(), "{name}: {status}");
    false
}

#[test]
fn engine_creation_preserves_the_process_nic_filter() {
    if !in_tcp_process("engine::tests::engine_creation_preserves_the_process_nic_filter") {
        return;
    }
    for nics in [vec!["temporary".into()], Vec::new()] {
        let engine =
            TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &nics).unwrap();
        assert_eq!(std::env::var("MC_TE_FILTERS").unwrap(), "original");
        engine.close_source(Duration::from_secs(1)).unwrap();
    }
}

#[test]
#[cfg(not(feature = "cuda-13"))]
fn grant_epoch_rejects_first_read_after_same_endpoint_restart() {
    if !in_tcp_process("engine::tests::grant_epoch_rejects_first_read_after_same_endpoint_restart")
    {
        return;
    }
    let source_engine =
        Arc::new(TransferEngine::new(P2P_METADATA, "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let endpoint = source_engine.local_segment_name().unwrap();
    let port: u64 = endpoint.rsplit_once(':').unwrap().1.parse().unwrap();
    let grant_epoch = source_engine.source_epoch().to_string();
    let mut source = vec![0u8; 8192];
    for (i, byte) in source.iter_mut().enumerate() {
        *byte = ((i * 17 + 31) % 251) as u8;
    }
    let mut destination = vec![0x5au8; source.len()];
    let source_pointer = NonNull::new(source.as_mut_ptr()).unwrap();
    let destination_pointer = NonNull::new(destination.as_mut_ptr()).unwrap();
    // SAFETY: source stays allocated and registered until its engine proves
    // source drain, including the same-address epoch replacement control.
    let mut source_registration = unsafe {
        source_engine
            .register_memory_owned(source_pointer, source.len(), "cpu:0")
            .unwrap()
    };
    let requester =
        Arc::new(TransferEngine::new(P2P_METADATA, "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    // SAFETY: destination remains exclusively available and registered until
    // all blocking READs have drained.
    let mut target_registration = unsafe {
        requester
            .register_memory_owned(destination_pointer, destination.len(), "cpu:0")
            .unwrap()
    };
    requester.open_segment(&endpoint).unwrap();
    let slices = [TransferSlice {
        priority: crate::TransferPriority::High,
        local: destination_pointer,
        remote_address: source_pointer.as_ptr() as u64,
        length: source.len(),
    }];
    // SAFETY: source/destination backing and registrations remain owned for
    // the blocking READ; the oracle runs only after successful drain.
    unsafe {
        requester.submit_and_wait(
            TransferOp::Read,
            &endpoint,
            &slices,
            Duration::from_secs(2),
            Some(&grant_epoch),
        )
    }
    .unwrap();
    assert_eq!(source, destination);
    destination.fill(0x5a);
    source_engine.close_source(Duration::from_secs(1)).unwrap();
    source_registration.try_unregister().unwrap();
    drop(source_registration);
    drop(source_engine);
    let replacement =
        Arc::new(TransferEngine::new(P2P_METADATA, &endpoint, "127.0.0.1", port, &[]).unwrap());
    assert_eq!(replacement.local_segment_name().unwrap(), endpoint);
    assert_ne!(replacement.source_epoch(), grant_epoch);
    for (i, byte) in source.iter_mut().enumerate() {
        *byte = ((i * 29 + 73) % 251) as u8;
    }
    // SAFETY: the old source epoch drained before these bytes were changed;
    // the replacement owner retains them through the new epoch's drain.
    let mut replacement_registration = unsafe {
        replacement
            .register_memory_owned(source_pointer, source.len(), "cpu:0")
            .unwrap()
    };
    let mutation = std::env::var_os("ORBITKV_TEST_CLEAR_GRANT_EPOCH").is_some();
    let expected = if mutation {
        None
    } else {
        Some(grant_epoch.as_str())
    };
    // SAFETY: the registered buffers are live and not accessed concurrently;
    // the stale epoch should reject before admission, with no replay.
    let old_read = unsafe {
        requester.submit_and_wait(
            TransferOp::Read,
            &endpoint,
            &slices,
            Duration::from_secs(2),
            expected,
        )
    };
    println!(
        "old_grant_result={old_read:?} sentinel_unchanged={} replacement_payload_observed={} mutation={mutation}",
        destination.iter().all(|byte| *byte == 0x5a),
        destination == source
    );
    assert!(
        old_read.is_err(),
        "stale grant captured a new source epoch and read replacement bytes"
    );
    assert!(destination.iter().all(|byte| *byte == 0x5a));
    {
        let admission = requester.admission.lock().unwrap();
        assert_eq!(admission.active, 0);
        assert_eq!(
            admission.request_bytes, 0,
            "pre-submit rejection retained a native batch"
        );
    }
    // SAFETY: the fresh epoch registration owns source and the requester owns
    // the disjoint destination until the blocking READ drains.
    let fresh_read = unsafe {
        requester.submit_and_wait(
            TransferOp::Read,
            &endpoint,
            &slices,
            Duration::from_secs(2),
            Some(replacement.source_epoch()),
        )
    }
    .unwrap();
    assert_eq!(fresh_read, source.len());
    assert_eq!(source, destination);
    replacement.close_source(Duration::from_secs(1)).unwrap();
    replacement_registration.try_unregister().unwrap();
    requester.close_source(Duration::from_secs(1)).unwrap();
    target_registration.try_unregister().unwrap();
    println!(
        "same_endpoint_same_address=true old_grant_rejected_before_batch=true fresh_payload_bytes={fresh_read} mutation={mutation}"
    );
}

#[test]
fn logical_failures_wait_for_actual_completion_and_preserve_first_error() {
    for (operation, status) in [
        ("deadline", -23),
        ("tent_submit", -23),
        ("tent_submit", -134),
        ("tent_submit", -135),
        ("tent_submit", -2),
        ("getTransferStatus", -23),
        ("tent_cancel_task", -23),
    ] {
        let polls = std::cell::Cell::new(0);
        let cancellations = std::cell::Cell::new(0);
        let frees = std::cell::Cell::new(0);
        let failure = || MooncakeError::Operation { operation, status };
        let result = drain_batch(
            2,
            Duration::ZERO,
            if operation == "tent_submit" {
                Err(failure())
            } else {
                Ok(())
            },
            |task| {
                if task == 0 {
                    return Ok(native::TransferStatus {
                        status: STATUS_COMPLETED,
                        transferred_bytes: 11,
                    });
                }
                polls.set(polls.get() + 1);
                if polls.get() == 1 && operation == "getTransferStatus" {
                    return Err(failure());
                }
                Ok(native::TransferStatus {
                    status: if polls.get() >= 4 {
                        STATUS_COMPLETED
                    } else {
                        STATUS_PENDING
                    },
                    transferred_bytes: 13,
                })
            },
            |_| {
                cancellations.set(cancellations.get() + 1);
                if operation == "tent_cancel_task" {
                    Err(failure())
                } else {
                    Ok(())
                }
            },
            || panic!("failed batch must not read completion bytes"),
            || {
                assert_eq!(
                    polls.get(),
                    4,
                    "keep both buffers until every task completes"
                );
                frees.set(frees.get() + 1);
                frees.get() == 2
            },
            |_, state, _| assert_eq!(state, STATUS_PENDING),
            || false,
        );
        assert_eq!(cancellations.get(), 1);
        assert_eq!(polls.get(), 4);
        assert_eq!(frees.get(), 2);
        if operation == "deadline" {
            assert!(matches!(result, Err(MooncakeError::Timeout)));
        } else {
            assert!(
                matches!(result, Err(MooncakeError::Operation { operation: cause, status: code }) if cause == operation && code == status)
            );
        }
    }
}

#[test]
fn native_retention_child() {
    let Ok(state) = std::env::var("ORBITKV_TEST_NATIVE_FAILURE_STATE") else {
        return;
    };
    let state = state.parse::<i32>().expect("test native state");
    struct CallerBuffers;
    impl Drop for CallerBuffers {
        fn drop(&mut self) {
            println!("caller_buffers_dropped");
        }
    }
    let _buffers = CallerBuffers;
    let first = MooncakeError::Operation {
        operation: "partial_submit",
        status: -23,
    };
    let _result = drain_batch(
        2,
        Duration::ZERO,
        Err(first),
        |task| {
            Ok(native::TransferStatus {
                status: if task == 0 { state } else { STATUS_COMPLETED },
                transferred_bytes: 0,
            })
        },
        |_| Ok(()),
        || panic!("cannot read bytes of uncertain batch"),
        || {
            println!("native_free_called");
            true
        },
        |task, observed, cause| {
            assert_eq!(task, 0);
            assert_eq!(observed, state);
            assert!(matches!(
                cause,
                Some(MooncakeError::Operation {
                    operation: "partial_submit",
                    status: -23
                })
            ));
            println!("retained_native_state={state}");
        },
        || false,
    );
    panic!("uncertain native task returned to caller");
}

#[test]
fn native_tcp_rejection_child() {
    if std::env::var_os("ORBITKV_TEST_NATIVE_TCP_REJECTION").is_none() {
        return;
    }
    let engine =
        Arc::new(TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let peer =
        Arc::new(TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let mut source = vec![0xa5u8; 4096];
    let mut destination = vec![0x5au8; 4096];
    let source_ptr = NonNull::new(source.as_mut_ptr()).unwrap();
    let destination_ptr = NonNull::new(destination.as_mut_ptr()).unwrap();
    // SAFETY: the child retains source and its registration indefinitely after
    // the intentionally rejected native request; it never frees on timeout.
    let _source_registration = unsafe {
        engine
            .register_memory_owned(source_ptr, source.len(), "cpu:0")
            .unwrap()
    };
    // SAFETY: destination is retained by this child through every native
    // operation, including the deliberately invalid remote address control.
    let _destination_registration = unsafe {
        peer.register_memory_owned(destination_ptr, destination.len(), "cpu:0")
            .unwrap()
    };
    let segment = peer.local_segment_name().unwrap();
    let transfer_engine = Arc::clone(&engine);
    let local = source_ptr.as_ptr() as usize;
    let remote = destination_ptr.as_ptr() as u64 + 4096;
    std::thread::spawn(move || {
        let slice = TransferSlice {
            priority: crate::TransferPriority::High,
            local: NonNull::new(local as *mut u8).unwrap(),
            remote_address: remote,
            length: 4096,
        };
        // SAFETY: the parent fixture keeps the registered local backing alive;
        // the deliberately unregistered remote range is rejected by native authority.
        // The child parks with all owners if physical completion remains uncertain.
        let result = unsafe {
            transfer_engine.submit_and_wait(
                TransferOp::Read,
                &segment,
                &[slice],
                Duration::from_secs(5),
                None,
            )
        };
        println!("native_call_returned={result:?}");
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let admission = engine.admission.lock().unwrap();
        if admission.active == 1 && admission.retained == 1 {
            drop(admission);
            std::thread::sleep(Duration::from_millis(100));
            println!("retained_native_tcp_rejection");
            loop {
                std::thread::park_timeout(Duration::from_secs(60));
            }
        }
        drop(admission);
        assert!(
            Instant::now() < deadline,
            "native rejected request never admitted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn native_failures_hold_caller_buffers_and_never_call_free() {
    use std::io::BufRead;
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    let mut cases = [
        STATUS_CANCELED,
        STATUS_FAILED,
        STATUS_INVALID,
        STATUS_TIMEOUT,
        -1,
        99,
    ]
    .into_iter()
    .map(|state| {
        (
            "engine::tests::native_retention_child",
            "ORBITKV_TEST_NATIVE_FAILURE_STATE",
            state.to_string(),
            format!("retained_native_state={state}"),
        )
    })
    .collect::<Vec<_>>();
    cases.push((
        "engine::tests::native_tcp_rejection_child",
        "ORBITKV_TEST_NATIVE_TCP_REJECTION",
        "1".into(),
        "retained_native_tcp_rejection".into(),
    ));
    cases.push((
        "engine::tests::native_panic_child",
        "ORBITKV_TEST_NATIVE_PANIC",
        "1".into(),
        "panic_injected_after_submit".into(),
    ));
    for (test, environment, value, expected) in cases {
        let mut child = Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", test, "--nocapture"])
            .env(environment, &value)
            .env("MC_FORCE_TCP", "1")
            .env_remove("MC_TENT_CONF")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn owned CPU control");
        let output = child.stdout.take().expect("control stdout");
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(output).lines() {
                if tx.send(line.expect("read control stdout")).is_err() {
                    break;
                }
            }
        });
        let evidence = (|| {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let line = rx
                    .recv_timeout(remaining)
                    .map_err(|error| error.to_string())?;
                if line.contains("native_free_called")
                    || line.contains("caller_buffers_dropped")
                    || line.contains("native_call_returned")
                    || line.contains("source_release_sent")
                {
                    return Err(format!("premature release: {line}"));
                }
                if line == expected {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
            if child.try_wait().expect("inspect control child").is_some() {
                return Err("retention returned prematurely".to_string());
            }
            if rx.try_iter().any(|line| {
                line.contains("native_free_called")
                    || line.contains("caller_buffers_dropped")
                    || line.contains("native_call_returned")
                    || line.contains("source_release_sent")
            }) {
                return Err("released while native termination unproven".to_string());
            }
            Ok(())
        })();
        // Only this synthetic CPU child is killed to finish the negative control.
        // This is not a transport termination or graceful-cleanup proof.
        child.kill().expect("stop owned synthetic child");
        child.wait().expect("reap owned synthetic child");
        reader.join().expect("join control stdout reader");
        assert!(evidence.is_ok(), "case={test}/{value}: {evidence:?}");
    }
}

#[test]
fn native_panic_child() {
    if std::env::var_os("ORBITKV_TEST_NATIVE_PANIC").is_none() {
        return;
    }
    struct CallerBuffers;
    impl Drop for CallerBuffers {
        fn drop(&mut self) {
            println!("caller_buffers_dropped");
        }
    }
    struct SourceCompletion;
    impl Drop for SourceCompletion {
        fn drop(&mut self) {
            println!("source_release_sent");
        }
    }
    let _buffers = CallerBuffers;
    let _source = SourceCompletion;
    let admission = Mutex::new(BatchAdmission::default());
    let mut permit = BatchAdmission::acquire(&admission, 1, 4096).unwrap();
    let requests = [native::TransferStatus {
        status: STATUS_FAILED,
        transferred_bytes: 0,
    }];
    contain_native_unwind(|| {
        println!("panic_injected_after_submit");
        let _ = drain_batch(
            1,
            Duration::ZERO,
            Ok(()),
            |_| {
                Ok(native::TransferStatus {
                    status: requests[0].status,
                    transferred_bytes: 0,
                })
            },
            |_| Ok(()),
            || panic!("no failed batch bytes"),
            || {
                println!("native_free_called");
                true
            },
            |_, _, _| {
                permit.retain();
                panic!("injected logging/retention panic");
            },
            || false,
        );
    });
    println!("native_call_returned");
}

#[test]
fn batch_admission_retains_slots_and_bounds_request_memory() {
    let admission = Mutex::new(BatchAdmission::default());
    let mut owners = (0..MAX_NATIVE_BATCHES)
        .map(|_| BatchAdmission::acquire(&admission, 1, 4096).unwrap())
        .collect::<Vec<_>>();
    owners[0].retain();
    owners[0].retain();
    assert_eq!(admission.lock().unwrap().active, MAX_NATIVE_BATCHES);
    assert_eq!(admission.lock().unwrap().retained, 1);
    assert!(BatchAdmission::acquire(&admission, 1, 1).is_err());
    owners.pop();
    assert!(BatchAdmission::acquire(&admission, 1, 1).is_ok());
    drop(owners);
    assert_eq!(admission.lock().unwrap().active, 0);
    assert_eq!(admission.lock().unwrap().retained, 0);
    assert_eq!(admission.lock().unwrap().request_bytes, 0);
    assert_eq!(admission.lock().unwrap().logical_bytes, 0);
    assert!(BatchAdmission::acquire(&admission, usize::MAX, 1).is_err());
    assert!(
        BatchAdmission::acquire(
            &admission,
            MAX_NATIVE_REQUEST_BYTES / std::mem::size_of::<native::TransferRequest>() + 1,
            1
        )
        .is_err()
    );
    let max = BatchAdmission::acquire(&admission, 1, i64::MAX as u64).unwrap();
    assert!(BatchAdmission::acquire(&admission, 1, 1).is_err());
    drop(max);
}

#[test]
fn batch_drain_uses_batch_bytes_for_merged_tasks() {
    let rounds = std::cell::Cell::new(0);
    let result = drain_batch(
        2,
        Duration::from_secs(5),
        Ok(()),
        |task| {
            if task == 1 {
                rounds.set(rounds.get() + 1);
            }
            Ok(native::TransferStatus {
                status: if task == 0 || rounds.get() == 3 {
                    STATUS_COMPLETED
                } else {
                    STATUS_PENDING
                },
                transferred_bytes: 22,
            })
        },
        |_| panic!("successful batch must not be cancelled"),
        || {
            assert_eq!(rounds.get(), 3);
            Ok(22)
        },
        || rounds.get() == 3,
        |_, _, _| panic!("successful batch must not be retained"),
        || false,
    );
    assert_eq!(result.unwrap(), 22);
}

#[test]
fn batch_counter_failure_still_reclaims_completed_tasks() {
    let freed = std::cell::Cell::new(false);
    let result = drain_batch(
        1,
        Duration::MAX,
        Ok(()),
        |_| {
            Ok(native::TransferStatus {
                status: STATUS_COMPLETED,
                transferred_bytes: 11,
            })
        },
        |_| panic!("completed task must not be cancelled"),
        || {
            Err(MooncakeError::Operation {
                operation: "tent_overall_status",
                status: -1,
            })
        },
        || {
            freed.set(true);
            true
        },
        |_, state, _| assert_eq!(state, STATUS_PENDING),
        || false,
    );
    assert!(result.is_err());
    assert!(freed.get());
}

#[test]
fn external_cancellation_waits_for_completion_before_freeing_the_batch() {
    let polls = std::cell::Cell::new(0);
    let cancellations = std::cell::Cell::new(0);
    let freed = std::cell::Cell::new(false);
    let result = drain_batch(
        1,
        Duration::from_secs(1),
        Ok(()),
        |_| {
            polls.set(polls.get() + 1);
            Ok(native::TransferStatus {
                status: if polls.get() >= 3 {
                    STATUS_COMPLETED
                } else {
                    STATUS_PENDING
                },
                transferred_bytes: 8,
            })
        },
        |_| {
            cancellations.set(cancellations.get() + 1);
            Ok(())
        },
        || panic!("cancelled data is not a successful service sample"),
        || {
            assert_eq!(polls.get(), 3);
            freed.set(true);
            true
        },
        |_, state, _| {
            assert_eq!(state, STATUS_PENDING);
            assert!(!freed.get());
        },
        || polls.get() >= 1,
    );
    assert!(matches!(result, Err(MooncakeError::Cancelled)));
    assert!(freed.get());
    assert_eq!(cancellations.get(), 1);
}
