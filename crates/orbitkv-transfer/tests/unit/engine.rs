use super::*;
use crate::types::{STATUS_CANCELED, STATUS_FAILED, STATUS_INVALID};

#[test]
fn fixed_tent_strings_are_bounded_even_without_a_terminator() {
    assert_eq!(fixed_c_string(&[b'a' as c_char, 0, b'b' as c_char]), "a");
    assert_eq!(fixed_c_string(&[b'a' as c_char, b'b' as c_char]), "ab");
}

#[test]
fn tcp_peers_move_merged_slices_and_count_each_byte_once() {
    unsafe {
        libc::setenv(c"MC_FORCE_TCP".as_ptr(), c"1".as_ptr(), 1);
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
    let registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory")
    };
    assert_eq!(registration.address(), base);
    let peer_registration = unsafe {
        peer.register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register peer memory")
    };
    let notification_generation = peer
        .open_notification_scope("loopback")
        .expect("open notification scope");
    let destination = unsafe { base.byte_add(4096) };
    let slices: Vec<_> = (0..4)
        .map(|index| TransferSlice {
            local: unsafe { base.byte_add(index * 1024) },
            remote_address: unsafe { destination.byte_add(index * 1024) }.as_ptr() as u64,
            length: 1024,
        })
        .collect();
    let transferred = engine
        .submit_and_notify(
            TransferOp::Write,
            &segment,
            &slices,
            Duration::from_secs(5),
            &Notification {
                name: "loopback".to_string(),
                message: "done".to_string(),
            },
        )
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
    let _ = engine.nic_load_stats().expect("query TENT NIC load stats");
    peer_registration
        .unregister()
        .expect("unregister peer memory");
    registration.unregister().expect("unregister memory");
    let registration = unsafe {
        engine
            .register_memory_owned(base, memory.len(), "cpu:0")
            .expect("register memory again")
    };
    #[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
    {
        engine
            .close_source(Duration::from_secs(1))
            .expect("physical CPU/TCP source receipt");
        assert!(unsafe { engine.register_memory_owned(base, memory.len(), "cpu:0") }.is_err());
        assert!(matches!(
            engine.submit_and_wait(
                TransferOp::Read,
                &segment,
                &slices,
                Duration::from_millis(1)
            ),
            Err(MooncakeError::BatchAdmissionExhausted)
        ));
        engine
            .close_source(Duration::ZERO)
            .expect("idempotent closed source receipt");
    }
    drop(engine);
    drop(registration);
    unsafe {
        libc::unsetenv(c"MC_FORCE_TCP".as_ptr());
    }
}

#[test]
fn engine_creation_restores_the_process_nic_filter() {
    let _guard = ENGINE_CREATE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let original = std::env::var_os("MC_TE_FILTERS");
    unsafe {
        libc::setenv(c"MC_TE_FILTERS".as_ptr(), c"original".as_ptr(), 1);
    }
    let filter =
        ScopedNicFilter::apply(&["temporary".to_string()]).expect("apply temporary NIC filter");
    assert_eq!(
        std::env::var_os("MC_TE_FILTERS").as_deref(),
        Some(std::ffi::OsStr::new("temporary"))
    );
    drop(filter);
    assert_eq!(
        std::env::var_os("MC_TE_FILTERS").as_deref(),
        Some(std::ffi::OsStr::new("original"))
    );
    unsafe {
        match original {
            Some(value) => {
                let value = CString::new(value.as_bytes()).expect("original environment");
                libc::setenv(c"MC_TE_FILTERS".as_ptr(), value.as_ptr(), 1);
            }
            None => {
                libc::unsetenv(c"MC_TE_FILTERS".as_ptr());
            }
        }
    }
}

#[test]
fn logical_failures_wait_for_actual_completion_and_preserve_first_error() {
    for operation in [
        "deadline",
        "tent_submit",
        "getTransferStatus",
        "tent_cancel_task",
    ] {
        let polls = std::cell::Cell::new(0);
        let cancellations = std::cell::Cell::new(0);
        let frees = std::cell::Cell::new(0);
        let failure = || MooncakeError::Operation {
            operation,
            status: -23,
        };
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
        );
        assert_eq!(cancellations.get(), 1);
        assert_eq!(polls.get(), 4);
        assert_eq!(frees.get(), 2);
        if operation == "deadline" {
            assert!(matches!(result, Err(MooncakeError::Timeout)));
        } else {
            assert!(
                matches!(result, Err(MooncakeError::Operation { operation: cause, status: -23 }) if cause == operation)
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
    );
    panic!("uncertain native task returned to caller");
}

#[test]
fn native_tcp_rejection_child() {
    if std::env::var_os("ORBITKV_TEST_NATIVE_TCP_REJECTION").is_none() {
        return;
    }
    unsafe {
        libc::setenv(c"MC_FORCE_TCP".as_ptr(), c"1".as_ptr(), 1);
    }
    let engine =
        Arc::new(TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let peer =
        Arc::new(TransferEngine::new("P2PHANDSHAKE", "127.0.0.1:0", "127.0.0.1", 0, &[]).unwrap());
    let mut source = vec![0xa5u8; 4096];
    let mut destination = vec![0x5au8; 4096];
    let source_ptr = NonNull::new(source.as_mut_ptr()).unwrap();
    let destination_ptr = NonNull::new(destination.as_mut_ptr()).unwrap();
    let _source_registration = unsafe {
        engine
            .register_memory_owned(source_ptr, source.len(), "cpu:0")
            .unwrap()
    };
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
            local: NonNull::new(local as *mut u8).unwrap(),
            remote_address: remote,
            length: 4096,
        };
        let result = transfer_engine.submit_and_wait(
            TransferOp::Read,
            &segment,
            &[slice],
            Duration::from_secs(5),
        );
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
    );
    assert!(result.is_err());
    assert!(freed.get());
}
