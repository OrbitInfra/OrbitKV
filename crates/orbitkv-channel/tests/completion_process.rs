#![cfg(target_os = "linux")]

use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use orbitkv_channel::{
    BootstrapServer, CacheClient, CallOptions, ChannelError, CommandCode, GrantState,
    PublishRequest, RESPONSE_FLAG_REQUEST_CONSUMED, Response, RestoreRequest, RestoreState,
    TransportServer,
};

struct ChildGuard {
    child: Child,
    start_ticks: String,
}

impl ChildGuard {
    fn start(directory: &Path) -> Self {
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "shared_completion_child", "--nocapture"])
            .env("ORBITKV_COMPLETION_TEST_DIRECTORY", directory)
            .spawn()
            .unwrap();
        let start_ticks = process_start_ticks(child.id()).unwrap();
        let mut child = Self { child, start_ticks };
        let deadline = Instant::now() + Duration::from_secs(10);
        while !directory.join("ready").exists() {
            assert!(Instant::now() < deadline, "child did not start");
            assert!(
                child.child.try_wait().unwrap().is_none(),
                "child exited early"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        child
    }

    fn wait_for_success(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                assert!(process_start_ticks(self.child.id()).is_none());
                println!(
                    "owned_pid={} start_ticks={} exit=0 reaped=1 forced_cleanup=0",
                    self.child.id(),
                    self.start_ticks
                );
                return;
            }
            assert!(Instant::now() < deadline, "child did not exit");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            if process_start_ticks(self.child.id()).as_ref() == Some(&self.start_ticks) {
                eprintln!(
                    "owned_pid={} start_ticks={} forced_cleanup=1",
                    self.child.id(),
                    self.start_ticks
                );
                let _ = self.child.kill();
                let _ = self.child.wait();
            } else {
                eprintln!("owned process identity unavailable; refusing signal");
            }
        }
    }
}

fn process_start_ticks(pid: u32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    Some(
        stat.rsplit_once(')')?
            .1
            .split_whitespace()
            .nth(19)?
            .to_string(),
    )
}

#[test]
fn native_wait_consumes_success_and_error_across_processes_without_poll_rpc() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handles: Vec<_> = ["success", "failure"]
        .into_iter()
        .map(|instance| {
            client
                .start_restore(&RestoreRequest {
                    instance_id: instance.into(),
                    tp_rank: 0,
                    device_id: 0,
                    layer_groups: vec![],
                    loads: vec![],
                })
                .unwrap()
        })
        .collect();
    for &handle in &handles {
        assert_eq!(
            client.poll_restore(handle).unwrap().state,
            RestoreState::Pending
        );
    }
    std::fs::write(dir.path().join("complete"), b"").unwrap();
    assert_eq!(
        client
            .wait_restore(handles[0], Duration::from_secs(5))
            .unwrap()
            .state,
        RestoreState::Succeeded
    );
    let failed = client
        .wait_restore(handles[1], Duration::from_secs(5))
        .unwrap();
    assert_eq!(failed.state, RestoreState::Failed);
    assert_eq!(failed.message, "GPU copy failed after drain: 错误");
    assert!(
        client.poll_restore(handles[0]).is_err(),
        "terminal records are consumed once"
    );
    client.close();
    child.wait_for_success();
}

#[test]
fn pending_restore_reports_actual_peer_process_exit_without_fabricating_completion() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "pending-until-exit".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert!(client.channel().peer_alive().unwrap());
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    std::fs::write(dir.path().join("exit"), b"").unwrap();
    child.wait_for_success();
    assert!(!client.channel().peer_alive().unwrap());
    assert!(
        client
            .restore_completions_ready(Duration::from_secs(5))
            .unwrap()
    );
    let response = client.poll_restore(handle).unwrap();
    assert_eq!(response.state, RestoreState::Pending);
    assert!(response.message.contains("resources retained"));
    assert!(matches!(
        client.wait_restore(handle, Duration::from_millis(2)),
        Err(ChannelError::RestoreTimeout { .. })
    ));
    assert_eq!(
        client
            .channel()
            .restore_completions()
            .state(handle.operation_id)
            .unwrap(),
        GrantState::Managed
    );
}

#[test]
fn local_grants_remain_pending_until_engine_drain_and_manager_reaping() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "local".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert_eq!(
        client.claim_local_restore(handle).unwrap().unwrap(),
        b"cross-process bounded plan"
    );
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    client.finish_local_restore(handle, Ok(()), None).unwrap();
    assert_eq!(
        client
            .wait_restore(handle, Duration::from_secs(5))
            .unwrap()
            .state,
        RestoreState::Succeeded
    );
    client.close();
    child.wait_for_success();
}

#[test]
fn manager_process_death_does_not_fence_active_engine_dma() {
    for instance in ["local", "local-parts"] {
        let dir = tempfile::tempdir().unwrap();
        let mut child = ChildGuard::start(dir.path());
        let client =
            CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
        let handle = client
            .start_restore(&RestoreRequest {
                instance_id: instance.into(),
                tp_rank: 0,
                device_id: 0,
                layer_groups: vec![],
                loads: vec![],
            })
            .unwrap();
        client.claim_local_restore(handle).unwrap().unwrap();
        std::fs::write(dir.path().join("exit"), b"").unwrap();
        child.wait_for_success();
        assert!(!client.channel().peer_alive().unwrap());
        assert_eq!(
            client.poll_restore(handle).unwrap().state,
            RestoreState::Pending
        );
        assert!(matches!(
            client.wait_restore(handle, Duration::from_millis(2)),
            Err(ChannelError::RestoreTimeout { .. })
        ));
        client.finish_local_restore(handle, Ok(()), None).unwrap();
        assert!(matches!(
            client.poll_restore(handle),
            Err(ChannelError::SessionRequiresReconnect)
        ));
    }
}

#[test]
fn peer_liveness_does_not_wait_for_reply_or_follow_replacement_manager() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "pending-until-exit".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    for _ in 0..8 {
        assert!(client.channel().peer_alive().unwrap());
        assert_eq!(
            client.poll_restore(handle).unwrap().state,
            RestoreState::Pending
        );
    }
    std::fs::write(dir.path().join("exit"), b"").unwrap();
    first.wait_for_success();
    assert!(!client.channel().peer_alive().unwrap());
    std::fs::remove_file(dir.path().join("ready")).unwrap();
    std::fs::remove_file(dir.path().join("exit")).unwrap();
    let mut replacement = ChildGuard::start(dir.path());
    let fresh =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    assert!(fresh.channel().peer_alive().unwrap());
    assert!(!client.channel().peer_alive().unwrap());
    fresh
        .start_restore(&RestoreRequest {
            instance_id: "pending-until-exit".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    std::fs::write(dir.path().join("exit"), b"").unwrap();
    replacement.wait_for_success();
}

#[test]
fn shared_completion_child() {
    let Some(directory) = std::env::var_os("ORBITKV_COMPLETION_TEST_DIRECTORY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let name = format!("orbitkv/test/completions/{}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let bootstrap =
        BootstrapServer::bind(directory.join("cache.sock"), &name, 91, 65536, 4096).unwrap();
    std::fs::write(directory.join("ready"), b"").unwrap();
    let mut session = bootstrap.accept().unwrap();
    let mut pending = Vec::new();
    let mut parts = std::collections::HashMap::new();
    let mut calls = 0;
    while session.is_alive().unwrap() {
        server
            .try_serve_for_epoch(91, |command| {
                assert_eq!(command.code, CommandCode::Restore);
                let slot = bootstrap
                    .descriptor_slot(command.descriptor.offset)
                    .unwrap();
                session
                    .validate_request(command.descriptor, command.arg0, slot)
                    .unwrap();
                // Any obsolete Poll command fails this decode. Only two Submits are allowed.
                let request =
                    RestoreRequest::decode(&bootstrap.arena().read(command.descriptor).unwrap())
                        .unwrap();
                calls += 1;
                assert!(calls <= 2);
                let id = command.arg1;
                session.completions().claim(id).unwrap();
                if matches!(request.instance_id.as_str(), "local" | "local-parts") {
                    let more = request.instance_id == "local-parts";
                    if more {
                        parts.insert(id, 2);
                    }
                    session
                        .completions()
                        .publish_local(id, b"cross-process bounded plan", more)
                        .unwrap();
                    session.notify().unwrap();
                } else {
                    session.completions().start_managed(id).unwrap();
                    pending.push((id, request.instance_id == "failure"));
                }
                let mut response = Response::ok(command);
                response.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                session.complete_request().unwrap();
                response
            })
            .unwrap();
        if directory.join("exit").exists() {
            assert_eq!(
                calls,
                usize::from(!directory.join("liveness-only").exists())
            );
            for (id, _) in &pending {
                assert_eq!(
                    session.completions().poll(*id).unwrap().state,
                    RestoreState::Pending
                );
            }
            return;
        }
        if directory.join("complete").exists() && !pending.is_empty() {
            for (id, failed) in pending.drain(..) {
                session
                    .completions()
                    .complete(
                        id,
                        if failed {
                            Err("GPU copy failed after drain: 错误".into())
                        } else {
                            Ok(())
                        },
                    )
                    .unwrap();
            }
            session.notify().unwrap();
        }
        for (id, state) in session.completions().manager_updates().unwrap() {
            match state {
                GrantState::Active | GrantState::ActiveMore => {
                    session.completions().release_plan(id).unwrap();
                }
                GrantState::PartDrained => {
                    session.completions().release_plan(id).unwrap();
                    session.completions().continue_local(id).unwrap();
                    let remaining = parts.get_mut(&id).unwrap();
                    *remaining -= 1;
                    session
                        .completions()
                        .publish_local(id, b"next part", *remaining != 0)
                        .unwrap();
                    session.notify().unwrap();
                }
                GrantState::Drained => session.completions().reap(id).unwrap(),
                _ => panic!("unexpected grant state: {state:?}"),
            }
        }
        std::thread::yield_now();
    }
    assert!(
        (1..=2).contains(&calls),
        "completion consumption must not issue Poll requests"
    );
}

#[test]
fn partitioned_grant_uses_one_cross_process_submission_and_terminal_result() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard::start(dir.path());
    let client =
        CacheClient::connect(dir.path().join("cache.sock"), CallOptions::default()).unwrap();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "local-parts".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    for part in 0..3 {
        let deadline = Instant::now() + Duration::from_secs(5);
        let plan = loop {
            if let Some(plan) = client.claim_local_restore(handle).unwrap() {
                break plan;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        assert_eq!(
            plan,
            if part == 0 {
                b"cross-process bounded plan".as_slice()
            } else {
                b"next part".as_slice()
            }
        );
        assert_eq!(
            client.poll_restore(handle).unwrap().state,
            RestoreState::Pending
        );
        assert_eq!(
            client.finish_local_restore(handle, Ok(()), None).unwrap(),
            part == 2
        );
    }
    assert_eq!(
        client
            .wait_restore(handle, Duration::from_secs(5))
            .unwrap()
            .state,
        RestoreState::Succeeded
    );
    client.close();
    child.wait_for_success();
}

#[test]
fn publish_peer_exit_does_not_release_a_submitted_source_owner() {
    for mode in [
        "exit-pending",
        "invalid-ack",
        "late-complete",
        "logger-unwind",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "publish_retention_child",
                "--nocapture",
            ])
            .env("ORBITKV_PUBLISH_RETENTION_DIRECTORY", directory.path())
            .env("ORBITKV_PUBLISH_RETENTION_MODE", mode)
            .spawn()
            .unwrap();
        let start_ticks = process_start_ticks(child.id()).unwrap();
        let mut child = ChildGuard { child, start_ticks };
        child.wait_for_success();
        println!("publish_control={mode} actual_wait=true exit=0");
    }
}

#[test]
#[ignore = "isolated CPU consumer for publish_peer_exit_does_not_release_a_submitted_source_owner"]
fn publish_retention_child() {
    use std::sync::{Arc, mpsc};
    let directory =
        std::path::PathBuf::from(std::env::var_os("ORBITKV_PUBLISH_RETENTION_DIRECTORY").unwrap());
    let mode = std::env::var("ORBITKV_PUBLISH_RETENTION_MODE").unwrap();
    let peer = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "publish_retention_peer",
            "--nocapture",
        ])
        .env("ORBITKV_PUBLISH_RETENTION_DIRECTORY", &directory)
        .env("ORBITKV_PUBLISH_RETENTION_MODE", &mode)
        .spawn()
        .unwrap();
    let start_ticks = process_start_ticks(peer.id()).unwrap();
    let mut peer = ChildGuard {
        child: peer,
        start_ticks,
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while !directory.join("ready").exists() {
        assert!(peer.child.try_wait().unwrap().is_none());
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    if mode == "logger-unwind" {
        struct PanickingLogger;
        impl log::Log for PanickingLogger {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }
            fn log(&self, _: &log::Record<'_>) {
                panic!("controlled post-submit logger unwind");
            }
            fn flush(&self) {}
        }
        static LOGGER: PanickingLogger = PanickingLogger;
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Warn);
    }
    let client = Arc::new(
        orbitkv_channel::ChannelClient::connect(
            directory.join("publish.sock"),
            CallOptions {
                timeout: Duration::from_millis(5),
                spin_iterations: 0,
            },
        )
        .unwrap(),
    );
    struct SourceOwner(std::path::PathBuf, Vec<u8>);
    impl Drop for SourceOwner {
        fn drop(&mut self) {
            assert_eq!(self.1, vec![37; 4096]);
            std::fs::write(&self.0, b"released").unwrap();
        }
    }
    let source = SourceOwner(directory.join("source-released"), vec![37; 4096]);
    let (returned, result) = mpsc::channel();
    let publisher_client = Arc::clone(&client);
    let publisher = std::thread::spawn(move || {
        let outcome = publisher_client.publish(
            1,
            &PublishRequest {
                instance_id: "test".into(),
                registration_generation: [1; 16],
                tp_rank: 0,
                pp_rank: 0,
                device_id: 0,
                layers: vec![],
            },
        );
        drop(source);
        returned.send(outcome.is_ok()).unwrap();
    });
    while !directory.join("admitted").exists() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!directory.join("source-released").exists());
    if mode == "late-complete" {
        assert!(result.recv_timeout(Duration::from_millis(20)).is_err());
        std::fs::write(directory.join("complete"), b"").unwrap();
        assert!(result.recv_timeout(Duration::from_secs(2)).unwrap());
        publisher.join().unwrap();
        assert!(directory.join("source-released").exists());
    } else {
        if mode == "logger-unwind" {
            assert!(result.recv_timeout(Duration::from_millis(30)).is_err());
            assert!(!directory.join("source-released").exists());
            std::fs::write(directory.join("exit"), b"").unwrap();
        }
        // The CPU peer submits no DMA. Its actual exit cannot release the simulated source.
        while peer.child.try_wait().unwrap().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(result.recv_timeout(Duration::from_millis(50)).is_err());
        assert!(!directory.join("source-released").exists());
        let (returned, other_result) = mpsc::channel();
        let other = std::thread::spawn(move || {
            returned.send(client.release(2, vec![])).unwrap();
        });
        let error = other_result
            .recv_timeout(Duration::from_secs(2))
            .expect("retained Publish must not hold the descriptor lock")
            .unwrap_err();
        if mode != "logger-unwind" {
            assert!(matches!(error, ChannelError::SessionRequiresReconnect));
        }
        other.join().unwrap();
        assert!(result.try_recv().is_err());
        assert!(!directory.join("source-released").exists());
    }
    peer.wait_for_success();
    println!("cpu_peer_mode={mode} actual_wait=true exit=0 no_DMA_submitted=true");
    // Isolated test process exit ends a deliberately parked CPU-only thread.
    // This is not a native drain or a production cleanup procedure.
}

#[test]
#[ignore = "CPU process peer fixture of publish_retention_child"]
fn publish_retention_peer() {
    let directory =
        std::path::PathBuf::from(std::env::var_os("ORBITKV_PUBLISH_RETENTION_DIRECTORY").unwrap());
    let mode = std::env::var("ORBITKV_PUBLISH_RETENTION_MODE").unwrap();
    let name = format!("orbitkv/test/publish-retention/{}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let bootstrap =
        BootstrapServer::bind(directory.join("publish.sock"), &name, 91, 65536, 4096).unwrap();
    std::fs::write(directory.join("ready"), b"").unwrap();
    let mut session = bootstrap.accept().unwrap();
    let mut admitted = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !admitted {
        assert!(Instant::now() < deadline);
        server
            .try_serve_deferred_for_epoch(91, |command, reply| {
                assert_eq!(command.code, CommandCode::Publish);
                let slot = bootstrap
                    .descriptor_slot(command.descriptor.offset)
                    .unwrap();
                session
                    .validate_request(command.descriptor, command.arg0, slot)
                    .unwrap();
                PublishRequest::decode(&bootstrap.arena().read(command.descriptor).unwrap())
                    .unwrap();
                admitted = true;
                std::fs::write(directory.join("admitted"), b"").unwrap();
                if mode == "exit-pending" || mode == "logger-unwind" {
                    if mode == "logger-unwind" {
                        while !directory.join("exit").exists() {
                            assert!(Instant::now() < deadline);
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    drop(reply);
                    return Ok(());
                }
                let mut response = Response::ok(command);
                if mode == "late-complete" {
                    while !directory.join("complete").exists() {
                        assert!(Instant::now() < deadline);
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    response.descriptor = bootstrap
                        .arena()
                        .write_response(command.descriptor, &[])
                        .unwrap();
                    session.complete_request().unwrap();
                    response.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                } else {
                    response.value1 = 0; // Deliberately missing the matched completion proof.
                }
                reply.send_and_notify(response, session.reply_notification_fd())
            })
            .unwrap();
        std::thread::yield_now();
    }
}
