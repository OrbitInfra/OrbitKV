#![cfg(target_os = "linux")]

use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use orbitkv_channel::{
    BootstrapServer, CacheClient, CallOptions, ChannelError, CommandCode, GrantState,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, RestoreRequest, RestoreState, TransportServer,
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
