use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command as ProcessCommand};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

#[test]
fn deferred_publishers_allow_an_unrelated_request_to_finish_first() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("orbitkv/test/{}/{nonce}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let publishers: Vec<_> = (0..8)
        .map(|_| {
            (
                TransportClient::connect(&name).unwrap(),
                Arc::new(rustix::event::eventfd(0, rustix::event::EventfdFlags::NONBLOCK).unwrap()),
            )
        })
        .collect();
    let reply_notifications: Vec<_> = publishers
        .iter()
        .map(|(_, notification)| Arc::clone(notification))
        .collect();
    let second = TransportClient::connect(&name).unwrap();
    let options = CallOptions {
        timeout: Duration::from_secs(5),
        spin_iterations: 0,
    };
    let publish_options = CallOptions {
        timeout: Duration::from_millis(10),
        spin_iterations: 0,
    };
    let peer = rustix::process::pidfd_open(
        rustix::process::getpid(),
        rustix::process::PidfdFlags::empty(),
    )
    .unwrap();
    let (pending_tx, pending_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let server_thread = thread::spawn(move || {
        let mut pending = Vec::new();
        let mut served_second = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while !served_second && Instant::now() < deadline {
            let served = server
                .try_serve_deferred_for_epoch(7, |command, reply| {
                    if command.request_id <= 8 {
                        pending.push((command, reply));
                        if pending.len() == 8 {
                            pending_tx.send(()).unwrap();
                        }
                        Ok(())
                    } else {
                        served_second = true;
                        reply.send(Response::ok(command))
                    }
                })
                .unwrap();
            if !served {
                server.wait_for_request(Duration::from_millis(10)).unwrap();
            }
        }
        assert!(served_second);
        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        for (command, reply) in pending {
            let mut response = Response::ok(command);
            if command.request_id == 8 {
                response.status = crate::StatusCode::Invalid;
            }
            reply
                .send_and_notify(
                    response,
                    &reply_notifications[command.request_id as usize - 1],
                )
                .unwrap();
        }
    });
    let peer = std::sync::Arc::new(peer);
    let threads: Vec<_> = publishers
        .into_iter()
        .enumerate()
        .map(|(index, (client, notification))| {
            let peer = std::sync::Arc::clone(&peer);
            thread::spawn(move || {
                client.call_until_peer_exit(
                    Command::ping(index as u64 + 1, 7),
                    publish_options,
                    &peer,
                    &notification,
                )
            })
        })
        .collect();
    pending_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let response = second.call(Command::ping(9, 7), options).unwrap();
    assert_eq!(response.request_id, 9);
    thread::sleep(Duration::from_millis(30));
    release_tx.send(()).unwrap();
    for (index, thread) in threads.into_iter().enumerate() {
        let response = thread.join().unwrap().unwrap();
        assert_eq!(response.request_id, index as u64 + 1);
        assert_eq!(
            response.status,
            if index == 7 {
                crate::StatusCode::Invalid
            } else {
                crate::StatusCode::Ok
            }
        );
    }
    server_thread.join().unwrap();
}

fn service_name(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("orbitkv/test/{label}/{}/{nonce}", std::process::id())
}

#[test]
fn failed_doorbell_does_not_end_a_submitted_publish() {
    let name = service_name("lost-wake");
    let transport = TransportServer::bind(&name).unwrap();
    let client = TransportClient::connect(&name).unwrap();
    let TransportServer {
        server,
        request_listener,
        _service,
        _node,
    } = transport;
    drop(request_listener);
    let peer = rustix::process::pidfd_open(
        rustix::process::getpid(),
        rustix::process::PidfdFlags::empty(),
    )
    .unwrap();
    let (returned_tx, returned_rx) = mpsc::channel();
    let reply_notification =
        rustix::event::eventfd(32, rustix::event::EventfdFlags::NONBLOCK).unwrap();
    let producer = thread::spawn(move || {
        let result = client.call_until_peer_exit(
            Command::ping(1, 7),
            CallOptions {
                timeout: Duration::from_millis(10),
                spin_iterations: 0,
            },
            &peer,
            &reply_notification,
        );
        returned_tx.send(result).unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    let request = loop {
        if let Some(request) = server.receive().unwrap() {
            break request;
        }
        assert!(
            Instant::now() < deadline,
            "Publish never reached the request queue"
        );
        thread::yield_now();
    };
    // No listener exists, but the command was submitted. Neither the missing
    // notification nor the normal call deadline proves source-page safety.
    // Preexisting reply wakes are hints too, never evidence of completion.
    assert!(matches!(
        returned_rx.recv_timeout(Duration::from_millis(30)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let command = Command::decode(*request).unwrap();
    // Deliberately omit the reply wake: bounded rechecks still find the response.
    request.send_copy(Response::ok(command).encode()).unwrap();
    assert_eq!(
        returned_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()
            .request_id,
        1
    );
    producer.join().unwrap();
}

#[test]
fn missing_request_service_reports_the_effective_discovery_path() {
    let name = service_name("missing-requests");
    let error = match TransportClient::connect(&name) {
        Err(TransportError::Service(error)) => error,
        _ => panic!("opening an absent request service must fail"),
    };
    assert!(
        error.contains("RequestResponseOpenError::DoesNotExist"),
        "{error}"
    );
    assert!(error.contains(&name), "{error}");
    assert!(error.contains("pattern=RequestResponse"), "{error}");
    assert!(
        error.contains(process_config().unwrap().global.service_dir().as_str()),
        "{error}"
    );
    assert!(error.contains("metadata=Err("), "{error}");
    assert!(
        error.contains(&format!("pid={}", std::process::id())),
        "{error}"
    );
}

#[test]
fn request_doorbell_is_required_and_scoped_to_the_manager_service() {
    let name = service_name("missing-wake");
    let node = NodeBuilder::new()
        .config(process_config().unwrap())
        .create::<ThreadSafeIpcService>()
        .unwrap();
    let _bare_requests = node
        .service_builder(&name.as_str().try_into().unwrap())
        .request_response::<WireMessage, WireMessage>()
        .create()
        .unwrap();
    let error = match TransportClient::connect(&name) {
        Err(TransportError::Service(error)) => error,
        _ => panic!("opening a service without its request doorbell must fail"),
    };
    assert!(error.contains("EventOpenError::DoesNotExist"), "{error}");
    assert!(error.contains(&format!("{name}/requests")), "{error}");
    assert!(error.contains("pattern=Event"), "{error}");
    assert!(error.contains("static_config="), "{error}");
    assert!(error.contains("metadata=Err("), "{error}");
    assert!(error.contains("mount_namespace=Ok("), "{error}");

    let old_name = service_name("old-manager");
    let old_server = TransportServer::bind(&old_name).unwrap();
    let old_client = TransportClient::connect(&old_name).unwrap();
    drop(old_server);
    let new_name = service_name("new-manager");
    let new_server = TransportServer::bind(&new_name).unwrap();
    let new_client = TransportClient::connect(&new_name).unwrap();
    assert_eq!(old_client.request_notifier.notify().unwrap(), 0);
    assert_eq!(new_server.request_listener.try_wait(|_| {}).unwrap(), 0);
    assert_eq!(new_client.request_notifier.notify().unwrap(), 1);
    assert_eq!(new_server.request_listener.try_wait(|_| {}).unwrap(), 1);
}

#[test]
fn request_wait_handles_expired_deadlines_and_check_to_park_across_processes() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("control.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let name = service_name("process-wake");
    let mut child = ChildGuard(
        ProcessCommand::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "transport::tests::request_notification_child",
                "--nocapture",
            ])
            .env("ORBITKV_REQUEST_WAKE_TEST_SOCKET", &socket)
            .env("ORBITKV_REQUEST_WAKE_TEST_SERVICE", &name)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut control = loop {
        match listener.accept() {
            Ok((control, _)) => break control,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "child exited before attach"
                );
                assert!(Instant::now() < deadline, "child attach timed out");
                thread::sleep(Duration::from_millis(1));
            }
            Err(error) => panic!("accept notification child: {error}"),
        }
    };
    control
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    control
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let client = TransportClient::connect(&name).unwrap();
    for phase in 1..=3 {
        expect_signal(&mut control, phase);
        let pending = client
            .client
            .send_copy(Command::ping(phase as u64, 7).encode())
            .unwrap();
        // Phase 1: notification before wait, including coalescing. Phase 2:
        // queued data with its wake omitted. Phase 3: notification strictly
        // between the child's empty-queue check and its actual blocking wait.
        if phase != 2 {
            for _ in 0..32 {
                assert_eq!(client.request_notifier.notify().unwrap(), 1);
            }
        }
        control.write_all(b"G").unwrap();
        expect_signal(&mut control, b'C');
        assert_eq!(
            Response::decode(*pending.receive().unwrap().unwrap())
                .unwrap()
                .request_id,
            phase as u64
        );
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "notification child failed: {status}");
            break;
        }
        assert!(Instant::now() < deadline, "notification child did not exit");
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "private exec fixture of request_wait_handles_expired_deadlines_and_check_to_park_across_processes"]
fn request_notification_child() {
    let Some(socket) = std::env::var_os("ORBITKV_REQUEST_WAKE_TEST_SOCKET") else {
        return;
    };
    let name = std::env::var("ORBITKV_REQUEST_WAKE_TEST_SERVICE").unwrap();
    let server = TransportServer::bind(&name).unwrap();
    // No notifier exists yet. Linux interprets a zero SO_RCVTIMEO as an
    // unbounded receive, including positive durations truncated below 1us.
    // The parent's attach deadline kills this child if any wait blocks.
    for timeout in [
        Duration::ZERO,
        Duration::from_nanos(1),
        Duration::from_nanos(999),
        Duration::from_micros(1),
    ] {
        server.wait_for_request(timeout).unwrap();
    }
    let mut control = UnixStream::connect(socket).unwrap();
    control
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    control
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    for phase in 1..=3 {
        if phase == 3 {
            server.request_listener.try_wait(|_| {}).unwrap();
            assert!(!server.server.has_requests().unwrap());
        }
        control.write_all(&[phase]).unwrap();
        expect_signal(&mut control, b'G');
        if phase == 3 {
            // Exercise the exact library boundary after wait_for_request's
            // final predicate check, without adding a production test hook.
            assert_eq!(
                server
                    .request_listener
                    .timed_wait(|_| {}, Duration::from_secs(30))
                    .unwrap(),
                32
            );
        } else {
            server.wait_for_request(Duration::from_secs(30)).unwrap();
        }
        assert!(server.try_serve(Response::ok).unwrap());
        control.write_all(b"C").unwrap();
    }
}

fn expect_signal(control: &mut UnixStream, expected: u8) {
    let mut received = [0u8];
    control.read_exact(&mut received).unwrap();
    assert_eq!(received, [expected]);
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
