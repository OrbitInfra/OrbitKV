use super::*;
use crate::PublishLayer;

#[test]
fn publish_send_failures_preserve_only_possibly_submitted_sources() {
    use std::process::Command;
    use std::time::Instant;

    for mode in ["uncertain-delivery", "not-submitted"] {
        let directory = tempfile::tempdir_in("/var/tmp").unwrap();
        let root = directory.path().join("ipc");
        std::fs::create_dir(&root).unwrap();
        let config = directory.path().join("channel.toml");
        let input = format!(
            "[global]\nroot-path={:?}\nprefix=\"send_proof_\"\n\
             [defaults.request-response]\nenable-safe-overflow-for-requests=false\n",
            root.to_str().unwrap()
        );
        let parsed: iceoryx2::config::Config = toml::from_str(&input).unwrap();
        for name in [
            parsed.global.node.directory.as_str(),
            parsed.global.service.directory.as_str(),
        ] {
            std::fs::create_dir_all(root.join(name)).unwrap();
        }
        std::fs::write(&config, input).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "client::tests::publish_send_retention_child",
                "--nocapture",
            ])
            .env("ORBITKV_CHANNEL_CONFIG", config)
            .env("ORBITKV_SEND_PROOF_DIRECTORY", directory.path())
            .env("ORBITKV_SEND_PROOF_MODE", mode)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let status = child.wait().unwrap();
                panic!("Publish send control {mode} exceeded deadline: {status}");
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert!(status.success(), "Publish send control {mode}: {status}");
        println!("publish_send_control={mode} actual_wait=true exit=0 no_DMA=true");
    }
}

#[test]
#[ignore = "isolated CPU process for publish_send_failures_preserve_only_possibly_submitted_sources"]
#[allow(
    unsafe_code,
    reason = "The isolated CPU fixture injects only a thread-local queue wait error."
)]
fn publish_send_retention_child() {
    use std::sync::mpsc;
    use std::time::Instant;

    use iceoryx2::port::SendError;
    use iceoryx2::port::client::RequestSendError;
    use iceoryx2::prelude::*;

    use crate::{BootstrapServer, Response, TransportServer};

    struct SourceOwner(std::path::PathBuf);
    impl Drop for SourceOwner {
        fn drop(&mut self) {
            std::fs::write(&self.0, b"released").unwrap();
        }
    }

    let directory =
        std::path::PathBuf::from(std::env::var_os("ORBITKV_SEND_PROOF_DIRECTORY").unwrap());
    let mode = std::env::var("ORBITKV_SEND_PROOF_MODE").unwrap();
    let name = format!("orbitkv/test/send-proof/{}", std::process::id());
    let server = TransportServer::bind(&name).unwrap();
    let bootstrap =
        BootstrapServer::bind(directory.join("s.sock"), &name, 91, 65536, 4096).unwrap();
    let peer = std::thread::spawn(move || bootstrap.accept().unwrap());
    let client = Arc::new(
        ChannelClient::connect(
            directory.join("s.sock"),
            CallOptions {
                timeout: Duration::from_millis(5),
                spin_iterations: 0,
            },
        )
        .unwrap(),
    );
    let _session = peer.join().unwrap();
    let config: iceoryx2::config::Config = toml::from_str(
        &std::fs::read_to_string(std::env::var_os("ORBITKV_CHANNEL_CONFIG").unwrap()).unwrap(),
    )
    .unwrap();
    let node = NodeBuilder::new()
        .config(&config)
        .create::<ipc_threadsafe::Service>()
        .unwrap();
    let service = node
        .service_builder(&name.as_str().try_into().unwrap())
        .request_response::<crate::WireMessage, crate::WireMessage>()
        .open()
        .unwrap();
    assert!(!service.static_config().has_safe_overflow_for_requests());
    assert_eq!(service.static_config().max_active_requests_per_client(), 4);
    assert_eq!(service.static_config().max_servers(), 1);
    // Establish the actual receiver connection before filling its queue.
    assert!(!server.try_serve(Response::ok).unwrap());
    let request = PublishRequest {
        instance_id: "send-proof".into(),
        registration_generation: [1; 16],
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: vec![],
    };
    let marker = directory.join("source-released");
    let source = SourceOwner(marker.clone());

    if mode == "not-submitted" {
        let mut workers = Vec::new();
        let mut replies = Vec::new();
        for request_id in 1..=4 {
            let client = Arc::clone(&client);
            workers.push(std::thread::spawn(move || {
                client.client.call(
                    Command::ping(request_id, 91),
                    CallOptions {
                        timeout: Duration::from_secs(5),
                        spin_iterations: 0,
                    },
                )
            }));
            let deadline = Instant::now() + Duration::from_secs(2);
            while replies.len() < request_id as usize {
                assert!(Instant::now() < deadline);
                server
                    .try_serve_deferred_for_epoch(91, |command, reply| {
                        replies.push((command, reply));
                        Ok(())
                    })
                    .unwrap();
                std::thread::yield_now();
            }
        }
        let error = client.publish(9, &request).unwrap_err();
        assert_eq!(client.publish_progress().unwrap().active, 0);
        assert!(matches!(
            error,
            ChannelError::Transport(TransportError::RequestSend(
                RequestSendError::ExceedsMaxActiveRequests
            ))
        ));
        drop(source);
        assert!(marker.exists());
        assert!(!server.try_serve(Response::ok).unwrap());
        for (command, reply) in replies {
            reply.send(Response::ok(command)).unwrap();
        }
        for worker in workers {
            assert!(worker.join().unwrap().is_ok());
        }
    } else {
        assert_eq!(mode, "uncertain-delivery");
        for request_id in 1..=4 {
            assert!(matches!(
                client
                    .client
                    .call(Command::ping(request_id, 91), client.options),
                Err(TransportError::Timeout { .. })
            ));
        }
        let (attempted, attempt) = mpsc::channel();
        let (returned, result) = mpsc::channel();
        let publisher = Arc::clone(&client);
        std::thread::spawn(move || {
            let mut filter = [
                libc::sock_filter {
                    code: 0x20,
                    jt: 0,
                    jf: 0,
                    k: 0,
                },
                libc::sock_filter {
                    code: 0x15,
                    jt: 0,
                    jf: 1,
                    k: libc::SYS_clock_nanosleep as u32,
                },
                libc::sock_filter {
                    code: 0x06,
                    jt: 0,
                    jf: 0,
                    k: 0x00050000 | libc::EINTR as u32,
                },
                libc::sock_filter {
                    code: 0x06,
                    jt: 0,
                    jf: 0,
                    k: 0x7fff0000,
                },
            ];
            let program = libc::sock_fprog {
                len: filter.len() as u16,
                filter: filter.as_mut_ptr(),
            };
            // Limit the fault to this CPU fixture thread's adaptive queue wait.
            unsafe {
                assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
                assert_eq!(libc::prctl(libc::PR_SET_SECCOMP, 2, &program), 0);
            }
            let error = publisher
                .client
                .call(Command::ping(8, 91), publisher.options)
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    TransportError::RequestSend(RequestSendError::SendError(
                        SendError::InternalError
                    ))
                ),
                "{error:?}"
            );
            attempted.send(()).unwrap();
            let outcome = publisher.publish(9, &request);
            drop(source);
            returned.send(outcome).unwrap();
        });
        attempt.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(
            result.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(!marker.exists());
        let progress = client.publish_progress().unwrap();
        assert_eq!(progress.active, 1);
        assert_eq!(progress.retained, 1);
        assert!(progress.overdue);
        assert_eq!(progress.manager_alive, Some(true));
        let (completed, completion) = mpsc::channel();
        let other = std::thread::spawn(move || {
            completed.send(client.release(10, vec![])).unwrap();
        });
        assert!(matches!(
            completion
                .recv_timeout(Duration::from_secs(2))
                .expect("closed session must reject another caller without waiting for Publish"),
            Err(ChannelError::SessionRequiresReconnect)
        ));
        other.join().unwrap();
        assert!(matches!(result.try_recv(), Err(mpsc::TryRecvError::Empty)));
        assert!(!marker.exists());
        // Only this isolated no-DMA fixture process ends the parked caller.
    }
}

#[test]
fn publish_chunks_preserve_all_layers_and_pages_within_the_slot_limit() {
    let request = PublishRequest {
        instance_id: "qwen3-8b".to_string(),
        registration_generation: [1; 16],
        tp_rank: 2,
        pp_rank: 1,
        device_id: 3,
        layers: (0..36)
            .map(|layer| PublishLayer {
                layer_name: format!("model.layers.{layer}.attn"),
                block_ids: (0..192).collect(),
                block_hashes: (0..192).map(|page| vec![page as u8; 32]).collect(),
            })
            .collect(),
    };
    let capacity = crate::arena::DEFAULT_SLOT_CAPACITY;
    let payloads = publish_payloads(&request, capacity).unwrap();
    assert!(payloads.len() > 1);
    let mut reconstructed = request.clone();
    for layer in &mut reconstructed.layers {
        layer.block_ids.clear();
        layer.block_hashes.clear();
    }
    for payload in payloads {
        assert!(payload.len() <= capacity);
        let chunk = PublishRequest::decode(&payload).unwrap();
        assert_eq!(chunk.instance_id, request.instance_id);
        assert_eq!((chunk.tp_rank, chunk.pp_rank, chunk.device_id), (2, 1, 3));
        assert_eq!(chunk.layers.len(), request.layers.len());
        for (layer, original) in chunk.layers.into_iter().zip(&mut reconstructed.layers) {
            assert_eq!(layer.layer_name, original.layer_name);
            original.block_ids.extend(layer.block_ids);
            original.block_hashes.extend(layer.block_hashes);
        }
    }
    assert_eq!(reconstructed, request);
    assert!(publish_payloads(&request, 32).is_err());
    assert_eq!(
        publish_payloads(&request, usize::MAX).unwrap(),
        vec![request.encode().unwrap()]
    );
}

#[test]
fn publish_chunks_preserve_ragged_cache_groups() {
    let request = PublishRequest {
        instance_id: "hybrid".to_string(),
        registration_generation: [1; 16],
        tp_rank: 0,
        pp_rank: 0,
        device_id: 0,
        layers: (0..6)
            .map(|index| PublishLayer {
                layer_name: format!("layer.{index}"),
                block_ids: (0..index * 4).collect(),
                block_hashes: (0..index * 4)
                    .map(|page| vec![page as u8; 8 + page as usize])
                    .collect(),
            })
            .collect(),
    };
    let mut reconstructed = request.clone();
    for layer in &mut reconstructed.layers {
        layer.block_ids.clear();
        layer.block_hashes.clear();
    }
    for payload in publish_payloads(&request, 512).unwrap() {
        assert!(payload.len() <= 512);
        for layer in PublishRequest::decode(&payload).unwrap().layers {
            let original = reconstructed
                .layers
                .iter_mut()
                .find(|candidate| candidate.layer_name == layer.layer_name)
                .unwrap();
            original.block_ids.extend(layer.block_ids);
            original.block_hashes.extend(layer.block_hashes);
        }
    }
    assert_eq!(reconstructed, request);
}

#[test]
fn publish_chunk_boundaries_include_variable_hashes_and_only_active_layers() {
    let request = PublishRequest {
        instance_id: "m".into(),
        registration_generation: [1; 16],
        tp_rank: 1,
        pp_rank: 2,
        device_id: 3,
        layers: vec![
            PublishLayer {
                layer_name: "a".into(),
                block_ids: vec![10, 11, 12],
                block_hashes: vec![vec![1; 1], vec![2; 20], vec![3; 2]],
            },
            PublishLayer {
                layer_name: "b".into(),
                block_ids: vec![20, 21],
                block_hashes: vec![vec![4; 3], vec![5; 8]],
            },
            PublishLayer {
                layer_name: "empty".into(),
                block_ids: vec![],
                block_hashes: vec![],
            },
        ],
    };

    for (capacity, first_count) in [(127, 2), (126, 1)] {
        let payloads = publish_payloads(&request, capacity).unwrap();
        assert_eq!(payloads.len(), 2);
        let chunks = payloads
            .iter()
            .map(|payload| {
                assert!(payload.len() <= capacity);
                PublishRequest::decode(payload).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(chunks[0].layers[0].block_ids.len(), first_count);
        assert_eq!(chunks[0].layers[1].block_ids.len(), first_count);
        for original in &request.layers[..2] {
            let layers = chunks
                .iter()
                .flat_map(|chunk| &chunk.layers)
                .filter(|layer| layer.layer_name == original.layer_name)
                .collect::<Vec<_>>();
            assert_eq!(
                layers
                    .iter()
                    .flat_map(|layer| &layer.block_ids)
                    .copied()
                    .collect::<Vec<_>>(),
                original.block_ids
            );
            assert_eq!(
                layers
                    .iter()
                    .flat_map(|layer| &layer.block_hashes)
                    .collect::<Vec<_>>(),
                original.block_hashes.iter().collect::<Vec<_>>()
            );
        }
        assert!(chunks.iter().all(|chunk| {
            chunk.instance_id == "m"
                && (chunk.tp_rank, chunk.pp_rank, chunk.device_id) == (1, 2, 3)
                && chunk.layers.iter().all(|layer| !layer.block_ids.is_empty())
        }));
    }

    let full = request.encode().unwrap();
    assert_eq!(publish_payloads(&request, full.len()).unwrap(), vec![full]);
    assert!(matches!(
        publish_payloads(&request, 82),
        Err(ChannelError::Bootstrap(BootstrapError::Arena(
            crate::ArenaError::PayloadTooLarge {
                len: 83,
                capacity: 82
            }
        )))
    ));
}

#[test]
fn query_target_export_accepts_only_bounded_payload_and_never_arena_descriptors() {
    use crate::lifecycle::{LIFECYCLE_HEADER_BYTES, send_lifecycle_fds};
    use crate::{BootstrapServer, TransportServer};
    use std::io::Read;
    use std::os::fd::AsFd;
    use std::sync::atomic::AtomicU64;
    use std::thread;

    static NAMES: AtomicU64 = AtomicU64::new(1);
    let cases = [
        (LifecycleCommand::ExportQueryTarget, false, false, true),
        (LifecycleCommand::ExportQueryTarget, true, false, false),
        (LifecycleCommand::ExportQueryTarget, false, true, false),
        (LifecycleCommand::Health, false, false, false),
        (LifecycleCommand::Session, false, false, false),
        (LifecycleCommand::Unregister, false, false, false),
    ];
    for (command, descriptor, oversized, accepted) in cases {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("lifecycle.sock");
        let name = format!(
            "orbitkv/test/lifecycle/{}/{}",
            std::process::id(),
            NAMES.fetch_add(1, Ordering::Relaxed)
        );
        let bootstrap = BootstrapServer::bind(&socket, &name, 81, 64 * 1024, 4096).unwrap();
        let _transport = TransportServer::bind(&name).unwrap();
        thread::scope(|scope| {
            let peer = scope.spawn(|| {
                let session = bootstrap.accept().unwrap();
                let mut stream = session.stream();
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut header = [0; LIFECYCLE_HEADER_BYTES];
                stream.read_exact(&mut header).unwrap();
                let request = LifecycleHeader::decode(header).unwrap();
                assert_eq!(request.code, command as u16);
                assert_eq!(request.epoch, 81);
                let file = tempfile::tempfile().unwrap();
                let fds = if descriptor {
                    vec![file.as_fd()]
                } else {
                    Vec::new()
                };
                send_lifecycle_fds(stream, &fds).unwrap();
                let payload = b"query-target-capability";
                stream
                    .write_all(
                        &LifecycleHeader {
                            code: 0,
                            epoch: 81,
                            payload_len: if oversized {
                                MAX_QUERY_TARGET_PAYLOAD + 1
                            } else {
                                payload.len()
                            },
                        }
                        .encode()
                        .unwrap(),
                    )
                    .unwrap();
                if !oversized {
                    stream.write_all(payload).unwrap();
                }
                if accepted {
                    stream.read_exact(&mut header).unwrap();
                    assert_eq!(
                        LifecycleHeader::decode(header).unwrap().code,
                        LifecycleCommand::Health as u16
                    );
                    send_lifecycle_fds(stream, &[]).unwrap();
                    stream
                        .write_all(
                            &LifecycleHeader {
                                code: 0,
                                epoch: 81,
                                payload_len: 0,
                            }
                            .encode()
                            .unwrap(),
                        )
                        .unwrap();
                }
            });
            let client = ChannelClient::connect(&socket, CallOptions::default()).unwrap();
            let result = client.lifecycle(command, &[]);
            if accepted {
                let reply = result.unwrap();
                assert_eq!(reply.payload, b"query-target-capability");
                assert!(reply.fds.is_empty());
                client.lifecycle(LifecycleCommand::Health, &[]).unwrap();
            } else {
                assert!(result.is_err(), "{command:?}/{descriptor}/{oversized}");
                assert!(matches!(
                    client.lifecycle(LifecycleCommand::Health, &[]),
                    Err(ChannelError::SessionRequiresReconnect)
                ));
            }
            peer.join().unwrap();
        });
    }
}
