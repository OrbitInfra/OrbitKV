use super::*;
use crate::PublishLayer;

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
