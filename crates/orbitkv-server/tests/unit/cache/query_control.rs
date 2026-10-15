use super::*;
use orbitkv_channel::QueryBundleRequest;

fn interest() -> Interest {
    Interest {
        target: RegisteredQueryTarget {
            instance_id: "registered".into(),
            ..Default::default()
        },
        token: 2,
        expires: Instant::now() + INTEREST_TIMEOUT,
        submitted: HashMap::new(),
        replies: HashMap::new(),
    }
}
fn query(operation: u64) -> QueryCommand {
    QueryCommand::Submit(QueryBundleRequest {
        ticket: QueryTicket {
            operation_id: operation,
            revision: 1,
        },
        instance_id: "registered".into(),
        request_id: "request".into(),
        block_hashes: vec![vec![1; 32]],
        group_id: 0,
        wait_for_full_prefix: false,
        warmup: false,
        discover: false,
        materialize: false,
        prepare: false,
        demand: None,
    })
}
#[test]
fn exact_replay_is_idempotent_and_modified_queries_do_not_replace_identity() {
    let mut interest = interest();
    let first = query(1);
    let encoded = first.encode().unwrap();
    assert_eq!(
        interest.admit(&first, &encoded).unwrap(),
        interest.admit(&first, &encoded).unwrap()
    );
    let mut changed = first.clone();
    let QueryCommand::Submit(request) = &mut changed else {
        unreachable!()
    };
    request.block_hashes[0][0] ^= 1;
    assert_eq!(
        interest
            .admit(&changed, &changed.encode().unwrap())
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    assert_eq!(interest.submitted.len(), 1);
    interest.admit(&first, &encoded).unwrap();
}
#[test]
fn registered_prefix_admission_rejects_other_instances_modes_and_unbounded_hashes() {
    for invalid in 0..10 {
        let mut command = query(1);
        let QueryCommand::Submit(request) = &mut command else {
            unreachable!()
        };
        match invalid {
            0 => request.instance_id = "other".into(),
            1 => request.group_id = 1,
            2 => request.warmup = true,
            3 => request.discover = true,
            4 => request.prepare = true,
            5 => request.materialize = true,
            6 => request.wait_for_full_prefix = true,
            7 => request.request_id = "x".repeat(257),
            8 => request.block_hashes[0].clear(),
            9 => request.block_hashes[0] = vec![0; 129],
            _ => unreachable!(),
        }
        let mut interest = interest();
        assert!(
            interest.admit(&command, b"untrusted").is_err(),
            "case {invalid}"
        );
        assert!(
            interest.submitted.is_empty(),
            "rejection must not admit an owner"
        );
    }
}
#[test]
fn operation_book_is_bounded_without_evicting_live_identity() {
    let mut interest = interest();
    for operation in 1..=MAX_OPERATIONS as u64 {
        let command = query(operation);
        interest
            .admit(&command, &command.encode().unwrap())
            .unwrap();
    }
    let command = query(MAX_OPERATIONS as u64 + 1);
    assert_eq!(
        interest
            .admit(&command, &command.encode().unwrap())
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    let replay = query(1);
    interest.admit(&replay, &replay.encode().unwrap()).unwrap();
    assert_eq!(interest.submitted.len(), MAX_OPERATIONS);
}
#[test]
fn busy_and_early_cancellation_retire_replays_without_canceling_newer_admission() {
    for admitted in [false, true] {
        let mut interest = interest();
        let first = query(1);
        let encoded = first.encode().unwrap();
        if admitted {
            // The control book has admitted Submit even if shared PendingQueries
            // has no capacity and returns Busy without keeping an operation.
            interest.admit(&first, &encoded).unwrap();
        }
        let ticket = QueryTicket {
            operation_id: 1,
            revision: 1,
        };
        interest.cancel(ticket).unwrap();
        interest.cancel(ticket).unwrap();
        for command in [first.clone(), QueryCommand::Poll(ticket)] {
            assert_eq!(
                interest
                    .admit(&command, &command.encode().unwrap())
                    .unwrap_err()
                    .code(),
                tonic::Code::FailedPrecondition
            );
        }
        let QueryCommand::Submit(mut second) = first else {
            unreachable!()
        };
        second.ticket.revision = 2;
        let second = QueryCommand::Submit(second);
        assert!(interest.admit(&second, &second.encode().unwrap()).is_err());
        assert_eq!(interest.submitted.len(), 1);
    }
    let mut interest = interest();
    let first = query(1);
    interest.admit(&first, &first.encode().unwrap()).unwrap();
    let QueryCommand::Submit(mut second) = first else {
        unreachable!()
    };
    second.ticket.revision = 2;
    let newer = second.ticket;
    let second = QueryCommand::Submit(second);
    interest.admit(&second, &second.encode().unwrap()).unwrap();
    interest
        .cancel(QueryTicket {
            operation_id: 1,
            revision: 1,
        })
        .unwrap();
    interest.admit(&second, &second.encode().unwrap()).unwrap();
    let poll = QueryCommand::Poll(newer);
    interest.admit(&poll, &poll.encode().unwrap()).unwrap();
}

#[test]
fn cancellation_tombstones_share_the_bounded_operation_book() {
    let mut interest = interest();
    for operation_id in 1..=MAX_OPERATIONS as u64 {
        interest
            .cancel(QueryTicket {
                operation_id,
                revision: 1,
            })
            .unwrap();
    }
    assert_eq!(
        interest
            .cancel(QueryTicket {
                operation_id: MAX_OPERATIONS as u64 + 1,
                revision: 1
            })
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    interest
        .cancel(QueryTicket {
            operation_id: 1,
            revision: 1,
        })
        .unwrap();
    assert_eq!(interest.submitted.len(), MAX_OPERATIONS);
}

#[test]
fn node_identity_rejects_truncated_or_nil_ids() {
    for bytes in [vec![], vec![0; 15], vec![0; 16], vec![1; 17]] {
        assert_eq!(
            uuid(&bytes, "id").unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
    }
    let id = Uuid::new_v4();
    assert_eq!(uuid(id.as_bytes(), "id").unwrap(), id);
}

async fn prefix(
    client: &mut orbitkv_proto::proto::engine::cache_query_control_client::CacheQueryControlClient<
        tonic::transport::Channel,
    >,
    interest: &QueryInterest,
    operation: u64,
    hashes: &[Vec<u8>],
) -> QueryBundleResponse {
    let QueryCommand::Submit(mut request) = query(operation) else {
        unreachable!()
    };
    request.block_hashes = hashes.to_vec();
    let command = QueryCommand::Submit(request);
    let ticket = QueryTicket {
        operation_id: operation,
        revision: 1,
    };
    let mut response = client
        .execute(QueryControlExecuteRequest {
            interest: Some(interest.clone()),
            command: command.encode().unwrap(),
        })
        .await
        .unwrap()
        .into_inner();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let result = QueryBundleResponse::decode(&response.payload).unwrap();
        if result.outcome != QueryOutcomeCode::Loading {
            return result;
        }
        assert!(Instant::now() < deadline, "remote query did not complete");
        tokio::time::sleep(Duration::from_millis(1)).await;
        response = client
            .execute(QueryControlExecuteRequest {
                interest: Some(interest.clone()),
                command: QueryCommand::Poll(ticket).encode().unwrap(),
            })
            .await
            .unwrap()
            .into_inner();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires CUDA publication/restore and real DRAM/io_uring query-control services"]
async fn real_query_control_replays_claims_and_fences_replaced_registrations() {
    use cudarc::driver::{CudaContext, DevicePtr};
    use orbitkv_core::transfer::local::{LocalRestoreExecutor, LocalTensor, RawRestorePart};
    use orbitkv_core::{
        EngineConfig, LayerSave, QueryAdmission, QueryLeaseId, QueryMode, RestoreExecution,
        SsdBackend, SsdCacheConfig, TransferMode,
    };
    use orbitkv_proto::proto::engine::cache_query_control_client::CacheQueryControlClient;
    use orbitkv_proto::proto::engine::cache_query_control_server::CacheQueryControlServer;

    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let hashes: Vec<_> = (0..4).map(|index| vec![index + 10; 32]).collect();
    for ssd in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut nodes = Vec::new();
        for shard in 0..2 {
            let config = EngineConfig {
                query_budget_bytes: Some(1 << 20),
                query_instance_budget_bytes: Some(1 << 20),
                ssd_cache_config: ssd.then(|| SsdCacheConfig {
                    cache_paths: vec![directory.path().join(format!("shard-{shard}"))],
                    capacity_bytes: 4 << 20,
                    backend: SsdBackend::Uring,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let engine = Arc::new(OrbitKVEngine::new_with_config(8 << 20, false, config).unwrap());
            let expected: Vec<u8> = (0..16384)
                .map(|i| ((i * 13 + shard * 37) % 251) as u8)
                .collect();
            context.bind_to_thread().unwrap();
            // Framework CUDA IPC tensors use synchronous allocations; async
            // pool allocations expose a null allocation context to the driver.
            // SAFETY: the primary context is current, and CudaSlice takes sole
            // ownership of this 16 KiB allocation until all restores drain below.
            let mut tensor = unsafe {
                stream.upgrade_device_ptr::<u8>(
                    cudarc::driver::result::malloc_sync(expected.len()).unwrap(),
                    expected.len(),
                )
            };
            stream.memcpy_htod(&expected, &mut tensor).unwrap();
            stream.synchronize().unwrap();
            let address = tensor.device_ptr(&stream).0;
            let register = |engine: &OrbitKVEngine| {
                engine
                    .register_context_layer_batch_strided(
                        "registered",
                        &format!("tp-shard-{shard}"),
                        0,
                        0,
                        0,
                        1,
                        1,
                        &["attention".into()],
                        &[address],
                        &[16384],
                        &[4],
                        &[4096],
                        &[0],
                        &[1],
                        None,
                        None,
                        None,
                        TransferMode::Direct,
                        false,
                    )
                    .unwrap();
            };
            register(&engine);
            engine
                .batch_save_kv_blocks_from_ipc(
                    "registered",
                    engine
                        .registration_generation("registered")
                        .unwrap_or_default(),
                    0,
                    0,
                    0,
                    vec![LayerSave {
                        layer_name: "attention".into(),
                        block_ids: (0..3 - shard).collect(),
                        block_hashes: hashes[..3 - shard].to_vec(),
                    }],
                )
                .await
                .unwrap();
            engine.flush_all().await;
            if ssd {
                let evicted = engine.cleanup_memory_cache();
                assert!(
                    evicted.evicted_blocks > 0,
                    "SSD restore must not use retained DRAM"
                );
                assert_eq!(evicted.still_referenced_blocks, 0);
            }
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let hll = Arc::new(Mutex::new(MultiWindowHllTracker::new(
                vec![("test".into(), Duration::from_secs(60))],
                4,
            )));
            let control = QueryControlService::new(
                endpoint.clone(),
                Arc::clone(&engine),
                Arc::clone(&hll),
                Handle::current(),
            );
            let lifecycle = crate::cache::lifecycle::LifecycleService::new(
                Arc::clone(&engine),
                crate::registry::RegistryHandle::spawn(crate::registry::CudaTensorRegistry::empty()),
            );
            let socket = directory.path().join(format!("local-{shard}.sock"));
            let mut local_endpoint = crate::endpoint::ProcessEndpoint::start(
                format!("orbitkv/test/query-control/{shard}"),
                91,
                socket.clone(),
                1 << 20,
                1 << 16,
                Arc::clone(&engine),
                Handle::current(),
                hll,
                Arc::new(tokio::sync::Notify::new()),
                lifecycle.clone(),
                0,
                None,
                usize::MAX,
                Some(control.clone()),
            )
            .unwrap();
            let local_channel = tokio::task::spawn_blocking(move || {
                let channel = orbitkv_channel::ChannelClient::connect(
                    socket,
                    orbitkv_channel::CallOptions::default(),
                )
                .unwrap();
                let reply = channel
                    .lifecycle(
                        orbitkv_channel::lifecycle::LifecycleCommand::ExportQueryTarget,
                        &SessionRequest {
                            instance_id: "registered".into(),
                            namespace: format!("tp-shard-{shard}"),
                            tp_size: 1,
                            world_size: 1,
                        }
                        .encode_to_vec(),
                    )
                    .unwrap();
                assert!(reply.fds.is_empty());
                let target = RegisteredQueryTarget::decode(reply.payload.as_slice()).unwrap();
                channel
                    .lifecycle(orbitkv_channel::lifecycle::LifecycleCommand::Health, &[])
                    .unwrap();
                (channel, target)
            })
            .await
            .unwrap();
            let (local_channel, target) = local_channel;
            let (stop, stopped) = tokio::sync::oneshot::channel();
            let service = control.clone();
            let server = tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(CacheQueryControlServer::new(service))
                    .serve_with_incoming_shutdown(
                        tokio_stream::wrappers::TcpListenerStream::new(listener),
                        async {
                            let _ = stopped.await;
                        },
                    )
                    .await
                    .unwrap();
            });
            let mut client = CacheQueryControlClient::connect(endpoint).await.unwrap();
            let coordinator = Uuid::new_v4();
            let interest = client
                .open_interest(OpenQueryInterestRequest {
                    target: Some(target.clone()),
                    coordinator_incarnation: coordinator.as_bytes().to_vec(),
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                control.book.lock().interests[&uuid(&interest.id, "id").unwrap()].token % 2,
                0
            );
            let initial = prefix(&mut client, &interest, 1, &hashes).await;
            assert_eq!(initial.num_hit_blocks as usize, 3 - shard);
            assert_eq!(initial.lease.len(), 16);
            // Discard the first transport reply, then retrieve precisely the retained lease.
            let replay = client
                .execute(QueryControlExecuteRequest {
                    interest: Some(interest.clone()),
                    command: QueryCommand::Poll(QueryTicket {
                        operation_id: 1,
                        revision: 1,
                    })
                    .encode()
                    .unwrap(),
                })
                .await
                .unwrap()
                .into_inner();
            assert_eq!(
                QueryBundleResponse::decode(&replay.payload).unwrap(),
                initial
            );
            client
                .cancel(QueryControlCancelRequest {
                    interest: Some(interest.clone()),
                    operation_id: 1,
                    revision: 1,
                })
                .await
                .unwrap();
            assert!(
                client
                    .execute(QueryControlExecuteRequest {
                        interest: Some(interest.clone()),
                        command: QueryCommand::Poll(QueryTicket {
                            operation_id: 1,
                            revision: 1
                        })
                        .encode()
                        .unwrap(),
                    })
                    .await
                    .is_err()
            );
            let selected = prefix(&mut client, &interest, 2, &hashes[..2]).await;
            assert_eq!(selected.num_hit_blocks, 2);
            assert!(
                client
                    .claim(QueryControlClaimRequest {
                        interest: Some(interest.clone()),
                        operation_id: 2,
                        revision: 1,
                        lease: vec![0; 16],
                    })
                    .await
                    .is_err()
            );
            let claim = QueryControlClaimRequest {
                interest: Some(interest.clone()),
                operation_id: 2,
                revision: 1,
                lease: selected.lease.clone(),
            };
            let delivered = client.claim(claim.clone()).await.unwrap().into_inner();
            // The claim response can be lost; its retry does not create or consume another lease.
            assert_eq!(
                client.claim(claim).await.unwrap().into_inner().payload,
                delivered.payload
            );
            stream.memset_zeros(&mut tensor).unwrap();
            stream.synchronize().unwrap();
            match engine
                .restore(
                    "registered",
                    0,
                    0,
                    &[vec!["attention"]],
                    &[(
                        QueryLeaseId::from_bytes(&selected.lease).unwrap(),
                        vec![vec![Some(0), Some(1)]],
                    )],
                )
                .unwrap()
            {
                RestoreExecution::Local(mut grant) => {
                    let mut executor = LocalRestoreExecutor::new(
                        0,
                        vec![
                            LocalTensor::new(
                                "attention".into(),
                                address,
                                16384,
                                address as usize,
                                4,
                                4096,
                                0,
                                1,
                            )
                            .unwrap(),
                        ],
                        engine.payload_arenas().unwrap(),
                        TransferMode::Direct,
                    )
                    .unwrap();
                    loop {
                        let (encoded, more) = grant.encoded_plan();
                        executor
                            .execute(
                                &RawRestorePart::decode(encoded).unwrap(),
                                &mut Default::default(),
                                !more,
                                || {},
                                None,
                            )
                            .unwrap();
                        if !grant.advance_plan() {
                            break;
                        }
                    }
                    grant.finish(true, None);
                }
                RestoreExecution::Managed(receiver) => receiver.await.unwrap().result.unwrap(),
            }
            let actual = stream.clone_dtoh(&tensor).unwrap();
            assert_eq!(&actual[..8192], &expected[..8192]);
            assert!(actual[8192..].iter().all(|byte| *byte == 0));
            client.close(interest.clone()).await.unwrap();
            assert!(
                engine.has_query_registration("registered", &target.registration_generation),
                "closing query interest must not close GPU lifecycle"
            );
            assert!(
                client
                    .execute(QueryControlExecuteRequest {
                        interest: Some(interest),
                        command: QueryCommand::Poll(QueryTicket {
                            operation_id: 2,
                            revision: 1
                        })
                        .encode()
                        .unwrap(),
                    })
                    .await
                    .is_err()
            );
            let race_interest = client
                .open_interest(OpenQueryInterestRequest {
                    target: Some(target.clone()),
                    coordinator_incarnation: coordinator.as_bytes().to_vec(),
                })
                .await
                .unwrap()
                .into_inner();
            let race_reply = prefix(&mut client, &race_interest, 1, &hashes[..1]).await;
            let mut closer = client.clone();
            let (claim_result, close_result) = tokio::join!(
                client.claim(QueryControlClaimRequest {
                    interest: Some(race_interest.clone()),
                    operation_id: 1,
                    revision: 1,
                    lease: race_reply.lease,
                }),
                closer.close(race_interest),
            );
            close_result.unwrap();
            if let Err(error) = claim_result {
                assert!(matches!(
                    error.code(),
                    tonic::Code::NotFound | tonic::Code::FailedPrecondition
                ));
            }
            let admission = engine
                .reserve_query("registered", 0, 256, QueryMode::Demand)
                .unwrap();
            assert!(
                matches!(admission, QueryAdmission::Admitted(_)),
                "close must retire all unconsumed query bytes"
            );
            drop(admission);

            let live = client
                .open_interest(OpenQueryInterestRequest {
                    target: Some(target.clone()),
                    coordinator_incarnation: coordinator.as_bytes().to_vec(),
                })
                .await
                .unwrap()
                .into_inner();
            let unclaimed = prefix(&mut client, &live, 1, &hashes[..1]).await;
            let QueryAdmission::Admitted(old_reservation) = engine
                .reserve_query("registered", 0, 1, QueryMode::Demand)
                .unwrap()
            else {
                panic!("query fits")
            };
            let old_sources = engine
                .count_prefix_hit_blocks_with_prefetch(
                    "registered",
                    "late-finish-control",
                    &hashes[..1],
                    QueryMode::Demand,
                )
                .await
                .unwrap()
                .blocks;
            assert_eq!(old_sources.len(), 1);
            engine
                .unregister_instance_and_wait("registered")
                .await
                .unwrap();
            register(&engine);
            assert_ne!(
                engine
                    .query_registration("registered")
                    .unwrap()
                    .generation
                    .as_slice(),
                target.registration_generation
            );
            assert!(
                engine
                    .finish_query(
                        old_reservation,
                        orbitkv_core::QueryOwner {
                            session: 2,
                            operation: 3,
                            revision: 1,
                        },
                        old_sources,
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("query instance registration changed"),
                "late finish cannot adopt an identical replacement layout"
            );
            assert!(
                client
                    .claim(QueryControlClaimRequest {
                        interest: Some(live),
                        operation_id: 1,
                        revision: 1,
                        lease: unclaimed.lease.clone(),
                    })
                    .await
                    .is_err()
            );
            assert!(
                engine
                    .restore(
                        "registered",
                        0,
                        0,
                        &[vec!["attention"]],
                        &[(
                            QueryLeaseId::from_bytes(&unclaimed.lease).unwrap(),
                            vec![vec![Some(0)]],
                        )]
                    )
                    .is_err(),
                "old registration lease must not reach replacement pages"
            );
            assert!(
                client
                    .open_interest(OpenQueryInterestRequest {
                        target: Some(target),
                        coordinator_incarnation: coordinator.as_bytes().to_vec(),
                    })
                    .await
                    .is_err()
            );
            control.stop();
            PendingQueries::stop_and_drain(&control.queries, &engine)
                .await
                .unwrap();
            engine
                .unregister_instance_and_wait("registered")
                .await
                .unwrap();
            engine.flush_all().await;
            local_channel.close();
            local_endpoint.stop_admission_and_drain_publishes().await;
            local_endpoint.stop_queries_and_drain().await.unwrap();
            local_endpoint.stop_lifecycle_and_drain_connections().await;
            lifecycle.shutdown().await.unwrap();
            local_endpoint.stop();
            stop.send(()).unwrap();
            server.await.unwrap();
            nodes.push(tensor);
        }
        drop(nodes);
    }
}
