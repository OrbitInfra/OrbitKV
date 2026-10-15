use super::*;
use orbitkv_core::EngineConfig;

fn request(operation_id: u64, revision: u64) -> QueryBundleRequest {
    QueryBundleRequest {
        ticket: QueryTicket {
            operation_id,
            revision,
        },
        instance_id: "model".into(),
        request_id: "same-request".into(),
        block_hashes: vec![vec![revision as u8]],
        group_id: 0,
        wait_for_full_prefix: false,
        warmup: false,
        discover: false,
        materialize: false,
        prepare: false,
        demand: None,
    }
}

fn engine() -> Arc<OrbitKVEngine> {
    Arc::new(OrbitKVEngine::new_with_config(1 << 20, false, EngineConfig::default()).unwrap())
}

fn tracker() -> Arc<Mutex<MultiWindowHllTracker>> {
    Arc::new(Mutex::new(MultiWindowHllTracker::new(
        vec![("test".into(), Duration::from_secs(60))],
        4,
    )))
}

#[test]
fn malformed_selected_demand_is_rejected_before_query_admission() {
    use orbitkv_state::{RecoveryDemand, TokenRange};
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    let _held = Arc::clone(&queries.capacity)
        .try_acquire_many_owned(MAX_ACTIVE_QUERIES as u32)
        .unwrap();
    let span = TokenRange { start: 0, end: 16 };
    let demand = RecoveryDemand {
        page_tokens: 16,
        span,
        groups: vec![(0, span)],
    };
    let mut missing = request(1, 1);
    missing.materialize = true;
    let mut unselected = request(2, 1);
    unselected.demand = Some(demand.clone());
    let mut malformed = request(3, 1);
    malformed.materialize = true;
    malformed.demand = Some(RecoveryDemand {
        page_tokens: 0,
        ..demand.clone()
    });
    let mut wrong_count = request(4, 1);
    wrong_count.materialize = true;
    wrong_count.block_hashes.push(vec![2]);
    wrong_count.demand = Some(demand);
    let mut auxiliary_prefix = request(5, 1);
    auxiliary_prefix.prepare = true;
    auxiliary_prefix.group_id = 1;
    let mut waiting_prefix = request(6, 1);
    waiting_prefix.prepare = true;
    waiting_prefix.wait_for_full_prefix = true;
    for request in [
        missing,
        unselected,
        malformed,
        wrong_count,
        auxiliary_prefix,
        waiting_prefix,
    ] {
        let error = queries
            .execute(
                1,
                QueryCommand::Submit(request),
                &engine,
                runtime.handle(),
                &hll,
            )
            .err()
            .expect("invalid demand must fail before waiting for capacity");
        assert!(matches!(error, EngineError::InvalidArgument(_)), "{error}");
        assert!(queries.pending.is_empty());
        assert!(queries.sessions.is_empty());
    }
}

#[test]
#[ignore = "requires CUDA registration and real GPU-to-host publication"]
fn manager_validates_complete_demand_and_never_leases_a_partial_selected_group() {
    use cudarc::driver::{CudaContext, DevicePtr};
    use orbitkv_core::{LayerSave, TransferMode};
    use orbitkv_state::{RecoveryDemand, TokenRange};

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let attention = stream.alloc_zeros::<u8>(2048).unwrap();
    let checkpoint = stream.alloc_zeros::<u8>(2048).unwrap();
    stream.synchronize().unwrap();
    let layers = vec!["attention".to_string(), "window".to_string()];
    engine
        .register_context_layer_batch_strided(
            "model",
            "registered-shard-identity",
            0,
            0,
            0,
            1,
            1,
            &layers,
            &[
                attention.device_ptr(&stream).0,
                checkpoint.device_ptr(&stream).0,
            ],
            &[2048, 2048],
            &[2, 2],
            &[1024, 1024],
            &[0, 0],
            &[1, 1],
            None,
            Some(&[0, 1]),
            None,
            TransferMode::Direct,
            false,
        )
        .unwrap();
    let present = vec![7; 32];
    runtime.block_on(async {
        engine
            .batch_save_kv_blocks_from_ipc(
                "model",
                engine.registration_generation("model").unwrap_or_default(),
                0,
                0,
                0,
                layers
                    .iter()
                    .map(|layer| LayerSave {
                        layer_name: layer.clone(),
                        block_ids: vec![0],
                        block_hashes: vec![present.clone()],
                    })
                    .collect(),
            )
            .await
            .unwrap();
        engine.flush_saves().await;
    });
    let span = TokenRange { start: 64, end: 96 };
    let demand = RecoveryDemand {
        page_tokens: 16,
        span,
        groups: vec![(0, span), (1, span)],
    };
    let mut queries = PendingQueries::default();
    // Even with capacity exhausted, missing registered groups are rejected
    // before allocating a ticket owner, budget or source read.
    let held = Arc::clone(&queries.capacity)
        .try_acquire_many_owned(MAX_ACTIVE_QUERIES as u32)
        .unwrap();
    for groups in [vec![(0, span)], vec![(0, span), (2, span)]] {
        let mut bad = request(1, 1);
        bad.materialize = true;
        bad.block_hashes = vec![present.clone(), vec![8; 32]];
        bad.demand = Some(RecoveryDemand {
            groups,
            ..demand.clone()
        });
        let error = queries
            .execute(
                1,
                QueryCommand::Submit(bad),
                &engine,
                runtime.handle(),
                &hll,
            )
            .err()
            .expect("all registered groups are required");
        assert!(
            error.to_string().contains("every registered storage group"),
            "{error}"
        );
        assert!(queries.pending.is_empty());
        assert!(queries.sessions.is_empty());
    }
    drop(held);
    let mut operation = 1;
    for group in [0, 1] {
        for (selected, complete) in [(false, false), (true, false), (true, true)] {
            let mut query = request(operation, 1);
            operation += 1;
            query.group_id = group;
            query.block_hashes = vec![present.clone()];
            if !complete {
                query.block_hashes.push(vec![8; 32]);
            }
            query.materialize = selected;
            query.demand = selected.then(|| {
                if complete {
                    let span = TokenRange { start: 64, end: 80 };
                    RecoveryDemand {
                        page_tokens: 16,
                        span,
                        groups: vec![(0, span), (1, span)],
                    }
                } else {
                    demand.clone()
                }
            });
            let ticket = query.ticket;
            let mut reply = queries
                .execute(
                    1,
                    QueryCommand::Submit(query),
                    &engine,
                    runtime.handle(),
                    &hll,
                )
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while reply.is_none() {
                assert!(Instant::now() < deadline, "query did not complete");
                runtime.block_on(tokio::task::yield_now());
                reply = queries
                    .execute(
                        1,
                        QueryCommand::Poll(ticket),
                        &engine,
                        runtime.handle(),
                        &hll,
                    )
                    .unwrap();
            }
            let reply = reply.unwrap();
            let QueryOutcome::Ready {
                num_hit_blocks,
                lease,
                ..
            } = reply.outcome.as_ref().unwrap()
            else {
                panic!("terminal query must be ready");
            };
            if selected && !complete {
                assert_eq!(*num_hit_blocks, 0);
                assert!(
                    lease.is_empty(),
                    "partial demanded group must never get a lease"
                );
            } else {
                assert_eq!(*num_hit_blocks, 1);
                assert!(!lease.is_empty());
            }
            drop(reply); // Retire the test consumer through the real reply owner.
            assert!(queries.pending.is_empty());
        }
    }
    let before = hll.lock().unwrap().metrics()[0].1.total_requests;
    let mut prefix = request(operation, 1);
    prefix.prepare = true;
    prefix.block_hashes = vec![present, vec![8; 32]];
    let ticket = prefix.ticket;
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(prefix),
                &engine,
                runtime.handle(),
                &hll
            )
            .unwrap()
            .is_none()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while queries.pending[&(1, ticket.operation_id)]
        .receiver
        .as_ref()
        .unwrap()
        .is_empty()
    {
        assert!(
            Instant::now() < deadline,
            "prefix preparation did not finish"
        );
        runtime.block_on(tokio::task::yield_now());
    }
    assert_eq!(
        hll.lock().unwrap().metrics()[0].1.total_requests,
        before,
        "preparation must not count a foreground lookup"
    );
    let reply = queries
        .execute(
            1,
            QueryCommand::Claim {
                ticket,
                count_lookup: true,
            },
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    let QueryOutcome::Ready {
        num_hit_blocks,
        lease,
        ..
    } = reply.outcome.as_ref().unwrap()
    else {
        panic!("prepared prefix must be ready");
    };
    assert_eq!(*num_hit_blocks, 1);
    assert!(
        !lease.is_empty(),
        "ordinary prefix preparation preserves partial hits"
    );
    assert_eq!(
        hll.lock().unwrap().metrics()[0].1.total_requests,
        before + 2
    );
    drop(reply);
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Poll(ticket),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
    assert_eq!(
        hll.lock().unwrap().metrics()[0].1.total_requests,
        before + 2,
        "retired claims cannot count again"
    );
    assert_eq!(queries.capacity.available_permits(), MAX_ACTIVE_QUERIES);
    runtime
        .block_on(engine.unregister_instance_and_wait("model"))
        .unwrap();
}

#[test]
fn revisions_and_session_teardown_drain_old_work_without_restarting_it() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    let capacity = Arc::clone(&queries.capacity);
    let held = Arc::clone(&capacity)
        .try_acquire_many_owned((MAX_ACTIVE_QUERIES - 2) as u32)
        .unwrap();
    let mut completions = Vec::new();
    for token in [1, 2] {
        queries.sessions.entry(token).or_default().last_operation = 1;
        queries.insert(token, request(1, 1));
        let (complete, wait) = oneshot::channel();
        completions.push(complete);
        let result_engine = Arc::clone(&engine);
        let permit = Arc::clone(&capacity).try_acquire_owned().unwrap();
        assert!(
            queries
                .start_query(
                    (token, 1),
                    async move {
                        wait.await.unwrap();
                        QueryReply {
                            outcome: Ok(QueryOutcome::Ready {
                                num_hit_blocks: 0,
                                lease: vec![],
                                hit_positions: vec![],
                            }),
                            engine: result_engine,
                            delivered: false,
                            _permits: vec![permit],
                        }
                    },
                    runtime.handle()
                )
                .is_none()
        );
    }
    let old = request(1, 1).ticket;
    let new = request(1, 2).ticket;
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .unwrap()
            .is_none()
    );
    queries.cancel(1, old, &engine);
    assert_eq!(queries.pending[&(1, 1)].request.ticket, new);
    assert_eq!(queries.pending[&(2, 1)].request.ticket, old);
    for (session, ticket) in [(1, old), (9, new)] {
        assert!(
            queries
                .execute(
                    session,
                    QueryCommand::Poll(ticket),
                    &engine,
                    runtime.handle(),
                    &hll
                )
                .is_err()
        );
    }
    assert_eq!(
        capacity.available_permits(),
        0,
        "superseding work does not finish its I/O"
    );
    queries.cancel(1, new, &engine);
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
    queries.retain_sessions(&engine, |_| false);
    for completion in completions {
        completion.send(()).unwrap();
    }
    runtime.block_on(async {
        let _done = tokio::time::timeout(Duration::from_secs(2), capacity.acquire_many(2))
            .await
            .unwrap()
            .unwrap();
    });
    assert_eq!(capacity.available_permits(), 2);
    assert!(queries.pending.is_empty());
    drop(held);
}

#[test]
fn expiration_and_capacity_pressure_never_turn_a_poll_into_submission() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    queries.sessions.entry(1).or_default().last_operation = MAX_PENDING_PER_SESSION as u64;
    for id in 1..=MAX_PENDING_PER_SESSION as u64 {
        queries.insert(1, request(id, 1));
    }
    let next = request(MAX_PENDING_PER_SESSION as u64 + 1, 1);
    let reply = queries
        .execute(
            1,
            QueryCommand::Submit(next.clone()),
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(reply.outcome, Ok(QueryOutcome::Busy)));
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Poll(next.ticket),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
    queries.pending.get_mut(&(1, 1)).unwrap().expires = Instant::now();
    queries.retain_sessions(&engine, |_| true);
    let error = queries
        .execute(
            1,
            QueryCommand::Poll(request(1, 1).ticket),
            &engine,
            runtime.handle(),
            &hll,
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("timed out"));
    assert!(
        queries
            .execute(
                1,
                QueryCommand::Submit(request(1, 2)),
                &engine,
                runtime.handle(),
                &hll
            )
            .is_err()
    );
}

#[test]
fn warmups_skip_full_capacity_and_reap_unpolled_completions() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    let held = Arc::clone(&queries.warming)
        .try_acquire_many_owned(MAX_ACTIVE_WARMUPS as u32)
        .unwrap();
    let mut warmup = request(1, 1);
    warmup.warmup = true;
    let reply = queries
        .execute(
            1,
            QueryCommand::Submit(warmup.clone()),
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(reply.outcome, Ok(QueryOutcome::Busy)));
    drop(reply);
    assert!(queries.pending.is_empty());
    assert_eq!(queries.capacity.available_permits(), MAX_ACTIVE_QUERIES);
    drop(held);

    let permit = Arc::clone(&queries.warming).try_acquire_owned().unwrap();
    warmup.ticket.operation_id = 2;
    queries.insert(1, warmup);
    let (sender, receiver) = oneshot::channel();
    queries.pending.get_mut(&(1, 2)).unwrap().receiver = Some(receiver);
    let _sent = sender.send(QueryReply {
        outcome: Ok(QueryOutcome::Ready {
            num_hit_blocks: 0,
            lease: vec![],
            hit_positions: vec![],
        }),
        engine: Arc::clone(&engine),
        delivered: false,
        _permits: vec![permit],
    });
    queries.retain_sessions(&engine, |_| true);
    assert!(queries.pending.is_empty());
    assert_eq!(queries.warming.available_permits(), MAX_ACTIVE_WARMUPS);
}

#[test]
fn prepared_results_remain_owned_until_claim_or_expiry_without_further_polling() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let engine = engine();
    let hll = tracker();
    let mut queries = PendingQueries::default();
    queries.sessions.entry(1).or_default();
    for id in 1..=3 {
        let mut request = request(id, 1);
        request.prepare = true;
        request.materialize = true;
        let span = orbitkv_state::TokenRange { start: 0, end: 16 };
        request.demand = Some(orbitkv_state::RecoveryDemand {
            page_tokens: 16,
            span,
            groups: vec![(0, span)],
        });
        queries.insert(1, request);
        let result_engine = Arc::clone(&engine);
        let permit = Arc::clone(&queries.capacity).try_acquire_owned().unwrap();
        assert!(
            queries
                .start_query(
                    (1, id),
                    async move {
                        QueryReply {
                            outcome: Ok(QueryOutcome::Ready {
                                num_hit_blocks: 0,
                                lease: vec![],
                                hit_positions: vec![],
                            }),
                            engine: result_engine,
                            delivered: false,
                            _permits: vec![permit],
                        }
                    },
                    runtime.handle()
                )
                .is_none()
        );
    }
    queries.retain_sessions(&engine, |_| true);
    assert_eq!(
        queries.pending.len(),
        3,
        "ready preparations need a consumer, not eager retirement"
    );
    let claimed = queries
        .execute(
            1,
            QueryCommand::Claim {
                ticket: request(1, 1).ticket,
                count_lookup: true,
            },
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(claimed.outcome, Ok(QueryOutcome::Ready { .. })));
    drop(claimed);
    let expired = queries.pending.get_mut(&(1, 2)).unwrap();
    expired.expires = Instant::now();
    let control = Arc::clone(&expired.control);
    queries.retain_sessions(&engine, |_| true);
    assert!(!control.can_submit(0));
    let retired = queries
        .execute(
            1,
            QueryCommand::Claim {
                ticket: request(2, 1).ticket,
                count_lookup: true,
            },
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(matches!(retired.outcome, Ok(QueryOutcome::Busy)));
    queries.pending.get_mut(&(1, 3)).unwrap().expires = Instant::now();
    let expired = queries
        .execute(
            1,
            QueryCommand::Claim {
                ticket: request(3, 1).ticket,
                count_lookup: true,
            },
            &engine,
            runtime.handle(),
            &hll,
        )
        .unwrap()
        .unwrap();
    assert!(
        matches!(expired.outcome, Ok(QueryOutcome::Busy)),
        "a claim cannot extend expired ownership between sweeps"
    );
    assert_eq!(queries.capacity.available_permits(), MAX_ACTIVE_QUERIES);
}

#[tokio::test]
#[ignore = "requires a CUDA pinned pool for Manager query drain ownership"]
async fn query_drain_fences_admission_before_waiting_and_after_permits_return() {
    let engine = engine();
    let hll = tracker();
    let runtime = Handle::current();
    let queries = Arc::new(parking_lot::Mutex::new(PendingQueries::default()));
    let capacity = Arc::clone(&queries.lock().capacity);
    let accepted_read = Arc::clone(&capacity).try_acquire_owned().unwrap();
    let mut drain = Box::pin(PendingQueries::stop_and_drain(&queries, &engine));
    std::future::poll_fn(|cx| {
        assert!(drain.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    assert_eq!(
        capacity.available_permits(),
        0,
        "the drainer reserves free permits while the accepted read still owns its permit"
    );
    let assert_closed = || {
        for (operation, mode) in [(1, "foreground"), (2, "warmup"), (3, "prepare")] {
            let mut request = request(operation, 1);
            request.warmup = mode == "warmup";
            request.prepare = mode == "prepare";
            let mut queries = queries.lock();
            let error = queries
                .execute(1, QueryCommand::Submit(request), &engine, &runtime, &hll)
                .err()
                .expect("shutdown must reject all new admission");
            assert!(
                error.to_string().contains("query admission stopped"),
                "{mode}: {error}"
            );
            assert!(queries.pending.is_empty());
            assert!(queries.sessions.is_empty());
        }
    };
    assert_closed();
    drop(accepted_read);
    drain.as_mut().await.unwrap();
    assert_eq!(capacity.available_permits(), MAX_ACTIVE_QUERIES);
    assert_closed();
    for operation in [1, 4] {
        let mut queries = queries.lock();
        assert!(
            queries
                .execute(
                    1,
                    QueryCommand::Submit(request(operation, 1)),
                    &engine,
                    &runtime,
                    &hll
                )
                .is_err()
        );
        assert!(queries.sessions.is_empty());
    }
}
