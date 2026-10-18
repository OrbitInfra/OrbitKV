use super::*;
use crate::{
    BootstrapServer, BootstrapSession, Command, CommandCode, RESPONSE_FLAG_REQUEST_CONSUMED,
    Response, TransportServer,
};
use std::sync::mpsc;
use std::thread;

const LOOKUP: QueryIntent = QueryIntent::Lookup {
    wait_for_full_prefix: false,
};

fn hashes(values: &[&[u8]]) -> BlockHashes {
    BlockHashes::new(values.iter().map(|value| value.to_vec()).collect())
}

fn key(instance: &str, group: u32) -> QueryKey {
    QueryKey {
        instance: instance.into(),
        request: "r".into(),
        group,
    }
}

#[test]
fn publish_progress_remains_bounded_during_a_real_late_reply() {
    let release = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&release);
    let peer = Peer::new(move |command, _, _| {
        assert_eq!(command.code, CommandCode::Publish);
        gate.load(Ordering::Acquire).then(Vec::new)
    });
    let client = Arc::new(
        CacheClient::connect(
            &peer.socket,
            CallOptions {
                timeout: Duration::from_millis(5),
                spin_iterations: 0,
            },
        )
        .unwrap(),
    );
    assert_eq!(client.publish_progress().unwrap().active, 0);
    let source = Arc::new(vec![37; 4096]);
    let observed = Arc::downgrade(&source);
    let publisher = Arc::clone(&client);
    let worker = thread::spawn(move || {
        let request = PublishRequest {
            instance_id: "progress".into(),
            registration_generation: [37; 16],
            tp_rank: 0,
            pp_rank: 0,
            device_id: 0,
            layers: vec![],
        };
        let result = publisher.publish(&request);
        drop(source);
        result
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let progress = match client.publish_progress() {
            Ok(progress) => progress,
            Err(ChannelError::PublishObservationUnavailable) => {
                assert!(Instant::now() < deadline);
                thread::yield_now();
                continue;
            }
            Err(error) => panic!("{error}"),
        };
        if progress.active == 1 && progress.overdue {
            assert_eq!(progress.retained, 0);
            assert_eq!(progress.manager_alive, Some(true));
            assert!(observed.upgrade().is_some());
            break;
        }
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    for _ in 0..1000 {
        assert_eq!(client.publish_progress().unwrap().active, 1);
    }
    release.store(true, Ordering::Release);
    assert!(worker.join().unwrap().is_ok());
    assert_eq!(client.publish_progress().unwrap().active, 0);
    assert!(observed.upgrade().is_none());
    let _locked = client.publisher.lock().unwrap();
    assert!(matches!(
        client.publish_progress(),
        Err(ChannelError::PublishObservationUnavailable)
    ));
}

#[test]
fn polls_reuse_hash_storage_and_changed_demand_revises_one_ticket() {
    let mut queries = Queries::default();
    let key = key("model", 0);
    let batch = hashes(&[b"one", b"two"]);
    let first = queries.prepare(&key, &batch, LOOKUP).unwrap();
    let QueryCommand::Submit(first) = first else {
        panic!("expected submit")
    };
    let storage = queries.pending[&key].hashes.as_slice()[0].as_ptr();
    for _ in 0..10 {
        assert_eq!(
            queries.prepare(&key, &batch, LOOKUP).unwrap(),
            QueryCommand::Poll(first.ticket)
        );
        assert_eq!(queries.pending[&key].hashes.as_slice()[0].as_ptr(), storage);
    }
    for (bytes, intent, revision) in [
        (vec![b"changed".as_slice()], LOOKUP, 2),
        (
            vec![b"changed".as_slice()],
            QueryIntent::Lookup {
                wait_for_full_prefix: true,
            },
            3,
        ),
        (vec![b"changed".as_slice()], QueryIntent::Candidates, 4),
        (
            vec![b"changed".as_slice()],
            QueryIntent::Recovery(RecoveryDemand {
                page_tokens: 16,
                span: orbitkv_state::TokenRange { start: 0, end: 16 },
                groups: vec![(0, orbitkv_state::TokenRange { start: 0, end: 16 })],
            }),
            5,
        ),
    ] {
        let QueryCommand::Submit(changed) = queries.prepare(&key, &hashes(&bytes), intent).unwrap()
        else {
            panic!("expected revision")
        };
        assert_eq!(
            changed.ticket,
            QueryTicket {
                operation_id: 1,
                revision
            }
        );
    }
    queries.pending.get_mut(&key).unwrap().ticket.revision = u64::MAX;
    assert!(queries.prepare(&key, &hashes(&[b"new"]), LOOKUP).is_err());
}

#[test]
fn terminal_and_busy_queries_retire_and_instances_and_groups_do_not_alias() {
    let mut queries = Queries::default();
    for (index, key) in [key("a", 0), key("a", 1), key("b", 0)].iter().enumerate() {
        let QueryCommand::Submit(request) =
            queries.prepare(key, &hashes(&[b"hash"]), LOOKUP).unwrap()
        else {
            panic!("expected submit")
        };
        assert_eq!(request.ticket.operation_id, index as u64 + 1);
    }
    for outcome in [QueryOutcomeCode::Ready, QueryOutcomeCode::Busy] {
        queries.complete(&key("a", 0), outcome);
        assert!(matches!(
            queries
                .prepare(&key("a", 0), &hashes(&[b"hash"]), LOOKUP)
                .unwrap(),
            QueryCommand::Submit(_)
        ));
    }
    queries.next_operation = u64::MAX;
    assert!(queries.ticket().is_err());
    assert!(next_id(&AtomicU64::new(u64::MAX)).is_err());
}

#[test]
fn hash_views_share_storage_but_do_not_hide_changed_content_or_bounds() {
    let batch = hashes(&[b"first", b"second", b"third"]);
    let view = batch.slice(1..3).unwrap();
    assert!(Arc::ptr_eq(&batch.hashes, &view.hashes));
    assert_eq!(view.as_slice()[0].as_ptr(), batch.as_slice()[1].as_ptr());
    let prefix = view.slice(0..1).unwrap();
    assert_eq!(prefix, hashes(&[b"second"]));
    assert_ne!(prefix, batch.slice(0..1).unwrap());
    assert!(batch.slice(0..4).is_none());
    assert!(batch.slice(Range { start: 1, end: 0 }).is_none());
    assert!(view.slice(2..2).unwrap().as_slice().is_empty());
    let mut queries = Queries::default();
    queries.prepare(&key("m", 0), &view, LOOKUP).unwrap();
    // Different allocations with equal values retain their operation too.
    assert!(matches!(
        queries
            .prepare(&key("m", 0), &hashes(&[b"second", b"third"]), LOOKUP)
            .unwrap(),
        QueryCommand::Poll(_)
    ));
    assert!(matches!(
        queries.prepare(&key("m", 0), &prefix, LOOKUP).unwrap(),
        QueryCommand::Submit(_)
    ));
}

#[test]
fn preparations_and_polls_require_the_complete_selected_demand() {
    use orbitkv_state::TokenRange;
    let span = TokenRange { start: 64, end: 96 };
    let demand = RecoveryDemand {
        page_tokens: 16,
        span,
        groups: vec![(0, span), (1, TokenRange { start: 80, end: 96 })],
    };
    let mut shifted = demand.clone();
    shifted.span = TokenRange {
        start: 80,
        end: 112,
    };
    shifted.groups = vec![
        (0, shifted.span),
        (
            1,
            TokenRange {
                start: 96,
                end: 112,
            },
        ),
    ];
    let mut other_group = demand.clone();
    other_group.groups[1].1.start = 64;
    let batch = hashes(&[b"same-a", b"same-b"]);
    let key = key("registered-shard", 0);
    for (intent, unchanged) in [
        (QueryIntent::Recovery(demand.clone()), true),
        (QueryIntent::Recovery(shifted), false),
        (QueryIntent::Recovery(other_group), false),
        (LOOKUP, false),
    ] {
        for prepared in [false, true] {
            let mut queries = Queries::default();
            let QueryCommand::Submit(first) = queries
                .prepare(&key, &batch, QueryIntent::Recovery(demand.clone()))
                .unwrap()
            else {
                panic!("initial submit")
            };
            if prepared {
                queries.pending.get_mut(&key).unwrap().prepared_until =
                    Some(Instant::now() + PREPARATION_TTL);
            }
            let command = queries.prepare(&key, &batch, intent.clone()).unwrap();
            if unchanged {
                assert_eq!(
                    command,
                    if prepared {
                        QueryCommand::Claim {
                            ticket: first.ticket,
                            count_lookup: false,
                        }
                    } else {
                        QueryCommand::Poll(first.ticket)
                    }
                );
            } else {
                let QueryCommand::Submit(next) = command else {
                    panic!("changed demand must submit")
                };
                assert_eq!(next.ticket.operation_id, first.ticket.operation_id);
                assert_eq!(next.ticket.revision, first.ticket.revision + 1);
                assert_eq!(
                    next.demand,
                    match &intent {
                        QueryIntent::Recovery(demand) => Some(demand.clone()),
                        _ => None,
                    }
                );
            }
            assert!(queries.pending[&key].prepared_until.is_none());
        }
    }
}

/// Real bootstrap/descriptor framing with controlled completion timing.
struct Peer {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
    _dir: tempfile::TempDir,
}

enum Reply {
    Pending,
    Payload(Vec<u8>),
    DropAcknowledgement,
}

impl Peer {
    fn new(
        mut reply: impl FnMut(Command, &[u8], &BootstrapSession) -> Option<Vec<u8>> + Send + 'static,
    ) -> Self {
        Self::with_reply(move |command, payload, session| {
            reply(command, payload, session).map_or(Reply::Pending, Reply::Payload)
        })
    }

    fn with_reply(
        mut reply: impl FnMut(Command, &[u8], &BootstrapSession) -> Reply + Send + 'static,
    ) -> Self {
        static NAMES: AtomicU64 = AtomicU64::new(1);
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("cache.sock");
        let name = format!(
            "orbitkv/test/client/{}/{}",
            std::process::id(),
            NAMES.fetch_add(1, Ordering::Relaxed)
        );
        let bootstrap = BootstrapServer::bind(&socket, &name, 71, 64 * 1024, 4096).unwrap();
        bootstrap.set_nonblocking(true).unwrap();
        let transport = TransportServer::bind(&name).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let mut sessions = HashMap::new();
            let mut pending = Vec::new();
            while !stopping.load(Ordering::Acquire) {
                while let Some(session) = bootstrap.try_accept().unwrap() {
                    sessions.insert(session.client_token(), session);
                }
                transport
                    .try_serve_deferred_for_epoch(71, |command, response| {
                        let session = sessions.get_mut(&command.arg0).unwrap();
                        let slot = bootstrap
                            .descriptor_slot(command.descriptor.offset)
                            .unwrap();
                        session
                            .validate_request(command.descriptor, command.arg0, slot)
                            .unwrap();
                        let payload = bootstrap.arena().read(command.descriptor).unwrap();
                        pending.push((command, payload, response));
                        Ok(())
                    })
                    .unwrap();
                let mut index = 0;
                while index < pending.len() {
                    let (command, payload, _) = &pending[index];
                    let session = sessions.get_mut(&command.arg0).unwrap();
                    match reply(*command, payload, session) {
                        Reply::Payload(bytes) => {
                            let (command, _, response) = pending.swap_remove(index);
                            let mut result = Response::ok(command);
                            if command.code == CommandCode::Restore {
                                assert!(bytes.is_empty(), "Restore ACK has no wire body");
                            } else {
                                result.descriptor = bootstrap
                                    .arena()
                                    .write_response(command.descriptor, &bytes)
                                    .unwrap();
                            }
                            result.value1 = RESPONSE_FLAG_REQUEST_CONSUMED;
                            session.complete_request().unwrap();
                            response.send(result).unwrap();
                        }
                        Reply::DropAcknowledgement => {
                            pending.swap_remove(index);
                            session.complete_request().unwrap();
                        }
                        Reply::Pending => index += 1,
                    }
                }
                thread::sleep(Duration::from_micros(100));
            }
        });
        Self {
            socket,
            stop,
            thread: Some(thread),
            _dir: dir,
        }
    }

    fn client(&self) -> CacheClient {
        CacheClient::connect(&self.socket, CallOptions::default()).unwrap()
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn loading() -> Vec<u8> {
    QueryBundleResponse {
        outcome: QueryOutcomeCode::Loading,
        num_hit_blocks: 0,
        lease: vec![],
        hit_positions: vec![],
    }
    .encode()
    .unwrap()
}

#[test]
fn completion_observation_uses_the_authenticated_descriptor_session() {
    let observed = Arc::new(Mutex::new(None));
    let captured = Arc::clone(&observed);
    let peer = Peer::new(move |command, payload, _| {
        assert_eq!(command.code, CommandCode::ObserveCompletion);
        *captured.lock().unwrap() = Some(CompletionObservationRequest::decode(payload).unwrap());
        Some(Vec::new())
    });
    let client = peer.client();
    let request = CompletionObservationRequest {
        instance_id: "decode".into(),
        destination_device_id: 1,
        source_endpoint: "tent://prefill".into(),
        transfer_generation: 2,
        intent: crate::CompletionIntent::EngineRestore,
        route: crate::CompletionRoute::PrefillToDecodeHandoff,
        representation: orbitkv_state::ReplicaRepresentation::Raw,
        logical_bytes: 4096,
        wire_bytes: 4096,
        fragment_count: 2,
        elapsed_ns: 25_000,
        decode_page_bytes: 4096,
        handoff_queue_depth: 1,
        handoff_queue_parallelism: 16,
        tent_inflight_bytes: 0,
        tent_bandwidth_bytes_per_second: 0,
        admission: crate::CompletionAdmission::Admitted,
        outcome: crate::CompletionOutcome::Completed,
    };

    client.observe_completion(&request).unwrap();

    assert_eq!(*observed.lock().unwrap(), Some(request));
}

#[test]
fn warming_is_bounded_expires_and_is_cancelled_before_demand() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let peer = Peer::new(move |command, payload, _| {
        captured.lock().unwrap().push(command.code);
        Some(match command.code {
            CommandCode::QueryBundle => loading(),
            CommandCode::CancelQuery => {
                CancelQueryRequest::decode(payload).unwrap();
                vec![]
            }
            _ => panic!("unexpected command"),
        })
    });
    let client = peer.client();
    assert!(!client.warm_prefix("m", &hashes(&[]), "empty").unwrap());
    for index in 0..MAX_WARMUPS {
        assert!(
            client
                .warm_prefix("m", &hashes(&[b"hash"]), &index.to_string())
                .unwrap()
        );
    }
    assert!(!client.warm_prefix("m", &hashes(&[b"hash"]), "0").unwrap());
    assert!(
        !client
            .warm_prefix("m", &hashes(&[b"hash"]), "overflow")
            .unwrap()
    );
    client
        .query("m", &hashes(&[b"changed"]), "0", 0, LOOKUP)
        .unwrap();
    assert_eq!(
        &events.lock().unwrap()[MAX_WARMUPS..],
        &[CommandCode::CancelQuery, CommandCode::QueryBundle]
    );
    client.cancel_query("m", "0", 0).unwrap();
    client.cancel_query("m", "0", 0).unwrap();
    for (_, submitted) in client.queries.lock().unwrap().warmups.values_mut() {
        *submitted = Instant::now() - WARMUP_TTL;
    }
    assert!(client.warm_prefix("m", &hashes(&[b"new"]), "new").unwrap());
    assert_eq!(client.queries.lock().unwrap().warmups.len(), 1);
    client.close();
    assert!(client.queries.lock().unwrap().warmups.is_empty());
    assert!(
        client
            .query("m", &hashes(&[b"h"]), "closed", 0, LOOKUP)
            .is_err()
    );
}

#[test]
fn rejected_revision_retires_the_previous_interest_without_reusing_its_ticket() {
    let cancels = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&cancels);
    let peer = Peer::new(move |command, payload, _| {
        Some(match command.code {
            CommandCode::QueryBundle => loading(),
            CommandCode::CancelQuery => {
                captured
                    .lock()
                    .unwrap()
                    .push(CancelQueryRequest::decode(payload).unwrap().ticket);
                vec![]
            }
            _ => panic!("unexpected command"),
        })
    });
    let client = peer.client();
    client
        .query("m", &hashes(&[b"small"]), "r", 0, LOOKUP)
        .unwrap();
    let oversized = BlockHashes::new(vec![vec![0; 32]; 1024]);
    assert!(client.query("m", &oversized, "r", 0, LOOKUP).is_err());
    assert!(client.queries.lock().unwrap().pending.is_empty());
    assert_eq!(
        *cancels.lock().unwrap(),
        vec![
            QueryTicket {
                operation_id: 1,
                revision: 2
            },
            QueryTicket {
                operation_id: 1,
                revision: 1
            },
        ]
    );
    client
        .query("m", &hashes(&[b"small"]), "r", 0, LOOKUP)
        .unwrap();
    assert_eq!(
        client.queries.lock().unwrap().pending[&key("m", 0)]
            .ticket
            .operation_id,
        2
    );
}

#[test]
fn restore_deadline_and_lost_notification_preserve_ownership_and_reject_other_clients() {
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let calls = Arc::new(AtomicU64::new(0));
    let observed_calls = Arc::clone(&calls);
    let peer = Peer::new(move |command, payload, session| {
        assert_eq!(
            command.code,
            CommandCode::Restore,
            "timeout must never release pages"
        );
        assert_eq!(
            observed_calls.fetch_add(1, Ordering::Relaxed),
            0,
            "poll/wait must not issue control requests"
        );
        RestoreRequest::decode(payload).unwrap();
        let operation_id = command.arg1;
        session.completions().claim(operation_id).unwrap();
        session.completions().start_managed(operation_id).unwrap();
        submitted_tx
            .send((Arc::clone(session.completions()), operation_id))
            .unwrap();
        Some(Vec::new())
    });
    let client = peer.client();
    let other = peer.client();
    let handle = client
        .start_restore(&RestoreRequest {
            instance_id: "m".into(),
            tp_rank: 0,
            device_id: 0,
            layer_groups: vec![],
            loads: vec![],
        })
        .unwrap();
    assert!(other.poll_restore(handle).is_err());
    assert!(matches!(
        client.wait_restore(handle, Duration::from_millis(2)),
        Err(ChannelError::RestoreTimeout { .. })
    ));
    let (records, id) = submitted_rx.recv().unwrap();
    thread::scope(|scope| {
        scope.spawn(|| {
            thread::sleep(Duration::from_millis(10));
            records.complete(id, Ok(())).unwrap(); // Deliberately lose the eventfd notification.
        });
        assert_eq!(
            client
                .wait_restore(handle, Duration::from_secs(1))
                .unwrap()
                .state,
            RestoreState::Succeeded
        );
    });
}

#[test]
fn blocked_publish_does_not_serialize_query_or_restore() {
    let finish = Arc::new(AtomicBool::new(false));
    let publishing = Arc::clone(&finish);
    let (started_tx, started_rx) = mpsc::channel();
    let mut started = false;
    let peer = Peer::new(move |command, _, _| {
        if command.code == CommandCode::Publish {
            if !started {
                started = true;
                started_tx.send(()).unwrap();
            }
            return publishing.load(Ordering::Acquire).then(Vec::new);
        }
        Some(loading())
    });
    let client = peer.client();
    thread::scope(|scope| {
        let publish = scope.spawn(|| {
            client.publish(&PublishRequest {
                instance_id: "m".into(),
                registration_generation: [1; 16],
                tp_rank: 0,
                pp_rank: 0,
                device_id: 0,
                layers: vec![],
            })
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let result = client.query("m", &hashes(&[b"h"]), "r", 0, LOOKUP);
        finish.store(true, Ordering::Release);
        assert_eq!(result.unwrap().outcome, QueryOutcomeCode::Loading);
        publish.join().unwrap().unwrap();
    });
    client.close();
    assert!(
        client
            .publish(&PublishRequest {
                instance_id: "m".into(),
                registration_generation: [1; 16],
                tp_rank: 0,
                pp_rank: 0,
                device_id: 0,
                layers: vec![]
            })
            .is_err()
    );
}

#[test]
fn observed_publish_and_query_share_the_manager_session_and_bound_the_call() {
    let peer = Peer::new(|command, _, _| {
        Some(if command.code == CommandCode::Publish {
            Vec::new()
        } else {
            loading()
        })
    });
    let client = peer.client();
    let publish = client
        .publish_observed(&PublishRequest {
            instance_id: "m".into(),
            registration_generation: [1; 16],
            tp_rank: 0,
            pp_rank: 0,
            device_id: 0,
            layers: vec![],
        })
        .unwrap();
    let (query, observed_query) = client
        .query_observed("m", &hashes(&[b"h"]), "r", 0, LOOKUP)
        .unwrap();
    assert_eq!(query.outcome, QueryOutcomeCode::Loading);
    assert_ne!(publish.request_id, observed_query.request_id);
    assert!(publish.submitted_mono_ns <= publish.returned_mono_ns);
    assert!(observed_query.submitted_mono_ns <= observed_query.returned_mono_ns);
    assert!(publish.session_epoch > 0);
    assert!(publish.session_token > 0);
    assert!(observed_query.session_epoch > 0);
    assert!(observed_query.session_token > 0);
}

#[test]
fn planned_read_fetches_only_its_window_and_releases_a_stale_partial_lease() {
    use orbitkv_state::{
        RecoveryContract, RecoveryRule, StateComponent, StateRequirement, TokenRange,
    };
    let contract = RecoveryContract::compile(
        "model".into(),
        16,
        vec![
            StateRequirement {
                group: 0,
                components: [StateComponent::AttentionKv].into(),
                rule: RecoveryRule::Prefix,
            },
            StateRequirement {
                group: 1,
                components: [StateComponent::SlidingWindowKv].into(),
                rule: RecoveryRule::Window { tokens: 32 },
            },
        ],
    )
    .unwrap();
    let released = Arc::new(AtomicBool::new(false));
    let observed = Arc::clone(&released);
    let preparations = Arc::new(Mutex::new(HashMap::new()));
    let peer = Peer::new(move |command, bytes, _| match command.code {
        CommandCode::QueryBundle => {
            let request = match QueryCommand::decode(bytes).unwrap() {
                QueryCommand::Submit(request) if request.prepare => {
                    assert_eq!(request.block_hashes, vec![b"c".to_vec(), b"d".to_vec()]);
                    assert_eq!(
                        request.demand.as_ref().unwrap().groups,
                        vec![
                            (
                                0,
                                TokenRange {
                                    start: 64,
                                    end: 128
                                }
                            ),
                            (
                                1,
                                TokenRange {
                                    start: 96,
                                    end: 128
                                }
                            ),
                        ]
                    );
                    preparations.lock().unwrap().insert(request.ticket, request);
                    return Some(loading());
                }
                QueryCommand::Submit(request) => request,
                QueryCommand::Claim {
                    ticket,
                    count_lookup: false,
                } => preparations.lock().unwrap().remove(&ticket).unwrap(),
                _ => panic!("unexpected poll or counted recovery"),
            };
            assert!(!request.discover);
            assert!(request.materialize);
            let demand = request.demand.as_ref().unwrap();
            assert_eq!(demand.page_tokens, 16);
            assert_eq!(
                demand.span,
                TokenRange {
                    start: 64,
                    end: 128
                }
            );
            assert_eq!(
                demand.groups,
                vec![
                    (
                        0,
                        TokenRange {
                            start: 64,
                            end: 128
                        }
                    ),
                    (
                        1,
                        TokenRange {
                            start: 96,
                            end: 128
                        }
                    ),
                ]
            );
            assert_eq!(request.block_hashes, vec![b"c".to_vec(), b"d".to_vec()]);
            let count = if request.request_id == "stale" { 1 } else { 2 };
            Some(
                QueryBundleResponse {
                    outcome: QueryOutcomeCode::Ready,
                    num_hit_blocks: count,
                    lease: vec![7],
                    hit_positions: (0..count as u32).collect(),
                }
                .encode()
                .unwrap(),
            )
        }
        CommandCode::Release => {
            observed.store(true, Ordering::Release);
            Some(Vec::new())
        }
        other => panic!("unexpected {other:?}"),
    });
    let client = peer.client();
    let batch = hashes(&[b"a", b"b", b"c", b"d"]);
    let read = || RecoveryRead {
        contract: &contract,
        namespace: "model",
        span: TokenRange {
            start: 64,
            end: 128,
        },
        group: 1,
    };
    let ready = client
        .read_recovery("m", &batch, "complete", read())
        .unwrap();
    assert_eq!(ready.hit_positions, vec![2, 3]);
    assert_eq!(ready.lease, vec![7]);
    assert!(!released.load(Ordering::Acquire));
    let miss = client.read_recovery("m", &batch, "stale", read()).unwrap();
    assert_eq!(miss.num_hit_blocks, 0);
    assert!(miss.lease.is_empty());
    assert!(released.load(Ordering::Acquire));
    assert!(
        client
            .prepare_recovery("m", &batch, "prepared", read())
            .unwrap()
    );
    assert!(
        !client
            .prepare_recovery("m", &batch, "prepared", read())
            .unwrap()
    );
    let prepared = client
        .read_recovery("m", &batch, "prepared", read())
        .unwrap();
    assert_eq!(prepared.hit_positions, vec![2, 3]);
    assert_eq!(prepared.lease, vec![7]);
}

#[test]
fn prepared_prefix_claims_partial_hits_without_an_extra_payload_query() {
    let preparations = Arc::new(Mutex::new(HashMap::new()));
    let calls = Arc::new(AtomicU64::new(0));
    let observed_calls = Arc::clone(&calls);
    let released = Arc::new(AtomicBool::new(false));
    let observed_release = Arc::clone(&released);
    let peer = Peer::new(move |command, bytes, _| match command.code {
        CommandCode::QueryBundle => {
            observed_calls.fetch_add(1, Ordering::Relaxed);
            match QueryCommand::decode(bytes).unwrap() {
                QueryCommand::Submit(request) => {
                    assert!(request.prepare);
                    assert!(!request.materialize && !request.discover && !request.warmup);
                    assert!(!request.wait_for_full_prefix && request.demand.is_none());
                    assert_eq!(request.group_id, 0);
                    assert_eq!(
                        request.block_hashes,
                        vec![b"present".to_vec(), b"missing".to_vec()]
                    );
                    preparations.lock().unwrap().insert(request.ticket, request);
                    Some(loading())
                }
                QueryCommand::Claim {
                    ticket,
                    count_lookup: true,
                } => {
                    preparations.lock().unwrap().remove(&ticket).unwrap();
                    Some(
                        QueryBundleResponse {
                            outcome: QueryOutcomeCode::Ready,
                            num_hit_blocks: 1,
                            lease: vec![7],
                            hit_positions: vec![],
                        }
                        .encode()
                        .unwrap(),
                    )
                }
                other => panic!("unexpected foreground operation: {other:?}"),
            }
        }
        CommandCode::Release => {
            observed_release.store(true, Ordering::Release);
            Some(Vec::new())
        }
        other => panic!("unexpected {other:?}"),
    });
    let client = peer.client();
    let batch = hashes(&[b"present", b"missing"]);
    assert!(client.prepare_prefix("m", &batch, "prefix").unwrap());
    assert!(!client.prepare_prefix("m", &batch, "prefix").unwrap());
    let ready = client.query("m", &batch, "prefix", 0, LOOKUP).unwrap();
    assert_eq!(ready.num_hit_blocks, 1);
    assert_eq!(ready.lease, vec![7]);
    assert_eq!(calls.load(Ordering::Acquire), 2);
    assert!(!released.load(Ordering::Acquire));
    client.release(ready.lease).unwrap();
    assert!(released.load(Ordering::Acquire));
}

fn restore_request() -> RestoreRequest {
    RestoreRequest {
        instance_id: "m".into(),
        tp_rank: 0,
        device_id: 0,
        layer_groups: vec![],
        loads: vec![],
    }
}

#[test]
fn equal_restore_ids_in_live_sessions_keep_results_and_handle_ownership_separate() {
    for same_peer in [true, false] {
        let (submitted_tx, submitted_rx) = mpsc::channel();
        let make_peer = || {
            let submitted_tx = submitted_tx.clone();
            Peer::new(move |command, payload, session| {
                assert_eq!(command.code, CommandCode::Restore);
                RestoreRequest::decode(payload).unwrap();
                session.completions().claim(command.arg1).unwrap();
                session.completions().start_managed(command.arg1).unwrap();
                submitted_tx
                    .send((session.client_token(), Arc::clone(session.completions())))
                    .unwrap();
                Some(Vec::new())
            })
        };
        let first_peer = make_peer();
        let second_peer = (!same_peer).then(make_peer);
        let first = first_peer.client();
        let second = second_peer.as_ref().unwrap_or(&first_peer).client();
        let first_handle = first.start_restore(&restore_request()).unwrap();
        let (first_token, first_records) = submitted_rx.recv().unwrap();
        let second_handle = second.start_restore(&restore_request()).unwrap();
        let (second_token, second_records) = submitted_rx.recv().unwrap();
        assert_eq!(first_handle.operation_id, second_handle.operation_id);
        assert_eq!(first_handle.session_epoch, second_handle.session_epoch);
        assert_eq!(first_handle.session_token, first_token);
        assert_eq!(second_handle.session_token, second_token);
        // Separate Managers can reuse an explicitly configured epoch and its
        // derived first token. A handle must still belong to its issuing client.
        assert_eq!(first_token == second_token, !same_peer);
        for (client, foreign) in [(&first, second_handle), (&second, first_handle)] {
            assert!(
                matches!(
                    client.poll_restore(foreign),
                    Err(ChannelError::SessionRequiresReconnect)
                ),
                "foreign handle accepted with same_peer={same_peer}"
            );
        }
        first_records
            .complete(first_handle.operation_id, Ok(()))
            .unwrap();
        assert_eq!(
            first.poll_restore(first_handle).unwrap().state,
            RestoreState::Succeeded
        );
        assert_eq!(
            second.poll_restore(second_handle).unwrap().state,
            RestoreState::Pending
        );
        second_records
            .complete(
                second_handle.operation_id,
                Err("second session failed".into()),
            )
            .unwrap();
        let result = second.poll_restore(second_handle).unwrap();
        assert_eq!(result.state, RestoreState::Failed);
        assert_eq!(result.message, "second session failed");
        assert!(first.poll_restore(first_handle).is_err());
    }
}

#[test]
fn lost_restore_acknowledgement_recovers_the_claimed_operation_without_resubmission() {
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let calls = Arc::new(AtomicU64::new(0));
    let observed_calls = Arc::clone(&calls);
    let peer = Peer::with_reply(move |command, payload, session| {
        assert_eq!(command.code, CommandCode::Restore);
        RestoreRequest::decode(payload).unwrap();
        assert_eq!(observed_calls.fetch_add(1, Ordering::Relaxed), 0);
        session.completions().claim(command.arg1).unwrap();
        session.completions().start_managed(command.arg1).unwrap();
        submitted_tx
            .send((Arc::clone(session.completions()), command.arg1))
            .unwrap();
        Reply::DropAcknowledgement
    });
    let client = CacheClient::connect(
        &peer.socket,
        CallOptions {
            timeout: Duration::from_millis(200),
            ..CallOptions::default()
        },
    )
    .unwrap();
    let handle = client.start_restore(&restore_request()).unwrap();
    let (records, id) = submitted_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(handle.operation_id, id);
    assert_eq!(
        client.poll_restore(handle).unwrap().state,
        RestoreState::Pending
    );
    assert!(matches!(
        client.query("m", &hashes(&[b"hash"]), "after-lost-ack", 0, LOOKUP),
        Err(ChannelError::SessionRequiresReconnect)
    ));
    assert!(matches!(
        client.start_restore(&restore_request()),
        Err(ChannelError::SessionRequiresReconnect)
    ));
    records
        .complete(id, Err("copy failed after drain".into()))
        .unwrap();
    let terminal = client.wait_restore(handle, Duration::from_secs(1)).unwrap();
    assert_eq!(terminal.operation_id, id);
    assert_eq!(terminal.state, RestoreState::Failed);
    assert_eq!(terminal.message, "copy failed after drain");
    assert!(client.poll_restore(handle).is_err());
    assert_eq!(calls.load(Ordering::Acquire), 1);
}

#[test]
fn failed_restore_before_claim_cancels_the_slot_and_rejects_a_delayed_execution() {
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let (claim_tx, claim_rx) = mpsc::channel();
    let (rejected_tx, rejected_rx) = mpsc::channel();
    let peer = Peer::with_reply(move |command, payload, session| {
        assert_eq!(command.code, CommandCode::Restore);
        RestoreRequest::decode(payload).unwrap();
        submitted_tx
            .send((Arc::clone(session.completions()), command.arg1))
            .unwrap();
        claim_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(session.completions().claim(command.arg1).is_err());
        assert!(
            session
                .completions()
                .complete(command.arg1, Ok(()))
                .is_err()
        );
        rejected_tx.send(()).unwrap();
        Reply::DropAcknowledgement
    });
    let client = CacheClient::connect(
        &peer.socket,
        CallOptions {
            timeout: Duration::from_millis(200),
            ..CallOptions::default()
        },
    )
    .unwrap();
    thread::scope(|scope| {
        let submitted = scope.spawn(|| client.start_restore(&restore_request()));
        let (records, id) = submitted_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            submitted.join().unwrap(),
            Err(ChannelError::Transport(
                crate::TransportError::Timeout { .. }
            ))
        ));
        assert!(
            records.poll(id).is_err(),
            "cancelled slots cannot remain pending"
        );
        claim_tx.send(()).unwrap();
        rejected_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    });
}

#[test]
fn closed_lifecycle_socket_preserves_pending_restore_until_the_live_peer_completes() {
    for terminal_before_disconnect in [false, true] {
        let (submitted_tx, submitted_rx) = mpsc::channel();
        let peer = Peer::new(move |command, payload, session| {
            assert_eq!(command.code, CommandCode::Restore);
            RestoreRequest::decode(payload).unwrap();
            let id = command.arg1;
            session.completions().claim(id).unwrap();
            session.completions().start_managed(id).unwrap();
            submitted_tx
                .send((Arc::clone(session.completions()), id))
                .unwrap();
            Some(Vec::new())
        });
        let client = peer.client();
        let handle = client.start_restore(&restore_request()).unwrap();
        let (records, id) = submitted_rx.recv().unwrap();
        if terminal_before_disconnect {
            records.complete(id, Ok(())).unwrap();
        }
        drop(peer);
        if !terminal_before_disconnect {
            assert_eq!(
                client.poll_restore(handle).unwrap().state,
                RestoreState::Pending
            );
            let waiting = Instant::now();
            assert!(
                !client
                    .channel
                    .wait_for_notification(Duration::from_millis(20))
                    .unwrap()
            );
            assert!(waiting.elapsed() >= Duration::from_millis(10));
            assert!(matches!(
                client.wait_restore(handle, Duration::from_millis(2)),
                Err(ChannelError::RestoreTimeout { .. })
            ));
            assert_eq!(records.poll(id).unwrap().state, RestoreState::Pending);
            records.complete(id, Ok(())).unwrap();
        }
        assert_eq!(
            client
                .wait_restore(handle, Duration::from_secs(1))
                .unwrap()
                .state,
            RestoreState::Succeeded
        );
    }
}
