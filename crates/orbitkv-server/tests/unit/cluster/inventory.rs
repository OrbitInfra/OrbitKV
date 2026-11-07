use super::*;
#[cfg(feature = "test-hooks")]
use crate::proto::engine::inventory_server::InventoryServer;
#[cfg(feature = "test-hooks")]
use orbitkv_state::{
    DiscoveryCoverage, ReplicaMedium, ReplicaMetadata, ReplicaRepresentation, StateKey,
};
#[cfg(feature = "test-hooks")]
use tonic::transport::Server;

#[cfg(feature = "test-hooks")]
use crate::cluster::tests::etcd::{Etcd, join, view, wait_for};
#[cfg(feature = "test-hooks")]
use crate::cluster::tests::gate::TcpGate;

#[cfg(feature = "test-hooks")]
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[cfg(feature = "test-hooks")]
async fn serve(runtime: InventoryRuntime, port: u16) -> tokio::task::JoinHandle<()> {
    let task = tokio::spawn(async move {
        Server::builder()
            .add_service(
                InventoryServer::new(runtime.service())
                    .max_decoding_message_size(MESSAGE_BYTES)
                    .max_encoding_message_size(MESSAGE_BYTES),
            )
            .serve(([127, 0, 0, 1], port).into())
            .await
            .unwrap();
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_err()
    {
        assert!(std::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    task
}

#[cfg(feature = "test-hooks")]
struct ScopedNode {
    cluster: crate::cluster::Cluster,
    runtime: InventoryRuntime,
    index: Arc<GlobalIndex>,
    server: tokio::task::JoinHandle<()>,
}

#[cfg(feature = "test-hooks")]
async fn scoped_node(
    etcd: &Etcd,
    cluster: &str,
    node: &str,
    port: u16,
    scope: InventoryScope,
) -> ScopedNode {
    let membership = view(port);
    let index = Arc::new(GlobalIndex::new(
        membership.clone(),
        1 << 20,
        Arc::new(scope),
    ));
    let inventory = Arc::new(ResidencyInventory::new(1 << 20));
    let cluster = crate::cluster::Cluster::join(
        &etcd.endpoints,
        cluster,
        node,
        60,
        membership,
        inventory,
        index.clone(),
    )
    .await
    .unwrap();
    let runtime = cluster.inventory();
    let server = serve(runtime.clone(), port).await;
    ScopedNode {
        cluster,
        runtime,
        index,
        server,
    }
}

#[cfg(feature = "test-hooks")]
fn metadata(medium: ReplicaMedium) -> ReplicaMetadata {
    ReplicaMetadata {
        medium,
        representation: ReplicaRepresentation::Raw,
        stored_bytes: Some(4096),
    }
}

fn runtime_for_protocol_test() -> (InventoryRuntime, Member, CacheOwner) {
    let own = CacheOwner {
        endpoint: "127.0.0.1:59001".into(),
        incarnation: Uuid::new_v4(),
    };
    let requester = CacheOwner {
        endpoint: "127.0.0.1:59002".into(),
        incarnation: Uuid::new_v4(),
    };
    let membership = Arc::new(MembershipView::new(own.clone()));
    assert!(membership.renew(std::time::Instant::now(), Duration::from_secs(60)));
    membership.replace_members([
        ("source".into(), 1, own.clone()),
        ("requester".into(), 1, requester.clone()),
    ]);
    let member = Member {
        node_id: "source".into(),
        epoch: 1,
        owner: own,
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        lease: 1,
    };
    let format = ClusterFormat {
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        cluster_uuid: Uuid::new_v4(),
        revision: 1,
        encoded: b"format".to_vec(),
    };
    let index = Arc::new(GlobalIndex::new(
        membership.clone(),
        1 << 20,
        Arc::new(InventoryScope::AllNamespaces),
    ));
    let runtime = InventoryRuntime::new(
        format,
        member.clone(),
        Arc::new(ResidencyInventory::new(1 << 20)),
        index,
        membership,
    );
    runtime.replace_members(
        1,
        [
            (member.node_id.clone(), member.clone()),
            (
                "requester".into(),
                Member {
                    node_id: "requester".into(),
                    epoch: 1,
                    owner: requester.clone(),
                    protocol: INVENTORY_STREAM_PROTOCOL.into(),
                    lease: 2,
                },
            ),
        ]
        .into_iter()
        .collect(),
    );
    (runtime, member, requester)
}

#[test]
fn open_and_frame_credit_reject_wrong_identity_and_future_ack() {
    let (runtime, source, requester) = runtime_for_protocol_test();
    let session = Uuid::new_v4();
    let valid = InventoryOpen {
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        cluster_uuid: runtime.shared.format.cluster_uuid.as_bytes().to_vec(),
        source_node: source.node_id.clone(),
        source_epoch: source.epoch,
        source_incarnation: source.owner.incarnation.as_bytes().to_vec(),
        requester_incarnation: requester.incarnation.as_bytes().to_vec(),
        scope_digest: all_namespaces_scope_digest().to_vec(),
        session_id: session.as_bytes().to_vec(),
        resume_sequence: None,
        installed_view_id: Vec::new(),
        scope_mode: InventoryScopeMode::InventoryScopeAllNamespaces as i32,
        scope_namespaces: Vec::new(),
    };
    assert!(validate_open(&runtime, &valid).is_ok());
    let mut wrong = valid.clone();
    wrong.cluster_uuid = Uuid::new_v4().as_bytes().to_vec();
    assert!(validate_open(&runtime, &wrong).is_err());
    let mut unspecified = valid.clone();
    unspecified.scope_mode = InventoryScopeMode::InventoryScopeUnspecified as i32;
    assert!(validate_open(&runtime, &unspecified).is_err());
    let mut wrong_digest = valid.clone();
    wrong_digest.scope_digest = InventoryScope::exact(["outside".into()])
        .unwrap()
        .digest()
        .to_vec();
    assert!(validate_open(&runtime, &wrong_digest).is_err());
    let exact_scope = InventoryScope::exact(["a".into(), "b".into()]).unwrap();
    let mut noncanonical = valid.clone();
    noncanonical.scope_mode = InventoryScopeMode::InventoryScopeExactNamespaces as i32;
    noncanonical.scope_namespaces = ["b", "a", "b"]
        .into_iter()
        .map(|value| value.as_bytes().to_vec())
        .collect();
    noncanonical.scope_digest = exact_scope.digest().to_vec();
    assert!(validate_open(&runtime, &noncanonical).is_err());
    let mut exact = noncanonical;
    exact.scope_namespaces = ["a", "b"]
        .into_iter()
        .map(|value| value.as_bytes().to_vec())
        .collect();
    assert_eq!(validate_open(&runtime, &exact).unwrap().scope, exact_scope);

    let (output, _) = mpsc::channel(1);
    let (control, controls) = watch::channel(ClientControl {
        consumed_frame_id: 1,
        ..ClientControl::default()
    });
    let mut sender = FrameSender::new(runtime, session, output, controls);
    assert!(sender.release_acknowledged().is_err());
    control.send_modify(|state| state.consumed_frame_id = 0);
    assert!(sender.release_acknowledged().is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(feature = "test-hooks")]
#[ignore = "requires ETCD_BIN; real membership plus inventory gRPC snapshot/delta/reset"]
async fn stream_installs_exact_owner_views_and_recovers_overflow_without_etcd_blocks() {
    let etcd = Etcd::start(1).await;
    let source_backend_port = free_port();
    let gate = TcpGate::start(&format!("http://127.0.0.1:{source_backend_port}")).await;
    let source_port = gate.endpoint.rsplit(':').next().unwrap().parse().unwrap();
    let target_port = free_port();
    let source_view = view(source_port);
    let target_view = view(target_port);
    let (source, _, source_inventory) =
        join(&etcd, "inventory-stream", "source", source_view.clone(), 60).await;
    let (target, target_index, _) =
        join(&etcd, "inventory-stream", "target", target_view.clone(), 60).await;
    let source_runtime = source.inventory();
    let target_runtime = target.inventory();
    let source_server = serve(source_runtime.clone(), source_backend_port).await;
    let target_server = serve(target_runtime.clone(), target_port).await;
    gate.set_downstream_delay(Duration::from_millis(2));
    wait_for(|| {
        source_view.permits(target_view.owner()) && target_view.permits(source_view.owner())
    })
    .await;

    let first = StateKey::new("model".into(), vec![1; 32]);
    let second = StateKey::new("model".into(), vec![2; 32]);
    source_inventory
        .test_change(
            &first,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    source_inventory
        .test_change(
            &first,
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    source_inventory
        .test_change(
            &second,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    let fence = source_runtime.capture_fence().unwrap();
    target_runtime
        .await_fence(
            &fence,
            &all_namespaces_scope_digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    let rows = target_index.lookup(&[first.clone(), second.clone()]);
    assert_eq!(rows[0].replicas.len(), 2);
    assert_eq!(rows[1].replicas.len(), 1);
    assert!(rows.iter().all(|row| {
        row.coverage == DiscoveryCoverage::CompleteAtWatermarks
            && row
                .replicas
                .iter()
                .all(|replica| replica.owner == *source_view.owner())
    }));

    gate.partition().await;
    for id in 0u32..400 {
        let key = StateKey::new("overflow".into(), id.to_le_bytes().to_vec());
        source_inventory
            .test_change(
                &key,
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
        source_inventory
            .test_change(&key, ReplicaMedium::Dram, None)
            .unwrap();
    }
    source_inventory
        .test_change(&first, ReplicaMedium::Dram, None)
        .unwrap();
    source_inventory
        .test_change(
            &second,
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    let final_fence = source_runtime.capture_fence().unwrap();
    gate.heal(Duration::ZERO);
    target_runtime
        .await_fence(
            &final_fence,
            &all_namespaces_scope_digest(),
            Duration::from_secs(15),
        )
        .await
        .unwrap();
    let rows = target_index.lookup(&[first.clone(), second.clone()]);
    assert_eq!(rows[0].replicas.len(), 1);
    assert_eq!(rows[0].replicas[0].metadata.medium, ReplicaMedium::Ssd);
    assert_eq!(rows[1].replicas.len(), 2);
    assert_eq!(
        target_index
            .owner_watermark(source_view.owner().incarnation)
            .unwrap()
            .1,
        final_fence.inventory_sequence
    );
    assert!(source_inventory.status().history_gaps > 0);
    assert!(target_runtime.status().resets > 0);

    let mut client = etcd_client::Client::connect(&etcd.endpoints, None)
        .await
        .unwrap();
    let metadata = client
        .get(
            "/orbitkv/v2/inventory-stream/",
            Some(
                etcd_client::GetOptions::new()
                    .with_prefix()
                    .with_keys_only(),
            ),
        )
        .await
        .unwrap();
    assert!(metadata.kvs().iter().all(|kv| {
        let key = kv.key_str().unwrap();
        !key.contains("/blocks/") && !key.contains("/publishers/")
    }));

    source_server.abort();
    target_server.abort();
    gate.shutdown().await;
    target.shutdown().await;
    source.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[cfg(feature = "test-hooks")]
#[ignore = "requires ETCD_BIN; real scoped snapshot, empty intervals and exact coverage"]
async fn scoped_streams_filter_at_source_and_install_complete_empty_views() {
    let etcd = Etcd::start(1).await;
    let selected = "orbitkv:v2:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let other = "orbitkv:v2:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let outside = "orbitkv:v2:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let source_backend_port = free_port();
    let gate = TcpGate::start(&format!("http://127.0.0.1:{source_backend_port}")).await;
    let source_port = gate.endpoint.rsplit(':').next().unwrap().parse().unwrap();
    let source_view = view(source_port);
    let (source, _, source_inventory) =
        join(&etcd, "scoped-stream", "source", source_view.clone(), 60).await;
    let source_runtime = source.inventory();
    let source_server = serve(source_runtime.clone(), source_backend_port).await;
    let single = scoped_node(
        &etcd,
        "scoped-stream",
        "single",
        free_port(),
        InventoryScope::exact([selected.into()]).unwrap(),
    )
    .await;
    let multi = scoped_node(
        &etcd,
        "scoped-stream",
        "multi",
        free_port(),
        InventoryScope::exact([other.into(), selected.into(), other.into()]).unwrap(),
    )
    .await;
    let empty = scoped_node(
        &etcd,
        "scoped-stream",
        "empty",
        free_port(),
        InventoryScope::exact(Vec::new()).unwrap(),
    )
    .await;
    wait_for(|| {
        source_view.permits(single.cluster.view.owner())
            && source_view.permits(multi.cluster.view.owner())
            && source_view.permits(empty.cluster.view.owner())
    })
    .await;

    let selected_key = StateKey::new(selected.into(), vec![7; 32]);
    let other_key = StateKey::new(other.into(), vec![7; 32]);
    let outside_key = StateKey::new(outside.into(), vec![7; 32]);
    for key in [&selected_key, &other_key, &outside_key] {
        source_inventory
            .test_change(
                key,
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
    }
    let fence = source_runtime.capture_fence().unwrap();
    for node in [&single, &multi, &empty] {
        node.runtime
            .await_fence(
                &fence,
                &node.index.scope().digest(),
                Duration::from_secs(10),
            )
            .await
            .unwrap();
    }
    wait_for(|| {
        [
            single.index.status(),
            multi.index.status(),
            empty.index.status(),
        ]
        .into_iter()
        .all(|status| status.coverage == DiscoveryCoverage::CompleteAtWatermarks)
    })
    .await;
    assert_eq!(
        single.index.lookup(std::slice::from_ref(&selected_key))[0]
            .replicas
            .len(),
        1
    );

    gate.partition().await;
    for id in 0u32..400 {
        let key = StateKey::new(outside.into(), id.to_le_bytes().to_vec());
        source_inventory
            .test_change(
                &key,
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
        source_inventory
            .test_change(&key, ReplicaMedium::Dram, None)
            .unwrap();
    }
    source_inventory
        .test_change(&selected_key, ReplicaMedium::Dram, None)
        .unwrap();
    source_inventory
        .test_change(
            &selected_key,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    let repaired = source_runtime.capture_fence().unwrap();
    gate.heal(Duration::ZERO);
    single
        .runtime
        .await_fence(
            &repaired,
            &single.index.scope().digest(),
            Duration::from_secs(15),
        )
        .await
        .unwrap();
    assert_eq!(
        single.index.lookup(std::slice::from_ref(&selected_key))[0]
            .replicas
            .len(),
        1
    );
    assert!(source_inventory.status().history_gaps > 0);
    assert_eq!(
        multi.index.lookup(std::slice::from_ref(&selected_key))[0]
            .replicas
            .len(),
        1
    );
    assert_eq!(
        multi.index.lookup(std::slice::from_ref(&other_key))[0]
            .replicas
            .len(),
        1
    );
    for row in single
        .index
        .lookup(&[other_key.clone(), outside_key.clone()])
    {
        assert!(row.replicas.is_empty());
        assert_eq!(row.coverage, DiscoveryCoverage::Unavailable);
    }
    assert_eq!(
        empty
            .index
            .owner_status(source_view.owner().incarnation)
            .unwrap()
            .records,
        0
    );
    assert!(
        empty
            .runtime
            .await_fence(
                &fence,
                &InventoryScope::AllNamespaces.digest(),
                Duration::from_millis(10),
            )
            .await
            .is_err()
    );

    let before = source_inventory.sequence();
    source_inventory
        .test_change(
            &StateKey::new(outside.into(), vec![8; 32]),
            ReplicaMedium::Ssd,
            Some(metadata(ReplicaMedium::Ssd)),
        )
        .unwrap();
    let empty_interval = source_runtime.capture_fence().unwrap();
    assert!(empty_interval.inventory_sequence > before);
    single
        .runtime
        .await_fence(
            &empty_interval,
            &single.index.scope().digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(
        single
            .index
            .owner_status(source_view.owner().incarnation)
            .unwrap()
            .applied_sequence,
        empty_interval.inventory_sequence
    );
    assert_eq!(
        single.index.lookup(std::slice::from_ref(&selected_key))[0]
            .replicas
            .len(),
        1
    );

    source_inventory
        .test_change(&selected_key, ReplicaMedium::Dram, None)
        .unwrap();
    source_inventory
        .test_change(
            &selected_key,
            ReplicaMedium::Dram,
            Some(metadata(ReplicaMedium::Dram)),
        )
        .unwrap();
    let replacement = source_runtime.capture_fence().unwrap();
    single
        .runtime
        .await_fence(
            &replacement,
            &single.index.scope().digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(
        single.index.lookup(std::slice::from_ref(&selected_key))[0]
            .replicas
            .len(),
        1
    );

    for node in [empty, multi, single] {
        node.server.abort();
        node.cluster.shutdown().await;
    }
    source_server.abort();
    source.shutdown().await;
    gate.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(feature = "test-hooks")]
#[ignore = "requires ETCD_BIN; scope-changing restart must bootstrap a new incarnation"]
async fn scope_change_restart_never_reuses_the_old_view_or_cursor() {
    let etcd = Etcd::start(1).await;
    let first_namespace =
        "orbitkv:v2:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    let second_namespace =
        "orbitkv:v2:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    let source_port = free_port();
    let source_view = view(source_port);
    let (source, _, source_inventory) =
        join(&etcd, "scope-restart", "source", source_view.clone(), 60).await;
    let source_runtime = source.inventory();
    let source_server = serve(source_runtime.clone(), source_port).await;
    let first = scoped_node(
        &etcd,
        "scope-restart",
        "target",
        free_port(),
        InventoryScope::exact([first_namespace.into()]).unwrap(),
    )
    .await;
    let first_key = StateKey::new(first_namespace.into(), vec![1; 32]);
    let second_key = StateKey::new(second_namespace.into(), vec![1; 32]);
    for key in [&first_key, &second_key] {
        source_inventory
            .test_change(
                key,
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
    }
    let fence = source_runtime.capture_fence().unwrap();
    first
        .runtime
        .await_fence(
            &fence,
            &first.index.scope().digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(
        first.index.lookup(std::slice::from_ref(&first_key))[0]
            .replicas
            .len(),
        1
    );
    let old_scope = first.index.scope().digest();
    let old_incarnation = first.cluster.view.owner().incarnation;
    let old_index = first.index.clone();
    first.server.abort();
    first.cluster.shutdown().await;

    let second = scoped_node(
        &etcd,
        "scope-restart",
        "target",
        free_port(),
        InventoryScope::exact([second_namespace.into()]).unwrap(),
    )
    .await;
    assert_ne!(second.cluster.view.owner().incarnation, old_incarnation);
    assert_ne!(second.index.scope().digest(), old_scope);
    second
        .runtime
        .await_fence(
            &fence,
            &second.index.scope().digest(),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    assert_eq!(
        second.index.lookup(std::slice::from_ref(&second_key))[0]
            .replicas
            .len(),
        1
    );
    assert_eq!(
        second.index.lookup(std::slice::from_ref(&first_key))[0].coverage,
        DiscoveryCoverage::Unavailable
    );
    let old_row = &old_index.lookup(std::slice::from_ref(&first_key))[0];
    assert!(old_row.replicas.is_empty());
    assert_eq!(old_row.coverage, DiscoveryCoverage::Unavailable);
    second.server.abort();
    second.cluster.shutdown().await;
    source_server.abort();
    source.shutdown().await;
}

#[cfg(feature = "test-hooks")]
#[path = "inventory_capacity.rs"]
mod capacity;

#[cfg(feature = "test-hooks")]
#[path = "inventory_pressure.rs"]
mod pressure;

#[tokio::test]
async fn fenced_source_stops_before_queueing_an_inventory_frame() {
    let (runtime, _, _) = runtime_for_protocol_test();
    let (output, mut response) = mpsc::channel(4);
    let (_control, controls) = watch::channel(ClientControl::default());
    let mut sender = FrameSender::new(runtime.clone(), Uuid::new_v4(), output, controls);
    let frame = sender
        .frame(inventory_server_frame::Body::Progress(InventoryProgress {
            through_sequence: 0,
            source_head_sequence: 0,
        }))
        .unwrap();
    runtime.shared.membership.fence();
    assert!(sender.send(frame).await.is_err());
    assert!(response.try_recv().is_err());
    assert_eq!(runtime.status().frames_sent, 0);
    assert_eq!(runtime.status().outbound_queue_bytes, 0);
}

#[tokio::test]
async fn aborted_receiver_session_releases_the_active_diagnostic() {
    let (runtime, _, _) = runtime_for_protocol_test();
    let shared = runtime.shared.clone();
    let task = tokio::spawn(async move {
        let _active = ActiveSession::new(shared, SessionDirection::Receiver);
        std::future::pending::<()>().await;
    });
    while runtime.status().receiver_sessions == 0 {
        tokio::task::yield_now().await;
    }
    assert_eq!(runtime.status().receiver_sessions, 1);
    task.abort();
    let _ = task.await;
    assert_eq!(runtime.status().receiver_sessions, 0);
}
