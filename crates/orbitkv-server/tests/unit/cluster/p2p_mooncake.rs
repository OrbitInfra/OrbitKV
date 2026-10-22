//! P2P Mooncake remote fetch integration test.
//!
//! Verifies the end-to-end flow:
//! Engine A saves blocks → etcd publishes locations → local index discovers them → Engine B fetches via Mooncake READ
//! → data integrity verified.
//!
//! Run with: `cargo test -p orbitkv-server --lib cluster::tests::p2p_mooncake -- --ignored`
//! Requires ETCD_BIN, CUDA, and a prebuilt Mooncake runtime.

use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::etcd::{Etcd, join, view, wait_for};
use super::gate::TcpGate;
use crate::P2pTransferService;
use crate::proto::engine::engine_server::EngineServer;
use crate::proto::engine::inventory_server::InventoryServer;
use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use orbitkv_catalog::GlobalIndex;
use orbitkv_core::transfer::local::{LocalRestoreExecutor, LocalTensor, RawRestorePart};
use orbitkv_core::*;
use orbitkv_proto::proto::engine::{
    OpenTransferWindowRequest, QueryBlocksForTransferRequest, ReleaseTransferLockRequest,
    TransferTicket, engine_client::EngineClient,
};
use orbitkv_state::group_hash;
use orbitkv_state::{BlockCandidates, StateKey};
use tonic::transport::Server;

// ── GPU buffer (from crates/orbitkv-core/tests/common/gpu_buffer.rs) ──────────────

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        assert!(len > 0);
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }

    fn zero(&self) {
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.ptr, 0, self.len) },
            "cuMemsetD8_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

// ── Helpers (from crates/orbitkv-core/tests/common/helpers.rs) ────────────────────

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = ((i % 251) + 1) as u8;
        block.fill(fill);
    }
}

fn make_block_ids(num_blocks: usize) -> Vec<usize> {
    (0..num_blocks).collect()
}

fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

async fn restore_and_wait(
    engine: &OrbitKVEngine,
    gpu: &GpuBuffer,
    layer: &str,
    blocks: usize,
    execution: RestoreExecution,
) {
    match execution {
        RestoreExecution::Local(mut grant) => {
            let tensor = LocalTensor::new(
                layer.into(),
                gpu.as_u64(),
                gpu.len,
                0,
                blocks,
                gpu.len / blocks,
                0,
                1,
            )
            .expect("local destination geometry");
            let mut executor = LocalRestoreExecutor::new(
                0,
                vec![tensor],
                engine.payload_arenas().expect("export payload arenas"),
                TransferMode::Direct,
            )
            .expect("import source payload arenas");
            let result = loop {
                let (bytes, more) = grant.encoded_plan();
                let plan = RawRestorePart::decode(bytes).expect("decode local Restore plan");
                let result = executor.execute(&plan, &mut Default::default(), true, || {}, None);
                if result.is_err() || !more {
                    break result;
                }
                assert!(grant.advance_plan());
            };
            grant.finish(result.is_ok(), None);
            result.expect("local Restore failed");
        }
        RestoreExecution::Managed(receiver) => {
            tokio::time::timeout(Duration::from_secs(5), receiver)
                .await
                .expect("restore timeout")
                .expect("restore worker disappeared")
                .result
                .expect("restore failed");
        }
    }
}

// ── Infrastructure ──────────────────────────────────────────────────────────

fn get_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local_addr")
        .port()
}

async fn wait_for_grpc_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .is_ok()
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for gRPC on port {port}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn spawn_engine_server(
    engine: Arc<OrbitKVEngine>,
    inventory: crate::cluster::InventoryRuntime,
    port: u16,
) -> PeerServer {
    let service = P2pTransferService::new(engine);
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .add_service(InventoryServer::new(inventory.service()))
            .serve_with_shutdown(addr, async {
                let _ = stopped.await;
            })
            .await
            .expect("peer service");
    });
    wait_for_grpc_ready(port).await;
    PeerServer {
        stop: Some(stop),
        task: Some(task),
    }
}

struct PeerServer {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl PeerServer {
    async fn shutdown(mut self) {
        let _ = self.stop.take().unwrap().send(());
        tokio::time::timeout(Duration::from_secs(5), self.task.as_mut().unwrap())
            .await
            .expect("peer service did not drain")
            .expect("peer service task failed");
        self.task.take();
    }
}

impl Drop for PeerServer {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn capture_inventory_fence(
    engine: &OrbitKVEngine,
    cluster: &crate::cluster::Cluster,
) -> orbitkv_state::InventoryFence {
    engine.flush_saves().await;
    cluster.inventory().capture_fence().unwrap()
}

async fn wait_for_fence(index: &GlobalIndex, fence: &orbitkv_state::InventoryFence) {
    wait_for(|| {
        index
            .owner_watermark(fence.source_incarnation)
            .is_some_and(|(_, sequence)| sequence >= fence.inventory_sequence)
    })
    .await;
}

fn locate(index: &GlobalIndex, namespace: &str, hashes: &[Vec<u8>]) -> Vec<BlockCandidates> {
    index.lookup(
        &hashes
            .iter()
            .map(|hash| StateKey::new(namespace.into(), hash.clone()))
            .collect::<Vec<_>>(),
    )
}

async fn wait_for_cache(
    engine: &OrbitKVEngine,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let status = engine
            .count_prefix_hit_blocks_with_prefetch(
                instance_id,
                "wait-for-cache",
                block_hashes,
                orbitkv_core::QueryMode::Demand,
            )
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        let hit = {
            let QueryResult { blocks, .. } = status;
            blocks.len()
        };
        if hit >= expected_hit {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected_hit} cached blocks (got {hit})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the live-store assertion keeps engine, layout and payload identities explicit"
)]
async fn restore_cached_image(
    engine: &OrbitKVEngine,
    gpu: &GpuBuffer,
    instance_id: &str,
    request_id: &str,
    layer: &str,
    block_hashes: &[Vec<u8>],
    expected: &[u8],
) {
    gpu.zero();
    let result = engine
        .count_prefix_hit_blocks_with_prefetch(
            instance_id,
            request_id,
            block_hashes,
            orbitkv_core::QueryMode::Demand,
        )
        .await
        .expect("query local cache");
    assert_eq!(result.blocks.len(), block_hashes.len());
    let lease = engine
        .create_query_lease(instance_id, result.blocks)
        .expect("create local query lease");
    let receiver = engine
        .restore(
            instance_id,
            0,
            DEVICE_ID,
            &[vec![layer]],
            &[(lease, vec![(0..block_hashes.len()).map(Some).collect()])],
        )
        .expect("restore local cache image");
    restore_and_wait(engine, gpu, layer, block_hashes.len(), receiver).await;
    assert_eq!(gpu.copy_to_host(), expected);
}

async fn wait_for_index_registration(
    store: &GlobalIndex,
    namespace: &str,
    hashes: &[Vec<u8>],
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = locate(store, namespace, hashes);
        let count = found
            .iter()
            .take_while(|row| !row.replicas.is_empty())
            .count();
        if count >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for global index registration ({} / {})",
            count,
            expected
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_index_ownership(
    store: &GlobalIndex,
    namespace: &str,
    hashes: &[Vec<u8>],
    node: &str,
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = locate(store, namespace, hashes);
        let owned = found
            .iter()
            .filter(|entry| entry.replicas.iter().any(|r| r.owner.endpoint == node))
            .count();
        if owned >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for global index ownership by {node} ({owned} / {expected})"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn ib_device() -> String {
    std::env::var("ORBITKV_IB_DEVICE").unwrap_or_else(|_| "mlx5_1".into())
}

fn mooncake_nics() -> Vec<String> {
    if std::env::var_os("MC_FORCE_TCP").is_some() {
        Vec::new()
    } else {
        vec![ib_device()]
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

// Cross multiple discovery/authorization segments so the opt-in lookahead path
// also exercises real source-budget pressure and Mooncake READ ownership.
const NUM_BLOCKS: usize = orbitkv_state::DISCOVERY_MAX_KEYS * 2 + 4;
const BLOCK_SIZE: usize = 1024;
const TOTAL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;
const NAMESPACE: &str = "test-p2p";
const LAYER: &str = "layer_0";
const DEVICE_ID: i32 = 0;

#[tokio::test]
#[ignore = "requires CUDA and Mooncake; set MC_FORCE_TCP=1 for the same-host TCP gate"]
async fn p2p_mooncake_remote_fetch_roundtrip() {
    peer_remote_fetch_roundtrip(PeerTransferBackend::Mooncake).await;
}

#[cfg(feature = "orbitte")]
#[tokio::test]
#[ignore = "OrbitTE peer hardware gate: requires CUDA and ETCD_BIN; actual Manager gRPC, inventory and TCP READ"]
async fn p2p_orbitte_remote_fetch_roundtrip() {
    peer_remote_fetch_roundtrip(PeerTransferBackend::OrbitTeTcp).await;
}

#[cfg(feature = "orbitte-rdma")]
#[tokio::test]
#[ignore = "requires CUDA13, ETCD_BIN and explicit ORBITKV_TEST_ORBITTE_RDMA_DEVICE/PORT/GID_INDEX; actual Manager GPU path"]
async fn p2p_orbitte_rdma_remote_fetch_roundtrip() {
    peer_remote_fetch_roundtrip(PeerTransferBackend::OrbitTeRdma).await;
}

#[cfg(feature = "orbitte-rdma")]
fn orbitte_rdma_config(backend: PeerTransferBackend) -> Option<orbitkv_core::OrbitTeRdmaConfig> {
    (backend == PeerTransferBackend::OrbitTeRdma).then(|| orbitkv_core::OrbitTeRdmaConfig {
        port: std::env::var("ORBITKV_TEST_ORBITTE_RDMA_PORT")
            .unwrap()
            .parse()
            .unwrap(),
        gid_index: std::env::var("ORBITKV_TEST_ORBITTE_RDMA_GID_INDEX")
            .unwrap()
            .parse()
            .unwrap(),
        ..orbitkv_core::OrbitTeRdmaConfig::new(
            std::env::var("ORBITKV_TEST_ORBITTE_RDMA_DEVICE").unwrap(),
        )
    })
}

async fn peer_remote_fetch_roundtrip(backend: PeerTransferBackend) {
    orbitkv_common::logging::init_stdout_colored("debug");
    let _cuda_ctx = CudaContext::new(0).expect("CUDA init");

    let coordinator = Etcd::start(1).await;
    let (_observer, stores, _) = join(
        &coordinator,
        "raw-gpu",
        "observer",
        view(get_free_port()),
        60,
    )
    .await;
    let port_a = get_free_port();
    let membership_a = view(port_a);
    let (cluster_a, index_a, inventory_a) =
        join(&coordinator, "raw-gpu", "a", membership_a.clone(), 60).await;
    let config_a = EngineConfig {
        membership: Some(membership_a.clone()),
        global_index: Some(index_a),
        inventory: Some(inventory_a),
        peer_transfer_backend: backend,
        #[cfg(feature = "orbitte-rdma")]
        orbitte_rdma: orbitte_rdma_config(backend),
        enable_numa_affinity: backend == PeerTransferBackend::Mooncake,
        mooncake_nic_names: if backend == PeerTransferBackend::Mooncake {
            mooncake_nics()
        } else {
            Vec::new()
        },
        transfer_budget_bytes: Some(TOTAL_SIZE),
        transfer_lock_timeout: Duration::ZERO,
        ..EngineConfig::default()
    };
    let engine_a = Arc::new(
        OrbitKVEngine::new_with_config(16 << 20, false, config_a).expect("engine A should start"),
    );

    // ── 3. Start Engine A gRPC server ──
    let _source_server =
        spawn_engine_server(Arc::clone(&engine_a), cluster_a.inventory(), port_a).await;

    // ── 4. Save blocks on Engine A ──
    let gpu_a = GpuBuffer::alloc(TOTAL_SIZE);
    let mut host_data = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut host_data, BLOCK_SIZE);
    gpu_a.copy_from_host(&host_data);

    engine_a
        .register_context_layer_batch(
            "inst-a",
            NAMESPACE,
            DEVICE_ID,
            0, // tp_rank
            0, // pp_rank
            1, // tp_size
            1, // world_size
            &[LAYER.to_string()],
            &[gpu_a.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0], // kv_strides
            &[1], // segments
            TransferMode::Direct,
            false,
        )
        .expect("register layer on engine A");

    let cache_namespace = engine_a
        .instance_namespace("inst-a")
        .expect("sealed namespace");
    let block_ids = make_block_ids(NUM_BLOCKS);
    let block_hashes = make_block_hashes(NUM_BLOCKS, 42);
    let stored_hashes: Vec<_> = block_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();

    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            engine_a
                .registration_generation("inst-a")
                .unwrap_or_default(),
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.to_string(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .expect("save blocks on engine A");

    // ── 5. Wait for Engine A cache ──
    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(5),
    )
    .await;

    // ── 6. Wait for acknowledged owner inventory ──
    let source_fence = capture_inventory_fence(&engine_a, &cluster_a).await;
    wait_for_fence(&stores, &source_fence).await;
    wait_for_index_registration(
        &stores,
        &cache_namespace,
        &stored_hashes,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // Source authorization fences both restarts and individual residency episodes.
    let evidence = locate(&stores, &cache_namespace, &stored_hashes);
    let grant_blocks = orbitkv_state::DISCOVERY_MAX_KEYS;
    let mut peer = EngineClient::connect(format!("http://127.0.0.1:{port_a}"))
        .await
        .unwrap();
    let window = peer
        .open_transfer_window(OpenTransferWindowRequest {
            owner_incarnation: evidence[0].replicas[0].owner.incarnation.to_string(),
            requester_incarnation: uuid::Uuid::new_v4().to_string(),
        })
        .await
        .unwrap()
        .into_inner()
        .window_id;
    let ticket = TransferTicket {
        window_id: window,
        slot: 0,
        generation: 1,
    };
    let authorization = QueryBlocksForTransferRequest {
        namespace: cache_namespace.clone(),
        block_hashes: stored_hashes[..grant_blocks].to_vec(),
        ticket: Some(ticket.clone()),
        owner_incarnation: evidence[0].replicas[0].owner.incarnation.to_string(),
        residency_sequences: evidence[..grant_blocks]
            .iter()
            .map(|r| r.replicas[0].sequence)
            .collect(),
    };
    let mut stale_runtime = authorization.clone();
    stale_runtime.owner_incarnation = uuid::Uuid::new_v4().to_string();
    let mut stale_residency = authorization.clone();
    stale_residency.residency_sequences[0] += 1;
    for stale in [stale_runtime, stale_residency] {
        assert_eq!(
            peer.query_blocks_for_transfer(stale)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
    }
    let granted = peer
        .query_blocks_for_transfer(authorization.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(granted.blocks.len(), grant_blocks);
    assert_eq!(
        granted.native_source_epoch,
        engine_a.transfer_source_epoch().unwrap()
    );
    assert_eq!(
        granted.transfer_endpoint,
        engine_a.transfer_endpoint().unwrap()
    );
    match backend {
        PeerTransferBackend::OrbitTeTcp => {
            assert!(granted.transfer_endpoint.starts_with("orbitte+tcp://"));
        }
        PeerTransferBackend::OrbitTeRdma => {
            assert!(granted.transfer_endpoint.starts_with("orbitte+rdma://"));
        }
        PeerTransferBackend::Mooncake => {}
    }
    assert_eq!(engine_a.expire_transfer_locks(), 1);
    assert_eq!(engine_a.expire_transfer_locks(), 0);
    assert_eq!(
        peer.query_blocks_for_transfer(authorization.clone())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    let mut concurrent = authorization.clone();
    concurrent.ticket = Some(TransferTicket {
        slot: 1,
        ..ticket.clone()
    });
    assert_eq!(
        peer.query_blocks_for_transfer(concurrent)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(ticket.clone()),
    })
    .await
    .unwrap();
    // Release-before-authorize must fence a queued RPC without ever pinning data.
    let cancelled = TransferTicket {
        generation: 2,
        ..ticket.clone()
    };
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(cancelled.clone()),
    })
    .await
    .unwrap();
    let mut queued = authorization.clone();
    queued.ticket = Some(cancelled);
    assert_eq!(
        peer.query_blocks_for_transfer(queued)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    for invalid in [
        None,
        Some(TransferTicket {
            slot: 64,
            ..ticket.clone()
        }),
        Some(TransferTicket {
            generation: 0,
            ..ticket.clone()
        }),
        Some(TransferTicket {
            window_id: "bad-id".into(),
            ..ticket.clone()
        }),
    ] {
        let mut query = authorization.clone();
        query.ticket = invalid.clone();
        assert_eq!(
            peer.query_blocks_for_transfer(query)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            peer.release_transfer_lock(ReleaseTransferLockRequest { ticket: invalid })
                .await
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
    }

    // ── 7. Create Engine B (fetcher) ──
    let port_b = get_free_port();
    let membership_b = view(port_b);
    let (cluster_b, index_b, inventory_b) =
        join(&coordinator, "raw-gpu", "b", membership_b.clone(), 60).await;
    wait_for(|| membership_b.permits(membership_a.owner())).await;
    let config_b = EngineConfig {
        membership: Some(membership_b.clone()),
        global_index: Some(index_b),
        inventory: Some(inventory_b),
        peer_transfer_backend: backend,
        #[cfg(feature = "orbitte-rdma")]
        orbitte_rdma: orbitte_rdma_config(backend),
        enable_numa_affinity: backend == PeerTransferBackend::Mooncake,
        mooncake_nic_names: if backend == PeerTransferBackend::Mooncake {
            mooncake_nics()
        } else {
            Vec::new()
        },
        ..EngineConfig::default()
    };
    let engine_b = Arc::new(
        OrbitKVEngine::new_with_config(16 << 20, false, config_b).expect("engine B should start"),
    );
    let _target_server =
        spawn_engine_server(Arc::clone(&engine_b), cluster_b.inventory(), port_b).await;

    let gpu_b = GpuBuffer::alloc(TOTAL_SIZE);
    gpu_b.zero();

    engine_b
        .register_context_layer_batch(
            "inst-b",
            NAMESPACE,
            DEVICE_ID,
            0, // tp_rank
            0, // pp_rank
            1, // tp_size
            1, // world_size
            &[LAYER.to_string()],
            &[gpu_b.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0],
            &[1],
            TransferMode::Direct,
            false,
        )
        .expect("register layer on engine B");

    // ── 8. Start the remote query before the producer registers the blocks ──
    let delayed_hashes = make_block_hashes(NUM_BLOCKS, 43);
    let stored_delayed: Vec<_> = delayed_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();
    let mut waiting = Box::pin(engine_b.count_prefix_hit_blocks_with_prefetch(
        "inst-b",
        "req-wait-for-producer",
        &delayed_hashes,
        orbitkv_core::QueryMode::WaitForFullPrefix,
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut waiting)
            .await
            .is_err()
    );

    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            engine_a
                .registration_generation("inst-a")
                .unwrap_or_default(),
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.to_string(),
                block_ids: block_ids.clone(),
                block_hashes: delayed_hashes.clone(),
            }],
        )
        .await
        .expect("save delayed blocks on engine A");

    wait_for_index_registration(
        &stores,
        &cache_namespace,
        &stored_delayed,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // ── 9. Engine B observes the producer and fetches via Mooncake READ ──
    let result = tokio::time::timeout(Duration::from_secs(30), waiting)
        .await
        .expect("remote fetch timeout")
        .expect("remote fetch");
    assert_eq!(result.blocks.len(), NUM_BLOCKS);
    let lease = engine_b
        .create_query_lease("inst-b", result.blocks)
        .expect("lease");

    // ── 9b. Verify Engine B re-registered fetched blocks to global index ──
    // Mooncake-fetched blocks are now resident on B, so B must advertise them so
    // other nodes can discover and fetch from B (not just from A).
    let restored_fence = capture_inventory_fence(&engine_b, &cluster_b).await;
    wait_for_fence(&stores, &restored_fence).await;
    wait_for_index_ownership(
        &stores,
        &cache_namespace,
        &stored_delayed,
        &format!("127.0.0.1:{port_b}"),
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // Eviction removes A's evidence while the copied replica on B remains usable.
    assert!(engine_a.cleanup_memory_cache().evicted_blocks > 0);
    let evicted_fence = capture_inventory_fence(&engine_a, &cluster_a).await;
    wait_for_fence(&stores, &evicted_fence).await;
    let remaining = locate(&stores, &cache_namespace, &stored_delayed);
    assert_eq!(remaining.len(), NUM_BLOCKS);
    for entry in remaining {
        assert_eq!(
            entry
                .replicas
                .into_iter()
                .map(|r| r.owner.endpoint)
                .collect::<Vec<_>>(),
            vec![format!("127.0.0.1:{port_b}")]
        );
    }
    assert!(
        locate(&stores, NAMESPACE, &delayed_hashes)
            .iter()
            .all(|row| row.replicas.is_empty())
    );

    assert_eq!(
        peer.query_blocks_for_transfer(authorization.clone())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            engine_a
                .registration_generation("inst-a")
                .unwrap_or_default(),
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.into(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .unwrap();
    let fresh_fence = capture_inventory_fence(&engine_a, &cluster_a).await;
    wait_for_fence(&stores, &fresh_fence).await;
    assert_eq!(
        peer.query_blocks_for_transfer(authorization)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );

    // ── 10. Load from Engine B cache → GPU ──
    let fresh = locate(&stores, &cache_namespace, &stored_hashes);
    let fresh_authorization = QueryBlocksForTransferRequest {
        namespace: cache_namespace.clone(),
        block_hashes: stored_hashes[..grant_blocks].to_vec(),
        ticket: Some(TransferTicket {
            generation: 3,
            ..ticket.clone()
        }),
        owner_incarnation: membership_a.owner().incarnation.to_string(),
        residency_sequences: fresh[..grant_blocks]
            .iter()
            .map(|row| {
                row.replicas
                    .iter()
                    .find(|replica| replica.owner == *membership_a.owner())
                    .unwrap()
                    .sequence
            })
            .collect(),
    };
    peer.query_blocks_for_transfer(fresh_authorization.clone())
        .await
        .unwrap()
        .into_inner();
    membership_a.fence();
    membership_b.fence();
    assert_eq!(
        peer.query_blocks_for_transfer(fresh_authorization)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    peer.release_transfer_lock(ReleaseTransferLockRequest {
        ticket: Some(TransferTicket {
            generation: 3,
            ..ticket
        }),
    })
    .await
    .unwrap();
    let resident = engine_b
        .count_prefix_hit_blocks_with_prefetch(
            "inst-b",
            "fenced-local-hit",
            &delayed_hashes,
            orbitkv_core::QueryMode::Demand,
        )
        .await
        .unwrap();
    assert_eq!(
        resident.blocks.len(),
        NUM_BLOCKS,
        "local hits survive membership loss"
    );
    let remote = engine_b
        .count_prefix_hit_blocks_with_prefetch(
            "inst-b",
            "fenced-remote-miss",
            &block_hashes,
            orbitkv_core::QueryMode::Demand,
        )
        .await
        .unwrap();
    assert!(
        remote.blocks.is_empty(),
        "fenced requester must not fetch remote-only blocks"
    );

    let receiver = engine_b
        .restore(
            "inst-b",
            0,
            DEVICE_ID,
            &[vec![LAYER]],
            &[(lease, vec![block_ids.iter().copied().map(Some).collect()])],
        )
        .expect("batch_load on engine B");

    restore_and_wait(&engine_b, &gpu_b, LAYER, NUM_BLOCKS, receiver).await;

    // ── 11. Verify data integrity ──
    let loaded = gpu_b.copy_to_host();
    assert_eq!(
        loaded, host_data,
        "GPU data mismatch: remote-fetched blocks differ from original"
    );
    cluster_b.shutdown().await;
    cluster_a.shutdown().await;
    _target_server.shutdown().await;
    _source_server.shutdown().await;
    engine_b
        .unregister_instance_and_wait("inst-b")
        .await
        .unwrap();
    engine_a
        .unregister_instance_and_wait("inst-a")
        .await
        .unwrap();
    assert!(engine_b.list_instance_ids().is_empty());
    assert!(engine_a.list_instance_ids().is_empty());
    _observer.shutdown().await;
    if matches!(
        backend,
        PeerTransferBackend::OrbitTeTcp | PeerTransferBackend::OrbitTeRdma
    ) {
        let artifact_root = std::env::var_os("ORBITKV_METADATA_ARTIFACT_DIR")
            .expect("explicit external peer evidence directory");
        std::fs::write(
            std::path::Path::new(&artifact_root).join("peer-maps.txt"),
            std::fs::read_to_string("/proc/self/maps").unwrap(),
        )
        .unwrap();
        println!(
            "ORBITTE_PEER_GATES GPU Save/Restore bytes verified; source grants, generations and membership controls passed; peer service tasks drained; backend={backend:?}"
        );
    }
}

#[tokio::test]
#[ignore = "requires CUDA, Mooncake and ETCD_BIN; live DRAM journal overflow and metadata loss"]
async fn live_dram_journal_overflow_rebuilds_and_local_payload_survives_metadata_loss() {
    orbitkv_common::logging::init_stdout_colored("debug");
    let _cuda_ctx = CudaContext::new(0).expect("CUDA init");

    let coordinator = Etcd::start(1).await;
    let observer_view = view(get_free_port());
    let (observer_cluster, observer_index, _) = join(
        &coordinator,
        "live-journal",
        "observer",
        observer_view.clone(),
        60,
    )
    .await;
    let etcd_gate = TcpGate::start(&coordinator.endpoints[0]).await;
    let source_backend_port = get_free_port();
    let peer_gate = TcpGate::start(&format!("http://127.0.0.1:{source_backend_port}")).await;
    let source_port = peer_gate
        .endpoint
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let source_view = view(source_port);
    let source_index = Arc::new(GlobalIndex::new(
        source_view.clone(),
        1 << 20,
        Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
    ));
    let inventory = Arc::new(ResidencyInventory::new(1024));
    let source_cluster = crate::cluster::Cluster::join(
        std::slice::from_ref(&etcd_gate.endpoint),
        "live-journal",
        "source",
        12,
        source_view.clone(),
        inventory.clone(),
        source_index.clone(),
    )
    .await
    .expect("join source through TCP gate");
    wait_for(|| {
        observer_view.permits(source_view.owner())
            && source_view.permits(observer_view.owner())
            && observer_index.status().registration_valid
    })
    .await;

    const BLOCKS: usize = 64;
    const BYTES: usize = BLOCKS * BLOCK_SIZE;
    const INSTANCE: &str = "live-journal-source";
    const NAMESPACE: &str = "live-journal-payload";
    let engine = Arc::new(
        OrbitKVEngine::new_with_config(
            4 << 20,
            false,
            EngineConfig {
                membership: Some(source_view.clone()),
                global_index: Some(source_index),
                inventory: Some(inventory.clone()),
                mooncake_nic_names: Vec::new(),
                ..EngineConfig::default()
            },
        )
        .expect("create live-store engine"),
    );
    let _source_server = spawn_engine_server(
        Arc::clone(&engine),
        source_cluster.inventory(),
        source_backend_port,
    )
    .await;
    let gpu = GpuBuffer::alloc(BYTES);
    let mut expected = vec![0u8; BYTES];
    fill_test_pattern(&mut expected, BLOCK_SIZE);
    gpu.copy_from_host(&expected);
    engine
        .register_context_layer_batch(
            INSTANCE,
            NAMESPACE,
            DEVICE_ID,
            0,
            0,
            1,
            1,
            &[LAYER.to_string()],
            &[gpu.as_u64()],
            &[BYTES],
            &[BLOCKS],
            &[BLOCK_SIZE],
            &[0],
            &[1],
            TransferMode::Direct,
            false,
        )
        .expect("register live-store GPU layer");
    let block_ids = make_block_ids(BLOCKS);
    let initial_hashes = make_block_hashes(BLOCKS, 71);
    let initial_stored: Vec<_> = initial_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();
    engine
        .batch_save_kv_blocks_from_ipc(
            INSTANCE,
            engine.registration_generation(INSTANCE).unwrap_or_default(),
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.into(),
                block_ids: block_ids.clone(),
                block_hashes: initial_hashes.clone(),
            }],
        )
        .await
        .expect("save initial live-store blocks");
    let initial_fence = capture_inventory_fence(&engine, &source_cluster).await;
    wait_for_fence(&observer_index, &initial_fence).await;
    wait_for_index_registration(
        &observer_index,
        &engine.instance_namespace(INSTANCE).unwrap(),
        &initial_stored,
        BLOCKS,
        Duration::from_secs(10),
    )
    .await;
    let initial_sequence = initial_fence.inventory_sequence;

    peer_gate.partition().await;
    let transient_partition_started = Instant::now();
    assert_eq!(engine.cleanup_memory_cache().evicted_blocks, BLOCKS);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let final_hashes = make_block_hashes(BLOCKS, 72);
    let final_stored: Vec<_> = final_hashes
        .iter()
        .map(|hash| group_hash(hash, 0))
        .collect();
    engine
        .batch_save_kv_blocks_from_ipc(
            INSTANCE,
            engine.registration_generation(INSTANCE).unwrap_or_default(),
            0,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.into(),
                block_ids: block_ids.clone(),
                block_hashes: final_hashes.clone(),
            }],
        )
        .await
        .expect("save replacement live-store blocks");
    wait_for_cache(
        &engine,
        INSTANCE,
        &final_hashes,
        BLOCKS,
        Duration::from_secs(5),
    )
    .await;
    let through = inventory.sequence();
    assert_eq!(
        inventory.changes(initial_sequence, through),
        Err(InventoryReadError::HistoryGap),
        "live-store churn did not overflow the retained journal"
    );
    assert!(
        locate(
            &observer_index,
            &engine.instance_namespace(INSTANCE).unwrap(),
            &final_stored,
        )
        .iter()
        .all(|row| row.replicas.is_empty()),
        "partition exposed an unpublished replacement"
    );
    assert!(
        locate(
            &observer_index,
            &engine.instance_namespace(INSTANCE).unwrap(),
            &initial_stored,
        )
        .iter()
        .all(|row| { row.replicas.len() == 1 && row.replicas[0].owner == *source_view.owner() }),
        "partition discarded the last complete source view"
    );
    restore_cached_image(
        &engine,
        &gpu,
        INSTANCE,
        "partition-local-restore",
        LAYER,
        &final_hashes,
        &expected,
    )
    .await;
    let transient_partition_ms = transient_partition_started.elapsed().as_secs_f64() * 1000.0;

    peer_gate.heal(Duration::ZERO);
    let reconciliation_started = Instant::now();
    let final_fence = capture_inventory_fence(&engine, &source_cluster).await;
    wait_for_fence(&observer_index, &final_fence).await;
    let namespace = engine.instance_namespace(INSTANCE).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let initial = locate(&observer_index, &namespace, &initial_stored);
        let final_rows = locate(&observer_index, &namespace, &final_stored);
        if observer_index
            .owner_watermark(source_view.owner().incarnation)
            .is_some_and(|(_, sequence)| sequence >= final_fence.inventory_sequence)
            && initial.iter().all(|row| row.replicas.is_empty())
            && final_rows
                .iter()
                .all(|row| row.replicas.len() == 1 && row.replicas[0].owner == *source_view.owner())
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "live-store snapshot did not converge after journal overflow"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reconciliation_ms = reconciliation_started.elapsed().as_secs_f64() * 1000.0;
    assert!(source_view.registration_valid());

    etcd_gate.partition().await;
    let expiry_started = Instant::now();
    wait_for(|| !source_view.registration_valid()).await;
    wait_for(|| {
        let observer = observer_index.status();
        observer.registration_valid
            && !observer_view.permits(source_view.owner())
            && locate(&observer_index, &namespace, &final_stored)
                .iter()
                .all(|row| row.replicas.is_empty())
    })
    .await;
    assert!(observer_index.status().registration_valid);
    assert!(!observer_view.permits(source_view.owner()));
    restore_cached_image(
        &engine,
        &gpu,
        INSTANCE,
        "expired-membership-local-restore",
        LAYER,
        &final_hashes,
        &expected,
    )
    .await;
    let expiry_ms = expiry_started.elapsed().as_secs_f64() * 1000.0;
    etcd_gate.heal(Duration::ZERO);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !source_view.registration_valid(),
        "expired runtime silently resumed its old incarnation"
    );

    std::fs::write(
        coordinator.directory.join("live-journal-overflow.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "initial_sequence": initial_sequence,
            "final_sequence": through,
            "final_fence": final_fence,
            "journal_bytes": 1024,
            "initial_records": BLOCKS,
            "final_records": BLOCKS,
            "transient_partition_ms": transient_partition_ms,
            "reconciliation_ms": reconciliation_ms,
            "lease_expiry_ms": expiry_ms,
            "local_payload_bytes": BYTES,
            "local_payload_exact_after_transient_partition": true,
            "local_payload_exact_after_lease_expiry": true,
            "last_complete_view_retained_during_transient_partition": true,
            "observer_available_after_source_expiry": true,
            "expired_incarnation_remained_fenced_after_heal": true,
        }))
        .unwrap(),
    )
    .unwrap();

    source_cluster.shutdown().await;
    observer_cluster.shutdown().await;
    peer_gate.shutdown().await;
    etcd_gate.shutdown().await;
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring, nvCOMP 5.3 and Mooncake; set MC_FORCE_TCP=1 for same-host TCP"]
async fn encoded_peer_payloads_restore_the_same_gpu_image() {
    use orbitkv_state::{AttentionRole, Scalar16, StorageFormat};
    let _cuda = CudaContext::new(0).unwrap();
    let coordinator = Etcd::start(1).await;
    for (index, codec) in [
        StorageCodec::Ans,
        StorageCodec::Fp8,
        StorageCodec::TurboQuant4,
        StorageCodec::TurboQuant3,
    ]
    .into_iter()
    .enumerate()
    {
        let port_a = get_free_port();
        let port_b = get_free_port();
        let view_a = view(port_a);
        let view_b = view(port_b);
        let cluster_name = format!("encoded-gpu-{index}");
        let (cluster_a, index_a, inventory_a) =
            join(&coordinator, &cluster_name, "a", view_a.clone(), 60).await;
        let (cluster_b, index_b, inventory_b) =
            join(&coordinator, &cluster_name, "b", view_b.clone(), 60).await;
        wait_for(|| view_b.permits(view_a.owner())).await;
        let stores = index_b.clone();
        let make_engine = |view, global_index, inventory, ssd_cache_config| {
            Arc::new(
                OrbitKVEngine::new_with_config(
                    16 << 20,
                    false,
                    EngineConfig {
                        codec,
                        ssd_cache_config,
                        membership: Some(view),
                        global_index: Some(global_index),
                        inventory: Some(inventory),
                        mooncake_nic_names: mooncake_nics(),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        };
        let source = make_engine(view_a.clone(), index_a, inventory_a, None);
        let target_disk = tempfile::tempdir().unwrap();
        let read_path = if index % 2 == 0 {
            SsdReadPath::Uring
        } else {
            SsdReadPath::Cufile
        };
        let target = make_engine(
            view_b.clone(),
            index_b,
            inventory_b,
            Some(SsdCacheConfig {
                cache_paths: vec![target_disk.path().join("peer-target.bin")],
                capacity_bytes: 4 << 20,
                backend: SsdBackend::Uring,
                read_path: Some(read_path),
                ..SsdCacheConfig::default()
            }),
        );
        let _source_server =
            spawn_engine_server(source.clone(), cluster_a.inventory(), port_a).await;
        let _target_server =
            spawn_engine_server(target.clone(), cluster_b.inventory(), port_b).await;
        let gpu_source = GpuBuffer::alloc(8192);
        let gpu_target = GpuBuffer::alloc(8192);
        gpu_source.copy_from_host(&[0xa0, 0x3f].repeat(4096));
        gpu_target.zero();
        for (engine, gpu, id) in [
            (&source, &gpu_source, "source"),
            (&target, &gpu_target, "target"),
        ] {
            engine
                .register_context_layer_batch_strided(
                    id,
                    "encoded-peers",
                    0,
                    0,
                    0,
                    1,
                    1,
                    &["layer".into()],
                    &[gpu.as_u64()],
                    &[8192],
                    &[1],
                    &[8192],
                    &[0],
                    &[1],
                    None,
                    None,
                    Some(&[StorageFormat::Attention {
                        scalar: Scalar16::Bf16,
                        role: AttentionRole::Key,
                        head_dim: 128,
                        layer_index: 2,
                        layer_count: 8,
                    }]),
                    TransferMode::Direct,
                    false,
                )
                .unwrap();
        }
        let hashes = vec![b"encoded-page".to_vec()];
        source
            .batch_save_kv_blocks_from_ipc(
                "source",
                source.registration_generation("source").unwrap_or_default(),
                0,
                0,
                0,
                vec![LayerSave {
                    layer_name: "layer".into(),
                    block_ids: vec![0],
                    block_hashes: hashes.clone(),
                }],
            )
            .await
            .unwrap();
        wait_for_cache(&source, "source", &hashes, 1, Duration::from_secs(10)).await;
        let source_fence = capture_inventory_fence(&source, &cluster_a).await;
        wait_for_fence(&stores, &source_fence).await;
        let namespace = source.instance_namespace("source").unwrap();
        wait_for_index_registration(
            &stores,
            &namespace,
            &[group_hash(&hashes[0], 0)],
            1,
            Duration::from_secs(10),
        )
        .await;
        let mut images = Vec::new();
        for (engine, gpu, id) in [
            (&source, &gpu_source, "source"),
            (&target, &gpu_target, "target"),
        ] {
            let result = engine
                .count_prefix_hit_blocks_with_prefetch(id, "codec", &hashes, QueryMode::Demand)
                .await
                .unwrap();
            assert_eq!(result.blocks.len(), 1);
            if id == "target" {
                assert!(
                    matches!(&result.blocks[0], RestoreSource::Memory(_)),
                    "{codec:?}: {read_path:?} must preserve remote DRAM recovery on a local SSD miss"
                );
            }
            assert!(
                result.blocks[0].memory_footprint() < 8192,
                "{codec:?} did not use encoded residency"
            );
            let lease = engine.create_query_lease(id, result.blocks).unwrap();
            gpu.zero();
            let execution = engine
                .restore(id, 0, 0, &[vec!["layer"]], &[(lease, vec![vec![Some(0)]])])
                .unwrap();
            restore_and_wait(engine, gpu, "layer", 1, execution).await;
            images.push(gpu.copy_to_host());
        }
        assert_eq!(
            images[0], images[1],
            "peer codec metadata/payload changed for {codec:?}"
        );
        source.unregister_instance_and_wait("source").await.unwrap();
        target.unregister_instance_and_wait("target").await.unwrap();
        cluster_b.shutdown().await;
        cluster_a.shutdown().await;
    }
}
