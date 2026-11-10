//! Actual CPU cache paths. Included as private unit tests to use the existing
//! owners, without exposing production APIs just for benchmarking.
use super::super::*;
use crate::block::{RawBlock, SealedBlock, Segment};
use crate::storage::{
    Storage,
    dram::DramStore,
    ssd::{SsdBackend, SsdCacheConfig},
};
use crate::transfer_engine::Permission;
use orbitkv_proto::proto::engine::TransferSlotInfo;
use std::{
    hint::black_box,
    num::{NonZeroU64, NonZeroUsize},
    path::Path,
};

fn block(storage: &Storage, bytes: usize) -> RawBlock {
    let allocation = storage
        .allocate(NonZeroU64::new(bytes as u64).unwrap(), None)
        .unwrap();
    // SAFETY: newly allocated exclusive bytes, before publication or native I/O.
    unsafe { allocation.as_non_null().as_ptr().write_bytes(0x35, bytes) };
    RawBlock::single_segment(Segment::new(allocation.as_non_null(), bytes, allocation).unwrap())
}

fn storage(root: &Path, bytes: usize, persistent: bool) -> Arc<Storage> {
    Storage::new_with_config(
        128 << 20,
        false,
        crate::EngineConfig {
            transfer_transport: crate::TransportHint::Tcp,
            ssd_cache_config: Some(SsdCacheConfig {
                cache_paths: vec![root.join("data")],
                capacity_bytes: (bytes as u64 * 8).max(64 << 20),
                shards: NonZeroUsize::new(1).unwrap(),
                backend: SsdBackend::Uring,
                persistent,
                ..Default::default()
            }),
            ..Default::default()
        },
        &[],
    )
    .unwrap()
}

fn key(sequence: u64) -> StateKey {
    StateKey::new(
        format!("orbitkv:v2:{}", "ab".repeat(32)),
        orbitkv_state::group_hash(&sequence.to_le_bytes(), 0),
    )
}

struct Peer {
    requester: Arc<TransferEngine>,
    source: Arc<TransferEngine>,
    registration: crate::transfer_engine::Registration,
    _segments: Arc<crate::peer::segment::PeerSegments>,
    identity: Arc<PeerIdentity>,
    blocks: Vec<TransferBlockInfo>,
    allocate: AllocateFn,
}
impl Peer {
    fn new(storage: &Arc<Storage>, source: &Arc<SealedBlock>) -> Self {
        let make = || {
            Arc::new(
                TransferEngine::new(
                    128 << 20,
                    "127.0.0.1:0",
                    orbite::Transport::Tcp,
                    None,
                    #[cfg(feature = "orbite-rdma")]
                    None,
                )
                .unwrap(),
            )
        };
        let engine = make();
        let requester = make();
        let registration = engine
            .clone()
            .register_pinned(
                &source.allocations().cloned().collect::<Vec<_>>(),
                Permission::RemoteRead,
            )
            .unwrap();
        let owner = orbitkv_state::CacheOwner {
            endpoint: "source".into(),
            incarnation: uuid::Uuid::new_v4(),
        };
        let membership = Arc::new(orbitkv_catalog::MembershipView::new(owner.clone()));
        assert!(membership.renew(Instant::now(), Duration::from_secs(600)));
        membership.replace_members([("source".into(), 1, owner.clone())]);
        let segments =
            crate::peer::segment::PeerSegments::new(requester.clone(), membership.clone());
        let identity = segments
            .get(
                membership.lease(&owner).unwrap(),
                engine.transfer_endpoint(),
                engine.source_epoch(),
            )
            .unwrap();
        let raw = &source.slots()[0];
        let mut slot = TransferSlotInfo {
            k_ptr: raw.segment_ptr(0).unwrap().as_ptr() as u64,
            k_size: raw.memory_footprint(),
            encoding: raw
                .encoding()
                .map(|value| serde_json::to_vec(value).unwrap())
                .unwrap_or_default(),
            ..Default::default()
        };
        // BENCH_REFERENCE_V2: the runner removes only this fixture assignment on pre-v2 revisions.
        slot.checksums = serde_json::to_vec(raw.checksums()).unwrap();
        let storage = storage.clone();
        Self {
            requester,
            source: engine,
            registration,
            _segments: segments,
            identity,
            blocks: vec![TransferBlockInfo {
                block_hash: key(1).hash,
                slots: vec![slot],
            }],
            allocate: Arc::new(move |bytes, node| storage.allocate(NonZeroU64::new(bytes)?, node)),
        }
    }
    async fn read(&self) -> Result<MaterializedBlocks, String> {
        fetch_blocks_via_peer(
            &self.requester,
            &self.allocate,
            &key(1).namespace,
            self.identity.clone(),
            &self.blocks,
            &[NumaNode::UNKNOWN],
            crate::transfer_engine::tests::demand(),
            TransferLockGuard::test_detached(),
            ExecutionResource::Peer(0),
        )
        .await
        .map(|(blocks, _)| blocks)
    }
    fn close(mut self) {
        // SAFETY: every benchmark READ has drained and all returned buffers were dropped.
        unsafe { self.source.unregister(&mut self.registration) }.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "explicit CPU path benchmark; writes CSV to the selected external artifacts"]
async fn cpu_paths() {
    assert_eq!(std::env::var("ORBITE_REQUIRE_IO_URING").as_deref(), Ok("1"));
    let root = std::env::var_os("ORBITKV_SSD_TEST_ROOT").unwrap();
    let output = std::env::var_os("ORBITKV_PATH_BENCH_OUTPUT").unwrap();
    let mut csv = String::from("path,bytes,iterations,seconds,ns_per_operation\n");
    for bytes in [4096, 1 << 20, 16 << 20] {
        let directory = tempfile::tempdir_in(&root).unwrap();
        let storage = storage(directory.path(), bytes, false);
        let source = Arc::new(SealedBlock::from_slots(vec![(
            block(&storage, bytes),
            NumaNode::UNKNOWN,
        )]));
        let dram = DramStore::new(64 << 20, false, None, None, 0);
        let keys = [key(1)];
        dram.batch_insert(vec![(keys[0].clone(), source.clone())]);
        let peer = Peer::new(&storage, &source);
        let ssd = storage.ssd_store.as_ref().unwrap();
        ssd.ingest_batch(std::iter::once((&keys[0], &source)), false);
        ssd.flush().await;
        let lease = ssd
            .discover(&keys)
            .remove(0)
            .unwrap()
            .pin(crate::transfer_engine::tests::demand())
            .unwrap();
        for path in ["dram_hit", "ssd_read", "peer_tcp_read"] {
            for phase in 0..2 {
                let start = Instant::now();
                let mut iterations = 0;
                while iterations < 3
                    || start.elapsed() < Duration::from_millis(if phase == 0 { 50 } else { 600 })
                {
                    match path {
                        "dram_hit" => {
                            assert_eq!(black_box(dram.get_prefix_blocks(&keys, false)).0, 1);
                        }
                        "ssd_read" => {
                            assert_eq!(
                                black_box(ssd.read_host_batch(vec![lease.clone()]).await.unwrap())
                                    .len(),
                                1
                            );
                        }
                        _ => {
                            assert_eq!(black_box(peer.read().await.unwrap()).len(), 1);
                        }
                    }
                    iterations += 1;
                }
                if phase == 1 {
                    row(&mut csv, path, bytes, iterations, start.elapsed());
                }
            }
        }
        drop(lease);
        peer.close();
        for persistent in [false, true] {
            let directory = tempfile::tempdir_in(&root).unwrap();
            let store = self::storage(directory.path(), bytes, persistent);
            let ssd = store.ssd_store.as_ref().unwrap();
            // Source allocation/initialization is outside timing; each commit
            // still seals a fresh immutable object and includes source CRC work.
            let raw = block(&store, bytes);
            let allocation = raw.allocations().next().unwrap().clone();
            let mut sequence = 0;
            for phase in 0..2 {
                let start = Instant::now();
                let mut iterations = 0;
                while iterations < 3
                    || start.elapsed() < Duration::from_millis(if phase == 0 { 50 } else { 600 })
                {
                    sequence += 1;
                    let source = Arc::new(SealedBlock::from_slots(vec![(
                        RawBlock::single_segment(
                            Segment::new(allocation.as_non_null(), bytes, allocation.clone())
                                .unwrap(),
                        ),
                        NumaNode::UNKNOWN,
                    )]));
                    let key = key(sequence);
                    ssd.ingest_batch(std::iter::once((&key, &source)), false);
                    ssd.flush().await;
                    assert!(
                        ssd.discover(&[key])[0].is_some(),
                        "write must publish, not time a refusal"
                    );
                    iterations += 1;
                }
                if phase == 1 {
                    row(
                        &mut csv,
                        if persistent {
                            "ssd_write_persistent"
                        } else {
                            "ssd_write_ephemeral"
                        },
                        bytes,
                        iterations,
                        start.elapsed(),
                    );
                }
            }
        }
    }
    std::fs::write(output, csv).unwrap();
}
fn row(csv: &mut String, path: &str, bytes: usize, iterations: usize, elapsed: Duration) {
    let seconds = elapsed.as_secs_f64();
    use std::fmt::Write;
    writeln!(
        csv,
        "{path},{bytes},{iterations},{seconds:.9},{:.3}",
        seconds * 1e9 / iterations as f64
    )
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real OrbiTE TCP and file I/O with CPU host-registration fixture"]
async fn tier_crossings_scan_once_and_never_publish_corrupt_receives() {
    let Some(metrics) = crate::integrity::tests::isolate(
        "peer::read::tests::path_bench::tier_crossings_scan_once_and_never_publish_corrupt_receives",
    ) else {
        return;
    };
    let root = std::env::var_os("ORBITKV_SSD_TEST_ROOT").unwrap();
    for encoded in [false, true] {
        let directory = tempfile::tempdir_in(&root).unwrap();
        let storage = storage(directory.path(), 4096, false);
        let mut raw = block(&storage, 4096);
        if encoded {
            raw.encoding = Some(vec![crate::EncodedSegment {
                version: 1,
                format: orbitkv_state::StorageFormat::Fp8FromBf16,
                logical_bytes: 274,
                stored_bytes: 137,
                checksum: crc32fast::hash(&[0x35; 137]),
            }]);
        }
        let before = crate::integrity::tests::bytes(&metrics, "seal");
        let source = Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)]));
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "seal") - before,
            4096
        );
        let seal = crate::integrity::tests::bytes(&metrics, "seal");
        let dram = DramStore::new(64 << 20, false, None, None, 0);
        let key = key(1);
        dram.batch_insert(vec![(key.clone(), source.clone())]);
        for _ in 0..3 {
            assert_eq!(
                dram.get_prefix_blocks(std::slice::from_ref(&key), false).0,
                1
            );
        }
        assert_eq!(crate::integrity::tests::bytes(&metrics, "seal"), seal);
        assert_eq!(crate::integrity::tests::bytes(&metrics, "dram"), 0);
        let ssd = storage.ssd_store.as_ref().unwrap();
        let before = crate::integrity::tests::bytes(&metrics, "ssd");
        ssd.ingest_batch(std::iter::once((&key, &source)), false);
        ssd.flush().await;
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "ssd"),
            before,
            "writes must not read back"
        );
        assert_eq!(crate::integrity::tests::bytes(&metrics, "seal"), seal);
        let lease = ssd
            .discover(std::slice::from_ref(&key))
            .remove(0)
            .unwrap()
            .pin(crate::transfer_engine::tests::demand())
            .unwrap();
        assert_eq!(
            ssd.read_host_batch(vec![lease.clone()])
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "ssd") - before,
            4096
        );
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "seal"),
            seal,
            "fill must not seal twice"
        );
        let peer = Peer::new(&storage, &source);
        let before = crate::integrity::tests::bytes(&metrics, "peer");
        assert_eq!(peer.read().await.unwrap().len(), 1);
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "peer") - before,
            4096
        );
        assert_eq!(crate::integrity::tests::bytes(&metrics, "seal"), seal);
        // SAFETY: previous transfer drained; no native reads race this deliberate flip.
        unsafe { *source.slots()[0].segment_ptr(0).unwrap().as_ptr().add(17) ^= 1 };
        assert!(
            peer.read().await.is_err(),
            "corrupt raw/encoded bytes must never be returned as a hit"
        );
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "peer") - before,
            8192
        );
        assert_eq!(crate::integrity::tests::bytes(&metrics, "seal"), seal);
        peer.close();
    }
}
