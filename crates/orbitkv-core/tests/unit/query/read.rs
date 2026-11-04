use super::*;

fn key(n: u8) -> StateKey {
    StateKey::new("ns".to_string(), vec![n])
}

fn block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}

#[test]
fn ready_result_rebuilds_prefix_in_requested_key_order() {
    let local = block();
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b2 = block();
    let b3 = block();

    let result = build_ready_result(
        vec![Arc::clone(&local)],
        4,
        Some(AttributionSource::Ssd),
        &[k1.clone(), k2.clone(), k3.clone()],
        vec![
            (k2, RestoreSource::Memory(Arc::clone(&b2))),
            (k1, RestoreSource::Memory(Arc::clone(&b1))),
            (k3, RestoreSource::Memory(Arc::clone(&b3))),
        ],
    );

    assert_eq!(result.ready_blocks.len(), 4);
    assert!(
        matches!(&result.ready_blocks[0], RestoreSource::Memory(block) if Arc::ptr_eq(block, &local))
    );
    assert!(
        matches!(&result.ready_blocks[1], RestoreSource::Memory(block) if Arc::ptr_eq(block, &b1))
    );
    assert!(
        matches!(&result.ready_blocks[2], RestoreSource::Memory(block) if Arc::ptr_eq(block, &b2))
    );
    assert!(
        matches!(&result.ready_blocks[3], RestoreSource::Memory(block) if Arc::ptr_eq(block, &b3))
    );
    assert_eq!(result.missing, 0);
    assert_eq!(result.cache_inserts.len(), 3);
}

#[test]
fn ready_result_stops_at_first_missing_prefetch_key() {
    let k1 = key(1);
    let k2 = key(2);
    let k3 = key(3);
    let b1 = block();
    let b3 = block();

    let result = build_ready_result(
        Vec::new(),
        3,
        Some(AttributionSource::Ssd),
        &[k1.clone(), k2, k3.clone()],
        vec![
            (k3, RestoreSource::Memory(b3)),
            (k1, RestoreSource::Memory(Arc::clone(&b1))),
        ],
    );

    assert_eq!(result.ready_blocks.len(), 1);
    assert!(
        matches!(&result.ready_blocks[0], RestoreSource::Memory(block) if Arc::ptr_eq(block, &b1))
    );
    assert_eq!(result.missing, 2);
    assert_eq!(result.cache_inserts.len(), 2);
}

#[tokio::test]
async fn shared_reads_require_the_same_placement_and_ssd_prefetch_permission() {
    use crate::{SsdBackend, SsdCacheConfig, SsdReadPath};

    for (read_path, mode, existing_permission, existing_node, existing_host, share) in [
        (None, QueryMode::Demand, true, NumaNode(1), false, true),
        (None, QueryMode::Demand, true, NumaNode(1), true, false),
        (None, QueryMode::Prepare, true, NumaNode(1), true, true),
        (None, QueryMode::Prepare, true, NumaNode(1), false, false),
        (
            Some(SsdReadPath::Cufile),
            QueryMode::Demand,
            true,
            NumaNode(1),
            false,
            false,
        ),
        (
            Some(SsdReadPath::Cufile),
            QueryMode::Prepare,
            false,
            NumaNode(1),
            true,
            false,
        ),
        (None, QueryMode::Prepare, true, NumaNode(0), true, false),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let store = crate::storage::ssd::SsdStore::new(
            SsdCacheConfig {
                cache_paths: vec![directory.path().join("cache.bin")],
                capacity_bytes: 4096,
                backend: SsdBackend::Uring,
                read_path,
                ..Default::default()
            },
            Arc::new(|_, _| None),
            Arc::new(|bytes, _| Some(bytes)),
            false,
            None,
            None,
        )
        .unwrap();
        let cache = Arc::new(DramStore::new(4096, false, None, None, 0));
        let scheduler = ReadCoordinator::new(
            cache.clone(),
            Some(store),
            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            None,
            0,
        );
        let shared = Arc::new(SharedRead::new());
        scheduler.reads.lock().insert(
            ReadKey {
                keys: vec![key(1)],
                hit: 0,
                wait_for_full_prefix: false,
                allow_ssd_prefetch: existing_permission,
                destination_nodes: vec![existing_node],
                host_ready: existing_host,
            },
            Arc::downgrade(&shared),
        );
        let (release, hold) = tokio::sync::oneshot::channel();
        let mut initializing = Box::pin(shared.get_or_init(|| async {
            hold.await.unwrap();
            MaterializedRead {
                source: Some(AttributionSource::Ssd),
                cache_inserts: Vec::new(),
                ready_blocks: vec![RestoreSource::Memory(block())],
                missing: 0,
            }
        }));
        assert!(futures::poll!(initializing.as_mut()).is_pending());
        let hashes = [vec![1]];
        let mut read =
            Box::pin(scheduler.read_prefix("query", "ns", &hashes, mode, &[NumaNode(1)]));
        if share {
            assert!(futures::poll!(read.as_mut()).is_pending());
            release.send(()).unwrap();
            initializing.await;
            let result = read.await;
            assert_eq!(result.blocks.len(), 1);
            assert_eq!(result.missing, 0);
        } else {
            let result = match futures::poll!(read.as_mut()) {
                std::task::Poll::Ready(result) => result,
                std::task::Poll::Pending => {
                    panic!("different placement or SSD permission shared a pending read: {mode:?}")
                }
            };
            assert!(result.blocks.is_empty());
            assert_eq!(result.missing, 1);
        }
    }
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[tokio::test]
async fn peer_ready_prefix_never_becomes_a_dram_cache_insert() {
    use crate::peer::restore::tests::{Control, source};
    let result = build_ready_result(
        vec![block()],
        2,
        Some(AttributionSource::Remote),
        &[key(1)],
        vec![(
            key(1),
            RestoreSource::Peer(source(Arc::new(Control::default()), vec![])),
        )],
    );
    assert_eq!(result.ready_blocks.len(), 2);
    assert!(matches!(result.ready_blocks[1], RestoreSource::Peer(_)));
    assert!(result.cache_inserts.is_empty());
}
