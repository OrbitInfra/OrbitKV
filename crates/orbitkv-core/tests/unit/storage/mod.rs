use super::*;

#[cfg(not(feature = "orbite-rdma"))]
#[test]
fn unavailable_rdma_backend_is_rejected_before_pinned_allocation() {
    let config = EngineConfig {
        transfer_transport: crate::TransportHint::Rdma,
        ..EngineConfig::default()
    };
    let Err(error) = Storage::new_with_config(4096, false, config, &[]) else {
        panic!("an unavailable RDMA backend must be rejected");
    };
    assert!(error.contains("orbite-rdma build feature"), "{error}");
    let configured = EngineConfig {
        orbite_rdma: Some(crate::OrbiteRdmaConfig::new("not-loaded")),
        ..EngineConfig::default()
    };
    assert_eq!(
        configured.validate_backend().unwrap_err().kind,
        crate::transfer_engine::ErrorKind::Unavailable
    );
}

fn make_engine() -> Arc<Storage> {
    Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap()
}

#[test]
fn invalid_memory_watermarks_fail_before_pinned_allocation() {
    for (low, high) in [
        (Some(50), None),
        (None, Some(80)),
        (Some(0), Some(0)),
        (Some(80), Some(80)),
        (Some(90), Some(80)),
        (Some(100), Some(101)),
        (Some(254), Some(255)),
    ] {
        let result = Storage::new_with_config(
            4096,
            false,
            EngineConfig {
                cache_low_watermark_percent: low,
                cache_high_watermark_percent: high,
                ..EngineConfig::default()
            },
            &[],
        );
        let Err(error) = result else {
            panic!("invalid watermarks accepted: {low:?}, {high:?}");
        };
        assert!(error.contains("0 <= low < high <= 100"), "{error}");
    }
}

fn insert_allocated_resident(storage: &Storage, id: u32, bytes: usize) -> StateKey {
    use crate::block::{RawBlock, Segment};

    let allocation = storage
        .allocate(NonZeroU64::new(bytes as u64).unwrap(), None)
        .unwrap();
    let slot = RawBlock::single_segment(
        Segment::new(allocation.as_non_null(), bytes, allocation).unwrap(),
    );
    let key = StateKey::new("watermarks".into(), id.to_le_bytes().to_vec());
    storage.dram.batch_insert(vec![(
        key.clone(),
        Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)])),
    )]);
    key
}

#[tokio::test]
async fn memory_watermarks_preserve_held_residents_and_finish_below_high() {
    const PAGE: usize = 4096;
    for enabled in [false, true] {
        let storage = Storage::new_with_config(
            16 * PAGE,
            false,
            EngineConfig {
                cache_low_watermark_percent: enabled.then_some(50),
                cache_high_watermark_percent: enabled.then_some(75),
                hint_value_size_bytes: Some(PAGE),
                ..EngineConfig::default()
            },
            &[],
        )
        .unwrap();
        let keys: Vec<_> = (0..12)
            .map(|id| insert_allocated_resident(&storage, id, PAGE))
            .collect();
        let held = storage.dram.get_blocks_aligned(&keys[2..]);
        assert_eq!(storage.allocator.usage().0, (12 * PAGE) as u64);
        let first = storage.maintain_memory_cache();
        if !enabled {
            assert_eq!(first, MemoryCacheCleanupStats::default());
            assert!(
                storage
                    .dram
                    .contains_keys(&keys)
                    .into_iter()
                    .all(|present| present)
            );
            continue;
        }
        assert_eq!(first.evicted_blocks, 2);
        assert_eq!(first.reclaimed_bytes, (2 * PAGE) as u64);
        assert_eq!(storage.allocator.usage().0, (10 * PAGE) as u64);
        assert_eq!(
            storage.maintain_memory_cache(),
            MemoryCacheCleanupStats::default()
        );
        assert!(
            storage
                .dram
                .contains_keys(&keys[2..])
                .into_iter()
                .all(|present| present)
        );
        drop(held);
        let second = storage.maintain_memory_cache();
        assert_eq!(
            second.evicted_blocks, 2,
            "continue below high until the low target"
        );
        assert_eq!(storage.allocator.usage().0, (8 * PAGE) as u64);
        for id in 12..15 {
            insert_allocated_resident(&storage, id, PAGE);
        }
        assert_eq!(storage.allocator.usage().0, (11 * PAGE) as u64);
        assert_eq!(
            storage.maintain_memory_cache(),
            MemoryCacheCleanupStats::default()
        );
        insert_allocated_resident(&storage, 15, PAGE);
        assert_eq!(storage.maintain_memory_cache().evicted_blocks, 4);
        assert_eq!(storage.allocator.usage().0, (8 * PAGE) as u64);
    }
}

#[tokio::test]
async fn memory_watermarks_count_allocations_retained_outside_residency() {
    use crate::block::{RawBlock, Segment};

    const PAGE: usize = 4096;
    let storage = Storage::new_with_config(
        4 * PAGE,
        false,
        EngineConfig {
            cache_low_watermark_percent: Some(25),
            cache_high_watermark_percent: Some(50),
            hint_value_size_bytes: Some(PAGE),
            ..EngineConfig::default()
        },
        &[],
    )
    .unwrap();
    let held = storage
        .allocate(NonZeroU64::new((2 * PAGE) as u64).unwrap(), None)
        .unwrap();
    let slot =
        RawBlock::single_segment(Segment::new(held.as_non_null(), 2 * PAGE, held.clone()).unwrap());
    storage.dram.batch_insert(vec![(
        StateKey::new("watermarks".into(), vec![0]),
        Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)])),
    )]);
    let other = insert_allocated_resident(&storage, 1, PAGE);
    let first = storage.maintain_memory_cache();
    assert_eq!(first.evicted_blocks, 1);
    assert_eq!(first.evicted_bytes, (2 * PAGE) as u64);
    assert_eq!(
        first.reclaimed_bytes, 0,
        "removing a cache entry is not physical release"
    );
    assert_eq!(storage.allocator.usage().0, (3 * PAGE) as u64);
    assert_eq!(storage.dram.contains_keys(&[other]), vec![true]);
    assert_eq!(storage.maintain_memory_cache().evicted_blocks, 1);
    assert_eq!(storage.allocator.usage().0, (2 * PAGE) as u64);
    assert_eq!(
        storage.maintain_memory_cache(),
        MemoryCacheCleanupStats::default()
    );
    drop(held);
    assert_eq!(storage.allocator.usage().0, 0);
    assert_eq!(
        storage.maintain_memory_cache(),
        MemoryCacheCleanupStats::default()
    );
    insert_allocated_resident(&storage, 2, PAGE);
    assert_eq!(
        storage.maintain_memory_cache(),
        MemoryCacheCleanupStats::default()
    );
}

#[tokio::test]
async fn memory_watermarks_bound_each_pass_and_resume_the_cycle() {
    const PAGE: usize = 4096;
    let storage = Storage::new_with_config(
        1024 * PAGE,
        false,
        EngineConfig {
            cache_low_watermark_percent: Some(0),
            cache_high_watermark_percent: Some(50),
            hint_value_size_bytes: Some(PAGE),
            ..EngineConfig::default()
        },
        &[],
    )
    .unwrap();
    for id in 0..768 {
        insert_allocated_resident(&storage, id, PAGE);
    }
    assert_eq!(storage.maintain_memory_cache().evicted_blocks, 512);
    assert_eq!(storage.allocator.usage().0, (256 * PAGE) as u64);
    assert_eq!(storage.maintain_memory_cache().evicted_blocks, 256);
    assert_eq!(storage.allocator.usage().0, 0);
}

#[tokio::test]
async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
    let storage = make_engine();
    let mut hashes: HashSet<Vec<u8>> = HashSet::new();

    storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
    assert!(hashes.is_empty());
}

#[tokio::test]
async fn cleanup_memory_cache_evicts_all_resident_blocks() {
    let storage = make_engine();
    let key1 = StateKey::new("ns".into(), vec![1]);
    let key2 = StateKey::new("ns".into(), vec![2]);
    let block1 = Arc::new(SealedBlock::from_slots(Vec::new()));
    let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));

    storage.dram.batch_insert(vec![(key1, Arc::clone(&block1))]);
    storage.dram.batch_insert(vec![(key2, block2)]);

    let stats = storage.cleanup_memory_cache();
    assert_eq!(stats.evicted_blocks, 2);
    assert_eq!(stats.still_referenced_blocks, 1);
    assert_eq!(stats.reclaimed_bytes, 0);

    let stats = storage.cleanup_memory_cache();
    assert_eq!(stats, MemoryCacheCleanupStats::default());
}

#[tokio::test]
async fn allocate_bounded_reclaim_terminates() {
    // With a tiny pool, allocation of a huge block should fail fast
    // (not loop forever) thanks to MAX_RECLAIM_ROUNDS.
    let storage = Storage::new_with_config(4096, false, EngineConfig::default(), &[]).unwrap();

    // Try to allocate more than the entire pool
    let result = storage.allocate(NonZeroU64::new(1 << 30).unwrap(), None);
    assert!(result.is_none(), "should fail, not loop forever");
}

#[tokio::test]
async fn reclaim_preserves_residents_and_rechecks_real_contiguous_capacity() {
    use crate::block::{RawBlock, Segment};

    const PAGE: usize = 64 * 1024;
    for fragmented in [false, true] {
        let storage = Storage::new_with_config(
            16 * PAGE,
            false,
            EngineConfig {
                hint_value_size_bytes: Some(PAGE),
                ..EngineConfig::default()
            },
            &[],
        )
        .unwrap();
        let mut blocks = Vec::new();
        for i in 0..16 {
            let allocation = storage
                .allocate(NonZeroU64::new(PAGE as u64).unwrap(), None)
                .unwrap();
            let slot = RawBlock::single_segment(
                Segment::new(allocation.as_non_null(), PAGE, allocation).unwrap(),
            );
            blocks.push((
                StateKey::new("ns".into(), vec![i]),
                Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)])),
            ));
        }
        let mut keys = blocks
            .iter()
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if fragmented {
            // First victims leave separate holes; footprint alone is not
            // proof that the allocator can satisfy a contiguous request.
            blocks.sort_by_key(|(key, _)| match key.hash[0] {
                0 => 0,
                2 => 1,
                4 => 2,
                6 => 3,
                1 => 4,
                3 => 5,
                other => other as usize + 6,
            });
            storage.dram.batch_insert(blocks);
        } else {
            let warm = blocks.pop().unwrap();
            warm.1.mark_warmed();
            storage.dram.batch_insert(blocks);
            storage.dram.batch_insert_reclaimable(vec![warm]);
            keys.rotate_right(1);
        }
        assert_eq!(storage.allocator.usage().0, (16 * PAGE) as u64);
        let required = if fragmented { 2 * PAGE } else { PAGE };
        let allocated = storage.allocate(NonZeroU64::new(required as u64).unwrap(), None);
        assert!(allocated.is_some(), "fragmented={fragmented}");
        let remaining = storage.dram.contains_keys(&keys);
        if fragmented {
            assert!(remaining.iter().filter(|present| **present).count() >= 8);
        } else {
            assert!(!remaining[0], "reclaimable warmup goes first");
            assert!(
                remaining[1..].iter().all(|present| *present),
                "one-page demand must not evict the retained cache"
            );
        }
    }
}

#[tokio::test]
async fn gc_stale_inflight_returns_zero_when_empty() {
    let storage = make_engine();
    let cleaned = storage
        .writes
        .gc_stale_inflight(std::time::Duration::from_secs(60))
        .await;
    assert_eq!(cleaned, 0);
}

#[tokio::test]
async fn filter_hashes_not_in_cache_removes_cached() {
    let storage = make_engine();
    let key1 = StateKey::new("ns".into(), vec![1]);
    let key2 = StateKey::new("ns".into(), vec![2]);
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));

    storage.dram.batch_insert(vec![(key1, block.clone())]);
    storage.dram.batch_insert(vec![(key2, block)]);

    let mut hashes: HashSet<Vec<u8>> = [vec![1], vec![2], vec![3]].into_iter().collect();

    storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);

    assert_eq!(hashes.len(), 1);
    assert!(hashes.contains(&vec![3]));
}

#[test]
fn invalid_ssd_setup_returns_an_error_without_panicking() {
    let result = Storage::new_with_config(
        1 << 20,
        false,
        EngineConfig {
            ssd_cache_config: Some(crate::SsdCacheConfig {
                cache_paths: vec![],
                backend: crate::SsdBackend::Uring,
                ..Default::default()
            }),
            ..Default::default()
        },
        &[],
    );
    assert!(
        result.is_err(),
        "SSD initialization errors must reach the caller"
    );
}

#[tokio::test]
async fn local_storage_owns_orbite_without_enabling_distributed_admission() {
    let storage = make_engine();
    assert!(!storage.transfer_endpoint().is_empty());
    assert!(!storage.transfer_source_epoch().is_empty());
    assert_eq!(
        storage.peer_transport().transfer_endpoint(),
        storage.transfer_endpoint()
    );
    assert_eq!(
        storage
            .exports
            .open(uuid::Uuid::new_v4(), uuid::Uuid::new_v4()),
        Err(crate::PeerError::Unavailable)
    );
    for hint in [crate::TransportHint::Auto, crate::TransportHint::Tcp] {
        let error = crate::transfer_engine::create_transfer_engine(
            hint,
            storage.allocator.clone(),
            "not-an-endpoint",
            None,
            None,
        )
        .err()
        .expect("native initialization failure must reach the caller");
        assert_eq!(error.kind, crate::transfer_engine::ErrorKind::Backend);
    }
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    for library in ["libtent", "libmooncake", "libibverbs", "libcufile"] {
        assert!(
            !maps.contains(library),
            "unexpected native dependency: {library}"
        );
    }
}

#[test]
fn invalid_data_ttl_is_rejected_before_allocating_pinned_memory() {
    for ttl in [std::time::Duration::ZERO, std::time::Duration::MAX] {
        let result = Storage::new_with_config(
            4096,
            false,
            EngineConfig {
                cache_ttl: Some(ttl),
                ..EngineConfig::default()
            },
            &[],
        );
        let Err(error) = result else {
            panic!("invalid TTL accepted")
        };
        assert!(error.contains("cache TTL"), "{error}");
    }
}

#[tokio::test]
async fn ttl_sweep_keeps_physical_allocation_charged_until_the_reader_releases_it() {
    let storage = make_engine(); // No local TTL: imported deadlines still apply.
    let key = insert_allocated_resident(&storage, 1, 4096);
    let held = storage
        .dram
        .get_prefix_blocks(std::slice::from_ref(&key), false)
        .1;
    storage.dram.remove_all();
    let expiry = expiry::Expiration::new(Some(std::time::Duration::from_secs(60)))
        .unwrap()
        .unwrap();
    held[0].set_expiration(Some(expiry.clone())).unwrap();
    storage
        .dram
        .batch_insert(vec![(key.clone(), held[0].clone())]);
    let used = storage.allocator.usage().0;
    expiry.expire_for_test();
    assert_eq!(storage.expire_cache().dram_blocks, 1);
    assert!(!storage.dram.contains_keys(&[key])[0]);
    assert!(held[0].is_verified());
    assert_eq!(storage.allocator.usage().0, used);
    drop(held);
    assert_eq!(storage.allocator.usage().0, 0);
}
