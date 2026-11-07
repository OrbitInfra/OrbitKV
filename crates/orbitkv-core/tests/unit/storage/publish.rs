use super::*;
use std::num::NonZeroU64;

use crate::EngineConfig;
use crate::block::RawBlock;
use crate::storage::Storage;

fn make_raw_block(engine: &Storage, size: u64) -> RawBlock {
    use crate::block::Segment;
    let alloc = engine
        .allocate(NonZeroU64::new(size).unwrap(), None)
        .expect("test pool should have space");
    let ptr = alloc.as_non_null();
    RawBlock::new(vec![Segment::new(ptr, size as usize, alloc).unwrap()])
}

fn make_worker(engine: &Storage) -> PublishWorker {
    PublishWorker::new(engine.dram.clone(), engine.ssd_store.clone())
}

#[tokio::test]
async fn single_slot_seals_immediately() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![1, 2, 3]);
    let block = make_raw_block(&engine, 64);

    let entries: InsertEntries = vec![(key.clone(), vec![(0, block)])];

    worker.insert(entries, 1, NumaNode::UNKNOWN, "ns", None);

    assert!(worker.inflight.is_empty(), "block should have been sealed");
    assert!(
        engine.dram.contains_keys(std::slice::from_ref(&key))[0],
        "sealed block should be in cache"
    );
}

#[tokio::test]
async fn ordered_multi_slot_batch_seals_immediately() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![4, 5, 6]);
    let block0 = make_raw_block(&engine, 64);
    let block1 = make_raw_block(&engine, 96);
    let block2 = make_raw_block(&engine, 128);
    let expected_footprint =
        block0.memory_footprint() + block1.memory_footprint() + block2.memory_footprint();

    let entries: InsertEntries = vec![(key.clone(), vec![(0, block0), (1, block1), (2, block2)])];

    let ordered_fast_path_seals = worker.insert(entries, 3, NumaNode(1), "ns", None);

    assert_eq!(ordered_fast_path_seals, 1);
    assert!(
        worker.inflight.is_empty(),
        "ordered complete batch should skip inflight storage"
    );
    let cached = engine.dram.get_blocks_aligned(std::slice::from_ref(&key));
    assert_eq!(cached.len(), 1, "sealed block should be in read cache");

    let sealed = cached[0].as_ref().expect("sealed block should be present");
    assert_eq!(sealed.memory_footprint(), expected_footprint);
    assert_eq!(sealed.slots().len(), 3);
    assert_eq!(sealed.slot_numas(), &[NumaNode(1); 3]);
}

#[tokio::test]
async fn multi_slot_partial_then_complete() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![1]);

    let block0 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block0)])];
    worker.insert(entries, 3, NumaNode::UNKNOWN, "ns", None);
    assert_eq!(worker.inflight.len(), 1, "block should still be inflight");
    assert!(!engine.dram.contains_keys(std::slice::from_ref(&key))[0]);

    let block1 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(1, block1)])];
    worker.insert(entries, 3, NumaNode::UNKNOWN, "ns", None);
    assert_eq!(worker.inflight.len(), 1, "still inflight after 2/3 slots");

    let block2 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(2, block2)])];
    worker.insert(entries, 3, NumaNode::UNKNOWN, "ns", None);
    assert!(
        worker.inflight.is_empty(),
        "block should be sealed after 3/3 slots"
    );
    assert!(engine.dram.contains_keys(std::slice::from_ref(&key))[0]);
}

#[tokio::test]
async fn duplicate_slot_is_idempotent() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![1]);

    let block_a = make_raw_block(&engine, 64);
    let block_b = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block_a), (0, block_b)])];

    worker.insert(entries, 2, NumaNode::UNKNOWN, "ns", None);

    assert_eq!(worker.inflight.len(), 1);
    let inflight_block = worker.inflight.get(&key).unwrap();
    assert_eq!(inflight_block.filled_count(), 1);
}

#[tokio::test]
async fn slot_count_mismatch_skips_key() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![1]);

    let block0 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block0)])];
    worker.insert(entries, 2, NumaNode::UNKNOWN, "ns", None);
    assert_eq!(worker.inflight.len(), 1);

    let block1 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(1, block1)])];
    worker.insert(
        entries,
        4, // mismatch
        NumaNode::UNKNOWN,
        "ns",
        None,
    );

    let inflight_block = worker.inflight.get(&key).unwrap();
    assert_eq!(inflight_block.filled_count(), 1);
}

#[tokio::test]
async fn late_save_for_resident_block_is_dropped() {
    // A block saved more times than `total_slots` (e.g. concurrent re-saves
    // of a shared prefix): the first full set seals it; a late duplicate
    // batch must not recreate a partial InflightBlock that never seals.
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![9]);

    // Seal the block: both slots arrive, block moves to the read cache.
    let block0 = make_raw_block(&engine, 64);
    let block1 = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block0), (1, block1)])];
    worker.insert(entries, 2, NumaNode::UNKNOWN, "ns", None);
    assert!(
        worker.inflight.is_empty(),
        "block should be sealed into read cache"
    );
    assert!(engine.dram.contains_keys(std::slice::from_ref(&key))[0]);

    // Late duplicate save of the already-resident block: a single column
    // for the same key arrives after the seal. It must be dropped, not
    // turned into a permanently-incomplete inflight block.
    let late = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, late)])];
    worker.insert(entries, 2, NumaNode::UNKNOWN, "ns", None);

    assert!(
        worker.inflight.is_empty(),
        "late save for an already-resident block must not leak a partial inflight block"
    );
}

#[tokio::test]
async fn gc_inflight_removes_old_blocks() {
    let key = StateKey::new("ns".into(), vec![1]);
    let mut inflight: HashMap<StateKey, InflightBlock> = HashMap::new();
    inflight.insert(key, InflightBlock::new(2));

    let cleaned = gc_inflight(&mut inflight, std::time::Duration::from_secs(60));
    assert_eq!(cleaned, 0);
    assert_eq!(inflight.len(), 1);

    let cleaned = gc_inflight(&mut inflight, std::time::Duration::ZERO);
    assert_eq!(cleaned, 1);
    assert!(inflight.is_empty());
}

#[tokio::test]
async fn sealed_blocks_are_resident_without_backing_stores() {
    let engine = Storage::new_with_config(1 << 20, false, EngineConfig::default(), &[]).unwrap();
    let mut worker = make_worker(&engine);

    let key = StateKey::new("ns".into(), vec![7]);
    let block = make_raw_block(&engine, 64);
    let entries: InsertEntries = vec![(key.clone(), vec![(0, block)])];

    worker.insert(entries, 1, NumaNode::UNKNOWN, "ns", None);

    assert!(worker.inflight.is_empty());
    assert!(engine.dram.contains_keys(std::slice::from_ref(&key))[0]);
}
