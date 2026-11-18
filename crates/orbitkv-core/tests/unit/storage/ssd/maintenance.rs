use super::*;
use crate::SlotMeta;
use crate::block::StateKey;
use crate::storage::ssd::{SsdCacheConfig, index, tests::queued_read_store};
use parking_lot::Mutex;
use std::sync::Arc;

fn key(id: usize) -> StateKey {
    StateKey::new(
        format!("orbitkv:v2:{}", "ab".repeat(32)),
        orbitkv_state::group_hash(&id.to_le_bytes(), 0),
    )
}

fn store(capacity: u64, shards: usize) -> Arc<SsdStore> {
    let (mut store, _queue) = queued_read_store();
    let owner = Arc::get_mut(&mut store).unwrap();
    owner.watermarks = Some((25, 75));
    owner.inner.get_mut().ring = index::SsdRingBuffer::new_sharded(vec![capacity; shards], 512);
    owner._files = (0..shards)
        .map(|_| {
            let file = tempfile::tempfile().unwrap();
            file.set_len(capacity).unwrap();
            Arc::new(file)
        })
        .collect();
    owner.inventory = Some(Arc::new(crate::ResidencyInventory::new(64 * 1024)));
    store
}

fn insert(store: &SsdStore, id: usize, commit: bool) -> index::SsdIndexEntry {
    let entry = store
        .inner
        .lock()
        .ring
        .reserve(
            &key(id),
            vec![SlotMeta::new(
                smallvec::smallvec![512],
                crate::NumaNode::UNKNOWN,
            )],
            index::Encoding::Raw,
            None,
        )
        .unwrap();
    entry
        .checksums
        .set(vec![crate::integrity::Checksum { end: 512, crc: 42 }])
        .unwrap();
    if let Some(journal) = &store.journal {
        let mut journal = journal.lock();
        journal.append(&key(id), &entry, None).unwrap();
        if commit {
            journal.append(&key(id), &entry, Some(42)).unwrap();
        }
    }
    if commit {
        store.commit_write(&key(id), true, Some(&entry));
    }
    entry
}

#[test]
fn watermarks_are_optional_paired_and_ordered_before_native_setup() {
    for (low, high, valid) in [
        (None, None, true),
        (Some(0), Some(100), true),
        (Some(25), Some(75), true),
        (None, Some(75), false),
        (Some(25), None, false),
        (Some(75), Some(75), false),
        (Some(90), Some(75), false),
        (Some(0), Some(101), false),
    ] {
        let config = SsdCacheConfig {
            low_watermark_percent: low,
            high_watermark_percent: high,
            ..Default::default()
        };
        assert_eq!(
            config.validate_transfer_engine().is_ok(),
            valid,
            "{low:?}/{high:?}"
        );
    }
}

#[test]
fn watermark_pass_is_bounded_and_keeps_hysteresis_until_low() {
    let mut store = store(600 * 512, 1);
    Arc::get_mut(&mut store).unwrap().watermarks = None;
    for id in 0..600 {
        insert(&store, id, true);
    }
    assert_eq!(
        store.maintain_cache().unwrap(),
        SsdCacheCleanupStats::default()
    );
    Arc::get_mut(&mut store).unwrap().watermarks = Some((0, 100));
    {
        let _busy = store.watermark_sweep.lock();
        assert_eq!(
            store.maintain_cache().unwrap(),
            SsdCacheCleanupStats::default()
        );
    }
    let first = store.maintain_cache().unwrap();
    assert_eq!((first.scanned_entries, first.evicted_blocks), (512, 512));
    assert_eq!(first.reclaimed_bytes, 512 * 512);
    assert!(
        store
            .discover(&[key(0), key(511)])
            .iter()
            .all(Option::is_none)
    );
    assert!(store.discover(&[key(512)])[0].is_some());
    let second = store.maintain_cache().unwrap();
    assert_eq!(second.evicted_blocks, 88, "continue below high");
    assert_eq!(second.reclaimed_bytes, 88 * 512);
    insert(&store, 601, true);
    assert_eq!(
        store.maintain_cache().unwrap().evicted_blocks,
        0,
        "latch cleared at low"
    );
    assert!(store.discover(&[key(601)])[0].is_some());
}

#[test]
fn reaching_low_at_the_scan_limit_ends_the_cycle_before_new_writes() {
    let mut store = store(512 * 512, 1);
    Arc::get_mut(&mut store).unwrap().watermarks = Some((0, 100));
    for id in 0..512 {
        insert(&store, id, true);
    }
    let stats = store.maintain_cache().unwrap();
    assert_eq!((stats.scanned_entries, stats.evicted_blocks), (512, 512));
    insert(&store, 512, true);
    assert_eq!(store.maintain_cache().unwrap().evicted_blocks, 0);
    assert!(store.discover(&[key(512)])[0].is_some());
}

#[test]
fn held_and_writing_fronts_block_only_their_shard_until_drain() {
    for barrier in ["writing", "reader", "invalid_reader"] {
        let store = store(2048, 2);
        let first = insert(&store, 0, barrier != "writing");
        for id in 1..6 {
            insert(&store, id, true);
        }
        let held = (barrier != "writing").then(|| {
            store.discover(&[key(0)])[0]
                .as_ref()
                .unwrap()
                .pin(crate::transfer_engine::tests::demand())
                .unwrap()
        });
        if barrier == "invalid_reader" {
            store.invalidate_entry(&key(0), &first);
        }
        let stats = store.maintain_cache().unwrap();
        assert_eq!(
            (
                stats.blocked_shards,
                stats.evicted_blocks,
                stats.reclaimed_bytes
            ),
            (1, 2, 1024),
            "{barrier}"
        );
        assert!(
            store
                .discover(&[key(2), key(4), key(5)])
                .iter()
                .all(Option::is_some)
        );
        assert!(
            store
                .discover(&[key(1), key(3)])
                .iter()
                .all(Option::is_none)
        );
        if barrier == "writing" {
            store.commit_write(&key(0), true, Some(&first));
        }
        drop(held);
        let stats = store.maintain_cache().unwrap();
        assert_eq!(stats.blocked_shards, 0);
        assert_eq!(stats.reclaimed_bytes, 1024);
        assert_eq!(
            stats.evicted_blocks,
            if barrier == "invalid_reader" { 1 } else { 2 }
        );
        assert!(
            store
                .discover(&[key(0), key(2)])
                .iter()
                .all(Option::is_none)
        );
        assert!(
            store
                .discover(&[key(4), key(5)])
                .iter()
                .all(Option::is_some)
        );
    }
}

#[test]
fn persistent_watermarks_remove_inventory_and_do_not_resurrect_after_reopen() {
    for fenced in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("index");
        let mut store = store(2048, 1);
        let (journal, recovered) =
            persistence::Journal::open(path.clone(), &store._files, 2048, 512, None).unwrap();
        assert!(recovered.is_empty());
        Arc::get_mut(&mut store).unwrap().journal = Some(Mutex::new(journal));
        for id in 0..3 {
            insert(&store, id, true);
        }
        if fenced {
            store.journal.as_ref().unwrap().lock().fence();
            assert!(store.maintain_cache().is_err());
            assert!(
                store
                    .discover(&[key(0), key(1), key(2)])
                    .iter()
                    .all(Option::is_some)
            );
        } else {
            let stats = store.maintain_cache().unwrap();
            assert_eq!((stats.evicted_blocks, stats.reclaimed_bytes), (2, 1024));
            let records = store.inventory.as_ref().unwrap().page(None).unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].key, key(2));
        }
        let files = store._files.clone();
        drop(store);
        let (_, recovered) = persistence::Journal::open(path, &files, 2048, 512, None).unwrap();
        assert_eq!(recovered.len(), if fenced { 3 } else { 1 });
        assert!(recovered.iter().any(|entry| entry.key == key(2)));
    }
}
