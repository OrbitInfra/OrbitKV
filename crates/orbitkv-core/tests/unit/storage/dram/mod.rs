use std::time::Duration;

use super::*;

fn make_cache() -> DramStore {
    DramStore::new(1 << 20, false, None, None, 0)
}

fn make_block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}

#[test]
fn discovery_preserves_image_identity_without_holding_or_promoting_residency() {
    let cache = DramStore::new(100, false, None, None, 60);
    let key = StateKey::new("ns".into(), vec![1]);
    let block = Arc::new(SealedBlock::for_policy_test(40));
    cache.batch_insert(vec![(key.clone(), Arc::clone(&block))]);
    let candidate = cache
        .discover(std::slice::from_ref(&key))
        .pop()
        .unwrap()
        .unwrap();
    assert!(Weak::ptr_eq(&candidate, &Arc::downgrade(&block)));
    assert_eq!(candidate.upgrade().unwrap().memory_footprint(), 40);
    assert_eq!(Arc::strong_count(&block), 2);
    assert_class(&cache, &key, ResidentClass::Probationary);
    drop(block);
    drop(cache.remove_all());
    assert!(candidate.upgrade().is_none());
}

#[test]
fn demand_protection_survives_scans_and_demotes_by_bytes() {
    let cache = DramStore::new(100, false, None, None, 60);
    let a = StateKey::new("ns".into(), vec![1]);
    let b = StateKey::new("ns".into(), vec![2]);
    let c = StateKey::new("ns".into(), vec![3]);
    for key in [&a, &b] {
        cache.batch_insert(vec![(
            key.clone(),
            Arc::new(SealedBlock::for_policy_test(40)),
        )]);
    }
    drop(cache.get_prefix_blocks(std::slice::from_ref(&a), false));
    assert_class(&cache, &a, ResidentClass::Retained);
    // A speculative lookup and duplicate publication cannot promote b.
    drop(cache.get_prefix_blocks(std::slice::from_ref(&b), true));
    cache.batch_insert(vec![(
        b.clone(),
        Arc::new(SealedBlock::for_policy_test(40)),
    )]);
    assert_class(&cache, &b, ResidentClass::Probationary);
    assert_eq!(
        cache.remove_lru_batch(1, 40, ReclaimReason::Pressure)[0].0,
        b
    );
    cache.batch_insert(vec![(
        c.clone(),
        Arc::new(SealedBlock::for_policy_test(30)),
    )]);
    drop(cache.get_prefix_blocks(std::slice::from_ref(&c), false));
    assert_class(&cache, &a, ResidentClass::Probationary);
    assert_class(&cache, &c, ResidentClass::Retained);
    assert_eq!(cache.inner.lock().retained_bytes, 30);
    // A live consumer still prevents reclaim after policy demotion.
    let held = cache.get_prefix_blocks(std::slice::from_ref(&a), true).1;
    assert_eq!(
        cache.remove_lru_batch(1, 40, ReclaimReason::Pressure)[0].0,
        c
    );
    assert_eq!(cache.inner.lock().retained_bytes, 0);
    drop(held);
    assert_eq!(
        cache.remove_lru_batch(1, 40, ReclaimReason::Pressure)[0].0,
        a
    );
}

#[test]
fn protection_revalidates_generations_and_bounds_retained_bytes() {
    let cache = DramStore::new(100, false, None, Some(16 * 1024), 60);
    let key = StateKey::new("ns".into(), vec![1]);
    let old = Arc::new(SealedBlock::for_policy_test(40));
    cache.batch_insert(vec![(key.clone(), Arc::clone(&old))]);
    cache.remove_all();
    let current = Arc::new(SealedBlock::for_policy_test(40));
    cache.batch_insert(vec![(key.clone(), Arc::clone(&current))]);
    cache.retain_demand(std::slice::from_ref(&key), &[old]);
    assert_class(&cache, &key, ResidentClass::Probationary);
    cache.retain_demand(std::slice::from_ref(&key), &[current]);
    assert_class(&cache, &key, ResidentClass::Retained);
    // A page larger than the protected allowance remains usable probation.
    cache.remove_all();
    cache.batch_insert(vec![(
        key.clone(),
        Arc::new(SealedBlock::for_policy_test(80)),
    )]);
    assert_eq!(
        cache.get_prefix_blocks(std::slice::from_ref(&key), false).0,
        1
    );
    assert_class(&cache, &key, ResidentClass::Probationary);
    cache.remove_all();
    assert_eq!(cache.inner.lock().retained_bytes, 0);
}

fn assert_class(cache: &DramStore, key: &StateKey, expected: ResidentClass) {
    let inner = cache.inner.lock();
    assert!(inner.cache.contains_key(key));
    assert_eq!(
        inner.reclaimable.contains_key(key),
        expected == ResidentClass::Reclaimable
    );
    assert_eq!(
        inner.probationary.contains_key(key),
        expected == ResidentClass::Probationary
    );
    assert_eq!(
        inner.retained.contains_key(key),
        expected == ResidentClass::Retained
    );
}

#[test]
fn watermark_scans_are_bounded_without_starving_other_classes_or_losing_inventory() {
    let cache = DramStore::new(4096, false, None, Some(16 * 1024), 0);
    let mut held = Vec::new();
    for id in 0..3 {
        let block = Arc::new(SealedBlock::for_policy_test(40));
        cache.batch_insert_reclaimable(vec![(StateKey::new("ns".into(), vec![id]), block.clone())]);
        held.push(block);
    }
    let victim = StateKey::new("ns".into(), vec![3]);
    cache.batch_insert(vec![(
        victim.clone(),
        Arc::new(SealedBlock::for_policy_test(40)),
    )]);
    let inventory = cache.inner.lock().inventory.as_ref().unwrap().clone();
    let before = inventory.page(None).unwrap();
    let removed = cache.remove_lru_batch(2, u64::MAX, ReclaimReason::Watermark);
    assert_eq!(
        removed.len(),
        1,
        "held reclaimable entries must not starve retained entries"
    );
    assert_eq!(removed[0].0, victim);
    assert_eq!(inventory.sequence(), 5);
    let changes = inventory.changes(4, 5).unwrap();
    assert_eq!(changes.len(), 1);
    assert!(!changes[0].present);
    assert!(cache.pin_residencies(&before).is_none());
    assert_eq!(
        cache.inner.lock().reclaimable.iter().next().unwrap().0.hash,
        vec![2]
    );
    drop(held);
    assert_eq!(
        cache
            .remove_lru_batch(2, u64::MAX, ReclaimReason::Watermark)
            .len(),
        2
    );
    assert_eq!(cache.inner.lock().reclaimable.len(), 1);
}

fn resident_metadata(cache: &DramStore, key: &StateKey) -> Option<ResidentMetadata> {
    let inner = cache.inner.lock();
    inner
        .reclaimable
        .peek(key)
        .or_else(|| inner.probationary.peek(key))
        .or_else(|| inner.retained.peek(key))
        .copied()
}

fn backdate_resident(cache: &DramStore, key: &StateKey, age: Duration) -> Instant {
    let inserted_at = Instant::now() - age;
    let mut inner = cache.inner.lock();
    let metadata = if let Some(metadata) = inner.reclaimable.peek_mut(key) {
        metadata
    } else {
        inner
            .retained
            .peek_mut(key)
            .expect("test resident must have replacement metadata")
    };
    metadata.inserted_at = inserted_at;
    inserted_at
}

#[test]
fn new_blocks_are_classified_by_source() {
    let cache = make_cache();
    let local = StateKey::new("ns".into(), vec![1]);
    let ssd = StateKey::new("ns".into(), vec![2]);
    let remote = StateKey::new("ns".into(), vec![3]);
    let local_block = make_block();

    cache.batch_insert_refs(&[(local.clone(), local_block)]);
    cache.batch_insert(vec![(ssd.clone(), make_block())]);
    cache.batch_insert_reclaimable(vec![(remote.clone(), make_block())]);

    assert_class(&cache, &local, ResidentClass::Retained);
    assert_class(&cache, &ssd, ResidentClass::Retained);
    assert_class(&cache, &remote, ResidentClass::Reclaimable);
}

#[test]
fn reclaimable_blocks_are_evicted_before_retained_blocks() {
    let cache = make_cache();
    let retained = StateKey::new("ns".into(), vec![1]);
    let reclaimable = StateKey::new("ns".into(), vec![2]);

    cache.batch_insert(vec![(retained.clone(), make_block())]);
    cache.batch_insert_reclaimable(vec![(reclaimable.clone(), make_block())]);

    let evicted = cache.remove_lru_batch(2, u64::MAX, ReclaimReason::Pressure);
    assert_eq!(
        evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
        vec![reclaimable, retained]
    );
}

#[test]
fn pressure_reclaim_ignores_weak_references() {
    let cache = make_cache();
    let key = StateKey::new("ns".into(), vec![1]);
    let block = make_block();
    let weak = Arc::downgrade(&block);
    cache.batch_insert(vec![(key.clone(), block)]);

    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        key
    );
    assert!(weak.upgrade().is_none());
}

#[test]
fn pressure_reclaim_waits_for_external_strong_reference() {
    let cache = make_cache();
    let key = StateKey::new("ns".into(), vec![1]);
    let block = make_block();
    let external = Arc::clone(&block);
    let weak = Arc::downgrade(&block);
    cache.batch_insert(vec![(key.clone(), block)]);

    assert!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .is_empty()
    );
    assert_class(&cache, &key, ResidentClass::Retained);

    drop(external);
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        key
    );
    assert!(weak.upgrade().is_none());
}

#[test]
fn local_hit_refreshes_recency_without_changing_class() {
    let cache = make_cache();
    let hit = StateKey::new("ns".into(), vec![1]);
    let oldest = StateKey::new("ns".into(), vec![2]);

    cache.batch_insert_reclaimable(vec![
        (hit.clone(), make_block()),
        (oldest.clone(), make_block()),
    ]);
    let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
    let (count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit), false);

    assert_eq!(count, 1);
    assert_eq!(
        resident_metadata(&cache, &hit).unwrap().inserted_at,
        inserted_at
    );
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        oldest
    );
    assert_class(&cache, &hit, ResidentClass::Reclaimable);
}

#[test]
fn warmup_hits_do_not_refresh_recency_and_unused_pages_are_reclaimable() {
    let cache = make_cache();
    let oldest = StateKey::new("ns".into(), vec![1]);
    let newest = StateKey::new("ns".into(), vec![2]);
    let warmed = StateKey::new("ns".into(), vec![3]);
    cache.batch_insert(vec![
        (oldest.clone(), make_block()),
        (newest.clone(), make_block()),
    ]);
    drop(cache.get_prefix_blocks(std::slice::from_ref(&oldest), true));
    let block = make_block();
    block.mark_warmed();
    cache.batch_insert_reclaimable(vec![(warmed.clone(), block)]);
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        warmed
    );
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        oldest
    );
    assert_class(&cache, &newest, ResidentClass::Retained);
}

#[test]
fn demand_promotes_only_the_matching_warmup_generation() {
    let cache = make_cache();
    let key = StateKey::new("ns".into(), vec![1]);
    let stale = make_block();
    stale.mark_warmed();
    let block = make_block();
    block.mark_warmed();
    cache.batch_insert_reclaimable(vec![(key.clone(), Arc::clone(&block))]);
    cache.retain_demand(std::slice::from_ref(&key), &[stale]);
    assert_class(&cache, &key, ResidentClass::Reclaimable);
    cache.retain_demand(std::slice::from_ref(&key), std::slice::from_ref(&block));
    assert_class(&cache, &key, ResidentClass::Retained);
    assert!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .is_empty()
    );
    drop(block);
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        key
    );
}

#[test]
fn serving_hit_refreshes_recency_without_changing_class() {
    let cache = make_cache();
    let hit = StateKey::new("ns".into(), vec![1]);
    let oldest = StateKey::new("ns".into(), vec![2]);

    cache.batch_insert(vec![
        (hit.clone(), make_block()),
        (oldest.clone(), make_block()),
    ]);
    let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
    assert_eq!(
        cache
            .get_blocks_aligned(std::slice::from_ref(&hit))
            .iter()
            .flatten()
            .count(),
        1
    );

    assert_eq!(
        resident_metadata(&cache, &hit).unwrap().inserted_at,
        inserted_at
    );
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        oldest
    );
    assert_class(&cache, &hit, ResidentClass::Retained);
}

#[test]
fn already_existing_insert_keeps_original_class() {
    let cache = make_cache();
    let remote_first = StateKey::new("ns".into(), vec![1]);
    let remote_other = StateKey::new("ns".into(), vec![2]);
    let local_first = StateKey::new("ns".into(), vec![3]);
    let local_other = StateKey::new("ns".into(), vec![4]);

    cache.batch_insert_reclaimable(vec![(remote_first.clone(), make_block())]);
    cache.batch_insert_reclaimable(vec![(remote_other.clone(), make_block())]);
    cache.batch_insert(vec![(remote_first.clone(), make_block())]);
    cache.batch_insert(vec![(local_first.clone(), make_block())]);
    cache.batch_insert(vec![(local_other.clone(), make_block())]);
    cache.batch_insert_reclaimable(vec![(local_first.clone(), make_block())]);

    assert_class(&cache, &remote_first, ResidentClass::Reclaimable);
    assert_class(&cache, &local_first, ResidentClass::Retained);
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        remote_other
    );
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        remote_first
    );
    assert_eq!(
        cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)[0].0,
        local_other
    );
    assert_class(&cache, &local_first, ResidentClass::Retained);
}

#[test]
fn already_existing_insert_preserves_residence_start() {
    let cache = make_cache();
    let key = StateKey::new("ns".into(), vec![1]);
    cache.batch_insert(vec![(key.clone(), make_block())]);
    let inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

    cache.batch_insert_reclaimable(vec![(key.clone(), make_block())]);

    assert_eq!(
        resident_metadata(&cache, &key).unwrap().inserted_at,
        inserted_at
    );
    assert_class(&cache, &key, ResidentClass::Retained);
}

#[test]
fn reinsert_after_eviction_starts_new_residence_episode() {
    let cache = make_cache();
    let key = StateKey::new("ns".into(), vec![1]);
    cache.batch_insert(vec![(key.clone(), make_block())]);
    let first_inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

    cache.remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure);
    cache.batch_insert(vec![(key.clone(), make_block())]);

    let second_inserted_at = resident_metadata(&cache, &key).unwrap().inserted_at;
    assert!(second_inserted_at > first_inserted_at);
}

#[test]
fn residence_duration_is_non_negative_and_finite() {
    let removed_at = Instant::now();
    let inserted_at = removed_at - Duration::from_secs(60);

    assert_eq!(residence_duration_seconds(inserted_at, removed_at), 60.0);
    assert_eq!(residence_duration_seconds(removed_at, inserted_at), 0.0);
    assert!(residence_duration_seconds(inserted_at, removed_at).is_finite());
}

#[test]
fn inventory_tracks_actual_residency_and_fences_old_grants() {
    let cache = DramStore::new(1 << 20, false, None, Some(16 * 1024), 0);
    let key = StateKey::new("ns".into(), vec![1]);
    cache.batch_insert_refs(&[(key.clone(), Arc::new(SealedBlock::for_policy_test(4096)))]);

    let first = cache
        .inner
        .lock()
        .inventory
        .as_ref()
        .unwrap()
        .page(None)
        .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(
        first[0].metadata,
        Some(orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Dram,
            representation: orbitkv_state::ReplicaRepresentation::Unknown,
            stored_bytes: Some(4096),
        })
    );
    cache.batch_insert_refs(&[(key.clone(), make_block())]);
    assert_eq!(cache.inner.lock().inventory.as_ref().unwrap().sequence(), 1);
    let pinned = cache.get_blocks_aligned(std::slice::from_ref(&key));
    assert!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .is_empty()
    );
    assert_eq!(cache.inner.lock().inventory.as_ref().unwrap().sequence(), 1);
    drop(pinned);
    assert_eq!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .len(),
        1
    );
    assert_eq!(cache.inner.lock().inventory.as_ref().unwrap().sequence(), 2);
    assert!(
        !cache
            .inner
            .lock()
            .inventory
            .as_ref()
            .unwrap()
            .changes(1, 2)
            .unwrap()[0]
            .present
    );
    // SSD restore uses the retained insertion path, and publishes a new episode.
    cache.batch_insert(vec![(key.clone(), make_block())]);
    assert!(cache.pin_residencies(&first).is_none());
    assert_class(&cache, &key, ResidentClass::Retained);
    cache.remove_all();
    assert_eq!(cache.inner.lock().inventory.as_ref().unwrap().sequence(), 4);
    assert!(
        cache
            .inner
            .lock()
            .inventory
            .as_ref()
            .unwrap()
            .page(None)
            .unwrap()
            .is_empty()
    );
    assert!(
        !cache
            .inner
            .lock()
            .inventory
            .as_ref()
            .unwrap()
            .changes(3, 4)
            .unwrap()[0]
            .present
    );
}

#[test]
fn pin_residencies_fences_eviction_and_reinsertion() {
    let cache = DramStore::new(1024 * 1024, false, None, Some(4096), 0);
    let key = StateKey::new("ns".into(), vec![1]);
    cache.batch_insert(vec![(key.clone(), make_block())]);

    let first = cache
        .inner
        .lock()
        .inventory
        .as_ref()
        .unwrap()
        .page(None)
        .unwrap();
    let pinned = cache.pin_residencies(&first).unwrap();
    assert!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .is_empty()
    );
    drop(pinned);
    assert_eq!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .len(),
        1
    );
    cache.batch_insert(vec![(key.clone(), make_block())]);
    assert!(cache.pin_residencies(&first).is_none());
    let current = cache
        .inner
        .lock()
        .inventory
        .as_ref()
        .unwrap()
        .page(None)
        .unwrap();
    let mut mixed = current.clone();
    mixed.extend(first);
    assert!(cache.pin_residencies(&mixed).is_none());
    assert!(cache.pin_residencies(&current).is_some());
    assert_eq!(
        cache
            .remove_lru_batch(1, u64::MAX, ReclaimReason::Pressure)
            .len(),
        1
    );
}

#[test]
fn remove_all_evicts_resident_blocks() {
    let cache = make_cache();
    let key1 = StateKey::new("ns".into(), vec![1]);
    let key2 = StateKey::new("ns".into(), vec![2]);

    cache.batch_insert(vec![
        (key1.clone(), make_block()),
        (key2.clone(), make_block()),
    ]);

    let removed = cache.remove_all();
    assert_eq!(removed.len(), 2);
    assert_eq!(
        cache
            .get_blocks_aligned(&[key1, key2])
            .iter()
            .flatten()
            .count(),
        0
    );
    let inner = cache.inner.lock();
    assert!(inner.reclaimable.is_empty());
    assert!(inner.retained.is_empty());
    drop(inner);
    assert!(cache.remove_all().is_empty());
}

#[test]
fn inventory_excludes_lfu_rejections_and_duplicate_restores() {
    let cache = DramStore::new(1, true, Some(1), Some(16 * 1024), 0);
    let hot = StateKey::new("ns".into(), vec![1]);
    let cold = StateKey::new("ns".into(), vec![2]);

    cache.batch_insert_reclaimable(vec![(hot.clone(), make_block())]);
    for _ in 0..2 {
        assert_eq!(
            cache
                .get_blocks_aligned(std::slice::from_ref(&hot))
                .iter()
                .flatten()
                .count(),
            1
        );
    }
    cache.batch_insert_reclaimable(vec![(cold.clone(), make_block())]);
    cache.batch_insert_reclaimable(vec![(hot.clone(), make_block())]);
    assert!(!cache.inner.lock().reclaimable.contains_key(&cold));
    assert_eq!(cache.inner.lock().inventory.as_ref().unwrap().sequence(), 1);
    assert_eq!(
        cache
            .inner
            .lock()
            .inventory
            .as_ref()
            .unwrap()
            .page(None)
            .unwrap()[0]
            .key,
        hot
    );
    assert_eq!(
        cache
            .inner
            .lock()
            .inventory
            .as_ref()
            .unwrap()
            .changes(0, 1)
            .unwrap()
            .len(),
        1
    );
}
