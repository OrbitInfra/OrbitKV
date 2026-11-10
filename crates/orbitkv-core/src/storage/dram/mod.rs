pub(crate) mod policy;

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
    time::Instant,
};

use hashlink::LruCache;
use orbitkv_state::InventoryRecord;
use parking_lot::Mutex;

use crate::storage::inventory::ResidencyInventory;

use crate::block::{SealedBlock, StateKey};
use crate::metrics::{
    CACHE_CLASS_PROBATIONARY, CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED,
    CACHE_RESIDENCE_REASON_CLEANUP, CACHE_RESIDENCE_REASON_PRESSURE, core_metrics,
};
use crate::storage::dram::policy::{CacheInsertOutcome, TinyLfuCache};

pub(crate) struct DramStore {
    inner: Mutex<DramStoreInner>,
}

struct DramStoreInner {
    inventory: Option<Arc<ResidencyInventory>>,
    cache: TinyLfuCache<StateKey, Arc<SealedBlock>>,
    reclaimable: LruCache<StateKey, ResidentMetadata>,
    probationary: LruCache<StateKey, ResidentMetadata>,
    retained: LruCache<StateKey, ResidentMetadata>,
    retained_bytes: u64,
    protected_limit: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResidentMetadata {
    inserted_at: Instant,
    bytes: u64,
}

struct RemovedResident {
    key: StateKey,
    block: Arc<SealedBlock>,
    /// Insertion time taken from the block's replacement-class metadata.
    inserted_at: Instant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidentClass {
    Reclaimable,
    Probationary,
    Retained,
}

impl DramStore {
    #[cfg(test)]
    pub(crate) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
        inventory_journal_bytes: Option<usize>,
        protected_limit: u64,
    ) -> Self {
        let inventory =
            inventory_journal_bytes.map(|bytes| Arc::new(ResidencyInventory::new(bytes)));
        Self::with_inventory(
            capacity_bytes,
            enable_lfu_admission,
            value_size_hint,
            inventory,
            protected_limit,
        )
    }

    pub(crate) fn with_inventory(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
        inventory: Option<Arc<ResidencyInventory>>,
        protected_limit: u64,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            inner: Mutex::new(DramStoreInner {
                inventory,
                cache,
                reclaimable: LruCache::new_unbounded(),
                probationary: LruCache::new_unbounded(),
                retained: LruCache::new_unbounded(),
                retained_bytes: 0,
                protected_limit,
            }),
        }
    }

    pub(crate) fn contains_keys(&self, keys: &[StateKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    pub(crate) fn discover(&self, keys: &[StateKey]) -> Vec<Option<Weak<SealedBlock>>> {
        let inner = self.inner.lock();
        keys.iter()
            .map(|key| inner.cache.peek(key).map(|block| Arc::downgrade(&block)))
            .collect()
    }

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(crate) fn get_prefix_blocks(
        &self,
        keys: &[StateKey],
        warming: bool,
    ) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut blocks = Vec::with_capacity(keys.len());
        {
            let mut inner = self.inner.lock();
            for key in keys {
                let block = if warming {
                    inner.cache.peek(key)
                } else {
                    inner.cache.get(key)
                };
                if let Some(block) = block {
                    if !warming {
                        if let Err(error) = retain_demand(&mut inner, key, &block) {
                            log::error!("DRAM retention failed: {error}");
                        }
                        refresh_recency(&mut inner, key);
                    }
                    blocks.push(block);
                } else {
                    break;
                }
            }
        }
        let valid = blocks
            .iter()
            .zip(keys)
            .take_while(|(block, key)| {
                if block.verify("dram") {
                    true
                } else {
                    self.invalidate(key, block);
                    false
                }
            })
            .count();
        blocks.truncate(valid);
        (valid, blocks)
    }

    pub(crate) fn retain_demand(&self, keys: &[StateKey], blocks: &[Arc<SealedBlock>]) {
        let mut inner = self.inner.lock();
        for (key, block) in keys.iter().zip(blocks) {
            if inner
                .cache
                .peek(key)
                .is_some_and(|resident| Arc::ptr_eq(&resident, block))
                && let Err(error) = retain_demand(&mut inner, key, block)
            {
                log::error!("DRAM retention failed: {error}");
            }
        }
    }

    pub(crate) fn batch_insert(&self, blocks: Vec<(StateKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Retained);
        }
    }

    pub(crate) fn batch_insert_reclaimable(&self, blocks: Vec<(StateKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(&mut inner, key, block, ResidentClass::Reclaimable);
        }
    }

    pub(crate) fn batch_insert_refs(&self, blocks: &[(StateKey, Arc<SealedBlock>)]) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Retained,
            );
        }
    }

    /// Check every insertion version and acquire payload Arcs under the same
    /// cache lock. A stale batch exposes no addresses and creates no session.
    pub(crate) fn pin_residencies(
        &self,
        records: &[InventoryRecord],
    ) -> Option<Vec<(StateKey, Arc<SealedBlock>)>> {
        let mut inner = self.inner.lock();
        let inventory = inner.inventory.as_ref()?;
        if records.is_empty()
            || !records
                .iter()
                .all(|record| inventory.contains_record(record, orbitkv_state::ReplicaMedium::Dram))
        {
            return None;
        }
        let mut found = Vec::with_capacity(records.len());
        for record in records {
            let block = inner.cache.get(&record.key)?;
            refresh_recency(&mut inner, &record.key);
            found.push((record.key.clone(), block));
        }
        drop(inner);
        for (key, block) in &found {
            if !block.verify("peer") {
                self.invalidate(key, block);
                return None;
            }
        }
        Some(found)
    }

    /// Position-aligned membership: entry `i` is the block for `keys[i]`, or
    /// `None` on miss. Unlike [`Self::get_prefix_blocks`] this never stops at
    /// the first gap — hybrid-cache checkpoint groups (recurrent state) have
    /// sparse hit patterns by design, where the caller picks the rightmost
    /// hit instead of a prefix.
    pub(crate) fn get_blocks_aligned(&self, keys: &[StateKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let mut inner = self.inner.lock();
        let found: Vec<_> = keys
            .iter()
            .map(|key| {
                inner.cache.get(key).inspect(|_| {
                    refresh_recency(&mut inner, key);
                })
            })
            .collect();
        drop(inner);
        found
            .into_iter()
            .zip(keys)
            .map(|(block, key)| {
                block.filter(|block| {
                    if block.verify("dram") {
                        true
                    } else {
                        self.invalidate(key, block);
                        false
                    }
                })
            })
            .collect()
    }

    pub(crate) fn invalidate(&self, key: &StateKey, failed: &Arc<SealedBlock>) {
        let mut inner = self.inner.lock();
        if !inner
            .cache
            .peek(key)
            .is_some_and(|current| Arc::ptr_eq(&current, failed))
        {
            return;
        }
        inner.cache.remove(key);
        for class in [
            ResidentClass::Retained,
            ResidentClass::Probationary,
            ResidentClass::Reclaimable,
        ] {
            if let Some(meta) = class_lru(&mut inner, class).remove(key) {
                if class == ResidentClass::Retained {
                    inner.retained_bytes -= meta.bytes;
                    if inner.protected_limit > 0 {
                        core_metrics()
                            .cache_protected_bytes
                            .add(-(meta.bytes as i64), &[]);
                    }
                }
                core_metrics()
                    .cache_resident_blocks
                    .add(-1, class.attributes());
                core_metrics()
                    .cache_resident_bytes
                    .add(-(meta.bytes as i64), &[]);
            }
        }
        if let Some(inventory) = &inner.inventory
            && let Err(error) = inventory.change(key, orbitkv_state::ReplicaMedium::Dram, None)
        {
            log::error!("Corrupt DRAM inventory invalidation failed: {error:?}");
        }
    }

    /// A restore grant carries owners rather than lookup keys. Only the rare
    /// corruption path scans the bounded replacement metadata for that owner.
    pub(crate) fn invalidate_owner(&self, failed: &Arc<SealedBlock>) {
        let keys: Vec<_> = {
            let inner = self.inner.lock();
            inner
                .retained
                .iter()
                .chain(inner.probationary.iter())
                .chain(inner.reclaimable.iter())
                .filter(|(key, _)| {
                    inner
                        .cache
                        .peek(key)
                        .is_some_and(|current| Arc::ptr_eq(&current, failed))
                })
                .map(|(key, _)| key.clone())
                .collect()
        };
        for key in keys {
            self.invalidate(&key, failed);
        }
    }

    pub(crate) fn remove_lru_batch(
        &self,
        batch_size: usize,
        target_bytes: u64,
    ) -> Vec<(StateKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let mut removed = Vec::with_capacity(batch_size);
            let mut removed_bytes = 0;
            for class in [
                ResidentClass::Reclaimable,
                ResidentClass::Probationary,
                ResidentClass::Retained,
            ] {
                if let Err(error) = remove_lru_batch_from_class(
                    &mut inner,
                    class,
                    batch_size,
                    target_bytes,
                    &mut removed,
                    &mut removed_bytes,
                ) {
                    log::error!("DRAM eviction failed: {error}");
                    break;
                }
            }
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_PRESSURE)
    }

    pub(crate) fn remove_all(&self) -> Vec<(StateKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let reclaimable_blocks = inner.reclaimable.len() as i64;
            let probationary_blocks = inner.probationary.len() as i64;
            let retained_blocks = inner.retained.len() as i64;
            let mut metadata = HashMap::with_capacity(
                inner.reclaimable.len() + inner.probationary.len() + inner.retained.len(),
            );
            metadata.extend(inner.reclaimable.drain());
            metadata.extend(inner.probationary.drain());
            metadata.extend(inner.retained.drain());
            let removed = inner
                .cache
                .remove_all()
                .into_iter()
                .map(|(key, block)| {
                    let inserted_at = metadata.remove(&key).map(|entry| entry.inserted_at);
                    debug_assert!(
                        inserted_at.is_some(),
                        "resident block is missing its replacement metadata"
                    );
                    RemovedResident {
                        inserted_at: inserted_at.unwrap_or_else(Instant::now),
                        key,
                        block,
                    }
                })
                .collect::<Vec<_>>();
            if let Some(inventory) = &mut inner.inventory {
                for entry in &removed {
                    if let Err(error) =
                        inventory.change(&entry.key, orbitkv_state::ReplicaMedium::Dram, None)
                    {
                        log::error!("Inventory mutation rejected: {error:?}");
                    }
                }
            }
            debug_assert_eq!(
                removed.len() as i64,
                reclaimable_blocks + probationary_blocks + retained_blocks,
                "resident cache and replacement classes diverged"
            );
            debug_assert!(
                metadata.is_empty(),
                "replacement metadata outlives its resident block"
            );
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
            metrics
                .cache_resident_blocks
                .add(-probationary_blocks, &*CACHE_CLASS_PROBATIONARY);
            metrics
                .cache_resident_blocks
                .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
            if inner.protected_limit > 0 {
                metrics
                    .cache_protected_bytes
                    .add(-(inner.retained_bytes as i64), &[]);
            }
            inner.retained_bytes = 0;
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_CLEANUP)
    }
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Reclaimable => &*CACHE_CLASS_RECLAIMABLE,
            Self::Probationary => &*CACHE_CLASS_PROBATIONARY,
            Self::Retained => &*CACHE_CLASS_RETAINED,
        }
    }
}

fn insert_block(
    inner: &mut DramStoreInner,
    key: StateKey,
    block: Arc<SealedBlock>,
    class: ResidentClass,
) -> CacheInsertOutcome {
    if block.was_warmed() && inner.cache.contains_key(&key) {
        return CacheInsertOutcome::AlreadyExists;
    }
    let footprint_bytes = block.memory_footprint();
    let replica_metadata = block.replica_metadata();
    let class = if inner.protected_limit > 0 && class == ResidentClass::Retained {
        ResidentClass::Probationary
    } else {
        class
    };
    let outcome = inner.cache.insert(key.clone(), block);
    match outcome {
        CacheInsertOutcome::InsertedNew => {
            if let Some(inventory) = &mut inner.inventory
                && let Err(error) = inventory.change(
                    &key,
                    orbitkv_state::ReplicaMedium::Dram,
                    Some(replica_metadata),
                )
            {
                log::error!("Inventory mutation rejected: {error:?}");
            }
            class_lru(inner, class).insert(
                key,
                ResidentMetadata {
                    inserted_at: Instant::now(),
                    bytes: footprint_bytes,
                },
            );
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
            if class == ResidentClass::Retained {
                inner.retained_bytes += footprint_bytes;
            }
        }
        CacheInsertOutcome::AlreadyExists => refresh_recency(inner, &key),
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn class_lru(
    inner: &mut DramStoreInner,
    class: ResidentClass,
) -> &mut LruCache<StateKey, ResidentMetadata> {
    match class {
        ResidentClass::Reclaimable => &mut inner.reclaimable,
        ResidentClass::Probationary => &mut inner.probationary,
        ResidentClass::Retained => &mut inner.retained,
    }
}

fn retain_demand(
    inner: &mut DramStoreInner,
    key: &StateKey,
    block: &SealedBlock,
) -> Result<(), String> {
    let from = if inner.probationary.contains_key(key) {
        ResidentClass::Probationary
    } else if block.was_warmed() {
        ResidentClass::Reclaimable
    } else {
        return Ok(());
    };
    if !class_lru(inner, from).contains_key(key) {
        return Ok(());
    }
    if inner.protected_limit > 0 {
        let Some(remaining) = inner.protected_limit.checked_sub(block.memory_footprint()) else {
            return Ok(());
        };
        while inner.retained_bytes > remaining {
            let key = inner
                .retained
                .iter()
                .next()
                .ok_or("protected bytes have no resident owner")?
                .0
                .clone();
            move_resident(
                inner,
                &key,
                ResidentClass::Retained,
                ResidentClass::Probationary,
            );
            core_metrics().cache_policy_demotions.add(1, &[]);
        }
    }
    move_resident(inner, key, from, ResidentClass::Retained);
    if inner.protected_limit > 0 {
        core_metrics().cache_policy_promotions.add(1, &[]);
    }
    Ok(())
}

fn move_resident(
    inner: &mut DramStoreInner,
    key: &StateKey,
    from: ResidentClass,
    to: ResidentClass,
) -> bool {
    let Some(metadata) = class_lru(inner, from).remove(key) else {
        return false;
    };
    let metrics = core_metrics();
    if from == ResidentClass::Retained {
        inner.retained_bytes -= metadata.bytes;
        if inner.protected_limit > 0 {
            metrics
                .cache_protected_bytes
                .add(-(metadata.bytes as i64), &[]);
        }
    }
    if to == ResidentClass::Retained {
        inner.retained_bytes += metadata.bytes;
        if inner.protected_limit > 0 {
            metrics
                .cache_protected_bytes
                .add(metadata.bytes as i64, &[]);
        }
    }
    class_lru(inner, to).insert(key.clone(), metadata);
    metrics.cache_resident_blocks.add(-1, from.attributes());
    metrics.cache_resident_blocks.add(1, to.attributes());
    true
}

fn refresh_recency(inner: &mut DramStoreInner, key: &StateKey) {
    let classified = inner.reclaimable.get(key).is_some()
        || inner.probationary.get(key).is_some()
        || inner.retained.get(key).is_some();
    debug_assert!(
        classified || !inner.cache.contains_key(key),
        "resident block is missing its replacement class"
    );
}

fn remove_lru(inner: &mut DramStoreInner, class: ResidentClass) -> Option<RemovedResident> {
    while let Some((key, metadata)) = class_lru(inner, class).remove_lru() {
        let block = inner.cache.remove(&key);
        debug_assert!(
            block.is_some(),
            "replacement class contains a non-resident block"
        );
        let Some(block) = block else {
            continue;
        };
        if let Some(inventory) = &mut inner.inventory
            && let Err(error) = inventory.change(&key, orbitkv_state::ReplicaMedium::Dram, None)
        {
            log::error!("Inventory mutation rejected: {error:?}");
        }
        let metrics = core_metrics();
        if class == ResidentClass::Retained {
            inner.retained_bytes -= metadata.bytes;
            if inner.protected_limit > 0 {
                metrics
                    .cache_protected_bytes
                    .add(-(metadata.bytes as i64), &[]);
            }
        }
        metrics.cache_resident_blocks.add(-1, class.attributes());
        metrics
            .cache_block_evictions_by_class
            .add(1, class.attributes());
        return Some(RemovedResident {
            key,
            block,
            inserted_at: metadata.inserted_at,
        });
    }
    None
}

fn remove_lru_batch_from_class(
    inner: &mut DramStoreInner,
    class: ResidentClass,
    batch_size: usize,
    target_bytes: u64,
    removed: &mut Vec<RemovedResident>,
    removed_bytes: &mut u64,
) -> Result<(), String> {
    let candidates = class_lru(inner, class).len();
    for _ in 0..candidates {
        if removed.len() == batch_size || *removed_bytes >= target_bytes {
            break;
        }

        let Some(key) = class_lru(inner, class)
            .iter()
            .next()
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if inner.cache.is_cache_owned_only(&key) {
            let block =
                remove_lru(inner, class).ok_or("cache-owned LRU candidate lost its resident")?;
            *removed_bytes = removed_bytes.saturating_add(block.block.memory_footprint());
            removed.push(block);
        } else {
            class_lru(inner, class).get(&key);
        }
    }
    Ok(())
}

fn record_residence_durations(
    removed: Vec<RemovedResident>,
    attributes: &[opentelemetry::KeyValue],
) -> Vec<(StateKey, Arc<SealedBlock>)> {
    let removed_at = Instant::now();
    let metrics = core_metrics();
    removed
        .into_iter()
        .map(|entry| {
            metrics.cache_residence_duration.record(
                residence_duration_seconds(entry.inserted_at, removed_at),
                attributes,
            );
            (entry.key, entry.block)
        })
        .collect()
}

fn residence_duration_seconds(inserted_at: Instant, removed_at: Instant) -> f64 {
    removed_at
        .saturating_duration_since(inserted_at)
        .as_secs_f64()
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/dram/mod.rs"]
mod tests;
