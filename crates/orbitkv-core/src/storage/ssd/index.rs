use crate::SlotMeta;
use crate::block::StateKey;
use crate::metrics::core_metrics;
use log::{debug, warn};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum Encoding {
    Raw,
    Encoded,
}

/// Metadata for a block stored in SSD cache
#[derive(Clone)]
pub(crate) struct SsdIndexEntry {
    /// Cache file shard containing this entry.
    pub shard_id: usize,
    /// Logical offset in the ring buffer (monotonically increasing)
    pub begin: u64,
    /// Aligned physical extent size; slot sizes describe the stored payload
    pub len: u64,
    /// Physical file offset for IO
    pub file_offset: u64,
    /// Per-slot metadata for rebuilding SealedBlock
    pub slots: Vec<SlotMeta>,
    pub encoding: Encoding,
    pub readers: Arc<AtomicUsize>,
}

impl SsdIndexEntry {
    pub(super) fn replica_metadata(&self) -> orbitkv_state::ReplicaMetadata {
        let mut representation = None;
        let mut stored_bytes = 0u64;
        for slot in &self.slots {
            stored_bytes = stored_bytes.saturating_add(slot.total_size());
            let next = match &slot.encoding {
                None => orbitkv_state::ReplicaRepresentation::Raw,
                Some(metadata) => {
                    let mut slot_representation = None;
                    for segment in metadata {
                        let current = orbitkv_state::ReplicaRepresentation::from(segment.format);
                        slot_representation = Some(match slot_representation {
                            None => current,
                            Some(previous) if previous == current => previous,
                            Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
                        });
                    }
                    slot_representation.unwrap_or_default()
                }
            };
            representation = Some(match representation {
                None => next,
                Some(previous) if previous == next => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
        }
        orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Ssd,
            representation: representation.unwrap_or_default(),
            stored_bytes: Some(stored_bytes),
        }
    }

    pub(super) fn fits_gpu_decode(&self, budget: usize) -> bool {
        self.slots
            .iter()
            .filter_map(|slot| slot.encoding.as_ref())
            .flatten()
            .all(|meta| {
                // Input assembly is 4096-aligned. One segment additionally needs
                // descriptors, a CRC reduction and the codec arena's base alignment.
                meta.stored_bytes
                    .checked_next_multiple_of(super::layout::ALIGNMENT)
                    .and_then(|bytes| bytes.checked_add(8192))
                    .is_some_and(|bytes| bytes <= budget)
            })
    }
}

/// State of an SSD index entry (two-phase commit)
#[derive(Clone)]
pub(super) enum SsdEntryState {
    /// IO in progress, not yet readable
    Writing(SsdIndexEntry),
    /// Recovered metadata is not a hit until its full stored-byte CRC verifies.
    Unverified(SsdIndexEntry),
    /// IO completed, readable
    Committed(SsdIndexEntry),
    /// Corruption hides this generation immediately, but active readers still
    /// protect its extent until their final GPU completion.
    Invalid(SsdIndexEntry),
}

impl SsdEntryState {
    #[inline]
    fn entry(&self) -> &SsdIndexEntry {
        match self {
            Self::Writing(e) | Self::Unverified(e) | Self::Committed(e) | Self::Invalid(e) => e,
        }
    }
}

struct SsdShardRing {
    capacity: u64,
    head: u64,
    tail: u64,
    order: VecDeque<(StateKey, u64)>,
}

/// SSD ring buffer: unified state for space allocation + block index.
///
/// Combines head/tail pointers with FIFO index. Maintains insertion order
/// for O(k) tail pruning while preserving O(1) lookup via HashMap.
///
/// Two-phase commit: reserve inserts Writing state, commit transitions
/// to Committed (or removes on failure). Only Committed entries are readable.
pub(super) struct SsdRingBuffer {
    /// Per-file ring state.
    shards: Vec<SsdShardRing>,
    /// Round-robin cursor for selecting the next write shard.
    next_shard: usize,
    alignment: u64,
    /// Fast lookup: key -> state (Writing or Committed)
    entries: HashMap<StateKey, SsdEntryState>,
    retired: Vec<StateKey>,
}

impl SsdRingBuffer {
    #[cfg(feature = "test-hooks")]
    pub(super) fn process_report(&self) -> serde_json::Value {
        serde_json::json!({
            "entries": self.entries.len(),
            "readers": self.entries.values().map(|s| s.entry().readers.load(Ordering::Acquire)).sum::<usize>(),
            "writing": self.entries.values().filter(|s| matches!(s, SsdEntryState::Writing(_))).count(),
            "bytes": self.entries.values().map(|s| s.entry().len).sum::<u64>(),
        })
    }
    pub(super) fn recover(&mut self, heads: &[u64], entries: &[super::persistence::Recovered]) {
        for (shard, &head) in self.shards.iter_mut().zip(heads) {
            shard.head = head;
            shard.tail = head.saturating_sub(shard.capacity);
        }
        for recovered in entries {
            let entry = recovered.entry.clone();
            self.shards[entry.shard_id]
                .order
                .push_back((recovered.key.clone(), entry.begin));
            self.entries
                .insert(recovered.key.clone(), SsdEntryState::Unverified(entry));
        }
    }

    pub(super) fn unverified(&self, key: &StateKey, entry: &SsdIndexEntry) -> bool {
        matches!(self.entries.get(key), Some(SsdEntryState::Unverified(current))
            if Arc::ptr_eq(&current.readers, &entry.readers))
    }

    pub(super) fn verify(&mut self, key: &StateKey, entry: &SsdIndexEntry) -> bool {
        if !self.unverified(key, entry) || !self.is_offset_valid(entry) {
            return false;
        }
        self.entries
            .insert(key.clone(), SsdEntryState::Committed(entry.clone()));
        true
    }

    pub(super) fn invalidate(&mut self, key: &StateKey, failed: &SsdIndexEntry) -> bool {
        if self
            .entries
            .get(key)
            .is_none_or(|state| !Arc::ptr_eq(&state.entry().readers, &failed.readers))
        {
            return false;
        }
        if failed.readers.load(Ordering::Acquire) == 0 {
            self.entries.remove(key);
        } else {
            self.entries
                .insert(key.clone(), SsdEntryState::Invalid(failed.clone()));
        }
        true
    }

    pub(super) fn new_sharded(shard_capacities: Vec<u64>, alignment: u64) -> Self {
        assert!(
            !shard_capacities.is_empty(),
            "SSD cache needs at least one shard"
        );
        Self {
            shards: shard_capacities
                .into_iter()
                .map(|capacity| SsdShardRing {
                    capacity,
                    head: 0,
                    tail: 0,
                    order: VecDeque::new(),
                })
                .collect(),
            next_shard: 0,
            alignment,
            entries: HashMap::new(),
            retired: Vec::new(),
        }
    }

    /// Lookup a Committed entry by key, returning None if Writing or expired.
    pub(super) fn get(&self, key: &StateKey) -> Option<&SsdIndexEntry> {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) if self.is_offset_valid(e) => Some(e),
            _ => None,
        }
    }

    /// Hide exactly the failed generation, retaining its extent while leased.
    pub(super) fn invalidate_encoded(&mut self, key: &StateKey, failed: &SsdIndexEntry) -> bool {
        if matches!(self.entries.get(key), Some(SsdEntryState::Committed(entry) | SsdEntryState::Invalid(entry))
            if entry.shard_id == failed.shard_id && entry.begin == failed.begin
                && matches!(entry.encoding, Encoding::Encoded))
        {
            if failed.readers.load(Ordering::Acquire) == 0 {
                self.entries.remove(key);
            } else {
                self.entries
                    .insert(key.clone(), SsdEntryState::Invalid(failed.clone()));
            }
            return true;
        }
        false
    }

    pub(super) fn release_invalid(&mut self, key: &StateKey, released: &SsdIndexEntry) {
        if matches!(self.entries.get(key), Some(SsdEntryState::Invalid(entry))
            if entry.shard_id == released.shard_id && entry.begin == released.begin
                && entry.readers.load(Ordering::Acquire) == 0)
        {
            self.entries.remove(key);
        }
    }

    /// Check if a logical offset is still valid (not yet overwritten).
    #[inline]
    pub(super) fn is_offset_valid(&self, entry: &SsdIndexEntry) -> bool {
        self.shards
            .get(entry.shard_id)
            .is_some_and(|shard| entry.begin >= shard.tail)
    }

    /// Allocate contiguous space for a batch and advance tail.
    /// Returns (begin, file_offset). Skips wrap-around gap if needed.
    fn allocate_contiguous(&mut self, shard_id: usize, size: u64) -> Option<(u64, u64)> {
        let shard = &self.shards[shard_id];
        if size == 0 || size > shard.capacity {
            return None;
        }
        let phys = shard.head % shard.capacity;
        let space_until_end = shard.capacity - phys;
        let begin = if size > space_until_end {
            shard.head.checked_add(space_until_end)?
        } else {
            shard.head
        };
        let head = begin.checked_add(size)?;
        let new_tail = head.saturating_sub(shard.capacity);
        // A query owns these extents through GPU completion. Do not move the
        // ring, evict their identity, or overwrite even part of a pinned block.
        for (key, generation) in &shard.order {
            if let Some(state) = self.entries.get(key)
                && state.entry().shard_id == shard_id
                && state.entry().begin == *generation
            {
                let entry = state.entry();
                if entry.begin >= new_tail {
                    break;
                }
                if matches!(state, SsdEntryState::Writing(_))
                    || entry.readers.load(Ordering::Acquire) != 0
                {
                    core_metrics().ssd_pinned_write_skips.add(1, &[]);
                    return None;
                }
            }
        }
        let capacity = shard.capacity;
        self.shards[shard_id].head = head;
        self.advance_tail(shard_id, new_tail);

        Some((begin, begin % capacity))
    }

    /// Advance tail and prune expired entries (FIFO order).
    /// Handles both Writing and Committed states uniformly.
    fn advance_tail(&mut self, shard_id: usize, new_tail: u64) {
        if new_tail <= self.shards[shard_id].tail {
            return;
        }
        self.shards[shard_id].tail = new_tail;

        while let Some((key, generation)) = self.shards[shard_id].order.front() {
            let current = self.entries.get(key).is_some_and(|state| {
                state.entry().shard_id == shard_id && state.entry().begin == *generation
            });
            if current && *generation >= new_tail {
                break;
            }
            let key = key.clone();
            self.shards[shard_id].order.pop_front();
            if current {
                self.entries.remove(&key);
                self.retired.push(key);
            }
        }
    }

    pub(super) fn take_retired(&mut self) -> Vec<StateKey> {
        std::mem::take(&mut self.retired)
    }

    /// Commit a write: success=true transitions Writing→Committed, success=false removes.
    /// Returns false if entry was already expired or missing.
    pub(super) fn commit(&mut self, key: &StateKey, success: bool) -> bool {
        let Some(state) = self.entries.get(key) else {
            // Already removed by advance_tail or previous abort
            return false;
        };

        // Only process Writing state
        let entry = match state {
            SsdEntryState::Writing(e) => e,
            SsdEntryState::Committed(_) => {
                warn!("SSD commit: key already committed, ignoring");
                return true;
            }
            SsdEntryState::Invalid(_) | SsdEntryState::Unverified(_) => return false,
        };

        // Check if expired (eviction faster than write)
        if !self.is_offset_valid(entry) {
            warn!("SSD commit: entry expired before IO completed");
            self.entries.remove(key);
            return false;
        }

        if success {
            // Writing → Committed
            let entry = entry.clone();
            self.entries
                .insert(key.clone(), SsdEntryState::Committed(entry));
            true
        } else {
            // Write failed, remove entry (order will be cleaned by advance_tail)
            self.entries.remove(key);
            false
        }
    }

    /// Reserve an unpublished object. Active writes and pinned readers cannot be overwritten.
    pub(super) fn reserve(
        &mut self,
        key: &StateKey,
        slots: Vec<SlotMeta>,
        encoding: Encoding,
    ) -> Option<SsdIndexEntry> {
        if self.entries.contains_key(key) {
            return None;
        }
        let payload = slots
            .iter()
            .try_fold(0u64, |sum, slot| sum.checked_add(slot.total_size()))?;
        let size = payload.checked_next_multiple_of(self.alignment)?;
        let available = (0..self.shards.len()).find_map(|offset| {
            let shard_id = (self.next_shard + offset) % self.shards.len();
            self.allocate_contiguous(shard_id, size)
                .map(|(begin, file_offset)| (shard_id, begin, file_offset))
        });
        let Some((shard_id, begin, file_offset)) = available else {
            debug!("SSD cache: cannot reserve block {key:?}, size {size} (oversized or pinned)");
            return None;
        };
        self.next_shard = (shard_id + 1) % self.shards.len();
        let entry = SsdIndexEntry {
            shard_id,
            begin,
            len: size,
            file_offset,
            slots,
            encoding,
            readers: Arc::new(AtomicUsize::new(0)),
        };
        self.entries
            .insert(key.clone(), SsdEntryState::Writing(entry.clone()));
        self.shards[shard_id].order.push_back((key.clone(), begin));
        Some(entry)
    }
}

impl Default for SsdRingBuffer {
    fn default() -> Self {
        Self::new_sharded(vec![0], 1)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/index.rs"]
mod tests;
