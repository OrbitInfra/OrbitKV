use super::*;

#[test]
fn failed_decode_hides_pinned_generation_until_drain_and_preserves_replacements() {
    let mut ring = SsdRingBuffer::new_sharded(vec![16384, 16384], 512);
    let key = make_key(1);
    let encoding = || Encoding::Encoded;
    let slots = || {
        vec![SlotMeta::new(
            smallvec::smallvec![512],
            crate::NumaNode::UNKNOWN,
        )]
    };
    let first = ring.reserve(&key, slots(), encoding()).unwrap();
    assert!(ring.commit(&key, true));
    ring.invalidate(&key, &first);
    assert!(ring.get(&key).is_none());
    let repaired = ring.reserve(&key, slots(), encoding()).unwrap();
    assert!(ring.commit(&key, true));
    ring.invalidate(&key, &first);
    assert!(ring.get(&key).is_some());
    repaired.readers.store(2, Ordering::Release);
    ring.invalidate(&key, &repaired);
    assert!(ring.get(&key).is_none());
    assert!(ring.reserve(&key, slots(), encoding()).is_none());
    assert!(matches!(
        ring.entries.get(&key),
        Some(SsdEntryState::Invalid(_))
    ));
    repaired.readers.store(1, Ordering::Release);
    ring.release_invalid(&key, &repaired);
    assert!(ring.entries.contains_key(&key));
    repaired.readers.store(0, Ordering::Release);
    ring.release_invalid(&key, &repaired);
    assert!(ring.get(&key).is_none());
    let next = ring.reserve(&key, slots(), encoding()).unwrap();
    assert!(ring.commit(&key, true));
    assert_eq!(next.shard_id, first.shard_id);
    ring.invalidate(&key, &first);
    ring.release_invalid(&key, &repaired);
    assert!(ring.get(&key).is_some());
}

#[test]
fn invalidated_extent_cannot_be_overwritten_until_gpu_leases_drain() {
    let mut ring = SsdRingBuffer::new_sharded(vec![4096], 4096);
    let slots = || {
        vec![SlotMeta::new(
            smallvec::smallvec![512],
            crate::NumaNode::UNKNOWN,
        )]
    };
    let key = make_key(1);
    let entry = ring.reserve(&key, slots(), Encoding::Encoded).unwrap();
    ring.commit(&key, true);
    entry.readers.store(1, Ordering::Release);
    ring.invalidate(&key, &entry);
    assert!(ring.get(&key).is_none());
    assert!(ring.reserve(&make_key(2), slots(), Encoding::Raw).is_none());
    assert_eq!((ring.shards[0].head, ring.shards[0].tail), (4096, 0));
    entry.readers.store(0, Ordering::Release);
    ring.release_invalid(&key, &entry);
    let repaired = ring.reserve(&key, slots(), Encoding::Encoded).unwrap();
    ring.commit(&key, true);
    assert_ne!(repaired.begin, entry.begin);
    ring.invalidate(&key, &entry);
    ring.release_invalid(&key, &entry);
    assert!(ring.get(&key).is_some());
}

#[test]
fn failed_write_retry_does_not_hide_or_evict_another_generation() {
    let mut ring = SsdRingBuffer::new_sharded(vec![16384, 16384], 4096);
    let slots = || {
        vec![SlotMeta::new(
            smallvec::smallvec![512],
            crate::memory::numa::NumaNode::UNKNOWN,
        )]
    };
    let key = make_key(1);
    let first = ring.reserve(&key, slots(), Encoding::Raw).unwrap();
    assert_eq!(first.len, 4096);
    assert_eq!(first.slots[0].total_size(), 512);
    assert!(ring.get(&key).is_none());
    assert!(ring.reserve(&key, slots(), Encoding::Raw).is_none());
    assert!(!ring.commit(&key, false));
    let retry = ring.reserve(&key, slots(), Encoding::Raw).unwrap();
    assert_ne!(first.shard_id, retry.shard_id);
    assert!(ring.commit(&key, true));
    ring.advance_tail(first.shard_id, 4096);
    assert_eq!(ring.get(&key).unwrap().shard_id, retry.shard_id);
    assert!(ring.shards[first.shard_id].order.is_empty());
}

fn make_key(n: u8) -> StateKey {
    StateKey::new("test".to_string(), vec![n])
}

#[test]
fn gpu_prefix_budget_falls_back_for_valid_encoded_segments_without_truncating_raw_hits() {
    let ring = SsdRingBuffer::new(4096);
    let mut entry = ring.test_entry(0, 0, 4096);
    entry.slots.push(SlotMeta::new(
        smallvec::smallvec![4096],
        crate::NumaNode::UNKNOWN,
    ));
    assert!(
        entry.fits_gpu_decode(4096),
        "raw reads require no codec workspace"
    );
    entry.slots[0].encoding = Some(vec![crate::codec::EncodedSegment {
        version: 1,
        format: orbitkv_state::StorageFormat::Fp8FromBf16,
        logical_bytes: 8192,
        stored_bytes: 4096,
        checksum: 0,
    }]);
    assert!(
        !entry.fits_gpu_decode(4096),
        "valid CPU fallback must use the host reader"
    );
    assert!(!entry.fits_gpu_decode(4096 + 8192 - 1));
    assert!(entry.fits_gpu_decode(4096 + 8192));
    entry.slots[0].encoding.as_mut().unwrap()[0].stored_bytes = usize::MAX;
    assert!(!entry.fits_gpu_decode(usize::MAX));
}

impl SsdRingBuffer {
    fn new(capacity: u64) -> Self {
        Self::new_sharded(vec![capacity], 1)
    }

    fn test_entry(&self, shard_id: usize, begin: u64, len: u64) -> SsdIndexEntry {
        SsdIndexEntry {
            shard_id,
            begin,
            len,
            file_offset: begin % self.shards[shard_id].capacity.max(1),
            slots: vec![],
            encoding: Encoding::Raw,
            readers: Arc::new(AtomicUsize::new(0)),
            checksum: Arc::default(),
        }
    }

    /// Insert a Committed entry for testing. Returns the key.
    fn insert_committed(&mut self, n: u8, begin: u64, len: u64) -> StateKey {
        let key = make_key(n);
        let entry = self.test_entry(0, begin, len);
        self.entries
            .insert(key.clone(), SsdEntryState::Committed(entry));
        self.shards[0].order.push_back((key.clone(), begin));
        key
    }

    /// Insert a Writing entry for testing. Returns the key.
    fn insert_writing(&mut self, n: u8, begin: u64, len: u64) -> StateKey {
        let key = make_key(n);
        let entry = self.test_entry(0, begin, len);
        self.entries
            .insert(key.clone(), SsdEntryState::Writing(entry));
        self.shards[0].order.push_back((key.clone(), begin));
        key
    }
}

#[test]
fn pinned_extent_and_pending_write_prevent_wrap_without_moving_the_ring() {
    let mut ring = SsdRingBuffer::new(4096);
    ring.allocate_contiguous(0, 4096).unwrap();
    let key = ring.insert_committed(1, 0, 2048);
    let pin = Arc::clone(&ring.get(&key).unwrap().readers);
    pin.store(1, Ordering::Release);
    assert!(ring.allocate_contiguous(0, 512).is_none());
    assert_eq!((ring.shards[0].head, ring.shards[0].tail), (4096, 0));
    assert!(ring.get(&key).is_some());
    pin.store(0, Ordering::Release);
    assert_eq!(ring.allocate_contiguous(0, 512), Some((4096, 0)));
    assert!(ring.get(&key).is_none());

    let mut ring = SsdRingBuffer::new(4096);
    ring.allocate_contiguous(0, 4096).unwrap();
    let key = ring.insert_writing(1, 0, 4096);
    assert!(ring.allocate_contiguous(0, 512).is_none());
    assert!(ring.commit(&key, true));
    assert!(ring.allocate_contiguous(0, 512).is_some());
}

// ========================================================================
// allocate_contiguous tests
// ========================================================================

#[test]
fn test_allocate_contiguous_basic() {
    let mut ring = SsdRingBuffer::new(1000);

    let (begin, offset) = ring.allocate_contiguous(0, 100).unwrap();
    assert_eq!(
        (begin, offset, ring.shards[0].head, ring.shards[0].tail),
        (0, 0, 100, 0)
    );

    let (begin, offset) = ring.allocate_contiguous(0, 200).unwrap();
    assert_eq!(
        (begin, offset, ring.shards[0].head, ring.shards[0].tail),
        (100, 100, 300, 0)
    );
}

#[test]
fn test_allocate_contiguous_wrap_around() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.allocate_contiguous(0, 900).unwrap();

    // 200 bytes doesn't fit in remaining 100, skips to wrap point
    let (begin, offset) = ring.allocate_contiguous(0, 200).unwrap();
    assert_eq!(begin, 1000); // skipped to wrap point
    assert_eq!(offset, 0); // wraps to file start
    assert_eq!(ring.shards[0].tail, 200); // head(1200) - capacity(1000)
}

#[test]
fn test_allocate_contiguous_prunes_expired() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_committed(1, 0, 100);

    ring.allocate_contiguous(0, 600).unwrap();
    ring.allocate_contiguous(0, 600).unwrap(); // head=1200, tail=200

    assert!(!ring.entries.contains_key(&key));
    assert!(ring.shards[0].order.is_empty());
}

// ========================================================================
// is_offset_valid tests
// ========================================================================

#[test]
fn test_is_offset_valid() {
    let mut ring = SsdRingBuffer::new(1000);
    assert!(ring.is_offset_valid(&ring.test_entry(0, 0, 10)));

    ring.shards[0].tail = 50;
    assert!(!ring.is_offset_valid(&ring.test_entry(0, 49, 10)));
    assert!(ring.is_offset_valid(&ring.test_entry(0, 50, 10)));
}

// ========================================================================
// commit tests
// ========================================================================

#[test]
fn test_commit_writing_to_committed() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    assert!(ring.commit(&key, true));
    assert!(matches!(
        ring.entries.get(&key),
        Some(SsdEntryState::Committed(_))
    ));
}

#[test]
fn test_commit_failure_removes_entry() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    assert!(!ring.commit(&key, false));
    assert!(!ring.entries.contains_key(&key));
    assert_eq!(ring.shards[0].order.len(), 1); // order cleaned by advance_tail later
}

#[test]
fn test_commit_expired_entry() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);
    ring.shards[0].tail = 200; // expire it

    assert!(!ring.commit(&key, true));
    assert!(!ring.entries.contains_key(&key));
}

#[test]
fn test_commit_edge_cases() {
    let mut ring = SsdRingBuffer::new(1000);

    // Missing key
    assert!(!ring.commit(&make_key(99), true));

    // Already committed (idempotent)
    let key = ring.insert_committed(1, 100, 50);
    assert!(ring.commit(&key, true));
}

// ========================================================================
// get tests
// ========================================================================

#[test]
fn test_get_committed_entries() {
    let mut ring = SsdRingBuffer::new(1000);
    let k_writing = ring.insert_writing(1, 100, 50);
    let k_committed = ring.insert_committed(2, 200, 50);

    // Writing: not readable
    assert!(ring.get(&k_writing).is_none());

    // Committed: readable
    let entry = ring.get(&k_committed).unwrap();
    assert_eq!((entry.begin, entry.len), (200, 50));

    // Expired: not readable
    ring.shards[0].tail = 250;
    assert!(ring.get(&k_committed).is_none());
}

// ========================================================================
// advance_tail tests
// ========================================================================

#[test]
fn test_advance_tail_prunes_expired() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.insert_committed(0, 0, 50);
    ring.insert_committed(1, 100, 50);
    ring.insert_committed(2, 200, 50);

    ring.advance_tail(0, 150);

    assert_eq!(ring.entries.len(), 1);
    assert!(ring.get(&make_key(2)).is_some());
}

#[test]
fn test_advance_tail_cleans_ghost_entries() {
    let mut ring = SsdRingBuffer::new(1000);
    // Ghost: in order but not in entries (aborted write)
    ring.shards[0].order.push_back((make_key(1), 0));
    ring.insert_committed(2, 200, 50);

    ring.advance_tail(0, 100);

    assert_eq!(ring.shards[0].order.len(), 1);
    assert_eq!(ring.shards[0].order.front(), Some(&(make_key(2), 200)));
}

#[test]
fn test_advance_tail_preserves_valid_writing() {
    let mut ring = SsdRingBuffer::new(1000);
    ring.insert_writing(1, 100, 50);

    ring.advance_tail(0, 50); // tail < begin, should preserve

    assert_eq!(ring.entries.len(), 1);
}

// ========================================================================
// Duplicate key filtering
// ========================================================================

#[test]
fn test_duplicate_key_filtered() {
    let mut ring = SsdRingBuffer::new(1000);
    let key = ring.insert_writing(1, 100, 50);

    let filtered: Vec<_> = vec![key]
        .into_iter()
        .filter(|k| !ring.entries.contains_key(k))
        .collect();

    assert!(filtered.is_empty());
}
