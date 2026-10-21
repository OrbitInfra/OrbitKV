// ============================================================================
// Block types for Storage
// ============================================================================

use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::memory::numa::NumaNode;
use crate::memory::pool::{MappedPinnedPtr, PinnedAllocation};

// ============================================================================
// StateKey
// ============================================================================

pub use orbitkv_state::StateKey;

pub type BlockHash = Vec<u8>;

/// Per-layer save input: layer name + block IDs + content hashes.
pub struct LayerSave {
    pub layer_name: String,
    pub block_ids: Vec<usize>,
    pub block_hashes: Vec<Vec<u8>>,
}

// ============================================================================
// Prefetch Status
// ============================================================================

/// Terminal prefix result. The query future owns all source blocks until return.
#[derive(Clone)]
pub struct QueryResult {
    pub blocks: Vec<RestoreSource>,
    pub missing: usize,
}

/// Owns either resident bytes or an immutable SSD extent until restore completes.
#[derive(Clone)]
pub enum RestoreSource {
    Memory(Arc<SealedBlock>),
    Ssd {
        lease: Arc<crate::storage::ssd::SsdReadLease>,
        path: crate::SsdReadPath,
        /// Automatic cuFile choice may fall back before submission when this
        /// target device has no staging owner. Explicit route controls cannot.
        allow_uring_fallback: bool,
    },
}

impl fmt::Debug for RestoreSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(match self {
            Self::Memory(_) => "Memory",
            Self::Ssd { .. } => "Ssd",
        })
        .field("slots", &self.slot_count())
        .field("bytes", &self.memory_footprint())
        .finish()
    }
}

impl RestoreSource {
    pub fn memory_footprint(&self) -> u64 {
        match self {
            Self::Memory(block) => block.memory_footprint(),
            Self::Ssd { lease, .. } => lease
                .entry
                .slots
                .iter()
                .map(crate::SlotMeta::total_size)
                .sum(),
        }
    }

    pub(crate) fn slot_count(&self) -> usize {
        match self {
            Self::Memory(block) => block.slots().len(),
            Self::Ssd { lease, .. } => lease.entry.slots.len(),
        }
    }
}

impl fmt::Debug for QueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryResult")
            .field("blocks", &self.blocks.len())
            .field("missing", &self.missing)
            .finish()
    }
}

// ============================================================================
// Segment + RawBlock (storage-level, layout-agnostic)
// ============================================================================

/// A single contiguous memory segment in pinned memory.
pub(crate) struct Segment {
    ptr: MappedPinnedPtr,
    size: usize,
    /// Held for RAII drop semantics -- memory freed when last Arc drops.
    _allocation: Arc<PinnedAllocation>,
}

impl Segment {
    pub(crate) fn host_ptr(&self) -> NonNull<u8> {
        self.ptr.host()
    }

    pub(crate) fn new(ptr: NonNull<u8>, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        let ptr = allocation.mapped_ptr_for_host_range(ptr, size);
        Self {
            ptr,
            size,
            _allocation: allocation,
        }
    }

    fn allocation_id(&self) -> usize {
        Arc::as_ptr(&self._allocation) as usize
    }
}

// Safety: Segment holds NonNull<u8> pointing to pinned memory whose lifetime
// is managed by Arc<PinnedAllocation>. PinnedAllocation itself is Send+Sync.
unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

/// Storage-level block: an ordered list of opaque memory segments.
/// No awareness of K/V layout -- that's the caller's concern.
///
/// RawBlock automatically derives Send+Sync from Segment.
pub struct RawBlock {
    pub(crate) storage_format: orbitkv_state::StorageFormat,
    pub(crate) encoding: Option<Vec<crate::codec::EncodedSegment>>,
    segments: BlockSegments,
    /// Total size across all segments (for footprint tracking).
    total_size: usize,
}

enum BlockSegments {
    One(Segment),
    Two([Segment; 2]),
    Many(Box<[Segment]>),
}

impl BlockSegments {
    fn from_vec(segments: Vec<Segment>) -> Self {
        match segments.len() {
            1 => {
                let mut segments = segments.into_iter();
                Self::One(segments.next().expect("length checked"))
            }
            2 => {
                let mut segments = segments.into_iter();
                Self::Two([
                    segments.next().expect("length checked"),
                    segments.next().expect("length checked"),
                ])
            }
            _ => Self::Many(segments.into_boxed_slice()),
        }
    }

    fn as_slice(&self) -> &[Segment] {
        match self {
            Self::One(segment) => std::slice::from_ref(segment),
            Self::Two(segments) => segments,
            Self::Many(segments) => segments,
        }
    }
}

impl RawBlock {
    /// Create from an arbitrary list of segments.
    /// Caller decides the segment layout; RawBlock is layout-agnostic.
    pub(crate) fn new(segments: Vec<Segment>) -> Self {
        debug_assert!(!segments.is_empty(), "RawBlock requires at least 1 segment");
        let total_size = segments.iter().map(|s| s.size).sum();
        Self {
            segments: BlockSegments::from_vec(segments),
            storage_format: Default::default(),
            encoding: None,
            total_size,
        }
    }

    pub(crate) fn single_segment(segment: Segment) -> Self {
        let total_size = segment.size;
        Self {
            segments: BlockSegments::One(segment),
            storage_format: Default::default(),
            encoding: None,
            total_size,
        }
    }

    pub(crate) fn two_segments(k_segment: Segment, v_segment: Segment) -> Self {
        let total_size = k_segment.size + v_segment.size;
        Self {
            segments: BlockSegments::Two([k_segment, v_segment]),
            storage_format: Default::default(),
            encoding: None,
            total_size,
        }
    }

    /// Stored representation metadata for wire adapters; absence means raw data.
    pub fn encoding(&self) -> Option<&[crate::EncodedSegment]> {
        self.encoding.as_deref()
    }

    pub(crate) fn validate_encoding(&self) -> Result<(), String> {
        if let Some(metadata) = &self.encoding {
            if metadata.len() != self.num_segments() {
                return Err("encoded segment count mismatch".into());
            }
            for (meta, (ptr, len)) in metadata.iter().zip(self.segment_iovecs()) {
                // SAFETY: sealed or freshly read segments are initialized and owned by this block.
                meta.validate(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) })?;
            }
        }
        Ok(())
    }

    /// Number of segments.
    pub(crate) fn num_segments(&self) -> usize {
        self.segments.as_slice().len()
    }

    /// Get segment pointer by index.
    pub(crate) fn segment_ptr(&self, index: usize) -> Option<NonNull<u8>> {
        self.segments.as_slice().get(index).map(|s| s.ptr.host())
    }

    /// Get mapped host/device pointer pair by index.
    pub(crate) fn segment_mapped_ptr(&self, index: usize) -> Option<MappedPinnedPtr> {
        self.segments.as_slice().get(index).map(|s| s.ptr)
    }

    pub(crate) fn segment_registration_id(&self, index: usize) -> Option<usize> {
        self.segments
            .as_slice()
            .get(index)
            .map(|segment| segment._allocation.registration_id())
    }

    pub(crate) fn source_range(
        &self,
        segment: usize,
        offset: usize,
        size: usize,
    ) -> Result<crate::transfer::local::SourceRange, String> {
        let segment = self
            .segments
            .as_slice()
            .get(segment)
            .ok_or_else(|| "missing raw source segment".to_string())?;
        if size == 0
            || offset
                .checked_add(size)
                .is_none_or(|end| end > segment.size)
        {
            return Err("raw source exceeds its host segment".into());
        }
        segment
            ._allocation
            .source_range(segment.ptr.add(offset).host(), size)
    }

    /// Get segment size by index.
    pub(crate) fn segment_size(&self, index: usize) -> Option<usize> {
        self.segments.as_slice().get(index).map(|s| s.size)
    }

    /// Iterator over (NonNull<u8>, size) pairs -- used for SSD I/O.
    pub(crate) fn segment_iovecs(&self) -> impl Iterator<Item = (NonNull<u8>, usize)> + '_ {
        self.segments
            .as_slice()
            .iter()
            .map(|s| (s.ptr.host(), s.size))
    }

    /// Total memory footprint.
    pub(crate) fn memory_footprint(&self) -> u64 {
        self.total_size as u64
    }

    fn representation(&self) -> orbitkv_state::ReplicaRepresentation {
        let Some(metadata) = &self.encoding else {
            return orbitkv_state::ReplicaRepresentation::Raw;
        };
        let mut representation = None;
        for segment in metadata {
            let next = orbitkv_state::ReplicaRepresentation::from(segment.format);
            representation = Some(match representation {
                None => next,
                Some(previous) if previous == next => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
        }
        representation.unwrap_or_default()
    }
}

// ============================================================================
// LayerBlock (thin KV wrapper, service/GPU layer only)
// ============================================================================

/// Layer-aware view: interprets RawBlock segments as K/V cache data.
/// Lives in the service/GPU layer, NOT in storage.
///
/// Invariant: inner RawBlock has >= 1 segment.
pub struct LayerBlock<'a> {
    raw: &'a RawBlock,
}

impl<'a> LayerBlock<'a> {
    /// Wrap a RawBlock as a KV layer block.
    /// Panics if the block has zero segments (violated invariant from construction).
    pub fn new(raw: &'a RawBlock) -> Self {
        debug_assert!(
            raw.num_segments() > 0,
            "LayerBlock requires at least 1 segment"
        );
        Self { raw }
    }

    /// K segment pointer (always segment 0).
    pub fn k_ptr(&self) -> *const u8 {
        self.raw.segment_ptr(0).unwrap().as_ptr()
    }

    /// K segment size.
    pub fn k_size(&self) -> usize {
        self.raw.segment_size(0).unwrap()
    }

    /// V segment pointer (segment 1 if split, None if contiguous).
    pub fn v_ptr(&self) -> Option<*const u8> {
        self.raw.segment_ptr(1).map(|p| p.as_ptr() as *const u8)
    }

    /// V segment size (None if contiguous).
    pub fn v_size(&self) -> Option<usize> {
        self.raw.segment_size(1)
    }

    /// Total size across all segments (K + V if split).
    pub fn size(&self) -> usize {
        self.raw.memory_footprint() as usize
    }
}

// ============================================================================
// Sealed Block (read path, immutable)
// ============================================================================

/// Immutable block after all slots are filled. Exposed to callers.
pub struct SealedBlock {
    slots: Box<[RawBlock]>,
    footprint: u64,
    /// Per-slot NUMA affinity; always covers every slot (`len == slots.len()`).
    /// Used by the SSD write path and advertised on cross-node transfer.
    slot_numas: Vec<NumaNode>,
    warmup: OnceLock<Warmup>,
}

#[cfg(test)]
#[path = "../tests/support/blocks.rs"]
mod test_blocks;

/// Tracks one physical read's pages through the last owner, including leases.
struct Warmup {
    bytes: u64,
    ready_at: Instant,
    restored: AtomicBool,
}

impl Warmup {
    fn finish(&self, outcome: &'static str) {
        let metrics = crate::metrics::core_metrics();
        metrics.warmup_pending_bytes.add(-(self.bytes as i64), &[]);
        metrics.warmup_wait_byte_seconds.add(
            self.bytes as f64 * self.ready_at.elapsed().as_secs_f64(),
            &[opentelemetry::KeyValue::new("outcome", outcome)],
        );
    }
}

impl Drop for Warmup {
    fn drop(&mut self) {
        if !*self.restored.get_mut() {
            crate::metrics::core_metrics()
                .warmup_unused_bytes
                .add(self.bytes, &[]);
            self.finish("unused");
        }
    }
}

impl SealedBlock {
    pub(crate) fn mark_warmed(&self) {
        self.warmup.get_or_init(|| {
            let metrics = crate::metrics::core_metrics();
            metrics.warmup_prepared_bytes.add(self.footprint, &[]);
            metrics.warmup_pending_bytes.add(self.footprint as i64, &[]);
            Warmup {
                bytes: self.footprint,
                ready_at: Instant::now(),
                restored: AtomicBool::new(false),
            }
        });
    }

    pub(crate) fn was_warmed(&self) -> bool {
        self.warmup.get().is_some()
    }

    /// Credit a page once after a successful GPU transfer, never at lookup.
    pub(crate) fn mark_warmup_restored(&self) {
        if let Some(warmup) = self.warmup.get()
            && !warmup.restored.swap(true, Ordering::Relaxed)
        {
            crate::metrics::core_metrics()
                .warmup_restored_bytes
                .add(self.footprint, &[]);
            warmup.finish("restored");
        }
    }

    pub(crate) fn get_slot(&self, slot_id: usize) -> Option<&RawBlock> {
        self.slots.get(slot_id)
    }

    pub(crate) fn memory_footprint(&self) -> u64 {
        self.footprint
    }

    pub(crate) fn replica_metadata(&self) -> orbitkv_state::ReplicaMetadata {
        let mut representation = None;
        for slot in &self.slots {
            let next = slot.representation();
            representation = Some(match representation {
                None => next,
                Some(previous) if previous == next => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
        }
        orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Dram,
            representation: representation.unwrap_or_default(),
            stored_bytes: Some(self.footprint),
        }
    }

    pub(crate) fn pinned_allocations(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.slots.iter().flat_map(|slot| {
            slot.segments
                .as_slice()
                .iter()
                .map(|segment| (segment.allocation_id(), segment._allocation.size_bytes()))
        })
    }

    #[cfg(feature = "orbitte")]
    pub(crate) fn allocations(&self) -> impl Iterator<Item = &Arc<PinnedAllocation>> {
        self.slots.iter().flat_map(|slot| {
            slot.segments
                .as_slice()
                .iter()
                .map(|segment| &segment._allocation)
        })
    }

    /// Get all slots (for serialization / cross-node transfer)
    pub fn slots(&self) -> &[RawBlock] {
        &self.slots
    }

    /// Per-slot NUMA affinity (used by SSD write path and cross-node transfer).
    pub fn slot_numas(&self) -> &[NumaNode] {
        &self.slot_numas
    }

    /// Create from per-slot `(block, NUMA)` pairs (deserialization / prefetch rebuild).
    /// Pairing keeps each slot and its advertised NUMA affinity in lockstep, so the
    /// cross-node serve path can never truncate slots against a short `slot_numas`.
    pub(crate) fn from_slots(slots: Vec<(RawBlock, NumaNode)>) -> Self {
        let mut blocks = Vec::with_capacity(slots.len());
        let mut slot_numas = Vec::with_capacity(slots.len());
        let mut footprint = 0u64;
        for (block, numa) in slots {
            footprint = footprint.saturating_add(block.memory_footprint());
            blocks.push(block);
            slot_numas.push(numa);
        }
        Self {
            slots: blocks.into_boxed_slice(),
            footprint,
            slot_numas,
            warmup: OnceLock::new(),
        }
    }

    /// Create from slots with pre-computed footprint (internal use)
    fn from_slots_with_footprint(
        slots: Box<[RawBlock]>,
        footprint: u64,
        slot_numas: Vec<NumaNode>,
    ) -> Self {
        Self {
            slots,
            footprint,
            slot_numas,
            warmup: OnceLock::new(),
        }
    }

    /// Create from a fully populated, slot-id ordered insert batch.
    pub(crate) fn from_ordered_slot_inserts(
        slots: Vec<(usize, RawBlock)>,
        total_slots: usize,
        numa_node: NumaNode,
    ) -> Result<Self, Vec<(usize, RawBlock)>> {
        if slots.len() != total_slots
            || slots
                .iter()
                .enumerate()
                .any(|(expected, (slot_id, _))| *slot_id != expected)
        {
            return Err(slots);
        }

        let mut footprint = 0u64;
        let mut blocks = Vec::with_capacity(total_slots);
        for (_, block) in slots {
            footprint = footprint.saturating_add(block.memory_footprint());
            blocks.push(block);
        }

        Ok(Self::from_slots_with_footprint(
            blocks.into_boxed_slice(),
            footprint,
            vec![numa_node; total_slots],
        ))
    }
}

// ============================================================================
// SlotInsertResult
// ============================================================================

/// Result of inserting a slot into an inflight block.
pub(crate) enum SlotInsertResult {
    /// Slot was newly inserted.
    Inserted {
        completed: bool,
        footprint_added: u64,
    },
    /// Slot already existed (no-op).
    Duplicate,
}

// ============================================================================
// Inflight Block (write path, mutable)
// ============================================================================

/// Block that is still being written. Internal to Storage.
pub(crate) struct InflightBlock {
    slots: Vec<Option<RawBlock>>,
    remaining: usize,
    total_slots: usize,
    footprint: u64,
    created_at: Instant,
    /// Per-slot NUMA node affinity, tracked during insertion for deterministic
    /// per-block NUMA assignment when the block is sealed.
    slot_numas: Vec<NumaNode>,
}

impl InflightBlock {
    pub(crate) fn new(total_slots: usize) -> Self {
        Self {
            slots: (0..total_slots).map(|_| None).collect(),
            remaining: total_slots,
            total_slots,
            footprint: 0,
            created_at: Instant::now(),
            slot_numas: vec![NumaNode::UNKNOWN; total_slots],
        }
    }

    /// Returns the age of this inflight block.
    pub(crate) fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the number of filled slots.
    pub(crate) fn filled_count(&self) -> usize {
        self.total_slots - self.remaining
    }

    /// Returns the total number of slots.
    pub(crate) fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Returns the current memory footprint of all inserted slots.
    pub(crate) fn footprint(&self) -> u64 {
        self.footprint
    }

    /// Insert a slot idempotently. Duplicate inserts are no-ops.
    pub(crate) fn insert_slot(
        &mut self,
        slot_id: usize,
        block: RawBlock,
        numa_node: NumaNode,
    ) -> SlotInsertResult {
        debug_assert!(
            slot_id < self.total_slots,
            "slot_id {} must be < total_slots {}",
            slot_id,
            self.total_slots
        );

        if self.slots[slot_id].is_some() {
            return SlotInsertResult::Duplicate;
        }

        let footprint_added = block.memory_footprint();
        self.footprint += footprint_added;
        self.slots[slot_id] = Some(block);
        self.slot_numas[slot_id] = numa_node;
        self.remaining = self
            .remaining
            .checked_sub(1)
            .expect("remaining should not underflow");

        SlotInsertResult::Inserted {
            completed: self.remaining == 0,
            footprint_added,
        }
    }

    /// Seal the block, converting to immutable SealedBlock.
    /// Panics if not all slots are filled.
    pub(crate) fn seal(self) -> SealedBlock {
        let slots: Vec<RawBlock> = self
            .slots
            .into_iter()
            .map(|opt| opt.expect("all slots must be filled before sealing"))
            .collect();
        SealedBlock::from_slots_with_footprint(
            slots.into_boxed_slice(),
            self.footprint,
            self.slot_numas,
        )
    }
}
