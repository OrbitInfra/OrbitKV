use std::{
    collections::HashMap,
    num::NonZeroU64,
    os::fd::OwnedFd,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

mod backend;

use bytesize::ByteSize;
use log::{error, info, warn};
use parking_lot::Mutex;

use crate::memory::allocator::{Allocation, ScaledOffsetAllocator};
use crate::memory::numa::{NumaNode, run_on_numa};
use crate::memory::pinned::{PagePolicy, PinnedMemory};
use crate::metrics::core_metrics;
use crate::transfer::local::SourceRange;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Size-sealed payload backing exported once to a GPU session.
#[derive(Debug)]
pub struct PayloadArena {
    pub id: u64,
    pub size: u64,
    pub fd: OwnedFd,
}

#[derive(Clone, Copy)]
pub(crate) struct MappedPinnedPtr {
    host: NonNull<u8>,
    device: NonNull<u8>,
}

// SAFETY: Both pointers identify the same CUDA pinned allocation. Ownership is
// carried separately by PinnedAllocation/RawBlock/LayerAlloc, and this value is
// only an address pair used to build CUDA copy descriptors.
unsafe impl Send for MappedPinnedPtr {}
// SAFETY: this copies addresses without accessing bytes. PinnedAllocation/RawBlock
// owns the backing; descriptor users retain it and serialize DMA/host access.
unsafe impl Sync for MappedPinnedPtr {}

impl MappedPinnedPtr {
    pub(crate) fn new(host: NonNull<u8>, device: NonNull<u8>) -> Self {
        Self { host, device }
    }

    pub(crate) fn host(self) -> NonNull<u8> {
        self.host
    }

    pub(crate) fn device(self) -> NonNull<u8> {
        self.device
    }

    pub(crate) fn add(self, offset: usize) -> Self {
        // SAFETY: callers create offsets from validated allocation/segment ranges.
        unsafe {
            Self {
                host: NonNull::new_unchecked(self.host.as_ptr().add(offset)),
                device: NonNull::new_unchecked(self.device.as_ptr().wrapping_add(offset)),
            }
        }
    }
}

/// RAII guard for a pinned memory allocation.
/// Automatically frees the allocation when dropped.
pub struct PinnedAllocation {
    id: u64,
    allocation: Allocation,
    ptr: NonNull<u8>,
    device_ptr: NonNull<u8>,
    pool: Arc<PinnedMemoryPool>,
    _engine_memory: backend::MemoryRegistration,
}

// SAFETY: moving the allocation does not move its backing. The retained pool
// owns the mapping and serializes allocator reuse; transfers retain this owner
// and coordinate byte access separately until physical drain.
unsafe impl Send for PinnedAllocation {}
// SAFETY: shared methods expose addresses, not mutable references; allocation
// owners and the transfer/grant protocol fence writes and retain backing through drain.
unsafe impl Sync for PinnedAllocation {}

impl PinnedAllocation {
    pub(crate) fn contains(&self, address: u64, bytes: usize) -> bool {
        address >= self.ptr.as_ptr() as u64
            && address
                .checked_add(bytes as u64)
                .is_some_and(|end| end <= self.ptr.as_ptr() as u64 + self.size_bytes())
    }
    pub(crate) fn allocate_sibling(&self, bytes: usize) -> Option<Arc<Self>> {
        self.pool
            .allocate(NonZeroU64::new(bytes as u64)?)
            .map(Arc::new)
    }

    pub(crate) fn registration_id(&self) -> usize {
        Arc::as_ptr(&self.pool) as usize
    }

    pub(crate) fn source_range(
        &self,
        host: NonNull<u8>,
        size: usize,
    ) -> Result<SourceRange, String> {
        let offset = (host.as_ptr() as usize)
            .checked_sub(self.ptr.as_ptr() as usize)
            .ok_or_else(|| "source begins before its pinned allocation".to_string())?
            as u64;
        let size = size as u64;
        if size == 0
            || offset
                .checked_add(size)
                .is_none_or(|end| end > self.size_bytes())
        {
            return Err("source exceeds its pinned allocation".into());
        }
        Ok(SourceRange {
            arena_id: self.pool.id,
            allocation_id: self.id,
            allocation_offset: self.allocation.offset_bytes,
            allocation_size: self.size_bytes(),
            offset: self.allocation.offset_bytes + offset,
            size,
        })
    }

    pub(crate) fn size_bytes(&self) -> u64 {
        self.allocation.size_bytes.get()
    }

    pub(crate) fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    pub(crate) fn mapped_ptr(&self) -> MappedPinnedPtr {
        MappedPinnedPtr::new(self.as_non_null(), self.device_ptr)
    }

    pub(crate) fn mapped_ptr_at(
        &self,
        offset: usize,
        size: usize,
    ) -> Result<MappedPinnedPtr, crate::EngineError> {
        let alloc_size = self.allocation.size_bytes.get() as usize;
        if offset.checked_add(size).is_none_or(|end| end > alloc_size) {
            return Err(crate::EngineError::InvalidArgument(
                "mapped pointer range exceeds pinned allocation".into(),
            ));
        }
        Ok(self.mapped_ptr().add(offset))
    }

    pub(crate) fn mapped_ptr_for_host_range(
        &self,
        host_ptr: NonNull<u8>,
        size: usize,
    ) -> Result<MappedPinnedPtr, crate::EngineError> {
        let base = self.ptr.as_ptr() as usize;
        let host = host_ptr.as_ptr() as usize;
        let offset = host.checked_sub(base).ok_or_else(|| {
            crate::EngineError::InvalidArgument(
                "host pointer is before pinned allocation base".into(),
            )
        })?;
        self.mapped_ptr_at(offset, size)
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Automatically free the allocation when the guard is dropped
        self.pool.free_internal(&self.allocation);
    }
}

/// Manages a CUDA pinned memory pool and a byte-addressable allocator.
pub(crate) struct PinnedMemoryPool {
    id: u64,
    next_allocation_id: AtomicU64,
    /// Backing pinned memory (handles mmap + cudaHostRegister)
    backing: Arc<PinnedMemory>,
    _local_pin: backend::PoolRegistration,
    allocator: Mutex<ScaledOffsetAllocator>,
    allocatable_bytes: u64,
}

impl std::fmt::Debug for PinnedMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedMemoryPool")
            .field("id", &self.id)
            .field("bytes", &self.allocatable_bytes)
            .finish_non_exhaustive()
    }
}

impl PinnedMemoryPool {
    /// Upper bound for simultaneous allocations in the pinned pool.
    const MAX_ALLOCS: u32 = 4_000_000;
    /// Alignment for unit size (512 bytes for Direct I/O compatibility)
    const UNIT_ALIGNMENT: u64 = 512;

    /// Calculate unit size that fits pool_size into u32 range
    fn compute_unit_size(pool_size: u64, hint: Option<NonZeroU64>) -> u64 {
        let max_units = u32::MAX as u64;
        let min_unit_for_capacity = pool_size.div_ceil(max_units);
        let base = hint.map(|h| h.get()).unwrap_or(Self::UNIT_ALIGNMENT);
        let unit = base.max(min_unit_for_capacity);
        unit.div_ceil(Self::UNIT_ALIGNMENT) * Self::UNIT_ALIGNMENT
    }

    /// Allocate a new pinned memory pool of `pool_size` bytes.
    ///
    /// `node` is the target NUMA node for first-touch placement of the backing
    /// region (used by Regular and HugePages strategies). Pass
    /// `NumaNode::UNKNOWN` for placement-agnostic allocation.
    ///
    /// If `use_hugepages` is true, uses huge pages (requires system config).
    /// If `unit_size_hint` is provided, the allocator rounds allocations up to this size.
    fn new(
        pool_size: usize,
        use_hugepages: bool,
        unit_size_hint: Option<NonZeroU64>,
        node: NumaNode,
    ) -> Result<Self, String> {
        let pages = if use_hugepages {
            PagePolicy::HugePages
        } else {
            PagePolicy::Regular
        };
        info!("Allocating shared pinned pool on {} with {:?}", node, pages);
        let backing = Arc::new(
            PinnedMemory::allocate(pool_size, pages, node).map_err(|error| error.to_string())?,
        );
        let local_pin = backend::pin(&backing)?;

        let actual_size = backing.size() as u64;
        let unit_size = Self::compute_unit_size(actual_size, unit_size_hint);
        let max_units = actual_size / unit_size;
        let allocatable_bytes = max_units * unit_size;

        info!(
            "Pinned pool: backing_size={}, unit_size={}, max_units={}, allocatable_size={}",
            ByteSize(actual_size),
            ByteSize(unit_size),
            max_units,
            ByteSize(allocatable_bytes)
        );

        let allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
            actual_size,
            unit_size,
            Self::MAX_ALLOCS,
        )
        .map_err(|error| error.to_string())?;

        let id = NEXT_ARENA_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "payload arena identity exhausted".to_string())?;

        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(allocatable_bytes) {
            metrics.pool_capacity_bytes.add(capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric update: allocatable_bytes={}",
                allocatable_bytes
            );
        }

        Ok(Self {
            id,
            next_allocation_id: AtomicU64::new(1),
            backing,
            _local_pin: local_pin,
            allocator: Mutex::new(allocator),
            allocatable_bytes,
        })
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    fn allocate(self: &Arc<Self>, size: NonZeroU64) -> Option<PinnedAllocation> {
        let id = self
            .next_allocation_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .ok()?;
        // Allocation is done under lock, metrics and pointer computation outside lock
        let allocation = {
            let mut allocator = self.allocator.lock();
            match allocator.allocate(size.get()) {
                Ok(Some(allocation)) => allocation,
                Ok(None) => return None, // Pool exhausted, caller can retry after eviction
                Err(err) => {
                    error!(
                        "Pinned memory allocation error: {} (requested {}): requested_bytes={}",
                        err,
                        ByteSize(size.get()),
                        size.get()
                    );
                    return None;
                }
            }
        };

        let size_bytes = allocation.size_bytes.get();
        let backing_size = self.backing.size() as u64;
        let allocation_end = allocation.offset_bytes.checked_add(size_bytes);
        assert!(
            matches!(allocation_end, Some(end) if end <= backing_size),
            "pinned allocator returned allocation outside backing memory: offset={} size={} backing_size={}",
            allocation.offset_bytes,
            size_bytes,
            backing_size
        );

        let offset = allocation.offset_bytes as usize;
        let mapped =
            MappedPinnedPtr::new(self.backing.as_non_null(), self.backing.device_ptr()).add(offset);
        let ptr = mapped.host();
        let device_ptr = mapped.device();

        if let Ok(size_i64) = i64::try_from(size_bytes) {
            core_metrics().pool_used_bytes.add(size_i64, &[]);
        }

        Some(PinnedAllocation {
            id,
            allocation,
            ptr,
            device_ptr,
            pool: Arc::clone(self),
            _engine_memory: Default::default(),
        })
    }

    fn allocation_footprint(&self, size: NonZeroU64) -> Option<u64> {
        self.allocator.lock().allocation_footprint(size.get()).ok()
    }

    /// Internal method to free a pinned memory allocation.
    /// This is called automatically by PinnedAllocation's Drop implementation.
    /// Users should not call this directly - use PinnedAllocation RAII instead.
    fn free_internal(&self, allocation: &Allocation) {
        // Free under lock, metrics update outside lock
        {
            let mut allocator = self.allocator.lock();
            allocator.free(allocation);
        }

        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            core_metrics().pool_used_bytes.add(-size_i64, &[]);
        }
    }

    /// Get (used_bytes, total_bytes) for the pool.
    fn usage(&self) -> (u64, u64) {
        let allocator = self.allocator.lock();
        let report = allocator.storage_report();
        let total = allocator.total_bytes();
        let used = total - report.total_free_bytes;
        (used, total)
    }

    /// Largest contiguous free region currently available, in bytes.
    fn largest_free_allocation(&self) -> u64 {
        let allocator = self.allocator.lock();
        allocator.storage_report().largest_free_allocation_bytes
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(self.allocatable_bytes) {
            metrics.pool_capacity_bytes.add(-capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric cleanup: allocatable_bytes={}",
                self.allocatable_bytes
            );
        }
    }
}

// PinnedMemory handles cleanup in its Drop impl, no manual Drop needed here.

// ============================================================================
// Sharded Pool Wrapper
// ============================================================================

/// Wrapper that distributes allocations across multiple `PinnedMemoryPool` shards
/// using round-robin. Each shard has its own backing memory and allocator, reducing
/// lock contention under concurrent save workloads.
///
/// `PinnedAllocation` returned by `allocate` holds an `Arc` to the specific shard
/// it was allocated from, so `Drop` automatically frees back to the correct shard.
#[derive(Debug)]
pub(crate) struct ShardedPinnedPool {
    shards: Vec<Arc<PinnedMemoryPool>>,
    cursor: AtomicUsize,
}

impl ShardedPinnedPool {
    /// Create a sharded pool by splitting `total_capacity` evenly across `num_shards` pools.
    ///
    /// `node` is the NUMA node every shard is bound to. `NumaNode::UNKNOWN`
    /// disables NUMA pinning (placement falls back to caller-thread affinity).
    ///
    /// When `num_shards <= 1`, a single pool is created (equivalent to the old behavior).
    fn new(
        total_capacity: usize,
        num_shards: usize,
        use_hugepages: bool,
        unit_size_hint: Option<NonZeroU64>,
        node: NumaNode,
    ) -> Result<Self, String> {
        let num_shards = num_shards.max(1);
        let per_shard = total_capacity / num_shards;

        if num_shards > 1 {
            info!(
                "Creating sharded pinned pool on {}: total={}, shards={}, per_shard={}",
                node,
                ByteSize(total_capacity as u64),
                num_shards,
                ByteSize(per_shard as u64)
            );
        }

        let shards: Vec<Arc<PinnedMemoryPool>> = (0..num_shards)
            .map(|_| {
                PinnedMemoryPool::new(per_shard, use_hugepages, unit_size_hint, node).map(Arc::new)
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            shards,
            cursor: AtomicUsize::new(0),
        })
    }

    /// Allocate from shards using round-robin with fallback to all shards.
    fn allocate(&self, size: NonZeroU64) -> Option<PinnedAllocation> {
        let n = self.shards.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % n;

        for i in 0..n {
            let idx = (start + i) % n;
            if let Some(alloc) = self.shards[idx].allocate(size) {
                return Some(alloc);
            }
        }
        None
    }

    fn allocation_footprint(&self, size: NonZeroU64) -> Option<u64> {
        self.shards.first()?.allocation_footprint(size)
    }

    /// Aggregate usage across all shards.
    fn usage(&self) -> (u64, u64) {
        let mut used = 0u64;
        let mut total = 0u64;
        for shard in &self.shards {
            let (u, t) = shard.usage();
            used += u;
            total += t;
        }
        (used, total)
    }

    /// Largest contiguous free region across all shards.
    fn largest_free_allocation(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.largest_free_allocation())
            .max()
            .unwrap_or(0)
    }
}

// ============================================================================
// NUMA-Aware Pool Management
// ============================================================================

/// Manages multiple pinned memory pools, one per NUMA node.
///
/// Each pool is allocated on its respective NUMA node using first-touch policy
/// (memory is allocated from a thread pinned to that NUMA node).
///
/// This enables NUMA-local memory allocation for GPU workers, which is critical
/// for achieving optimal D2H/H2D transfer bandwidth on multi-socket systems.
///
/// # Note
/// This implementation does NOT provide fallback for unknown NUMA nodes.
/// If a GPU's NUMA affinity cannot be determined, the system should either:
/// - Disable NUMA-aware allocation (use global pool)
/// - Fail early during registration with a clear error
#[derive(Debug)]
pub(crate) struct NumaAwarePinnedPools {
    /// Per-NUMA sharded pools indexed by NUMA node ID
    pools: HashMap<u32, ShardedPinnedPool>,
}

impl NumaAwarePinnedPools {
    /// Create NUMA-aware pools, evenly distributing capacity across nodes.
    ///
    /// # Arguments
    /// * `total_capacity` - Total memory to allocate across all NUMA nodes
    /// * `numa_nodes` - List of NUMA nodes to create pools for
    /// * `use_hugepages` - Whether to use huge pages for allocation
    /// * `unit_size_hint` - Optional hint for allocator unit size
    ///
    /// # Behavior
    /// - Each NUMA node gets `total_capacity / num_nodes` bytes
    /// - Pools are allocated on threads pinned to their respective NUMA nodes
    ///
    /// Returns an error if any NUMA pool allocation or thread setup fails.
    fn new(
        total_capacity: usize,
        numa_nodes: &[NumaNode],
        num_shards: usize,
        use_hugepages: bool,
        unit_size_hint: Option<NonZeroU64>,
    ) -> Result<Self, String> {
        let num_nodes = numa_nodes.len();
        if num_nodes == 0 {
            warn!("No NUMA nodes provided, creating empty NumaAwarePinnedPools");
            return Ok(Self {
                pools: HashMap::new(),
            });
        }

        let per_node_capacity = total_capacity / num_nodes;
        info!(
            "Creating NUMA-aware pools: total={}, nodes={}, per_node={}, shards_per_node={}",
            ByteSize(total_capacity as u64),
            num_nodes,
            ByteSize(per_node_capacity as u64),
            num_shards
        );

        let mut pools = HashMap::new();

        for node in numa_nodes {
            if node.is_unknown() {
                continue;
            }

            let node_id = node.0;
            let hint = unit_size_hint;
            let target_node = *node;

            // Allocate sharded pool on a thread pinned to this NUMA node
            let result = run_on_numa(target_node, move || {
                ShardedPinnedPool::new(
                    per_node_capacity,
                    num_shards,
                    use_hugepages,
                    hint,
                    target_node,
                )
            });

            let pool =
                result.map_err(|error| format!("NUMA{node_id} pool initialization: {error}"))??;
            info!(
                "Created pinned pool on NUMA{}: capacity={}",
                node_id,
                ByteSize(per_node_capacity as u64)
            );
            pools.insert(node_id, pool);
        }

        Ok(Self { pools })
    }

    /// Allocate memory from the pool for a specific NUMA node.
    ///
    ///
    /// Returns `None` if:
    /// - `numa_node` is `UNKNOWN` (should be caught at registration time)
    /// - The NUMA node has no pool
    /// - The pool is exhausted
    fn allocate(&self, numa_node: NumaNode, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
        if numa_node.is_unknown() {
            error!("UNEXPECTED: allocate called with UNKNOWN NUMA node");
            return None;
        }

        self.pools.get(&numa_node.0)?.allocate(size).map(Arc::new)
    }

    fn allocation_footprint(&self, numa_node: NumaNode, size: NonZeroU64) -> Option<u64> {
        if numa_node.is_unknown() {
            return None;
        }
        self.pools.get(&numa_node.0)?.allocation_footprint(size)
    }

    /// Largest contiguous free region for a specific NUMA node.
    fn largest_free_allocation_for_node(&self, numa_node: NumaNode) -> u64 {
        if numa_node.is_unknown() {
            return 0;
        }
        self.pools
            .get(&numa_node.0)
            .map(|pool| pool.largest_free_allocation())
            .unwrap_or(0)
    }

    /// Get aggregate usage across all pools: (used_bytes, total_bytes)
    fn total_usage(&self) -> (u64, u64) {
        let mut used = 0u64;
        let mut total = 0u64;

        for pool in self.pools.values() {
            let (u, t) = pool.usage();
            used += u;
            total += t;
        }

        (used, total)
    }
}

// ============================================================================
// Unified Allocator Interface
// ============================================================================

/// Unified pinned memory allocator that hides NUMA details from callers.
///
/// This enum encapsulates both global and NUMA-aware allocation strategies,
/// providing a single interface for the rest of the system.
#[derive(Debug)]
pub(crate) enum PinnedAllocator {
    /// Global pool (NUMA disabled or single-node systems)
    Global(ShardedPinnedPool),
    /// NUMA-aware pools (multi-socket systems)
    Numa(NumaAwarePinnedPools),
}

impl PinnedAllocator {
    /// Create a new global allocator.
    pub(crate) fn new_global(
        capacity: usize,
        num_shards: usize,
        use_hugepages: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Result<Self, String> {
        Ok(Self::Global(ShardedPinnedPool::new(
            capacity,
            num_shards,
            use_hugepages,
            unit_hint,
            NumaNode::UNKNOWN,
        )?))
    }

    /// Create a new NUMA-aware allocator.
    ///
    /// If `numa_nodes` is empty, falls back to a global allocator.
    pub(crate) fn new_numa(
        capacity: usize,
        numa_nodes: &[NumaNode],
        num_shards: usize,
        use_hugepages: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Result<Self, String> {
        if numa_nodes.is_empty() {
            warn!(
                "NUMA allocator requested but no nodes provided, falling back to global allocator"
            );
            return Self::new_global(capacity, num_shards, use_hugepages, unit_hint);
        }
        Ok(Self::Numa(NumaAwarePinnedPools::new(
            capacity,
            numa_nodes,
            num_shards,
            use_hugepages,
            unit_hint,
        )?))
    }

    /// Allocate pinned memory.
    ///
    /// For global allocators, `numa_node` is ignored.
    /// For NUMA allocators, allocates from the specified node's pool.
    pub(crate) fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: NumaNode,
    ) -> Option<Arc<PinnedAllocation>> {
        match self {
            Self::Global(pool) => pool.allocate(size).map(Arc::new),
            Self::Numa(pools) => pools.allocate(numa_node, size),
        }
    }

    /// Exact rounded footprint an allocation would consume from its target pool.
    pub(crate) fn allocation_footprint(
        &self,
        size: NonZeroU64,
        numa_node: NumaNode,
    ) -> Option<u64> {
        match self {
            Self::Global(pool) => pool.allocation_footprint(size),
            Self::Numa(pools) => pools.allocation_footprint(numa_node, size),
        }
    }

    /// Get aggregate usage: (used_bytes, total_bytes)
    pub(crate) fn usage(&self) -> (u64, u64) {
        match self {
            Self::Global(pool) => pool.usage(),
            Self::Numa(pools) => pools.total_usage(),
        }
    }

    /// Largest contiguous free region for the requested allocation target.
    ///
    /// For global allocators, `numa_node` is ignored.
    /// For NUMA allocators, this checks only the target node's pool.
    pub(crate) fn largest_free_allocation_for_node(&self, numa_node: NumaNode) -> u64 {
        match self {
            Self::Global(pool) => pool.largest_free_allocation(),
            Self::Numa(pools) => pools.largest_free_allocation_for_node(numa_node),
        }
    }

    /// Check if this is a NUMA allocator.
    pub(crate) fn is_numa(&self) -> bool {
        matches!(self, Self::Numa(_))
    }

    pub(crate) fn payload_arenas(&self) -> std::io::Result<Vec<PayloadArena>> {
        let shards: Vec<_> = match self {
            Self::Global(pool) => pool.shards.iter().collect(),
            Self::Numa(pools) => pools.pools.values().flat_map(|pool| &pool.shards).collect(),
        };
        let mut arenas = shards
            .into_iter()
            .map(|shard| {
                Ok(PayloadArena {
                    id: shard.id,
                    size: shard.backing.size() as u64,
                    fd: shard.backing.export_fd()?,
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        arenas.sort_unstable_by_key(|arena| arena.id);
        Ok(arenas)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/memory/pool.rs"]
mod tests;
