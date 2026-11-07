//! Native registrations sharing each pinned allocation and physical pool pin.
use super::*;

#[cfg(feature = "orbite")]
pub(super) type MemoryRegistration = Mutex<std::sync::Weak<orbite::HostMemory>>;
#[cfg(not(feature = "orbite"))]
#[derive(Default)]
pub(super) struct MemoryRegistration;
#[cfg(feature = "orbite")]
pub(super) type PoolRegistration = Arc<orbite::cuda::local_copy::PinnedHostRegistration>;
#[cfg(not(feature = "orbite"))]
pub(super) struct PoolRegistration;

pub(super) fn pin(backing: &Arc<PinnedMemory>) -> PoolRegistration {
    #[cfg(feature = "orbite")]
    {
        // SAFETY: backing owns the physical CUDA pin until the last local loan.
        unsafe {
            orbite::cuda::local_copy::PinnedHostRegistration::from_raw_parts(
                NonNull::new(backing.as_ptr().cast_mut()).expect("pinned base"),
                backing.size(),
                backing.clone(),
            )
        }
        .expect("valid pinned pool")
    }
    #[cfg(not(feature = "orbite"))]
    {
        let _ = backing;
        PoolRegistration
    }
}

#[cfg(feature = "orbite")]
impl PinnedAllocation {
    /// Shares this allocation and its access lock with OrbiTE registrations.
    /// Retaining the returned memory keeps this exact allocator slot occupied,
    /// even after the allocator, registration or batch handle is dropped.
    ///
    /// # Safety
    /// Until the last returned memory, registration and admitted transfer releases
    /// this allocation, coordinate all external CPU, CUDA and process accesses
    /// with OrbiTE. No conflicting Rust references or unsynchronized device
    /// accesses may remain. All OrbiTE aliases must be obtained through this
    /// method so they share the same access lock.
    pub unsafe fn orbite_memory(self: &Arc<Self>) -> orbite::Result<Arc<orbite::HostMemory>> {
        let mut cached = self._engine_memory.lock();
        if let Some(memory) = cached.upgrade() {
            return Ok(memory);
        }
        let length = usize::try_from(self.size_bytes()).map_err(|_| orbite::Error::InvalidRange)?;
        // SAFETY: this Arc retains the actual allocation and its pinned pool;
        // its Drop is the only path returning this slot to the allocator. The
        // caller coordinates accesses outside the shared OrbiTE lock.
        let memory = Arc::new(unsafe {
            orbite::HostMemory::from_cuda_registered_parts(
                self.ptr,
                length,
                self.clone(),
                self.pool._local_pin.clone(),
            )?
        });
        *cached = Arc::downgrade(&memory);
        Ok(memory)
    }
}

#[cfg(feature = "mooncake")]
impl PinnedMemoryPool {
    /// Return the base pointer and length of the backing pinned memory.
    fn memory_region(&self) -> (NonNull<u8>, usize) {
        let ptr = NonNull::new(self.backing.as_ptr() as *mut u8)
            .expect("PinnedMemoryPool backing pointer is null");
        (ptr, self.backing.size())
    }
}

#[cfg(feature = "mooncake")]
impl ShardedPinnedPool {
    /// Return all backing memory regions (one per shard).
    fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        self.shards.iter().map(|s| s.memory_region()).collect()
    }
}

#[cfg(feature = "mooncake")]
impl NumaAwarePinnedPools {
    /// Return all backing memory regions across all NUMA nodes and shards.
    fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        self.pools
            .values()
            .flat_map(|p| p.memory_regions())
            .collect()
    }
}

#[cfg(feature = "mooncake")]
impl PinnedAllocator {
    /// Return all backing memory regions as `(ptr, len)` pairs.
    pub(crate) fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        match self {
            Self::Global(pool) => pool.memory_regions(),
            Self::Numa(pools) => pools.memory_regions(),
        }
    }
}
#[cfg(all(test, feature = "orbite"))]
#[path = "../../../tests/unit/memory/orbite.rs"]
mod orbite_tests;
