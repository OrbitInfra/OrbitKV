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

#[allow(
    clippy::unnecessary_wraps,
    reason = "the adapter contract is fallible when OrbiTE registrations are compiled"
)]
pub(super) fn pin(backing: &Arc<PinnedMemory>) -> Result<PoolRegistration, String> {
    #[cfg(feature = "orbite")]
    {
        // SAFETY: backing owns the physical CUDA pin until the last local loan.
        unsafe {
            orbite::cuda::local_copy::PinnedHostRegistration::from_raw_parts(
                backing.as_non_null(),
                backing.size(),
                backing.clone(),
            )
        }
        .map_err(|error| error.to_string())
    }
    #[cfg(not(feature = "orbite"))]
    {
        let _ = backing;
        Ok(PoolRegistration)
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

#[cfg(all(test, feature = "orbite"))]
#[path = "../../../tests/unit/memory/orbite.rs"]
mod orbite_tests;
