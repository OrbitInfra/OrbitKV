use offset_allocator::{Allocation as RawAllocation, Allocator as RawAllocator};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    num::NonZeroU64,
};

/// Error variants returned by the scaled allocator wrapper.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AllocatorError {
    InvalidAllocationSize,
    /// The configured unit size must be greater than zero.
    InvalidUnitSize,
    /// The requested capacity cannot be represented with the underlying u32 allocator.
    CapacityTooLarge {
        total_bytes: u64,
        unit_size: u64,
    },
    /// The requested allocation is too large for the underlying allocator.
    RequestTooLarge {
        requested_bytes: u64,
        unit_size: u64,
    },
}

impl Display for AllocatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            AllocatorError::InvalidAllocationSize => {
                write!(f, "allocator returned an empty allocation")
            }
            AllocatorError::InvalidUnitSize => write!(f, "unit size must be greater than zero"),
            AllocatorError::CapacityTooLarge {
                total_bytes,
                unit_size,
            } => write!(
                f,
                "capacity {} bytes with unit {} overflows u32 backing allocator",
                total_bytes, unit_size
            ),
            AllocatorError::RequestTooLarge {
                requested_bytes,
                unit_size,
            } => write!(
                f,
                "request {} bytes with unit {} overflows u32 backing allocator",
                requested_bytes, unit_size
            ),
        }
    }
}

/// Represents an allocated region in bytes.
pub(crate) struct Allocation {
    /// Offset from the start of the managed region, in bytes.
    pub offset_bytes: u64,
    /// Actual size of the allocation in bytes (rounded up to `unit_size`).
    pub size_bytes: NonZeroU64,
    raw: RawAllocation,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocation")
            .field("offset_bytes", &self.offset_bytes)
            .field("size_bytes", &self.size_bytes)
            .finish()
    }
}

/// A thin scaling wrapper around `offset-allocator` that allows byte-level APIs
/// while the underlying allocator works with u32-sized units.
#[derive(Debug)]
pub(crate) struct ScaledOffsetAllocator {
    unit_size: NonZeroU64,
    total_units: u32,
    inner: RawAllocator,
}

impl ScaledOffsetAllocator {
    /// Create a new allocator with explicit `unit_size` and `max_allocs`.
    pub(crate) fn new_with_unit_size_and_max_allocs(
        total_bytes: u64,
        unit_size: u64,
        max_allocs: u32,
    ) -> Result<Self, AllocatorError> {
        let unit_size = NonZeroU64::new(unit_size).ok_or(AllocatorError::InvalidUnitSize)?;

        let total_units = total_bytes / unit_size.get();
        let total_units_u32 =
            u32::try_from(total_units).map_err(|_| AllocatorError::CapacityTooLarge {
                total_bytes,
                unit_size: unit_size.get(),
            })?;

        Ok(Self {
            unit_size,
            total_units: total_units_u32,
            inner: RawAllocator::with_max_allocs(total_units_u32, max_allocs),
        })
    }

    /// Allocate `size_bytes`, rounded up to the allocator's unit size.
    pub(crate) fn allocate(
        &mut self,
        size_bytes: u64,
    ) -> Result<Option<Allocation>, AllocatorError> {
        if size_bytes == 0 {
            return Ok(None);
        }

        let units_u32 = self.allocation_units(size_bytes)?;

        let Some(raw) = self.inner.allocate(units_u32) else {
            return Ok(None);
        };

        let allocated_units = self.inner.allocation_size(raw) as u64;
        let Some(size_bytes) = NonZeroU64::new(allocated_units * self.unit_size.get()) else {
            self.inner.free(raw);
            return Err(AllocatorError::InvalidAllocationSize);
        };
        Ok(Some(Allocation {
            offset_bytes: raw.offset as u64 * self.unit_size.get(),
            size_bytes,
            raw,
        }))
    }

    /// Exact allocator footprint for a request, without mutating allocator state.
    pub(crate) fn allocation_footprint(&self, size_bytes: u64) -> Result<u64, AllocatorError> {
        let units = self.allocation_units(size_bytes)?;
        Ok(units as u64 * self.unit_size.get())
    }

    fn allocation_units(&self, size_bytes: u64) -> Result<u32, AllocatorError> {
        let units = size_bytes.div_ceil(self.unit_size.get());
        u32::try_from(units).map_err(|_| AllocatorError::RequestTooLarge {
            requested_bytes: size_bytes,
            unit_size: self.unit_size.get(),
        })
    }

    /// Free a previously allocated region.
    pub(crate) fn free(&mut self, allocation: &Allocation) {
        self.inner.free(allocation.raw);
    }

    /// Return the total managed capacity in bytes (rounded down to `unit_size`).
    pub(crate) fn total_bytes(&self) -> u64 {
        self.unit_size.get() * self.total_units as u64
    }

    /// Return a storage report converted to bytes.
    pub(crate) fn storage_report(&self) -> StorageReportBytes {
        let report = self.inner.storage_report();
        StorageReportBytes {
            total_free_bytes: report.total_free_space as u64 * self.unit_size.get(),
            largest_free_allocation_bytes: report.largest_free_region as u64 * self.unit_size.get(),
        }
    }
}

/// Storage report expressed in bytes.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct StorageReportBytes {
    pub total_free_bytes: u64,
    pub largest_free_allocation_bytes: u64,
}

#[cfg(test)]
#[path = "../../tests/unit/memory/allocator.rs"]
mod tests;
