//! SSD file intervals and GPU layout descriptors; engines move the bytes.
use crate::transfer_engine::{FileExtent, TransferSlice};
use std::ptr::NonNull;

pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const FILE_INFLIGHT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRange {
    pub file_offset: u64,
    pub device: u64,
    pub bytes: usize,
}

#[derive(Debug)]
pub(crate) struct IoBatch {
    pub file_offset: u64,
    pub bytes: usize,
    pub copies: Vec<CopyRange>,
}
impl IoBatch {
    pub(crate) fn into_extent(self) -> Result<FileExtent, String> {
        if !self.file_offset.is_multiple_of(ALIGNMENT as u64)
            || !self.bytes.is_multiple_of(ALIGNMENT)
        {
            return Err("SSD GPU extent must be 4 KiB aligned".into());
        }
        let slices = self
            .copies
            .into_iter()
            .map(|copy| {
                Ok(TransferSlice {
                    local: NonNull::new(copy.device as *mut u8).ok_or("null GPU range")?,
                    remote_address: copy.file_offset,
                    length: copy.bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        FileExtent::from_slices(self.file_offset, self.bytes, slices).map_err(|e| e.to_string())
    }
}

/// One vector per extent, with holes and aligned tails described as padding.
/// Repeated file ranges (fan-out) require another vector; neither engine ABI
/// can map the same file bytes to two destinations in one request.
pub(crate) fn plan_reads(mut copies: Vec<CopyRange>) -> Result<Vec<IoBatch>, String> {
    copies.sort_unstable_by_key(|copy| copy.file_offset);
    let mut batches: Vec<IoBatch> = Vec::new();
    for copy in copies {
        let end = copy
            .file_offset
            .checked_add(copy.bytes as u64)
            .and_then(|n| n.checked_next_multiple_of(ALIGNMENT as u64))
            .filter(|&n| i64::try_from(n).is_ok())
            .ok_or("SSD read overflow")?;
        if copy.bytes == 0
            || copy.device == 0
            || copy.device.checked_add(copy.bytes as u64).is_none()
        {
            return Err("invalid GPU read range".into());
        }
        if let Some(batch) = batches.iter_mut().find(|batch| {
            batch
                .copies
                .last()
                .is_some_and(|last| last.file_offset + last.bytes as u64 <= copy.file_offset)
        }) {
            batch.bytes =
                usize::try_from(end - batch.file_offset).map_err(|_| "SSD extent overflow")?;
            batch.copies.push(copy);
        } else {
            let start = copy.file_offset / ALIGNMENT as u64 * ALIGNMENT as u64;
            batches.push(IoBatch {
                file_offset: start,
                bytes: (end - start) as usize,
                copies: vec![copy],
            });
        }
    }
    Ok(batches)
}

pub(crate) fn plan_writes(
    file_offset: u64,
    bytes: u64,
    copies: Vec<CopyRange>,
) -> Result<Vec<IoBatch>, String> {
    let batch = IoBatch {
        file_offset,
        bytes: usize::try_from(bytes).map_err(|_| "SSD extent overflow")?,
        copies,
    };
    // Validate the whole vector before any engine submission or publication.
    IoBatch {
        file_offset,
        bytes: batch.bytes,
        copies: batch.copies.clone(),
    }
    .into_extent()?;
    Ok(vec![batch])
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/layout.rs"]
mod tests;
