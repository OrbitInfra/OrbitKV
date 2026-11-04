//! SSD file intervals and GPU layout descriptors; no native I/O.

pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const STAGING_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const STAGING_SLOTS: usize = 2;

/// One file interval and the corresponding engine-owned GPU range.
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
    pub(super) fn validate(&self) -> Result<(), String> {
        let end = self
            .file_offset
            .checked_add(self.bytes as u64)
            .filter(|&end| i64::try_from(end).is_ok())
            .ok_or("SSD batch offset overflow")?;
        if self.bytes == 0
            || self.bytes > STAGING_BYTES
            || !self.bytes.is_multiple_of(ALIGNMENT)
            || !self.file_offset.is_multiple_of(ALIGNMENT as u64)
        {
            return Err("SSD batch exceeds aligned GPU staging".into());
        }
        for copy in &self.copies {
            if copy.device == 0
                || copy.file_offset < self.file_offset
                || copy
                    .file_offset
                    .checked_add(copy.bytes as u64)
                    .is_none_or(|limit| limit > end)
                || copy.device.checked_add(copy.bytes as u64).is_none()
            {
                return Err("SSD batch copy exceeds its owned ranges".into());
            }
        }
        Ok(())
    }
}

/// Merge adjacent aligned intervals, splitting large state checkpoints to keep
/// staging bounded. Read amplification is restricted to the aligned edges.
pub(crate) fn plan_reads(mut copies: Vec<CopyRange>) -> Result<Vec<IoBatch>, String> {
    copies.sort_unstable_by_key(|copy| copy.file_offset);
    let mut batches: Vec<IoBatch> = Vec::new();
    for mut copy in copies {
        copy.file_offset
            .checked_add(copy.bytes as u64)
            .filter(|&end| i64::try_from(end).is_ok())
            .ok_or("SSD read offset overflow")?;
        copy.device
            .checked_add(copy.bytes as u64)
            .ok_or("GPU destination overflow")?;
        while copy.bytes != 0 {
            let aligned = copy.file_offset / ALIGNMENT as u64 * ALIGNMENT as u64;
            let append = batches.last().is_some_and(|batch| {
                copy.file_offset >= batch.file_offset
                    && aligned <= batch.file_offset + batch.bytes as u64
                    && copy.file_offset < batch.file_offset + STAGING_BYTES as u64
            });
            if !append {
                batches.push(IoBatch {
                    file_offset: aligned,
                    bytes: 0,
                    copies: Vec::new(),
                });
            }
            let batch = batches.last_mut().expect("batch inserted");
            let offset = (copy.file_offset - batch.file_offset) as usize;
            let bytes = copy.bytes.min(STAGING_BYTES - offset);
            batch.bytes = batch
                .bytes
                .max((offset + bytes).next_multiple_of(ALIGNMENT));
            batch.copies.push(CopyRange { bytes, ..copy });
            copy.file_offset += bytes as u64;
            copy.device += bytes as u64;
            copy.bytes -= bytes;
        }
    }
    Ok(batches)
}

/// A complete object owns its aligned extent, including padding. Gather only
/// its valid GPU bytes and zero padding before any chunk becomes visible.
pub(crate) fn plan_writes(
    file_offset: u64,
    bytes: u64,
    copies: Vec<CopyRange>,
) -> Result<Vec<IoBatch>, String> {
    let end = file_offset
        .checked_add(bytes)
        .filter(|&end| i64::try_from(end).is_ok())
        .ok_or("SSD write offset overflow")?;
    if !file_offset.is_multiple_of(ALIGNMENT as u64)
        || !bytes.is_multiple_of(ALIGNMENT as u64)
        || bytes == 0
    {
        return Err("SSD writes require an aligned, nonempty reservation".into());
    }
    for copy in &copies {
        if copy.file_offset < file_offset
            || copy
                .file_offset
                .checked_add(copy.bytes as u64)
                .is_none_or(|limit| limit > end)
            || copy.device.checked_add(copy.bytes as u64).is_none()
        {
            return Err("GPU write range exceeds its reservation".into());
        }
    }
    let mut batches = Vec::new();
    let mut start = file_offset;
    while start < end {
        let limit = end.min(start + STAGING_BYTES as u64);
        let pieces = copies
            .iter()
            .filter_map(|copy| {
                let begin = copy.file_offset.max(start);
                let finish = (copy.file_offset + copy.bytes as u64).min(limit);
                (begin < finish).then(|| CopyRange {
                    file_offset: begin,
                    device: copy.device + (begin - copy.file_offset),
                    bytes: (finish - begin) as usize,
                })
            })
            .collect();
        batches.push(IoBatch {
            file_offset: start,
            bytes: (limit - start) as usize,
            copies: pieces,
        });
        start = limit;
    }
    Ok(batches)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/layout.rs"]
mod tests;
