use std::collections::HashMap;
use std::sync::Arc;

mod decode;
mod queue;
pub(super) use queue::{MAX_WRITES, run};

use orbitkv_state::StorageFormat;

use crate::codec::{EncodedSegment, segment_format};
use crate::storage::ssd::GpuFile;
use crate::storage::ssd::layout::{CopyRange, IoBatch, plan_reads};
use crate::storage::ssd::{SsdReadLease, WriteLease};
use crate::transfer::layout::BlockCopies;
use crate::{EngineError, SlotMeta};

use super::{LayerTransferData, TransferPayload};

type FileReads = (Arc<GpuFile>, Vec<IoBatch>);

pub(super) struct EncodedRead {
    source: Arc<SsdReadLease>,
    file_offset: u64,
    target: u64,
    meta: EncodedSegment,
}

#[derive(Default)]
pub(super) struct ReadPlan {
    raw: Vec<FileReads>,
    encoded: Vec<EncodedRead>,
    owners: Vec<Arc<SsdReadLease>>,
}

pub(crate) struct GpuWrite {
    pub lease: Arc<WriteLease>,
    pub batches: Vec<IoBatch>,
}

/// Merge validated ranges per immutable extent. The worker retains the whole task and every
/// extent lease separately until all submitted reads and scatters complete.
pub(super) fn plan(layers: &[LayerTransferData]) -> Result<ReadPlan, EngineError> {
    let mut plan = ReadPlan::default();
    let mut sources: HashMap<usize, (Arc<GpuFile>, Vec<CopyRange>)> = HashMap::new();
    for layer in layers {
        for block in &layer.blocks {
            let TransferPayload::Ssd {
                source,
                slot_id,
                offset,
                ..
            } = &block.block
            else {
                continue;
            };
            plan.owners.push(Arc::clone(source));
            let (slot, base) = source_slot(source, *slot_id)?;
            let copies = layer
                .layout
                .block_copies(block.block_idx)
                .map_err(EngineError::Storage)?;
            if slot.encoding.is_some() {
                for (copy, meta) in
                    encoded_copies(slot, base, *offset, copies, layer.layout.storage_format)?
                {
                    plan.encoded.push(EncodedRead {
                        source: Arc::clone(source),
                        file_offset: copy.file_offset,
                        target: copy.device,
                        meta,
                    });
                }
                continue;
            }
            let file = source.file()?;
            let reads = &mut sources
                .entry(Arc::as_ptr(&source.entry.readers) as usize)
                .or_insert_with(|| (Arc::clone(file), Vec::new()))
                .1;
            let (first, second) = raw_copies(slot, base, *offset, copies)?;
            reads.push(first);
            reads.extend(second);
        }
    }
    plan.raw = sources
        .into_values()
        .map(|(file, copies)| {
            plan_reads(copies)
                .map(|batches| (file, batches))
                .map_err(EngineError::Storage)
        })
        .collect::<Result<_, _>>()?;
    // Keep corruption attribution to one immutable generation per decode batch.
    plan.encoded.sort_by_key(|read| {
        (
            Arc::as_ptr(&read.source.entry.readers) as usize,
            read.file_offset,
        )
    });
    Ok(plan)
}

fn source_slot(source: &SsdReadLease, slot_id: usize) -> Result<(&SlotMeta, u64), EngineError> {
    let slot = source
        .entry
        .slots
        .get(slot_id)
        .ok_or_else(|| EngineError::Storage("SSD slot is missing".into()))?;
    let base = source.entry.slots[..slot_id]
        .iter()
        .try_fold(source.entry.file_offset, |base, slot| {
            base.checked_add(slot.total_size())
        })
        .ok_or_else(|| EngineError::Storage("SSD slot offset overflow".into()))?;
    validate_slot(slot)
        .and_then(|()| {
            if base.checked_add(slot.total_size()).is_none_or(|end| {
                source
                    .entry
                    .file_offset
                    .checked_add(source.entry.len)
                    .is_none_or(|limit| end > limit)
            }) {
                return Err("SSD slot exceeds its leased extent".into());
            }
            Ok(())
        })
        .inspect_err(|_| source.invalidate_encoded())
        .map_err(EngineError::Storage)?;
    Ok((slot, base))
}

fn raw_copies(
    slot: &SlotMeta,
    base: u64,
    offset: usize,
    copies: BlockCopies,
) -> Result<(CopyRange, Option<CopyRange>), EngineError> {
    let range = |segment, relative, copy: crate::transfer::layout::BlockCopy| {
        Ok(CopyRange {
            file_offset: segment_offset(slot, base, segment, relative, copy.bytes)?,
            device: copy.addr,
            bytes: copy.bytes,
        })
    };
    match copies {
        BlockCopies::Contiguous(copy) => Ok((range(0, offset, copy)?, None)),
        BlockCopies::Split { k, v } => {
            let second = if slot.num_segments() > 1 {
                range(1, offset, v)?
            } else {
                range(
                    0,
                    offset
                        .checked_add(k.bytes)
                        .ok_or_else(|| EngineError::Storage("SSD slot offset overflow".into()))?,
                    v,
                )?
            };
            Ok((range(0, offset, k)?, Some(second)))
        }
    }
}

pub(super) fn validate_host_sources(layers: &[LayerTransferData]) -> Result<(), EngineError> {
    for layer in layers {
        for block in &layer.blocks {
            if let TransferPayload::Ssd {
                source,
                slot_id,
                offset,
                ..
            } = &block.block
            {
                let (slot, base) = source_slot(source, *slot_id)?;
                let copies = layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?;
                if slot.encoding.is_some() {
                    encoded_copies(slot, base, *offset, copies, layer.layout.storage_format)?;
                } else {
                    raw_copies(slot, base, *offset, copies)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_slot(slot: &SlotMeta) -> Result<(), String> {
    if slot.segment_sizes.is_empty()
        || slot.segment_sizes.contains(&0)
        || slot
            .segment_sizes
            .iter()
            .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
            != Some(slot.total_size())
    {
        return Err("invalid SSD slot bounds".into());
    }
    if let Some(metadata) = &slot.encoding {
        if metadata.len() != slot.num_segments() {
            return Err("encoded SSD segment count mismatch".into());
        }
        for (meta, &physical) in metadata.iter().zip(&slot.segment_sizes) {
            meta.validate_metadata(usize::try_from(physical).map_err(|_| "SSD segment overflow")?)?;
        }
    }
    Ok(())
}

fn encoded_copies(
    slot: &SlotMeta,
    base: u64,
    offset: usize,
    copies: BlockCopies,
    format: StorageFormat,
) -> Result<Vec<(CopyRange, EncodedSegment)>, EngineError> {
    let ranges = match copies {
        BlockCopies::Contiguous(copy) => vec![copy],
        BlockCopies::Split { k, v } => vec![k, v],
    };
    let metadata = slot
        .encoding
        .as_ref()
        .ok_or_else(|| EngineError::Storage("encoded SSD metadata is missing".into()))?;
    if offset != 0 || metadata.len() != ranges.len() {
        return Err(EngineError::Storage(
            "encoded SSD restore layout mismatch".into(),
        ));
    }
    metadata
        .iter()
        .zip(ranges)
        .enumerate()
        .map(|(index, (meta, copy))| {
            if meta.logical_bytes != copy.bytes
                || (meta.format != StorageFormat::Exact
                    && meta.format != segment_format(format, index))
            {
                return Err(EngineError::Storage(
                    "encoded SSD restore representation mismatch".into(),
                ));
            }
            let file_offset = segment_offset(slot, base, index, 0, meta.stored_bytes)?;
            Ok((
                CopyRange {
                    file_offset,
                    device: copy.addr,
                    bytes: meta.stored_bytes,
                },
                meta.clone(),
            ))
        })
        .collect()
}

fn segment_offset(
    slot: &SlotMeta,
    base: u64,
    segment: usize,
    offset: usize,
    bytes: usize,
) -> Result<u64, EngineError> {
    let size = slot
        .segment_sizes
        .get(segment)
        .copied()
        .ok_or_else(|| EngineError::Storage("SSD segment missing".into()))?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end as u64 > size)
    {
        return Err(EngineError::Storage(
            "SSD segment is smaller than the registered GPU layout".into(),
        ));
    }
    slot.segment_sizes[..segment]
        .iter()
        .try_fold(base, |base, &bytes| base.checked_add(bytes))
        .and_then(|start| start.checked_add(offset as u64))
        .ok_or_else(|| EngineError::Storage("SSD segment offset overflow".into()))
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/ssd.rs"]
mod tests;
