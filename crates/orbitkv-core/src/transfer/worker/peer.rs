//! Remote bytes land in registered GPU targets or GPU codec input allocations.
//! Transport staging and GPU Direct selection belong entirely to the engine.

use super::{LayerTransferData, TransferPayload, WorkerRuntime};
use crate::codec::{
    EncodedSegment,
    gpu::{DecodeInput, GpuCodec},
    segment_format,
};
use crate::peer::restore::PeerRestoreBlock;
use crate::transfer::layout::BlockCopies;
use crate::transfer_engine::PeerTransferError;
use cudarc::driver::DevicePtr;
use std::sync::Arc;

struct Copy<'a> {
    source: &'a PeerRestoreBlock,
    remote: u64,
    target: u64,
    bytes: usize,
    encoded: Option<&'a EncodedSegment>,
    owner: Arc<dyn Send + Sync>,
}

fn plan(layers: &[LayerTransferData], budget: usize) -> Result<Vec<Copy<'_>>, String> {
    let mut plan = Vec::new();
    for layer in layers {
        for block in &layer.blocks {
            let TransferPayload::Peer {
                source,
                slot_id,
                offset,
            } = &block.block
            else {
                continue;
            };
            let owner = layer
                .layout
                .owner
                .as_ref()
                .ok_or("peer GPU restore requires an owned CUDA registration")?;
            let slot = source.slots.get(*slot_id).ok_or("peer slot is missing")?;
            let copies = match layer.layout.block_copies(block.block_idx)? {
                BlockCopies::Contiguous(copy) => vec![copy],
                BlockCopies::Split { k, v } => vec![k, v],
            };
            if let Some(encoding) = &slot.meta.encoding {
                if *offset != 0 || encoding.len() != copies.len() {
                    return Err("peer codec layout mismatch".into());
                }
                for (index, (meta, copy)) in encoding.iter().zip(copies).enumerate() {
                    if meta.logical_bytes != copy.bytes
                        || (meta.format != orbitkv_state::StorageFormat::Exact
                            && meta.format != segment_format(layer.layout.storage_format, index))
                    {
                        return Err("peer codec representation mismatch".into());
                    }
                    // Codec inputs are compute owners, not transport staging.
                    if meta.format != orbitkv_state::StorageFormat::Exact
                        && meta.stored_bytes > budget / 2
                    {
                        return Err("peer codec input exceeds its GPU budget".into());
                    }
                    plan.push(Copy {
                        source,
                        remote: slot.range(index, 0, meta.stored_bytes)?,
                        target: copy.addr,
                        bytes: copy.bytes,
                        encoded: (meta.format != orbitkv_state::StorageFormat::Exact)
                            .then_some(meta),
                        owner: Arc::clone(owner),
                    });
                }
            } else {
                let mut packed_offset = *offset;
                for (index, copy) in copies.into_iter().enumerate() {
                    let (segment, relative) = if slot.meta.num_segments() > 1 {
                        (index, *offset)
                    } else {
                        (0, packed_offset)
                    };
                    plan.push(Copy {
                        source,
                        remote: slot.range(segment, relative, copy.bytes)?,
                        target: copy.addr,
                        bytes: copy.bytes,
                        encoded: None,
                        owner: Arc::clone(owner),
                    });
                    packed_offset = packed_offset
                        .checked_add(copy.bytes)
                        .ok_or("peer layout offset overflow")?;
                }
            }
        }
    }
    Ok(plan)
}

pub(super) fn validate(layers: &[LayerTransferData], budget: usize) -> Result<(), String> {
    plan(layers, budget).map(|_| ())
}

pub(super) fn restore(
    runtime: &WorkerRuntime,
    layers: &[LayerTransferData],
    budget: usize,
) -> Result<usize, PeerTransferError> {
    let fail = PeerTransferError::Failed;
    let copies = plan(layers, budget).map_err(fail)?;
    let stream = &runtime.stream;
    let mut codec = None;
    let mut bytes = 0usize;
    for copy in copies {
        let owner: Arc<dyn Send + Sync> = Arc::new((copy.owner, Arc::clone(stream.context())));
        if let Some(meta) = copy.encoded {
            if codec.is_none() {
                codec = Some(GpuCodec::new(stream.context()).map_err(fail)?);
            }
            // SAFETY: the engine fills the entire input before codec reads it.
            let input = Arc::new(
                unsafe { stream.alloc::<u8>(meta.stored_bytes) }
                    .map_err(|e| fail(e.to_string()))?,
            );
            // Make allocator/initialization work ready before an independent engine writes.
            crate::transfer::finish_gpu_transfer(stream, Ok(()))
                .map_err(|e| fail(e.to_string()))?;
            let (address, _record) = input.device_ptr(stream);
            let input_owner: Arc<dyn Send + Sync> = Arc::clone(&input) as Arc<dyn Send + Sync>;
            // SAFETY: actual codec input and context are retained by registration; the
            // layout owner keeps the disjoint target through synchronous codec drain.
            unsafe {
                copy.source.read_gpu(
                    copy.remote,
                    address,
                    meta.stored_bytes,
                    stream.context().ordinal() as i32,
                    input_owner,
                )?;
            }
            // SAFETY: engine completion proves input readiness; decode_batch drains
            // every accepted CUDA operation before releasing input/target owners.
            unsafe {
                codec.as_mut().expect("codec initialized").decode_batch(
                    stream,
                    &[DecodeInput {
                        source: address,
                        source_bytes: meta.stored_bytes,
                        target: copy.target,
                        target_bytes: copy.bytes,
                        meta,
                    }],
                    budget / 2,
                )
            }
            .map_err(|e| fail(e.to_string()))?;
            drop(owner);
        } else {
            // SAFETY: the admitted restore exclusively owns these validated pages;
            // the actual imported tensor and context accompany the native registration.
            unsafe {
                copy.source.read_gpu(
                    copy.remote,
                    copy.target,
                    copy.bytes,
                    stream.context().ordinal() as i32,
                    owner,
                )?;
            }
        }
        bytes = bytes
            .checked_add(copy.bytes)
            .ok_or_else(|| fail("peer restore byte count overflow".into()))?;
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/peer.rs"]
mod tests;
