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
use crate::transfer_engine::{CudaRegistration, PeerTransferError, TransferSlice};
use cudarc::driver::DevicePtr;
use std::{collections::VecDeque, ptr::NonNull, sync::Arc};

#[derive(Clone)]
struct Copy<'a> {
    source: &'a PeerRestoreBlock,
    remote: u64,
    target: u64,
    bytes: usize,
    encoded: Option<&'a EncodedSegment>,
    owner: Arc<CudaRegistration>,
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
                        && meta.stored_bytes > (budget / 2).min(BATCH_BYTES)
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
    for copy in &plan {
        if !copy.owner.covers(copy.target, copy.bytes) {
            return Err("peer GPU destination exceeds its registered tensor".into());
        }
    }
    Ok(plan)
}

pub(super) fn validate(layers: &[LayerTransferData], budget: usize) -> Result<(), String> {
    plan(layers, budget).map(|_| ())
}

const BATCH_BYTES: usize = 64 << 20;
const BATCH_RANGES: usize = 16_384;

struct Batch<'a> {
    source: &'a PeerRestoreBlock,
    slices: Vec<TransferSlice>,
    owners: Vec<Arc<CudaRegistration>>,
    decodes: Vec<DecodeInput<'a>>,
    bytes: usize,
}
fn batch<'a>(
    pending: &mut VecDeque<Copy<'a>>,
    input: Option<&Arc<CudaRegistration>>,
) -> Result<Batch<'a>, String> {
    let source = pending.front().ok_or("empty peer restore")?.source;
    let mut batch = Batch {
        source,
        slices: Vec::new(),
        owners: Vec::new(),
        decodes: Vec::new(),
        bytes: 0,
    };
    let mut wire_bytes = 0;
    let mut input_offset = 0;
    while let Some(copy) = pending.front() {
        if !source.same_grant(copy.source)
            || batch.slices.len() == BATCH_RANGES
            || wire_bytes == BATCH_BYTES
        {
            break;
        }
        let mut copy = copy.clone();
        let (target, length, owner) = if let Some(meta) = copy.encoded {
            let input = input.ok_or("missing persistent codec input pool")?;
            let offset = (input_offset + 7) & !7;
            if meta.stored_bytes > input.length || meta.stored_bytes > BATCH_BYTES {
                return Err("peer codec segment exceeds the input or engine batch budget".into());
            }
            if offset + meta.stored_bytes > input.length
                || wire_bytes + meta.stored_bytes > BATCH_BYTES
            {
                break;
            }
            let target = input.address + offset as u64;
            input_offset = offset + meta.stored_bytes;
            batch.decodes.push(DecodeInput {
                source: target,
                source_bytes: meta.stored_bytes,
                target: copy.target,
                target_bytes: copy.bytes,
                meta,
            });
            (target, meta.stored_bytes, Arc::clone(input))
        } else {
            (
                copy.target,
                copy.bytes.min(BATCH_BYTES - wire_bytes),
                Arc::clone(&copy.owner),
            )
        };
        pending.pop_front();
        let remote_address = copy.remote;
        if copy.encoded.is_none() && length < copy.bytes {
            copy.remote += length as u64;
            copy.target += length as u64;
            copy.bytes -= length;
            pending.push_front(copy.clone());
            copy.bytes = length;
        }
        batch.slices.push(TransferSlice {
            local: NonNull::new(target as *mut u8).ok_or("invalid GPU range")?,
            remote_address,
            length,
        });
        batch.owners.push(owner);
        wire_bytes += length;
        batch.bytes += copy.bytes;
    }
    Ok(batch)
}

pub(super) fn restore(
    runtime: &WorkerRuntime,
    layers: &[LayerTransferData],
    budget: usize,
) -> Result<usize, PeerTransferError> {
    let fail = PeerTransferError::Failed;
    let mut copies = plan(layers, budget).map_err(fail)?;
    copies.sort_by_key(|copy| copy.source.grant_id());
    if copies
        .iter()
        .any(|copy| copy.owner.device != runtime.stream.context().ordinal() as i32)
    {
        return Err(fail(
            "peer destination is on a different CUDA device".into(),
        ));
    }
    if let Some(copy) = copies.iter().find(|copy| copy.encoded.is_some()) {
        let mut input = runtime.peer_input.borrow_mut();
        if input.is_none() {
            runtime.codec.borrow_mut().take();
            let capacity = (budget / 2).min(BATCH_BYTES);
            // SAFETY: no codec reads this pool before engine completion.
            let allocation = Arc::new(
                unsafe { runtime.stream.alloc::<u8>(capacity) }.map_err(|e| fail(e.to_string()))?,
            );
            crate::transfer::finish_gpu_transfer(&runtime.stream, Ok(()))
                .map_err(|e| fail(e.to_string()))?;
            let (address, _record) = allocation.device_ptr(&runtime.stream);
            // SAFETY: the pool owns its allocation and CUDA context through unregister.
            *input = Some(unsafe {
                CudaRegistration::new(
                    Some(copy.source.backend()),
                    address,
                    capacity,
                    copy.owner.device,
                    allocation.clone(),
                )
            }?);
        }
    }
    let input = runtime.peer_input.borrow();
    if input
        .as_ref()
        .is_some_and(|input| input.length > budget / 2)
    {
        return Err(fail(
            "codec budget is below the persistent input pool".into(),
        ));
    }
    let mut pending = VecDeque::from(copies);
    let mut bytes = 0;
    while !pending.is_empty() {
        let batch = batch(&mut pending, input.as_ref()).map_err(fail)?;
        // SAFETY: this batch owns registered destinations and source authority through drain.
        unsafe { batch.source.read_gpu(&batch.slices, &batch.owners) }?;
        if !batch.decodes.is_empty() {
            let mut codec = runtime.codec.borrow_mut();
            if codec.is_none() {
                *codec = Some(
                    GpuCodec::new(runtime.stream.context())
                        .map_err(fail)?
                        .with_transfer(runtime.engine.clone()),
                );
            }
            for decode in &batch.decodes {
                // SAFETY: engine-complete input and actual targets outlive the codec drain.
                unsafe {
                    codec.as_mut().expect("codec initialized").decode_batch(
                        &runtime.stream,
                        std::slice::from_ref(decode),
                        budget / 2,
                    )
                }
                .map_err(|e| fail(e.to_string()))?;
            }
        }
        bytes += batch.bytes;
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/peer.rs"]
mod tests;
