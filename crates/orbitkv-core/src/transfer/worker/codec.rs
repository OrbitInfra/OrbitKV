use std::{
    num::NonZeroU64,
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::result;
use opentelemetry::KeyValue;
use orbitkv_state::StorageFormat;

use super::{LayerTransferData, TransferPayload, WorkerRuntime};
use crate::{
    EngineError, SlotMeta,
    block::{RawBlock, Segment, StateKey},
    codec::{
        EncodedSegment,
        gpu::{EncodeInput, GpuCodec, HostDecodeInput},
        segment_format,
    },
    cost::Observation,
    memory::numa::NumaNode,
    metrics::core_metrics,
    storage::Storage,
    storage::ssd::cufile::{CopyRange, GpuSlot, plan_writes},
    transfer::{finish_gpu_transfer, layout::BlockCopies},
};

/// A complete independently restorable object, with blocks in storage-slot order.
pub(crate) struct SaveGroup {
    pub key: StateKey,
    pub blocks: Vec<(usize, usize)>,
}

struct Reservation {
    bytes: usize,
    start: Instant,
    operation: &'static str,
}
impl Reservation {
    fn new(bytes: usize, operation: &'static str) -> Self {
        core_metrics()
            .storage_codec_reserved_bytes
            .add(bytes as i64, &[]);
        Self {
            bytes,
            start: Instant::now(),
            operation,
        }
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        core_metrics()
            .storage_codec_reserved_bytes
            .add(-(self.bytes as i64), &[]);
        core_metrics().storage_codec_seconds.record(
            self.start.elapsed().as_secs_f64(),
            &[KeyValue::new("operation", self.operation)],
        );
    }
}

fn ranges(copies: BlockCopies) -> Vec<(u64, usize)> {
    match copies {
        BlockCopies::Contiguous(c) => vec![(c.addr, c.bytes)],
        BlockCopies::Split { k, v } => vec![(k.addr, k.bytes), (v.addr, v.bytes)],
    }
}

/// Owns a batch's pinned destinations through every submitted D2H operation.
struct SavedSegment {
    segment: Segment,
    meta: EncodedSegment,
    device: Option<u64>,
}

pub(super) fn save(
    runtime: &WorkerRuntime,
    layers: &mut [LayerTransferData],
    storage: Option<&Storage>,
    numa: NumaNode,
    groups: &[SaveGroup],
    observation: &mut Observation,
) -> Result<(), EngineError> {
    if !layers.iter().any(|l| {
        l.blocks
            .iter()
            .any(|b| matches!(b.block, TransferPayload::Pending))
    }) {
        return Ok(());
    }
    let storage = storage.ok_or_else(|| EngineError::Storage("missing codec allocator".into()))?;
    let mut owner = runtime.codec.borrow_mut();
    if owner.is_none() {
        *owner = Some(GpuCodec::new(runtime.stream.context()).map_err(EngineError::Storage)?);
    }
    let codec = owner.as_mut().expect("initialized codec");
    let _reservation = Reservation::new(storage.codec_budget, "encode");
    for group in groups {
        save_blocks(
            runtime,
            codec,
            layers,
            storage,
            numa,
            &group.blocks,
            Some(&group.key),
            observation,
        )?;
    }
    let remaining: Vec<_> = layers
        .iter()
        .enumerate()
        .flat_map(|(layer, data)| {
            data.blocks
                .iter()
                .enumerate()
                .filter_map(|(block, data)| {
                    matches!(data.block, TransferPayload::Pending).then_some((layer, block))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if !remaining.is_empty() {
        save_blocks(
            runtime,
            codec,
            layers,
            storage,
            numa,
            &remaining,
            None,
            observation,
        )?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "Keep existing resource owners and their observation explicit"
)]
fn save_blocks(
    runtime: &WorkerRuntime,
    codec: &mut GpuCodec,
    layers: &mut [LayerTransferData],
    storage: &Storage,
    numa: NumaNode,
    blocks: &[(usize, usize)],
    key: Option<&StateKey>,
    observation: &mut Observation,
) -> Result<(), EngineError> {
    let stream = &runtime.stream;
    let alignment = if storage.is_ssd_enabled() { 512 } else { 1 };
    let mut inputs = Vec::new();
    let mut destinations = Vec::new();
    for (slot, &(layer, block)) in blocks.iter().enumerate() {
        let layer = &layers[layer];
        for (segment, (source, bytes)) in ranges(
            layer
                .layout
                .block_copies(layer.blocks[block].block_idx)
                .map_err(EngineError::Storage)?,
        )
        .into_iter()
        .enumerate()
        {
            inputs.push(EncodeInput {
                source,
                bytes,
                format: segment_format(layer.layout.storage_format, segment),
            });
            destinations.push(slot);
        }
    }
    let mut saved: Vec<Vec<SavedSegment>> = (0..blocks.len()).map(|_| Vec::new()).collect();
    let mut offset = 0;
    let mut transferred = 0u64;
    while offset < inputs.len() {
        observation.submitted();
        // SAFETY: the save task owns every registered source page until completion.
        let batch = unsafe { codec.encode_batch(stream, &inputs[offset..], storage.codec_budget) }
            .map_err(EngineError::Storage)?;
        if batch.processed == 0 {
            return Err(EngineError::Storage("codec batch made no progress".into()));
        }
        let mut allocated = Vec::with_capacity(batch.processed);
        for (input, encoded) in inputs[offset..offset + batch.processed]
            .iter()
            .zip(&batch.outputs)
        {
            let raw_size = input.bytes.next_multiple_of(alignment);
            let encoded = encoded.as_ref().filter(|output| {
                output.meta.stored_bytes.next_multiple_of(alignment) <= raw_size - raw_size / 8
            });
            let (source, meta) = encoded.map_or_else(
                || {
                    (
                        input.source,
                        EncodedSegment {
                            version: 1,
                            format: StorageFormat::Exact,
                            logical_bytes: input.bytes,
                            stored_bytes: input.bytes,
                            checksum: 0,
                        },
                    )
                },
                |output| (output.device, output.meta.clone()),
            );
            let physical = meta.stored_bytes.next_multiple_of(alignment);
            let allocation = storage
                .allocate(
                    NonZeroU64::new(physical as u64)
                        .ok_or_else(|| EngineError::Storage("empty encoded segment".into()))?,
                    Some(numa),
                )
                .ok_or_else(|| {
                    EngineError::Storage("pinned pool exhausted during encoded save".into())
                })?;
            let ptr = allocation.mapped_ptr().host();
            // Only padding needs initialization; DMA fills every payload byte.
            unsafe {
                ptr.as_ptr()
                    .add(meta.stored_bytes)
                    .write_bytes(0, physical - meta.stored_bytes);
            }
            allocated.push((
                source,
                SavedSegment {
                    segment: Segment::new(ptr, physical, allocation),
                    meta,
                    device: Some(source),
                },
            ));
        }
        let copied = crate::transfer::contain_gpu_unwind(|| {
            for (source, saved) in &allocated {
                // SAFETY: all pinned destinations are exclusive and retained through drain.
                unsafe {
                    result::memcpy_dtoh_async(
                        std::slice::from_raw_parts_mut(
                            saved.segment.host_ptr().as_ptr(),
                            saved.meta.stored_bytes,
                        ),
                        *source,
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| e.to_string())?;
            }
            Ok(())
        });
        finish_gpu_transfer(stream, copied)?;
        for (index, (_, mut saved_segment)) in allocated.into_iter().enumerate() {
            let input = &inputs[offset + index];
            transferred += saved_segment.meta.stored_bytes as u64;
            if saved_segment.meta.format == StorageFormat::Exact
                && matches!(
                    input.format,
                    StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
                )
                && input.bytes <= 16 * 1024 * 1024
            {
                let packed_size = (input.bytes / 2).next_multiple_of(alignment);
                if packed_size
                    <= input.bytes.next_multiple_of(alignment)
                        - input.bytes.next_multiple_of(alignment) / 8
                    && let Some(packed) = storage.allocate(
                        NonZeroU64::new(packed_size as u64).expect("nonzero"),
                        Some(numa),
                    )
                {
                    let host = unsafe {
                        std::slice::from_raw_parts(
                            saved_segment.segment.host_ptr().as_ptr(),
                            input.bytes,
                        )
                    };
                    let output = unsafe {
                        std::slice::from_raw_parts_mut(
                            packed.mapped_ptr().host().as_ptr(),
                            packed_size,
                        )
                    };
                    output.fill(0);
                    if crate::codec::cpu::encode(input.format, host, &mut output[..input.bytes / 2])
                    {
                        saved_segment.meta.format = input.format;
                        saved_segment.meta.stored_bytes = input.bytes / 2;
                        saved_segment.meta.checksum = crc32fast::hash(&output[..input.bytes / 2]);
                        saved_segment.segment =
                            Segment::new(packed.mapped_ptr().host(), packed_size, packed);
                        saved_segment.device = None;
                    }
                }
            }
            if saved_segment.meta.format == StorageFormat::Exact
                && input.format != StorageFormat::Exact
            {
                core_metrics()
                    .storage_codec_skips
                    .add(1, &[KeyValue::new("reason", "range_ratio_or_budget")]);
            }
            saved[destinations[offset + index]].push(saved_segment);
        }
        if offset + batch.processed == inputs.len() {
            let mut devices = Vec::new();
            for ((layer, block), segments) in blocks.iter().copied().zip(&mut saved) {
                let compressed = segments
                    .iter()
                    .any(|s| s.meta.format != StorageFormat::Exact);
                if compressed {
                    for saved in segments
                        .iter_mut()
                        .filter(|s| s.meta.format == StorageFormat::Exact)
                    {
                        saved.meta.checksum = crc32fast::hash(unsafe {
                            std::slice::from_raw_parts(
                                saved.segment.host_ptr().as_ptr(),
                                saved.meta.stored_bytes,
                            )
                        });
                    }
                }
                let metadata: Vec<_> = segments.iter().map(|s| s.meta.clone()).collect();
                devices.push(
                    segments
                        .iter()
                        .map(|s| s.device.map(|device| (device, s.meta.stored_bytes)))
                        .collect::<Option<Vec<_>>>(),
                );
                let mut raw = RawBlock::new(
                    std::mem::take(segments)
                        .into_iter()
                        .map(|s| s.segment)
                        .collect(),
                );
                raw.storage_format = layers[layer].layout.storage_format;
                if compressed {
                    core_metrics().storage_codec_bytes.add(
                        metadata.iter().map(|m| m.logical_bytes as u64).sum(),
                        &[KeyValue::new("representation", "logical")],
                    );
                    core_metrics().storage_codec_bytes.add(
                        raw.memory_footprint(),
                        &[KeyValue::new("representation", "stored")],
                    );
                }
                raw.encoding = compressed.then_some(metadata);
                layers[layer].blocks[block].block = TransferPayload::Owned(raw);
            }
            if let Some(key) = key {
                // A split batch or CPU fallback no longer has every encoded byte
                // in the current arena. Its sealed DRAM object uses normal writeback.
                if offset == 0 && devices.iter().all(Option::is_some) {
                    write_gpu(runtime, storage, key, numa, blocks, layers, devices)?;
                } else {
                    core_metrics().ssd_gpu_write_fallbacks.add(1, &[]);
                }
            }
        }
        offset += batch.processed;
        // The batch borrow ends only after every D2H and direct SSD write drains.
    }
    core_metrics()
        .storage_codec_transfer_bytes
        .add(transferred, &[KeyValue::new("direction", "d2h")]);
    Ok(())
}

fn write_gpu(
    runtime: &WorkerRuntime,
    storage: &Storage,
    key: &StateKey,
    numa: NumaNode,
    blocks: &[(usize, usize)],
    layers: &[LayerTransferData],
    devices: Vec<Option<Vec<(u64, usize)>>>,
) -> Result<(), EngineError> {
    let Some(store) = &storage.ssd_store else {
        return Ok(());
    };
    let mut metadata = Vec::with_capacity(blocks.len());
    let mut copies = Vec::new();
    let mut offset = 0;
    for ((layer, block), devices) in blocks.iter().copied().zip(devices) {
        let raw = layers[layer].blocks[block].block.raw();
        let mut meta = SlotMeta::new(
            raw.segment_iovecs().map(|(_, size)| size as u64).collect(),
            numa,
        );
        meta.encoding = raw.encoding.clone();
        for ((source, bytes), &physical) in devices
            .expect("GPU sources checked")
            .into_iter()
            .zip(&meta.segment_sizes)
        {
            copies.push(CopyRange {
                file_offset: offset,
                device: source,
                bytes,
            });
            offset += physical;
        }
        metadata.push(meta);
    }
    let Some(lease) = store.reserve_gpu(key.clone(), metadata) else {
        return Ok(());
    };
    for copy in &mut copies {
        copy.file_offset += lease.entry.file_offset;
    }
    let batches = plan_writes(lease.entry.file_offset, lease.entry.len, copies)
        .map_err(EngineError::Storage)?;
    let mut owner = runtime.codec_write.borrow_mut();
    if owner.is_none() {
        *owner = Some(GpuSlot::new(runtime.stream.context()).map_err(|error| {
            store.gpu_io.failed(&error);
            EngineError::Storage(error)
        })?);
    }
    let slot = owner.as_mut().expect("initialized GPU storage slot");
    for batch in batches {
        #[cfg(feature = "test-hooks")]
        while crate::test_faults::active("cufile_write") {
            std::thread::sleep(Duration::from_micros(50));
        }
        if let Err(error) = slot.submit(Arc::clone(lease.file()), batch, true) {
            *owner = None;
            return Err(EngineError::Storage(error));
        }
        loop {
            if let Some(result) = slot.poll() {
                if let Err(error) = result {
                    // A failed stream must not be reused by a subsequent batch.
                    *owner = None;
                    return Err(EngineError::Storage(error));
                }
                break;
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }
    lease.commit();
    Ok(())
}

pub(super) fn restore(
    runtime: &WorkerRuntime,
    layers: &[LayerTransferData],
    budget: usize,
    observation: &mut Observation,
) -> Result<usize, EngineError> {
    let mut inputs = Vec::new();
    for layer in layers {
        for block in &layer.blocks {
            if matches!(block.block, TransferPayload::Ssd { .. }) {
                continue;
            }
            let raw = block.block.raw();
            let Some(metadata) = &raw.encoding else {
                continue;
            };
            let copies = ranges(
                layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?,
            );
            if block.block.host_offset() != 0
                || metadata.len() != copies.len()
                || raw.num_segments() != metadata.len()
            {
                return Err(EngineError::Storage(
                    "encoded restore layout mismatch".into(),
                ));
            }
            for (index, (meta, (target, bytes))) in metadata.iter().zip(copies).enumerate() {
                if meta.logical_bytes != bytes
                    || (meta.format != StorageFormat::Exact
                        && meta.format != segment_format(layer.layout.storage_format, index))
                {
                    return Err(EngineError::Storage(
                        "encoded restore representation mismatch".into(),
                    ));
                }
                meta.validate_metadata(raw.segment_size(index).expect("segment count checked"))
                    .map_err(EngineError::Storage)?;
                let host = unsafe {
                    std::slice::from_raw_parts(
                        raw.segment_ptr(index)
                            .expect("segment count checked")
                            .as_ptr(),
                        meta.stored_bytes,
                    )
                };
                inputs.push(HostDecodeInput {
                    source: host,
                    target,
                    target_bytes: bytes,
                    meta,
                });
            }
        }
    }
    if inputs.is_empty() {
        return Ok(0);
    }
    let _reservation = Reservation::new(budget, "decode");
    let stream = &runtime.stream;
    let mut owner = runtime.codec.borrow_mut();
    if owner.is_none() {
        *owner = Some(GpuCodec::new(stream.context()).map_err(EngineError::Storage)?);
    }
    let codec = owner.as_mut().expect("initialized codec");
    let mut total = 0;
    let mut transferred = 0;
    let cpu: Vec<bool> = inputs
        .iter()
        .map(|input| {
            if matches!(
                input.meta.format,
                StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
            ) {
                codec.host_decode_fits(input.meta, budget).map(|fits| !fits)
            } else {
                Ok(false)
            }
        })
        .collect::<Result<_, _>>()
        .map_err(EngineError::Storage)?;
    let mut offset = 0;
    while offset < inputs.len() {
        let input = &inputs[offset];
        if cpu[offset] {
            input
                .meta
                .validate(input.source)
                .map_err(EngineError::Storage)?;
            let mut decoded = vec![0; input.meta.logical_bytes];
            if !crate::codec::cpu::decode(input.meta.format, input.source, &mut decoded) {
                return Err(EngineError::Storage("CPU FP8 decode mismatch".into()));
            }
            observation.submitted();
            let copied =
                unsafe { result::memcpy_htod_async(input.target, &decoded, stream.cu_stream()) }
                    .map_err(|e| e.to_string());
            finish_gpu_transfer(stream, copied)?;
            total += input.meta.logical_bytes;
            transferred += decoded.len();
            offset += 1;
        } else {
            let end = cpu[offset..]
                .iter()
                .position(|&fallback| fallback)
                .map_or(inputs.len(), |n| offset + n);
            observation.submitted();
            // SAFETY: source host allocations and destination pages are owned by this task.
            let count = unsafe { codec.decode_host_batch(stream, &inputs[offset..end], budget) }
                .map_err(EngineError::Storage)?;
            if count == 0 {
                return Err(EngineError::Storage(
                    "codec restore made no progress".into(),
                ));
            }
            for input in &inputs[offset..offset + count] {
                total += input.meta.logical_bytes;
                transferred += input.source.len();
            }
            offset += count;
        }
    }
    core_metrics()
        .storage_codec_transfer_bytes
        .add(transferred as u64, &[KeyValue::new("direction", "h2d")]);
    Ok(total)
}
