use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::Arc;

use log::{debug, info};

use orbitkv_state::group_hash;

use super::{EngineError, OrbitKVEngine};
use crate::SlotMeta;
use crate::block::{LayerSave, RawBlock, Segment, StateKey};
use crate::memory::numa::NumaNode;
use crate::memory::pool::PinnedAllocation;
use crate::metrics::core_metrics;
use crate::storage::publish::{RawSaveBatch, RawSaveLayer};
use crate::storage::ssd::SsdStore;
use crate::storage::ssd::cufile::{CopyRange, plan_writes};
use crate::transfer::layout::{BlockCopies, KVCacheLayout};
use crate::transfer::worker::ssd::GpuWrite;
use crate::transfer::worker::{LayerTransferData, SaveGroup, TransferBlock, TransferPayload};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublishDiagnostic {
    pub request_id: u64,
    pub session_epoch: u64,
    pub session_token: u64,
}

impl PublishDiagnostic {
    pub(crate) fn record(
        &self,
        stage: &'static str,
        fields: orbitkv_common::timeline::DiagnosticFields,
    ) {
        orbitkv_common::timeline::record_diagnostic(
            stage,
            fields.with_operation(self.request_id, self.session_epoch, self.session_token),
        );
    }
}

/// Unified per-layer context for the save pipeline.
/// Combines metadata, filtered blocks, and allocation results.
struct LayerContext {
    layer_name: String,
    layout: KVCacheLayout,
    slot_id: usize,
    /// Hybrid-cache storage group of this layer; keyed and sealed per group.
    group: u32,
    /// Blocks to save: (block_idx, hash). Filtered in Phase 1.
    blocks_to_save: Vec<(usize, Vec<u8>)>,
    /// Host-side block images, parallel to `blocks_to_save`.
    /// Filled after the GPU save task returns the moved blocks.
    raw_blocks: Vec<RawBlock>,
}

/// Only a complete state group can be written from this publisher's GPU pages.
/// Fragmented/multi-writer publications still seal in DRAM before SSD ingestion.
fn prepare_gpu_writes(
    store: &Arc<SsdStore>,
    namespace: &str,
    topology: &crate::engine::instance::LayerTopology,
    layers: &[LayerContext],
    numa: NumaNode,
) -> Result<Vec<GpuWrite>, EngineError> {
    use std::collections::BTreeMap;
    type SlotCopies = BTreeMap<usize, (SlotMeta, Vec<CopyRange>)>;
    let mut objects: HashMap<(u32, Vec<u8>), SlotCopies> = HashMap::new();
    for layer in layers {
        let page = topology.is_page_first();
        let layer_id = topology.layer_id(&layer.layer_name)?;
        let offset = if page {
            topology.page_placement(layer_id).expect("page layout").0
        } else {
            0
        };
        let sizes = if page {
            smallvec::smallvec![
                topology
                    .shard_page_size(layer.slot_id)
                    .expect("page layout") as u64
            ]
        } else if layer.layout.geometry().is_split() {
            smallvec::smallvec![layer.layout.geometry().padded_segment_bytes() as u64; 2]
        } else {
            smallvec::smallvec![layer.layout.geometry().padded_block_bytes() as u64]
        };
        for (block, hash) in &layer.blocks_to_save {
            let (_, copies) = objects
                .entry((layer.group, hash.clone()))
                .or_default()
                .entry(layer.slot_id)
                .or_insert_with(|| (SlotMeta::new(sizes.clone(), numa), Vec::new()));
            match layer
                .layout
                .block_copies(*block)
                .map_err(EngineError::Storage)?
            {
                BlockCopies::Contiguous(copy) => copies.push(CopyRange {
                    file_offset: offset as u64,
                    device: copy.addr,
                    bytes: copy.bytes,
                }),
                BlockCopies::Split { k, v } => {
                    copies.push(CopyRange {
                        file_offset: 0,
                        device: k.addr,
                        bytes: k.bytes,
                    });
                    copies.push(CopyRange {
                        file_offset: layer.layout.geometry().padded_segment_bytes() as u64,
                        device: v.addr,
                        bytes: v.bytes,
                    });
                }
            }
        }
    }
    let mut writes = Vec::new();
    for ((group, hash), slots) in objects {
        let expected = topology.group_total_slots(group)?;
        if slots.len() != expected || !slots.keys().copied().eq(0..expected) {
            continue;
        }
        let mut metadata = Vec::with_capacity(expected);
        let mut copies = Vec::new();
        let mut offset = 0;
        for (meta, ranges) in slots.into_values() {
            copies.extend(ranges.into_iter().map(|mut range| {
                range.file_offset += offset;
                range
            }));
            offset += meta.total_size();
            metadata.push(meta);
        }
        let key = StateKey::new(namespace.to_owned(), group_hash(&hash, group));
        let Some(lease) = store.reserve_gpu(key, metadata) else {
            continue;
        };
        for copy in &mut copies {
            copy.file_offset += lease.entry.file_offset;
        }
        let batches = plan_writes(lease.entry.file_offset, lease.entry.len, copies)
            .map_err(EngineError::Storage)?;
        writes.push(GpuWrite { lease, batches });
    }
    Ok(writes)
}

/// Encoded sizes are known only on the GPU worker. Keep complete groups in
/// slot order so their SSD reservation can be made before that arena is reused.
fn prepare_codec_groups(
    namespace: &str,
    topology: &crate::engine::instance::LayerTopology,
    layers: &[LayerContext],
) -> Result<Vec<SaveGroup>, EngineError> {
    let mut groups = HashMap::new();
    for (layer_index, layer) in layers.iter().enumerate() {
        for (block_position, (_, hash)) in layer.blocks_to_save.iter().enumerate() {
            groups
                .entry((layer.group, hash.clone()))
                .or_insert_with(std::collections::BTreeMap::new)
                .insert(layer.slot_id, (layer_index, block_position));
        }
    }
    let mut result = Vec::new();
    for ((group, hash), slots) in groups {
        let expected = topology.group_total_slots(group)?;
        if slots.len() == expected && slots.keys().copied().eq(0..expected) {
            result.push(SaveGroup {
                key: StateKey::new(namespace.to_owned(), group_hash(&hash, group)),
                blocks: slots.into_values().collect(),
            });
        }
    }
    // Preserve the publisher's block order for adjacent-file writes.
    result.sort_by_key(|group| group.blocks.first().map(|&(layer, block)| (block, layer)));
    Ok(result)
}

/// Drop `(group, hash)` candidates whose group-encoded key already exists in
/// the read cache. Filtering is per group because groups key the same content
/// hash independently (e.g. attention block vs. recurrent checkpoint).
fn filter_new_hashes_per_group(
    storage: &crate::storage::Storage,
    namespace: &str,
    candidates: HashSet<(u32, Vec<u8>)>,
) -> HashSet<(u32, Vec<u8>)> {
    let mut by_group: HashMap<u32, Vec<Vec<u8>>> = HashMap::new();
    for (group, hash) in &candidates {
        by_group.entry(*group).or_default().push(hash.clone());
    }

    let mut survivors = HashSet::with_capacity(candidates.len());
    for (group, raw_hashes) in by_group {
        let mut encoded: HashSet<Vec<u8>> = raw_hashes
            .iter()
            .map(|hash| group_hash(hash, group))
            .collect();
        storage.filter_hashes_not_in_cache_inplace(namespace, &mut encoded);
        for raw in raw_hashes {
            if encoded.contains(&group_hash(&raw, group)) {
                survivors.insert((group, raw));
            }
        }
    }
    survivors
}

impl OrbitKVEngine {
    /// Page-first allocation: one contiguous pinned page per block, holding
    /// this writer's shard layers at their sealed shard-relative offsets. Each
    /// layer's GPU copy targets a sub-view into the pages; the pages themselves
    /// are sealed as the shard's stored slot (see Phase 4). Fills `pages` and
    /// `gpu_save_layers`.
    fn alloc_page_first_targets(
        &self,
        topology: &crate::engine::instance::LayerTopology,
        layer_contexts: &[LayerContext],
        numa_node: Option<NumaNode>,
        pages: &mut Vec<Arc<PinnedAllocation>>,
        gpu_save_layers: &mut Vec<LayerTransferData>,
    ) -> Result<(), EngineError> {
        // One writer saves exactly one shard (its registered layers), so every
        // layer in this batch maps to the same shard/slot.
        let shard = layer_contexts[0].slot_id;
        let page_size = topology
            .shard_page_size(shard)
            .expect("page-first topology must have a shard page size");

        // A page holds all of the shard's layers, so every layer must save the
        // same blocks in the same order. Block hashes are content-derived
        // (identical across layers) and block_idx is the same physical KV slot
        // across layers, so this invariant holds for MLA saves.
        let reference = &layer_contexts[0].blocks_to_save;
        for ctx in layer_contexts {
            if ctx.layout.geometry().is_split() {
                return Err(EngineError::InvalidArgument(format!(
                    "page-first save does not support split K/V (layer {})",
                    ctx.layer_name
                )));
            }
            if ctx.slot_id != shard {
                return Err(EngineError::InvalidArgument(
                    "page-first save spans multiple shards (one writer must save one shard)".into(),
                ));
            }
            if ctx.blocks_to_save != *reference {
                return Err(EngineError::InvalidArgument(
                    "page-first save requires the same block set across all layers".into(),
                ));
            }
        }

        // A shard's page is sealed by this single save, so the save must carry
        // EVERY layer of the shard. A partial save (e.g. a pipeline stage that
        // holds only some of the shard's layers) would seal a page whose
        // missing-layer offsets are never written — they stay as stale pool
        // memory and load returns another block's KV. Reject it loudly.
        let mut layer_ids: Vec<usize> = layer_contexts
            .iter()
            .map(|ctx| topology.layer_id(&ctx.layer_name))
            .collect::<Result<_, _>>()?;
        layer_ids.sort_unstable();
        layer_ids.dedup();
        let shard_layers = topology
            .shard_layer_count(shard)
            .expect("page-first topology must have a shard layer count");
        if layer_ids.len() != shard_layers {
            return Err(EngineError::InvalidArgument(format!(
                "page-first save must cover all {shard_layers} layers of shard {shard}, \
                 got {} distinct (one worker must write a shard's entire page)",
                layer_ids.len()
            )));
        }

        let mut padding = Vec::with_capacity(layer_contexts.len());
        for ctx in layer_contexts {
            let layer_id = topology.layer_id(&ctx.layer_name)?;
            let (offset, padded) = topology.page_placement(layer_id).expect("page layout");
            let BlockCopies::Contiguous(copy) = ctx
                .layout
                .block_copies(ctx.blocks_to_save[0].0)
                .map_err(EngineError::Storage)?
            else {
                unreachable!("split rejected above")
            };
            if copy.bytes < padded {
                padding.push((offset + copy.bytes, padded - copy.bytes));
            }
        }

        // One page per block.
        let page_bytes = NonZeroU64::new(page_size as u64)
            .ok_or_else(|| EngineError::Storage("page size is zero".into()))?;
        for _ in 0..reference.len() {
            let page = self
                .storage
                .allocate(page_bytes, numa_node)
                .ok_or_else(|| {
                    EngineError::Storage("pinned pool exhausted while allocating page".into())
                })?;
            for &(offset, len) in &padding {
                // SAFETY: these disjoint holes belong to the fresh page; no DMA has started.
                unsafe {
                    page.mapped_ptr()
                        .add(offset)
                        .host()
                        .as_ptr()
                        .write_bytes(0, len);
                }
            }
            pages.push(page);
        }

        // Each layer copies its block into its slice of every page.
        for ctx in layer_contexts {
            let layer_id = topology.layer_id(&ctx.layer_name)?;
            let (offset, layer_bytes) = topology
                .page_placement(layer_id)
                .expect("page-first layer placement");
            let save_blocks: Vec<TransferBlock> = ctx
                .blocks_to_save
                .iter()
                .zip(pages.iter())
                .map(|((block_idx, _), page)| TransferBlock {
                    block_idx: *block_idx,
                    block: TransferPayload::Owned(RawBlock::single_segment(Segment::new(
                        page.mapped_ptr().add(offset).host(),
                        layer_bytes,
                        Arc::clone(page),
                    ))),
                })
                .collect();
            gpu_save_layers.push(LayerTransferData {
                layer_name: ctx.layer_name.clone(),
                layout: ctx.layout.clone(),
                blocks: save_blocks,
            });
        }
        Ok(())
    }

    /// Batch save KV blocks from multiple layers.
    ///
    /// More efficient than calling `save_kv_blocks_from_ipc` in a loop as it
    /// reduces Python-Rust boundary crossings and batches GPU copies + storage
    /// operations across all layers for minimal lock overhead and a single
    /// CUDA stream synchronization.
    ///
    /// Pinned memory is allocated on the saving GPU's own NUMA node, keeping
    /// every D2H copy socket-local.
    pub async fn batch_save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        registration_generation: [u8; 16],
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
    ) -> Result<(), EngineError> {
        self.batch_save_kv_blocks_from_ipc_inner(
            instance_id,
            registration_generation,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            None,
        )
        .await
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "binds registered GPU identity, batched inputs and diagnostic ownership"
    )]
    pub async fn batch_save_kv_blocks_from_ipc_diagnostic(
        &self,
        instance_id: &str,
        registration_generation: [u8; 16],
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
        diagnostic: PublishDiagnostic,
    ) -> Result<(), EngineError> {
        self.batch_save_kv_blocks_from_ipc_inner(
            instance_id,
            registration_generation,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            Some(diagnostic),
        )
        .await
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "binds registered GPU identity, batched inputs and diagnostic ownership"
    )]
    async fn batch_save_kv_blocks_from_ipc_inner(
        &self,
        instance_id: &str,
        registration_generation: [u8; 16],
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
        diagnostic: Option<PublishDiagnostic>,
    ) -> Result<(), EngineError> {
        self.query_leases.sweep_expired();

        let batch_start = std::time::Instant::now();
        let total_layers = saves.len();

        let instance = self.get_instance(instance_id)?;
        if registration_generation == [0; 16] || instance.generation() != registration_generation {
            return Err(EngineError::InvalidArgument(
                "stale GPU registration generation".into(),
            ));
        }
        let topology = instance.sealed_topology()?;
        let namespace = topology.cache_namespace.clone();

        // ── Phase 0: Resolve per-layer metadata and build valid_blocks ──
        trace_scope!("save.resolve_metadata", _s);

        let gpu_context = instance.get_gpu_for_save_group(device_id, tp_rank, pp_rank)?;

        let mut layer_contexts: Vec<LayerContext> = Vec::with_capacity(saves.len());
        // Dedupe candidates per (group, hash): the same content hash may exist
        // in the attention group while the recurrent group still needs it.
        let mut hashes_to_save: HashSet<(u32, Vec<u8>)> = HashSet::new();

        for LayerSave {
            layer_name,
            block_ids,
            block_hashes,
        } in saves
        {
            // Engine-level contract: in-process callers bypass the RPC-layer
            // validation in service.rs, and zip below would silently truncate.
            if block_ids.len() != block_hashes.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "block_ids length {} does not match block_hashes {} for layer {}",
                    block_ids.len(),
                    block_hashes.len(),
                    layer_name
                )));
            }

            let layer_id = topology.layer_id(&layer_name)?;

            let layout = gpu_context.get_layout(&layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} not registered on device"))
            })?;

            let slot_id = topology.slot_index(layer_id, tp_rank)?;
            let group = topology.group_of_layer(layer_id);

            let num_blocks = layout.geometry().num_blocks();

            let blocks_to_save: Vec<(usize, Vec<u8>)> = block_ids
                .into_iter()
                .zip(block_hashes)
                .map(|(idx, hash)| {
                    if idx >= num_blocks {
                        return Err(EngineError::InvalidArgument(format!(
                            "block_id {} out of range (num_blocks={}) for layer {}",
                            idx, num_blocks, layer_name
                        )));
                    }
                    Ok((idx, hash))
                })
                .collect::<Result<_, _>>()?;

            if blocks_to_save.is_empty() {
                continue;
            }

            for (_, hash) in &blocks_to_save {
                hashes_to_save.insert((group, hash.clone()));
            }

            layer_contexts.push(LayerContext {
                layer_name,
                layout,
                slot_id,
                group,
                blocks_to_save,
                raw_blocks: Vec::new(),
            });
        }
        trace_drop!(_s);

        if layer_contexts.is_empty() {
            info!(
                "save_batch skipped (no valid blocks): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // ── Phase 1: Filter hashes against cache ──

        trace_scope!("save.hash_filter", _s);

        let hashes_to_save = filter_new_hashes_per_group(
            &self.storage,
            &namespace,
            std::mem::take(&mut hashes_to_save),
        );

        if hashes_to_save.is_empty() {
            trace_drop!(_s);
            return Ok(());
        }

        // Per-layer filter: keep only blocks whose (group, hash) needs saving
        let mut total_blocks_to_save = 0usize;
        for ctx in &mut layer_contexts {
            let group = ctx.group;
            ctx.blocks_to_save
                .retain(|(_, hash)| hashes_to_save.contains(&(group, hash.clone())));
            total_blocks_to_save += ctx.blocks_to_save.len();
        }
        // Remove layers with no blocks to save
        layer_contexts.retain(|ctx| !ctx.blocks_to_save.is_empty());

        trace_drop!(_s, || {
            [
                ("unique_hashes", hashes_to_save.len().to_string()),
                ("to_save", total_blocks_to_save.to_string()),
            ]
        });

        if layer_contexts.is_empty() {
            return Ok(());
        }

        // ── Phase 2: Allocate pinned memory + build SaveBlocks for all layers ──

        trace_scope!("save.pinned_alloc", _s);
        let save_numa_node = gpu_context.preferred_numa();
        let numa_node = Some(save_numa_node);
        let blockwise = self.storage.blockwise_alloc();
        let page_first = topology.is_page_first();

        let mut gpu_save_layers: Vec<LayerTransferData> = Vec::with_capacity(layer_contexts.len());
        // Page-first only: one contiguous page per block holds every layer.
        // Each layer's GPU copy targets a sub-view into these pages, and the
        // pages themselves become the single stored slot in Phase 4. Kept alive
        // across the GPU copy so the segments stay valid.
        let mut pages: Vec<Arc<PinnedAllocation>> = Vec::new();

        if page_first {
            self.alloc_page_first_targets(
                &topology,
                &layer_contexts,
                numa_node,
                &mut pages,
                &mut gpu_save_layers,
            )?;
        } else {
            for ctx in &mut layer_contexts {
                let layout = &ctx.layout;
                let num_blocks = ctx.blocks_to_save.len();
                if self.storage.codec != crate::StorageCodec::None {
                    gpu_save_layers.push(LayerTransferData {
                        layer_name: ctx.layer_name.clone(),
                        layout: layout.clone(),
                        blocks: ctx
                            .blocks_to_save
                            .iter()
                            .map(|(block_idx, _)| TransferBlock {
                                block_idx: *block_idx,
                                block: TransferPayload::Pending,
                            })
                            .collect(),
                    });
                    continue;
                }

                // Blockwise: allocate once per block; Batch: allocate once for all blocks
                let alloc_count = if blockwise { num_blocks } else { 1 };
                let blocks_per_alloc = if blockwise { 1 } else { num_blocks };

                let source_bytes = match layout
                    .block_copies(ctx.blocks_to_save[0].0)
                    .map_err(EngineError::Storage)?
                {
                    BlockCopies::Contiguous(copy) => copy.bytes,
                    BlockCopies::Split { k, v } => k.bytes + v.bytes,
                };

                let alloc_pinned = |stride: usize, what: &str| {
                    let alloc_size = (stride as u64)
                        .checked_mul(blocks_per_alloc as u64)
                        .and_then(NonZeroU64::new)
                        .ok_or_else(|| {
                            EngineError::Storage(format!("allocation size overflow for {what}"))
                        })?;
                    self.storage.allocate(alloc_size, numa_node).ok_or_else(|| {
                        EngineError::Storage(format!(
                            "pinned pool exhausted while allocating {what}"
                        ))
                    })
                };

                // Allocate pinned memory and construct the host-side RawBlocks in
                // save order. Strides are padded (SSD-aligned); GPU copies later
                // use actual (unpadded) sizes; initialize padding before codec/SSD reads.
                let mut raw_blocks: Vec<RawBlock> = Vec::with_capacity(num_blocks);
                for _ in 0..alloc_count {
                    if layout.geometry().is_split() {
                        let stride = layout.geometry().padded_segment_bytes();
                        let k_alloc = alloc_pinned(stride, "K segment buffer")?;
                        let v_alloc = alloc_pinned(stride, "V segment buffer")?;
                        for i in 0..blocks_per_alloc {
                            let offset = i * stride;
                            for alloc in [&k_alloc, &v_alloc] {
                                // SAFETY: only the padding tail of this newly allocated segment.
                                unsafe {
                                    alloc
                                        .mapped_ptr()
                                        .add(offset + layout.geometry().segment_bytes())
                                        .host()
                                        .as_ptr()
                                        .write_bytes(0, stride - layout.geometry().segment_bytes());
                                }
                            }
                            // Safety: offsets are within the just-made allocations.
                            raw_blocks.push(RawBlock::two_segments(
                                Segment::new(
                                    k_alloc.mapped_ptr().add(offset).host(),
                                    stride,
                                    Arc::clone(&k_alloc),
                                ),
                                Segment::new(
                                    v_alloc.mapped_ptr().add(offset).host(),
                                    stride,
                                    Arc::clone(&v_alloc),
                                ),
                            ));
                        }
                    } else {
                        let stride = layout.geometry().padded_block_bytes();
                        let alloc = alloc_pinned(stride, "block buffer")?;
                        for i in 0..blocks_per_alloc {
                            // SAFETY: only the padding tail, before the GPU writes the payload.
                            unsafe {
                                alloc
                                    .mapped_ptr()
                                    .add(i * stride + source_bytes)
                                    .host()
                                    .as_ptr()
                                    .write_bytes(0, stride - source_bytes);
                            }
                            // Safety: offset is within the just-made allocation.
                            raw_blocks.push(RawBlock::single_segment(Segment::new(
                                alloc.mapped_ptr().add(i * stride).host(),
                                stride,
                                Arc::clone(&alloc),
                            )));
                        }
                    }
                }

                let save_blocks: Vec<TransferBlock> = ctx
                    .blocks_to_save
                    .iter()
                    .zip(raw_blocks)
                    .map(|((block_idx, _), mut block)| {
                        block.storage_format = layout.storage_format;
                        TransferBlock {
                            block_idx: *block_idx,
                            block: TransferPayload::Owned(block),
                        }
                    })
                    .collect();

                gpu_save_layers.push(LayerTransferData {
                    layer_name: ctx.layer_name.clone(),
                    layout: layout.clone(),
                    blocks: save_blocks,
                });
            }
        }
        trace_drop!(_s);

        // ── Phase 3: Submit GPU encoding/copies while retaining source ownership ──

        let ssd_writes = match &self.storage.ssd_store {
            Some(store)
                if store.gpu_io.available() && self.storage.codec == crate::StorageCodec::None =>
            {
                prepare_gpu_writes(
                    store,
                    &namespace,
                    &topology,
                    &layer_contexts,
                    save_numa_node,
                )?
            }
            _ => Vec::new(),
        };
        let codec_groups = if self.storage.codec != crate::StorageCodec::None
            && self
                .storage
                .ssd_store
                .as_ref()
                .is_some_and(|store| store.gpu_io.available())
        {
            prepare_codec_groups(&namespace, &topology, &layer_contexts)?
        } else {
            Vec::new()
        };
        let returned_layers = trace_future!(
            "save.gpu_copy",
            gpu_context.worker_pool().batch_save(
                Arc::clone(&instance),
                gpu_save_layers,
                ssd_writes,
                codec_groups,
                Some(Arc::clone(&self.storage))
            )
        )
        .await?;

        if page_first {
            // The pages already hold the copied data; the returned sub-view
            // blocks share those pages via Arc and are dropped here.
            drop(returned_layers);
        } else {
            for (ctx, layer) in layer_contexts.iter_mut().zip(returned_layers) {
                ctx.raw_blocks = layer
                    .blocks
                    .into_iter()
                    .map(|transfer_block| match transfer_block.block {
                        TransferPayload::Owned(block) => block,
                        TransferPayload::Pending
                        | TransferPayload::Cached { .. }
                        | TransferPayload::Ssd { .. } => {
                            panic!("save path must return TransferPayload::Owned blocks")
                        }
                    })
                    .collect();
            }
        }

        // ── Phase 4 (deferred): Build RawBlocks + insert — sent to worker ──

        // Record metrics on the RPC path (cheap, measures RPC-visible latency)
        let metrics = core_metrics();
        let mut total_bytes = 0u64;
        for ctx in &layer_contexts {
            let num_blocks = ctx.blocks_to_save.len();
            let bytes = (ctx.layout.geometry().padded_block_bytes() as u64)
                .checked_mul(num_blocks as u64)
                .unwrap_or(0);
            total_bytes += bytes;
        }
        if total_blocks_to_save > 0 {
            metrics.save_bytes.add(total_bytes, &[]);
            metrics
                .save_duration_seconds
                .record(batch_start.elapsed().as_secs_f64(), &[]);
        }

        debug!(
            "save_batch completed: instance_id={} registration_generation={:?} tp_rank={} pp_rank={} device_id={} layers={} layers_saved={} blocks_saved={} bytes={} numa={} total_ms={:.2}",
            instance_id,
            registration_generation,
            tp_rank,
            pp_rank,
            device_id,
            total_layers,
            layer_contexts.len(),
            total_blocks_to_save,
            total_bytes,
            save_numa_node,
            batch_start.elapsed().as_secs_f64() * 1000.0
        );

        // Build per-group RawSaveBatches and send to the insert worker
        // (fire-and-forget). Each group seals blocks against its own slot
        // count and stores them under the same versioned group encoding.
        if page_first {
            // Page-first collapses a shard's layers into one page slot per
            // block, so a whole shard is a single RawSaveLayer regardless of
            // its layer count. Sealing rejects multi-group page-first, so the
            // whole batch is group 0.
            let slot_id = layer_contexts[0].slot_id;
            let page_size = topology
                .shard_page_size(slot_id)
                .expect("page-first shard page size");
            let block_hashes: Vec<Vec<u8>> = layer_contexts[0]
                .blocks_to_save
                .iter()
                .map(|(_, hash)| group_hash(hash, 0))
                .collect();
            let format = layer_contexts[0].layout.storage_format;
            let format = if layer_contexts
                .iter()
                .all(|ctx| ctx.layout.storage_format == format)
            {
                format
            } else {
                orbitkv_state::StorageFormat::Exact
            };
            let blocks: Vec<RawBlock> = pages
                .into_iter()
                .map(|page| {
                    let ptr = page.mapped_ptr().host();
                    let mut block = RawBlock::single_segment(Segment::new(ptr, page_size, page));
                    block.storage_format = format;
                    block
                })
                .collect();
            if let Some(diagnostic) = diagnostic {
                diagnostic.record(
                    "publish_storage_enqueue",
                    orbitkv_common::timeline::DiagnosticFields::default()
                        .group(0)
                        .blocks(blocks.len()),
                );
            }
            self.storage.writes.insert(RawSaveBatch {
                namespace,
                total_slots: topology.total_slots(),
                numa_node: save_numa_node,
                diagnostic,
                layers: vec![RawSaveLayer {
                    slot_id,
                    padded_block_size: page_size,
                    blocks,
                    block_hashes,
                }],
            });
        } else {
            let mut by_group: std::collections::BTreeMap<u32, Vec<LayerContext>> =
                Default::default();
            for ctx in layer_contexts {
                by_group.entry(ctx.group).or_default().push(ctx);
            }
            for (group, contexts) in by_group {
                let raw_layers: Vec<RawSaveLayer> = contexts
                    .into_iter()
                    .map(|ctx| {
                        let block_hashes: Vec<Vec<u8>> = ctx
                            .blocks_to_save
                            .into_iter()
                            .map(|(_, hash)| group_hash(&hash, group))
                            .collect();
                        RawSaveLayer {
                            slot_id: ctx.slot_id,
                            padded_block_size: ctx.layout.geometry().padded_block_bytes(),
                            blocks: ctx.raw_blocks,
                            block_hashes,
                        }
                    })
                    .collect();
                if let Some(diagnostic) = diagnostic {
                    let blocks = raw_layers
                        .first()
                        .map(|layer| layer.block_hashes.len())
                        .unwrap_or(0);
                    diagnostic.record(
                        "publish_storage_enqueue",
                        orbitkv_common::timeline::DiagnosticFields::default()
                            .group(group)
                            .blocks(blocks),
                    );
                }
                self.storage.writes.insert(RawSaveBatch {
                    namespace: namespace.clone(),
                    total_slots: topology.group_total_slots(group)?,
                    numa_node: save_numa_node,
                    diagnostic,
                    layers: raw_layers,
                });
            }
        }

        Ok(())
    }
}
