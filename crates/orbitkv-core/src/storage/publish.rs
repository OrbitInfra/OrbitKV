use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use log::{debug, error, info, warn};
use tokio::sync::oneshot;

use crate::PublishDiagnostic;
use crate::block::{InflightBlock, RawBlock, SealedBlock, SlotInsertResult, StateKey};
use crate::memory::numa::NumaNode;
use crate::metrics::core_metrics;
use crate::storage::ssd::SsdStore;

use super::dram::DramStore;

type InsertEntries = Vec<(StateKey, Vec<(usize, RawBlock)>)>;

/// One layer's saved blocks, ready for cache insertion.
///
/// The `RawBlock`s are constructed at allocation time (Phase 2) and shared
/// with the GPU copy task; their segments own the pinned allocations, so no
/// separate allocation bookkeeping is needed.
pub(crate) struct RawSaveLayer {
    pub slot_id: usize,
    /// Padded block size (SSD-aligned). Becomes `RawBlock.total_size` → `SlotMeta.total_size()`.
    pub padded_block_size: usize,
    /// Saved blocks, parallel to `block_hashes`.
    pub blocks: Vec<RawBlock>,
    /// Block hashes in save order.
    pub block_hashes: Vec<Vec<u8>>,
}

/// Deferred save batch: sent to insert worker after GPU copy completes.
pub(crate) struct RawSaveBatch {
    pub namespace: String,
    pub total_slots: usize,
    pub numa_node: NumaNode,
    pub diagnostic: Option<PublishDiagnostic>,
    pub layers: Vec<RawSaveLayer>,
}

/// Build insert entries from a raw batch (called by the insert worker).
///
/// Returns `(entries, total_bytes, total_blocks)` where entries is grouped
/// by hash: `Vec<(StateKey, Vec<(slot_id, RawBlock)>)>`.
fn build_insert_entries(batch: RawSaveBatch) -> (InsertEntries, u64, usize) {
    let mut total_bytes: u64 = 0;
    let mut total_blocks: usize = 0;
    for layer in &batch.layers {
        let layer_blocks = layer.block_hashes.len();
        total_blocks += layer_blocks;
        total_bytes += (layer.padded_block_size as u64).saturating_mul(layer_blocks as u64);
    }

    let entries = if can_use_ordered_fast_path(&batch.layers) {
        build_ordered_insert_entries(batch.namespace, batch.layers)
    } else {
        build_hashed_insert_entries(batch.namespace, batch.layers)
    };
    (entries, total_bytes, total_blocks)
}

fn can_use_ordered_fast_path(layers: &[RawSaveLayer]) -> bool {
    let Some(first_layer) = layers.first() else {
        return false;
    };
    layers
        .iter()
        .all(|layer| layer.block_hashes == first_layer.block_hashes)
}

/// Fast path: all layers share one hash order, so entries can be built
/// block-by-block without a hash map.
fn build_ordered_insert_entries(namespace: String, layers: Vec<RawSaveLayer>) -> InsertEntries {
    let hashes = layers
        .first()
        .map(|layer| layer.block_hashes.clone())
        .unwrap_or_default();
    let num_blocks = hashes.len();
    let mut per_block_slots: Vec<Vec<(usize, RawBlock)>> =
        (0..num_blocks).map(|_| Vec::new()).collect();

    for layer in layers {
        let slot_id = layer.slot_id;
        for (block_idx, block) in layer.blocks.into_iter().enumerate() {
            per_block_slots[block_idx].push((slot_id, block));
        }
    }

    hashes
        .into_iter()
        .zip(per_block_slots)
        .map(|(hash, slots)| (StateKey::new(namespace.clone(), hash), slots))
        .collect()
}

/// Fallback for heterogeneous per-layer hash sets: group via a hash map.
fn build_hashed_insert_entries(namespace: String, layers: Vec<RawSaveLayer>) -> InsertEntries {
    let mut hash_entries: HashMap<Vec<u8>, Vec<(usize, RawBlock)>> = HashMap::new();
    for layer in layers {
        for (block, hash) in layer.blocks.into_iter().zip(layer.block_hashes) {
            hash_entries
                .entry(hash)
                .or_default()
                .push((layer.slot_id, block));
        }
    }

    hash_entries
        .into_iter()
        .map(|(hash, slots)| (StateKey::new(namespace.clone(), hash), slots))
        .collect()
}

pub(super) enum InsertWorkerCommand {
    RawInsert(RawSaveBatch),
    Flush(oneshot::Sender<()>),
    Gc {
        max_age: std::time::Duration,
        reply: oneshot::Sender<usize>,
    },
}

pub(crate) struct PublishQueue {
    insert_tx: Sender<InsertWorkerCommand>,
}

impl PublishQueue {
    pub(super) fn spawn(dram: Arc<DramStore>, ssd: Option<Arc<SsdStore>>) -> std::io::Result<Self> {
        let (insert_tx, insert_rx) = std::sync::mpsc::channel();
        let worker = PublishWorker::new(dram, ssd);
        std::thread::Builder::new()
            .name("orbitkv-insert".into())
            .spawn(move || worker.run(insert_rx))?;
        Ok(Self { insert_tx })
    }

    pub(crate) fn insert(&self, batch: RawSaveBatch) {
        let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    }

    /// Wait until the worker has handled every preceding publication.
    pub(crate) async fn flush(&self) {
        let (tx, rx) = oneshot::channel();
        if self.insert_tx.send(InsertWorkerCommand::Flush(tx)).is_ok() {
            let _ = rx.await;
        }
    }

    pub(crate) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .insert_tx
            .send(InsertWorkerCommand::Gc {
                max_age,
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.await.unwrap_or(0)
    }
}

struct PublishWorker {
    dram: Arc<DramStore>,
    ssd: Option<Arc<SsdStore>>,
    inflight: HashMap<StateKey, InflightBlock>,
}

impl PublishWorker {
    fn new(dram: Arc<DramStore>, ssd: Option<Arc<SsdStore>>) -> Self {
        Self {
            dram,
            ssd,
            inflight: HashMap::new(),
        }
    }

    fn run(mut self, rx: Receiver<InsertWorkerCommand>) {
        while let Ok(cmd) = rx.recv() {
            let mut cmds = vec![cmd];
            while let Ok(more) = rx.try_recv() {
                cmds.push(more);
            }

            for cmd in cmds {
                match cmd {
                    InsertWorkerCommand::RawInsert(batch) => {
                        self.save(batch);
                    }
                    InsertWorkerCommand::Flush(tx) => {
                        let _ = tx.send(());
                    }
                    InsertWorkerCommand::Gc { max_age, reply } => {
                        let cleaned = gc_inflight(&mut self.inflight, max_age);
                        let _ = reply.send(cleaned);
                    }
                }
            }
        }

        info!(
            "Insert worker shutting down, {} inflight blocks remaining",
            self.inflight.len()
        );
    }

    fn save(&mut self, batch: RawSaveBatch) {
        let diagnostic = batch.diagnostic;
        if let Some(diagnostic) = diagnostic {
            diagnostic.record(
                "publish_storage_dequeue",
                orbitkv_common::timeline::DiagnosticFields::default(),
            );
        }
        let start = std::time::Instant::now();
        let namespace = batch.namespace.clone();
        let numa_node = batch.numa_node;
        let total_slots = batch.total_slots;

        let (entries, total_bytes, total_blocks) = build_insert_entries(batch);

        self.insert(entries, total_slots, numa_node, &namespace, diagnostic);

        if let Some(diagnostic) = diagnostic {
            diagnostic.record(
                "publish_storage_complete",
                orbitkv_common::timeline::DiagnosticFields::default()
                    .blocks(total_blocks)
                    .bytes(total_bytes)
                    .elapsed_us(start.elapsed().as_micros() as u64),
            );
        }

        debug!(
            "insert_worker: batch sealed blocks={} bytes={} ms={:.2}",
            total_blocks,
            total_bytes,
            start.elapsed().as_secs_f64() * 1000.0,
        );
    }

    fn insert(
        &mut self,
        entries: InsertEntries,
        total_slots: usize,
        numa_node: NumaNode,
        namespace: &str,
        diagnostic: Option<PublishDiagnostic>,
    ) -> usize {
        let mut sealed_blocks: Vec<(StateKey, Arc<SealedBlock>)> = Vec::new();
        let mut inflight_bytes_added: u64 = 0;
        let mut inflight_bytes_removed: u64 = 0;
        let mut ordered_fast_path_seals = 0usize;

        let inflight = &mut self.inflight;

        for (key, slots) in entries {
            // Drop a late duplicate save of an already-resident block.
            if self.dram.contains_keys(std::slice::from_ref(&key))[0] {
                continue;
            }

            if !inflight.contains_key(&key) {
                match SealedBlock::from_ordered_slot_inserts(slots, total_slots, numa_node) {
                    Ok(sealed) => {
                        ordered_fast_path_seals += 1;
                        sealed_blocks.push((key, Arc::new(sealed)));
                        continue;
                    }
                    Err(slots) => {
                        insert_partial_slots(
                            inflight,
                            key,
                            slots,
                            total_slots,
                            numa_node,
                            namespace,
                            &mut sealed_blocks,
                            &mut inflight_bytes_added,
                            &mut inflight_bytes_removed,
                        );
                        continue;
                    }
                }
            }

            insert_partial_slots(
                inflight,
                key,
                slots,
                total_slots,
                numa_node,
                namespace,
                &mut sealed_blocks,
                &mut inflight_bytes_added,
                &mut inflight_bytes_removed,
            );
        }

        if inflight_bytes_added > 0 {
            core_metrics()
                .inflight_bytes
                .add(inflight_bytes_added as i64, &[]);
        }
        if inflight_bytes_removed > 0 {
            core_metrics()
                .inflight_bytes
                .add(-(inflight_bytes_removed as i64), &[]);
        }

        if !sealed_blocks.is_empty() {
            self.dram.batch_insert_refs(&sealed_blocks);
            if let Some(ssd) = &self.ssd {
                ssd.ingest_batch_diagnostic(
                    sealed_blocks.iter().map(|(key, block)| (key, block)),
                    false,
                    diagnostic,
                );
            }
        }

        ordered_fast_path_seals
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "insert worker threads batch-local accounting through the fallback path"
)]
fn insert_partial_slots(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    key: StateKey,
    slots: Vec<(usize, crate::block::RawBlock)>,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
    sealed_blocks: &mut Vec<(StateKey, Arc<SealedBlock>)>,
    inflight_bytes_added: &mut u64,
    inflight_bytes_removed: &mut u64,
) {
    let inflight_block = match inflight.entry(key.clone()) {
        Entry::Vacant(v) => v.insert(InflightBlock::new(total_slots)),
        Entry::Occupied(o) => {
            let ib = o.into_mut();
            if ib.total_slots() != total_slots {
                error!(
                    "insert worker: slot count mismatch: key namespace={} expected={} got={}",
                    namespace,
                    ib.total_slots(),
                    total_slots
                );
                return;
            }
            ib
        }
    };

    let mut completed = false;
    for (slot_id, block) in slots {
        match inflight_block.insert_slot(slot_id, block, numa_node) {
            Ok(SlotInsertResult::Inserted {
                completed: c,
                footprint_added,
            }) => {
                *inflight_bytes_added = inflight_bytes_added.saturating_add(footprint_added);
                completed = c;
                if completed {
                    break;
                }
            }
            Ok(SlotInsertResult::Duplicate) => {}
            Err(error) => {
                log::error!("Cannot insert publish slot: {error}");
                return;
            }
        }
    }

    if completed {
        let Some(inflight_block) = inflight.remove(&key) else {
            return;
        };
        let total_footprint = inflight_block.footprint();
        *inflight_bytes_removed = inflight_bytes_removed.saturating_add(total_footprint);
        let sealed = match inflight_block.seal() {
            Ok(block) => Arc::new(block),
            Err(error) => {
                log::error!("Cannot publish incomplete block: {error}");
                return;
            }
        };

        sealed_blocks.push((key, sealed));
    }
}

fn gc_inflight(
    inflight: &mut HashMap<StateKey, InflightBlock>,
    max_age: std::time::Duration,
) -> usize {
    let before = inflight.len();

    inflight.retain(|key, block| {
        let age = block.age();
        if age > max_age {
            warn!(
                "GC: removing stale inflight block: namespace={} hash_len={} filled={} total={} age_secs={}",
                key.namespace,
                key.hash.len(),
                block.filled_count(),
                block.total_slots(),
                age.as_secs()
            );
            core_metrics().inflight_bytes.add(-(block.footprint() as i64), &[]);
            false
        } else {
            true
        }
    });

    let cleaned = before - inflight.len();
    if cleaned > 0 {
        core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
        info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
    }
    cleaned
}

#[cfg(test)]
#[path = "../../tests/unit/storage/publish.rs"]
mod tests;
