use super::{SsdStore, index::SsdIndexEntry, io::SsdIo};
use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
};
use crate::metrics::core_metrics;
use crate::peer::export::{PeerError, StagingReservation};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, warn};
use mea::oneshot;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Result of a single prefetch I/O.
type SingleMaterializedBlocks = (
    StateKey,
    SsdIndexEntry,
    Option<Arc<SealedBlock>>,
    f64,
    u64,
    Arc<BatchContext>,
);

type BatchResult = Result<crate::storage::MaterializedBlocks, PeerError>;

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub(super) struct PrefetchBatch {
    pub requests: Vec<Arc<super::SsdReadLease>>,
    pub done_tx: oneshot::Sender<BatchResult>,
    pub observation: Observation,
    pub reservation: Option<StagingReservation>,
}

impl PrefetchBatch {
    pub(super) fn new(
        requests: Vec<Arc<super::SsdReadLease>>,
        done_tx: oneshot::Sender<BatchResult>,
        resource: ExecutionResource,
    ) -> Self {
        Self::build(requests, done_tx, resource, None)
    }

    pub(super) fn for_export(
        requests: Vec<Arc<super::SsdReadLease>>,
        done_tx: oneshot::Sender<BatchResult>,
        resource: ExecutionResource,
        reservation: StagingReservation,
    ) -> Self {
        Self::build(requests, done_tx, resource, Some(reservation))
    }

    fn build(
        requests: Vec<Arc<super::SsdReadLease>>,
        done_tx: oneshot::Sender<BatchResult>,
        resource: ExecutionResource,
        reservation: Option<StagingReservation>,
    ) -> Self {
        if !crate::cost::enabled() {
            return Self {
                requests,
                done_tx,
                observation: Observation::disabled(),
                reservation,
            };
        }
        let mut logical_bytes = Some(0u64);
        let mut stored_bytes = 0u64;
        let mut representation = None;
        for request in &requests {
            for slot in &request.entry.slots {
                stored_bytes = stored_bytes.saturating_add(slot.total_size());
                if let Some(metadata) = &slot.encoding {
                    if metadata.len() != slot.num_segments() {
                        logical_bytes = None;
                    }
                    for meta in metadata {
                        logical_bytes = logical_bytes
                            .and_then(|bytes| bytes.checked_add(meta.logical_bytes as u64));
                        let next = Representation::from(meta.format);
                        representation = Some(match representation {
                            None => next,
                            Some(previous) if previous == next => previous,
                            Some(_) => Representation::Mixed,
                        });
                    }
                } else {
                    // Raw segment lengths include allocation alignment. Only
                    // the GPU layout owner knows their logical data size.
                    logical_bytes = None;
                    representation = Some(match representation {
                        None | Some(Representation::Raw) => Representation::Raw,
                        Some(_) => Representation::Mixed,
                    });
                }
            }
        }
        let observation = Observation::new(
            CostEstimateKey::new(
                CostObservationKind::LocalSsdHostReady,
                resource,
                representation.unwrap_or(Representation::Unknown),
                stored_bytes,
                requests.len(),
            ),
            logical_bytes,
        );
        Self {
            requests,
            done_tx,
            observation,
            reservation,
        }
    }
}

/// Shared context for a batch of prefetch operations.
/// Collects successful blocks and delivers them as a batch when all reads finish.
pub(super) struct BatchContext {
    results: Mutex<crate::storage::MaterializedBlocks>,
    remaining: AtomicUsize,
    done_tx: Mutex<Option<oneshot::Sender<BatchResult>>>,
    observation: Mutex<Option<Observation>>,
    reservation: Mutex<Option<StagingReservation>>,
    failed: AtomicBool,
    unquiesced: AtomicBool,
    submitted: AtomicBool,
    /// Pinned generations stay owned until every queued/submitted read drains.
    leases: Vec<Arc<super::SsdReadLease>>,
}

impl BatchContext {
    fn new(
        count: usize,
        done_tx: oneshot::Sender<BatchResult>,
        observation: Observation,
        leases: Vec<Arc<super::SsdReadLease>>,
        reservation: Option<StagingReservation>,
    ) -> Self {
        Self {
            results: Mutex::new(Vec::with_capacity(count)),
            remaining: AtomicUsize::new(count),
            done_tx: Mutex::new(Some(done_tx)),
            observation: Mutex::new(Some(observation)),
            reservation: Mutex::new(reservation),
            failed: AtomicBool::new(false),
            unquiesced: AtomicBool::new(false),
            submitted: AtomicBool::new(false),
            leases,
        }
    }

    fn complete_one(&self, key: StateKey, block: Option<Arc<SealedBlock>>) {
        if let Some(block) = block {
            self.results.lock().push((key, block));
        } else {
            self.failed.store(true, Ordering::Release);
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(tx) = self.done_tx.lock().take()
        {
            let results = std::mem::take(&mut *self.results.lock());
            let failed = self.failed.load(Ordering::Acquire);
            let unquiesced = self.unquiesced.load(Ordering::Acquire);
            // The detached file owner retains this context on uncertainty. Keep
            // its export reservation charged even after logical failure delivery.
            let reservation = if unquiesced {
                None
            } else {
                self.reservation.lock().take()
            };
            let mut prepared = None;
            let result = if unquiesced {
                Err(PeerError::StagingFailed)
            } else if let Some(reservation) = reservation {
                if failed {
                    Err(PeerError::StagingFailed)
                } else {
                    match reservation.commit(results.clone()) {
                        Ok(transfer) => {
                            prepared = Some(transfer);
                            Ok(results)
                        }
                        Err(error) => Err(error),
                    }
                }
            } else {
                Ok(results)
            };
            let terminal_failed = failed || result.is_err();
            let delivered = tx.send(result).is_ok();
            if delivered && let Some(transfer) = prepared {
                transfer.publish();
            }
            if let Some(observation) = self.observation.lock().take() {
                let outcome = if terminal_failed {
                    Outcome::Failed
                } else if !delivered {
                    Outcome::Cancelled
                } else {
                    Outcome::Completed
                };
                // io_uring owns physical byte accounting. This composite also
                // includes allocation, queueing, validation and reconstruction.
                observation.finish(outcome, None);
            }
        }
    }
}

/// Internal: single block prefetch task with per-slot allocated memory.
struct PrefetchTask {
    key: StateKey,
    entry: SsdIndexEntry,
    /// One per slot (parallel to `entry.slots`), each from the correct NUMA pool.
    slots: Vec<RawBlock>,
    /// Shared batch context: per-block callback + completion counter.
    ctx: Arc<BatchContext>,
    store: Arc<SsdStore>,
}

/// SSD prefetch entry point. Spawns dispatcher + worker pipeline internally.
pub(super) async fn ssd_prefetch_loop(
    store: Weak<SsdStore>,
    rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    io: Arc<SsdIo>,
    prefetch_inflight: usize,
) {
    let prefetch_inflight = prefetch_inflight.max(1);

    // Bounded channel: capacity = max inflight tasks
    let (task_tx, task_rx) = tokio::sync::mpsc::channel(prefetch_inflight);

    // Spawn dispatcher and worker
    let dispatcher = tokio::spawn(ssd_prefetch_dispatcher(store.clone(), rx, task_tx));
    let worker = tokio::spawn(ssd_prefetch_worker(store, task_rx, io, prefetch_inflight));

    // Wait for both to complete
    let _ = dispatcher.await;
    let _ = worker.await;

    debug!("SSD prefetch pipeline exiting");
}

/// Dispatcher: receives batches, allocates page segments on their NUMA node,
/// then submits block-level read tasks.
async fn ssd_prefetch_dispatcher(
    store: Weak<SsdStore>,
    mut batch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    task_tx: tokio::sync::mpsc::Sender<PrefetchTask>,
) {
    while let Some(batch) = batch_rx.recv().await {
        if batch.requests.is_empty() {
            let result = if batch.reservation.is_some() {
                Err(PeerError::StagingFailed)
            } else {
                Ok(Vec::new())
            };
            let _ = batch.done_tx.send(result);
            continue;
        }

        let Some(s) = store.upgrade() else { break };
        if !dispatch_prefetch_batch(&s, &task_tx, batch).await {
            return;
        }
    }

    debug!("SSD prefetch dispatcher exiting");
}

/// Allocate each stored segment independently, then enqueue block reads.
/// Read and write allocations use the same page/segment lifetime and sizes;
/// a surviving prefix cannot pin unrelated pages from a larger batch.
async fn dispatch_prefetch_batch(
    store: &Arc<SsdStore>,
    task_tx: &tokio::sync::mpsc::Sender<PrefetchTask>,
    batch: PrefetchBatch,
) -> bool {
    let PrefetchBatch {
        requests,
        done_tx,
        mut observation,
        reservation,
    } = batch;
    observation.admitted();
    let mut block_slots = Vec::with_capacity(requests.len());
    for req in &requests {
        let mut slots = Vec::with_capacity(req.entry.slots.len());
        for meta in &req.entry.slots {
            let numa_node =
                (store.is_numa() && !meta.numa_node.is_unknown()).then_some(meta.numa_node);
            let mut segments = Vec::with_capacity(meta.segment_sizes.len());
            for &size in &meta.segment_sizes {
                let Some(allocation) = store.allocate_prefetch(size, numa_node) else {
                    warn!(
                        "SSD prefetch dispatcher: alloc failed for {size} bytes numa={numa_node:?}, failing entire batch"
                    );
                    observation.finish(Outcome::Failed, None);
                    let result = if reservation.is_some() {
                        Err(PeerError::StagingFailed)
                    } else {
                        Ok(Vec::new())
                    };
                    let _ = done_tx.send(result);
                    return true;
                };
                segments.push(Segment::new(
                    allocation.mapped_ptr().host(),
                    size as usize,
                    allocation,
                ));
            }
            let mut raw = RawBlock::new(segments);
            raw.encoding = meta.encoding.clone();
            slots.push(raw);
        }
        block_slots.push(slots);
    }

    let ctx = Arc::new(BatchContext::new(
        requests.len(),
        done_tx,
        observation,
        requests,
        reservation,
    ));
    let mut iter = ctx.leases.iter().zip(block_slots);
    while let Some((req, slots)) = iter.next() {
        let task = PrefetchTask {
            key: req.key.clone(),
            entry: req.entry.clone(),
            slots,
            ctx: Arc::clone(&ctx),
            store: Arc::clone(store),
        };

        if let Err(err) = task_tx.send(task).await {
            debug!("SSD prefetch dispatcher: worker channel closed");
            let task = err.0;
            ctx.complete_one(task.key, None);
            for (req, _) in iter {
                ctx.complete_one(req.key.clone(), None);
            }
            return false;
        }
    }
    true
}

/// Worker: maintains FuturesUnordered with max_inflight concurrent I/O operations.
async fn ssd_prefetch_worker(
    store: Weak<SsdStore>,
    mut task_rx: tokio::sync::mpsc::Receiver<PrefetchTask>,
    io: Arc<SsdIo>,
    max_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type PrefetchFuture = Pin<Box<dyn Future<Output = SingleMaterializedBlocks> + Send>>;

    let metrics = core_metrics();
    let mut inflight: FuturesUnordered<PrefetchFuture> = FuturesUnordered::new();

    loop {
        tokio::select! {
            biased;

            // Complete finished tasks first (priority)
            Some((key, entry, result, duration_secs, block_size, ctx)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_prefetch_inflight.add(-1, &[]);

                // Validate data wasn't overwritten during read
                let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(&entry));
                let result = if result.is_some() && !valid {
                    warn!("SSD prefetch: data overwritten during read, discarding");
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                } else if result.is_some() {
                    metrics.ssd_prefetch_success.add(1, &[]);
                    metrics.ssd_prefetch_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_prefetch_throughput_bytes_per_second.record(throughput, &[]);
                    result
                } else {
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                };
                ctx.complete_one(key, result);
            }

            // Accept new task if below limit
            task = task_rx.recv(), if inflight.len() < max_inflight => {
                match task {
                    Some(t) => {
                        metrics.ssd_prefetch_inflight.add(1, &[]);
                        inflight.push(Box::pin(execute_prefetch(t, io.clone())));
                    }
                    None => {
                        // Channel closed, drain remaining
                        break;
                    }
                }
            }
        }
    }

    // Drain remaining inflight tasks
    while let Some((key, entry, result, duration_secs, block_size, ctx)) = inflight.next().await {
        metrics.ssd_prefetch_inflight.add(-1, &[]);

        let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(&entry));
        let result = if result.is_some() && !valid {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        } else if result.is_some() {
            metrics.ssd_prefetch_success.add(1, &[]);
            metrics.ssd_prefetch_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_prefetch_throughput_bytes_per_second
                .record(throughput, &[]);
            result
        } else {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        };
        ctx.complete_one(key, result);
    }

    debug!("SSD prefetch worker exiting");
}

/// Execute a single prefetch operation.
async fn execute_prefetch(task: PrefetchTask, io: Arc<SsdIo>) -> SingleMaterializedBlocks {
    let start = Instant::now();
    let duration_secs = || start.elapsed().as_secs_f64();

    let key = task.key;
    let block_size: u64 = task
        .entry
        .slots
        .iter()
        .map(crate::SlotMeta::total_size)
        .sum();
    let ctx = task.ctx;

    let expected_len = block_size as usize;
    let read_result = {
        let mut offset = task.entry.file_offset;
        let slices = task
            .slots
            .iter()
            .flat_map(|slot| slot.segment_iovecs())
            .map(|(local, length)| {
                let slice = crate::transfer_engine::TransferSlice {
                    local,
                    remote_address: offset,
                    length,
                };
                offset += length as u64;
                slice
            })
            .collect();
        if !ctx.submitted.swap(true, Ordering::Relaxed)
            && let Some(observation) = ctx.observation.lock().as_mut()
        {
            observation.submitted();
        }
        io.submit(
            crate::transfer_engine::TransferOp::Read,
            task.entry.shard_id,
            slices,
            task.slots
                .iter()
                .flat_map(|slot| slot.allocations().cloned())
                .collect(),
            ctx.clone(),
        )
    };

    #[cfg(feature = "test-hooks")]
    crate::test_faults::pause("ssd").await;

    let result = match read_result {
        Ok(rx) => rx.await.unwrap_or_else(|error| {
            Err(crate::transfer_engine::TransferError::new(
                crate::transfer_engine::ErrorKind::Unquiesced,
                error.to_string(),
            ))
        }),
        Err(error) => Err(error),
    };
    if matches!(&result, Err(error) if error.kind == crate::transfer_engine::ErrorKind::Unquiesced)
    {
        ctx.unquiesced.store(true, Ordering::Release);
    }
    let read_ok = matches!(result, Ok(bytes) if bytes == expected_len);
    let slots = if !read_ok {
        warn!("SSD prefetch: failed or short read for {key:?}");
        None
    } else if task
        .slots
        .iter()
        .any(|slot| slot.validate_encoding().is_err())
    {
        core_metrics().storage_codec_decode_failures.add(1, &[]);
        task.store.invalidate_encoded_entry(&key, &task.entry);
        warn!("SSD prefetch: corrupt encoded object {key:?}");
        None
    } else {
        Some(task.slots)
    };
    let block = slots.map(|slots| {
        Arc::new(SealedBlock::from_slots(
            slots
                .into_iter()
                .zip(&task.entry.slots)
                .map(|(slot, meta)| (slot, meta.numa_node))
                .collect(),
        ))
    });
    (key, task.entry, block, duration_secs(), block_size, ctx)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/reader.rs"]
mod tests;
