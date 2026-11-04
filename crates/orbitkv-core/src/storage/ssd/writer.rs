use super::{SsdStore, index::Encoding, io::SsdIo};
use crate::PublishDiagnostic;
use crate::block::{SealedBlock, StateKey};
use crate::cost::{Observation, Outcome};
use crate::metrics::core_metrics;
use futures::stream::{FuturesOrdered, StreamExt};
use log::{debug, warn};
use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Batch of sealed blocks to write to SSD
pub(super) struct SsdWriteBatch {
    pub blocks: Vec<(StateKey, Weak<SealedBlock>)>,
    pub observation: Observation,
    pub diagnostic: Option<Box<PublishDiagnostic>>,
}

struct WriteBatchObservation {
    remaining: usize,
    outcome: Outcome,
    observation: Observation,
    diagnostic: Option<PublishDiagnostic>,
}

fn complete_write_batch(
    batches: &mut VecDeque<WriteBatchObservation>,
    success: bool,
    bytes: u64,
) -> Option<(PublishDiagnostic, Outcome)> {
    let batch = batches.front_mut()?;
    batch.remaining -= 1;
    if !success {
        if bytes != 0 {
            batch.outcome = Outcome::Failed;
        } else if batch.outcome == Outcome::Completed {
            batch.outcome = Outcome::Cancelled;
        }
    }
    if batch.remaining == 0
        && let Some(batch) = batches.pop_front()
    {
        // FuturesOrdered returns writes in submission order. Completion follows
        // every ring commit; the file engine counts actual bytes.
        batch.observation.finish(batch.outcome, None);
        return batch
            .diagnostic
            .map(|diagnostic| (diagnostic, batch.outcome));
    }
    None
}

/// Commands sent to the SSD writer task.
pub(super) enum SsdWriteCommand {
    Write(SsdWriteBatch),
    Flush(tokio::sync::oneshot::Sender<()>),
}

/// Internal: single block write task
struct WriteTask {
    key: StateKey,
    block: Arc<SealedBlock>,
}

/// Result: key, success, elapsed seconds, physical bytes (zero for unreserved skips).
type WriteResult = (StateKey, bool, f64, u64);

const fn outcome_name(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Completed => "Completed",
        Outcome::Failed => "Failed",
        Outcome::Cancelled => "Cancelled",
        Outcome::TimedOut => "TimedOut",
        Outcome::Abandoned => "Abandoned",
    }
}

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives batches of sealed blocks and writes them.
pub(super) async fn ssd_writer_loop(
    store: Weak<SsdStore>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
    io: Arc<SsdIo>,
    write_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type WriteFuture = Pin<Box<dyn Future<Output = WriteResult> + Send>>;

    let metrics = core_metrics();
    let max_inflight = write_inflight.max(1);

    let mut pending: VecDeque<WriteTask> = VecDeque::new();
    let mut inflight: FuturesOrdered<WriteFuture> = FuturesOrdered::new();
    let mut batches: VecDeque<WriteBatchObservation> = VecDeque::new();
    let mut flush_waiters: Vec<tokio::sync::oneshot::Sender<()>> = Vec::new();

    loop {
        // If a flush is pending and all work is drained, fire it.
        if !flush_waiters.is_empty() && pending.is_empty() && inflight.is_empty() {
            for tx in flush_waiters.drain(..) {
                let _ = tx.send(());
            }
        }

        tokio::select! {
            biased;

            // Priority 1: Complete writes
            Some((key, success, duration_secs, block_size)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_write_inflight.add(-1, &[]);

                if success {
                    metrics.ssd_write_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_write_throughput_bytes_per_second.record(throughput, &[]);
                } else if block_size != 0 {
                    metrics.ssd_write_failures.add(1, &[]);
                    warn!("SSD cache write failed for {:?}", key);
                }
                if let Some((diagnostic, outcome)) =
                    complete_write_batch(&mut batches, success, block_size)
                {
                    diagnostic.record(
                        "publish_ssd_complete",
                        orbitkv_common::timeline::DiagnosticFields::default()
                            .outcome(outcome_name(outcome))
                            .pending_blocks(pending.len())
                            .inflight_writes(inflight.len())
                            .max_inflight_writes(max_inflight),
                    );
                }
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                // Only the newest dequeued batch can still have pending tasks.
                // This composite starts service at worker admission; low-level
                // file submission and completion are observed by the engine boundary.
                if let Some(batch) = batches.back_mut() {
                    batch.observation.submitted();
                }
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, store.clone(), io.clone())));
            }

            // Priority 3: Receive new command
            cmd = rx.recv(), if pending.is_empty() && flush_waiters.is_empty() => {
                match cmd {
                    Some(SsdWriteCommand::Write(mut b)) => {
                        // Dequeue metric
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);
                        b.observation.admitted();
                        if let Some(diagnostic) = b.diagnostic.as_deref().copied() {
                            let blocks = b.blocks.len();
                            diagnostic.record(
                                "publish_ssd_dequeue",
                                orbitkv_common::timeline::DiagnosticFields::default()
                                    .blocks(blocks)
                                    .pending_blocks(pending.len())
                                    .inflight_writes(inflight.len())
                                    .max_inflight_writes(max_inflight),
                            );
                        }

                        let Some(s) = store.upgrade() else {
                            if let Some(diagnostic) = b.diagnostic.as_deref().copied() {
                                diagnostic.record(
                                    "publish_ssd_complete",
                                    orbitkv_common::timeline::DiagnosticFields::default()
                                        .outcome("Cancelled:store_dropped"),
                                );
                            }
                            b.observation.finish(Outcome::Cancelled, None);
                            continue;
                        };
                        let mut outcome = Outcome::Completed;
                        for (key, weak) in b.blocks {
                            if let Some(block) = weak.upgrade() {
                                pending.push_back(WriteTask { key, block });
                            } else {
                                s.inner.lock().pending_writes.remove(&key);
                                outcome = Outcome::Cancelled;
                            }
                        }
                        if pending.is_empty() {
                            if let Some(diagnostic) = b.diagnostic.as_deref().copied() {
                                diagnostic.record(
                                    "publish_ssd_complete",
                                    orbitkv_common::timeline::DiagnosticFields::default()
                                        .outcome("Cancelled:sources_dropped"),
                                );
                            }
                            b.observation.finish(Outcome::Cancelled, None);
                        } else {
                            batches.push_back(WriteBatchObservation {
                                remaining: pending.len(),
                                outcome,
                                observation: b.observation,
                                diagnostic: b.diagnostic.as_deref().copied(),
                            });
                        }
                    }
                    Some(SsdWriteCommand::Flush(tx)) => {
                        flush_waiters.push(tx);
                    }
                    None => break,
                }
            }
        }
    }

    // Drain remaining inflight writes
    drain_inflight(metrics, &mut inflight, &mut batches).await;

    // Fire any remaining flush waiters
    for tx in flush_waiters.drain(..) {
        let _ = tx.send(());
    }

    debug!("SSD writer task exiting");
}

async fn drain_inflight(
    metrics: &crate::metrics::CoreMetrics,
    inflight: &mut FuturesOrdered<
        std::pin::Pin<Box<dyn std::future::Future<Output = WriteResult> + Send>>,
    >,
    batches: &mut VecDeque<WriteBatchObservation>,
) {
    while let Some((key, success, duration_secs, block_size)) = inflight.next().await {
        metrics.ssd_write_inflight.add(-1, &[]);

        if success {
            metrics.ssd_write_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_write_throughput_bytes_per_second
                .record(throughput, &[]);
        } else if block_size != 0 {
            metrics.ssd_write_failures.add(1, &[]);
            warn!("SSD cache write failed for {:?}", key);
        }
        if let Some((diagnostic, outcome)) = complete_write_batch(batches, success, block_size) {
            diagnostic.record(
                "publish_ssd_complete",
                orbitkv_common::timeline::DiagnosticFields::default()
                    .outcome(outcome_name(outcome))
                    .pending_blocks(0)
                    .inflight_writes(inflight.len()),
            );
        }
    }
}

/// Reserve extents at the stored size; retain sealed buffers through I/O completion.
async fn execute_write(task: WriteTask, store: Weak<SsdStore>, io: Arc<SsdIo>) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let Some(store) = store.upgrade() else {
        return (key, false, 0.0, 0);
    };
    let encoded = task.block.slots().iter().any(|s| s.encoding.is_some());
    let slots = task
        .block
        .slots()
        .iter()
        .zip(task.block.slot_numas())
        .map(|(slot, numa)| {
            let mut meta = crate::SlotMeta::new(
                slot.segment_iovecs().map(|(_, size)| size as u64).collect(),
                *numa,
            );
            meta.encoding = slot.encoding.clone();
            meta
        })
        .collect();
    let encoding = if encoded {
        Encoding::Encoded
    } else {
        Encoding::Raw
    };
    let (entry, retired) = {
        let mut inner = store.inner.lock();
        let entry = inner.ring.reserve(&key, slots, encoding);
        let retired = inner.ring.take_retired();
        (entry, retired)
    };
    store.retire_inventory(retired);
    let Some(entry) = entry else {
        store.commit_write(&key, false);
        return (key, false, start.elapsed().as_secs_f64(), 0);
    };
    let lease = Arc::new(super::WriteLease {
        entry: entry.clone(),
        key: key.clone(),
        committed: std::sync::atomic::AtomicBool::new(false),
        gpu: false,
        store: Arc::clone(&store),
    });
    let bytes = entry.len;
    let rx = {
        let mut offset = entry.file_offset;
        let slices = task
            .block
            .slots()
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
        crate::transfer_engine::FileExtent::from_slices(
            entry.file_offset,
            entry.len as usize,
            slices,
        )
        .and_then(|extent| {
            io.submit(
                crate::transfer_engine::TransferOp::Write,
                entry.shard_id,
                extent,
                task.block.allocations().cloned().collect(),
                lease.clone(),
            )
        })
    };
    let success = match rx {
        Ok(rx) => matches!(rx.await, Ok(Ok(written)) if written as u64 == bytes),
        Err(_) => false,
    };
    if success {
        lease.commit();
    }
    let duration = start.elapsed().as_secs_f64();
    core_metrics()
        .ssd_write_duration_seconds
        .record(duration, &[]);
    (key, success, duration, bytes)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/writer.rs"]
mod tests;
