use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use bytesize::ByteSize;
use hashlink::LruCache;
use log::{debug, info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::block::{SealedBlock, StateKey};
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
};
use crate::memory::numa::NumaNode;
use crate::memory::pool::PinnedAllocation;
use crate::metrics::core_metrics;

mod config;
pub(crate) mod cufile;
mod files;
pub(crate) mod index;
pub(crate) mod metadata;
mod reader;
mod uring;
mod writer;

use super::MaterializedBlocks;
use crate::memory::{AllocateFn, AllocationFootprintFn};
pub(crate) use config::SSD_ALIGNMENT;
pub use config::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdBackend, SsdCacheConfig, SsdReadPath, SsdWritePolicy,
};
use cufile::CufileFile;
use index::{SsdIndexEntry, SsdRingBuffer};
use reader::{PrefetchBatch, ssd_prefetch_loop};
use uring::{UringConfig, UringIoEngine};
use writer::{SsdWriteBatch, SsdWriteCommand, ssd_writer_loop};

/// Owns an immutable SSD source until the last query/GPU consumer releases it.
pub struct SsdReadLease {
    pub(crate) entry: SsdIndexEntry,
    pub(crate) key: StateKey,
    pub(crate) store: Arc<SsdStore>,
}

/// An index snapshot does not reserve disk space or keep payload readers alive.
/// The readers allocation identifies the generation even if a ring offset wraps.
pub(crate) struct SsdReadCandidate {
    pub(crate) entry: SsdIndexEntry,
    key: StateKey,
    store: Weak<SsdStore>,
}

impl SsdReadCandidate {
    pub(crate) fn cufile_eligible(&self, codec_budget: usize) -> bool {
        self.store
            .upgrade()
            .is_some_and(|store| store.cufile_eligible(&self.entry, codec_budget))
    }

    pub(crate) fn pin(&self) -> Option<Arc<SsdReadLease>> {
        let store = self.store.upgrade()?;
        let inner = store.inner.lock();
        let current = inner.ring.get(&self.key)?;
        if !Arc::ptr_eq(&current.readers, &self.entry.readers) {
            return None;
        }
        current.readers.fetch_add(1, Ordering::Relaxed);
        core_metrics()
            .ssd_read_pinned_bytes
            .add(current.len as i64, &[]);
        let entry = current.clone();
        drop(inner);
        Some(Arc::new(SsdReadLease {
            entry,
            key: self.key.clone(),
            store,
        }))
    }

    #[cfg(feature = "peer")]
    pub(crate) fn cost_resource(&self) -> Option<ExecutionResource> {
        Some(self.store.upgrade()?.io.cost_resource)
    }

    #[cfg(feature = "peer")]
    pub(crate) fn replica_metadata(&self) -> orbitkv_state::ReplicaMetadata {
        self.entry.replica_metadata()
    }
}

impl SsdReadLease {
    pub(crate) fn cufile_eligible(&self, codec_budget: usize) -> bool {
        self.store.cufile_eligible(&self.entry, codec_budget)
    }

    pub(crate) fn file(&self) -> Result<&Arc<CufileFile>, crate::EngineError> {
        self.store
            .cufile_files
            .get(self.entry.shard_id)
            .ok_or_else(|| {
                crate::EngineError::Storage("SSD source has no cuFile registration".into())
            })
    }

    pub(crate) fn cost_resource(&self) -> ExecutionResource {
        self.store.io.cost_resource
    }

    pub(crate) fn invalidate_encoded(&self) {
        self.store.invalidate_encoded_entry(&self.key, &self.entry);
    }
}

impl Drop for SsdReadLease {
    fn drop(&mut self) {
        if self.entry.readers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.store
                .inner
                .lock()
                .ring
                .release_invalid(&self.key, &self.entry);
        }
        core_metrics()
            .ssd_read_pinned_bytes
            .add(-(self.entry.len as i64), &[]);
    }
}

/// An unpublished extent. Dropping a failed/unsubmitted write rolls it back;
/// the GPU worker commits only after every cuFile chunk has completed.
pub(crate) struct GpuWriteLease {
    pub(crate) entry: SsdIndexEntry,
    key: Option<StateKey>,
    store: Arc<SsdStore>,
}

impl GpuWriteLease {
    pub(crate) fn file(&self) -> &Arc<CufileFile> {
        &self.store.cufile_files[self.entry.shard_id]
    }

    pub(crate) fn commit(mut self) {
        if let Some(key) = self.key.take() {
            self.store.commit_write(&key, true);
            core_metrics().ssd_write_bytes.add(self.entry.len, &[]);
        }
    }
}

impl Drop for GpuWriteLease {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            self.store.commit_write(&key, false);
            core_metrics().ssd_write_failures.add(1, &[]);
        }
        core_metrics().ssd_write_inflight.add(-1, &[]);
    }
}

struct SsdInner {
    ring: SsdRingBuffer,
    pending_writes: HashSet<StateKey>,
    reuse_history: LruCache<StateKey, ()>,
}

const REUSE_HISTORY_BLOCKS: usize = 16_384;

/// Shared admission state. Disabling new GPU I/O never revokes existing leases.
pub(crate) struct GpuIo {
    automatic: bool,
    enabled: AtomicBool,
}

impl GpuIo {
    pub(crate) fn available(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn failed(&self, error: &str) {
        if self.automatic && self.enabled.swap(false, Ordering::AcqRel) {
            core_metrics().ssd_backend_fallbacks.add(1, &[]);
            warn!("SSD auto backend selected uring for new operations: {error}");
        }
    }
}

impl SsdInner {
    fn admission_skip(
        &mut self,
        key: &StateKey,
        reused: bool,
        policy: SsdWritePolicy,
    ) -> Option<&'static str> {
        if self.ring.get(key).is_some() {
            return Some("resident");
        }
        if self.pending_writes.contains(key) {
            return Some("pending");
        }
        if policy == SsdWritePolicy::Reuse {
            let seen = self.reuse_history.get(key).is_some();
            self.reuse_history.insert(key.clone(), ());
            if !reused && !seen {
                return Some("cold");
            }
        }
        None
    }
}

pub(crate) struct SsdStore {
    pub(crate) gpu_io: Arc<GpuIo>,
    pub(crate) read_path: Option<SsdReadPath>,
    /// Keeps file descriptors alive for io_uring operations.
    _files: Vec<std::fs::File>,
    cufile_files: Vec<Arc<CufileFile>>,
    io: Arc<UringIoEngine>,
    write_tx: tokio::sync::mpsc::Sender<SsdWriteCommand>,
    write_policy: SsdWritePolicy,
    prefetch_tx: tokio::sync::mpsc::Sender<PrefetchBatch>,
    inner: Mutex<SsdInner>,
    allocate_fn: AllocateFn,
    allocation_footprint_fn: AllocationFootprintFn,
    is_numa: bool,
    inventory: Option<Arc<crate::storage::inventory::ResidencyInventory>>,
}

impl SsdStore {
    fn cufile_eligible(&self, entry: &SsdIndexEntry, codec_budget: usize) -> bool {
        self.gpu_io.available()
            && self.cufile_files.get(entry.shard_id).is_some()
            && entry.fits_gpu_decode(codec_budget)
    }
    pub(crate) fn reserve_gpu(
        self: &Arc<Self>,
        key: StateKey,
        slots: Vec<crate::SlotMeta>,
    ) -> Option<GpuWriteLease> {
        if !self.gpu_io.available() {
            return None;
        }
        let mut inner = self.inner.lock();
        if self.write_policy == SsdWritePolicy::Reuse && !inner.reuse_history.contains_key(&key) {
            return None;
        }
        if inner
            .admission_skip(&key, false, self.write_policy)
            .is_some()
        {
            return None;
        }
        let encoding = if slots.iter().any(|slot| slot.encoding.is_some()) {
            index::Encoding::Encoded
        } else {
            index::Encoding::Raw
        };
        let entry = inner.ring.reserve(&key, slots, encoding)?;
        let retired = inner.ring.take_retired();
        inner.pending_writes.insert(key.clone());
        core_metrics().ssd_write_inflight.add(1, &[]);
        drop(inner);
        self.retire_inventory(retired);
        Some(GpuWriteLease {
            entry,
            key: Some(key),
            store: Arc::clone(self),
        })
    }

    pub(crate) fn new(
        config: SsdCacheConfig,
        allocate_fn: AllocateFn,
        allocation_footprint_fn: AllocationFootprintFn,
        is_numa: bool,
        inventory: Option<Arc<crate::storage::inventory::ResidencyInventory>>,
    ) -> std::io::Result<Arc<Self>> {
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        let shards_per_path = config.shards.get();
        let total_shards = config.cache_paths.len() * shards_per_path;
        let try_gpu = config.backend != SsdBackend::Uring
            && (config.backend == SsdBackend::Cufile
                || config.capacity_bytes / total_shards.max(1) as u64 >= cufile::ALIGNMENT as u64);
        let gpu_io = Arc::new(GpuIo {
            automatic: config.backend == SsdBackend::Auto,
            enabled: AtomicBool::new(try_gpu),
        });
        let alignment = if try_gpu {
            cufile::ALIGNMENT
        } else {
            SSD_ALIGNMENT
        };
        let shard_capacity =
            files::aligned_shard_capacity(config.capacity_bytes, total_shards, alignment)?;
        let files = files::open_cache_files(
            &config.cache_paths,
            shards_per_path,
            shard_capacity,
            &mut OpenOptions::new(),
        )?;
        let fds: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();
        let cufile_files = if try_gpu {
            let registered = files
                .iter()
                .map(|file| {
                    CufileFile::new(file.try_clone()?, Arc::clone(&gpu_io))
                        .map(Arc::new)
                        .map_err(std::io::Error::other)
                })
                .collect::<std::io::Result<Vec<_>>>();
            match registered {
                Ok(files) => files,
                Err(error) if gpu_io.automatic => {
                    gpu_io.failed(&error.to_string());
                    Vec::new()
                }
                Err(error) => return Err(error),
            }
        } else {
            Vec::new()
        };
        if !cufile_files.is_empty() {
            files::reserve_cache_space(&files, shard_capacity)?;
        }
        let ring_alignment = if gpu_io.available() {
            cufile::ALIGNMENT
        } else {
            SSD_ALIGNMENT
        };

        let io = Arc::new(UringIoEngine::new_multi(fds, UringConfig::default())?);

        let (write_tx, write_rx) = tokio::sync::mpsc::channel(config.write_queue_depth);
        let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(config.prefetch_queue_depth);

        let capacity = shard_capacity * total_shards as u64;
        let write_inflight = config.write_inflight;
        let prefetch_inflight = config.prefetch_inflight;

        info!(
            "SSD cache initialized at {} (capacity {}, shards {}, shard capacity {})",
            config
                .cache_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            ByteSize(capacity),
            total_shards,
            ByteSize(shard_capacity)
        );
        info!(
            "SSD backend requested={:?} selected={}",
            config.backend,
            if gpu_io.available() {
                "cufile"
            } else {
                "uring"
            },
        );

        let store = Arc::new(Self {
            gpu_io,
            read_path: config.read_path,
            _files: files,
            cufile_files,
            io: Arc::clone(&io),
            write_tx,
            write_policy: config.write_policy,
            prefetch_tx,
            inner: Mutex::new(SsdInner {
                ring: SsdRingBuffer::new_sharded(
                    vec![shard_capacity; total_shards],
                    ring_alignment as u64,
                ),
                pending_writes: HashSet::new(),
                reuse_history: LruCache::new(REUSE_HISTORY_BLOCKS),
            }),
            allocate_fn,
            allocation_footprint_fn,
            is_numa,
            inventory,
        });

        Self::spawn_workers(
            &store,
            write_rx,
            prefetch_rx,
            write_inflight,
            prefetch_inflight,
        );

        Ok(store)
    }

    pub(crate) fn discover(self: &Arc<Self>, keys: &[StateKey]) -> Vec<Option<SsdReadCandidate>> {
        let inner = self.inner.lock();
        keys.iter()
            .map(|key| {
                let entry = inner.ring.get(key)?.clone();
                Some(SsdReadCandidate {
                    entry,
                    key: key.clone(),
                    store: Arc::downgrade(self),
                })
            })
            .collect()
    }

    /// Validate the advertised inventory versions and pin their exact SSD
    /// generations under the index lock before source staging begins.
    pub(crate) fn pin_residencies(
        self: &Arc<Self>,
        records: &[orbitkv_state::InventoryRecord],
    ) -> Option<Vec<Arc<SsdReadLease>>> {
        let inner = self.inner.lock();
        let inventory = self.inventory.as_ref()?;
        if records.is_empty()
            || !records
                .iter()
                .all(|record| inventory.contains_record(record, orbitkv_state::ReplicaMedium::Ssd))
        {
            return None;
        }
        let entries = records
            .iter()
            .map(|record| inner.ring.get(&record.key).cloned())
            .collect::<Option<Vec<_>>>()?;
        let leases = records
            .iter()
            .zip(entries)
            .map(|(record, entry)| {
                entry.readers.fetch_add(1, Ordering::Relaxed);
                core_metrics()
                    .ssd_read_pinned_bytes
                    .add(entry.len as i64, &[]);
                Arc::new(SsdReadLease {
                    entry,
                    key: record.key.clone(),
                    store: Arc::clone(self),
                })
            })
            .collect();
        Some(leases)
    }

    /// Exact upper bound for the pool allocations made by the host reader.
    /// Each stored segment is materialized into its own rounded allocation.
    pub(crate) fn staging_footprint(&self, leases: &[Arc<SsdReadLease>]) -> Option<u64> {
        let mut bytes = 0u64;
        for lease in leases {
            if !std::ptr::eq(self, Arc::as_ptr(&lease.store)) {
                return None;
            }
            for slot in &lease.entry.slots {
                let numa_node = if self.is_numa {
                    if slot.numa_node.is_unknown() {
                        return None;
                    }
                    slot.numa_node
                } else {
                    NumaNode::UNKNOWN
                };
                for &size in &slot.segment_sizes {
                    let numa_node = (!numa_node.is_unknown()).then_some(numa_node);
                    let footprint = (self.allocation_footprint_fn)(size, numa_node)?;
                    bytes = bytes.checked_add(footprint)?;
                }
            }
        }
        Some(bytes)
    }

    pub(super) fn is_offset_valid(&self, entry: &SsdIndexEntry) -> bool {
        self.inner.lock().ring.is_offset_valid(entry)
    }

    pub(super) fn allocate_prefetch(
        &self,
        size: u64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        (self.allocate_fn)(size, numa_node)
    }

    pub(super) fn commit_write(&self, key: &StateKey, success: bool) {
        let mut inner = self.inner.lock();
        let committed = inner.ring.commit(key, success);
        let metadata = if committed {
            inner.ring.get(key).map(SsdIndexEntry::replica_metadata)
        } else {
            None
        };
        let retired = inner.ring.take_retired();
        inner.pending_writes.remove(key);
        drop(inner);
        self.retire_inventory(retired);
        if let (Some(inventory), Some(metadata)) = (&self.inventory, metadata) {
            inventory.change(key, orbitkv_state::ReplicaMedium::Ssd, Some(metadata));
        }
    }

    pub(super) fn invalidate_encoded_entry(&self, key: &StateKey, entry: &SsdIndexEntry) {
        let invalidated = self.inner.lock().ring.invalidate_encoded(key, entry);
        if invalidated && let Some(inventory) = &self.inventory {
            inventory.change(key, orbitkv_state::ReplicaMedium::Ssd, None);
        }
    }

    pub(super) fn retire_inventory(&self, keys: Vec<StateKey>) {
        if let Some(inventory) = &self.inventory {
            for key in keys {
                inventory.change(&key, orbitkv_state::ReplicaMedium::Ssd, None);
            }
        }
    }

    pub(super) fn is_numa(&self) -> bool {
        self.is_numa
    }

    fn spawn_workers(
        store: &Arc<Self>,
        write_rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
        prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
        write_inflight: usize,
        prefetch_inflight: usize,
    ) {
        let io = Arc::clone(&store.io);

        let writer_weak = Arc::downgrade(store);
        let writer_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_writer_loop(writer_weak, write_rx, writer_io, write_inflight).await;
        });

        let prefetch_weak = Arc::downgrade(store);
        let prefetch_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_prefetch_loop(prefetch_weak, prefetch_rx, prefetch_io, prefetch_inflight).await;
        });

        debug!("SSD backing store workers spawned");
    }

    /// Fire-and-forget write.
    ///
    /// Queue weak sources so waiting writes cannot prevent pressure eviction.
    /// Foreground demand may retry a selectively admitted page; speculative
    /// warming must not call this with `reused = true`.
    pub(crate) fn ingest_batch<'a>(
        &self,
        blocks: impl IntoIterator<Item = (&'a StateKey, &'a Arc<SealedBlock>)>,
        reused: bool,
    ) {
        self.ingest_batch_diagnostic(blocks, reused, None);
    }

    pub(crate) fn ingest_batch_diagnostic<'a>(
        &self,
        blocks: impl IntoIterator<Item = (&'a StateKey, &'a Arc<SealedBlock>)>,
        reused: bool,
        diagnostic: Option<crate::PublishDiagnostic>,
    ) {
        if reused && self.write_policy == SsdWritePolicy::All {
            return;
        }
        let mut inner = self.inner.lock();
        let metrics = core_metrics();
        let mut admitted = Vec::new();
        let mut seen_batch = HashSet::new();
        let observe = crate::cost::enabled();
        let mut logical_bytes = Some(0u64);
        let mut stored_bytes = 0u64;
        let mut fragments = 0;
        let mut representation = None;
        for (key, block) in blocks {
            let skip = if seen_batch.insert(key) {
                inner.admission_skip(key, reused, self.write_policy)
            } else {
                Some("duplicate")
            };
            if let Some(reason) = skip {
                metrics
                    .ssd_write_admission_skips
                    .add(1, &[opentelemetry::KeyValue::new("reason", reason)]);
            } else {
                if observe {
                    for slot in block.slots() {
                        fragments += slot.num_segments();
                        stored_bytes = stored_bytes.saturating_add(slot.memory_footprint());
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
                            logical_bytes = None;
                            representation = Some(match representation {
                                None | Some(Representation::Raw) => Representation::Raw,
                                Some(_) => Representation::Mixed,
                            });
                        }
                    }
                }
                admitted.push((key.clone(), Arc::downgrade(block)));
            }
        }
        if admitted.is_empty() {
            return;
        }
        let len = admitted.len();
        let observation = Observation::new(
            CostEstimateKey::new(
                CostObservationKind::SsdWriteBatch,
                self.io.cost_resource,
                representation.unwrap_or(Representation::Unknown),
                logical_bytes.unwrap_or(stored_bytes),
                fragments,
            ),
            logical_bytes,
        );
        match self.write_tx.try_reserve() {
            Ok(permit) => {
                inner
                    .pending_writes
                    .extend(admitted.iter().map(|(key, _)| key.clone()));
                metrics.ssd_write_queue_pending.add(len as i64, &[]);
                if let Some(diagnostic) = diagnostic {
                    diagnostic.record(
                        "publish_ssd_enqueue",
                        orbitkv_common::timeline::DiagnosticFields::default().blocks(len),
                    );
                }
                permit.send(SsdWriteCommand::Write(SsdWriteBatch {
                    blocks: admitted,
                    observation,
                    diagnostic: diagnostic.map(Box::new),
                }));
            }
            Err(_) => {
                warn!("SSD write queue full, dropping {len} blocks");
                metrics.ssd_write_queue_full.add(len as u64, &[]);
                if let Some(diagnostic) = diagnostic {
                    diagnostic.record(
                        "publish_ssd_rejected",
                        orbitkv_common::timeline::DiagnosticFields::default()
                            .blocks(len)
                            .outcome("queue_full"),
                    );
                }
                observation.finish(Outcome::Cancelled, None);
            }
        }
    }

    /// Flush the SSD writer: waits until all enqueued writes complete.
    pub(crate) async fn flush(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.write_tx.send(SsdWriteCommand::Flush(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }

    /// Submit only acquired generations. The queue and batch completion owner
    /// retain every source through cancellation and the last physical read.
    pub(crate) async fn read_host_batch(
        &self,
        leases: Vec<Arc<SsdReadLease>>,
    ) -> Result<MaterializedBlocks, crate::EngineError> {
        if leases
            .iter()
            .any(|lease| !std::ptr::eq(self, Arc::as_ptr(&lease.store)))
        {
            return Err(crate::EngineError::Storage(
                "SSD leases belong to another store".into(),
            ));
        }
        if leases.is_empty() {
            return Ok(Vec::new());
        }
        let started = std::time::Instant::now();
        let (done_tx, done_rx) = oneshot::channel();
        let batch = PrefetchBatch::new(leases, done_tx, self.io.cost_resource);
        if let Err(error) = self.prefetch_tx.send(batch).await {
            core_metrics()
                .ssd_prefetch_queue_closed
                .add(error.0.requests.len() as u64, &[]);
            error.0.observation.finish(Outcome::Failed, None);
            return Err(crate::EngineError::Storage(
                "SSD host reader is closed".into(),
            ));
        }
        let result = done_rx
            .await
            .map_err(|_| crate::EngineError::Storage("SSD host reader lost completion".into()))?
            .map_err(|_| crate::EngineError::Storage("SSD host reader failed".into()));
        core_metrics()
            .ssd_prefetch_duration_seconds
            .record(started.elapsed().as_secs_f64(), &[]);
        result
    }

    /// Materialize a remote export while the detached batch owns its source
    /// admission. A successful result atomically publishes the transfer grant.
    pub(crate) async fn read_host_batch_for_export(
        &self,
        leases: Vec<Arc<SsdReadLease>>,
        reservation: crate::peer::export::StagingReservation,
    ) -> Result<MaterializedBlocks, crate::PeerError> {
        if leases.is_empty()
            || leases
                .iter()
                .any(|lease| !std::ptr::eq(self, Arc::as_ptr(&lease.store)))
        {
            return Err(crate::PeerError::StagingFailed);
        }
        let started = std::time::Instant::now();
        let count = leases.len();
        let (done_tx, done_rx) = oneshot::channel();
        let batch = PrefetchBatch::for_export(leases, done_tx, self.io.cost_resource, reservation);
        if let Err(error) = self.prefetch_tx.send(batch).await {
            core_metrics()
                .ssd_prefetch_queue_closed
                .add(count as u64, &[]);
            error.0.observation.finish(Outcome::Failed, None);
            return Err(crate::PeerError::StagingFailed);
        }
        let result = done_rx
            .await
            .unwrap_or(Err(crate::PeerError::StagingFailed));
        core_metrics()
            .ssd_prefetch_duration_seconds
            .record(started.elapsed().as_secs_f64(), &[]);
        result
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/mod.rs"]
mod tests;
