use crate::transfer::copy_and_drain;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Weak, mpsc as std_mpsc};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use log::{debug, error, info, warn};
use logforth::diagnostic::ThreadLocalDiagnostic;
use parking_lot::Mutex;
use tokio::sync::{OnceCell, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::CompletionResourceEvidence;
use crate::EngineError;
use crate::block::{RawBlock, SealedBlock};
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
    enabled, record_resource_evidence, shadow,
};
use crate::engine::instance::InstanceContext;
use crate::memory::numa::{NumaNode, pin_thread_to_numa_node};
use crate::metrics::core_metrics;
use crate::planning::restore::RestorePlan;
use crate::transfer::layout::{BlockCopies, KVCacheLayout};
use crate::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend, TransferMode};

mod codec;
mod restore;
pub(crate) use codec::SaveGroup;
pub(crate) mod ssd;
use ssd::GpuWrite;

/// A task to restore KV blocks from leased sources to GPU layers
pub(crate) struct LoadTask {
    pub plan: RestorePlan,
    pub layers: Vec<LayerTransferData>,
    pub completion: oneshot::Sender<LoadOutcome>,
    pub reservations: Vec<crate::QueryReservation>,
    pub codec_budget: usize,
    pub decode_ready_started: Instant,
    pub decode_ready_observation: Box<Observation>,
    pub decode_admission: Option<DecodeRestorePermit>,
}

/// Terminal GPU transfer evidence, timestamped before notifying the dispatcher.
#[derive(Debug)]
pub struct LoadOutcome {
    pub result: Result<(), EngineError>,
    pub completed_at: Instant,
}

/// One layer in a transfer task (either direction): GPU layout plus the
/// blocks to move.
pub(crate) struct LayerTransferData {
    pub layer_name: String,
    pub layout: KVCacheLayout,
    pub blocks: Vec<TransferBlock>,
}

/// Owned payload or leased source for a GPU transfer.
pub(crate) enum TransferPayload {
    /// Save path: uniquely owned, moved through the GPU worker and returned.
    Owned(RawBlock),
    /// Codec saves allocate host residency after the GPU reports its encoded size.
    Pending,
    Ssd {
        source: Arc<crate::SsdReadLease>,
        path: crate::SsdReadPath,
        slot_id: usize,
        offset: usize,
    },
    /// Load path: slot borrowed from a cached sealed block.
    Cached {
        sealed: Arc<SealedBlock>,
        slot_id: usize,
        /// Byte offset into the slot's `RawBlock` where this layer begins.
        /// Page-first reads every layer from one page slot at distinct
        /// offsets; layer-first is always 0 (the slot is the layer).
        offset: usize,
    },
}

impl TransferPayload {
    pub(crate) fn raw(&self) -> &RawBlock {
        match self {
            Self::Pending => unreachable!("unallocated save"),
            Self::Ssd { .. } => unreachable!("SSD sources do not own host memory"),
            Self::Owned(block) => block,
            Self::Cached {
                sealed, slot_id, ..
            } => sealed
                .get_slot(*slot_id)
                .expect("cached slot_id validated at construction"),
        }
    }

    /// Byte offset of this layer within its host slot (0 unless page-first).
    fn host_offset(&self) -> usize {
        match self {
            Self::Ssd { offset, .. } => *offset,
            Self::Owned(_) | Self::Pending => 0,
            Self::Cached { offset, .. } => *offset,
        }
    }
}

/// One block in a transfer: the GPU slot index and its host-side image.
/// Loads copy `block` -> GPU slot, saves copy GPU slot -> `block`.
pub(crate) struct TransferBlock {
    pub block_idx: usize,
    pub block: TransferPayload,
}

/// A task to save KV blocks from GPU to CPU for multiple layers.
/// Raw saves own preallocated buffers; encoded saves allocate after GPU encoding.
/// The worker retains source mappings and destinations through transfer completion.
pub(crate) struct SaveTask {
    pub layers: Vec<LayerTransferData>,
    pub reply: oneshot::Sender<Result<Vec<LayerTransferData>, EngineError>>,
    pub ssd_writes: Vec<GpuWrite>,
    pub codec_groups: Vec<SaveGroup>,
    pub storage: Option<Arc<crate::storage::Storage>>,
    pub numa: NumaNode,
    pub ssd_admission: Option<OwnedSemaphorePermit>,
    admission: Arc<SaveAdmission>,
    _instance: Arc<InstanceContext>,
    #[cfg(feature = "tracing")]
    pub trace_ctx: Option<::fastrace::prelude::SpanContext>,
}

enum WorkerCommand {
    Load(LoadTask, Observation),
    Save(SaveTask, Observation),
    Drain(oneshot::Sender<Result<(), String>>),
}

/// Independent memory read/write lanes and one lazily allocated GPU storage lane.
pub(crate) struct GpuWorkerPool {
    device_id: i32,
    numa_node: NumaNode,
    pub(crate) transfer_mode: TransferMode,
    ssd_tx: Mutex<Option<mpsc::UnboundedSender<WorkerCommand>>>,
    ssd_host_tx: Mutex<Option<mpsc::UnboundedSender<WorkerCommand>>>,
    codec_write_tx: Mutex<Option<mpsc::UnboundedSender<WorkerCommand>>>,
    ssd_write_admission: Arc<Semaphore>,
    decode_restore_admission: Arc<DeviceRestoreAdmission>,
    cufile_worker_admission: Arc<Semaphore>,
    cufile_worker_owner: Mutex<Option<OwnedSemaphorePermit>>,
    load_tx: mpsc::UnboundedSender<WorkerCommand>,
    save_tx: mpsc::UnboundedSender<WorkerCommand>,
    closed: Mutex<bool>,
    drained: OnceCell<Result<(), String>>,
}

const MAX_DEVICE_RESTORES: usize = 128;
const MAX_SAVE_OPERATIONS: usize = 1024;
const MAX_SAVE_METADATA_BYTES: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct SaveUsage {
    operations: usize,
    metadata_bytes: usize,
    logical_bytes: u64,
    retained: usize,
}

struct SaveAdmission {
    usage: Arc<Mutex<SaveUsage>>,
    metadata_bytes: usize,
    logical_bytes: u64,
    completed: std::sync::atomic::AtomicBool,
    retained: std::sync::atomic::AtomicBool,
}

impl SaveAdmission {
    fn acquire(metadata_bytes: usize, logical_bytes: u64) -> Result<Arc<Self>, EngineError> {
        static USAGE: LazyLock<Arc<Mutex<SaveUsage>>> =
            LazyLock::new(|| Arc::new(Mutex::new(SaveUsage::default())));
        Self::reserve(Arc::clone(&USAGE), metadata_bytes, logical_bytes)
    }

    fn reserve(
        usage: Arc<Mutex<SaveUsage>>,
        metadata_bytes: usize,
        logical_bytes: u64,
    ) -> Result<Arc<Self>, EngineError> {
        let mut state = usage.lock();
        let metadata = state.metadata_bytes.checked_add(metadata_bytes);
        let logical = state.logical_bytes.checked_add(logical_bytes);
        if state.operations >= MAX_SAVE_OPERATIONS
            || metadata.is_none_or(|bytes| bytes > MAX_SAVE_METADATA_BYTES)
            || logical.is_none_or(|bytes| bytes > i64::MAX as u64)
        {
            core_metrics().save_admission_rejections.add(1, &[]);
            return Err(EngineError::Storage("GPU save admission is full".into()));
        }
        state.operations += 1;
        state.metadata_bytes += metadata_bytes;
        state.logical_bytes += logical_bytes;
        drop(state);
        core_metrics().save_active_operations.add(1, &[]);
        core_metrics()
            .save_active_metadata_bytes
            .add(metadata_bytes as i64, &[]);
        core_metrics()
            .save_active_logical_bytes
            .add(logical_bytes as i64, &[]);
        Ok(Arc::new(Self {
            usage,
            metadata_bytes,
            logical_bytes,
            completed: std::sync::atomic::AtomicBool::new(false),
            retained: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    fn retain(&self) {
        let mut usage = self.usage.lock();
        if !self.completed.load(Ordering::Acquire) && !self.retained.swap(true, Ordering::AcqRel) {
            usage.retained += 1;
            core_metrics().save_retained_operations.add(1, &[]);
        }
    }

    fn complete(&self) {
        let mut usage = self.usage.lock();
        if self.completed.swap(true, Ordering::AcqRel) {
            return;
        }
        usage.operations -= 1;
        usage.metadata_bytes -= self.metadata_bytes;
        usage.logical_bytes -= self.logical_bytes;
        if self.retained.load(Ordering::Acquire) {
            usage.retained -= 1;
            core_metrics().save_retained_operations.add(-1, &[]);
        }
        core_metrics().save_active_operations.add(-1, &[]);
        core_metrics()
            .save_active_metadata_bytes
            .add(-(self.metadata_bytes as i64), &[]);
        core_metrics()
            .save_active_logical_bytes
            .add(-(self.logical_bytes as i64), &[]);
    }
}

impl Drop for SaveAdmission {
    fn drop(&mut self) {
        // Loss of every logical owner is not completion evidence. Keep the
        // process budget charged even when no worker can deliver a receipt.
        self.retain();
    }
}

#[allow(
    clippy::ptr_arg,
    reason = "admission charges retained vector capacity, not just populated elements"
)]
fn save_shape(
    layers: &Vec<LayerTransferData>,
    ssd_writes: &Vec<GpuWrite>,
    codec_groups: &Vec<SaveGroup>,
) -> Result<(usize, u64), EngineError> {
    let overflow = || EngineError::InvalidArgument("GPU save shape overflow".into());
    let mut metadata = std::mem::size_of::<SaveTask>();
    let mut logical = 0u64;
    let mut charge = |count: usize, size: usize| -> Result<(), EngineError> {
        metadata = metadata
            .checked_add(count.checked_mul(size).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        Ok(())
    };
    charge(layers.capacity(), std::mem::size_of::<LayerTransferData>())?;
    for layer in layers {
        charge(layer.layer_name.capacity(), 1)?;
        charge(
            layer.blocks.capacity(),
            std::mem::size_of::<TransferBlock>(),
        )?;
        for block in &layer.blocks {
            let bytes = match layer
                .layout
                .block_copies(block.block_idx)
                .map_err(EngineError::Storage)?
            {
                BlockCopies::Contiguous(copy) => copy.bytes,
                BlockCopies::Split { k, v } => k.bytes.checked_add(v.bytes).ok_or_else(overflow)?,
            };
            logical = logical.checked_add(bytes as u64).ok_or_else(overflow)?;
        }
    }
    charge(ssd_writes.capacity(), std::mem::size_of::<GpuWrite>())?;
    for write in ssd_writes {
        charge(
            write.batches.capacity(),
            std::mem::size_of::<crate::storage::ssd::cufile::IoBatch>(),
        )?;
        for batch in &write.batches {
            charge(
                batch.copies.capacity(),
                std::mem::size_of::<crate::storage::ssd::cufile::CopyRange>(),
            )?;
        }
    }
    charge(codec_groups.capacity(), std::mem::size_of::<SaveGroup>())?;
    for group in codec_groups {
        charge(group.key.namespace.capacity(), 1)?;
        charge(group.key.hash.capacity(), 1)?;
        charge(
            group.blocks.capacity(),
            std::mem::size_of::<(usize, usize)>(),
        )?;
    }
    Ok((metadata, logical))
}

struct DeviceRestoreAdmission {
    permits: Arc<Semaphore>,
    active: AtomicUsize,
}

pub(crate) struct DecodeRestorePermit {
    _permit: OwnedSemaphorePermit,
    admission: Arc<DeviceRestoreAdmission>,
    depth: u32,
}

impl Drop for DecodeRestorePermit {
    fn drop(&mut self) {
        self.admission.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One process-wide GPU-storage write budget per physical CUDA device. Worker
/// pools are instance-owned, but their staging pressure is not.
fn device_ssd_write_admission(device_id: i32) -> Arc<Semaphore> {
    static ADMISSIONS: LazyLock<Mutex<HashMap<i32, Weak<Semaphore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut admissions = ADMISSIONS.lock();
    admissions.retain(|_, admission| admission.strong_count() > 0);
    if let Some(admission) = admissions.get(&device_id).and_then(Weak::upgrade) {
        return admission;
    }
    let admission = Arc::new(Semaphore::new(ssd::MAX_WRITES));
    admissions.insert(device_id, Arc::downgrade(&admission));
    admission
}

/// A cuFile worker owns two persistent registered GPU staging slots. Until
/// staging is shared directly, one instance worker may own those slots per GPU.
fn device_cufile_worker_admission(device_id: i32) -> Arc<Semaphore> {
    static ADMISSIONS: LazyLock<Mutex<HashMap<i32, Weak<Semaphore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut admissions = ADMISSIONS.lock();
    admissions.retain(|_, admission| admission.strong_count() > 0);
    if let Some(admission) = admissions.get(&device_id).and_then(Weak::upgrade) {
        return admission;
    }
    let admission = Arc::new(Semaphore::new(1));
    admissions.insert(device_id, Arc::downgrade(&admission));
    admission
}

fn device_restore_admission(device_id: i32) -> Arc<DeviceRestoreAdmission> {
    static ADMISSIONS: LazyLock<Mutex<HashMap<i32, Weak<DeviceRestoreAdmission>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut admissions = ADMISSIONS.lock();
    admissions.retain(|_, admission| admission.strong_count() > 0);
    if let Some(admission) = admissions.get(&device_id).and_then(Weak::upgrade) {
        return admission;
    }
    let admission = Arc::new(DeviceRestoreAdmission {
        permits: Arc::new(Semaphore::new(MAX_DEVICE_RESTORES)),
        active: AtomicUsize::new(0),
    });
    admissions.insert(device_id, Arc::downgrade(&admission));
    admission
}

impl DeviceRestoreAdmission {
    fn try_acquire(self: &Arc<Self>) -> Result<DecodeRestorePermit, EngineError> {
        let permit = Arc::clone(&self.permits)
            .try_acquire_owned()
            .map_err(|_| EngineError::Storage("decode restore queue is full".into()))?;
        let depth = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        Ok(DecodeRestorePermit {
            _permit: permit,
            admission: Arc::clone(self),
            depth: depth as u32,
        })
    }
}

impl GpuWorkerPool {
    pub(crate) fn new(
        device_id: i32,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            device_id,
            numa_node,
            transfer_mode,
            load_tx: spawn_worker(device_id, numa_node, transfer_mode, "load")?,
            save_tx: spawn_worker(device_id, numa_node, transfer_mode, "save")?,
            ssd_tx: Mutex::new(None),
            ssd_host_tx: Mutex::new(None),
            codec_write_tx: Mutex::new(None),
            ssd_write_admission: device_ssd_write_admission(device_id),
            decode_restore_admission: device_restore_admission(device_id),
            cufile_worker_admission: device_cufile_worker_admission(device_id),
            cufile_worker_owner: Mutex::new(None),
            closed: Mutex::new(false),
            drained: OnceCell::new(),
        })
    }

    fn submit(&self, command: WorkerCommand, disk: bool) -> Result<(), EngineError> {
        let reject = |command: WorkerCommand| match command {
            WorkerCommand::Load(task, observation) => {
                observation.finish(Outcome::Failed, Some(0));
                task.decode_ready_observation.finish(Outcome::Failed, None);
            }
            WorkerCommand::Save(_, observation) => {
                observation.finish(Outcome::Failed, Some(0));
            }
            WorkerCommand::Drain(_) => {}
        };
        let closed = self.closed.lock();
        if *closed {
            reject(command);
            return Err(EngineError::Storage("GPU worker is draining".into()));
        }
        if matches!(&command, WorkerCommand::Save(task, _) if !task.codec_groups.is_empty()) {
            let mut sender = self.codec_write_tx.lock();
            if sender.is_none() {
                match spawn_worker(
                    self.device_id,
                    self.numa_node,
                    self.transfer_mode,
                    "encoded-writeback",
                ) {
                    Ok(worker) => *sender = Some(worker),
                    Err(error) => {
                        reject(command);
                        return Err(error);
                    }
                }
            }
            sender
                .as_ref()
                .expect("encoded writeback worker initialized")
                .send(command)
        } else if matches!(&command, WorkerCommand::Load(task, _) if task.plan.ssd_path() == Some(crate::SsdReadPath::Uring)) {
            let mut sender = self.ssd_host_tx.lock();
            if sender.is_none() {
                match spawn_worker(self.device_id, self.numa_node, self.transfer_mode, "ssd-host") {
                    Ok(worker) => *sender = Some(worker),
                    Err(error) => {
                        reject(command);
                        return Err(error);
                    }
                }
            }
            sender.as_ref().expect("SSD host worker initialized").send(command)
        } else if disk {
            let mut sender = self.ssd_tx.lock();
            if sender.is_none() {
                match spawn_worker(self.device_id, self.numa_node, self.transfer_mode, "ssd") {
                    Ok(worker) => *sender = Some(worker),
                    Err(error) => {
                        reject(command);
                        return Err(error);
                    }
                }
            }
            sender
                .as_ref()
                .expect("SSD worker initialized")
                .send(command)
        } else {
            match command {
                WorkerCommand::Load(..) => self.load_tx.send(command),
                _ => self.save_tx.send(command),
            }
        }
        .map_err(|error| {
            reject(error.0);
            EngineError::Storage(format!(
                "GPU worker channel closed for device {}",
                self.device_id
            ))
        })
    }

    fn own_cufile_worker(&self) -> bool {
        let mut owner = self.cufile_worker_owner.lock();
        if owner.is_some() {
            return true;
        }
        match Arc::clone(&self.cufile_worker_admission).try_acquire_owned() {
            Ok(permit) => {
                *owner = Some(permit);
                true
            }
            Err(_) => false,
        }
    }

    pub(crate) fn admit_restore(
        &self,
        plan: &mut RestorePlan,
        bytes: u64,
        fragments: usize,
    ) -> Result<DecodeRestorePermit, EngineError> {
        if plan.device_id() != self.device_id {
            return Err(EngineError::InvalidArgument(
                "restore admission device mismatch".into(),
            ));
        }
        let permit = self.decode_restore_admission.try_acquire()?;
        plan.bind_target_shape(bytes, fragments)
            .map_err(EngineError::InvalidArgument)?;
        Ok(permit)
    }

    pub(crate) fn submit_load(&self, mut task: LoadTask) -> Result<(), EngineError> {
        if task.plan.device_id() != self.device_id {
            return Err(EngineError::InvalidArgument(format!(
                "restore plan targets device {} but worker owns device {}",
                task.plan.device_id(),
                self.device_id
            )));
        }
        let layers = &mut task.layers;
        let mut targets = Vec::new();
        for layer in layers.iter() {
            for block in &layer.blocks {
                match layer
                    .layout
                    .block_copies(block.block_idx)
                    .map_err(EngineError::Storage)?
                {
                    BlockCopies::Contiguous(copy) => targets.push((copy.addr, copy.bytes)),
                    BlockCopies::Split { k, v } => {
                        targets.extend([(k.addr, k.bytes), (v.addr, v.bytes)]);
                    }
                }
            }
        }
        let target_bytes = targets.iter().try_fold(0u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes as u64)
                .ok_or_else(|| EngineError::InvalidArgument("decode page bytes overflow".into()))
        })?;
        let target_fragments = targets.len();
        crate::codec::gpu::validate_targets(targets).map_err(EngineError::Storage)?;
        let decode_admission =
            self.admit_restore(&mut task.plan, target_bytes, target_fragments)?;
        let decode_queue_depth = decode_admission.depth;
        task.decode_admission = Some(decode_admission);
        restore::validate_plan(&task.plan, layers)?;
        let mut ssd_path = task.plan.ssd_path();
        if ssd_path == Some(crate::SsdReadPath::Cufile)
            && layers.iter().flat_map(|layer| &layer.blocks).any(|block| {
                matches!(&block.block, TransferPayload::Ssd { source, .. } if !source.cufile_eligible(task.codec_budget))
            })
        {
            return Err(EngineError::Storage("cuFile read route is no longer eligible".into()));
        }
        if ssd_path == Some(crate::SsdReadPath::Cufile) && !self.own_cufile_worker() {
            task.plan
                .fallback_from_cufile()
                .map_err(EngineError::Storage)?;
            restore::set_ssd_path(layers, crate::SsdReadPath::Uring);
            core_metrics().ssd_gpu_read_fallbacks.add(1, &[]);
            ssd_path = task.plan.ssd_path();
        }
        let disk = ssd_path.is_some();
        let (observation, decode_ready_observation) = if enabled() {
            let (mut key, bytes) =
                transfer_key(layers, self.device_id, self.transfer_mode, false, disk);
            if let Some(path) = ssd_path {
                key = restore::cost_estimate_key(&task.plan, layers, self.transfer_mode, path, key);
                restore::shadow(layers, task.codec_budget, path, key);
            }
            let target_shape = task
                .plan
                .target_shape()
                .expect("restore target shape bound before cost observation");
            debug_assert_eq!(target_shape.device_id(), self.device_id);
            debug_assert_eq!(target_shape.bytes(), bytes);
            debug_assert_eq!(target_shape.fragments(), transfer_shape(&task.layers).1);
            let decode_resource = ExecutionResource::CacheRestore {
                source_set_hash: task.plan.source_set_hash(),
                destination_device: self.device_id as u64,
                copy_backend: self.transfer_mode as u8,
            };
            let decode_ready_key = key
                .with_observation_kind_and_resource(
                    CostObservationKind::CacheRestore,
                    decode_resource,
                )
                .with_source_shape(task.plan.source_bytes(), task.plan.source_fragments())
                .with_wire_bytes(task.plan.source_bytes());
            record_resource_evidence(
                decode_resource,
                CompletionResourceEvidence {
                    decode_page_bytes: target_shape.bytes(),
                    queue_depth: decode_queue_depth,
                    queue_parallelism: 1,
                    tent_inflight_bytes: 0,
                    tent_bandwidth_bytes_per_second: 0,
                },
                std::time::Duration::ZERO,
            );
            (
                Observation::new(key, Some(bytes)),
                Observation::new_enqueued(decode_ready_key, Some(bytes), task.decode_ready_started),
            )
        } else {
            (Observation::disabled(), Observation::disabled())
        };
        task.decode_ready_observation = Box::new(decode_ready_observation);
        self.submit(WorkerCommand::Load(task, observation), disk)
    }

    pub(crate) async fn batch_save(
        &self,
        instance: Arc<InstanceContext>,
        layers: Vec<LayerTransferData>,
        mut ssd_writes: Vec<GpuWrite>,
        mut codec_groups: Vec<SaveGroup>,
        storage: Option<Arc<crate::storage::Storage>>,
    ) -> Result<Vec<LayerTransferData>, EngineError> {
        let (metadata_bytes, logical_bytes) = save_shape(&layers, &ssd_writes, &codec_groups)?;
        let admission = SaveAdmission::acquire(metadata_bytes, logical_bytes)?;
        let (reply, receiver) = oneshot::channel();
        if (!ssd_writes.is_empty() || !codec_groups.is_empty()) && !self.own_cufile_worker() {
            ssd_writes.clear();
            codec_groups.clear();
            core_metrics().ssd_gpu_write_fallbacks.add(1, &[]);
        }
        let ssd_admission = if ssd_writes.is_empty() && codec_groups.is_empty() {
            None
        } else {
            match Arc::clone(&self.ssd_write_admission).try_acquire_owned() {
                Ok(permit) => Some(permit),
                Err(_) => {
                    // Roll back unsubmitted GPU extents, preserving the normal
                    // D2H publication and bounded io_uring writeback path.
                    ssd_writes.clear();
                    codec_groups.clear();
                    core_metrics().ssd_gpu_write_fallbacks.add(1, &[]);
                    None
                }
            }
        };
        let disk = !ssd_writes.is_empty();
        let observation = if enabled() {
            let (key, bytes) = transfer_key(
                &layers,
                self.device_id,
                self.transfer_mode,
                true,
                disk || !codec_groups.is_empty(),
            );
            Observation::new(key, Some(bytes))
        } else {
            Observation::disabled()
        };
        self.submit(
            WorkerCommand::Save(
                SaveTask {
                    layers,
                    reply,
                    ssd_writes,
                    codec_groups,
                    ssd_admission,
                    admission: Arc::clone(&admission),
                    _instance: instance,
                    storage,
                    numa: self.numa_node,
                    #[cfg(feature = "tracing")]
                    trace_ctx: ::fastrace::prelude::SpanContext::current_local_parent(),
                },
                observation,
            ),
            disk,
        )
        .inspect_err(|_| admission.complete())?;
        match receiver.await {
            Ok(result) => result,
            Err(_) => {
                admission.retain();
                // The caller owns the imported GPU context until this future returns.
                // Losing a reply says nothing about already submitted device access.
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "GPU save reply lost; retaining admitted Publish without a completion proof"
                );
                futures::future::pending().await
            }
        }
    }

    /// Reject new work; every lane releases staging and imported pointers before acknowledgement.
    pub(crate) async fn drain(&self) -> Result<(), EngineError> {
        self.drained
            .get_or_init(|| async {
                let mut acknowledgements = Vec::new();
                {
                    let mut closed = self.closed.lock();
                    *closed = true;
                    let ssd = self.ssd_tx.lock();
                    let ssd_host = self.ssd_host_tx.lock();
                    let codec_write = self.codec_write_tx.lock();
                    for sender in [
                        Some(&self.load_tx),
                        Some(&self.save_tx),
                        ssd.as_ref(),
                        ssd_host.as_ref(),
                        codec_write.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let (reply, receiver) = oneshot::channel();
                        sender
                            .send(WorkerCommand::Drain(reply))
                            .map_err(|_| "GPU worker unavailable while draining".to_owned())?;
                        acknowledgements.push(receiver);
                    }
                }
                for receiver in acknowledgements {
                    receiver
                        .await
                        .map_err(|_| "GPU worker exited before draining".to_owned())??;
                }
                self.cufile_worker_owner.lock().take();
                Ok(())
            })
            .await
            .clone()
            .map_err(EngineError::Storage)
    }
}

fn spawn_worker(
    device_id: i32,
    numa: NumaNode,
    mode: TransferMode,
    name: &str,
) -> Result<mpsc::UnboundedSender<WorkerCommand>, EngineError> {
    let (sender, receiver) = mpsc::unbounded_channel();
    let (ready_tx, ready_rx) = std_mpsc::channel();
    let storage = name == "ssd";
    std::thread::Builder::new()
        .name(format!("gpu{device_id}-{name}"))
        .spawn(move || {
            if numa.is_valid()
                && let Err(error) = pin_thread_to_numa_node(numa)
            {
                warn!("Failed to pin GPU worker: {error}");
            }
            match init_worker(device_id, mode) {
                Ok(runtime) => {
                    let _ = ready_tx.send(Ok(()));
                    if storage {
                        ssd::run(receiver, runtime);
                    } else {
                        worker_loop(device_id, receiver, runtime);
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            }
        })
        .map_err(|e| EngineError::CudaInit(e.to_string()))?;
    ready_rx
        .recv()
        .map_err(|e| EngineError::CudaInit(e.to_string()))??;
    Ok(sender)
}

struct WorkerRuntime {
    stream: Arc<CudaStream>,
    backend: Box<dyn TransferBackend>,
    max_dma_pitch: usize,
    codec: std::cell::RefCell<Option<crate::codec::gpu::GpuCodec>>,
    codec_write: std::cell::RefCell<Option<crate::storage::ssd::cufile::GpuSlot>>,
}

fn init_worker(device_id: i32, transfer_mode: TransferMode) -> Result<WorkerRuntime, EngineError> {
    // Initialize CUDA context for this thread
    let ctx = CudaContext::new(device_id as usize)
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA context: {e:?}")))?;
    let stream = ctx
        .new_stream()
        .map_err(|e| EngineError::CudaInit(format!("Failed to create CUDA stream: {e:?}")))?;

    // Set thread-local diagnostic info
    ThreadLocalDiagnostic::insert("device_id", device_id.to_string());

    let direct = MemcpyBackend::new(&ctx).map_err(EngineError::CudaInit)?;
    let max_dma_pitch = direct.max_pitch;
    let backend: Box<dyn TransferBackend> = match transfer_mode {
        TransferMode::Direct => Box::new(direct),
        TransferMode::Kernel => Box::new(KernelBackend::new(&ctx).map_err(EngineError::CudaInit)?),
    };

    info!(
        "GPU worker initialized: device={} backend={}",
        device_id,
        backend.name()
    );

    Ok(WorkerRuntime {
        stream,
        backend,
        max_dma_pitch,
        codec: Default::default(),
        codec_write: Default::default(),
    })
}

fn worker_loop(
    device_id: i32,
    mut receiver: mpsc::UnboundedReceiver<WorkerCommand>,
    runtime: WorkerRuntime,
) {
    while let Some(command) = receiver.blocking_recv() {
        match command {
            WorkerCommand::Drain(reply) => {
                let result = runtime
                    .stream
                    .synchronize()
                    .map_err(|e| format!("GPU drain failed: {e}"));
                drop(runtime);
                let _ = reply.send(result);
                return;
            }
            WorkerCommand::Load(mut task, mut observation) => {
                let started = Instant::now();
                observation.admitted();
                task.decode_ready_observation.admitted();
                task.decode_ready_observation.submitted();
                let host_staged = task.plan.ssd_path() == Some(crate::SsdReadPath::Uring);
                let mut cancelled = false;
                let mut gpu_observation = Observation::disabled();
                let result = (|| {
                    if task.completion.is_closed() {
                        cancelled = true;
                        return Err(EngineError::Storage("GPU transfer consumer closed".into()));
                    }
                    let layers = &mut task.layers;
                    if host_staged {
                        observation.submitted();
                        restore::materialize_host(layers)?;
                        if task.completion.is_closed() {
                            cancelled = true;
                            return Err(EngineError::Storage(
                                "GPU transfer consumer closed".into(),
                            ));
                        }
                        if enabled() {
                            let mode = if runtime.backend.name() == "kernel" {
                                TransferMode::Kernel
                            } else {
                                TransferMode::Direct
                            };
                            let (key, bytes) = transfer_key(layers, device_id, mode, false, false);
                            gpu_observation = Observation::new(key, Some(bytes));
                            gpu_observation.admitted();
                        }
                    }
                    let gpu_cost = if host_staged {
                        &mut gpu_observation
                    } else {
                        &mut observation
                    };
                    let decoded_bytes =
                        codec::restore(&runtime, layers, task.codec_budget, gpu_cost).inspect_err(
                            |_| {
                                core_metrics().storage_codec_decode_failures.add(1, &[]);
                            },
                        )?;
                    let (copies, bytes) = build_copy_descs(layers)?;
                    if decoded_bytes == 0 {
                        observe_raw_copies(
                            &copies,
                            device_id as u64,
                            runtime.backend.name(),
                            false,
                            runtime.max_dma_pitch,
                            gpu_cost,
                        );
                    }
                    gpu_cost.submitted();
                    copy_and_drain(
                        runtime.backend.as_ref(),
                        &runtime.stream,
                        &copies,
                        true,
                        runtime.max_dma_pitch,
                    )?;
                    Ok(bytes + decoded_bytes)
                })();
                let bytes = result.as_ref().copied().unwrap_or(0);
                let outcome = if cancelled {
                    Outcome::Cancelled
                } else {
                    terminal_outcome(result.is_ok(), task.completion.is_closed())
                };
                let actual_io = (enabled()
                    && result.is_ok()
                    && runtime.backend.name() == "direct"
                    && !has_encoded(&task.layers))
                .then_some(bytes as u64);
                if host_staged {
                    gpu_observation.finish(outcome, actual_io);
                    observation.finish(outcome, None);
                } else {
                    observation.finish(outcome, actual_io);
                }
                finish_load(task, result.map(|_| ()), started, bytes, outcome);
            }
            WorkerCommand::Save(mut task, mut observation) => {
                // Borrow every task-owned allocation/admission across execution;
                // an unwind must not drop the task before device completion.
                let result = crate::transfer::contain_gpu_unwind(|| {
                    observation.admitted();
                    let encoded = task
                        .storage
                        .as_ref()
                        .is_some_and(|storage| storage.codec != crate::StorageCodec::None);
                    let result = codec::save(
                        &runtime,
                        &mut task.layers,
                        task.storage.as_deref(),
                        task.numa,
                        &task.codec_groups,
                        &mut observation,
                    )
                    .and_then(|()| {
                        if encoded {
                            return Ok(());
                        }
                        process_save_task(
                            &task.layers,
                            &runtime,
                            &mut observation,
                            #[cfg(feature = "tracing")]
                            task.trace_ctx,
                        )
                    });
                    let outcome = terminal_outcome(result.is_ok(), task.reply.is_closed());
                    let actual_io = if enabled() && runtime.backend.name() == "direct" && !encoded {
                        result.as_ref().ok().map(|_| transfer_shape(&task.layers).0)
                    } else {
                        None
                    };
                    observation.finish(outcome, actual_io);
                    result
                });
                drop(task.ssd_admission);
                task.admission.complete();
                let _ = task.reply.send(result.map(|()| task.layers));
            }
        }
    }
    info!("GPU worker shutting down: device={device_id}");
}

fn terminal_outcome(completed: bool, consumer_closed: bool) -> Outcome {
    if !completed {
        Outcome::Failed
    } else if consumer_closed {
        Outcome::Cancelled
    } else {
        Outcome::Completed
    }
}

fn has_encoded(layers: &[LayerTransferData]) -> bool {
    layers.iter().any(|layer| {
        layer.blocks.iter().any(|block| match &block.block {
            TransferPayload::Pending => true,
            TransferPayload::Owned(raw) => raw.encoding.is_some(),
            TransferPayload::Cached {
                sealed, slot_id, ..
            } => sealed
                .get_slot(*slot_id)
                .is_some_and(|raw| raw.encoding.is_some()),
            TransferPayload::Ssd {
                source, slot_id, ..
            } => source
                .entry
                .slots
                .get(*slot_id)
                .is_some_and(|slot| slot.encoding.is_some()),
        })
    })
}

fn transfer_shape(layers: &[LayerTransferData]) -> (u64, usize) {
    let mut bytes = 0u64;
    let mut fragments = 0usize;
    for layer in layers {
        for block in &layer.blocks {
            if let Ok(copies) = layer.layout.block_copies(block.block_idx) {
                match copies {
                    BlockCopies::Contiguous(copy) => {
                        bytes = bytes.saturating_add(copy.bytes as u64);
                        fragments = fragments.saturating_add(1);
                    }
                    BlockCopies::Split { k, v } => {
                        bytes = bytes
                            .saturating_add(k.bytes as u64)
                            .saturating_add(v.bytes as u64);
                        fragments = fragments.saturating_add(2);
                    }
                }
            }
        }
    }
    (bytes, fragments)
}

fn transfer_key(
    layers: &[LayerTransferData],
    device: i32,
    mode: TransferMode,
    write: bool,
    disk: bool,
) -> (CostEstimateKey, u64) {
    let (bytes, fragments) = transfer_shape(layers);
    let encoded = has_encoded(layers);
    let path = match (disk, encoded, write, mode) {
        (true, _, false, _) => CostObservationKind::GpuSsdLoad,
        (true, _, true, _) => CostObservationKind::GpuSsdSave,
        (false, true, false, _) => CostObservationKind::GpuDecode,
        (false, true, true, _) => CostObservationKind::GpuEncode,
        (false, false, false, TransferMode::Direct) => CostObservationKind::GpuLoadDirect,
        (false, false, false, TransferMode::Kernel) => CostObservationKind::GpuLoadKernel,
        (false, false, true, TransferMode::Direct) => CostObservationKind::GpuSaveDirect,
        (false, false, true, TransferMode::Kernel) => CostObservationKind::GpuSaveKernel,
    };
    let mut representation = None;
    let mut add_format = |format| {
        let next = Representation::from(format);
        representation = Some(match representation {
            Some(previous) if previous != next => Representation::Mixed,
            _ => next,
        });
    };
    for layer in layers {
        for block in &layer.blocks {
            let metadata = match &block.block {
                TransferPayload::Pending => {
                    add_format(layer.layout.storage_format);
                    continue;
                }
                TransferPayload::Owned(raw) => raw.encoding.as_ref(),
                TransferPayload::Cached {
                    sealed, slot_id, ..
                } => sealed
                    .get_slot(*slot_id)
                    .and_then(|raw| raw.encoding.as_ref()),
                TransferPayload::Ssd {
                    source, slot_id, ..
                } => source
                    .entry
                    .slots
                    .get(*slot_id)
                    .and_then(|slot| slot.encoding.as_ref()),
            };
            match metadata {
                Some(metadata) => metadata.iter().for_each(|meta| add_format(meta.format)),
                None => add_format(orbitkv_state::StorageFormat::Exact),
            }
        }
    }
    (
        CostEstimateKey::new(
            path,
            ExecutionResource::Gpu(device as u64),
            representation.unwrap_or(Representation::Raw),
            bytes,
            fragments,
        ),
        bytes,
    )
}

fn raw_copy_keys(
    copies: &[CopyDesc],
    device: u64,
    write: bool,
    max_pitch: usize,
) -> ([CostEstimateKey; 2], u64) {
    let bytes = copies
        .iter()
        .fold(0u64, |bytes, copy| bytes.saturating_add(copy.size as u64));
    let dma_ranges = crate::transfer::memcpy::dma_copies(copies, max_pitch).count();
    let paths = if write {
        [
            CostObservationKind::GpuSaveDirect,
            CostObservationKind::GpuSaveKernel,
        ]
    } else {
        [
            CostObservationKind::GpuLoadDirect,
            CostObservationKind::GpuLoadKernel,
        ]
    };
    let keys = paths.map(|path| {
        CostEstimateKey::new(
            path,
            ExecutionResource::Gpu(device),
            Representation::Raw,
            bytes,
            copies.len(),
        )
        .with_dma_ranges(dma_ranges)
    });
    (keys, bytes)
}

fn observe_raw_copies(
    copies: &[CopyDesc],
    device: u64,
    backend: &str,
    write: bool,
    max_pitch: usize,
    observation: &mut Observation,
) {
    if !enabled() || copies.is_empty() {
        return;
    }
    let (keys, bytes) = raw_copy_keys(copies, device, write, max_pitch);
    let selected = usize::from(backend == "kernel");
    if !observation.refine_raw_copy(keys[selected], bytes) {
        return;
    }
    // Mapped pinned ranges are the existing kernel backend's required evidence.
    let candidates = if copies.iter().all(|copy| copy.host_device != 0) {
        &keys[..]
    } else {
        &keys[..1]
    };
    shadow(candidates, selected);
}

/// Bind validated device ranges to checked host segments without changing direction.
/// Appends only when every segment fits and returns the actual transfer byte count.
pub(crate) fn append_copy_descs(
    copies: &mut Vec<CopyDesc>,
    device_allocation: usize,
    block_copies: BlockCopies,
    raw: &RawBlock,
    host_offset: usize,
) -> Result<usize, EngineError> {
    if raw.encoding.is_some() {
        return Err(EngineError::InvalidArgument(
            "encoded source cannot be submitted as raw copies".into(),
        ));
    }
    let descriptor = |segment: usize, offset: usize, device, size: usize| {
        let invalid = || EngineError::Storage("raw copy exceeds its host segment".into());
        let end = offset.checked_add(size).ok_or_else(invalid)?;
        let segment_size = raw.segment_size(segment).ok_or_else(invalid)?;
        if end > segment_size {
            return Err(invalid());
        }
        let ptr = raw
            .segment_mapped_ptr(segment)
            .ok_or_else(invalid)?
            .add(offset);
        Ok(CopyDesc {
            device,
            host: ptr.host().as_ptr(),
            host_device: ptr.device().as_ptr() as u64,
            size,
            device_allocation,
            host_registration: raw.segment_registration_id(segment).ok_or_else(invalid)?,
        })
    };
    match block_copies {
        BlockCopies::Contiguous(copy) => {
            copies.push(descriptor(0, host_offset, copy.addr, copy.bytes)?);
            Ok(copy.bytes)
        }
        BlockCopies::Split { k, v } => {
            let k_copy = descriptor(0, host_offset, k.addr, k.bytes)?;
            let (v_segment, v_offset) = if raw.num_segments() > 1 {
                (1, host_offset)
            } else {
                (
                    0,
                    host_offset.checked_add(k.bytes).ok_or_else(|| {
                        EngineError::Storage("raw copy host offset overflow".into())
                    })?,
                )
            };
            let v_copy = descriptor(v_segment, v_offset, v.addr, v.bytes)?;
            let bytes = k
                .bytes
                .checked_add(v.bytes)
                .ok_or_else(|| EngineError::Storage("raw copy byte count overflow".into()))?;
            copies.extend([k_copy, v_copy]);
            Ok(bytes)
        }
    }
}

/// Encoded/SSD restore and Publish retain layer payloads until their physical route
/// is ready; compile their remaining raw ranges with the same source-bound checks.
fn build_copy_descs(layers: &[LayerTransferData]) -> Result<(Vec<CopyDesc>, usize), EngineError> {
    let capacity = layers
        .iter()
        .map(|layer| {
            layer.blocks.len()
                * if layer.layout.geometry().is_split() {
                    2
                } else {
                    1
                }
        })
        .sum();
    let mut copies = Vec::with_capacity(capacity);
    let mut total_bytes = 0usize;
    for (layer_index, layer) in layers.iter().enumerate() {
        for block in &layer.blocks {
            if matches!(block.block, TransferPayload::Ssd { .. }) {
                continue;
            }
            let raw = block.block.raw();
            if raw.encoding.is_some() {
                continue;
            }
            let ranges = layer
                .layout
                .block_copies(block.block_idx)
                .map_err(|error| {
                    EngineError::Storage(format!("layer {}: {error}", layer.layer_name))
                })?;
            let bytes = append_copy_descs(
                &mut copies,
                layer_index,
                ranges,
                raw,
                block.block.host_offset(),
            )?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| EngineError::Storage("raw transfer byte count overflow".into()))?;
        }
    }
    // Copy ownership IDs prevent coalescing across distinct allocations.
    copies.sort_unstable_by_key(|copy| copy.device);
    Ok((copies, total_bytes))
}

/// Publish completion only after the worker establishes that all GPU access has ended.
fn finish_load(
    mut task: LoadTask,
    result: Result<(), EngineError>,
    started: Instant,
    bytes: usize,
    outcome: Outcome,
) {
    if result.is_ok() {
        for layer in &task.layers {
            for block in &layer.blocks {
                if let TransferPayload::Cached { sealed, .. } = &block.block {
                    sealed.mark_warmup_restored();
                }
            }
        }
        if bytes != 0 {
            core_metrics().load_bytes.add(bytes as u64, &[]);
            core_metrics()
                .load_duration_seconds
                .record(started.elapsed().as_secs_f64(), &[]);
        }
    } else if let Err(error) = &result {
        error!("GPU restore failed: {error}");
        core_metrics().load_failures.add(1, &[]);
    }
    let wire_bytes = result.is_ok().then_some(task.plan.source_bytes());
    task.decode_ready_observation.finish(outcome, wire_bytes);
    drop(task.layers);
    drop(task.reservations);
    drop(task.decode_admission.take());
    let _ = task.completion.send(LoadOutcome {
        result,
        completed_at: Instant::now(),
    });
}

/// Process a save task: copy blocks from GPU to CPU pinned memory. All layers
/// and segments are collected into one descriptor batch, handed to a single
/// backend, then synchronized once.
fn process_save_task(
    layers: &[LayerTransferData],
    runtime: &WorkerRuntime,
    observation: &mut Observation,
    #[cfg(feature = "tracing")] trace_ctx: Option<::fastrace::prelude::SpanContext>,
) -> Result<(), EngineError> {
    trace_child!("gpu.save_task", trace_ctx);
    let start = std::time::Instant::now();
    let stream = &runtime.stream;
    let backend = runtime.backend.as_ref();
    let total_blocks: usize = layers.iter().map(|l| l.blocks.len()).sum();

    let (copies, total_bytes) = build_copy_descs(layers)?;

    observe_raw_copies(
        &copies,
        stream.context().ordinal() as u64,
        backend.name(),
        true,
        runtime.max_dma_pitch,
        observation,
    );
    observation.submitted();
    copy_and_drain(backend, stream, &copies, false, runtime.max_dma_pitch)?;

    let elapsed = start.elapsed();
    let bandwidth_gbps = if elapsed.as_secs_f64() > 0.0 {
        (total_bytes as f64 / 1e9) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    debug!(
        "Save task completed: layers={} blocks={} copies={} bytes={} elapsed_ms={:.2} bandwidth_gbps={:.2} backend={}",
        layers.len(),
        total_blocks,
        copies.len(),
        total_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bandwidth_gbps,
        backend.name()
    );

    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/mod.rs"]
mod drain_tests;
