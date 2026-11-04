//! File byte submission. Extent/index authority stays with the supplied lease.

use std::fs::File;
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use super::SSD_ALIGNMENT;
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
    resource_id,
};
use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileMemory, FileSegment, FileTransfer, FileTransport, TransferBackend,
    TransferError, TransferOp, TransferSlice, validate_slices,
};

const MAX_OPERATIONS: usize = 32;
const MAX_BYTES: usize = 64 << 20;
const MAX_SLICES: usize = 16_384;
const DEADLINE: Duration = Duration::from_secs(30);

pub(super) type Receipt = oneshot::Receiver<Result<usize, TransferError>>;

/// Defaults to retention, including unwind or a dropped completion receiver.
/// Only a physical receipt may turn this into ordinary destruction.
struct DrainOwner<T> {
    value: ManuallyDrop<T>,
    drained: bool,
}

impl<T> DrainOwner<T> {
    fn new(value: T) -> Self {
        Self {
            value: ManuallyDrop::new(value),
            drained: false,
        }
    }
    fn drain(mut self) {
        self.drained = true;
    }
}

impl<T> Drop for DrainOwner<T> {
    fn drop(&mut self) {
        if self.drained {
            // SAFETY: this is the only destruction site, enabled by a drain receipt.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        }
    }
}

pub(crate) struct EngineFile {
    backend: Arc<dyn TransferBackend>,
    segment: FileSegment,
    // /proc/self/fd binds the engine open to this exact inode, not a replaceable path.
    file: Arc<File>,
    admission: Arc<Semaphore>,
    runtime: tokio::runtime::Handle,
    pub(crate) cost_resource: ExecutionResource,
}

impl EngineFile {
    pub(crate) fn submit(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        slices: Vec<TransferSlice>,
        memory: FileMemory,
        extent: Arc<dyn Send + Sync>,
    ) -> Result<Receipt, TransferError> {
        validate(&slices)?;
        let owns = |slice: &TransferSlice| match &memory {
            FileMemory::Pinned(allocations) => {
                allocations.len() <= MAX_SLICES
                    && allocations
                        .iter()
                        .any(|a| a.source_range(slice.local, slice.length).is_ok())
            }
            FileMemory::Cuda {
                address, length, ..
            } => (slice.local.as_ptr() as u64)
                .checked_sub(*address)
                .and_then(|offset| offset.checked_add(slice.length as u64))
                .is_some_and(|end| end <= *length as u64),
        };
        if !slices.iter().all(owns) {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "file descriptor is outside retained backing",
            ));
        }
        let permit = admit(&self.admission)?;
        let file = Arc::clone(self);
        let deadline = Instant::now() + DEADLINE;
        let (tx, rx) = oneshot::channel();
        self.runtime.spawn_blocking(move || {
            let owners = DrainOwner::new(Arc::new(FileTransfer {
                slices,
                memory,
                _lease: Arc::new((Arc::clone(&file), extent, permit)),
            }));
            let result = file.execute(operation, transport, owners, deadline);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    fn execute(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        owners: DrainOwner<Arc<FileTransfer>>,
        deadline: Instant,
    ) -> Result<usize, TransferError> {
        let slices = &owners.value.slices;
        let mut observation = Observation::new(
            CostEstimateKey::new(
                if operation == TransferOp::Read {
                    CostObservationKind::SsdRead
                } else {
                    CostObservationKind::SsdWrite
                },
                self.cost_resource,
                Representation::Unknown,
                slices.iter().map(|s| s.length as u64).sum(),
                slices.len(),
            ),
            None,
        );
        observation.admitted();
        observation.submitted();
        // SAFETY: registration, actual backing, exact file and extent lease are
        // held by owners. An uncertain receipt retains that entire owner and permit.
        let mut result = unsafe {
            Arc::clone(&self.backend).transfer_file(
                operation,
                &self.segment,
                transport,
                Arc::clone(&owners.value),
                deadline,
                &Cancellation::default(),
            )
        };
        if result
            .as_ref()
            .is_ok_and(|bytes| *bytes != slices.iter().map(|s| s.length).sum::<usize>())
        {
            result = Err(TransferError::new(
                ErrorKind::Backend,
                "short file transfer",
            ));
        }
        if operation == TransferOp::Read && result.is_ok() {
            crate::metrics::core_metrics()
                .ssd_engine_read_batches
                .add(1, &[]);
        }
        observation.finish(
            if result.is_ok() {
                Outcome::Completed
            } else {
                Outcome::Failed
            },
            result.as_ref().ok().map(|n| *n as u64),
        );
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            crate::metrics::core_metrics()
                .ssd_engine_retained_operations
                .add(1, &[]);
            log::error!(
                "Retaining SSD file, registration, backing, extent and admission after unknown drain"
            );
        } else {
            owners.drain();
        }
        result
    }
}

impl Drop for EngineFile {
    #[allow(
        clippy::mem_forget,
        reason = "failed file close retains its engine and exact inode"
    )]
    fn drop(&mut self) {
        if let Err(error) = self
            .backend
            .close_file(&self.segment, Instant::now() + DEADLINE)
        {
            log::error!("Retaining SSD file after close failed: {error}");
            std::mem::forget((
                Arc::clone(&self.backend),
                Arc::clone(&self.file),
                Arc::clone(&self.admission),
            ));
        }
    }
}

pub(super) struct SsdIo {
    files: Vec<Arc<EngineFile>>,
    pub(super) cost_resource: ExecutionResource,
}

impl SsdIo {
    pub(super) fn new(
        files: &[Arc<File>],
        transfer: Arc<dyn TransferBackend>,
    ) -> std::io::Result<Self> {
        let admission = Arc::new(Semaphore::new(MAX_OPERATIONS));
        let mut handles = Vec::with_capacity(files.len());
        for file in files {
            let segment = transfer
                .open_file(&PathBuf::from(format!(
                    "/proc/self/fd/{}",
                    file.as_raw_fd()
                )))
                .map_err(std::io::Error::other)?;
            let handle = Arc::new(EngineFile {
                backend: Arc::clone(&transfer),
                segment,
                file: Arc::clone(file),
                admission: Arc::clone(&admission),
                runtime: tokio::runtime::Handle::current(),
                cost_resource: ExecutionResource::SsdFile(resource_id(&(
                    transfer.source_epoch(),
                    file.as_raw_fd(),
                ))),
            });
            if handle.segment.length != file.metadata()?.len() {
                return Err(std::io::Error::other(
                    "engine file length differs from reserved SSD shard",
                ));
            }
            handles.push(handle);
        }
        let cost_resource = ExecutionResource::SsdStore(resource_id(
            &handles.iter().map(|f| f.cost_resource).collect::<Vec<_>>(),
        ));
        Ok(Self {
            files: handles,
            cost_resource,
        })
    }

    pub(super) fn engine_file(&self, shard: usize) -> Option<Arc<EngineFile>> {
        self.files.get(shard).cloned()
    }

    pub(super) fn submit(
        &self,
        operation: TransferOp,
        shard: usize,
        slices: Vec<TransferSlice>,
        allocations: Vec<Arc<PinnedAllocation>>,
        extent: Arc<dyn Send + Sync>,
    ) -> Result<Receipt, TransferError> {
        self.files
            .get(shard)
            .ok_or_else(|| TransferError::new(ErrorKind::InvalidRequest, "invalid SSD shard"))?
            .submit(
                operation,
                FileTransport::IoUring,
                slices,
                FileMemory::Pinned(allocations),
                extent,
            )
    }
}

fn admit(admission: &Arc<Semaphore>) -> Result<OwnedSemaphorePermit, TransferError> {
    Arc::clone(admission).try_acquire_owned().map_err(|_| {
        TransferError::new(
            ErrorKind::ResourceExhausted,
            "SSD operation/retention budget exhausted",
        )
    })
}

fn validate(slices: &[TransferSlice]) -> Result<(), TransferError> {
    if slices.is_empty() || slices.len() > MAX_SLICES || validate_slices(slices)? > MAX_BYTES {
        return Err(TransferError::new(
            ErrorKind::ResourceExhausted,
            "SSD batch exceeds descriptor or byte budget",
        ));
    }
    for s in slices {
        if !(s.local.as_ptr() as usize).is_multiple_of(SSD_ALIGNMENT)
            || !s.length.is_multiple_of(SSD_ALIGNMENT)
            || !s.remote_address.is_multiple_of(SSD_ALIGNMENT as u64)
        {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "SSD I/O must preserve 512-byte O_DIRECT alignment",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/io.rs"]
pub(super) mod tests;
