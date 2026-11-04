//! File byte submission. Extent/index authority stays with the supplied lease.

use std::fs::File;
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use super::SSD_ALIGNMENT;
use super::uring::{UringConfig, UringIoEngine};
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
    resource_id,
};
use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileSegment, FileTransport, Permission, Registration, TransferBackend,
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

pub(crate) enum FileMemory {
    Pinned(Vec<Arc<PinnedAllocation>>),
    Cuda {
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
    },
}

struct FileOwners {
    _file: Arc<EngineFile>,
    memory: FileMemory,
    _extent: Arc<dyn Send + Sync>,
    _permit: OwnedSemaphorePermit,
    registration: Option<Registration>,
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
            let owners = DrainOwner::new(FileOwners {
                _file: Arc::clone(&file),
                memory,
                _extent: extent,
                _permit: permit,
                registration: None,
            });
            let result = file.execute(operation, transport, &slices, owners, deadline);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    fn execute(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        slices: &[TransferSlice],
        mut owners: DrainOwner<FileOwners>,
        deadline: Instant,
    ) -> Result<usize, TransferError> {
        let registration = match &owners.value.memory {
            FileMemory::Pinned(allocations) => {
                Arc::clone(&self.backend).register_pinned(allocations, Permission::LocalOnly)
            }
            FileMemory::Cuda {
                address,
                length,
                device,
                owner,
            } => {
                // SAFETY: the slot retains this allocation/context, and observes its
                // gather event before admission. Scatter waits for this receipt.
                unsafe {
                    Arc::clone(&self.backend).register_cuda(
                        *address,
                        *length,
                        *device,
                        Arc::clone(owner),
                        Permission::LocalOnly,
                    )
                }
            }
        };
        let registration = match registration {
            Ok(registration) => registration,
            Err(error) => {
                if error.kind != ErrorKind::Unquiesced {
                    owners.drain();
                } else {
                    crate::metrics::core_metrics()
                        .ssd_engine_retained_operations
                        .add(1, &[]);
                }
                return Err(error);
            }
        };
        owners.value.registration = Some(registration);
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
            self.backend.transfer_file(
                operation,
                &self.segment,
                transport,
                slices,
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
        if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced)
            && let Some(registration) = owners.value.registration.as_mut()
        {
            // SAFETY: all results except Unquiesced prove local drain.
            if let Err(error) = unsafe { self.backend.unregister(registration) } {
                result = Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    format!("file registration cleanup failed: {error}"),
                ));
            }
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

enum Backend {
    Legacy {
        io: UringIoEngine,
        _files: Vec<Arc<File>>,
    },
    Transfer(Vec<Arc<EngineFile>>),
}

pub(super) struct SsdIo {
    backend: Backend,
    admission: Arc<Semaphore>,
    pub(super) cost_resource: ExecutionResource,
}

impl SsdIo {
    pub(super) fn new(
        files: &[Arc<File>],
        transfer: Option<Arc<dyn TransferBackend>>,
        legacy_config: UringConfig,
    ) -> std::io::Result<Self> {
        let admission = Arc::new(Semaphore::new(MAX_OPERATIONS));
        let backend = if let Some(transfer) = transfer {
            let mut handles = Vec::with_capacity(files.len());
            for file in files {
                let segment = transfer
                    .open_file(&PathBuf::from(format!(
                        "/proc/self/fd/{}",
                        file.as_raw_fd()
                    )))
                    .map_err(std::io::Error::other)?;
                if segment.length != file.metadata()?.len() {
                    let _ = transfer.close_file(&segment, Instant::now() + DEADLINE);
                    return Err(std::io::Error::other(
                        "engine file length differs from reserved SSD shard",
                    ));
                }
                handles.push(Arc::new(EngineFile {
                    backend: Arc::clone(&transfer),
                    segment,
                    file: Arc::clone(file),
                    admission: Arc::clone(&admission),
                    runtime: tokio::runtime::Handle::current(),
                    cost_resource: ExecutionResource::SsdFile(resource_id(&(
                        transfer.source_epoch(),
                        file.as_raw_fd(),
                    ))),
                }));
            }
            Backend::Transfer(handles)
        } else {
            Backend::Legacy {
                io: UringIoEngine::new_multi(
                    files.iter().map(|f| f.as_raw_fd()).collect(),
                    legacy_config,
                )?,
                _files: files.to_vec(),
            }
        };
        let cost_resource = match &backend {
            Backend::Legacy { io, .. } => io.cost_resource,
            Backend::Transfer(handles) => ExecutionResource::SsdStore(resource_id(
                &handles.iter().map(|f| f.cost_resource).collect::<Vec<_>>(),
            )),
        };
        Ok(Self {
            backend,
            admission,
            cost_resource,
        })
    }

    pub(super) fn engine_file(&self, shard: usize) -> Option<Arc<EngineFile>> {
        match &self.backend {
            Backend::Transfer(files) => files.get(shard).cloned(),
            _ => None,
        }
    }

    pub(super) fn submit(
        self: &Arc<Self>,
        operation: TransferOp,
        shard: usize,
        slices: Vec<TransferSlice>,
        allocations: Vec<Arc<PinnedAllocation>>,
        extent: Arc<dyn Send + Sync>,
    ) -> Result<Receipt, TransferError> {
        validate(&slices)?;
        if slices.iter().any(|s| {
            !allocations
                .iter()
                .any(|a| a.source_range(s.local, s.length).is_ok())
        }) {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "SSD descriptor is outside retained backing",
            ));
        }
        match &self.backend {
            Backend::Transfer(files) => files
                .get(shard)
                .ok_or_else(|| TransferError::new(ErrorKind::InvalidRequest, "invalid SSD shard"))?
                .submit(
                    operation,
                    FileTransport::IoUring,
                    slices,
                    FileMemory::Pinned(allocations),
                    extent,
                ),
            Backend::Legacy { io, .. } => {
                let permit = admit(&self.admission)?;
                let offset = slices
                    .first()
                    .ok_or_else(|| {
                        TransferError::new(ErrorKind::InvalidRequest, "empty SSD batch")
                    })?
                    .remote_address;
                let mut expected = offset;
                for slice in &slices {
                    if slice.remote_address != expected {
                        return Err(TransferError::new(
                            ErrorKind::InvalidRequest,
                            "legacy SSD iovecs must be consecutive",
                        ));
                    }
                    expected += slice.length as u64;
                }
                let owners = DrainOwner::new((Arc::clone(self), allocations, extent, permit));
                let receiver = match operation {
                    TransferOp::Read => io.readv_at_async(
                        shard,
                        slices
                            .iter()
                            .map(|s| (s.local.as_ptr(), s.length))
                            .collect(),
                        offset,
                    ),
                    TransferOp::Write => io.writev_at_async(
                        shard,
                        slices
                            .iter()
                            .map(|s| (s.local.as_ptr().cast_const(), s.length))
                            .collect(),
                        offset,
                    ),
                };
                let receiver = match receiver {
                    Ok(receiver) => receiver,
                    Err(error) => {
                        owners.drain();
                        return Err(TransferError::new(ErrorKind::Backend, error.to_string()));
                    }
                };
                let (tx, rx) = oneshot::channel();
                tokio::spawn(async move {
                    let result = match receiver.await {
                        Ok(result) => {
                            owners.drain();
                            result
                                .map_err(|e| TransferError::new(ErrorKind::Backend, e.to_string()))
                        }
                        Err(error) => {
                            crate::metrics::core_metrics()
                                .ssd_engine_retained_operations
                                .add(1, &[]);
                            Err(TransferError::new(ErrorKind::Unquiesced, error.to_string()))
                        }
                    };
                    let _ = tx.send(result);
                });
                Ok(rx)
            }
        }
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
mod tests;
