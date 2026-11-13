//! File byte submission. Extent/index authority stays with the supplied lease.

use std::fs::File;
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use super::SSD_ALIGNMENT;
use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileExtent, FileIoElement, FileMemory, FileSegment, FileTransfer,
    FileTransport, MAX_FILE_ELEMENTS, TransferEngine, TransferError, TransferOp, TransferOptions,
};

const MAX_OPERATIONS: usize = 32;
pub(super) const DEADLINE: Duration = Duration::from_secs(30);

pub(crate) type Receipt = oneshot::Receiver<Result<usize, TransferError>>;

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
    #[cfg(test)]
    receipt_control: Option<Arc<tests::engine::ReceiptControl>>,
    pub(crate) backend: Arc<TransferEngine>,
    segment: FileSegment,
    // /proc/self/fd binds the engine open to this exact inode, not a replaceable path.
    file: Arc<File>,
    admission: Arc<Semaphore>,
    runtime: tokio::runtime::Handle,
}

impl EngineFile {
    /// The caller serializes scratch access and holds the immutable extent.
    /// Even an unsuccessful checksum keeps submitted owners until native drain.
    pub(super) fn checksum(
        self: &Arc<Self>,
        offset: u64,
        length: u64,
        scratch: &Arc<PinnedAllocation>,
        lease: Arc<dyn Send + Sync>,
        options: TransferOptions,
    ) -> Result<u32, TransferError> {
        let mut read = 0u64;
        let mut checksum = crc32fast::Hasher::new();
        while read < length {
            let bytes = (length - read).min(super::persistence::VERIFY_CHUNK as u64) as usize;
            let extent = FileExtent {
                offset: offset + read,
                elements: vec![FileIoElement::Range {
                    address: scratch.as_non_null().as_ptr() as u64,
                    length: bytes,
                }],
            };
            validate(&extent)?;
            if bytes as u64 > scratch.size_bytes() {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "SSD checksum scratch is too small",
                ));
            }
            let permit = admit(&self.admission)?;
            let owner = DrainOwner::new(Arc::new(FileTransfer {
                extent,
                memory: FileMemory::Pinned(vec![Arc::clone(scratch)]),
                _lease: Arc::new((Arc::clone(self), Arc::clone(&lease), permit)),
            }));
            self.execute(TransferOp::Read, FileTransport::IoUring, owner, options)?;
            // SAFETY: this range is in the retained scratch allocation. The
            // checksum caller holds its exclusive mutex; the successful native
            // read receipt initialized all bytes and ended every device access.
            let payload =
                unsafe { std::slice::from_raw_parts(scratch.as_non_null().as_ptr(), bytes) };
            checksum.update(payload);
            read += bytes as u64;
        }
        Ok(checksum.finalize())
    }

    pub(crate) fn submit(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        extent: FileExtent,
        memory: FileMemory,
        lease: Arc<dyn Send + Sync>,
        options: TransferOptions,
    ) -> Result<Receipt, TransferError> {
        validate(&extent)?;
        let owns = |address, length| match &memory {
            FileMemory::Pinned(allocations) => {
                allocations.len() <= MAX_FILE_ELEMENTS
                    && NonNull::new(address as *mut u8).is_some_and(|pointer| {
                        allocations
                            .iter()
                            .any(|a| a.source_range(pointer, length).is_ok())
                    })
            }
            FileMemory::RegisteredCuda { owners, context } => {
                owners.len() <= MAX_FILE_ELEMENTS
                    && owners.iter().any(|owner| {
                        owner.device == context.ordinal() as i32 && owner.covers(address, length)
                    })
            }
        };
        if extent.elements.iter().any(|element| match *element {
            FileIoElement::Range { address, length } => !owns(address, length),
            FileIoElement::Padding(_) => false,
        }) {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "file descriptor is outside retained backing",
            ));
        }
        let permit = admit(&self.admission)?;
        let file = Arc::clone(self);
        let (tx, rx) = oneshot::channel();
        self.runtime.spawn_blocking(move || {
            let owners = DrainOwner::new(Arc::new(FileTransfer {
                extent,
                memory,
                _lease: Arc::new((Arc::clone(&file), lease, permit)),
            }));
            let result = file.execute(operation, transport, owners, options);
            let _ = tx.send(result);
        });
        Ok(rx)
    }

    fn execute(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        owners: DrainOwner<Arc<FileTransfer>>,
        options: TransferOptions,
    ) -> Result<usize, TransferError> {
        let bytes = owners.value.extent.bytes()?;
        let gpu = matches!(owners.value.memory, FileMemory::RegisteredCuda { .. });
        let started = Instant::now();
        let metrics = crate::metrics::core_metrics();
        if gpu {
            metrics.ssd_cufile_inflight_batches.add(1, &[]);
        }
        let registrations = match &owners.value.memory {
            FileMemory::RegisteredCuda { owners, .. } => owners.as_slice(),
            _ => &[],
        };
        let mut unique: Vec<_> = registrations.iter().collect();
        unique.sort_unstable_by_key(|owner| Arc::as_ptr(owner));
        unique.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let loans = unique
            .iter()
            .map(|owner| owner.loan(&self.backend))
            .collect::<Result<Vec<_>, _>>();
        let mut result = match loans.as_ref() {
            Err(error) => Err(TransferError::new(error.kind, error.message.clone())),
            Ok(_) => catch_unwind(AssertUnwindSafe(|| {
                #[cfg(test)]
                if let Some(control) = &self.receipt_control {
                    control.observe(options)?;
                }
                // SAFETY: actual backing, persistent loans, file and extent/admission leases outlive drain.
                unsafe {
                    Arc::clone(&self.backend).transfer_file(
                        operation,
                        &self.segment,
                        transport,
                        Arc::clone(&owners.value),
                        options,
                        &Cancellation::default(),
                    )
                }
            }))
            .unwrap_or_else(|_| {
                Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "file backend unwound without a drain receipt",
                ))
            }),
        };
        #[cfg(test)]
        if self.receipt_control.as_ref().is_some_and(|c| c.short) {
            result = result.map(|bytes| bytes / 2);
        }
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            for owner in registrations {
                owner.retain();
            }
        }
        drop(loans);
        if result
            .as_ref()
            .is_ok_and(|transferred| *transferred != bytes)
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
        if gpu {
            let (seconds, transferred, failures) = if operation == TransferOp::Read {
                (
                    &metrics.ssd_cufile_read_seconds,
                    &metrics.ssd_cufile_read_bytes,
                    &metrics.ssd_cufile_read_failures,
                )
            } else {
                (
                    &metrics.ssd_cufile_write_seconds,
                    &metrics.ssd_cufile_write_bytes,
                    &metrics.ssd_cufile_write_failures,
                )
            };
            seconds.record(started.elapsed().as_secs_f64(), &[]);
            if result.is_ok() {
                transferred.add(bytes as u64, &[]);
            } else {
                failures.add(1, &[]);
            }
            if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
                metrics.ssd_cufile_inflight_batches.add(-1, &[]);
            }
        }

        #[cfg(test)]
        if self.receipt_control.as_ref().is_some_and(|c| c.short) {
            result = result.map(|bytes| bytes / 2);
        }
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            crate::metrics::core_metrics()
                .ssd_engine_retained_operations
                .add(1, &[]);
            log::error!(
                "Retaining SSD file, registration, backing, extent and admission after unknown drain"
            );
        } else {
            if operation == TransferOp::Read {
                super::persistence::fault("ssd_read_receipt");
            }
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
    transport: FileTransport,
    files: Vec<Arc<EngineFile>>,
}

impl SsdIo {
    pub(super) fn new(
        files: &[Arc<File>],
        transfer: Arc<TransferEngine>,
        transport: FileTransport,
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
                #[cfg(test)]
                receipt_control: None,
                backend: Arc::clone(&transfer),
                segment,
                file: Arc::clone(file),
                admission: Arc::clone(&admission),
                runtime: tokio::runtime::Handle::current(),
            });
            if handle.segment.length != file.metadata()?.len() {
                return Err(std::io::Error::other(
                    "engine file length differs from reserved SSD shard",
                ));
            }
            handles.push(handle);
        }
        Ok(Self {
            transport,
            files: handles,
        })
    }

    pub(super) fn engine_file(&self, shard: usize) -> Option<Arc<EngineFile>> {
        self.files.get(shard).cloned()
    }

    pub(super) fn submit(
        &self,
        operation: TransferOp,
        shard: usize,
        extent: FileExtent,
        allocations: Vec<Arc<PinnedAllocation>>,
        lease: Arc<dyn Send + Sync>,
        options: TransferOptions,
    ) -> Result<Receipt, TransferError> {
        self.files
            .get(shard)
            .ok_or_else(|| TransferError::new(ErrorKind::InvalidRequest, "invalid SSD shard"))?
            .submit(
                operation,
                self.transport,
                extent,
                FileMemory::Pinned(allocations),
                lease,
                options,
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

fn validate(extent: &FileExtent) -> Result<(), TransferError> {
    let bytes = extent.bytes()?;
    if !extent.offset.is_multiple_of(SSD_ALIGNMENT as u64) || !bytes.is_multiple_of(SSD_ALIGNMENT) {
        return Err(TransferError::new(
            ErrorKind::InvalidRequest,
            "SSD file extent must preserve 512-byte O_DIRECT alignment",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/io.rs"]
pub(super) mod tests;
