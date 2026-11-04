//! TENT's file ABI has ranges but no padding or unaligned-vector staging.
//! Keep the repair here until the fork can consume FileIoElement directly.
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    mem::ManuallyDrop,
    ptr::NonNull,
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, result, sys};
use orbitkv_transfer::{TransferSlice, files::FileSegment};

use super::tent_error;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileIoElement, FileMemory, FileTransfer, FileTransport, TransferError,
    TransferOp,
};

const SCRATCH_BYTES: usize = 4 << 20;
struct Scratch {
    address: u64,
    bytes: usize,
    cuda: Option<(Arc<CudaStream>, CudaEvent)>,
}
impl Scratch {
    fn new(bytes: usize, context: Option<Arc<CudaContext>>) -> Result<Arc<Self>, TransferError> {
        let cuda = context
            .as_ref()
            .map(|context| {
                Ok::<_, TransferError>((
                    context.new_stream().map_err(cuda_error)?,
                    context.new_event(None).map_err(cuda_error)?,
                ))
            })
            .transpose()?;
        let address = if context.is_some() {
            // SAFETY: caller bound the retained context before allocation.
            unsafe { result::malloc_sync(bytes) }.map_err(cuda_error)?
        } else {
            let layout =
                Layout::from_size_align(bytes, 4096).map_err(|e| invalid(e.to_string()))?;
            // SAFETY: nonzero bounded layout; this owner releases it after drain.
            let ptr = unsafe { alloc_zeroed(layout) };
            NonNull::new(ptr)
                .ok_or_else(|| {
                    TransferError::new(
                        ErrorKind::ResourceExhausted,
                        "TENT alignment allocation failed",
                    )
                })?
                .as_ptr() as u64
        };
        if context.is_some() {
            crate::metrics::core_metrics()
                .ssd_gpu_staging_bytes
                .add(bytes as i64, &[]);
        }
        Ok(Arc::new(Self {
            address,
            bytes,
            cuda,
        }))
    }
    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<(), TransferError> {
        if let Some((stream, _)) = &self.cuda {
            // SAFETY: preflight bounds both ranges; drain retains them through the event.
            unsafe { result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }
                .map_err(cuda_error)
        } else {
            // SAFETY: scratch and retained host ranges are disjoint and valid for bytes.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
            Ok(())
        }
    }
    unsafe fn zero(&self, bytes: usize) -> Result<(), TransferError> {
        if let Some((stream, _)) = &self.cuda {
            // SAFETY: exclusively owned scratch, bounded by its allocation.
            unsafe { result::memset_d8_async(self.address, 0, bytes, stream.cu_stream()) }
                .map_err(cuda_error)
        } else {
            // SAFETY: exclusively owned scratch, bounded by its allocation.
            unsafe { std::ptr::write_bytes(self.address as *mut u8, 0, bytes) };
            Ok(())
        }
    }
}
impl Scratch {
    fn drain(&self, deadline: Instant, cancellation: &Cancellation) -> Result<(), TransferError> {
        let Some((stream, event)) = &self.cuda else {
            return Ok(());
        };
        event.record(stream).map_err(cuda_error)?;
        loop {
            // SAFETY: this event and its exact stream remain retained by scratch.
            match unsafe { result::event::query(event.cu_event()) } {
                Ok(()) => return Ok(()),
                Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => {}
                Err(error) => return Err(cuda_error(error)),
            }
            if Instant::now() >= deadline || cancellation.is_cancelled() {
                return Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "TENT alignment copy has no physical drain receipt",
                ));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    // Called once after both file and copy drain. Drop intentionally cannot retry
    // a failed cleanup; the transfer retains scratch and its permit on failure.
    fn release(&self) -> Result<(), TransferError> {
        if let Some((stream, _)) = &self.cuda {
            stream.context().bind_to_thread().map_err(cuda_error)?;
            // SAFETY: the file receipt and CUDA event prove every borrower has drained.
            unsafe { result::free_sync(self.address) }.map_err(cuda_error)?;
            crate::metrics::core_metrics()
                .ssd_gpu_staging_bytes
                .add(-(self.bytes as i64), &[]);
        } else {
            let layout = Layout::from_size_align(self.bytes, 4096).map_err(cuda_error)?;
            // SAFETY: same layout used at allocation; every borrower has drained.
            unsafe {
                dealloc(self.address as *mut u8, layout);
            }
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> TransferError {
    TransferError::new(ErrorKind::InvalidRequest, message)
}
fn cuda_error(error: impl std::fmt::Display) -> TransferError {
    // Even a synchronous CUDA error can report an earlier asynchronous failure.
    TransferError::new(ErrorKind::Unquiesced, error.to_string())
}

/// # Safety
/// The transfer owns exclusive/ready ranges and their registrations through drain.
#[allow(
    clippy::mem_forget,
    reason = "uncertain CUDA copies retain scratch and actual backing"
)]
pub(super) unsafe fn transfer(
    file: &Arc<FileSegment>,
    operation: TransferOp,
    transport: FileTransport,
    owner: Arc<FileTransfer>,
    deadline: Instant,
    cancellation: &Cancellation,
) -> Result<usize, TransferError> {
    if cancellation.is_cancelled() {
        return Err(TransferError::new(
            ErrorKind::Cancelled,
            "file transfer cancelled",
        ));
    }
    crate::transfer_engine::remaining(deadline)?;
    let bytes = owner.extent.bytes()?;
    if owner.extent.offset + bytes as u64 > file.length {
        return Err(invalid("file vector exceeds admitted file length"));
    }
    let context = match &owner.memory {
        FileMemory::Pinned(_) => None,
        FileMemory::RegisteredCuda { context, .. } => Some(Arc::clone(context)),
    };
    if let Some(context) = &context {
        context
            .bind_to_thread()
            .map_err(|e| invalid(e.to_string()))?;
    }
    let gds = transport == FileTransport::Gds;
    if gds
        && owner
            .extent
            .elements
            .iter()
            .any(|e| matches!(e, FileIoElement::Padding(_)))
    {
        return Err(TransferError::new(
            ErrorKind::Unavailable,
            "TENT GDS cannot express padding",
        ));
    }
    let native = if gds {
        orbitkv_transfer::files::FileTransport::Gds(
            context
                .clone()
                .ok_or_else(|| invalid("GDS requires CUDA memory"))?,
        )
    } else {
        orbitkv_transfer::files::FileTransport::IoUring
    };
    let op = if operation == TransferOp::Read {
        orbitkv_transfer::TransferOp::Read
    } else {
        orbitkv_transfer::TransferOp::Write
    };
    let alignment = if gds { 4096 } else { 512 };
    let direct = owner.extent.offset.is_multiple_of(alignment as u64)
        && owner.extent.elements.iter().all(|e| match e {
            FileIoElement::Range { address, length } => {
                length.is_multiple_of(alignment) && (!gds || address.is_multiple_of(4096))
            }
            FileIoElement::Padding(_) => false,
        });
    if gds && !direct {
        return Err(TransferError::new(
            ErrorKind::Unavailable,
            "TENT GDS cannot express padded or unaligned file vectors",
        ));
    }
    if direct {
        let mut offset = owner.extent.offset;
        let slices = owner
            .extent
            .elements
            .iter()
            .map(|e| {
                let FileIoElement::Range { address, length } = *e else {
                    return Err(invalid("unexpected padding"));
                };
                let slice = TransferSlice {
                    local: NonNull::new(address as *mut u8).ok_or_else(|| invalid("null range"))?,
                    remote_address: offset,
                    length,
                };
                offset += length as u64;
                Ok(slice)
            })
            .collect::<Result<Vec<_>, TransferError>>()?;
        // SAFETY: callback retains all actual backing and leases until native drain.
        return unsafe { file.transfer(op, native, &slices, owner, deadline, cancellation.flag()) }
            .map_err(tent_error);
    }
    if !owner.extent.offset.is_multiple_of(512) || !bytes.is_multiple_of(512) {
        return Err(invalid("TENT repaired extent must be 512-byte aligned"));
    }
    let mut scratch = ManuallyDrop::new(Scratch::new(bytes.min(SCRATCH_BYTES), context)?);
    let mut result = (|| {
        for start in (0..bytes).step_by(SCRATCH_BYTES) {
            crate::transfer_engine::remaining(deadline)?;
            if cancellation.is_cancelled() {
                return Err(TransferError::new(
                    ErrorKind::Cancelled,
                    "file transfer cancelled",
                ));
            }
            let size = (bytes - start).min(SCRATCH_BYTES);
            let copy_ranges = || {
                let mut position = 0;
                for element in &owner.extent.elements {
                    let end = position + element.length();
                    if let FileIoElement::Range { address, .. } = element {
                        let begin = position.max(start);
                        let limit = end.min(start + size);
                        if begin < limit {
                            let local = address + (begin - position) as u64;
                            let staged = scratch.address + (begin - start) as u64;
                            let (dst, src) = if operation == TransferOp::Write {
                                (staged, local)
                            } else {
                                (local, staged)
                            };
                            // SAFETY: intersection is within validated actual range and scratch.
                            unsafe {
                                scratch.copy(dst, src, limit - begin)?;
                            }
                        }
                    }
                    position = end;
                }
                Ok::<(), TransferError>(())
            };
            if operation == TransferOp::Write {
                // SAFETY: size is bounded by scratch allocation.
                unsafe {
                    scratch.zero(size)?;
                }
                copy_ranges()?;
                scratch.drain(deadline, cancellation)?;
            }
            let slice = TransferSlice {
                local: NonNull::new(scratch.address as *mut u8)
                    .ok_or_else(|| invalid("null scratch"))?,
                remote_address: owner.extent.offset + start as u64,
                length: size,
            };
            // SAFETY: native callback owns both scratch and original extent/backing.
            unsafe {
                file.transfer(
                    op,
                    orbitkv_transfer::files::FileTransport::IoUring,
                    &[slice],
                    Arc::new((Arc::clone(&owner), Arc::clone(&scratch))),
                    deadline,
                    cancellation.flag(),
                )
            }
            .map_err(tent_error)?;
            if operation == TransferOp::Read {
                copy_ranges()?;
                scratch.drain(deadline, cancellation)?;
            }
        }
        crate::transfer_engine::remaining(deadline)?;
        if cancellation.is_cancelled() {
            return Err(TransferError::new(
                ErrorKind::Cancelled,
                "file transfer cancelled after drain",
            ));
        }
        Ok(bytes)
    })();
    if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced)
        && let Err(error) = scratch.release()
    {
        result = Err(error);
    }
    if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
        std::mem::forget((owner, scratch));
    } else {
        // SAFETY: file/copy drain and explicit allocation cleanup succeeded.
        unsafe {
            ManuallyDrop::drop(&mut scratch);
        }
    }
    result
}
