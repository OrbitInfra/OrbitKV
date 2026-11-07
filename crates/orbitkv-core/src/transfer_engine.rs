//! Transfer-engine contracts. Logical failure never establishes physical drain.

pub(crate) mod backend;
pub(crate) use backend::create_transfer_engine;
pub use backend::{FileSegment, PeerSegment, Registration, TransferEngine};
mod cuda;
pub use cuda::CudaRegistration;
pub mod local;

use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

use crate::memory::pool::PinnedAllocation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
}

/// KV purpose supplied to the engine, independently of byte layout or transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferIntent {
    ForegroundGet,
    BackgroundPrefetch,
    Checkpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferPriority {
    High,
    Low,
}

impl TransferIntent {
    pub fn priority(self) -> TransferPriority {
        match self {
            Self::ForegroundGet => TransferPriority::High,
            Self::BackgroundPrefetch | Self::Checkpoint => TransferPriority::Low,
        }
    }
}

/// One admission deadline, preserved across queueing and all extent submissions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransferOptions {
    pub intent: TransferIntent,
    pub deadline: Instant,
}

impl TransferOptions {
    pub fn read(mode: crate::QueryMode, deadline: Instant) -> Self {
        Self {
            intent: match mode {
                crate::QueryMode::Prepare | crate::QueryMode::Warmup => {
                    TransferIntent::BackgroundPrefetch
                }
                crate::QueryMode::Demand | crate::QueryMode::WaitForFullPrefix => {
                    TransferIntent::ForegroundGet
                }
            },
            deadline,
        }
    }

    pub fn checkpoint(deadline: Instant) -> Self {
        Self {
            intent: TransferIntent::Checkpoint,
            deadline,
        }
    }
}

/// Requested native access, additionally constrained by authoritative source grants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permission {
    LocalOnly,
    RemoteRead,
    RemoteReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Timeout,
    Cancelled,
    Unquiesced,
    InvalidRequest,
    EpochMismatch,
    PermissionDenied,
    ResourceExhausted,
    Unavailable,
    Backend,
}

#[derive(Debug)]
pub struct TransferError {
    pub kind: ErrorKind,
    pub message: String,
}

impl TransferError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}
impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for TransferError {}

/// Cooperative cancellation is a request to stop work, never a release receipt.
#[derive(Default)]
pub struct Cancellation(AtomicBool);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy)]
pub struct TransferSlice {
    pub local: NonNull<u8>,
    pub remote_address: u64,
    pub length: usize,
}
// SAFETY: submission requires the actual allocation and exclusive access owners.
unsafe impl Send for TransferSlice {}
// SAFETY: the descriptor does not itself dereference the address.
unsafe impl Sync for TransferSlice {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileTransport {
    Auto,
    IoUring,
    Gds,
}

pub(crate) const MAX_FILE_BYTES: usize = 64 << 20;
pub(crate) const MAX_FILE_ELEMENTS: usize = 1024;

/// File-ordered layout. Padding has no consumer-owned byte buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileIoElement {
    Range { address: u64, length: usize },
    Padding(usize),
}

impl FileIoElement {
    pub(crate) fn length(&self) -> usize {
        match self {
            Self::Range { length, .. } | Self::Padding(length) => *length,
        }
    }
}

pub(crate) struct FileExtent {
    pub offset: u64,
    pub elements: Vec<FileIoElement>,
}

impl FileExtent {
    /// Layout planning only: local ranges cover the data; the engine owns gaps.
    pub(crate) fn from_slices(
        offset: u64,
        bytes: usize,
        mut slices: Vec<TransferSlice>,
    ) -> Result<Self, TransferError> {
        let invalid = || {
            TransferError::new(
                ErrorKind::InvalidRequest,
                "file ranges overlap or exceed their extent",
            )
        };
        let end = offset.checked_add(bytes as u64).ok_or_else(invalid)?;
        if slices.len() > MAX_FILE_ELEMENTS || bytes > MAX_FILE_BYTES {
            return Err(TransferError::new(
                ErrorKind::ResourceExhausted,
                "file vector budget exceeded",
            ));
        }
        slices.sort_unstable_by_key(|slice| slice.remote_address);
        let mut elements = Vec::new();
        let mut cursor = offset;
        for slice in slices {
            let limit = slice
                .remote_address
                .checked_add(slice.length as u64)
                .ok_or_else(invalid)?;
            if slice.remote_address < cursor || limit > end || slice.length == 0 {
                return Err(invalid());
            }
            if slice.remote_address > cursor {
                elements.push(FileIoElement::Padding(
                    (slice.remote_address - cursor) as usize,
                ));
            }
            elements.push(FileIoElement::Range {
                address: slice.local.as_ptr() as u64,
                length: slice.length,
            });
            cursor = limit;
        }
        if cursor < end {
            elements.push(FileIoElement::Padding((end - cursor) as usize));
        }
        let extent = Self { offset, elements };
        extent.bytes()?;
        Ok(extent)
    }

    pub(crate) fn bytes(&self) -> Result<usize, TransferError> {
        let invalid = || TransferError::new(ErrorKind::InvalidRequest, "invalid file vector");
        if self.elements.is_empty() || self.elements.len() > MAX_FILE_ELEMENTS {
            return Err(TransferError::new(
                ErrorKind::ResourceExhausted,
                "file vector descriptor limit exceeded",
            ));
        }
        let mut bytes = 0usize;
        for element in &self.elements {
            if element.length() == 0 {
                return Err(invalid());
            }
            if let FileIoElement::Range { address, length } = element
                && (*address == 0 || address.checked_add(*length as u64).is_none())
            {
                return Err(invalid());
            }
            bytes = bytes.checked_add(element.length()).ok_or_else(invalid)?;
        }
        self.offset
            .checked_add(bytes as u64)
            .filter(|end| i64::try_from(*end).is_ok())
            .ok_or_else(invalid)?;
        if bytes > MAX_FILE_BYTES {
            return Err(TransferError::new(
                ErrorKind::ResourceExhausted,
                "file vector byte limit exceeded",
            ));
        }
        Ok(bytes)
    }
}

pub(crate) enum FileMemory {
    Pinned(Vec<Arc<PinnedAllocation>>),
    RegisteredCuda {
        owners: Vec<Arc<CudaRegistration>>,
        context: Arc<cudarc::driver::CudaContext>,
    },
}

/// Actual file-transfer backing, immutable layout and the SSD owner's lease.
/// Native callbacks may retain this owner, including after a dropped receiver.
pub struct FileTransfer {
    pub(crate) extent: FileExtent,
    pub(crate) memory: FileMemory,
    pub(crate) _lease: Arc<dyn Send + Sync>,
}

pub(crate) struct PeerReadSlice {
    pub(crate) local: NonNull<u8>,
    pub(crate) remote_address: u64,
    pub(crate) length: usize,
    pub(crate) allocation: Arc<PinnedAllocation>,
}
// SAFETY: each descriptor retains its actual allocation; READ owns exclusive access.
unsafe impl Send for PeerReadSlice {}

#[derive(Debug)]
pub(crate) enum PeerTransferError {
    Timeout,
    Failed(String),
    Unquiesced,
}
impl fmt::Display for PeerTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("peer READ timed out"),
            Self::Failed(reason) => f.write_str(reason),
            Self::Unquiesced => f.write_str("peer READ has not established local quiescence"),
        }
    }
}
impl From<TransferError> for PeerTransferError {
    fn from(error: TransferError) -> Self {
        match error.kind {
            ErrorKind::Timeout => Self::Timeout,
            ErrorKind::Unquiesced => Self::Unquiesced,
            _ => Self::Failed(error.to_string()),
        }
    }
}

impl TransferEngine {
    #[allow(
        clippy::mem_forget,
        reason = "unquiesced work retains actual registrations and engine admission"
    )]
    pub(crate) fn read(
        self: &Arc<Self>,
        peer: crate::peer::segment::PeerOperation,
        slices: &[PeerReadSlice],
        options: TransferOptions,
    ) -> Result<usize, PeerTransferError> {
        let descriptors = slices
            .iter()
            .map(|slice| {
                slice
                    .allocation
                    .source_range(slice.local, slice.length)
                    .map_err(PeerTransferError::Failed)?;
                Ok(TransferSlice {
                    local: slice.local,
                    remote_address: slice.remote_address,
                    length: slice.length,
                })
            })
            .collect::<Result<Vec<_>, PeerTransferError>>()?;
        validate_slices(&descriptors)?;
        let allocations = slices
            .iter()
            .map(|slice| Arc::clone(&slice.allocation))
            .collect::<Vec<_>>();
        let registration = Arc::clone(self).register_pinned(&allocations, Permission::LocalOnly)?;
        // SAFETY: source grants and destination owners stay with TransferLockGuard.
        let result = unsafe {
            peer.transfer(
                TransferOp::Read,
                &descriptors,
                options,
                &Cancellation::default(),
            )
        };
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            std::mem::forget((registration, Arc::clone(self)));
        }
        result.map_err(Into::into)
    }
}

pub(crate) fn remaining(deadline: Instant) -> Result<Duration, TransferError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| TransferError::new(ErrorKind::Timeout, "transfer deadline expired"))
}
pub(crate) fn validate_slices(slices: &[TransferSlice]) -> Result<usize, TransferError> {
    slices.iter().try_fold(0usize, |total, slice| {
        if slice.length == 0
            || (slice.local.as_ptr() as usize)
                .checked_add(slice.length)
                .is_none()
            || slice
                .remote_address
                .checked_add(slice.length as u64)
                .is_none()
        {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "invalid transfer range",
            ));
        }
        total
            .checked_add(slice.length)
            .filter(|bytes| i64::try_from(*bytes).is_ok())
            .ok_or_else(|| {
                TransferError::new(ErrorKind::ResourceExhausted, "transfer byte count overflow")
            })
    })
}

#[cfg(test)]
#[path = "../tests/unit/transfer_engine.rs"]
pub(crate) mod tests;
