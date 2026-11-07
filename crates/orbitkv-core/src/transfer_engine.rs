//! Transfer-engine contracts. Logical failure never establishes physical drain.

mod cuda;
pub use cuda::CudaRegistration;
pub mod local;

use std::fmt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

#[cfg(any(feature = "mooncake", feature = "orbite"))]
use crate::PeerTransferBackend;
use crate::memory::pool::PinnedAllocation;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
use crate::memory::pool::PinnedAllocator;

#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) const MAX_REGISTERED_POOLS: usize = 1024;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

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

/// Requested native access. TENT retains its existing whole-pool registration;
/// per-allocation authorization remains the responsibility of source grants.
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
    #[cfg(feature = "mooncake")]
    pub(crate) fn flag(&self) -> &AtomicBool {
        &self.0
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

pub struct PeerSegment {
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) owner_epoch: String,
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) inner: Segment,
}
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) enum Segment {
    #[cfg(test)]
    Control,
    #[cfg(feature = "mooncake")]
    Tent {
        endpoint: String,
        source_epoch: String,
    },
    #[cfg(feature = "orbite")]
    Orbite(orbite::RemoteSegment),
}

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

/// Engine-local file handle. The SSD owner keeps the original fd and extents.
pub struct FileSegment {
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) owner_epoch: String,
    pub length: u64,
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) inner: FileSegmentInner,
}

#[cfg(any(feature = "mooncake", feature = "orbite"))]
#[derive(Clone)]
pub(crate) enum FileSegmentInner {
    #[cfg(feature = "mooncake")]
    Tent(Arc<orbitkv_transfer::files::FileSegment>),
    #[cfg(feature = "orbite")]
    Orbite(orbite::FileSegment),
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

/// Retains registration, engine and actual backing. Failed cleanup retains all
/// three and their admission charge. Drop is only legal after transfer drain.
pub struct Registration {
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) inner: RegistrationInner,
}
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) enum RegistrationInner {
    #[cfg(test)]
    #[allow(
        dead_code,
        reason = "CPU control retains the actual borrowed owner until registration drop"
    )]
    Control(Arc<dyn Send + Sync>),
    #[cfg(feature = "mooncake")]
    Tent(crate::peer::transport::TentRegistration),
    #[cfg(feature = "orbite")]
    Orbite(crate::peer::orbite::RegistrationSet),
}

/// Selected once per Manager. Source authority stays in OrbitKV's grant owner.
pub trait TransferBackend: Send + Sync {
    fn local_engine(&self) -> crate::PeerTransferBackend {
        crate::PeerTransferBackend::Mooncake
    }

    fn bind_local(
        self: Arc<Self>,
        _stream: Arc<cudarc::driver::CudaStream>,
        _mode: crate::transfer::TransferMode,
        _device: u64,
    ) -> Result<Box<dyn local::LocalCopy>, TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "selected engine has no local copy implementation",
        ))
    }
    fn transfer_endpoint(&self) -> &str;
    fn source_epoch(&self) -> &str;
    fn accepts_endpoint(&self, endpoint: &str) -> bool;

    fn open_file(&self, _path: &Path) -> Result<FileSegment, TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "selected backend does not expose file transports",
        ))
    }

    fn close_file(&self, _file: &FileSegment, _deadline: Instant) -> Result<(), TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "selected backend does not expose file drain",
        ))
    }

    /// # Safety
    /// The supplied owner keeps actual backing and immutable/read-exclusive
    /// extents. The backend owns any file registrations. `remote_address` is a
    /// file byte offset. Unquiesced retains all owners; never replay a failed WRITE.
    /// Every other result proves local drain, including a pre-admission rejection.
    unsafe fn transfer_file(
        self: Arc<Self>,
        _operation: TransferOp,
        _file: &FileSegment,
        _transport: FileTransport,
        _transfer: Arc<FileTransfer>,
        _options: TransferOptions,
        _cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "selected backend does not expose file transfers",
        ))
    }

    /// The caller transfers exclusive access to destination allocations, or
    /// supplies immutable source allocations, through the last admitted operation.
    fn register_pinned(
        self: Arc<Self>,
        allocations: &[Arc<PinnedAllocation>],
        permission: Permission,
    ) -> Result<Registration, TransferError>;

    /// # Safety
    /// `owner` must retain the allocation and CUDA context. Producers must be
    /// complete and aliases coordinated until unregister and all transfers drain.
    unsafe fn register_cuda(
        self: Arc<Self>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
        permission: Permission,
    ) -> Result<Registration, TransferError>;

    /// # Safety
    /// All operations using this registration must have physically drained.
    /// On failure retain the handle; a deadline alone never permits unregister.
    unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError>;

    fn open_peer(
        &self,
        endpoint: &str,
        expected_epoch: &str,
        deadline: Instant,
    ) -> Result<PeerSegment, TransferError>;

    /// # Safety
    /// All slices must be registered, with exclusive READ destinations and
    /// immutable WRITE sources. Retain actual local owners and remote grants
    /// through return; Unquiesced requires retaining them until process exit.
    /// Every other result proves local drain. WRITE failures are never replayed.
    unsafe fn transfer(
        &self,
        operation: TransferOp,
        peer: &PeerSegment,
        slices: &[TransferSlice],
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError>;

    /// Irreversibly fence admission and drain both source and requester work.
    /// An error requires retaining this backend and its registrations.
    fn shutdown(&self, deadline: Instant) -> Result<(), TransferError>;
    /// Retire a membership identity. Errors never supply a release receipt.
    /// Repeated calls may wait again, but must never retry native destruction.
    /// TENT's reversible cache close additionally requires all wrapper requests
    /// to have drained, since a pending wrapper request could reopen its handle.
    fn retire_peer(
        &self,
        _endpoint: &str,
        _epoch: &str,
        _timeout: Duration,
        _requesters_drained: bool,
    ) -> Result<(), TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "peer retirement unavailable",
        ))
    }
    fn invalidate_segment(&self, endpoint: &str);
    fn record_nic_load(&self);
}

#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) struct PeerReadSlice {
    pub(crate) local: NonNull<u8>,
    pub(crate) remote_address: u64,
    pub(crate) length: usize,
    pub(crate) allocation: Arc<PinnedAllocation>,
}
// SAFETY: each descriptor retains its actual allocation; READ owns exclusive access.
#[cfg(any(feature = "mooncake", feature = "orbite"))]
unsafe impl Send for PeerReadSlice {}

#[cfg(any(feature = "mooncake", feature = "orbite"))]
#[derive(Debug)]
pub(crate) enum PeerTransferError {
    Timeout,
    Failed(String),
    Unquiesced,
}
#[cfg(any(feature = "mooncake", feature = "orbite"))]
impl fmt::Display for PeerTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("peer READ timed out"),
            Self::Failed(reason) => f.write_str(reason),
            Self::Unquiesced => f.write_str("peer READ has not established local quiescence"),
        }
    }
}
#[cfg(any(feature = "mooncake", feature = "orbite"))]
impl From<TransferError> for PeerTransferError {
    fn from(error: TransferError) -> Self {
        match error.kind {
            ErrorKind::Timeout => Self::Timeout,
            ErrorKind::Unquiesced => Self::Unquiesced,
            _ => Self::Failed(error.to_string()),
        }
    }
}

#[cfg(any(feature = "mooncake", feature = "orbite"))]
impl dyn TransferBackend {
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

#[cfg(any(feature = "mooncake", feature = "orbite"))]
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

#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) fn create_transfer_backend(
    backend: PeerTransferBackend,
    nics: &[String],
    allocator: Arc<PinnedAllocator>,
    advertise: &str,
    storage: Option<&crate::SsdCacheConfig>,
    #[cfg(feature = "orbite-rdma")] rdma: Option<crate::OrbiteRdmaConfig>,
) -> Result<Arc<dyn TransferBackend>, String> {
    backend.validate_compiled()?;
    #[cfg(feature = "orbite-rdma")]
    if (backend == PeerTransferBackend::OrbiteRdma && rdma.is_none())
        || (rdma.is_some()
            && !matches!(
                backend,
                PeerTransferBackend::Orbite | PeerTransferBackend::OrbiteRdma
            ))
    {
        return Err(
            "OrbiTE RDMA requires a device configuration and orbite or orbite-rdma selection"
                .into(),
        );
    }
    match backend {
        #[cfg(feature = "mooncake")]
        PeerTransferBackend::Mooncake => Ok(Arc::new(
            crate::peer::transport::MooncakeTransport::new(nics, allocator, advertise, storage)?,
        )),
        #[cfg(feature = "orbite")]
        PeerTransferBackend::Orbite
        | PeerTransferBackend::OrbiteTcp
        | PeerTransferBackend::OrbiteRdma => {
            if !nics.is_empty() {
                return Err("OrbiTE does not accept a Mooncake RDMA NIC filter".into());
            }
            Ok(Arc::new(crate::peer::orbite::OrbiteTransport::new(
                allocator.usage().1,
                advertise,
                if backend == PeerTransferBackend::OrbiteRdma {
                    orbite::Transport::Rdma
                } else if backend == PeerTransferBackend::OrbiteTcp {
                    orbite::Transport::Tcp
                } else {
                    orbite::Transport::Unspec
                },
                storage,
                #[cfg(feature = "orbite-rdma")]
                rdma,
            )?))
        }
        #[allow(
            unreachable_patterns,
            reason = "feature combinations omit unavailable constructors"
        )]
        _ => Err("selected transfer backend was not compiled".into()),
    }
}

#[cfg(test)]
#[path = "../tests/unit/transfer_engine.rs"]
pub(crate) mod tests;
