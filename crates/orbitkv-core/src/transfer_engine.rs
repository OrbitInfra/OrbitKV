//! Transfer-engine contracts. Logical failure never establishes physical drain.

use std::fmt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use std::time::Duration;
use std::time::Instant;

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use crate::PeerTransferBackend;
use crate::memory::pool::PinnedAllocation;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use crate::memory::pool::PinnedAllocator;

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) const MAX_REGISTERED_POOLS: usize = 1024;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferOp {
    Read,
    Write,
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
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) owner_epoch: String,
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) inner: Segment,
}
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) enum Segment {
    #[cfg(feature = "mooncake")]
    Tent {
        endpoint: String,
        source_epoch: String,
    },
    #[cfg(feature = "orbitte")]
    OrbitTe(orbit_te::RemoteSegment),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileTransport {
    IoUring,
    Gds,
}

/// Engine-local file handle. The SSD owner keeps the original fd and extents.
pub struct FileSegment {
    #[cfg(feature = "orbitte")]
    pub(crate) owner_epoch: String,
    pub length: u64,
    #[cfg(feature = "orbitte")]
    pub(crate) inner: orbit_te::FileSegment,
}

/// Retains registration, engine and actual backing. Failed cleanup retains all
/// three and their admission charge. Drop is only legal after transfer drain.
pub struct Registration {
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) inner: RegistrationInner,
}
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) enum RegistrationInner {
    #[cfg(feature = "mooncake")]
    Tent(crate::peer::transport::TentRegistration),
    #[cfg(feature = "orbitte")]
    OrbitTe(crate::peer::orbitte::RegistrationSet),
}

/// Selected once per Manager. Source authority stays in OrbitKV's grant owner.
pub trait TransferBackend: Send + Sync {
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
    /// The registration, actual backing and immutable/read-exclusive extents
    /// must remain owned until drain. `remote_address` is a file byte offset.
    /// Unquiesced requires retaining all of them; never replay a failed WRITE.
    /// Every other result proves local drain, including a pre-admission rejection.
    unsafe fn transfer_file(
        &self,
        _operation: TransferOp,
        _file: &FileSegment,
        _transport: FileTransport,
        _slices: &[TransferSlice],
        _deadline: Instant,
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
        deadline: Instant,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError>;

    /// Irreversibly fence admission and drain both source and requester work.
    /// An error requires retaining this backend and its registrations.
    fn shutdown(&self, deadline: Instant) -> Result<(), TransferError>;
    fn invalidate_segment(&self, endpoint: &str);
    fn record_nic_load(&self);
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) struct PeerReadSlice {
    pub(crate) local: NonNull<u8>,
    pub(crate) remote_address: u64,
    pub(crate) length: usize,
    pub(crate) allocation: Arc<PinnedAllocation>,
}
// SAFETY: each descriptor retains its actual allocation; READ owns exclusive access.
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
unsafe impl Send for PeerReadSlice {}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[derive(Debug)]
pub(crate) enum PeerTransferError {
    Timeout,
    Failed(String),
    Unquiesced,
}
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
impl fmt::Display for PeerTransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("peer READ timed out"),
            Self::Failed(reason) => f.write_str(reason),
            Self::Unquiesced => f.write_str("peer READ has not established local quiescence"),
        }
    }
}
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
impl From<TransferError> for PeerTransferError {
    fn from(error: TransferError) -> Self {
        match error.kind {
            ErrorKind::Timeout => Self::Timeout,
            ErrorKind::Unquiesced => Self::Unquiesced,
            _ => Self::Failed(error.to_string()),
        }
    }
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
impl dyn TransferBackend {
    #[allow(
        clippy::mem_forget,
        reason = "unquiesced work retains actual registrations and engine admission"
    )]
    pub(crate) fn read(
        self: &Arc<Self>,
        endpoint: &str,
        slices: &[PeerReadSlice],
        timeout: Duration,
        expected_epoch: &str,
    ) -> Result<usize, PeerTransferError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(PeerTransferError::Timeout)?;
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
        let peer = self.open_peer(endpoint, expected_epoch, deadline)?;
        // SAFETY: source grants and destination owners stay with TransferLockGuard.
        let result = unsafe {
            self.transfer(
                TransferOp::Read,
                &peer,
                &descriptors,
                deadline,
                &Cancellation::default(),
            )
        };
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            std::mem::forget((registration, Arc::clone(self)));
        }
        result.map_err(Into::into)
    }
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
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

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) fn create_transfer_backend(
    backend: PeerTransferBackend,
    nics: &[String],
    allocator: Arc<PinnedAllocator>,
    advertise: &str,
    storage: Option<&crate::SsdCacheConfig>,
    #[cfg(feature = "orbitte-rdma")] rdma: Option<crate::OrbitTeRdmaConfig>,
) -> Result<Arc<dyn TransferBackend>, String> {
    backend.validate_compiled()?;
    if storage.is_some() && backend == PeerTransferBackend::Mooncake {
        return Err("TENT file transport configuration is not exposed".into());
    }
    #[cfg(feature = "orbitte-rdma")]
    if (backend == PeerTransferBackend::OrbitTeRdma) != rdma.is_some() {
        return Err(
            "OrbitTE RDMA requires an explicit configuration and orbitte-rdma selection".into(),
        );
    }
    match backend {
        #[cfg(feature = "mooncake")]
        PeerTransferBackend::Mooncake => Ok(Arc::new(
            crate::peer::transport::MooncakeTransport::new(nics, allocator, advertise)?,
        )),
        #[cfg(feature = "orbitte")]
        PeerTransferBackend::OrbitTeTcp | PeerTransferBackend::OrbitTeRdma => {
            if !nics.is_empty() {
                return Err("OrbitTE does not accept a Mooncake RDMA NIC filter".into());
            }
            Ok(Arc::new(crate::peer::orbitte::OrbitTeTransport::new(
                allocator.usage().1,
                advertise,
                if backend == PeerTransferBackend::OrbitTeRdma {
                    orbit_te::Transport::Rdma
                } else {
                    orbit_te::Transport::Tcp
                },
                storage,
                #[cfg(feature = "orbitte-rdma")]
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
