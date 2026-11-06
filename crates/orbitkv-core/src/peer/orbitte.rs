mod local;
use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orbit_te::{
    EngineConfig, Error, Permission, RegisteredMemory, TransferEngine, TransferOp, TransferRequest,
    Transport,
};

use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileIoElement, FileMemory, FileSegment, FileSegmentInner,
    FileTransfer, FileTransport, MAX_REGISTERED_BYTES, MAX_REGISTERED_POOLS, PeerSegment,
    REGISTERED_POOLS, Registration, RegistrationInner, Segment, TransferBackend, TransferError,
    TransferSlice, remaining, validate_slices,
};

fn prefix(transport: Transport) -> &'static str {
    match transport {
        Transport::Tcp => "orbitte+tcp://",
        Transport::Rdma => "orbitte+rdma://",
        _ => "orbitte+unsupported://",
    }
}
const MAX_BATCH_REQUESTS: usize = 16_384;
const MAX_BATCH_BYTES: usize = 64 << 20;

pub(super) fn parse_endpoint(transport: Transport, endpoint: &str) -> Result<SocketAddr, String> {
    endpoint
        .strip_prefix(prefix(transport))
        .ok_or("peer OrbitTE payload protocol mismatch")?
        .parse()
        .map_err(|error| format!("invalid OrbitTE endpoint: {error}"))
}

pub(crate) struct OrbitTeTransport {
    engine: Arc<TransferEngine>,
    pub(super) endpoint: String,
    pub(super) epoch: String,
    pool_bytes: u64,
    files_enabled: bool,
    pub(super) payload: Transport,
}

/// Explicit registration references plus the actual allocation and provider
/// lifetime. Source tickets and requester READ owners hold these together.
pub(crate) struct RegistrationSet {
    transport: Arc<OrbitTeTransport>,
    regions: Vec<RegisteredMemory>,
    cleanup_failed: bool,
}

impl RegistrationSet {
    fn unregister(&mut self) -> Result<(), TransferError> {
        if self.cleanup_failed {
            return Err(TransferError::new(
                ErrorKind::Unquiesced,
                "OrbitTE registration cleanup remains retained",
            ));
        }
        let result = self.transport.unregister(&mut self.regions);
        self.cleanup_failed = result.is_err();
        result
    }
}

impl Drop for RegistrationSet {
    #[allow(
        clippy::mem_forget,
        reason = "failed unregister must retain the actual backing and charged engine admission"
    )]
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            log::error!(
                "Retaining OrbitTE registration ownership after unregister failed: {error}"
            );
            std::mem::forget((
                Arc::clone(&self.transport),
                std::mem::take(&mut self.regions),
            ));
        }
    }
}

impl OrbitTeTransport {
    pub(crate) fn new(
        pool_bytes: u64,
        advertise: &str,
        payload: Transport,
        storage: Option<&crate::SsdCacheConfig>,
        #[cfg(feature = "orbitte-rdma")] rdma: Option<crate::OrbitTeRdmaConfig>,
    ) -> Result<Self, String> {
        if !matches!(payload, Transport::Tcp | Transport::Rdma) {
            return Err("unsupported OrbitTE peer payload".into());
        }
        #[cfg(not(feature = "orbitte-rdma"))]
        if payload == Transport::Rdma {
            return Err("OrbitTE RDMA requires the orbitte-rdma build feature".into());
        }
        #[cfg(feature = "orbitte-rdma")]
        if (payload == Transport::Rdma) != rdma.is_some() {
            return Err("OrbitTE RDMA requires an explicit device configuration".into());
        }
        let endpoint = advertise
            .strip_prefix("http://")
            .or_else(|| advertise.strip_prefix("https://"))
            .unwrap_or(advertise);
        let (host, _) = endpoint
            .rsplit_once(':')
            .ok_or("invalid peer advertise address")?;
        let host: IpAddr = host
            .trim_matches(['[', ']'])
            .parse()
            .map_err(|_| "OrbitTE requires a numeric advertised IP")?;
        if host.is_unspecified() {
            return Err("OrbitTE requires a concrete advertised IP".into());
        }
        let mut policy = vec![orbit_te::SelectionPolicy {
            transports: vec![payload],
            ..Default::default()
        }];
        if storage.is_some() {
            policy.push(orbit_te::SelectionPolicy {
                name: "ssd".into(),
                segment_type: orbit_te::SegmentType::File,
                transports: vec![Transport::IoUring, Transport::Gds],
                ..Default::default()
            });
        }
        if storage.is_some_and(|s| s.backend == crate::SsdBackend::Cufile)
            && !cfg!(feature = "orbitte-gds")
        {
            return Err("OrbitTE cuFile requires the orbitte-gds build feature".into());
        }
        let config = EngineConfig {
            #[cfg(feature = "orbitte-rdma")]
            rdma,
            bind_addr: SocketAddr::new(host, 0),
            max_registered_regions: 4096,
            max_registered_bytes: usize::try_from(pool_bytes)
                .map_err(|_| "pool exceeds addressable bytes")?
                .checked_add(2 << 30) // Persistent GPU registrations share this headroom.
                .ok_or("registration budget overflow")?,
            max_batch_requests: MAX_BATCH_REQUESTS,
            max_inflight_batches: 32,
            max_inflight_bytes: MAX_BATCH_BYTES,
            max_transfer_bytes: MAX_BATCH_BYTES,
            max_allocated_batches: 32,
            max_allocated_requests: 32 * MAX_BATCH_REQUESTS,
            max_file_segments: if storage.is_some() { 256 } else { 0 },
            #[cfg(feature = "orbitte-gds")]
            gds_require_native: true,
            #[cfg(feature = "orbitte-gds")]
            gds_enabled: storage.is_some_and(|s| s.backend == crate::SsdBackend::Cufile),
            max_failover_attempts: 0,
            enable_auto_failover_on_poll: false,
            tcp: payload == Transport::Tcp,
            selector: Some(orbit_te::TransportSelectorConfig {
                policy,
                ..Default::default()
            }),
            // No implicit expansion of the qualified payload namespace.
            hp_tcp: Default::default(),
            shm: Default::default(),
            nvlink: false,
            mnnvl: None,
            remote_staging: false,
            ..EngineConfig::default()
        };
        let engine = Arc::new(TransferEngine::new(config).map_err(|error| error.to_string())?);
        let epoch = engine.source_epoch().to_string();
        let endpoint = format!("{}{}", prefix(payload), engine.endpoint());
        {
            let mut budget = REGISTERED_POOLS
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let bytes = budget
                .1
                .checked_add(pool_bytes)
                .ok_or("source pool accounting overflow")?;
            if budget.0 >= MAX_REGISTERED_POOLS || bytes > MAX_REGISTERED_BYTES {
                return Err("registered source pool admission exhausted".into());
            }
            budget.0 += 1;
            budget.1 = bytes;
        }
        log::info!("OrbitTE peer payload initialized: {endpoint}");
        Ok(Self {
            engine,
            endpoint,
            epoch,
            pool_bytes,
            files_enabled: storage.is_some(),
            payload,
        })
    }

    fn transfer_target(
        &self,
        operation: crate::transfer_engine::TransferOp,
        target: orbit_te::TransferTarget<'_>,
        transport: Transport,
        slices: &[TransferSlice],
        deadline: Instant,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let total = validate_profile(slices)?;
        if cancellation.is_cancelled() {
            return Err(TransferError::new(
                ErrorKind::Cancelled,
                "transfer cancelled before submission",
            ));
        }
        let requests: Vec<_> = slices
            .iter()
            .map(|slice| {
                TransferRequest::new(
                    transport,
                    slice.local.as_ptr() as u64,
                    slice.remote_address,
                    slice.length,
                )
                .with_deadline(deadline)
            })
            .collect();
        self.transfer_requests(operation, target, &requests, total, deadline, cancellation)
    }

    fn transfer_requests(
        &self,
        operation: crate::transfer_engine::TransferOp,
        target: orbit_te::TransferTarget<'_>,
        requests: &[TransferRequest],
        total: usize,
        deadline: Instant,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let op = match operation {
            crate::transfer_engine::TransferOp::Read => TransferOp::Read,
            crate::transfer_engine::TransferOp::Write => TransferOp::Write,
        };
        let mut batch = self
            .engine
            .submit_async(op, target, requests, None, remaining(deadline)?)
            .map_err(transfer_error)?;
        let mut cancelled = false;
        loop {
            if cancellation.is_cancelled() && !cancelled {
                // Unsupported running cancellation still requires drain; queued
                // physical owners have received cancellation before this returns.
                let _ = batch.cancel_all();
                cancelled = true;
            }
            let status = batch
                .status()
                .map_err(|e| TransferError::new(ErrorKind::Unquiesced, e.to_string()))?;
            if status.is_terminal() {
                if !status.locally_drained {
                    return Err(TransferError::new(
                        ErrorKind::Unquiesced,
                        "OrbitTE physical owners remain active",
                    ));
                }
                let submissions = batch
                    .submissions()
                    .map_err(|e| TransferError::new(ErrorKind::Unquiesced, e.to_string()))?;
                // Per-task bytes can alias a merged physical owner. Only the
                // submission report is additive across public descriptors.
                let bytes = submissions
                    .iter()
                    .map(|report| report.transferred_bytes)
                    .sum::<usize>();
                let failure = submissions
                    .iter()
                    .find_map(|report| report.failure.as_ref())
                    .map(|failure| {
                        if failure.remote_effect == orbit_te::RemoteEffect::Unknown {
                            TransferError::new(
                                ErrorKind::Unquiesced,
                                "OrbitTE remote WRITE effect is unknown",
                            )
                        } else {
                            transfer_error_ref(&failure.error)
                        }
                    });
                batch
                    .free()
                    .map_err(|e| TransferError::new(ErrorKind::Unquiesced, e.to_string()))?;
                if let Some(error) = failure {
                    return Err(error);
                }
                if cancelled {
                    return Err(TransferError::new(
                        ErrorKind::Cancelled,
                        "transfer cancelled after drain",
                    ));
                }
                if bytes != total {
                    return Err(TransferError::new(
                        ErrorKind::Backend,
                        "OrbitTE returned incomplete bytes",
                    ));
                }
                return Ok(bytes);
            }
            let Ok(wait) = remaining(deadline) else {
                let _ = batch.cancel_all();
                // The batch owns accepted work after handle drop; the caller
                // must retain registrations, source grants and actual buffers.
                return Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "OrbitTE deadline expired before physical drain",
                ));
            };
            match batch.wait(wait.min(Duration::from_millis(5))) {
                Ok(_) | Err(Error::Timeout) => {}
                Err(error) => {
                    return Err(TransferError::new(ErrorKind::Unquiesced, error.to_string()));
                }
            }
        }
    }

    pub(super) fn register<'a>(
        self: &Arc<Self>,
        allocations: impl Iterator<Item = &'a Arc<PinnedAllocation>>,
        permission: Permission,
    ) -> Result<RegistrationSet, TransferError> {
        let mut seen = HashSet::new();
        let ranges = allocations
            .filter(|allocation| seen.insert(Arc::as_ptr(allocation) as usize))
            .map(|allocation| {
                // SAFETY: sealed source blocks are immutable. Requester ranges
                // are newly allocated and handed exclusively to the READ owner.
                // All aliases use the allocation's canonical shared access lock.
                let memory = unsafe { allocation.orbitte_memory() }.map_err(transfer_error)?;
                let length = memory.len();
                Ok((memory, 0, length))
            })
            .collect::<Result<Vec<_>, TransferError>>()?;
        let regions = self
            .engine
            .register_host_memory_batch(&ranges, permission)
            .map_err(transfer_error)?;
        Ok(RegistrationSet {
            transport: Arc::clone(self),
            regions,
            cleanup_failed: false,
        })
    }

    fn unregister(&self, regions: &mut Vec<RegisteredMemory>) -> Result<(), TransferError> {
        while let Some(region) = regions.last() {
            self.engine
                .unregister_memory(region)
                .map_err(transfer_error)?;
            regions.pop();
        }
        Ok(())
    }
}

impl TransferBackend for OrbitTeTransport {
    fn local_engine(&self) -> crate::PeerTransferBackend {
        match self.payload {
            Transport::Rdma => crate::PeerTransferBackend::OrbitTeRdma,
            _ => crate::PeerTransferBackend::OrbitTeTcp,
        }
    }

    fn bind_local(
        self: Arc<Self>,
        stream: Arc<cudarc::driver::CudaStream>,
        mode: crate::transfer::TransferMode,
        device: u64,
    ) -> Result<Box<dyn crate::transfer_engine::local::LocalCopy>, TransferError> {
        Ok(Box::new(local::Copies::new(self, stream, mode, device)?))
    }

    fn transfer_endpoint(&self) -> &str {
        &self.endpoint
    }
    fn source_epoch(&self) -> &str {
        &self.epoch
    }
    fn accepts_endpoint(&self, endpoint: &str) -> bool {
        parse_endpoint(self.payload, endpoint).is_ok()
    }

    fn register_pinned(
        self: Arc<Self>,
        allocations: &[Arc<PinnedAllocation>],
        permission: crate::transfer_engine::Permission,
    ) -> Result<Registration, TransferError> {
        self.register(allocations.iter(), permission.into())
            .map(|registration| Registration {
                inner: RegistrationInner::OrbitTe(registration),
            })
    }

    unsafe fn register_cuda(
        self: Arc<Self>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
        permission: crate::transfer_engine::Permission,
    ) -> Result<Registration, TransferError> {
        // SAFETY: the caller retains the allocation/context and coordinates aliases.
        let memory = Arc::new(
            unsafe { orbit_te::cuda::CudaMemory::from_raw_parts(address, length, owner) }
                .map_err(transfer_error)?,
        );
        if !matches!(memory.location(0, length).map_err(transfer_error)?.location,
            orbit_te::MemoryLocation::Cuda { device: actual, .. } if actual == device)
        {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "CUDA device does not match the retained allocation",
            ));
        }
        let regions = self
            .engine
            .register_cuda_memory_batch(&[(memory, 0, length)], permission.into())
            .map_err(transfer_error)?;
        Ok(Registration {
            inner: RegistrationInner::OrbitTe(RegistrationSet {
                transport: self,
                regions,
                cleanup_failed: false,
            }),
        })
    }

    unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError> {
        match &mut registration.inner {
            RegistrationInner::OrbitTe(registration)
                if registration.transport.epoch == self.epoch =>
            {
                registration.unregister()
            }
            _ => Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign registration",
            )),
        }
    }

    fn open_peer(
        &self,
        endpoint: &str,
        expected_epoch: &str,
        deadline: Instant,
    ) -> Result<PeerSegment, TransferError> {
        let endpoint = parse_endpoint(self.payload, endpoint)
            .map_err(|e| TransferError::new(ErrorKind::InvalidRequest, e))?;
        let epoch = expected_epoch.parse().map_err(transfer_error)?;
        let peer = self
            .engine
            .open_segment(endpoint, epoch, remaining(deadline)?)
            .map_err(transfer_error)?;
        Ok(PeerSegment {
            owner_epoch: self.epoch.clone(),
            inner: Segment::OrbitTe(peer),
        })
    }

    unsafe fn transfer(
        &self,
        operation: crate::transfer_engine::TransferOp,
        peer: &PeerSegment,
        slices: &[TransferSlice],
        deadline: Instant,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        validate_profile(slices)?;
        if peer.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign peer segment",
            ));
        }
        #[cfg_attr(
            not(feature = "mooncake"),
            allow(
                clippy::infallible_destructuring_match,
                reason = "single-provider builds have one segment variant"
            )
        )]
        let peer = match &peer.inner {
            Segment::OrbitTe(peer) => peer,
            #[cfg(any(feature = "mooncake", test))]
            _ => {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "foreign peer protocol",
                ));
            }
        };
        self.transfer_target(
            operation,
            peer.into(),
            Transport::Unspec,
            slices,
            deadline,
            cancellation,
        )
    }

    fn open_file(&self, path: &std::path::Path) -> Result<FileSegment, TransferError> {
        if !self.files_enabled {
            return Err(TransferError::new(
                ErrorKind::Unavailable,
                "OrbitTE file admission is disabled",
            ));
        }
        let path = path.to_str().ok_or_else(|| {
            TransferError::new(ErrorKind::InvalidRequest, "file path must be UTF-8")
        })?;
        let inner = self
            .engine
            .open_file_segment(&format!("file://{path}"), Permission::RemoteReadWrite)
            .map_err(transfer_error)?;
        match self.engine.file_segment_info(&inner) {
            Ok(info) => Ok(FileSegment {
                owner_epoch: self.epoch.clone(),
                length: info.length,
                inner: FileSegmentInner::OrbitTe(inner),
            }),
            Err(error) => {
                let _ = self.engine.close_file_segment(&inner, Duration::ZERO);
                Err(transfer_error(error))
            }
        }
    }

    fn close_file(&self, file: &FileSegment, deadline: Instant) -> Result<(), TransferError> {
        if file.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        #[cfg(feature = "mooncake")]
        let FileSegmentInner::OrbitTe(inner) = &file.inner else {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file provider",
            ));
        };
        #[cfg(not(feature = "mooncake"))]
        let FileSegmentInner::OrbitTe(inner) = &file.inner;
        self.engine
            .close_file_segment(inner, deadline.saturating_duration_since(Instant::now()))
            .map_err(transfer_error)
    }

    #[allow(
        clippy::mem_forget,
        reason = "file backing, registrations and extent leases survive unknown drain or cleanup"
    )]
    unsafe fn transfer_file(
        self: Arc<Self>,
        operation: crate::transfer_engine::TransferOp,
        file: &FileSegment,
        transport: FileTransport,
        transfer: Arc<FileTransfer>,
        deadline: Instant,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        if file.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        #[cfg(feature = "mooncake")]
        let FileSegmentInner::OrbitTe(inner) = &file.inner else {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file provider",
            ));
        };
        #[cfg(not(feature = "mooncake"))]
        let FileSegmentInner::OrbitTe(inner) = &file.inner;
        remaining(deadline)?;
        if cancellation.is_cancelled() {
            return Err(TransferError::new(
                ErrorKind::Cancelled,
                "file transfer cancelled before admission",
            ));
        }
        let elements: Vec<_> = transfer
            .extent
            .elements
            .iter()
            .map(|element| match *element {
                FileIoElement::Range { address, length } => {
                    orbit_te::FileIoElement::Range { address, length }
                }
                FileIoElement::Padding(bytes) => orbit_te::FileIoElement::Padding(bytes),
            })
            .collect();
        let total = transfer.extent.bytes()?;
        let request = TransferRequest::file_vectored(
            match transport {
                FileTransport::IoUring => Transport::IoUring,
                FileTransport::Gds => Transport::Gds,
            },
            transfer.extent.offset,
            &elements,
        )
        .map_err(transfer_error)?
        .with_deadline(deadline);
        let registration = match &transfer.memory {
            FileMemory::Pinned(allocations) if allocations.is_empty() => Ok(None),
            FileMemory::Pinned(allocations) => Arc::clone(&self)
                .register_pinned(allocations, crate::transfer_engine::Permission::LocalOnly)
                .map(Some),
            // EngineFile holds persistent registration loans through this receipt.
            FileMemory::RegisteredCuda { context, .. } => {
                context
                    .bind_to_thread()
                    .map_err(|e| TransferError::new(ErrorKind::Backend, e.to_string()))?;
                Ok(None)
            }
        };
        let mut registration = ManuallyDrop::new(match registration {
            Ok(registration) => registration,
            Err(error) => {
                if error.kind == ErrorKind::Unquiesced {
                    std::mem::forget(transfer);
                }
                return Err(error);
            }
        });
        let mut result = self.transfer_requests(
            operation,
            inner.into(),
            &[request],
            total,
            deadline,
            cancellation,
        );
        if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced)
            && let Some(registration) = registration.as_mut()
        {
            // SAFETY: every other result proves physical drain or rejection.
            if let Err(error) = unsafe { TransferBackend::unregister(self.as_ref(), registration) }
            {
                result = Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    format!("file registration cleanup failed: {error}"),
                ));
            }
        }
        if matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
            std::mem::forget(transfer);
        } else {
            // SAFETY: transfer drain and explicit registration cleanup both succeeded.
            unsafe { ManuallyDrop::drop(&mut registration) };
        }
        result
    }

    fn shutdown(&self, deadline: Instant) -> Result<(), TransferError> {
        self.engine
            .close(deadline.saturating_duration_since(Instant::now()))
            .map_err(transfer_error)
    }
    fn invalidate_segment(&self, _endpoint: &str) {}
    fn record_nic_load(&self) {}
}

impl From<crate::transfer_engine::Permission> for Permission {
    fn from(permission: crate::transfer_engine::Permission) -> Self {
        match permission {
            crate::transfer_engine::Permission::LocalOnly => Self::LocalOnly,
            crate::transfer_engine::Permission::RemoteRead => Self::RemoteRead,
            crate::transfer_engine::Permission::RemoteReadWrite => Self::RemoteReadWrite,
        }
    }
}
fn validate_profile(slices: &[TransferSlice]) -> Result<usize, TransferError> {
    if slices.len() > MAX_BATCH_REQUESTS {
        return Err(TransferError::new(
            ErrorKind::ResourceExhausted,
            "transfer descriptor budget exhausted",
        ));
    }
    let total = validate_slices(slices)?;
    if total > MAX_BATCH_BYTES {
        return Err(TransferError::new(
            ErrorKind::ResourceExhausted,
            "transfer exceeds the 64 MiB profile budget",
        ));
    }
    Ok(total)
}

fn transfer_error_ref(error: &Error) -> TransferError {
    let kind = match error {
        Error::Timeout | Error::DeadlineInfeasible => ErrorKind::Timeout,
        Error::Cancelled => ErrorKind::Cancelled,
        Error::Unquiesced { .. } | Error::BatchBusy => ErrorKind::Unquiesced,
        Error::EpochMismatch => ErrorKind::EpochMismatch,
        Error::PermissionDenied => ErrorKind::PermissionDenied,
        Error::ResourceExhausted => ErrorKind::ResourceExhausted,
        Error::Unavailable(_) | Error::SourceClosed | Error::RequesterClosed => {
            ErrorKind::Unavailable
        }
        Error::InvalidRange
        | Error::ForeignRegistration
        | Error::RegionNotFound
        | Error::RegistrationConflict
        | Error::Protocol(_) => ErrorKind::InvalidRequest,
        _ => ErrorKind::Backend,
    };
    TransferError::new(kind, error.to_string())
}
fn transfer_error(error: Error) -> TransferError {
    transfer_error_ref(&error)
}

impl Drop for OrbitTeTransport {
    #[allow(
        clippy::mem_forget,
        reason = "a failed drain must retain the engine and its process admission until exit"
    )]
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let drained = self.shutdown(deadline);
        if let Err(error) = drained {
            log::error!(
                "Retaining OrbitTE engine and {} source pool budget bytes after drain failed: {error}",
                self.pool_bytes
            );
            std::mem::forget(Arc::clone(&self.engine));
            return;
        }
        let mut budget = REGISTERED_POOLS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        budget.0 -= 1;
        budget.1 -= self.pool_bytes;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/peer/orbitte.rs"]
mod tests;
