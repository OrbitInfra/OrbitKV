mod local;
use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use orbite::{
    BatchState, EngineConfig, Error, Permission, RegisteredMemory, TransferEngine as NativeEngine,
    TransferOp, TransferRequest, Transport,
};

use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::backend::{
    MAX_REGISTERED_BYTES, MAX_REGISTERED_POOLS, REGISTERED_POOLS, RegistrationInner, Segment,
};
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileIoElement, FileMemory, FileSegment, FileTransfer, FileTransport,
    PeerSegment, Registration, TransferError, TransferOptions, TransferSlice, remaining,
    validate_slices,
};

fn request_options(request: TransferRequest, options: TransferOptions) -> TransferRequest {
    use crate::transfer_engine::{TransferIntent, TransferPriority};
    request
        .with_deadline(options.deadline)
        .with_intent(match options.intent {
            TransferIntent::ForegroundGet => orbite::IntentType::ForegroundGet,
            TransferIntent::BackgroundPrefetch => orbite::IntentType::BackgroundPrefetch,
            TransferIntent::Checkpoint => orbite::IntentType::Checkpoint,
        })
        .with_priority(match options.intent.priority() {
            TransferPriority::High => orbite::Priority::High,
            TransferPriority::Low => orbite::Priority::Low,
        })
}

fn prefix(transport: Transport) -> &'static str {
    match transport {
        Transport::Unspec => "orbite://",
        Transport::Tcp => "orbite+tcp://",
        Transport::Rdma => "orbite+rdma://",
        _ => "orbite+unsupported://",
    }
}
const MAX_BATCH_REQUESTS: usize = 16_384;
const MAX_BATCH_BYTES: usize = 64 << 20;

pub(super) fn parse_endpoint(endpoint: &str) -> Result<SocketAddr, String> {
    ["orbite://", "orbite+tcp://", "orbite+rdma://"]
        .into_iter()
        .find_map(|scheme| endpoint.strip_prefix(scheme))
        .ok_or("peer OrbiTE payload protocol mismatch")?
        .parse()
        .map_err(|error| format!("invalid OrbiTE endpoint: {error}"))
}

pub struct TransferEngine {
    engine: Arc<NativeEngine>,
    pub(super) endpoint: String,
    pub(super) epoch: String,
    pool_bytes: u64,
    files_enabled: bool,
    pub(super) payload: Transport,
    drained: AtomicBool,
    #[cfg(test)]
    pub(crate) control: Option<Arc<crate::peer::restore::tests::Control>>,
}

/// Explicit registration references plus the actual allocation and provider
/// lifetime. Source tickets and requester READ owners hold these together.
pub(crate) struct RegistrationSet {
    transport: Arc<TransferEngine>,
    regions: Vec<RegisteredMemory>,
    cleanup_failed: bool,
}

impl RegistrationSet {
    fn unregister(&mut self) -> Result<(), TransferError> {
        if self.cleanup_failed {
            return Err(TransferError::new(
                ErrorKind::Unquiesced,
                "OrbiTE registration cleanup remains retained",
            ));
        }
        let result = self.transport.unregister_regions(&mut self.regions);
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
            log::error!("Retaining OrbiTE registration ownership after unregister failed: {error}");
            std::mem::forget((
                Arc::clone(&self.transport),
                std::mem::take(&mut self.regions),
            ));
        }
    }
}

impl TransferEngine {
    pub(crate) fn new(
        pool_bytes: u64,
        advertise: &str,
        payload: Transport,
        storage: Option<&crate::SsdCacheConfig>,
        #[cfg(feature = "orbite-rdma")] rdma: Option<crate::OrbiteRdmaConfig>,
    ) -> Result<Self, String> {
        if !matches!(
            payload,
            Transport::Unspec | Transport::Tcp | Transport::Rdma
        ) {
            return Err("unsupported OrbiTE peer payload".into());
        }
        #[cfg(not(feature = "orbite-rdma"))]
        if payload == Transport::Rdma {
            return Err("OrbiTE RDMA requires the orbite-rdma build feature".into());
        }
        #[cfg(feature = "orbite-rdma")]
        if (payload == Transport::Rdma && rdma.is_none())
            || (payload == Transport::Tcp && rdma.is_some())
        {
            return Err("OrbiTE RDMA requires an explicit device configuration".into());
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
            .map_err(|_| "OrbiTE requires a numeric advertised IP")?;
        if host.is_unspecified() {
            return Err("OrbiTE requires a concrete advertised IP".into());
        }
        let mut policy = vec![orbite::SelectionPolicy::default()];
        if storage.is_some() {
            let mut file_policy = orbite::SelectionPolicy::default();
            file_policy.name = "ssd".into();
            file_policy.segment_type = orbite::SegmentType::File;
            file_policy.transports = vec![Transport::IoUring, Transport::Gds];
            policy.push(file_policy);
        }
        if storage.is_some_and(|s| s.backend == crate::SsdBackend::Cufile)
            && !cfg!(feature = "orbite-gds")
        {
            return Err("OrbiTE cuFile requires the orbite-gds build feature".into());
        }
        let mut selector = orbite::TransportSelectorConfig::default();
        selector.policy = policy;
        let mut config = EngineConfig::default();
        #[cfg(feature = "orbite-rdma")]
        {
            config.rdma = rdma;
        }
        config.bind_addr = SocketAddr::new(host, 0);
        config.max_registered_regions = 4096;
        config.max_registered_bytes = usize::try_from(pool_bytes)
            .map_err(|_| "pool exceeds addressable bytes")?
            .checked_add(2 << 30) // Persistent GPU registrations share this headroom.
            .ok_or("registration budget overflow")?;
        config.max_batch_requests = MAX_BATCH_REQUESTS;
        config.max_inflight_batches = 32;
        config.max_inflight_bytes = MAX_BATCH_BYTES;
        config.max_transfer_bytes = MAX_BATCH_BYTES;
        config.max_allocated_batches = 32;
        config.max_allocated_requests = 32 * MAX_BATCH_REQUESTS;
        config.max_file_segments = if storage.is_some() { 256 } else { 0 };
        #[cfg(feature = "orbite-gds")]
        {
            config.gds_require_native = true;
            config.gds_enabled = storage.is_some_and(|s| s.backend == crate::SsdBackend::Cufile);
        }
        config.max_failover_attempts = 0;
        config.enable_auto_failover_on_poll = false;
        config.tcp = payload != Transport::Rdma;
        config.selector = Some(selector);
        // Auto selection uses only enabled providers; extra transports stay opt-in.
        config.hp_tcp = Default::default();
        config.shm = Default::default();
        config.nvlink = false;
        config.mnnvl = None;
        config.remote_staging = false;
        let engine = Arc::new(NativeEngine::new(config).map_err(|error| error.to_string())?);
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
        log::info!("OrbiTE peer payload initialized: {endpoint}");
        Ok(Self {
            engine,
            endpoint,
            epoch,
            pool_bytes,
            files_enabled: storage.is_some(),
            payload,
            drained: AtomicBool::new(false),
            #[cfg(test)]
            control: None,
        })
    }

    fn transfer_target(
        &self,
        operation: crate::transfer_engine::TransferOp,
        target: orbite::TransferTarget<'_>,
        transport: Transport,
        slices: &[TransferSlice],
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let deadline = options.deadline;
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
                request_options(
                    TransferRequest::new(
                        transport,
                        slice.local.as_ptr() as u64,
                        slice.remote_address,
                        slice.length,
                    ),
                    options,
                )
            })
            .collect();
        self.transfer_requests(operation, target, &requests, total, deadline, cancellation)
    }

    fn transfer_requests(
        &self,
        operation: crate::transfer_engine::TransferOp,
        target: orbite::TransferTarget<'_>,
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
            let terminal = match status.state {
                BatchState::Open | BatchState::Pending => false,
                BatchState::Completed | BatchState::Failed | BatchState::Cancelled => true,
                _ => {
                    return Err(TransferError::new(
                        ErrorKind::Unquiesced,
                        "unknown OrbiTE batch state is not a drain receipt",
                    ));
                }
            };
            if terminal {
                if !status.locally_drained {
                    return Err(TransferError::new(
                        ErrorKind::Unquiesced,
                        "OrbiTE physical owners remain active",
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
                        if failure.remote_effect != orbite::RemoteEffect::None {
                            TransferError::new(
                                ErrorKind::Unquiesced,
                                "OrbiTE remote WRITE effect is unknown",
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
                if status.state != BatchState::Completed {
                    return Err(TransferError::new(
                        ErrorKind::Backend,
                        "OrbiTE batch did not complete successfully",
                    ));
                }
                if bytes != total {
                    return Err(TransferError::new(
                        ErrorKind::Backend,
                        "OrbiTE returned incomplete bytes",
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
                    "OrbiTE deadline expired before physical drain",
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
                let memory = unsafe { allocation.orbite_memory() }.map_err(transfer_error)?;
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

    fn unregister_regions(&self, regions: &mut Vec<RegisteredMemory>) -> Result<(), TransferError> {
        while let Some(region) = regions.last() {
            self.engine
                .unregister_memory(region)
                .map_err(transfer_error)?;
            regions.pop();
        }
        Ok(())
    }
}

impl TransferEngine {
    pub fn bind_local(
        self: Arc<Self>,
        stream: Arc<cudarc::driver::CudaStream>,
        mode: crate::transfer::TransferMode,
        device: u64,
    ) -> Result<Box<dyn crate::transfer_engine::local::LocalCopy>, TransferError> {
        Ok(Box::new(local::Copies::new(self, stream, mode, device)?))
    }

    pub fn transfer_endpoint(&self) -> &str {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.transfer_endpoint();
        }

        &self.endpoint
    }
    pub fn source_epoch(&self) -> &str {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.source_epoch();
        }

        &self.epoch
    }
    pub fn accepts_endpoint(&self, endpoint: &str) -> bool {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.accepts_endpoint(endpoint);
        }

        parse_endpoint(endpoint).is_ok()
    }

    pub fn register_pinned(
        self: Arc<Self>,
        allocations: &[Arc<PinnedAllocation>],
        permission: crate::transfer_engine::Permission,
    ) -> Result<Registration, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return Arc::clone(control).register_pinned(allocations, permission);
        }

        self.register(allocations.iter(), permission.into())
            .map(|registration| Registration {
                inner: RegistrationInner::Orbite(registration),
            })
    }

    /// # Safety
    /// The owner retains the CUDA allocation and context; coordinate producers and aliases through physical drain and unregister.
    pub unsafe fn register_cuda(
        self: Arc<Self>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
        permission: crate::transfer_engine::Permission,
    ) -> Result<Registration, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            // SAFETY: the same caller ownership and drain contract applies to the fault control.
            return unsafe {
                Arc::clone(control).register_cuda(address, length, device, owner, permission)
            };
        }

        // SAFETY: the caller retains the allocation/context and coordinates aliases.
        let memory = Arc::new(
            unsafe { orbite::cuda::CudaMemory::from_raw_parts(address, length, owner) }
                .map_err(transfer_error)?,
        );
        if !matches!(memory.location(0, length).map_err(transfer_error)?.location,
            orbite::MemoryLocation::Cuda { device: actual, .. } if actual == device)
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
            inner: RegistrationInner::Orbite(RegistrationSet {
                transport: self,
                regions,
                cleanup_failed: false,
            }),
        })
    }

    /// # Safety
    /// Every operation using this registration has physically drained; errors require retaining the registration.
    pub unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            // SAFETY: the same caller ownership and drain contract applies to the fault control.
            return unsafe { control.unregister(registration) };
        }

        match &mut registration.inner {
            RegistrationInner::Orbite(registration)
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

    pub fn open_peer(
        &self,
        endpoint: &str,
        expected_epoch: &str,
        deadline: Instant,
    ) -> Result<PeerSegment, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.open_peer(endpoint, expected_epoch, deadline);
        }

        let endpoint = parse_endpoint(endpoint)
            .map_err(|e| TransferError::new(ErrorKind::InvalidRequest, e))?;
        let epoch = expected_epoch.parse().map_err(transfer_error)?;
        let peer = self
            .engine
            .open_segment(endpoint, epoch, remaining(deadline)?)
            .map_err(transfer_error)?;
        Ok(PeerSegment {
            owner_epoch: self.epoch.clone(),
            inner: Segment::Orbite(peer),
        })
    }

    /// # Safety
    /// Retain registered local ranges and remote grants, with exclusive READ destinations and immutable WRITE sources, through physical drain.
    pub unsafe fn transfer(
        &self,
        operation: crate::transfer_engine::TransferOp,
        peer: &PeerSegment,
        slices: &[TransferSlice],
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            // SAFETY: the same caller ownership and drain contract applies to the fault control.
            return unsafe { control.transfer(operation, peer, slices, options, cancellation) };
        }

        validate_profile(slices)?;
        if peer.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign peer segment",
            ));
        }
        #[cfg_attr(
            not(test),
            allow(
                clippy::infallible_destructuring_match,
                reason = "single-provider builds have one segment variant"
            )
        )]
        let peer = match &peer.inner {
            Segment::Orbite(peer) => peer,
            #[cfg(test)]
            _ => {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "foreign peer protocol",
                ));
            }
        };
        let bytes = self.transfer_target(
            operation,
            peer.into(),
            self.payload,
            slices,
            options,
            cancellation,
        )?;
        self.record_nic_load();
        Ok(bytes)
    }

    pub fn open_file(&self, path: &std::path::Path) -> Result<FileSegment, TransferError> {
        if !self.files_enabled {
            return Err(TransferError::new(
                ErrorKind::Unavailable,
                "OrbiTE file admission is disabled",
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
                inner,
            }),
            Err(error) => {
                let _ = self.engine.close_file_segment(&inner, Duration::ZERO);
                Err(transfer_error(error))
            }
        }
    }

    pub fn close_file(&self, file: &FileSegment, deadline: Instant) -> Result<(), TransferError> {
        if file.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        let inner = &file.inner;
        self.engine
            .close_file_segment(inner, deadline.saturating_duration_since(Instant::now()))
            .map_err(transfer_error)
    }

    #[allow(
        clippy::mem_forget,
        reason = "file backing, registrations and extent leases survive unknown drain or cleanup"
    )]
    /// # Safety
    /// The transfer retains the actual backing and immutable or exclusively held extents through physical drain; never replay a failed WRITE.
    pub unsafe fn transfer_file(
        self: Arc<Self>,
        operation: crate::transfer_engine::TransferOp,
        file: &FileSegment,
        transport: FileTransport,
        transfer: Arc<FileTransfer>,
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let deadline = options.deadline;
        if file.owner_epoch != self.epoch {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        let inner = &file.inner;
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
                    orbite::FileIoElement::Range { address, length }
                }
                FileIoElement::Padding(bytes) => orbite::FileIoElement::Padding(bytes),
            })
            .collect();
        let total = transfer.extent.bytes()?;
        let request = request_options(
            TransferRequest::file_vectored(
                match transport {
                    FileTransport::Auto => Transport::Unspec,
                    FileTransport::IoUring => Transport::IoUring,
                    FileTransport::Gds => Transport::Gds,
                },
                transfer.extent.offset,
                &elements,
            )
            .map_err(transfer_error)?,
            options,
        );
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
            if let Err(error) = unsafe { TransferEngine::unregister(self.as_ref(), registration) } {
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

    pub fn shutdown(&self, deadline: Instant) -> Result<(), TransferError> {
        if self.drained.load(Ordering::Acquire) {
            return Ok(());
        }
        self.engine
            .close(deadline.saturating_duration_since(Instant::now()))
            .map_err(transfer_error)?;
        self.drained.store(true, Ordering::Release);
        Ok(())
    }
    pub fn retire_peer(
        &self,
        endpoint: &str,
        epoch: &str,
        timeout: Duration,
        _requesters_drained: bool,
    ) -> Result<(), TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.retire_peer(endpoint, epoch, timeout, _requesters_drained);
        }

        // A successful whole-engine close also drains these handles. Its
        // RequesterClosed fence alone, or a failed shutdown, supplies no receipt.
        if self.drained.load(Ordering::Acquire) {
            return Ok(());
        }
        let endpoint = parse_endpoint(endpoint)
            .map_err(|error| TransferError::new(ErrorKind::InvalidRequest, error))?;
        let epoch = epoch.parse().map_err(transfer_error)?;
        self.engine
            .close_segment(endpoint, epoch, timeout)
            .map_err(transfer_error)
    }
    pub fn record_nic_load(&self) {
        #[cfg(feature = "orbite-rdma")]
        if let Ok(stats) = self.engine.nic_load_stats() {
            static METRICS: std::sync::OnceLock<crate::transfer_engine::backend::NicMetrics> =
                std::sync::OnceLock::new();
            let metrics = METRICS.get_or_init(|| {
                crate::transfer_engine::backend::NicMetrics::new("orbite", "OrbiTE")
            });
            for stat in stats {
                let attributes = [opentelemetry::KeyValue::new("nic", stat.device_name)];
                metrics
                    .inflight
                    .record(stat.inflight_bytes as u64, &attributes);
                metrics
                    .bandwidth
                    .record(stat.ewma_bandwidth_bps, &attributes);
            }
        }
    }
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
        Error::Unavailable(_)
        | Error::SourceClosed
        | Error::RequesterClosed
        | Error::PeerClosed => ErrorKind::Unavailable,
        Error::InvalidRange
        | Error::ForeignRegistration
        | Error::RegionNotFound
        | Error::RegistrationConflict
        | Error::Protocol(_) => ErrorKind::InvalidRequest,
        Error::InvalidConfig(_)
        | Error::Config { .. }
        | Error::InvalidQosConfig(_)
        | Error::InvalidSelectorConfig(_)
        | Error::TransportPolicy { .. }
        | Error::BatchClosed
        | Error::BatchOpen
        | Error::InvalidTask
        | Error::Backend { .. }
        | Error::FileSegmentClosed
        | Error::ForeignQosPolicy
        | Error::Internal(_)
        | Error::Io(_) => ErrorKind::Backend,
        // A future native failure may carry uncertain cleanup semantics.
        _ => ErrorKind::Unquiesced,
    };
    TransferError::new(kind, error.to_string())
}
fn transfer_error(error: Error) -> TransferError {
    transfer_error_ref(&error)
}

impl Drop for TransferEngine {
    #[allow(
        clippy::mem_forget,
        reason = "a failed drain must retain the engine and its process admission until exit"
    )]
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let drained = self.shutdown(deadline);
        if let Err(error) = drained {
            log::error!(
                "Retaining OrbiTE engine and {} source pool budget bytes after drain failed: {error}",
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
#[path = "../../tests/unit/peer/orbite.rs"]
mod tests;
