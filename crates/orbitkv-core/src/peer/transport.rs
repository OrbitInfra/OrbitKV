use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use log::{error, info};
use orbitkv_transfer::{AUTO_MEMORY_LOCATION, MemoryRegistration, P2P_METADATA, TransferEngine};

use crate::memory::pool::PinnedAllocator;
use crate::transfer_engine::{
    Cancellation, ErrorKind, FileIoElement, FileMemory, FileSegment, FileSegmentInner,
    FileTransfer, FileTransport, MAX_REGISTERED_BYTES, MAX_REGISTERED_POOLS, PeerSegment,
    Permission, REGISTERED_POOLS, Registration, RegistrationInner, Segment, TransferBackend,
    TransferError, TransferOp, TransferOptions, TransferSlice, remaining, validate_slices,
};

struct RetainedSource {
    _engine: Arc<TransferEngine>,
    _registrations: Vec<MemoryRegistration>,
    _pool: Arc<PinnedAllocator>,
}
static RETAINED_SOURCES: Mutex<Vec<RetainedSource>> = Mutex::new(Vec::new());
static SOURCE_METRICS: LazyLock<(
    opentelemetry::metrics::Counter<u64>,
    opentelemetry::metrics::Counter<u64>,
)> = LazyLock::new(|| {
    let meter = opentelemetry::global::meter("orbitkv-peer");
    let pools = meter
        .u64_counter("orbitkv_peer_retained_source_pools")
        .build();
    let bytes = meter
        .u64_counter("orbitkv_peer_retained_source_pool_bytes")
        .build();
    pools.add(0, &[]);
    bytes.add(0, &[]);
    (pools, bytes)
});

/// Mooncake TENT plus the lifetime of the registered pinned pool.
pub(crate) struct MooncakeTransport {
    engine: Arc<TransferEngine>,
    registrations: Vec<MemoryRegistration>,
    pool: Option<Arc<PinnedAllocator>>,
    registered_bytes: u64,
    transfer_endpoint: String,
    files: Option<Arc<orbitkv_transfer::files::FileEngine>>,
}

impl MooncakeTransport {
    pub(crate) fn new(
        nic_names: &[String],
        allocator: Arc<PinnedAllocator>,
        advertise_addr: &str,
        storage: Option<&crate::SsdCacheConfig>,
    ) -> Result<Self, String> {
        LazyLock::force(&SOURCE_METRICS);
        let t0 = Instant::now();
        for nic in nic_names {
            let path = Path::new("/sys/class/infiniband").join(nic);
            if !path.exists() {
                return Err(format!("configured RDMA NIC {nic:?} does not exist"));
            }
        }
        let bind_host = host_from_endpoint(advertise_addr)?;
        let local_server_name = format!("{bind_host}:0");
        let engine = Arc::new(
            TransferEngine::new(P2P_METADATA, &local_server_name, &bind_host, 0, nic_names)
                .map_err(|e| e.to_string())?,
        );
        let transfer_endpoint = engine.local_segment_name().map_err(|e| e.to_string())?;

        let regions = allocator.memory_regions();
        let registered_bytes = regions.iter().try_fold(0u64, |total, (_, len)| {
            total
                .checked_add(*len as u64)
                .ok_or("registered pool bytes overflow")
        })?;
        {
            let mut budget = REGISTERED_POOLS.lock().unwrap_or_else(|e| e.into_inner());
            let total = budget.1.checked_add(registered_bytes);
            if budget.0 >= MAX_REGISTERED_POOLS || total.is_none_or(|n| n > MAX_REGISTERED_BYTES) {
                return Err("registered source pool admission exhausted".into());
            }
            budget.0 += 1;
            budget.1 = total.ok_or("registered pool bytes overflow")?;
        }
        // Own partial registration and its backing before the first native call.
        let mut transport = Self {
            engine,
            registrations: Vec::with_capacity(regions.len()),
            pool: Some(allocator),
            registered_bytes,
            transfer_endpoint,
            files: None,
        };
        for (ptr, len) in regions {
            transport.registrations.push(
                unsafe {
                    transport
                        .engine
                        .register_memory_owned(ptr, len, AUTO_MEMORY_LOCATION)
                }
                .map_err(|e| e.to_string())?,
            );
        }

        if let Some(storage) = storage {
            transport.files = Some(
                orbitkv_transfer::files::FileEngine::new(
                    storage.backend == crate::SsdBackend::Cufile,
                )
                .map_err(|error| format!("TENT file engine unavailable: {error}"))?,
            );
        }

        info!(
            "Mooncake TENT initialised: endpoint={}, nics={}, registered {} memory region(s), elapsed={:?}",
            transport.transfer_endpoint,
            nic_names.len(),
            transport.registrations.len(),
            t0.elapsed(),
        );

        Ok(transport)
    }
}

impl Drop for MooncakeTransport {
    fn drop(&mut self) {
        // A native source receipt covers the whole irreversible engine gate.
        // Local borrowers retain the transport while requester work runs.
        let result = self
            .engine
            .close_source(Duration::from_secs(5))
            .and_then(|()| {
                for registration in &mut self.registrations {
                    registration.try_unregister()?;
                }
                Ok(())
            });
        if let Err(error) = result {
            error!(
                "Retaining source engine, registrations and {} pool bytes: {error}",
                self.registered_bytes
            );
            SOURCE_METRICS.0.add(1, &[]);
            SOURCE_METRICS.1.add(self.registered_bytes, &[]);
            // These owners and their process admission remain charged until exit.
            // No unregister retry, engine destructor or allocator release is safe.
            if let Some(pool) = self.pool.take() {
                RETAINED_SOURCES
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(RetainedSource {
                        _engine: Arc::clone(&self.engine),
                        _registrations: std::mem::take(&mut self.registrations),
                        _pool: pool,
                    });
            }
            return;
        }
        self.registrations.clear();
        let mut budget = REGISTERED_POOLS.lock().unwrap_or_else(|e| e.into_inner());
        budget.0 -= 1;
        budget.1 -= self.registered_bytes;
    }
}

fn host_from_endpoint(endpoint: &str) -> Result<String, String> {
    let endpoint = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    endpoint
        .rsplit_once(':')
        .map(|(host, _)| host.trim_matches(['[', ']']).to_string())
        .filter(|host| !host.is_empty())
        .ok_or_else(|| format!("invalid advertise address: {endpoint}"))
}

pub(crate) struct TentRegistration {
    transport: Arc<MooncakeTransport>,
    native: Option<MemoryRegistration>,
    owner: Option<Arc<dyn Send + Sync>>,
    charged_bytes: Option<u64>,
    cleanup_failed: bool,
}
impl TentRegistration {
    fn unregister(&mut self) -> Result<(), crate::transfer_engine::TransferError> {
        if self.cleanup_failed {
            return Err(TransferError::new(
                ErrorKind::Unquiesced,
                "TENT registration cleanup remains retained",
            ));
        }
        if let Some(native) = &mut self.native
            && let Err(error) = native.try_unregister()
        {
            self.cleanup_failed = true;
            return Err(tent_error(error));
        }
        self.native = None;
        self.owner = None;
        if let Some(bytes) = self.charged_bytes.take() {
            let mut budget = REGISTERED_POOLS.lock().unwrap_or_else(|e| e.into_inner());
            budget.0 -= 1;
            budget.1 -= bytes;
        }
        Ok(())
    }
}
impl Drop for TentRegistration {
    #[allow(
        clippy::mem_forget,
        reason = "failed unregister retains native token and actual backing without a retry"
    )]
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            log::error!("Retaining TENT registration and backing: {error}");
            std::mem::forget((
                Arc::clone(&self.transport),
                self.native.take(),
                self.owner.take(),
            ));
        }
    }
}

impl TransferBackend for MooncakeTransport {
    fn bind_local(
        self: Arc<Self>,
        stream: Arc<cudarc::driver::CudaStream>,
        mode: crate::transfer::TransferMode,
        _device: u64,
    ) -> Result<Box<dyn crate::transfer_engine::local::LocalCopy>, TransferError> {
        crate::transfer_engine::local::TentLocalCopy::bind(stream, mode)
    }

    fn transfer_endpoint(&self) -> &str {
        &self.transfer_endpoint
    }
    fn source_epoch(&self) -> &str {
        self.engine.source_epoch()
    }
    fn accepts_endpoint(&self, endpoint: &str) -> bool {
        !endpoint.starts_with("orbite+") && !endpoint.is_empty()
    }

    fn open_file(&self, path: &Path) -> Result<FileSegment, TransferError> {
        let files = self.files.as_ref().ok_or_else(|| {
            TransferError::new(ErrorKind::Unavailable, "TENT file admission is disabled")
        })?;
        let file = files.open(path).map_err(tent_error)?;
        Ok(FileSegment {
            owner_epoch: self.source_epoch().into(),
            length: file.length,
            inner: FileSegmentInner::Tent(file),
        })
    }

    fn close_file(&self, file: &FileSegment, deadline: Instant) -> Result<(), TransferError> {
        if file.owner_epoch != self.source_epoch() {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        #[cfg(feature = "orbite")]
        let FileSegmentInner::Tent(file) = &file.inner else {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file provider",
            ));
        };
        #[cfg(not(feature = "orbite"))]
        let FileSegmentInner::Tent(file) = &file.inner;
        file.close(deadline).map_err(tent_error)
    }

    unsafe fn transfer_file(
        self: Arc<Self>,
        operation: TransferOp,
        file: &FileSegment,
        transport: FileTransport,
        transfer: Arc<FileTransfer>,
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let deadline = options.deadline;
        if file.owner_epoch != self.source_epoch() {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file segment",
            ));
        }
        #[cfg(feature = "orbite")]
        let FileSegmentInner::Tent(file) = &file.inner else {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign file provider",
            ));
        };
        #[cfg(not(feature = "orbite"))]
        let FileSegmentInner::Tent(file) = &file.inner;
        transfer.extent.bytes()?;
        let context = match &transfer.memory {
            FileMemory::Pinned(_) => None,
            FileMemory::RegisteredCuda { context, .. } => {
                context
                    .bind_to_thread()
                    .map_err(|e| TransferError::new(ErrorKind::InvalidRequest, e.to_string()))?;
                Some(Arc::clone(context))
            }
        };
        let transport = match transport {
            FileTransport::Auto | FileTransport::IoUring => {
                orbitkv_transfer::files::FileTransport::IoUring
            }
            FileTransport::Gds => {
                orbitkv_transfer::files::FileTransport::Gds(context.ok_or_else(|| {
                    TransferError::new(
                        ErrorKind::Unavailable,
                        "TENT native GDS requires CUDA ranges",
                    )
                })?)
            }
        };
        let elements: Vec<_> = transfer
            .extent
            .elements
            .iter()
            .map(|element| match *element {
                FileIoElement::Range { address, length } => {
                    orbitkv_transfer::files::FileElement::Range { address, length }
                }
                FileIoElement::Padding(length) => {
                    orbitkv_transfer::files::FileElement::Padding(length)
                }
            })
            .collect();
        let operation = match operation {
            TransferOp::Read => orbitkv_transfer::TransferOp::Read,
            TransferOp::Write => orbitkv_transfer::TransferOp::Write,
        };
        // SAFETY: FileTransfer retains actual backing, registrations, CUDA context and
        // extent through the native physical receipt, including TENT-owned staging.
        unsafe {
            file.transfer(
                operation,
                transport,
                orbitkv_transfer::files::FileExtent {
                    offset: transfer.extent.offset,
                    elements: &elements,
                },
                transfer,
                deadline,
                cancellation.flag(),
            )
        }
        .map_err(tent_error)
    }

    fn register_pinned(
        self: Arc<Self>,
        allocations: &[Arc<crate::memory::pool::PinnedAllocation>],
        _permission: Permission,
    ) -> Result<Registration, TransferError> {
        let regions = self
            .pool
            .as_ref()
            .ok_or_else(|| TransferError::new(ErrorKind::Unavailable, "TENT pool closed"))?
            .memory_regions();
        for allocation in allocations {
            let address = allocation.as_non_null().as_ptr() as usize;
            let end = address
                .checked_add(allocation.size_bytes() as usize)
                .ok_or_else(|| {
                    TransferError::new(ErrorKind::InvalidRequest, "allocation overflow")
                })?;
            if !regions.iter().any(|(ptr, len)| {
                address >= ptr.as_ptr() as usize && end <= ptr.as_ptr() as usize + len
            }) {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "allocation is outside the registered TENT pool",
                ));
            }
        }
        Ok(Registration {
            inner: RegistrationInner::Tent(TentRegistration {
                transport: self,
                native: None,
                owner: Some(Arc::new(allocations.to_vec())),
                charged_bytes: None,
                cleanup_failed: false,
            }),
        })
    }

    #[allow(
        clippy::mem_forget,
        reason = "partial native registration errors retain backing and bounded admission"
    )]
    unsafe fn register_cuda(
        self: Arc<Self>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
        _permission: Permission,
    ) -> Result<Registration, TransferError> {
        let ptr = usize::try_from(address)
            .ok()
            .and_then(|a| std::ptr::NonNull::new(a as *mut u8))
            .filter(|_| length != 0 && address.checked_add(length as u64).is_some() && device >= 0)
            .ok_or_else(|| TransferError::new(ErrorKind::InvalidRequest, "invalid CUDA region"))?;
        {
            let mut budget = REGISTERED_POOLS.lock().unwrap_or_else(|e| e.into_inner());
            let bytes = budget
                .1
                .checked_add(length as u64)
                .filter(|b| *b <= MAX_REGISTERED_BYTES)
                .filter(|_| budget.0 < MAX_REGISTERED_POOLS)
                .ok_or_else(|| {
                    TransferError::new(
                        ErrorKind::ResourceExhausted,
                        "registered CUDA memory budget exhausted",
                    )
                })?;
            budget.0 += 1;
            budget.1 = bytes;
        }
        let mut registration = TentRegistration {
            transport: self,
            native: None,
            owner: Some(owner),
            charged_bytes: Some(length as u64),
            cleanup_failed: false,
        };
        // SAFETY: actual allocation/context ownership is installed before native registration.
        match unsafe {
            registration.transport.engine.register_memory_owned(
                ptr,
                length,
                &format!("cuda:{device}"),
            )
        } {
            Ok(native) => {
                registration.native = Some(native);
                Ok(Registration {
                    inner: RegistrationInner::Tent(registration),
                })
            }
            Err(error) => {
                std::mem::forget(registration);
                Err(tent_error(error))
            }
        }
    }

    unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError> {
        match &mut registration.inner {
            RegistrationInner::Tent(registration)
                if std::ptr::eq(registration.transport.as_ref(), self) =>
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
        if !self.accepts_endpoint(endpoint) {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "peer payload protocol mismatch",
            ));
        }
        remaining(deadline)?;
        self.engine
            .prepare_segment(endpoint, expected_epoch)
            .map_err(tent_error)?;
        remaining(deadline)?;
        Ok(PeerSegment {
            owner_epoch: self.source_epoch().to_owned(),
            inner: Segment::Tent {
                endpoint: endpoint.to_owned(),
                source_epoch: expected_epoch.to_owned(),
            },
        })
    }
    unsafe fn transfer(
        &self,
        operation: TransferOp,
        peer: &PeerSegment,
        slices: &[TransferSlice],
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        let deadline = options.deadline;
        let total = validate_slices(slices)?;
        if peer.owner_epoch != self.source_epoch() {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "foreign peer segment",
            ));
        }
        #[cfg_attr(
            not(feature = "orbite"),
            allow(
                clippy::infallible_destructuring_match,
                reason = "single-provider builds have one segment variant"
            )
        )]
        let (endpoint, source_epoch) = match &peer.inner {
            Segment::Tent {
                endpoint,
                source_epoch,
            } => (endpoint, source_epoch),
            #[cfg(any(feature = "orbite", test))]
            _ => {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "foreign peer protocol",
                ));
            }
        };
        if cancellation.is_cancelled() {
            return Err(TransferError::new(
                ErrorKind::Cancelled,
                "transfer cancelled before submission",
            ));
        }
        let native = slices
            .iter()
            .map(|s| orbitkv_transfer::TransferSlice {
                priority: match options.intent.priority() {
                    crate::transfer_engine::TransferPriority::High => {
                        orbitkv_transfer::TransferPriority::High
                    }
                    crate::transfer_engine::TransferPriority::Low => {
                        orbitkv_transfer::TransferPriority::Low
                    }
                },
                local: s.local,
                remote_address: s.remote_address,
                length: s.length,
            })
            .collect::<Vec<_>>();
        let op = match operation {
            TransferOp::Read => orbitkv_transfer::TransferOp::Read,
            TransferOp::Write => orbitkv_transfer::TransferOp::Write,
        };
        let bytes = self
            .engine
            .submit_and_wait_cancellable(
                op,
                endpoint,
                &native,
                remaining(deadline)?,
                Some(source_epoch),
                cancellation.flag(),
            )
            .map_err(tent_error)?;
        if bytes != total {
            return Err(TransferError::new(
                ErrorKind::Backend,
                "TENT returned incomplete bytes",
            ));
        }
        self.record_nic_load();
        Ok(bytes)
    }
    fn shutdown(&self, deadline: Instant) -> Result<(), TransferError> {
        if let Some(files) = &self.files {
            files.shutdown(deadline).map_err(tent_error)?;
        }
        self.engine
            .close_source(deadline.saturating_duration_since(Instant::now()))
            .map_err(|e| TransferError::new(ErrorKind::Unquiesced, e.to_string()))
    }
    fn invalidate_segment(&self, endpoint: &str) {
        if let Err(error) = self.engine.invalidate_segment(endpoint) {
            log::warn!("Retaining failed TENT peer close: {error}");
        }
    }
    fn retire_peer(
        &self,
        endpoint: &str,
        _epoch: &str,
        _timeout: std::time::Duration,
        requesters_drained: bool,
    ) -> Result<(), TransferError> {
        if !requesters_drained {
            return Err(TransferError::new(
                ErrorKind::Unquiesced,
                "TENT peer requests still own the reversible handle",
            ));
        }
        self.engine
            .invalidate_segment(endpoint)
            .map_err(|error| TransferError::new(ErrorKind::Unquiesced, error.to_string()))
    }
    fn record_nic_load(&self) {
        if let Ok(stats) = self.engine.nic_load_stats() {
            let metrics = crate::metrics::core_metrics();
            for stat in stats {
                let attributes = [opentelemetry::KeyValue::new("nic", stat.device_name)];
                metrics
                    .tent_nic_inflight_bytes
                    .record(stat.inflight_bytes, &attributes);
                metrics
                    .tent_nic_bandwidth_bytes_per_second
                    .record(stat.ewma_bandwidth_bps, &attributes);
            }
        }
    }
}
fn tent_error(error: orbitkv_transfer::TransferError) -> TransferError {
    use orbitkv_transfer::TransferError as E;
    let kind = match error {
        E::File { status, .. } => match status {
            1 => ErrorKind::InvalidRequest,
            2 | 3 => ErrorKind::Unavailable,
            6 => ErrorKind::Cancelled,
            7 => ErrorKind::Timeout,
            8 => ErrorKind::Unquiesced,
            9 => ErrorKind::ResourceExhausted,
            10 => ErrorKind::PermissionDenied,
            _ => ErrorKind::Backend,
        },
        E::Timeout => ErrorKind::Timeout,
        E::Cancelled => ErrorKind::Cancelled,
        E::BatchAdmissionExhausted | E::SegmentAdmissionExhausted => ErrorKind::ResourceExhausted,
        E::NativeRuntime(_) | E::Create => ErrorKind::Unavailable,
        E::InvalidString(_) => ErrorKind::InvalidRequest,
        // RPC errors take precedence over the operation's application error.
        // Only 134 proves the request was not sent; 135 never permits replay.
        E::Operation { status: -134, .. } => ErrorKind::Unavailable,
        E::Operation { status: -135, .. } => ErrorKind::Unquiesced,
        E::Operation { status: -2, .. } => ErrorKind::ResourceExhausted,
        E::Operation {
            operation: "validateSourceEpochBeforeSubmit",
            ..
        } => ErrorKind::EpochMismatch,
        E::Operation {
            operation: "invalidSourceEpoch",
            ..
        } => ErrorKind::InvalidRequest,
        _ => ErrorKind::Backend,
    };
    TransferError::new(kind, error.to_string())
}

#[cfg(test)]
#[path = "../../tests/unit/peer/transport.rs"]
mod tests;
