//! OrbiTE construction and native handles; absent features never load an engine.
use super::*;
use crate::TransportHint;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};
#[cfg(feature = "orbite")]
pub use crate::peer::orbite::TransferEngine;
#[cfg(not(feature = "orbite"))]
use std::path::Path;
#[cfg(feature = "orbite")]
use std::sync::Mutex;
#[cfg(feature = "orbite")]
pub(crate) const MAX_REGISTERED_POOLS: usize = 1024;
#[cfg(feature = "orbite")]
pub(crate) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
#[cfg(feature = "orbite")]
pub(crate) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

pub struct PeerSegment {
    #[cfg(any(feature = "orbite", test))]
    pub(crate) owner_epoch: String,
    #[cfg(any(feature = "orbite", test))]
    pub(crate) inner: Segment,
}
#[cfg(any(feature = "orbite", test))]
pub(crate) enum Segment {
    #[cfg(test)]
    Control,
    #[cfg(feature = "orbite")]
    Orbite(orbite::RemoteSegment),
}
#[cfg_attr(test, derive(Clone))]
pub struct FileSegment {
    #[cfg(feature = "orbite")]
    pub(crate) owner_epoch: String,
    pub length: u64,
    #[cfg(feature = "orbite")]
    pub(crate) inner: orbite::FileSegment,
}
/// Failed cleanup retains registration, engine, backing and admission.
pub struct Registration {
    #[cfg(any(feature = "orbite", test))]
    pub(crate) inner: RegistrationInner,
}
#[cfg(any(feature = "orbite", test))]
pub(crate) enum RegistrationInner {
    #[cfg(test)]
    #[allow(
        dead_code,
        reason = "fault control retains the actual borrowed owner until registration drop"
    )]
    Control(Arc<dyn Send + Sync>),
    #[cfg(feature = "orbite")]
    Orbite(crate::peer::orbite::RegistrationSet),
}
impl TransportHint {
    pub fn validate_compiled(self) -> Result<(), TransferError> {
        let (compiled, feature) = match self {
            Self::Auto | Self::Tcp => (cfg!(feature = "orbite"), "orbite"),
            Self::Rdma => (cfg!(feature = "orbite-rdma"), "orbite-rdma"),
        };
        if compiled {
            Ok(())
        } else {
            Err(TransferError::new(
                ErrorKind::Unavailable,
                format!("transfer transport requires the {feature} build feature"),
            ))
        }
    }
}
impl crate::EngineConfig {
    pub(crate) fn validate_backend(&self) -> Result<(), TransferError> {
        if self.global_index.is_some()
            || self.ssd_cache_config.is_some()
            || self.transfer_transport != TransportHint::Auto
        {
            self.transfer_transport.validate_compiled()?;
        }
        if self.orbite_rdma.is_some() {
            TransportHint::Rdma.validate_compiled()?;
        }
        Ok(())
    }
}
pub(crate) fn create_transfer_engine(
    hint: TransportHint,
    allocator: Arc<PinnedAllocator>,
    advertise: &str,
    storage: Option<&crate::SsdCacheConfig>,
    rdma: Option<crate::OrbiteRdmaConfig>,
) -> Result<Option<Arc<TransferEngine>>, TransferError> {
    if hint != TransportHint::Auto {
        hint.validate_compiled()?;
    }
    if rdma.is_some() {
        TransportHint::Rdma.validate_compiled()?;
    }
    if (hint == TransportHint::Rdma && rdma.is_none())
        || (hint == TransportHint::Tcp && rdma.is_some())
    {
        return Err(TransferError::new(
            ErrorKind::InvalidRequest,
            "RDMA requires a device configuration and auto or rdma transport",
        ));
    }
    #[cfg(feature = "orbite")]
    {
        hint.validate_compiled()?;
        TransferEngine::new(
            allocator.usage().1,
            advertise,
            match hint {
                TransportHint::Auto => orbite::Transport::Unspec,
                TransportHint::Tcp => orbite::Transport::Tcp,
                TransportHint::Rdma => orbite::Transport::Rdma,
            },
            storage,
            #[cfg(feature = "orbite-rdma")]
            rdma,
        )
        .map(|engine| Some(Arc::new(engine)))
        .map_err(|error| TransferError::new(ErrorKind::Backend, error))
    }
    #[cfg(not(feature = "orbite"))]
    {
        let _ = (allocator, advertise);
        if storage.is_some() || hint != TransportHint::Auto {
            hint.validate_compiled()?;
        }
        Ok(None)
    }
}
#[cfg(feature = "orbite-rdma")]
pub use orbite::transport::rdma::{
    RdmaConfig as OrbiteRdmaConfig, RdmaCudaMemory as OrbiteRdmaCudaMemory,
};
#[cfg(not(feature = "orbite-rdma"))]
pub use unavailable_rdma::{OrbiteRdmaConfig, OrbiteRdmaCudaMemory};
#[cfg(not(feature = "orbite-rdma"))]
mod unavailable_rdma {
    #[derive(Clone, Debug)]
    pub enum OrbiteRdmaCudaMemory {
        Direct,
        Staged,
    }
    #[derive(Clone, Debug)]
    pub struct OrbiteRdmaConfig {
        pub device: String,
        pub port: Option<u8>,
        pub gid_index: Option<u8>,
        pub cuda_memory: OrbiteRdmaCudaMemory,
    }
    impl OrbiteRdmaConfig {
        pub fn new(device: impl Into<String>) -> Self {
            Self {
                device: device.into(),
                port: None,
                gid_index: None,
                cuda_memory: OrbiteRdmaCudaMemory::Staged,
            }
        }
    }
}

impl crate::SsdCacheConfig {
    pub fn validate_transfer_engine(&self) -> Result<(), String> {
        TransportHint::Auto
            .validate_compiled()
            .map_err(|e| e.to_string())?;
        if self.backend == crate::SsdBackend::Cufile && !cfg!(feature = "orbite-gds") {
            return Err("OrbiTE cuFile requires the orbite-gds build feature".into());
        }
        if self.persistent && self.backend == crate::SsdBackend::Cufile {
            return Err("persistent SSD requires io_uring (uring or auto); GDS process-exit lifetime is not admitted".into());
        }
        Ok(())
    }
}
/// Actual host backing; imported client mappings use their grant's lifetime.
#[derive(Clone)]
pub enum LocalHost {
    Pinned(Arc<PinnedAllocation>),
    #[cfg(feature = "orbite")]
    Imported(Arc<orbite::HostMemory>),
}

#[cfg(feature = "orbite-rdma")]
pub(crate) struct NicMetrics {
    pub inflight: opentelemetry::metrics::Histogram<u64>,
    pub bandwidth: opentelemetry::metrics::Histogram<f64>,
}
#[cfg(feature = "orbite-rdma")]
impl NicMetrics {
    pub(crate) fn new(name: &str, display: &str) -> Self {
        let meter = opentelemetry::global::meter("orbitkv-core");
        Self {
            inflight: meter
                .u64_histogram(format!("orbitkv_{name}_nic_inflight_bytes"))
                .with_unit("bytes")
                .with_description(format!(
                    "{display} live RDMA rail inflight-byte snapshot after a peer batch"
                ))
                .build(),
            bandwidth: meter
                .f64_histogram(format!("orbitkv_{name}_nic_bandwidth_bytes_per_second"))
                .with_unit("By/s")
                .with_description(format!(
                    "{display} RDMA rail EWMA bandwidth snapshot after a peer batch"
                ))
                .build(),
        }
    }
}

/// An unavailable build has no native handle and cannot submit byte movement.
#[cfg(not(feature = "orbite"))]
pub struct TransferEngine {
    #[cfg(test)]
    pub(crate) control: Option<Arc<crate::peer::restore::tests::Control>>,
}
#[cfg(not(feature = "orbite"))]
impl TransferEngine {
    pub fn bind_local(
        self: Arc<Self>,
        _stream: Arc<cudarc::driver::CudaStream>,
        _mode: crate::transfer::TransferMode,
        _device: u64,
    ) -> Result<Box<dyn local::LocalCopy>, TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn transfer_endpoint(&self) -> &str {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.transfer_endpoint();
        }
        ""
    }
    pub fn source_epoch(&self) -> &str {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.source_epoch();
        }
        ""
    }
    pub fn accepts_endpoint(&self, endpoint: &str) -> bool {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.accepts_endpoint(endpoint);
        }
        let _ = (endpoint,);
        false
    }
    pub fn open_file(&self, _path: &Path) -> Result<FileSegment, TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn close_file(&self, _file: &FileSegment, _deadline: Instant) -> Result<(), TransferError> {
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    /// # Safety
    /// The transfer retains the actual backing and immutable or exclusively held extents through physical drain; never replay a failed WRITE.
    pub unsafe fn transfer_file(
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
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn register_pinned(
        self: Arc<Self>,
        allocations: &[Arc<PinnedAllocation>],
        permission: Permission,
    ) -> Result<Registration, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return Arc::clone(control).register_pinned(allocations, permission);
        }
        let _ = (allocations, permission);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    /// # Safety
    /// The owner retains the CUDA allocation and context; coordinate producers and aliases through physical drain and unregister.
    pub unsafe fn register_cuda(
        self: Arc<Self>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
        permission: Permission,
    ) -> Result<Registration, TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            // SAFETY: the same caller ownership and drain contract applies to the fault control.
            return unsafe {
                Arc::clone(control).register_cuda(address, length, device, owner, permission)
            };
        }
        let _ = (address, length, device, owner, permission);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    /// # Safety
    /// Every operation using this registration has physically drained; errors require retaining the registration.
    pub unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            // SAFETY: the same caller ownership and drain contract applies to the fault control.
            return unsafe { control.unregister(registration) };
        }
        let _ = (registration,);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
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
        let _ = (endpoint, expected_epoch, deadline);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    /// # Safety
    /// Retain registered local ranges and remote grants, with exclusive READ destinations and immutable WRITE sources, through physical drain.
    pub unsafe fn transfer(
        &self,
        operation: TransferOp,
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
        let _ = (operation, peer, slices, options, cancellation);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn shutdown(&self, deadline: Instant) -> Result<(), TransferError> {
        let _ = (deadline,);
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn retire_peer(
        &self,
        _endpoint: &str,
        _epoch: &str,
        _timeout: Duration,
        _requesters_drained: bool,
    ) -> Result<(), TransferError> {
        #[cfg(test)]
        if let Some(control) = &self.control {
            return control.retire_peer(_endpoint, _epoch, _timeout, _requesters_drained);
        }
        Err(TransferError::new(
            ErrorKind::Unavailable,
            "OrbiTE requires the orbite build feature",
        ))
    }
    pub fn record_nic_load(&self) {}
}
