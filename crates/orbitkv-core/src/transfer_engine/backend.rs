//! Required OrbiTE engine and native handles; RDMA and GDS remain opt-in.
use super::*;
use crate::TransportHint;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};
pub use crate::peer::orbite::TransferEngine;
use std::sync::Mutex;
pub(crate) const MAX_REGISTERED_POOLS: usize = 1024;
pub(crate) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
pub(crate) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

pub struct PeerSegment {
    pub(crate) owner_epoch: String,
    pub(crate) inner: Segment,
}
pub(crate) enum Segment {
    #[cfg(test)]
    Control,
    Orbite(orbite::RemoteSegment),
}
#[cfg_attr(test, derive(Clone))]
pub struct FileSegment {
    pub(crate) owner_epoch: String,
    pub length: u64,
    pub(crate) inner: orbite::FileSegment,
}
/// Failed cleanup retains registration, engine, backing and admission.
pub struct Registration {
    pub(crate) inner: RegistrationInner,
}
pub(crate) enum RegistrationInner {
    #[cfg(test)]
    #[allow(
        dead_code,
        reason = "fault control retains the actual borrowed owner until registration drop"
    )]
    Control(Arc<dyn Send + Sync>),
    Orbite(crate::peer::orbite::RegistrationSet),
}
impl TransportHint {
    pub fn validate_compiled(self) -> Result<(), TransferError> {
        let (compiled, feature) = match self {
            Self::Auto | Self::Tcp => return Ok(()),
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
        self.transfer_transport.validate_compiled()?;
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
) -> Result<Arc<TransferEngine>, TransferError> {
    hint.validate_compiled()?;
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
    .map(Arc::new)
    .map_err(|error| TransferError::new(ErrorKind::Backend, error))
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
        self.watermarks()?;
        self.shared_partition()?;
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
