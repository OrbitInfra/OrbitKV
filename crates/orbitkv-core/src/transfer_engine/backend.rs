//! Native handle representations and provider construction.

use super::*;
use crate::PeerTransferBackend;
use crate::memory::pool::PinnedAllocator;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
use std::sync::Mutex;

#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) const MAX_REGISTERED_POOLS: usize = 1024;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

#[cfg(feature = "mooncake")]
impl Cancellation {
    pub(crate) fn flag(&self) -> &AtomicBool {
        &self.0
    }
}

pub struct PeerSegment {
    #[cfg(any(feature = "mooncake", feature = "orbite", test))]
    pub(crate) owner_epoch: String,
    #[cfg(any(feature = "mooncake", feature = "orbite", test))]
    pub(crate) inner: Segment,
}
#[cfg(any(feature = "mooncake", feature = "orbite", test))]
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

/// Engine-local file handle. The SSD owner keeps the original fd and extents.
#[cfg_attr(test, derive(Clone))]
pub struct FileSegment {
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) owner_epoch: String,
    pub length: u64,
    #[cfg(any(feature = "mooncake", feature = "orbite"))]
    pub(crate) inner: FileSegmentInner,
}

#[derive(Clone)]
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) enum FileSegmentInner {
    #[cfg(feature = "mooncake")]
    Tent(Arc<orbitkv_transfer::files::FileSegment>),
    #[cfg(feature = "orbite")]
    Orbite(orbite::FileSegment),
}

/// Retains registration, engine and actual backing. Failed cleanup retains all
/// three and their admission charge. Drop is only legal after transfer drain.
pub struct Registration {
    #[cfg(any(feature = "mooncake", feature = "orbite", test))]
    pub(crate) inner: RegistrationInner,
}
#[cfg(any(feature = "mooncake", feature = "orbite", test))]
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

impl PeerTransferBackend {
    /// Reject unavailable providers before CUDA or pinned-pool initialization.
    pub fn validate_compiled(self) -> Result<(), TransferError> {
        let (compiled, feature) = match self {
            Self::Mooncake => (cfg!(feature = "mooncake"), "mooncake"),
            Self::Orbite | Self::OrbiteTcp => (cfg!(feature = "orbite"), "orbite"),
            Self::OrbiteRdma => (cfg!(feature = "orbite-rdma"), "orbite-rdma"),
        };
        if compiled {
            Ok(())
        } else {
            Err(TransferError::new(
                ErrorKind::Unavailable,
                format!("selected peer transfer backend requires the {feature} build feature"),
            ))
        }
    }
}

impl crate::EngineConfig {
    pub(crate) fn validate_backend(&self) -> Result<(), TransferError> {
        if self.global_index.is_some()
            || self.ssd_cache_config.is_some()
            || self.peer_transfer_backend != PeerTransferBackend::Mooncake
        {
            self.peer_transfer_backend.validate_compiled()?;
        }
        if self.orbite_rdma.is_some() {
            PeerTransferBackend::OrbiteRdma.validate_compiled()?;
        }
        Ok(())
    }
}

pub(crate) fn create_transfer_backend(
    backend: PeerTransferBackend,
    nics: &[String],
    allocator: Arc<PinnedAllocator>,
    advertise: &str,
    storage: Option<&crate::SsdCacheConfig>,
    rdma: Option<crate::OrbiteRdmaConfig>,
) -> Result<Arc<dyn TransferBackend>, TransferError> {
    backend.validate_compiled()?;
    if rdma.is_some() {
        PeerTransferBackend::OrbiteRdma.validate_compiled()?;
    }
    if (backend == PeerTransferBackend::OrbiteRdma && rdma.is_none())
        || (rdma.is_some()
            && !matches!(
                backend,
                PeerTransferBackend::Orbite | PeerTransferBackend::OrbiteRdma
            ))
    {
        return Err(TransferError::new(
            ErrorKind::InvalidRequest,
            "OrbiTE RDMA requires a device configuration and orbite or orbite-rdma selection",
        ));
    }
    match backend {
        #[cfg(feature = "mooncake")]
        PeerTransferBackend::Mooncake => Ok(Arc::new(
            crate::peer::transport::MooncakeTransport::new(nics, allocator, advertise, storage)
                .map_err(|e| TransferError::new(ErrorKind::Backend, e))?,
        )),
        #[cfg(feature = "orbite")]
        PeerTransferBackend::Orbite
        | PeerTransferBackend::OrbiteTcp
        | PeerTransferBackend::OrbiteRdma => {
            if !nics.is_empty() {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "OrbiTE does not accept a Mooncake RDMA NIC filter",
                ));
            }
            Ok(Arc::new(
                crate::peer::orbite::OrbiteTransport::new(
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
                )
                .map_err(|e| TransferError::new(ErrorKind::Backend, e))?,
            ))
        }
        #[allow(
            unreachable_patterns,
            reason = "feature combinations omit unavailable constructors"
        )]
        _ => {
            let _ = (nics, allocator, advertise, storage, rdma);
            Err(TransferError::new(
                ErrorKind::Unavailable,
                "selected transfer backend was not compiled",
            ))
        }
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

use crate::{SsdBackend, SsdCacheConfig};

impl SsdCacheConfig {
    pub fn validate_transfer_engine(
        &self,
        backend: crate::PeerTransferBackend,
    ) -> Result<(), String> {
        backend.validate_compiled().map_err(|e| e.to_string())?;
        if self.backend == SsdBackend::Cufile {
            if backend == crate::PeerTransferBackend::Mooncake {
                if !cfg!(feature = "cuda-13") {
                    return Err("TENT GDS requires a CUDA-enabled native runtime".into());
                }
                if self.cache_paths.len().saturating_mul(self.shards.get()) > 128 {
                    return Err(
                        "TENT GDS allows at most 128 shards (256 IOURING/GDS file handles)".into(),
                    );
                }
            } else if !cfg!(feature = "orbite-gds") {
                return Err("OrbiTE cuFile requires the orbite-gds build feature".into());
            }
        }
        Ok(())
    }
}

use crate::memory::pool::PinnedAllocation;
/// Actual host backing; imported client mappings use their grant's lifetime.
#[derive(Clone)]
pub enum LocalHost {
    Pinned(Arc<PinnedAllocation>),
    #[cfg(feature = "orbite")]
    Imported(Arc<orbite::HostMemory>),
}

#[cfg(any(feature = "mooncake", feature = "orbite-rdma"))]
pub(crate) struct NicMetrics {
    pub inflight: opentelemetry::metrics::Histogram<u64>,
    pub bandwidth: opentelemetry::metrics::Histogram<f64>,
}
#[cfg(any(feature = "mooncake", feature = "orbite-rdma"))]
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
