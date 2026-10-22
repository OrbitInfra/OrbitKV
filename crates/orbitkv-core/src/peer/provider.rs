use std::fmt;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::PeerTransferBackend;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};

pub(super) const MAX_REGISTERED_POOLS: usize = 1024;
pub(super) const MAX_REGISTERED_BYTES: u64 = 1 << 40;
pub(super) static REGISTERED_POOLS: Mutex<(usize, u64)> = Mutex::new((0, 0));

pub(crate) enum PeerTransport {
    #[cfg(feature = "mooncake")]
    Mooncake(super::transport::MooncakeTransport),
    #[cfg(feature = "orbitte")]
    OrbitTe(Arc<super::orbitte::OrbitTeTransport>),
}

pub(super) struct PeerReadSlice {
    pub(super) local: NonNull<u8>,
    pub(super) remote_address: u64,
    pub(super) length: usize,
    pub(super) allocation: Arc<PinnedAllocation>,
}

// SAFETY: Each address is retained by its actual pinned allocation. Consumers
// validate the range and transfer exclusive destination access to the worker.
unsafe impl Send for PeerReadSlice {}

#[derive(Debug)]
pub(super) enum PeerTransferError {
    Timeout,
    Failed(String),
    /// The backend still retains resources; source grants and buffers must stay.
    #[cfg_attr(
        not(feature = "orbitte"),
        allow(
            dead_code,
            reason = "the TENT adapter drains before returning; OrbitTE can expose retained work"
        )
    )]
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

impl PeerTransport {
    pub(crate) fn new(
        backend: PeerTransferBackend,
        nics: &[String],
        allocator: Arc<PinnedAllocator>,
        advertise: &str,
        #[cfg(feature = "orbitte-rdma")] rdma: Option<crate::OrbitTeRdmaConfig>,
    ) -> Result<Self, String> {
        #[cfg(feature = "orbitte-rdma")]
        if (backend == PeerTransferBackend::OrbitTeRdma) != rdma.is_some() {
            return Err(
                "OrbitTE RDMA requires an explicit configuration and orbitte-rdma selection".into(),
            );
        }
        match backend {
            PeerTransferBackend::Mooncake => {
                #[cfg(feature = "mooncake")]
                return super::transport::MooncakeTransport::new(nics, allocator, advertise)
                    .map(Self::Mooncake);
                #[cfg(not(feature = "mooncake"))]
                Err("Mooncake peer transport requires the mooncake build feature".into())
            }
            PeerTransferBackend::OrbitTeTcp => {
                #[cfg(feature = "orbitte")]
                {
                    if !nics.is_empty() {
                        return Err("orbitte-tcp does not accept an RDMA NIC filter".into());
                    }
                    super::orbitte::OrbitTeTransport::new(
                        allocator.usage().1,
                        advertise,
                        orbit_te::Transport::Tcp,
                        #[cfg(feature = "orbitte-rdma")]
                        None,
                    )
                    .map(Arc::new)
                    .map(Self::OrbitTe)
                }
                #[cfg(not(feature = "orbitte"))]
                Err("OrbitTE TCP peer transport requires the orbitte build feature".into())
            }
            PeerTransferBackend::OrbitTeRdma => {
                #[cfg(feature = "orbitte-rdma")]
                {
                    if !nics.is_empty() {
                        return Err("orbitte-rdma uses its explicit device configuration, not a Mooncake NIC filter".into());
                    }
                    super::orbitte::OrbitTeTransport::new(
                        allocator.usage().1,
                        advertise,
                        orbit_te::Transport::Rdma,
                        rdma,
                    )
                    .map(Arc::new)
                    .map(Self::OrbitTe)
                }
                #[cfg(not(feature = "orbitte-rdma"))]
                Err("OrbitTE RDMA peer transport requires the orbitte-rdma build feature".into())
            }
        }
    }

    pub(crate) fn transfer_endpoint(&self) -> &str {
        match self {
            #[cfg(feature = "mooncake")]
            Self::Mooncake(transport) => transport.transfer_endpoint(),
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(transport) => &transport.endpoint,
        }
    }

    pub(crate) fn source_epoch(&self) -> &str {
        match self {
            #[cfg(feature = "mooncake")]
            Self::Mooncake(transport) => transport.engine().source_epoch(),
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(transport) => &transport.epoch,
        }
    }

    pub(super) fn accepts_endpoint(&self, endpoint: &str) -> bool {
        match self {
            #[cfg(feature = "mooncake")]
            Self::Mooncake(_) => !endpoint.starts_with("orbitte+") && !endpoint.is_empty(),
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(transport) => {
                super::orbitte::parse_endpoint(transport.payload, endpoint).is_ok()
            }
        }
    }

    pub(super) fn read(
        &self,
        endpoint: &str,
        slices: &[PeerReadSlice],
        timeout: Duration,
        expected_epoch: &str,
    ) -> Result<usize, PeerTransferError> {
        if !self.accepts_endpoint(endpoint) {
            return Err(PeerTransferError::Failed(
                "peer payload protocol mismatch".into(),
            ));
        }
        for slice in slices {
            slice
                .allocation
                .source_range(slice.local, slice.length)
                .map_err(PeerTransferError::Failed)?;
        }
        match self {
            #[cfg(feature = "mooncake")]
            Self::Mooncake(transport) => {
                let slices = slices
                    .iter()
                    .map(|slice| orbitkv_transfer::TransferSlice {
                        local: slice.local,
                        remote_address: slice.remote_address,
                        length: slice.length,
                    })
                    .collect::<Vec<_>>();
                transport
                    .engine()
                    .submit_and_wait(
                        orbitkv_transfer::TransferOp::Read,
                        endpoint,
                        &slices,
                        timeout,
                        Some(expected_epoch),
                    )
                    .map_err(|error| match error {
                        orbitkv_transfer::TransferError::Timeout => PeerTransferError::Timeout,
                        error => PeerTransferError::Failed(error.to_string()),
                    })
            }
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(transport) => transport.read(endpoint, expected_epoch, slices, timeout),
        }
    }

    pub(super) fn invalidate_segment(&self, _endpoint: &str) {
        #[cfg(feature = "mooncake")]
        match self {
            Self::Mooncake(transport) => transport.engine().invalidate_segment(_endpoint),
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(_) => {}
        }
        // OrbitTE opens each descriptor using the grant's epoch; no peer cache.
    }

    pub(super) fn record_nic_load(&self) {
        #[cfg(feature = "mooncake")]
        match self {
            Self::Mooncake(transport) => {
                if let Ok(stats) = transport.engine().nic_load_stats() {
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
            #[cfg(feature = "orbitte")]
            Self::OrbitTe(_) => {}
        }
    }
}
