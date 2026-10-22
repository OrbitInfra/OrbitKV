use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orbit_te::{
    EngineConfig, Error, Permission, RegisteredMemory, TransferEngine, TransferOp, TransferRequest,
    Transport,
};

use super::provider::{
    MAX_REGISTERED_BYTES, MAX_REGISTERED_POOLS, PeerReadSlice, PeerTransferError, REGISTERED_POOLS,
};
use crate::memory::pool::PinnedAllocation;

const PREFIX: &str = "orbitte+tcp://";
const MAX_READ_REQUESTS: usize = 16_384;
const MAX_READ_BYTES: usize = 64 << 20;
const SLICE_BYTES: usize = 8 << 20;

pub(super) fn parse_endpoint(endpoint: &str) -> Result<SocketAddr, String> {
    endpoint
        .strip_prefix(PREFIX)
        .ok_or("peer is not an OrbitTE TCP endpoint")?
        .parse()
        .map_err(|error| format!("invalid OrbitTE TCP endpoint: {error}"))
}

pub(crate) struct OrbitTeTransport {
    engine: Arc<TransferEngine>,
    pub(super) endpoint: String,
    pub(super) epoch: String,
    pool_bytes: u64,
}

/// Explicit registration references plus the actual allocation and provider
/// lifetime. Source tickets and requester READ owners hold these together.
pub(super) struct RegistrationSet {
    transport: Arc<OrbitTeTransport>,
    regions: Vec<RegisteredMemory>,
}

impl Drop for RegistrationSet {
    #[allow(
        clippy::mem_forget,
        reason = "failed unregister must retain the actual backing and charged engine admission"
    )]
    fn drop(&mut self) {
        for region in &self.regions {
            if let Err(error) = self.transport.engine.unregister_memory(region) {
                log::error!(
                    "Retaining OrbitTE registration ownership after unregister failed: {error}"
                );
                std::mem::forget((
                    Arc::clone(&self.transport),
                    std::mem::take(&mut self.regions),
                ));
                return;
            }
        }
    }
}

impl OrbitTeTransport {
    pub(super) fn new(pool_bytes: u64, advertise: &str) -> Result<Self, String> {
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
            .map_err(|_| "OrbitTE TCP requires a numeric advertised IP")?;
        if host.is_unspecified() {
            return Err("OrbitTE TCP requires a concrete advertised IP".into());
        }
        let config = EngineConfig {
            bind_addr: SocketAddr::new(host, 0),
            max_registered_regions: 4096,
            max_registered_bytes: usize::try_from(pool_bytes)
                .map_err(|_| "pool exceeds addressable bytes")?,
            max_batch_requests: MAX_READ_REQUESTS,
            ..EngineConfig::default()
        };
        let engine = Arc::new(TransferEngine::new(config).map_err(|error| error.to_string())?);
        let epoch = engine.source_epoch().to_string();
        let endpoint = format!("{PREFIX}{}", engine.endpoint());
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
        Ok(Self {
            engine,
            endpoint,
            epoch,
            pool_bytes,
        })
    }

    pub(super) fn register<'a>(
        self: &Arc<Self>,
        allocations: impl Iterator<Item = &'a Arc<PinnedAllocation>>,
        permission: Permission,
    ) -> Result<RegistrationSet, String> {
        let mut seen = HashSet::new();
        let ranges = allocations
            .filter(|allocation| seen.insert(Arc::as_ptr(allocation) as usize))
            .map(|allocation| {
                // SAFETY: sealed source blocks are immutable. Requester ranges
                // are newly allocated and handed exclusively to the READ owner.
                // All aliases use the allocation's canonical shared access lock.
                let memory =
                    unsafe { allocation.orbitte_memory() }.map_err(|error| error.to_string())?;
                let length = memory.len();
                Ok((memory, 0, length))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let regions = self
            .engine
            .register_host_memory_batch(&ranges, permission)
            .map_err(|error| error.to_string())?;
        Ok(RegistrationSet {
            transport: Arc::clone(self),
            regions,
        })
    }

    pub(super) fn read(
        self: &Arc<Self>,
        endpoint: &str,
        expected_epoch: &str,
        slices: &[PeerReadSlice],
        timeout: Duration,
    ) -> Result<usize, PeerTransferError> {
        let started = Instant::now();
        let endpoint = parse_endpoint(endpoint).map_err(PeerTransferError::Failed)?;
        let epoch = expected_epoch
            .parse()
            .map_err(|error: Error| PeerTransferError::Failed(error.to_string()))?;
        let mut requests = Vec::new();
        let mut total = 0usize;
        for slice in slices {
            total = total
                .checked_add(slice.length)
                .filter(|bytes| *bytes <= MAX_READ_BYTES)
                .ok_or_else(|| {
                    PeerTransferError::Failed(
                        "OrbitTE TCP READ exceeds the 64 MiB profile budget".into(),
                    )
                })?;
            let count = slice.length.div_ceil(SLICE_BYTES);
            if requests
                .len()
                .checked_add(count)
                .is_none_or(|count| count > MAX_READ_REQUESTS)
            {
                return Err(PeerTransferError::Failed(
                    "OrbitTE TCP READ descriptor budget exhausted".into(),
                ));
            }
            slice
                .remote_address
                .checked_add(slice.length as u64)
                .ok_or_else(|| PeerTransferError::Failed("remote READ address overflow".into()))?;
            for offset in (0..slice.length).step_by(SLICE_BYTES) {
                requests.push(TransferRequest {
                    transport: Transport::Tcp,
                    local_address: slice.local.as_ptr() as usize as u64 + offset as u64,
                    remote_address: slice.remote_address + offset as u64,
                    length: SLICE_BYTES.min(slice.length - offset),
                });
            }
        }
        let _registrations = self
            .register(
                slices.iter().map(|slice| &slice.allocation),
                Permission::LocalOnly,
            )
            .map_err(PeerTransferError::Failed)?;
        let remaining = || {
            timeout
                .checked_sub(started.elapsed())
                .filter(|value| !value.is_zero())
                .ok_or(PeerTransferError::Timeout)
        };
        // Never learn authority from a refreshed descriptor. The source grant
        // supplies the exact epoch used for both metadata open and payload READ.
        let peer = self
            .engine
            .open_segment(endpoint, epoch, remaining()?)
            .map_err(transfer_error)?;
        let report = self
            .engine
            .submit_and_wait(TransferOp::Read, &peer, &requests, remaining()?)
            .map_err(transfer_error)?;
        if !report.locally_drained {
            return Err(PeerTransferError::Unquiesced);
        }
        if let Some(failure) = report.failure {
            return Err(transfer_error(failure.error));
        }
        if report.transferred_bytes != total {
            return Err(PeerTransferError::Failed(
                "OrbitTE TCP READ returned incomplete bytes".into(),
            ));
        }
        Ok(total)
    }
}

fn transfer_error(error: Error) -> PeerTransferError {
    match error {
        Error::Timeout => PeerTransferError::Timeout,
        Error::Unquiesced { .. } => PeerTransferError::Unquiesced,
        error => PeerTransferError::Failed(error.to_string()),
    }
}

impl Drop for OrbitTeTransport {
    #[allow(
        clippy::mem_forget,
        reason = "a failed drain must retain the engine and its process admission until exit"
    )]
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let source = self
            .engine
            .close_source(deadline.saturating_duration_since(Instant::now()));
        let requester = self
            .engine
            .close_requester(deadline.saturating_duration_since(Instant::now()));
        let drained = source.and(requester);
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
