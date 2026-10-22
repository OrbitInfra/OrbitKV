use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use log::{error, info};
use orbitkv_transfer::{AUTO_MEMORY_LOCATION, MemoryRegistration, P2P_METADATA, TransferEngine};

use crate::memory::pool::PinnedAllocator;

use super::provider::{MAX_REGISTERED_BYTES, MAX_REGISTERED_POOLS, REGISTERED_POOLS};
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
}

impl MooncakeTransport {
    pub(crate) fn engine(&self) -> &TransferEngine {
        &self.engine
    }

    pub(crate) fn transfer_endpoint(&self) -> &str {
        &self.transfer_endpoint
    }

    pub(crate) fn new(
        nic_names: &[String],
        allocator: Arc<PinnedAllocator>,
        advertise_addr: &str,
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

#[cfg(test)]
#[path = "../../tests/unit/peer/transport.rs"]
mod tests;
