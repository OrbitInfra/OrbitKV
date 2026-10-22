mod cache;
mod check_cuda_version;
mod cluster;
mod endpoint;
mod peer;
pub use peer::P2pTransferService;
pub mod http_server;
pub mod metric;
pub mod proto;
pub mod registry;
#[cfg(feature = "tracing")]
mod trace;
#[cfg(not(feature = "tracing"))]
mod trace {
    pub(crate) fn init() {}
    pub(crate) fn flush() {}
}
mod utils;
mod wire;

pub use registry::{CudaTensorRegistry, RegistryHandle};

use clap::Parser;
use cudarc::driver::result as cuda_driver;
use log::{error, info, warn};
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use orbitkv_common::grpc::{
    GRPC_SERVER_HTTP2_KEEPALIVE_INTERVAL, GRPC_SERVER_HTTP2_KEEPALIVE_TIMEOUT,
};
use orbitkv_core::OrbitKVEngine;
use prometheus::Registry;
use proto::engine::engine_server::EngineServer;
use proto::engine::inventory_server::InventoryServer;
use pyo3::{PyErr, Python, types::PyAnyMethods};
use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tonic::transport::Server;
use utils::parse_memory_size;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(
    name = "orbitkv-cache-manager",
    bin_name = "orbitkv-cache-manager",
    version,
    about = "OrbitKV local cache with optional distributed control"
)]
pub struct Cli {
    /// Peer control address in distributed mode; its port also names the default local socket.
    #[arg(long, default_value = "127.0.0.1:50055")]
    pub addr: SocketAddr,

    /// Membership endpoint advertised to peers when it differs from --addr.
    #[arg(long)]
    pub peer_advertise_addr: Option<SocketAddr>,

    /// Enable registered TP query control for trusted processes over loopback.
    #[arg(long, default_value_t = false)]
    pub enable_query_control: bool,

    /// CUDA devices to initialize (comma-separated, e.g., "0,1,2,3").
    /// If not specified, auto-detects and initializes all available GPUs.
    #[arg(long, value_delimiter = ',')]
    pub devices: Vec<i32>,

    /// Pinned memory pool size (supports units: kb, mb, gb, tb)
    /// Examples: "10gb", "500mb", "1tb"
    #[arg(long, default_value = "30gb", value_parser = parse_memory_size)]
    pub pool_size: usize,

    /// Hint for typical value size (supports units: kb, mb, gb, tb); tunes cache + allocator
    #[arg(long, value_parser = parse_memory_size)]
    pub hint_value_size: Option<usize>,

    /// Bytes owned by queries through preparation, ready leases and GPU loads.
    /// Defaults to 75% of --pool-size.
    #[arg(long, value_parser = parse_memory_size)]
    pub query_budget: Option<usize>,

    /// Query bytes per registered instance; defaults to the global query budget.
    #[arg(long, value_parser = parse_memory_size)]
    pub query_instance_budget: Option<usize>,

    /// Maximum bytes in each ordinary read batch (0 keeps the demand baseline).
    /// Prepared reads always use at most 32 MiB, or one oversized page.
    #[arg(long, default_value = "0", value_parser = parse_memory_size)]
    pub query_read_batch: usize,

    /// Stop submitting new ordinary reads after this relative deadline (0 disables).
    #[arg(long, default_value_t = 0)]
    pub query_read_timeout_ms: u64,

    /// Maximum read batches per ordinary query (0 unlimited, 1 best effort).
    #[arg(long, default_value_t = 0)]
    pub query_read_max_batches: usize,

    /// Use huge pages for pinned memory pool (faster allocation).
    /// Requires pre-configured huge pages via /proc/sys/vm/nr_hugepages
    #[arg(long, default_value_t = false)]
    pub use_hugepages: bool,

    /// Enable TinyLFU admission policy for cache (default: plain LRU)
    #[arg(long, default_value_t = false)]
    pub enable_lfu_admission: bool,

    /// Maximum host-cache percent protected after demand reuse (0 disables segmented LRU).
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=100))]
    pub cache_protected_percent: u8,

    /// Disable NUMA-aware memory allocation (use single pool instead of per-node pools)
    #[arg(long, default_value_t = false)]
    pub disable_numa_affinity: bool,

    /// HTTP server address for health check and Prometheus metrics.
    /// Always enabled for health check endpoint.
    #[arg(long, default_value = "0.0.0.0:9091")]
    pub http_addr: SocketAddr,

    /// Enable Prometheus /metrics endpoint on the HTTP server.
    #[arg(long, default_value_t = true)]
    pub enable_prometheus: bool,

    /// Enable OTLP metrics export over gRPC (e.g. http://127.0.0.1:4317).
    #[arg(long)]
    pub metrics_otel_endpoint: Option<String>,

    /// Period (seconds) for exporting OTLP metrics (only used when endpoint is set).
    #[arg(long, default_value_t = 10)]
    pub metrics_period_secs: u64,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable SSD cache for sealed blocks. Provide one or more cache directories.
    /// Repeat the flag to use multiple paths (e.g. one per SSD device).
    #[arg(long, num_args = 1..)]
    pub ssd_cache_path: Vec<String>,

    /// SSD cache capacity (supports units: kb, mb, gb, tb). Default: 512gb
    #[arg(long, default_value = "512gb", value_parser = parse_memory_size)]
    pub ssd_cache_capacity: usize,

    /// SSD cache file shards per path. Use >1 for parallel filesystems such as GPFS.
    /// When multiple --ssd-cache-path values are given each path receives this many
    /// shards so that every device is utilised. Default: 1
    #[arg(long, default_value = "1")]
    pub ssd_cache_shards: std::num::NonZeroUsize,

    /// SSD write queue depth (max pending write batches). Default: 8
    #[arg(long, default_value_t = orbitkv_core::DEFAULT_SSD_WRITE_QUEUE_DEPTH)]
    pub ssd_write_queue_depth: usize,

    /// SSD write admission: all publications, or reuse observed by demand/republication.
    #[arg(long, default_value = "all")]
    pub ssd_write_policy: orbitkv_core::SsdWritePolicy,

    /// SSD I/O: auto tries native cuFile and falls back to uring; explicit cufile follows NVIDIA configuration.
    /// Hardware path selection does not establish a performance advantage; qualify native GDS separately.
    #[arg(long, default_value = "auto", requires = "ssd_cache_path")]
    pub ssd_backend: orbitkv_core::SsdBackend,

    /// Pin SSD demand data and use this read route independently of write/backend selection.
    /// An unavailable explicit route returns a cache miss; it never silently switches routes.
    #[arg(long, requires = "ssd_cache_path")]
    pub ssd_read_path: Option<orbitkv_core::SsdReadPath>,

    /// GPU storage representation: none, ans, fp8, turboquant-4 or turboquant-3.
    #[arg(long, default_value = "none")]
    pub storage_codec: orbitkv_core::StorageCodec,

    /// Maximum GPU codec scratch per transfer worker (4 KiB to 4 GiB - 1).
    #[arg(long, default_value = "64mb", value_parser = parse_memory_size)]
    pub storage_codec_budget: usize,

    /// SSD prefetch queue depth (max pending prefetch batches). Default: 2
    #[arg(long, default_value_t = orbitkv_core::DEFAULT_SSD_PREFETCH_QUEUE_DEPTH)]
    pub ssd_prefetch_queue_depth: usize,

    /// SSD write inflight (max concurrent block writes). Default: 2
    #[arg(long, default_value_t = orbitkv_core::DEFAULT_SSD_WRITE_INFLIGHT)]
    pub ssd_write_inflight: usize,

    /// SSD prefetch inflight (max concurrent block reads). Default: 16
    #[arg(long, default_value_t = orbitkv_core::DEFAULT_SSD_PREFETCH_INFLIGHT)]
    pub ssd_prefetch_inflight: usize,

    /// Trace sampling rate (0.0–1.0). E.g. 0.01 = 1%. Default: 1.0 (100%)
    #[arg(long, default_value_t = 1.0, value_parser = parse_sample_rate)]
    pub trace_sample_rate: f64,

    /// Number of shards for the pinned memory pool (reduces allocator lock contention).
    /// The pool is split into this many independent sub-pools with round-robin allocation.
    #[arg(long, default_value_t = 1)]
    pub pool_shards: usize,

    /// Allocate each block separately in DRAM-only mode (always enabled with SSD).
    /// Lets evicted pages return memory independently of surviving batch pages.
    #[arg(long, default_value_t = false)]
    pub blockwise_alloc: bool,

    /// Peer payload provider; OrbitTE modes require their corresponding build feature.
    #[arg(long, default_value = "mooncake")]
    pub peer_transfer_backend: orbitkv_core::PeerTransferBackend,

    /// Explicit single RNIC for the experimental OrbitTE RDMA staging profile.
    #[cfg(feature = "orbitte-rdma")]
    #[arg(long, value_parser = parse_nic_name, required_if_eq("peer_transfer_backend", "orbitte-rdma"))]
    pub orbitte_rdma_device: Option<String>,

    /// RNIC port for OrbitTE RDMA; select from the actual device topology.
    #[cfg(feature = "orbitte-rdma")]
    #[arg(long, default_value_t = 1, requires = "orbitte_rdma_device", value_parser = clap::value_parser!(u8).range(1..))]
    pub orbitte_rdma_port: u8,

    /// Routable GID index for OrbitTE RDMA; no automatic GID selection.
    #[cfg(feature = "orbitte-rdma")]
    #[arg(long, default_value_t = 0, requires = "orbitte_rdma_device")]
    pub orbitte_rdma_gid_index: u8,

    /// Optional Mooncake RDMA rail filter (e.g. --nics mlx5_0,mlx5_1).
    /// Without it, Mooncake selects the available transport, including TCP.
    #[arg(long, value_delimiter = ',', value_parser = parse_nic_name, num_args = 1..)]
    pub nics: Option<Vec<String>>,

    /// etcd endpoints for membership and background block-location synchronization.
    #[arg(long, value_delimiter = ',', requires = "node_id")]
    pub etcd_endpoints: Vec<String>,

    /// Maximum accounted bytes in this Manager's complete global index.
    #[arg(long, default_value = "256mb", value_parser = parse_memory_size)]
    pub index_budget: usize,

    /// Stable, unique identity of this Manager across process restarts.
    #[arg(long, requires = "etcd_endpoints", value_parser = cluster::parse_label)]
    pub node_id: Option<String>,

    /// Isolates member registrations and persistent node epochs in etcd.
    #[arg(long, default_value = "orbitkv", value_parser = cluster::parse_label)]
    pub cluster_name: String,

    /// Lease TTL; new remote operations stop after half the acknowledged TTL.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(i64).range(12..=3600))]
    pub membership_ttl_secs: i64,

    /// Retained residency journal bytes; overflow triggers an inventory snapshot.
    #[arg(long, default_value_t = orbitkv_core::DEFAULT_INVENTORY_JOURNAL_BYTES)]
    pub inventory_journal_bytes: usize,

    /// Quiet window for bounded inventory-stream delta coalescing (0 disables).
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u64).range(0..=5))]
    pub inventory_stream_coalesce_ms: u64,

    /// Exact storage namespace to subscribe to; repeat for an explicit allowlist.
    #[arg(
        long,
        action = clap::ArgAction::Append,
        requires = "etcd_endpoints",
        conflicts_with = "metadata_empty_scope",
        value_parser = parse_metadata_namespace
    )]
    pub metadata_namespace: Vec<String>,

    /// Subscribe to no remote namespaces while retaining local cache service.
    #[arg(
        long,
        default_value_t = false,
        requires = "etcd_endpoints",
        conflicts_with = "metadata_namespace"
    )]
    pub metadata_empty_scope: bool,

    /// HLL sliding-window list for hit-rate estimation. Comma-separated humantime
    /// durations; each becomes a canonical `window` label in metrics (e.g. `15m,1h,1d`).
    /// Slot duration is derived as `clamp(window/24, 1min, 1h)`.
    #[arg(long, default_value = "15m,1h,1d", value_parser = parse_hll_windows_arg)]
    pub metric_hll_windows: String,

    /// HLL bucket index bits 4–18 (default: 16 → 65536 buckets, ~0.4% error)
    #[arg(long, default_value_t = 16, value_parser = parse_hll_bucket_bits)]
    pub metric_hll_bucket_bits: u8,

    /// Mark source transfers overdue after this many seconds; timeout never releases pins.
    #[arg(long, default_value_t = 120)]
    pub transfer_lock_timeout_secs: u64,

    /// Source transfer allocation budget, including overdue sessions. Defaults to half the pool.
    #[arg(long, value_parser = parse_memory_size)]
    pub transfer_budget: Option<usize>,

    /// iceoryx2 service name prefix for the inference control path.
    /// Each startup appends a unique incarnation, advertised through UDS.
    /// Defaults to a name derived from --addr.
    #[arg(long)]
    pub channel_service: Option<String>,

    /// Explicit Cache Manager session epoch for stale-client fencing. A random epoch
    /// is generated when omitted.
    #[arg(long)]
    pub channel_session_epoch: Option<u64>,

    /// Unix socket used to bootstrap inference clients and pass descriptor arena FDs.
    /// Defaults to /tmp/orbitkv-<addr-port>.sock.
    #[arg(long)]
    pub bootstrap_socket: Option<std::path::PathBuf>,

    /// Shared descriptor arena size for inference clients.
    #[arg(long, default_value = "8mb", value_parser = parse_memory_size)]
    pub descriptor_arena_size: usize,

    /// Per-client descriptor slot capacity.
    #[arg(long, default_value = "64kb", value_parser = parse_memory_size)]
    pub descriptor_slot_size: usize,
}

impl Cli {
    fn query_control_address(&self) -> Result<Option<SocketAddr>, String> {
        if !self.enable_query_control {
            return Ok(None);
        }
        let advertised = self.peer_advertise_addr.unwrap_or(self.addr);
        if !self.addr.ip().is_loopback()
            || self.addr.port() == 0
            || !advertised.ip().is_loopback()
            || advertised.port() == 0
        {
            return Err(
                "--enable-query-control requires loopback --addr and --peer-advertise-addr with nonzero ports; query control is restricted to trusted same-host processes"
                    .into(),
            );
        }
        Ok(Some(advertised))
    }
}

fn parse_hll_bucket_bits(s: &str) -> Result<u8, String> {
    use crate::metric::hll::{MAX_BUCKET_BITS, MIN_BUCKET_BITS};
    let v: u8 = s.parse().map_err(|e| format!("{e}"))?;
    if !(MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&v) {
        return Err(format!(
            "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {v}"
        ));
    }
    Ok(v)
}

fn parse_nic_name(s: &str) -> Result<String, String> {
    let name = s.trim();
    if name.is_empty() {
        return Err("--nics contains an empty NIC name".into());
    }
    Ok(name.to_string())
}

fn parse_metadata_namespace(value: &str) -> Result<String, String> {
    if orbitkv_state::is_storage_namespace(value) {
        Ok(value.to_string())
    } else {
        Err("--metadata-namespace requires an exact orbitkv:v2 storage namespace".into())
    }
}

fn parse_hll_windows_arg(s: &str) -> Result<String, String> {
    parse_hll_windows(s)?;
    Ok(s.to_string())
}

/// Parse a comma-separated list of humantime windows (e.g. `15m,1h,1d`).
/// Each entry becomes `(label, duration)` where label is canonicalized from
/// the parsed duration.
fn parse_hll_windows(s: &str) -> Result<Vec<(String, Duration)>, String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (idx, token) in s.split(',').map(str::trim).enumerate() {
        if token.is_empty() {
            return Err(format!(
                "--metric-hll-windows contains an empty window at position {}",
                idx + 1
            ));
        }
        let dur = parse_humantime(token)?;
        if dur < Duration::from_secs(60) {
            return Err(format!("HLL window {token} must be at least 1 minute"));
        }
        if !seen.insert(dur) {
            return Err(format!(
                "duplicate HLL window duration: {token} ({})",
                format_hll_window_label(dur)
            ));
        }
        out.push((format_hll_window_label(dur), dur));
    }
    if out.is_empty() {
        return Err("--metric-hll-windows must list at least one window".into());
    }
    Ok(out)
}

/// Minimal humantime parser: `<number><unit>` where unit ∈ {s, m, h, d}.
fn parse_humantime(s: &str) -> Result<Duration, String> {
    let (num_str, unit) = s.split_at(
        s.find(|c: char| !c.is_ascii_digit())
            .ok_or_else(|| format!("missing time unit in {s}"))?,
    );
    let n: u64 = num_str
        .parse()
        .map_err(|e| format!("invalid number in {s}: {e}"))?;
    let unit_secs = match unit {
        "s" => n,
        "m" => n
            .checked_mul(60)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        "h" => n
            .checked_mul(3600)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        "d" => n
            .checked_mul(86400)
            .ok_or_else(|| format!("duration overflows seconds in {s}"))?,
        _ => return Err(format!("unknown time unit {unit:?} in {s}; use s/m/h/d")),
    };
    Ok(Duration::from_secs(unit_secs))
}

fn format_hll_window_label(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs.is_multiple_of(86400) {
        format!("{}d", secs / 86400)
    } else if secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

fn parse_sample_rate(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|e| format!("{e}"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("sample rate must be between 0.0 and 1.0, got {v}"));
    }
    Ok(v)
}

fn format_py_err(err: PyErr) -> String {
    Python::attach(|py| err.value(py).to_string())
}

fn init_cuda_driver() -> Result<(), std::io::Error> {
    cuda_driver::init()
        .map_err(|err| std::io::Error::other(format!("failed to initialize CUDA driver: {err}")))
}

fn detect_cuda_devices() -> Result<Vec<i32>, std::io::Error> {
    Python::attach(|py| -> pyo3::PyResult<Vec<i32>> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        let device_count: i32 = cuda.call_method0("device_count")?.extract()?;

        // Probe each device ID from 0 to device_count-1 to see if it's available
        let mut available_devices = Vec::new();
        for device_id in 0..device_count {
            // Try to get device properties to verify it's accessible
            match cuda.call_method1("get_device_properties", (device_id,)) {
                Ok(_) => available_devices.push(device_id),
                Err(_) => continue, // Skip unavailable devices
            }
        }
        Ok(available_devices)
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to detect CUDA devices: {}",
            format_py_err(err)
        ))
    })
}

fn init_python_cuda(device_ids: &[i32]) -> Result<(), std::io::Error> {
    if device_ids.is_empty() {
        return Err(std::io::Error::other("no CUDA devices to initialize"));
    }

    Python::attach(|py| -> pyo3::PyResult<()> {
        let torch = py.import("torch")?;
        let cuda = torch.getattr("cuda")?;
        cuda.call_method0("init")?;

        // Initialize CUDA context for each device by performing a real CUDA operation
        // PyTorch uses lazy initialization, so we need to actually allocate something
        // to force context creation on each device
        for &device_id in device_ids {
            let start = std::time::Instant::now();
            cuda.call_method1("set_device", (device_id,))?;

            // Allocate a small tensor to force CUDA context creation on this device
            // This ensures the CUDA driver creates a context for the device
            let device_str = format!("cuda:{}", device_id);
            let empty_args = (vec![1i64],);
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("device", device_str)?;
            let _ = torch.call_method("empty", empty_args, Some(&kwargs))?;

            // Synchronize to ensure context is fully initialized
            cuda.call_method0("synchronize")?;

            let elapsed = start.elapsed();
            log::info!(
                "Initialized CUDA context for device {} in {:.2}s",
                device_id,
                elapsed.as_secs_f64()
            );
        }

        // Set the first device as the default
        cuda.call_method1("set_device", (device_ids[0],))?;
        Ok(())
    })
    .map_err(|err| {
        std::io::Error::other(format!(
            "failed to initialize python/tensor CUDA runtime: {}",
            format_py_err(err)
        ))
    })
}

struct MetricsState {
    meter_provider: Option<SdkMeterProvider>,
    prometheus_registry: Option<Registry>,
}

fn init_metrics(
    prometheus_enabled: bool,
    otlp_endpoint: Option<String>,
    otlp_period_secs: u64,
) -> Result<MetricsState, Box<dyn Error>> {
    let otlp_endpoint = otlp_endpoint.filter(|s| !s.is_empty());

    // If neither Prometheus nor OTLP is enabled, return empty state
    if !prometheus_enabled && otlp_endpoint.is_none() {
        info!("Metrics disabled (no Prometheus addr or OTLP endpoint configured)");
        return Ok(MetricsState {
            meter_provider: None,
            prometheus_registry: None,
        });
    }

    let mut builder = SdkMeterProvider::builder();
    let mut prometheus_registry = None;

    // Add Prometheus exporter if enabled
    if prometheus_enabled {
        let registry = Registry::new();
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()?;
        builder = builder.with_reader(exporter);
        prometheus_registry = Some(registry);
        info!("Prometheus metrics exporter enabled");
    }

    // Add OTLP exporter if endpoint is configured
    if let Some(endpoint) = otlp_endpoint {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?;

        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(otlp_period_secs))
            .build();

        builder = builder.with_reader(reader);
        info!(
            "OTLP metrics exporter enabled (period={}s)",
            otlp_period_secs
        );
    }

    let meter_provider = builder.build();
    global::set_meter_provider(meter_provider.clone());

    Ok(MetricsState {
        meter_provider: Some(meter_provider),
        prometheus_registry,
    })
}

/// Main entry point for the Cache Manager.
pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let query_control_address = cli.query_control_address()?;
    orbitkv_common::logging::init_stdout_colored(&cli.log_level);
    info!(
        "Starting orbitkv-cache-manager v{}",
        env!("CARGO_PKG_VERSION")
    );
    trace::init();
    orbitkv_core::set_trace_sample_rate(cli.trace_sample_rate);

    // Initialize CUDA in the main thread before starting Tokio runtime
    init_cuda_driver()?;
    check_cuda_version::preflight()?;

    // Determine which devices to initialize
    let devices = if cli.devices.is_empty() {
        // Auto-detect all available devices
        let detected = detect_cuda_devices()?;
        info!(
            "Auto-detected {} CUDA device(s): {:?}",
            detected.len(),
            detected
        );
        detected
    } else {
        info!("Using specified CUDA device(s): {:?}", cli.devices);
        cli.devices.clone()
    };

    if devices.is_empty() {
        return Err("No CUDA devices available".into());
    }

    init_python_cuda(&devices)?;
    info!(
        "CUDA runtime initialized for {} device(s): {:?}",
        devices.len(),
        devices
    );

    let registry = CudaTensorRegistry::new().map_err(|err| {
        let msg = format_py_err(err);
        std::io::Error::other(format!("failed to initialize torch CUDA context: {msg}"))
    })?;
    // Confine the registry to its own thread: GIL + CUDA work now happens off
    // the async runtime, so a wedged `empty_cache` can't starve tokio workers.
    let registry = RegistryHandle::spawn(registry);

    if let Some(hint_value_size) = cli.hint_value_size {
        if hint_value_size == 0 {
            return Err("--hint-value-size must be greater than zero when set".into());
        }
        info!("Value size hint set to {} bytes", hint_value_size);
    }

    info!(
        "Creating OrbitKVEngine with pinned memory pool: {:.2} GiB ({} bytes), hugepages={}",
        cli.pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.pool_size,
        cli.use_hugepages
    );

    let ssd_cache_config = if cli.ssd_cache_path.is_empty() {
        None
    } else {
        info!(
            "SSD cache enabled: paths={}, capacity={:.2} GiB, shards={}, write_queue={}, prefetch_queue={}, write_inflight={}, prefetch_inflight={}",
            cli.ssd_cache_path.join(", "),
            cli.ssd_cache_capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            cli.ssd_cache_shards,
            cli.ssd_write_queue_depth,
            cli.ssd_prefetch_queue_depth,
            cli.ssd_write_inflight,
            cli.ssd_prefetch_inflight,
        );
        Some(orbitkv_core::SsdCacheConfig {
            cache_paths: cli.ssd_cache_path.iter().map(|p| p.into()).collect(),
            capacity_bytes: cli.ssd_cache_capacity as u64,
            shards: cli.ssd_cache_shards,
            write_queue_depth: cli.ssd_write_queue_depth,
            write_policy: cli.ssd_write_policy,
            backend: cli.ssd_backend,
            read_path: cli.ssd_read_path,
            prefetch_queue_depth: cli.ssd_prefetch_queue_depth,
            write_inflight: cli.ssd_write_inflight,
            prefetch_inflight: cli.ssd_prefetch_inflight,
        })
    };

    let peer_control_enabled = !cli.etcd_endpoints.is_empty();
    if cli.nics.as_ref().is_some_and(|nics| !nics.is_empty()) && !peer_control_enabled {
        log::warn!("--nics has no effect without distributed cache configuration");
    }
    let metadata_scope = Arc::new(if cli.metadata_empty_scope {
        orbitkv_state::InventoryScope::exact(Vec::new())?
    } else if cli.metadata_namespace.is_empty() {
        orbitkv_state::InventoryScope::AllNamespaces
    } else {
        orbitkv_state::InventoryScope::exact(cli.metadata_namespace.clone())?
    });
    let membership_view = if peer_control_enabled {
        let peer_addr = cli.peer_advertise_addr.unwrap_or(cli.addr);
        if peer_addr.ip().is_unspecified() || peer_addr.port() == 0 {
            return Err("distributed peer advertise address must be concrete and routable".into());
        }
        if cli.index_budget < orbitkv_state::INVENTORY_BATCH_BYTES * 2 {
            return Err("--index-budget must hold at least two inventory batches".into());
        }
        Some(Arc::new(orbitkv_catalog::MembershipView::new(
            orbitkv_state::CacheOwner {
                endpoint: peer_addr.to_string(),
                incarnation: uuid::Uuid::new_v4(),
            },
        )))
    } else {
        None
    };
    let global_index = membership_view.as_ref().map(|view| {
        Arc::new(orbitkv_catalog::GlobalIndex::new(
            view.clone(),
            cli.index_budget,
            Arc::clone(&metadata_scope),
        ))
    });
    let inventory = membership_view
        .as_ref()
        .map(|_| {
            orbitkv_core::ResidencyInventory::with_publish_coalescing(
                cli.inventory_journal_bytes,
                Duration::from_millis(cli.inventory_stream_coalesce_ms),
            )
            .map(Arc::new)
        })
        .transpose()?;
    let storage_config = orbitkv_core::EngineConfig {
        query_budget_bytes: cli.query_budget,
        query_instance_budget_bytes: cli.query_instance_budget,
        enable_lfu_admission: cli.enable_lfu_admission,
        cache_protected_percent: cli.cache_protected_percent,
        hint_value_size_bytes: cli.hint_value_size,
        ssd_cache_config,
        codec: cli.storage_codec,
        codec_budget: cli.storage_codec_budget,
        mooncake_nic_names: cli.nics.clone().unwrap_or_default(),
        peer_transfer_backend: cli.peer_transfer_backend,
        #[cfg(feature = "orbitte-rdma")]
        orbitte_rdma: cli.orbitte_rdma_device.clone().map(|device| {
            orbitkv_core::OrbitTeRdmaConfig {
                port: cli.orbitte_rdma_port,
                gid_index: cli.orbitte_rdma_gid_index,
                ..orbitkv_core::OrbitTeRdmaConfig::new(device)
            }
        }),
        enable_numa_affinity: !cli.disable_numa_affinity,
        blockwise_alloc: cli.blockwise_alloc,
        transfer_lock_timeout: Duration::from_secs(cli.transfer_lock_timeout_secs),
        transfer_budget_bytes: cli.transfer_budget,
        membership: membership_view.clone(),
        global_index: global_index.clone(),
        inventory: inventory.clone(),
        pool_shards: cli.pool_shards,
    };

    if cli.pool_shards > 1 {
        info!(
            "Pinned memory pool sharding enabled: {} shards",
            cli.pool_shards
        );
    }
    if cli.enable_lfu_admission {
        info!("TinyLFU cache admission enabled");
    }
    if cli.disable_numa_affinity {
        info!("NUMA-aware memory allocation disabled");
    }

    // Create Tokio runtime early - needed for OTLP metrics gRPC exporter
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Initialize OTEL metrics BEFORE creating OrbitKVEngine, so that core metrics
    // (pool, cache, save/load) use the real meter provider instead of noop.
    let metrics_state = runtime.block_on(async {
        init_metrics(
            cli.enable_prometheus,
            cli.metrics_otel_endpoint.clone(),
            cli.metrics_period_secs,
        )
    })?;

    let hll_tracker = Arc::new(std::sync::Mutex::new(
        crate::metric::hll::MultiWindowHllTracker::new(
            parse_hll_windows(&cli.metric_hll_windows)
                .map_err(|err| format!("invalid --metric-hll-windows: {err}"))?,
            cli.metric_hll_bucket_bits,
        ),
    ));
    crate::metric::register_hll_gauges(&hll_tracker);

    let shutdown = Arc::new(Notify::new());
    let channel_config = {
        let service_name = cli
            .channel_service
            .clone()
            .unwrap_or_else(|| format!("orbitkv/channel/{}", cli.addr.port()));
        let session_epoch = cli
            .channel_session_epoch
            .unwrap_or_else(random_nonzero_session_epoch);
        if session_epoch == 0 {
            return Err("--channel-session-epoch must be non-zero".into());
        }
        let bootstrap_socket = cli.bootstrap_socket.clone().unwrap_or_else(|| {
            std::path::PathBuf::from(format!("/tmp/orbitkv-{}.sock", cli.addr.port()))
        });
        (
            service_name,
            session_epoch,
            bootstrap_socket,
            cli.descriptor_arena_size,
            cli.descriptor_slot_size,
        )
    };
    let runtime_handle = runtime.handle().clone();
    let result = runtime.block_on(async move {
        let membership = match membership_view.clone() {
            Some(view) => Some(
                cluster::Cluster::join(
                    &cli.etcd_endpoints,
                    &cli.cluster_name,
                    cli.node_id
                        .as_deref()
                        .ok_or("--node-id is required with etcd")?,
                    cli.membership_ttl_secs,
                    view,
                    inventory.clone().ok_or("missing distributed inventory")?,
                    global_index.clone().ok_or("missing global index")?,
                )
                .await?,
            ),
            None => None,
        };
        let inventory_runtime = membership.as_ref().map(cluster::Cluster::inventory);
        // Create OrbitKVEngine inside tokio runtime context (needed for SSD cache tokio::spawn)
        let engine = Arc::new(OrbitKVEngine::new_with_config(
            cli.pool_size,
            cli.use_hugepages,
            storage_config,
        )?);
        let lifecycle = cache::lifecycle::LifecycleService::new(Arc::clone(&engine), registry);
        let (service_name, session_epoch, bootstrap_socket, arena_size, slot_size) = channel_config;
        let query_control = query_control_address.map(|address| {
            cache::query_control::QueryControlService::new(
                format!("http://{address}"),
                Arc::clone(&engine),
                Arc::clone(&hll_tracker),
                runtime_handle.clone(),
            )
        });
        let mut channel_endpoint = endpoint::ProcessEndpoint::start(
            service_name,
            session_epoch,
            bootstrap_socket,
            arena_size,
            slot_size,
            Arc::clone(&engine),
            runtime_handle,
            Arc::clone(&hll_tracker),
            Arc::clone(&shutdown),
            lifecycle.clone(),
            cli.query_read_batch as u64,
            (cli.query_read_timeout_ms != 0).then(|| Duration::from_millis(cli.query_read_timeout_ms)),
            if cli.query_read_max_batches == 0 { usize::MAX } else { cli.query_read_max_batches },
            query_control.clone(),
        )?;
        if let Some(control) = &query_control { control.start_reaper(); }

        // Spawn background GC task for stale inflight blocks and expired transfer locks
        {
            let engine = Arc::clone(&engine);
            let shutdown = Arc::clone(&shutdown);
            tokio::spawn(async move {
                const GC_INTERVAL: Duration = Duration::from_secs(60);
                const INFLIGHT_MAX_AGE: Duration = Duration::from_secs(300); // 5 min

                let mut interval = tokio::time::interval(GC_INTERVAL);
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            let cleaned = engine
                                .gc_stale_inflight(INFLIGHT_MAX_AGE)
                                .await;
                            if cleaned > 0 {
                                info!("Inflight GC: cleaned {} stale blocks", cleaned);
                            }

                            let expired_locks = engine.expire_transfer_locks();
                            if expired_locks > 0 {
                                warn!("Retaining source pins for {} overdue transfer sessions", expired_locks);
                            }
                        }
                        _ = shutdown.notified() => {
                            info!("Background GC task shutting down");
                            break;
                        }
                    }
                }
            });
            info!("Background GC task started (interval=60s, inflight_max_age=5m)");
        }

        // Start HTTP server for health check (always enabled)
        let http_server_handle = http_server::start_http_server_with_lifecycle(
            cli.http_addr,
            Arc::clone(&engine),
            lifecycle.clone(),
            cli.enable_prometheus,
            metrics_state.prometheus_registry.clone(),
            Arc::clone(&shutdown),
            inventory_runtime.clone(),
        )
        .await?;

        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let shutdown_signal = {
            let notify = Arc::clone(&shutdown);
            let membership_view = membership_view.clone();
            async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl+C received, shutting down");
                    }
                    _ = terminate.recv() => {
                        info!("SIGTERM received, shutting down");
                    }
                    _ = notify.notified() => {
                        info!("Shutdown requested via control endpoint");
                    }
                }
                if let Some(view) = membership_view {
                    view.fence();
                }
                notify.notify_waiters();
            }
        };

        if membership_view.is_some() || query_control.is_some() {
            const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
            let peer_service = membership_view.as_ref().map(|_| {
                EngineServer::new(P2pTransferService::new(Arc::clone(&engine)))
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            });
            let inventory_service = inventory_runtime.as_ref().map(|inventory| {
                InventoryServer::new(inventory.service())
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024)
            });
            let query_service = query_control.clone().map(|control| {
                proto::engine::cache_query_control_server::CacheQueryControlServer::new(control)
                    .max_decoding_message_size(128 * 1024)
                    .max_encoding_message_size(128 * 1024)
            });
            info!("Cache Manager control listening on {}", cli.addr);
            if let Err(error) = Server::builder()
                .http2_keepalive_interval(Some(GRPC_SERVER_HTTP2_KEEPALIVE_INTERVAL))
                .http2_keepalive_timeout(Some(GRPC_SERVER_HTTP2_KEEPALIVE_TIMEOUT))
                .concurrency_limit_per_connection(16)
                .add_optional_service(peer_service)
                .add_optional_service(inventory_service)
                .add_optional_service(query_service)
                .serve_with_shutdown(cli.addr, shutdown_signal)
                .await
            {
                if let Some(control) = &query_control { control.stop(); }
                error!("Server error: {error}");
                return Err(error.into());
            }
        } else {
            info!("Standalone Cache Manager ready; network control is disabled");
            shutdown_signal.await;
        }
        if let Some(control) = &query_control { control.stop(); }

        info!("Cache Manager stopped");
        channel_endpoint.stop_admission_and_drain_publishes().await;
        channel_endpoint.stop_queries_and_drain().await?;
        channel_endpoint
            .stop_lifecycle_and_drain_connections()
            .await;

        // Stop HTTP server
        shutdown.notify_waiters();
        lifecycle.shutdown().await?;
        engine.flush_all().await;
        channel_endpoint.stop();
        let _ = http_server_handle.await;

        if let Some(membership) = membership {
            membership.shutdown().await;
        }

        // Flush metrics before exit
        if let Some(provider) = metrics_state.meter_provider
            && let Err(err) = provider.shutdown()
        {
            error!("Failed to shutdown metrics provider: {err}");
        }

        trace::flush();

        Ok(())
    });
    orbitkv_common::timeline::flush_diagnostic();
    result
}

fn random_nonzero_session_epoch() -> u64 {
    loop {
        let epoch = uuid::Uuid::new_v4().as_u128() as u64;
        if epoch != 0 {
            return epoch;
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/lib.rs"]
mod tests;
