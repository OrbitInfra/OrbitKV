//! GPU<->CPU KV-cache transfer backends.
//!
//! A transfer is a batch of independent contiguous copies, each described by a
//! [`CopyDesc`] (`device` address, `host` address, `size`). The same batch can
//! be moved by two backends with different tradeoffs:
//!
//! - [`MemcpyBackend`]: coalesces contiguous ranges and equal-width strided rows
//!   into directional 1D or 2D DMA submissions. Drives the copy engines
//!   (DMA); best for few/large transfers and overlaps with compute.
//! - [`KernelBackend`]: a single grid-strided kernel reads/writes mapped pinned
//!   host memory directly (zero-copy over PCIe). One launch regardless of the
//!   number of fragments; best when the batch is highly fragmented and the
//!   per-call launch latency of the memcpy path dominates.
//!
//! Both backends only *enqueue* onto the caller's stream and never synchronize,
//! so a caller can batch many layers and synchronize exactly once.

mod kernel;
pub(crate) mod layout;
pub mod local;
mod memcpy;
pub(crate) mod worker;

pub use kernel::KernelBackend;
pub use memcpy::MemcpyBackend;

use std::sync::Arc;

use cudarc::driver::CudaStream;

/// One contiguous copy between device memory and mapped, pinned host memory.
///
/// `device` is an absolute device virtual address. `host` is the host virtual
/// address used by CUDA memcpy. `host_device` is the device-visible address for
/// the same mapped pinned memory, queried at allocation time; registered and
/// write-combined host memory must not assume host and device pointer values are
/// identical. The direction is fixed by which [`TransferBackend`] method is
/// called, not stored.
#[derive(Clone, Copy)]
pub struct CopyDesc {
    pub device: u64,
    pub host: *mut u8,
    pub host_device: u64,
    pub size: usize,
    pub device_allocation: usize,
    /// Physical CUDA host registration, distinct from each retained suballocation.
    pub host_registration: usize,
}

// SAFETY: `host` points into pinned memory owned by the caller, who guarantees
// it outlives the transfer. The pointer is only dereferenced by CUDA, never by
// the Rust side, so moving the descriptor across threads is sound.
unsafe impl Send for CopyDesc {}

/// Enqueue a batch of copies onto a CUDA stream. Implementations must only
/// submit work; synchronization is the caller's responsibility.
pub trait TransferBackend: Send {
    /// Host -> device.
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String>;
    /// Device -> host.
    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String>;
    /// Short name for logging (matches the CLI value that selects it).
    fn name(&self) -> &'static str;
}

/// Which [`TransferBackend`] an instance's GPU worker pools use. The connector
/// picks this per model (kernel for MLA, direct otherwise) and sends it with
/// the registration; the pools are spawned per (instance, GPU), so each
/// instance can run a different backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferMode {
    /// Directional DMA over contiguous ranges or explicit constant-pitch rows.
    /// The default backend uses the GPU copy engines.
    #[default]
    Direct,
    /// A single grid-strided copy kernel that reads/writes mapped pinned host
    /// memory directly. Wins when the batch is so fragmented that per-call
    /// launch latency on the direct path dominates.
    Kernel,
}

pub(crate) fn retain_gpu_work() -> ! {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr().lock(),
        "GPU transfer has no completion proof; retaining device and payload owners"
    );
    loop {
        std::thread::park_timeout(std::time::Duration::from_secs(60));
    }
}

pub(super) fn contain_gpu_unwind<R>(operation: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(_) => retain_gpu_work(),
    }
}

pub(crate) fn finish_gpu_transfer(
    stream: &Arc<CudaStream>,
    submitted: Result<(), String>,
) -> Result<(), crate::EngineError> {
    // A backend can fail after enqueueing part of a batch. Keep both host
    // allocations and GPU mappings alive until even that partial work drains.
    if let Err(error) = stream.synchronize() {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr().lock(),
            "GPU synchronization failed: {error}"
        );
        retain_gpu_work();
    }
    submitted.map_err(crate::EngineError::Storage)
}

fn raw_copy_shape(
    copies: &[CopyDesc],
    max_pitch: usize,
) -> orbitkv_common::timeline::RawCopyTimelineFields {
    let mut fields = orbitkv_common::timeline::RawCopyTimelineFields {
        descriptors: copies.len() as u64,
        dma_ranges: memcpy::dma_copies(copies, max_pitch).count() as u64,
        min_bytes: copies.first().map_or(0, |copy| copy.size as u64),
        ..Default::default()
    };
    const LIMITS: [usize; 7] = [256, 4096, 16384, 65536, 262144, 1048576, 4194304];
    for copy in copies {
        let size = copy.size as u64;
        fields.bytes = fields.bytes.saturating_add(size);
        fields.zero_descriptors += u64::from(size == 0);
        fields.min_bytes = fields.min_bytes.min(size);
        fields.max_bytes = fields.max_bytes.max(size);
        let bucket = LIMITS.partition_point(|limit| copy.size > *limit);
        fields.size_histogram[bucket] += 1;
    }
    fields
}

/// Manager raw batches keep their original enqueue and physical drain boundary.
pub(crate) fn copy_and_drain(
    backend: &dyn TransferBackend,
    stream: &Arc<CudaStream>,
    copies: &[CopyDesc],
    to_device: bool,
    max_pitch: usize,
) -> Result<(), crate::EngineError> {
    let diagnostic = orbitkv_common::timeline::diagnostic_enabled() && !copies.is_empty();
    let started = diagnostic.then(std::time::Instant::now);
    let submitted = contain_gpu_unwind(|| {
        if to_device {
            backend.h2d(copies, stream)
        } else {
            backend.d2h(copies, stream)
        }
    });
    let cpu_submit_ns = started.map(|started| started.elapsed().as_nanos() as u64);
    let drain_started = diagnostic.then(std::time::Instant::now);
    let result = finish_gpu_transfer(stream, submitted);
    if let (Some(cpu_submit_ns), Some(drain_started)) = (cpu_submit_ns, drain_started) {
        let host_drain_ns = drain_started.elapsed().as_nanos() as u64;
        let mut fields = raw_copy_shape(copies, max_pitch);
        fields.device = stream.context().ordinal() as u64;
        fields.backend = backend.name();
        fields.direction = if to_device { "h2d" } else { "d2h" };
        fields.cpu_submit_ns = cpu_submit_ns;
        fields.host_drain_ns = host_drain_ns;
        fields.success = result.is_ok();
        orbitkv_common::timeline::record_raw_copy(fields);
    }
    result
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/mod.rs"]
mod tests;
