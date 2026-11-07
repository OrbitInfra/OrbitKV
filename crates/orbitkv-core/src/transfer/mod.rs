//! KV layout descriptors and OrbiTE local-copy completion ownership.

pub(crate) mod layout;
pub mod local;
pub(crate) mod worker;

use std::sync::Arc;

use cudarc::driver::CudaStream;

/// One contiguous copy between device memory and mapped, pinned host memory.
///
/// `device` is an absolute device virtual address. `host` is the host virtual
/// address used by CUDA memcpy. `host_device` is the device-visible address for
/// the same mapped pinned memory, queried at allocation time; registered and
/// write-combined host memory must not assume host and device pointer values are
/// identical. The local-copy submission supplies the direction.
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

/// Operator/layout hint consumed by OrbiTE's local CUDA copy engine.
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

pub(crate) fn contain_gpu_unwind<R>(operation: impl FnOnce() -> R) -> R {
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
    _max_pitch: usize,
) -> orbitkv_common::timeline::RawCopyTimelineFields {
    let mut fields = orbitkv_common::timeline::RawCopyTimelineFields {
        descriptors: copies.len() as u64,
        // OrbiTE owns coalescing; zero means the compiled DMA count is unavailable.
        dma_ranges: 0,
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
    backend: &dyn crate::transfer_engine::local::LocalCopy,
    stream: &Arc<CudaStream>,
    copies: &[CopyDesc],
    hosts: &[crate::transfer_engine::local::LocalHost],
    to_device: bool,
    max_pitch: usize,
) -> Result<(), crate::EngineError> {
    let diagnostic = orbitkv_common::timeline::diagnostic_enabled() && !copies.is_empty();
    let started = diagnostic.then(std::time::Instant::now);
    let submitted = contain_gpu_unwind(|| {
        // SAFETY: worker owns source grants, GPU registrations and host slots until drain.
        unsafe { backend.submit(copies, hosts, to_device) }
    });
    let cpu_submit_ns = started.map(|started| started.elapsed().as_nanos() as u64);
    let drain_started = diagnostic.then(std::time::Instant::now);
    let submitted = match submitted {
        Err(error) if error.kind == crate::transfer_engine::ErrorKind::Unquiesced => {
            retain_gpu_work()
        }
        result => result,
    };
    let result = submitted
        .and_then(|mut completion| crate::transfer_engine::local::drain(completion.as_mut()))
        .map_err(|e| crate::EngineError::Storage(e.to_string()));
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
