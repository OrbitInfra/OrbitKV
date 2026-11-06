//! Single-launch transfer backend.
//!
//! Instead of one `cuMemcpyAsync` per fragment, a single grid-strided kernel
//! copies the whole batch: bounded threadblocks per descriptor read from the source
//! address and writes to the destination address. For host<->device copies the
//! host side is mapped pinned memory, which the kernel dereferences directly
//! (zero-copy over PCIe). This collapses N driver submissions into one small
//! descriptor copy plus one launch, which wins when the batch is so fragmented
//! that per-call launch latency on the memcpy path dominates.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::compile_ptx;

use super::{CopyDesc, TentCopy};

/// Descriptor layout is a flat `u64` array, 3 entries per copy:
/// `[dst, src, size, dst, src, size, ...]`. A flat array sidesteps any host/
/// device struct-layout mismatch. The 16-byte vectorized path is taken only
/// when both addresses are 16-byte aligned, otherwise a scalar loop is used.
const KERNEL_SRC: &str = include_str!("kernel.cu");

const BLOCK_DIM: u32 = 256;
const MAX_GRID: u32 = 65535;

/// Single-launch transfer backend. Compiled once per worker; the descriptor
/// scratch buffer is reused across calls.
pub struct KernelBackend {
    func: CudaFunction,
    cta_budget: usize,
    /// Device-side descriptor buffer, grow-only and reused. The owning worker
    /// processes tasks serially and synchronizes after each, so the previous
    /// task's kernel has consumed the buffer before the next call overwrites it.
    scratch: RefCell<Option<CudaSlice<u64>>>,
}

impl KernelBackend {
    /// Compile and load the copy kernel into `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc compile failed: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e:?}"))?;
        let func = module
            .load_function("orbitkv_batch_copy")
            .map_err(|e| format!("load_function failed: {e:?}"))?;
        let multiprocessors = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| format!("multiprocessor count failed: {e:?}"))?;
        let cta_budget = usize::try_from(multiprocessors)
            .ok()
            .filter(|count| *count > 0)
            .ok_or("invalid multiprocessor count")?
            .saturating_mul(4);
        Ok(Self {
            func,
            cta_budget,
            scratch: RefCell::new(None),
        })
    }

    /// Enqueue the batch. `host_is_src` selects the direction: H2D reads the
    /// host address and writes the device address, D2H is reversed.
    fn submit(
        &self,
        copies: &[CopyDesc],
        host_is_src: bool,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        if copies.is_empty() {
            return Ok(());
        }

        let n = copies.len();
        let n_arg = i32::try_from(n).map_err(|_| "copy descriptor count exceeds i32")?;
        let capacity = n.checked_mul(3).ok_or("copy descriptor size overflow")?;
        let mut max_bytes = 0;
        let mut descs = Vec::with_capacity(capacity);
        for c in copies {
            let (dst, src) = if host_is_src {
                (c.device, c.host_device)
            } else {
                (c.host_device, c.device)
            };
            descs.push(dst);
            descs.push(src);
            descs.push(c.size as u64);
            max_bytes = max_bytes.max(c.size);
        }

        let mut guard = self.scratch.borrow_mut();
        let needs_realloc = guard.as_ref().is_none_or(|s| s.len() < descs.len());
        if needs_realloc {
            *guard = Some(
                stream
                    .alloc_zeros::<u64>(descs.len())
                    .map_err(|e| format!("scratch alloc failed: {e:?}"))?,
            );
        }
        let scratch = guard
            .as_mut()
            .ok_or("copy descriptor scratch is unavailable")?;

        // Async H2D of the descriptor array. The source is pageable, so the
        // driver consumes it before returning — the local `descs` is safe to
        // drop. The device buffer outlives the launch because it lives in
        // `self.scratch`, not on this stack frame.
        stream
            .memcpy_htod(&descs, scratch)
            .map_err(|e| format!("descriptor upload failed: {e:?}"))?;

        let ctas = (self.cta_budget / n)
            .clamp(1, 16)
            .min(max_bytes.div_ceil(65536).clamp(1, 16));
        let grid = (n as u64 * ctas as u64).min(u64::from(MAX_GRID)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let ctas_arg = ctas as i32;
        let mut builder = stream.launch_builder(&self.func);
        builder.arg(&*scratch).arg(&n_arg).arg(&ctas_arg);
        // SAFETY: the kernel reads exactly `n` descriptors from `scratch` (len
        // >= 3*n) and copies `size` bytes between the device and mapped pinned
        // host addresses, all kept valid by the caller until it synchronizes.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e:?}"))?;
        Ok(())
    }
}

impl TentCopy for KernelBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, true, stream)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, false, stream)
    }

    fn name(&self) -> &'static str {
        "kernel"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/kernel.rs"]
mod tests;
