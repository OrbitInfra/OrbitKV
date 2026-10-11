//! Compile validated DMA ranges without copying gaps or reordering operations.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, sys};

use super::{CopyDesc, TransferBackend};

mod ranges;
pub(super) use ranges::dma_copies;

/// DMA copy-engine backend. The device pitch limit is read once at binding.
pub struct MemcpyBackend {
    pub(super) max_pitch: usize,
}

impl MemcpyBackend {
    pub fn new(context: &CudaContext) -> Result<Self, String> {
        let max_pitch = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
            .map_err(|error| error.to_string())?;
        let max_pitch = usize::try_from(max_pitch)
            .ok()
            .filter(|pitch| *pitch > 0)
            .ok_or("invalid CUDA maximum DMA pitch")?;
        Ok(Self { max_pitch })
    }

    fn submit(
        &self,
        copies: &[CopyDesc],
        stream: &Arc<CudaStream>,
        to_device: bool,
    ) -> Result<(), String> {
        for copy in dma_copies(copies, self.max_pitch) {
            // SAFETY: descriptors were bounds-checked and their owners remain
            // alive through stream drain, including on a partial enqueue error.
            // Compilation copies only explicit rows within one host registration
            // and one device allocation; row gaps are never accessed.
            let result = unsafe {
                if copy.rows == 1 {
                    if to_device {
                        sys::cuMemcpyHtoDAsync_v2(
                            copy.device,
                            copy.host.cast(),
                            copy.width,
                            stream.cu_stream(),
                        )
                    } else {
                        sys::cuMemcpyDtoHAsync_v2(
                            copy.host.cast(),
                            copy.device,
                            copy.width,
                            stream.cu_stream(),
                        )
                    }
                } else {
                    let host = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
                    let device = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
                    let params = sys::CUDA_MEMCPY2D {
                        srcXInBytes: 0,
                        srcY: 0,
                        srcMemoryType: if to_device { host } else { device },
                        srcHost: copy.host.cast(),
                        srcDevice: copy.device,
                        srcArray: std::ptr::null_mut(),
                        srcPitch: if to_device {
                            copy.host_pitch
                        } else {
                            copy.device_pitch
                        },
                        dstXInBytes: 0,
                        dstY: 0,
                        dstMemoryType: if to_device { device } else { host },
                        dstHost: copy.host.cast(),
                        dstDevice: copy.device,
                        dstArray: std::ptr::null_mut(),
                        dstPitch: if to_device {
                            copy.device_pitch
                        } else {
                            copy.host_pitch
                        },
                        WidthInBytes: copy.width,
                        Height: copy.rows,
                    };
                    sys::cuMemcpy2DAsync_v2(&params, stream.cu_stream())
                }
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "DMA copy failed (to_device={to_device}, rows={}): {result:?}",
                    copy.rows
                ));
            }
        }
        Ok(())
    }
}

impl TransferBackend for MemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, stream, true)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, stream, false)
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/memcpy.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/transfer/memcpy/ranges.rs"]
mod range_tests;
