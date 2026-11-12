//! OrbiTE owns DMA coalescing, kernel dispatch, events and physical byte loans.
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::CudaStream;
use orbite::cuda::local_copy::{
    CopyDirection, LocalCopyBackend, LocalCopyConfig, LocalCopyDesc, LocalCopyEngine,
    LocalCopyOptions,
};

use super::{RegistrationSet, TransferEngine, transfer_error};
use crate::transfer::{CopyDesc, TransferMode};
use crate::transfer_engine::local::{LocalCompletion, LocalCopy, LocalHost};
use crate::transfer_engine::{ErrorKind, TransferError};

const DESCRIPTORS: usize = 1024;
const BYTES: usize = 256 << 20;

pub(super) struct Copies {
    transport: Arc<TransferEngine>,
    mode: TransferMode,
    bound: LocalCopyEngine,
}
impl Copies {
    pub(super) fn new(
        transport: Arc<TransferEngine>,
        stream: Arc<CudaStream>,
        mode: TransferMode,
        device: u64,
    ) -> Result<Self, TransferError> {
        // Anchor the stream in a persistent tensor, never a resizable codec arena.
        let (device, _) = transport
            .engine
            .resolve_local_address(device, 1)
            .map_err(transfer_error)?;
        let mut config = LocalCopyConfig::default();
        config.backend = match mode {
            TransferMode::Direct => LocalCopyBackend::Dma,
            TransferMode::Kernel => LocalCopyBackend::Kernel,
        };
        config.max_descriptors = DESCRIPTORS;
        config.max_inflight = 8;
        config.max_bytes = BYTES;
        // SAFETY: registered tensor retains its exact context. OrbiTE retains the
        // caller stream and anchor through every completion and failed cleanup.
        let bound = unsafe {
            LocalCopyEngine::bind(&device, stream.cu_stream().cast(), stream.clone(), config)
        }
        .map_err(transfer_error)?;
        Ok(Self {
            transport,
            mode,
            bound,
        })
    }

    fn window(
        &self,
        copies: &[CopyDesc],
        hosts: &std::collections::BTreeMap<u64, &LocalHost>,
        to_device: bool,
    ) -> Result<Completion, TransferError> {
        let mut ranges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for copy in copies {
            let (_, host) = hosts
                .range(..=copy.host as u64)
                .next_back()
                .ok_or_else(|| {
                    TransferError::new(ErrorKind::InvalidRequest, "missing pinned copy owner")
                })?;
            let memory = match host {
                // SAFETY: submit's caller coordinates all aliases until drain.
                LocalHost::Pinned(allocation) => {
                    // SAFETY: the retained allocation backs this entire host range; its canonical
                    // OrbiTE alias shares the access lock and the completion retains the loan.
                    unsafe { allocation.orbite_memory() }.map_err(transfer_error)?
                }
                LocalHost::Imported(memory) => (*memory).clone(),
            };
            if (copy.host as u64)
                .checked_add(copy.size as u64)
                .is_none_or(|end| end > memory.base_address() + memory.len() as u64)
            {
                return Err(TransferError::new(
                    ErrorKind::InvalidRequest,
                    "local copy exceeds its pinned slot",
                ));
            }
            if seen.insert(Arc::as_ptr(&memory) as usize) {
                let length = memory.len();
                ranges.push((memory, 0, length));
            }
        }
        let regions = self
            .transport
            .engine
            .register_host_memory_batch(&ranges, orbite::Permission::LocalOnly)
            .map_err(transfer_error)?;
        let mut completion = Completion {
            native: None,
            registration: Some(RegistrationSet {
                transport: self.transport.clone(),
                regions,
                cleanup_failed: false,
            }),
            drained: false,
        };
        let descs = copies
            .iter()
            .map(|copy| {
                let (device, device_offset) = self
                    .transport
                    .engine
                    .resolve_local_address(copy.device, copy.size)
                    .map_err(transfer_error)?;
                let (host, host_offset) = self
                    .transport
                    .engine
                    .resolve_local_address(copy.host as u64, copy.size)
                    .map_err(transfer_error)?;
                Ok(LocalCopyDesc::new(
                    &device,
                    device_offset,
                    &host,
                    host_offset,
                    copy.size,
                ))
            })
            .collect::<Result<Vec<_>, TransferError>>()?;
        // Producer compute is already ordered on this stream; framework readiness
        // is fenced before claiming Restore grants. The engine compiles 1D/2D rows.
        let bytes = copies.iter().map(|copy| copy.size).sum::<usize>();
        completion.native = Some(crate::transfer::contain_gpu_unwind(|| {
            let result = (|| {
                loop {
                    let usage = self.bound.progress().map_err(transfer_error)?;
                    if usage.active < 8
                        && bytes <= BYTES - usage.descriptor_bytes - usage.payload_bytes
                    {
                        break;
                    }
                    // Admission backpressure only: no submitted copy is replayed.
                    std::thread::sleep(Duration::from_micros(50));
                }
                self.bound
                    .submit(
                        if to_device {
                            CopyDirection::HostToDevice
                        } else {
                            CopyDirection::DeviceToHost
                        },
                        &descs,
                        LocalCopyOptions::default(),
                    )
                    .map_err(transfer_error)
            })();
            match result {
                Err(error) if error.kind == ErrorKind::Unquiesced => {
                    crate::transfer::retain_gpu_work()
                }
                result => result,
            }
        })?);
        Ok(completion)
    }
}
impl LocalCopy for Copies {
    unsafe fn submit(
        &self,
        copies: &[CopyDesc],
        hosts: &[LocalHost],
        to_device: bool,
    ) -> Result<Box<dyn LocalCompletion>, TransferError> {
        let hosts = hosts
            .iter()
            .map(|host| {
                (
                    match host {
                        LocalHost::Pinned(owner) => owner.as_non_null().as_ptr() as u64,
                        LocalHost::Imported(owner) => owner.base_address(),
                    },
                    host,
                )
            })
            .collect();
        let mut window = Vec::with_capacity(DESCRIPTORS.min(copies.len()));
        let mut bytes = 0;
        let mut completion = Completion {
            native: None,
            registration: None,
            drained: false,
        };
        for copy in copies {
            let mut offset = 0;
            while offset < copy.size {
                // Reserve half the byte budget for another admitted SSD job and
                // engine descriptor storage. Every prefix drains before reuse.
                let size = (copy.size - offset).min(BYTES / 2);
                if window.len() == DESCRIPTORS || bytes + size > BYTES / 2 {
                    crate::transfer_engine::local::drain(&mut completion)?;
                    completion = self.window(&window, &hosts, to_device)?;
                    window.clear();
                    bytes = 0;
                }
                window.push(CopyDesc {
                    device: copy
                        .device
                        .checked_add(offset as u64)
                        .ok_or_else(invalid_range)?,
                    host: (copy.host as usize)
                        .checked_add(offset)
                        .ok_or_else(invalid_range)? as *mut u8,
                    host_device: copy
                        .host_device
                        .checked_add(offset as u64)
                        .ok_or_else(invalid_range)?,
                    size,
                    ..*copy
                });
                offset += size;
                bytes += size;
            }
        }
        crate::transfer_engine::local::drain(&mut completion)?;
        if !window.is_empty() {
            completion = self.window(&window, &hosts, to_device)?;
        }
        Ok(Box::new(completion))
    }
    fn name(&self) -> &'static str {
        match self.mode {
            TransferMode::Direct => "direct",
            TransferMode::Kernel => "kernel",
        }
    }
    fn close(&self) -> Result<(), TransferError> {
        self.bound
            .close(Duration::from_secs(30))
            .map_err(|error| TransferError::new(ErrorKind::Unquiesced, error.to_string()))
    }
}
fn invalid_range() -> TransferError {
    TransferError::new(ErrorKind::InvalidRequest, "local copy address overflow")
}

struct Completion {
    native: Option<orbite::cuda::local_copy::LocalCopyCompletion>,
    registration: Option<RegistrationSet>,
    drained: bool,
}
impl LocalCompletion for Completion {
    fn poll(&mut self) -> Result<bool, TransferError> {
        let result = self
            .native
            .as_ref()
            .map_or(Ok(true), |native| native.poll().map_err(transfer_error));
        if matches!(&result, Ok(true))
            || result
                .as_ref()
                .is_err_and(|e| e.kind != ErrorKind::Unquiesced)
        {
            self.drained = true;
            if let Some(mut registration) = self.registration.take() {
                registration.unregister()?;
            }
        }
        result
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if !self.drained {
            let _ = crate::transfer_engine::local::drain(self);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/peer/orbite_local.rs"]
mod tests;
