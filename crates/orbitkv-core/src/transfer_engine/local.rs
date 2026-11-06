//! Stream-bound local copies. A completion owns bytes until physical drain.
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::{CudaEvent, CudaStream, result, sys};

use super::{ErrorKind, TransferBackend, TransferError};
use crate::memory::pool::PinnedAllocation;
use crate::transfer::{CopyDesc, TransferMode};

/// Actual host backing; imported client mappings use their grant's lifetime.
#[derive(Clone)]
pub enum LocalHost {
    Pinned(Arc<PinnedAllocation>),
    #[cfg(feature = "orbitte")]
    Imported(Arc<orbit_te::HostMemory>),
}

pub trait LocalCompletion: Send {
    /// `Ok(true)` and non-Unquiesced errors prove physical drain.
    fn poll(&mut self) -> Result<bool, TransferError>;
    fn wait(&mut self) -> Result<(), TransferError> {
        loop {
            if self.poll()? {
                return Ok(());
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

/// One implementation per selected engine, bound once to a worker stream.
pub trait LocalCopy: Send {
    /// # Safety
    /// Coordinate all aliases and retain GPU allocations and source grants until
    /// the completion drains. Unquiesced is never a release receipt.
    unsafe fn submit(
        &self,
        copies: &[CopyDesc],
        hosts: &[LocalHost],
        to_device: bool,
    ) -> Result<Box<dyn LocalCompletion>, TransferError>;
    fn name(&self) -> &'static str;
    fn close(&self) -> Result<(), TransferError>;
}

pub(crate) fn bind(
    engine: Option<Arc<dyn TransferBackend>>,
    stream: Arc<CudaStream>,
    mode: TransferMode,
    device: u64,
) -> Result<Box<dyn LocalCopy>, TransferError> {
    match engine {
        Some(engine) => engine.bind_local(stream, mode, device),
        // Local-only TENT configurations do not start a peer/file engine.
        None => TentLocalCopy::bind(stream, mode),
    }
}

pub(crate) fn failure(error: TransferError) -> String {
    if error.kind == ErrorKind::Unquiesced {
        crate::transfer::retain_gpu_work();
    }
    error.to_string()
}

pub(crate) fn drain(completion: &mut dyn LocalCompletion) -> Result<(), TransferError> {
    match completion.wait() {
        Err(error) if error.kind == ErrorKind::Unquiesced => {
            log::error!("Local copy has no drain receipt: {error}");
            crate::transfer::retain_gpu_work();
        }
        result => result,
    }
}

pub(crate) struct TentLocalCopy {
    stream: Arc<CudaStream>,
    native: Box<dyn crate::transfer::TentCopy>,
}

impl TentLocalCopy {
    #[cfg(test)]
    pub(crate) fn control(
        stream: Arc<CudaStream>,
        native: Box<dyn crate::transfer::TentCopy>,
    ) -> Box<dyn LocalCopy> {
        Box::new(Self { stream, native })
    }

    pub(crate) fn bind(
        stream: Arc<CudaStream>,
        mode: TransferMode,
    ) -> Result<Box<dyn LocalCopy>, TransferError> {
        use crate::transfer::{KernelBackend, MemcpyBackend};
        let native: Box<dyn crate::transfer::TentCopy> = match mode {
            TransferMode::Direct => {
                Box::new(MemcpyBackend::new(stream.context()).map_err(native_error)?)
            }
            TransferMode::Kernel => {
                Box::new(KernelBackend::new(stream.context()).map_err(native_error)?)
            }
        };
        Ok(Box::new(Self { stream, native }))
    }
}

impl LocalCopy for TentLocalCopy {
    unsafe fn submit(
        &self,
        copies: &[CopyDesc],
        hosts: &[LocalHost],
        to_device: bool,
    ) -> Result<Box<dyn LocalCompletion>, TransferError> {
        let event = self
            .stream
            .context()
            .new_event(None)
            .map_err(native_error)?;
        let submitted = crate::transfer::contain_gpu_unwind(|| {
            if to_device {
                self.native.h2d(copies, &self.stream)
            } else {
                self.native.d2h(copies, &self.stream)
            }
        });
        if let Err(error) =
            submitted.and_then(|()| event.record(&self.stream).map_err(|e| e.to_string()))
        {
            crate::transfer::finish_gpu_transfer(&self.stream, Ok(())).map_err(native_error)?;
            return Err(native_error(error));
        }
        Ok(Box::new(TentCompletion {
            stream: self.stream.clone(),
            event,
            _hosts: hosts.to_vec(),
            drained: false,
        }))
    }
    fn name(&self) -> &'static str {
        self.native.name()
    }
    fn close(&self) -> Result<(), TransferError> {
        crate::transfer::finish_gpu_transfer(&self.stream, Ok(())).map_err(native_error)
    }
}

struct TentCompletion {
    stream: Arc<CudaStream>,
    event: CudaEvent,
    _hosts: Vec<LocalHost>,
    drained: bool,
}
impl LocalCompletion for TentCompletion {
    fn poll(&mut self) -> Result<bool, TransferError> {
        // SAFETY: this owner retains the event and stream through query/drain.
        match unsafe { result::event::query(self.event.cu_event()) } {
            Ok(()) => {
                self.drained = true;
                Ok(true)
            }
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            Err(error) => {
                crate::transfer::finish_gpu_transfer(&self.stream, Ok(())).map_err(native_error)?;
                self.drained = true;
                Err(native_error(error))
            }
        }
    }
}
impl Drop for TentCompletion {
    fn drop(&mut self) {
        if !self.drained {
            let _ = crate::transfer::finish_gpu_transfer(&self.stream, Ok(()));
        }
    }
}
fn native_error(error: impl std::fmt::Display) -> TransferError {
    TransferError::new(ErrorKind::Backend, error.to_string())
}
