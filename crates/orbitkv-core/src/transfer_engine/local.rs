//! Stream-bound local copies. A completion owns bytes until physical drain.
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::CudaStream;

use super::{ErrorKind, TransferEngine, TransferError};
use crate::transfer::{CopyDesc, TransferMode};

pub use super::backend::LocalHost;

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

/// OrbiTE local-copy binding, retained once per worker stream.
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

/// Every GPU worker requires OrbiTE, including a local-only deployment.
pub(crate) fn bind(
    engine: Option<Arc<TransferEngine>>,
    stream: Arc<CudaStream>,
    mode: TransferMode,
    device: u64,
) -> Result<Box<dyn LocalCopy>, TransferError> {
    engine
        .ok_or_else(|| {
            TransferError::new(ErrorKind::Unavailable, "OrbiTE local copy is unavailable")
        })?
        .bind_local(stream, mode, device)
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
