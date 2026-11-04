//! Imported tensors and codec pools own one registration through their last loan.
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{ErrorKind, Permission, Registration, TransferBackend, TransferError};
use parking_lot::RwLock;

/// Persistent native registration and the actual CUDA allocation/context owner.
/// Layouts share this owner; close fences and drains active transfer loans.
pub struct CudaRegistration {
    pub(crate) address: u64,
    pub(crate) length: usize,
    pub(crate) device: i32,
    native: RwLock<Option<(Arc<dyn TransferBackend>, Registration)>>,
    owner: Option<Arc<dyn Send + Sync>>,
    uncertain: AtomicBool,
}
impl std::fmt::Debug for CudaRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRegistration")
            .field("address", &self.address)
            .field("length", &self.length)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
impl CudaRegistration {
    /// # Safety
    /// The owner retains this entire valid CUDA range and its context. Importer
    /// readiness precedes this call; destination aliases stay coordinated.
    #[allow(
        clippy::mem_forget,
        reason = "uncertain registration retains backing and its budget"
    )]
    pub unsafe fn new(
        backend: Option<Arc<dyn TransferBackend>>,
        address: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
    ) -> Result<Arc<Self>, TransferError> {
        if address == 0 || length == 0 || address.checked_add(length as u64).is_none() || device < 0
        {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "invalid CUDA allocation",
            ));
        }
        let native = if let Some(backend) = backend {
            let result = catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: actual backing is retained outside this unwind boundary.
                unsafe {
                    Arc::clone(&backend).register_cuda(
                        address,
                        length,
                        device,
                        Arc::clone(&owner),
                        Permission::LocalOnly,
                    )
                }
            }));
            match result {
                Ok(Ok(registration)) => Some((backend, registration)),
                Ok(Err(error)) if error.kind != ErrorKind::Unquiesced => return Err(error),
                uncertain => {
                    std::mem::forget((uncertain, backend, owner));
                    return Err(unquiesced());
                }
            }
        } else {
            None
        };
        Ok(Arc::new(Self {
            address,
            length,
            device,
            native: RwLock::new(native),
            owner: Some(owner),
            uncertain: AtomicBool::new(false),
        }))
    }
    /// Check allocation bounds; admission still requires a live registration loan.
    pub fn covers(&self, address: u64, length: usize) -> bool {
        address >= self.address
            && address
                .checked_add(length as u64)
                .is_some_and(|end| end <= self.address + self.length as u64)
    }
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn loan(
        &self,
        backend: &Arc<dyn TransferBackend>,
    ) -> Result<impl Drop + '_, TransferError> {
        let native = self.native.read();
        if self.uncertain.load(Ordering::Acquire) {
            return Err(unquiesced());
        }
        if !native
            .as_ref()
            .is_some_and(|(owner, _)| Arc::ptr_eq(owner, backend))
        {
            return Err(TransferError::new(
                ErrorKind::Unavailable,
                "CUDA registration is closed or belongs to another engine",
            ));
        }
        Ok(native)
    }
    pub(crate) fn retain(&self) {
        self.uncertain.store(true, Ordering::Release);
    }
    /// Fence admission and unregister after every transfer loan has drained.
    #[allow(
        clippy::mem_forget,
        reason = "unknown cleanup must not unwind through backing owners"
    )]
    pub fn close(&self) -> Result<(), TransferError> {
        let mut native = self.native.write();
        if self.uncertain.load(Ordering::Acquire) {
            return Err(unquiesced());
        }
        if let Some((backend, registration)) = native.as_mut() {
            let result = catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: the exclusive lock proves all transfer loans ended.
                unsafe { backend.unregister(registration) }
            }));
            if !matches!(result, Ok(Ok(()))) {
                self.retain();
                std::mem::forget(result);
                return Err(unquiesced());
            }
        }
        native.take();
        Ok(())
    }
}
fn unquiesced() -> TransferError {
    TransferError::new(
        ErrorKind::Unquiesced,
        "CUDA registration has no cleanup receipt",
    )
}
impl Drop for CudaRegistration {
    #[allow(
        clippy::mem_forget,
        reason = "failed native cleanup retains actual backing, engine and budget"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            std::mem::forget((self.native.get_mut().take(), self.owner.take()));
        }
    }
}
