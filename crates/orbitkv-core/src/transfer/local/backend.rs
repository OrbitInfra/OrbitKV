//! Optional OrbiTE binding for client mappings and retained tensor owners.
use super::*;
#[cfg(feature = "orbite")]
#[path = "orbite.rs"]
mod orbite;

type Binding = (
    Box<dyn LocalCopy>,
    Vec<LocalHost>,
    Vec<Arc<crate::transfer_engine::CudaRegistration>>,
);

pub(super) fn bind(
    owner: Arc<dyn Send + Sync>,
    bindings: &HashMap<String, LocalTensor>,
    imported: &HashMap<u64, Arc<ImportedArena>>,
    stream: Arc<CudaStream>,
    mode: TransferMode,
) -> Result<Binding, String> {
    #[cfg(feature = "orbite")]
    {
        orbite::bind(bindings, imported, stream, mode, owner)
    }
    #[cfg(not(feature = "orbite"))]
    {
        let _ = (owner, bindings, imported, stream, mode);
        Err("OrbiTE local copy requires the orbite build feature".into())
    }
}
