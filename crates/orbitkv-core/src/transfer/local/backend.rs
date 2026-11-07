//! Client-side provider dispatch, imported registrations and native tensor metadata.
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
    engine: Option<(crate::PeerTransferBackend, Arc<dyn Send + Sync>)>,
    bindings: &HashMap<String, LocalTensor>,
    imported: &HashMap<u64, Arc<ImportedArena>>,
    stream: Arc<CudaStream>,
    mode: TransferMode,
) -> Result<Binding, String> {
    match engine {
        Some((
            crate::PeerTransferBackend::Orbite
            | crate::PeerTransferBackend::OrbiteTcp
            | crate::PeerTransferBackend::OrbiteRdma,
            owner,
        )) => {
            #[cfg(feature = "orbite")]
            {
                orbite::bind(bindings, imported, stream.clone(), mode, owner)
            }
            #[cfg(not(feature = "orbite"))]
            {
                let _ = (owner, bindings, imported);
                Err("Manager selected orbite, but this client was built without it".into())
            }
        }
        None | Some((crate::PeerTransferBackend::Mooncake, _)) => Ok((
            crate::transfer_engine::local::bind(
                crate::PeerTransferBackend::Mooncake,
                None,
                stream.clone(),
                mode,
                0,
            )
            .map_err(|e| e.to_string())?,
            Vec::new(),
            Vec::new(),
        )),
    }
}
