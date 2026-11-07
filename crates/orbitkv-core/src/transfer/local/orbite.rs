//! Client mappings and exporter owners for OrbiTE local copies.
use super::*;
use crate::transfer_engine::{CudaRegistration, TransferEngine};
use ::orbite::{HostMemory, Transport, cuda::local_copy::PinnedHostRegistration};

type Binding = (
    Box<dyn LocalCopy>,
    Vec<LocalHost>,
    Vec<Arc<CudaRegistration>>,
);

pub(super) fn bind(
    tensors: &HashMap<String, LocalTensor>,
    arenas: &HashMap<u64, Arc<ImportedArena>>,
    stream: Arc<CudaStream>,
    mode: TransferMode,
    owner: Arc<dyn Send + Sync>,
) -> Result<Binding, String> {
    let pool_bytes = arenas.values().try_fold(0u64, |total, arena| {
        total
            .checked_add(arena.size as u64)
            .ok_or("arena byte count overflow")
    })?;
    // This process has its own addresses and registration authority. Peer
    // transport is unused; no RDMA devices or storage SDKs are configured here.
    let engine: Arc<TransferEngine> = Arc::new(crate::peer::orbite::TransferEngine::new(
        pool_bytes,
        "127.0.0.1:0",
        Transport::Tcp,
        None,
        #[cfg(feature = "orbite-rdma")]
        None,
    )?);
    let mut ranges = HashMap::<usize, (u64, u64)>::new();
    for tensor in tensors.values() {
        if tensor.allocation == 0 || tensor.allocation as u64 > tensor.address {
            return Err("OrbiTE requires the tensor's actual storage base".into());
        }
        let range = ranges
            .entry(tensor.allocation)
            .or_insert((tensor.allocation as u64, tensor.allocation_end));
        range.0 = range.0.min(tensor.allocation as u64);
        range.1 = range.1.max(tensor.allocation_end);
    }
    let mut registrations = Vec::with_capacity(ranges.len());
    for (address, end) in ranges.into_values() {
        // SAFETY: exporter owner retains every tensor and its original storage;
        // readiness is fenced before admission and executor retains registration.
        registrations.push(
            unsafe {
                CudaRegistration::new(
                    Some(engine.clone()),
                    address,
                    (end - address) as usize,
                    stream.context().ordinal() as i32,
                    owner.clone(),
                )
            }
            .map_err(|e| e.to_string())?,
        );
    }
    let mut hosts = Vec::with_capacity(arenas.len());
    for arena in arenas.values() {
        // SAFETY: one pin owner per physical cuMemHostRegister, retaining mmap/fd.
        let pin = unsafe {
            PinnedHostRegistration::from_raw_parts(arena.pointer, arena.size, arena.clone())
        }
        .map_err(|e| e.to_string())?;
        // SAFETY: whole-arena aliases share this memory/access lock. Active source
        // grants are held by execute until all completions drain, including errors.
        let memory = unsafe {
            HostMemory::from_cuda_registered_parts(arena.pointer, arena.size, arena.clone(), pin)
        }
        .map_err(|e| e.to_string())?;
        hosts.push(LocalHost::Imported(Arc::new(memory)));
    }
    let device = registrations
        .first()
        .ok_or("local copy requires a registered tensor")?
        .address;
    Ok((
        engine
            .bind_local(stream, mode, device)
            .map_err(|e| e.to_string())?,
        hosts,
        registrations,
    ))
}
