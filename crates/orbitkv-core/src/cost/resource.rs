use std::hash::{Hash, Hasher};

/// Process-local identities supplied by the resource's execution owner.
/// Peer identities include the source runtime; they imply no memory capability.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum ExecutionResource {
    Gpu(u64),
    SsdStore(u64),
    SsdFile(u64),
    PrefillToDecodeHandoff {
        source_endpoint_hash: u64,
        destination_device: u64,
    },
    CacheRestore {
        source_set_hash: u64,
        destination_device: u64,
        copy_backend: u8,
    },
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    Peer(u64),
    SsdRestore {
        device: u64,
        copy_backend: u8,
        stores: u64,
        has_memory: bool,
    },
}

pub(crate) fn resource_id(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
