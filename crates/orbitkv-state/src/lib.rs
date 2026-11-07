#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! Framework-neutral contracts shared by OrbitKV engine adapters.
//!
//! These types describe logical model state and recovery requirements without
//! importing vLLM block IDs, SGLang radix nodes, CUDA handles, or transport
//! implementations. Framework adapters translate their native objects into
//! this vocabulary before calling the OrbitKV data plane.

#![forbid(unsafe_code)]

mod bundle;
mod component;
mod discovery;
mod format;
mod inventory;
mod key;

pub use bundle::{
    BundleComponent, RecoveryContract, RecoveryDemand, RecoveryError, RecoveryRule, StateBundle,
    StateRequirement,
};
pub use component::StateComponent;
pub use discovery::{
    BlockCandidates, CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_ENDPOINT_BYTES,
    DISCOVERY_MAX_KEYS, DISCOVERY_MAX_REPLICAS_PER_MEDIUM, DiscoveryCoverage, ReplicaLocation,
    ReplicaMedium, ReplicaMetadata, ReplicaRepresentation, validate_discovery_query,
};
pub use format::{AttentionRole, Scalar16, StateDType, StateFormat, StateLayout, StorageFormat};
pub use inventory::{
    INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, INVENTORY_OPEN_MAX_BYTES,
    INVENTORY_SCOPE_MAX_NAMESPACES, INVENTORY_STREAM_PROTOCOL, InventoryFence, InventoryRecord,
    InventoryScope,
};
pub use key::{
    ContractError, Digest, StateDescriptor, StateKey, StorageSlot, TokenRange, generation_hash,
    group_hash, is_storage_namespace, storage_namespace,
};
