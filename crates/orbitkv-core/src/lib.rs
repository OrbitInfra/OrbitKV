//! Framework-neutral cache engine with leased state, tiered storage and GPU transfer.

#[cfg(feature = "test-hooks")]
#[path = "../tests/support/faults.rs"]
pub mod test_faults;

#[macro_use]
mod trace;
mod codec;
mod cost;
pub use codec::{EncodedSegment, StorageCodec};
mod block;
mod completion;
mod engine;
mod memory;
mod metrics;
mod peer;
mod planning;
mod query;
mod storage;
pub mod transfer;

pub use block::{
    BlockHash, LayerBlock, LayerSave, QueryResult, RawBlock, RestoreSource, SealedBlock, StateKey,
};
pub use completion::{
    CompletionAdmission, CompletionIntent, CompletionObservation, CompletionOutcome,
    CompletionResourceEvidence, CompletionRoute,
};
pub use engine::config::{EngineConfig, PeerTransferBackend};
pub use engine::instance::{GpuContext, InstanceContext, QueryRegistration};
pub use engine::{
    EngineError, MetadataStatus, OrbitKVEngine, PublishDiagnostic, RawRestoreGrant,
    RestoreExecution,
};
pub use memory::numa::NumaNode;
pub use memory::pool::{PayloadArena, PinnedAllocation};
pub use orbitkv_state::{
    BundleComponent, RecoveryContract, StateBundle, StateComponent, StateDescriptor, StateFormat,
    TokenRange,
};
pub use peer::export::{PeerError, PeerExports, TransferTicket};
pub use query::lease::QueryLeaseId;
pub use query::{QueryAdmission, QueryMode, QueryOwner, QueryReservation};
pub use storage::MemoryCacheCleanupStats;
pub use storage::inventory::{
    DEFAULT_INVENTORY_JOURNAL_BYTES, InventoryDelta, InventoryReadError, InventoryStatus,
    ResidencyInventory,
};
pub use storage::ssd::metadata::SlotMeta;
pub use storage::ssd::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdBackend, SsdCacheConfig, SsdReadLease, SsdReadPath,
    SsdWritePolicy,
};
pub use trace::{set_trace_sample_rate, should_sample};
pub use transfer::TransferMode;
pub use transfer::worker::LoadOutcome;
