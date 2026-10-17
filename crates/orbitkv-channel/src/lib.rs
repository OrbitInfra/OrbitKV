//! Process channel between inference engines and an OrbitKV Cache Manager.
//!
//! Payload bytes do not travel through this crate. Messages refer to
//! descriptors in separately registered CUDA IPC or shared-memory regions.

#![deny(unsafe_code)]

#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "file-backed mmap creation requires an audited unsafe boundary"
)]
mod arena;
#[cfg(target_os = "linux")]
mod bootstrap;
#[cfg(target_os = "linux")]
mod cache_client;
mod cache_protocol;
#[cfg(target_os = "linux")]
mod client;
#[cfg(target_os = "linux")]
#[allow(
    unsafe_code,
    reason = "sealed shared completion memory uses aligned atomic access"
)]
mod completion;
#[cfg(all(target_os = "linux", any(test, feature = "test-hooks")))]
#[path = "../tests/support/claim.rs"]
pub mod test_faults;

pub mod lifecycle;
mod protocol;
mod transport;

#[cfg(target_os = "linux")]
pub use arena::{ArenaError, DEFAULT_ARENA_BYTES, DEFAULT_SLOT_CAPACITY, DescriptorArena};
#[cfg(target_os = "linux")]
pub use bootstrap::{
    BootstrapClient, BootstrapError, BootstrapInfo, BootstrapServer, BootstrapSession,
    PeerCredentials,
};
#[cfg(target_os = "linux")]
pub use cache_client::{
    BlockHashes, CacheClient, ChannelCallObservation, QueryIntent, RecoveryRead, RestoreHandle,
};
pub use cache_protocol::{
    CacheProtocolError, CancelQueryRequest, CompletionAdmission, CompletionIntent,
    CompletionObservationRequest, CompletionOutcome, CompletionRoute, PublishLayer, PublishRequest,
    QueryBundleRequest, QueryBundleResponse, QueryCommand, QueryOutcomeCode, QueryTicket,
    ReleaseRequest, RestoreLease, RestoreRequest, RestoreResponse, RestoreState,
    ShardQueryResponse,
};
#[cfg(target_os = "linux")]
pub use client::{ChannelClient, ChannelError, PublishProgress};
#[cfg(target_os = "linux")]
pub use completion::{
    CompletionError, GrantState, RESTORE_COMPLETION_SLOTS, RESTORE_ERROR_BYTES, RESTORE_PLAN_BYTES,
    RestoreCompletions, RestoreTiming,
};
pub use protocol::{
    ABI_VERSION, Command, CommandCode, DescriptorRef, ProtocolError,
    RESPONSE_FLAG_REQUEST_CONSUMED, Response, StatusCode, WIRE_MESSAGE_BYTES, WireMessage,
};
pub use transport::{
    CallOptions, DeferredResponse, TransportClient, TransportError, TransportServer,
};
