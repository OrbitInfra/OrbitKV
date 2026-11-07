use std::time::Duration;

use orbitkv_state::ReplicaRepresentation;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompletionIntent {
    HostReady,
    EngineRestore,
    SourceRelease,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompletionRoute {
    PrefillToDecodeHandoff,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionAdmission {
    Admitted,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionOutcome {
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

/// Bounded resource state captured when the physical route was admitted.
/// These values are observations for freshness/admission checks, never metric
/// labels or stable cost-key dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletionResourceEvidence {
    pub decode_page_bytes: u64,
    pub queue_depth: u32,
    pub queue_parallelism: u32,
    pub transfer_inflight_bytes: u64,
    pub transfer_bandwidth_bytes_per_second: u64,
}

/// One measured physical completion interval reported by its execution owner.
///
/// Request identifiers and state keys are intentionally absent. The
/// transfer generation identifies a fenced report,
/// but it is freshness evidence rather than a cost-model dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionObservation {
    pub instance_id: String,
    pub destination_device_id: i32,
    pub source_endpoint: String,
    pub transfer_generation: u64,
    pub intent: CompletionIntent,
    pub route: CompletionRoute,
    pub representation: ReplicaRepresentation,
    pub logical_bytes: u64,
    pub wire_bytes: u64,
    pub fragment_count: u32,
    pub elapsed: Duration,
    pub resources: CompletionResourceEvidence,
    pub admission: CompletionAdmission,
    pub outcome: CompletionOutcome,
}
