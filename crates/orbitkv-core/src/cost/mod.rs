//! Bounded, host-observed costs. Observations never own execution resources.
//!
//! A path is one measurement boundary, not an additive edge: codec, prefetch
//! and SSD restore paths include child operations. No device timing is inferred.

use std::sync::LazyLock;
use std::time::Duration;

use crate::completion::CompletionIntent;

const CAPACITY: usize = 512;
const MIN_SAMPLES: u64 = 4;
const MAX_AGE: Duration = Duration::from_secs(300);
const ALPHA: f64 = 0.2;

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_COST_OBSERVATIONS").as_deref() == Ok("1"));
#[cfg(feature = "peer")]
static SELECTION_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_COST_SELECTION").as_deref() == Ok("1"));
#[cfg(feature = "peer")]
static CROSS_MEDIUM_SELECTION_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_CROSS_MEDIUM_SELECTION").as_deref() == Ok("1"));

pub(crate) fn enabled() -> bool {
    *ENABLED
}

#[cfg(feature = "peer")]
pub(crate) fn selection_enabled() -> bool {
    *ENABLED && *SELECTION_ENABLED
}

#[cfg(feature = "peer")]
pub(crate) fn cross_medium_selection_enabled() -> bool {
    selection_enabled() && *CROSS_MEDIUM_SELECTION_ENABLED
}

#[cfg(all(test, feature = "peer"))]
pub(crate) fn observe_for_test(key: CostEstimateKey, seconds: f64, now: std::time::Instant) {
    estimates::ESTIMATES.lock().observe(key, seconds, now);
}

#[cfg(feature = "peer")]
mod decision;
mod estimates;
mod observation;
mod resource;
mod resource_evidence;
mod shadow;

#[cfg(feature = "peer")]
pub(crate) use decision::{SelectionScope, select_route, shadow_routes};
pub(crate) use observation::{Observation, Outcome, record_completion_observation};
pub(crate) use orbitkv_state::ReplicaRepresentation as Representation;
pub(crate) use resource::{ExecutionResource, resource_id};
pub(crate) use resource_evidence::{
    current as current_resource_evidence, record as record_resource_evidence,
};
pub(crate) use shadow::shadow;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleBoundary {
    SubmittedToCompletion,
    EnqueuedToCompletion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CompletionTarget {
    intent: CompletionIntent,
    resource: Option<ExecutionResource>,
}

impl CompletionTarget {
    const fn host_ready() -> Self {
        Self {
            intent: CompletionIntent::HostReady,
            resource: None,
        }
    }

    const fn engine_restore(device: u64) -> Self {
        Self {
            intent: CompletionIntent::EngineRestore,
            resource: Some(ExecutionResource::Gpu(device)),
        }
    }

    const fn source_release(device: u64) -> Self {
        Self {
            intent: CompletionIntent::SourceRelease,
            resource: Some(ExecutionResource::Gpu(device)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CostObservationKind {
    GpuLoadDirect,
    GpuLoadKernel,
    GpuSaveDirect,
    GpuSaveKernel,
    GpuDecode,
    GpuEncode,
    GpuSsdLoad,
    SsdUringRestore,
    SsdCufileRestore,
    GpuSsdSave,
    SsdRead,
    SsdWrite,
    SsdWriteBatch,
    SsdCufileRead,
    SsdCufileWrite,
    LocalSsdHostReady,
    #[cfg(feature = "peer")]
    RemoteRead,
    #[cfg(feature = "peer")]
    RemoteAuthorization,
    #[cfg(feature = "peer")]
    RemoteSsdAuthorization,
    #[cfg(feature = "peer")]
    PeerDramHostReady,
    #[cfg(feature = "peer")]
    PeerSsdHostReady,
    CacheRestore,
    // Caller-to-drain evidence cannot be pooled with Manager-preparation routes.
    // No comparable completion target until both routes share that boundary.
    EngineLocalRestore,
    PrefillToDecodeHandoff,
}

impl CostObservationKind {
    #[cfg(feature = "peer")]
    fn is_decode_ready_route(self) -> bool {
        matches!(self, Self::CacheRestore | Self::PrefillToDecodeHandoff)
    }

    fn is_raw_copy(self) -> bool {
        matches!(
            self,
            Self::GpuLoadDirect | Self::GpuLoadKernel | Self::GpuSaveDirect | Self::GpuSaveKernel
        )
    }

    fn sample_boundary(self) -> SampleBoundary {
        match self {
            Self::SsdUringRestore
            | Self::SsdCufileRestore
            | Self::LocalSsdHostReady
            | Self::CacheRestore
            | Self::EngineLocalRestore
            | Self::PrefillToDecodeHandoff => SampleBoundary::EnqueuedToCompletion,
            Self::GpuLoadDirect
            | Self::GpuLoadKernel
            | Self::GpuSaveDirect
            | Self::GpuSaveKernel
            | Self::GpuDecode
            | Self::GpuEncode
            | Self::GpuSsdLoad
            | Self::GpuSsdSave
            | Self::SsdRead
            | Self::SsdWrite
            | Self::SsdWriteBatch
            | Self::SsdCufileRead
            | Self::SsdCufileWrite => SampleBoundary::SubmittedToCompletion,
            #[cfg(feature = "peer")]
            Self::RemoteRead | Self::RemoteAuthorization | Self::RemoteSsdAuthorization => {
                SampleBoundary::SubmittedToCompletion
            }
            #[cfg(feature = "peer")]
            Self::PeerDramHostReady | Self::PeerSsdHostReady => {
                SampleBoundary::SubmittedToCompletion
            }
        }
    }

    fn completion_target(self, resource: ExecutionResource) -> Option<CompletionTarget> {
        match (self, resource) {
            (Self::GpuLoadDirect | Self::GpuLoadKernel, ExecutionResource::Gpu(device)) => {
                Some(CompletionTarget::engine_restore(device))
            }
            (Self::GpuSaveDirect | Self::GpuSaveKernel, ExecutionResource::Gpu(device)) => {
                Some(CompletionTarget::source_release(device))
            }
            (
                Self::SsdUringRestore | Self::SsdCufileRestore,
                ExecutionResource::SsdRestore { device, .. },
            ) => Some(CompletionTarget::engine_restore(device)),
            #[cfg(feature = "peer")]
            (Self::LocalSsdHostReady | Self::PeerDramHostReady | Self::PeerSsdHostReady, _) => {
                Some(CompletionTarget::host_ready())
            }
            (
                Self::PrefillToDecodeHandoff,
                ExecutionResource::PrefillToDecodeHandoff {
                    destination_device, ..
                },
            ) => Some(CompletionTarget::engine_restore(destination_device)),
            (
                Self::CacheRestore,
                ExecutionResource::CacheRestore {
                    destination_device, ..
                },
            ) => Some(CompletionTarget::engine_restore(destination_device)),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GpuLoadDirect => "gpu_load_direct",
            Self::GpuLoadKernel => "gpu_load_kernel",
            Self::GpuSaveDirect => "gpu_save_direct",
            Self::GpuSaveKernel => "gpu_save_kernel",
            Self::GpuDecode => "gpu_decode",
            Self::GpuEncode => "gpu_encode",
            Self::GpuSsdLoad => "gpu_ssd_load",
            Self::SsdUringRestore => "ssd_uring_restore",
            Self::SsdCufileRestore => "ssd_cufile_restore",
            Self::GpuSsdSave => "gpu_ssd_save",
            Self::SsdRead => "ssd_read",
            Self::SsdWrite => "ssd_write",
            Self::SsdWriteBatch => "ssd_write_batch",
            Self::SsdCufileRead => "ssd_cufile_read",
            Self::SsdCufileWrite => "ssd_cufile_write",
            Self::LocalSsdHostReady => "local_ssd_host_ready",
            #[cfg(feature = "peer")]
            Self::RemoteRead => "remote_read",
            #[cfg(feature = "peer")]
            Self::RemoteAuthorization => "remote_authorization",
            #[cfg(feature = "peer")]
            Self::RemoteSsdAuthorization => "remote_ssd_authorization",
            #[cfg(feature = "peer")]
            Self::PeerDramHostReady => "peer_dram_host_ready",
            #[cfg(feature = "peer")]
            Self::PeerSsdHostReady => "peer_ssd_host_ready",
            Self::CacheRestore => "cache_restore",
            Self::EngineLocalRestore => "engine_local_restore",
            Self::PrefillToDecodeHandoff => "prefill_to_decode_handoff",
        }
    }
}

/// Neither request IDs nor state keys belong here. Resources identify a GPU,
/// disk owner or peer incarnation; they are never exported as metric labels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CostEstimateKey {
    kind: CostObservationKind,
    resource: ExecutionResource,
    completion: Option<CompletionTarget>,
    representation: Representation,
    size: u8,
    fragments: u8,
    dma_ranges: u8,
    source_size: u8,
    source_fragments: u8,
    wire_size: u8,
    ssd_size: u8,
    ssd_fragments: u8,
}

impl CostEstimateKey {
    fn comparable(self, other: Self) -> bool {
        self.completion.is_some()
            && self.completion == other.completion
            && self.representation != Representation::Unknown
            && self.with_observation_kind(other.kind) == other
    }

    #[cfg(feature = "peer")]
    fn route_comparable(self, other: Self) -> bool {
        self.completion.is_some()
            && self.completion == other.completion
            && self.kind.is_decode_ready_route() == other.kind.is_decode_ready_route()
            && self.representation != Representation::Unknown
            && self.with_observation_kind_and_resource(other.kind, other.resource) == other
    }

    pub(crate) fn with_dma_ranges(self, ranges: usize) -> Self {
        Self {
            dma_ranges: bucket(ranges as u64),
            ..self
        }
    }

    pub(crate) fn with_observation_kind(self, kind: CostObservationKind) -> Self {
        Self {
            completion: kind.completion_target(self.resource),
            kind,
            ..self
        }
    }
    pub(crate) fn with_ssd_shape(
        self,
        source_bytes: u64,
        source_fragments: usize,
        target_bytes: u64,
        target_fragments: usize,
    ) -> Self {
        Self {
            source_size: bucket(source_bytes),
            source_fragments: bucket(source_fragments as u64),
            ssd_size: bucket(target_bytes),
            ssd_fragments: bucket(target_fragments as u64),
            ..self
        }
    }
    pub(crate) fn with_source_shape(self, source_bytes: u64, source_fragments: usize) -> Self {
        Self {
            source_size: bucket(source_bytes),
            source_fragments: bucket(source_fragments as u64),
            ..self
        }
    }
    pub(crate) fn with_wire_bytes(self, wire_bytes: u64) -> Self {
        Self {
            wire_size: bucket(wire_bytes),
            ..self
        }
    }
    pub(crate) fn with_observation_kind_and_resource(
        self,
        kind: CostObservationKind,
        resource: ExecutionResource,
    ) -> Self {
        Self {
            kind,
            resource,
            completion: kind.completion_target(resource),
            ..self
        }
    }
    pub(crate) fn new(
        kind: CostObservationKind,
        resource: ExecutionResource,
        representation: Representation,
        bytes: u64,
        fragments: usize,
    ) -> Self {
        let completion = kind.completion_target(resource);
        Self {
            kind,
            resource,
            completion,
            representation,
            size: bucket(bytes),
            fragments: bucket(fragments as u64),
            dma_ranges: 0,
            source_size: 0,
            source_fragments: 0,
            wire_size: 0,
            ssd_size: 0,
            ssd_fragments: 0,
        }
    }
}

fn bucket(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}
