// Candidate discovery -> requester planning -> source authorization -> peer READ.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{info, warn};
use orbitkv_proto::proto::engine::{QueryBlocksForTransferResponse, TransferBlockInfo};

use crate::memory::numa::NumaNode;

use opentelemetry::KeyValue;

use super::completion::{TransferCompletions, TransferLockGuard};
use super::execute::{
    AuthorizationError, AuthorizationMode, FetchResult, SegmentFetcher, execute_fetch_plan,
};
use super::provider::{PeerReadSlice, PeerTransferError, PeerTransport};
use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, Observation, Outcome, Representation,
    resource_id,
};
use crate::memory::AllocateFn;
use crate::metrics::core_metrics;
use crate::planning::peer::{FetchPlan, FetchSegment, PeerSource};
use crate::storage::MaterializedBlocks;
use orbitkv_catalog::GlobalIndex;

/// Minimum usable transfer timeout. If the server's lock timeout minus the
/// safety margin falls below this, we use this floor to avoid instant timeouts.
const MIN_TRANSFER_TIMEOUT: Duration = Duration::from_secs(10);

/// Stop submitting work before the source marks the session overdue.
const LOCK_TIMEOUT_MARGIN: Duration = Duration::from_secs(60);

/// Upper bound for a single pinned-pool allocation while staging a Peer fetch.
/// LRU reclaim must carve a contiguous hole of the requested size, so a
/// whole-prefix slab can force eviction of far more bytes than the fetch needs.
const FETCH_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

/// Peer remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries Catalog for their
/// location, picks the best remote node, and uses gRPC authorization plus a
/// peer READ to fetch them.
pub(crate) struct PeerReader {
    global_index: Arc<GlobalIndex>,
    completions: Arc<TransferCompletions>,
    membership: Arc<orbitkv_catalog::MembershipView>,
    transfer: Arc<PeerTransport>,
    allocate_fn: AllocateFn,
}

pub(super) struct AuthorizedSegment {
    response: QueryBlocksForTransferResponse,
    lock_guard: TransferLockGuard,
    authorized_at: Instant,
    query_elapsed: Duration,
    resource: ExecutionResource,
    route_observation: Observation,
}

fn valid_transfer_authorization(
    response: &QueryBlocksForTransferResponse,
    records: &[orbitkv_state::InventoryRecord],
) -> bool {
    !response.transfer_endpoint.is_empty()
        && response.native_source_epoch.len() == 32
        && response
            .native_source_epoch
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        && response.blocks.len() == records.len()
        && response
            .blocks
            .iter()
            .zip(records)
            .all(|(block, record)| block.block_hash == record.key.hash)
}

#[tonic::async_trait]
impl SegmentFetcher for PeerReader {
    type Grant = AuthorizedSegment;

    async fn authorize_segment(
        &self,
        segment: &FetchSegment,
        mode: AuthorizationMode,
    ) -> Result<Self::Grant, AuthorizationError> {
        if !self.membership.permits(&segment.owner) {
            return Err(AuthorizationError::Rejected);
        }
        let remote_addr = &segment.owner.endpoint;
        // A prepared grant can wait for the preceding READ. That residence time
        // is not this route's service cost; keep leaf authorization/READ samples
        // but train the composite route only on demand-authorized segments.
        let mut route_observation = if mode == AuthorizationMode::Demand && crate::cost::enabled() {
            Observation::new(segment.cost_estimate_key(), None)
        } else {
            Observation::disabled()
        };
        route_observation.admitted();
        route_observation.submitted();

        // Query the OrbitKV authority before exposing any physical addresses.
        let query_start = Instant::now();
        let resource = ExecutionResource::Peer(if crate::cost::enabled() {
            resource_id(&segment.owner)
        } else {
            0
        });
        let mut authorization_observation = Observation::new(
            CostEstimateKey::new(
                match segment.source {
                    PeerSource::Dram => CostObservationKind::RemoteAuthorization,
                    PeerSource::Ssd => CostObservationKind::RemoteSsdAuthorization,
                },
                resource,
                segment.representation,
                segment.stored_bytes.unwrap_or(0),
                segment.records.len(),
            ),
            None,
        );
        authorization_observation.admitted();
        authorization_observation.submitted();
        let authorization = self
            .completions
            .authorize(segment, self.membership.owner().incarnation)
            .await;
        authorization_observation.finish(
            match &authorization {
                Ok(_) => Outcome::Completed,
                Err(error) if error.code() == tonic::Code::DeadlineExceeded => Outcome::TimedOut,
                Err(error) if error.code() == tonic::Code::Cancelled => Outcome::Cancelled,
                Err(_) => Outcome::Failed,
            },
            None,
        );
        let query_elapsed = query_start.elapsed();
        core_metrics().remote_stage_duration_seconds.record(
            query_elapsed.as_secs_f64(),
            &[
                KeyValue::new("stage", "authorization"),
                KeyValue::new("status", if authorization.is_ok() { "ok" } else { "error" }),
            ],
        );
        let (lock_guard, response) = match authorization {
            Ok(cr) => cr,
            Err(error) if error.code() == tonic::Code::FailedPrecondition => {
                route_observation.finish(Outcome::Failed, None);
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "rejected")]);
                // A batch rejection does not identify the stale key. Exclude the
                // route only from this request; Watch owns shared index updates.
                return Err(AuthorizationError::Rejected);
            }
            Err(error)
                if matches!(
                    error.code(),
                    tonic::Code::ResourceExhausted | tonic::Code::Unavailable
                ) =>
            {
                route_observation.finish(Outcome::Failed, None);
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "rejected")]);
                return Err(AuthorizationError::Rejected);
            }
            Err(e) => {
                route_observation.finish(
                    if e.code() == tonic::Code::DeadlineExceeded {
                        Outcome::TimedOut
                    } else if e.code() == tonic::Code::Cancelled {
                        Outcome::Cancelled
                    } else {
                        Outcome::Failed
                    },
                    None,
                );
                warn!("Remote query to {remote_addr} failed: {e}");
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "error")]);
                return Err(AuthorizationError::Failed);
            }
        };

        // The guard moves into the blocking transfer with the destination buffers.
        // Cancelling this future cannot release either while the READ is running.
        if !valid_transfer_authorization(&response, &segment.records)
            || !self.transfer.accepts_endpoint(&response.transfer_endpoint)
        {
            route_observation.finish(Outcome::Failed, None);
            warn!("Remote query to {remote_addr} returned invalid transfer authorization");
            drop(lock_guard);
            core_metrics()
                .remote_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Err(AuthorizationError::Failed);
        }

        Ok(AuthorizedSegment {
            response,
            lock_guard,
            authorized_at: Instant::now(),
            query_elapsed,
            resource,
            route_observation,
        })
    }

    async fn fetch_segment(
        &self,
        segment: &FetchSegment,
        grant: Self::Grant,
        req_id: &str,
        destination_nodes: &[NumaNode],
    ) -> Result<MaterializedBlocks, ()> {
        let AuthorizedSegment {
            response,
            lock_guard,
            authorized_at,
            query_elapsed,
            resource,
            route_observation,
        } = grant;
        let prepared_wait = authorized_at.elapsed();
        let transfer_start = Instant::now();
        let remote_addr = &segment.owner.endpoint;
        let namespace = &segment.records[0].key.namespace;

        // peer READ all blocks + build SealedBlocks.
        let transfer_timeout = transfer_timeout_from_server(response.lock_timeout_secs);
        let blocks = response.blocks;
        let total_bytes: u64 = blocks
            .iter()
            .flat_map(|b| &b.slots)
            .map(|s| s.k_size + s.v_size)
            .sum();
        let (result, transfer_timing) = match fetch_blocks_via_peer(
            &self.transfer,
            &self.allocate_fn,
            namespace,
            &response.transfer_endpoint,
            &response.native_source_epoch,
            &blocks,
            destination_nodes,
            transfer_timeout,
            lock_guard,
            resource,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                route_observation.finish(Outcome::Failed, None);
                warn!("peer transfer from {remote_addr} failed: {e}");
                self.transfer
                    .invalidate_segment(&response.transfer_endpoint);
                core_metrics()
                    .remote_fetch_total
                    .add(1, &[KeyValue::new("status", "error")]);
                return Err(());
            }
        };
        route_observation.finish(Outcome::Completed, Some(total_bytes));

        // Retain authorization + payload service timing. Lookahead residence
        // belongs to the pipeline queue, not the READ or its throughput sample.
        let elapsed = query_elapsed + transfer_start.elapsed();
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
            mb / elapsed.as_secs_f64()
        } else {
            0.0
        };
        info!(
            "Peer fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} slots={} descs={} slabs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
            result.len(),
            segment.records.len(),
            transfer_timing.slot_count,
            transfer_timing.transfer_desc_count,
            transfer_timing.numa_slab_count,
        );
        info!(
            "Peer fetch stages: req_id={req_id} remote={remote_addr} query_ms={:.2} prepared_wait_ms={:.2} build_transfer_tasks_ms={:.2} transfer_wait_ms={:.2} rebuild_ms={:.2}",
            query_elapsed.as_secs_f64() * 1000.0,
            prepared_wait.as_secs_f64() * 1000.0,
            transfer_timing.build_transfer_tasks.as_secs_f64() * 1000.0,
            transfer_timing.mooncake_wait.as_secs_f64() * 1000.0,
            transfer_timing.rebuild.as_secs_f64() * 1000.0,
        );
        let m = core_metrics();
        let ok = &[KeyValue::new("status", "ok")];
        m.remote_fetch_total.add(1, ok);
        m.remote_fetch_duration_seconds
            .record(elapsed.as_secs_f64(), ok);
        m.remote_fetch_bytes.add(total_bytes, ok);
        self.transfer.record_nic_load();
        for (stage, duration) in [
            ("prepared_wait", prepared_wait),
            ("allocation", transfer_timing.build_transfer_tasks),
            ("read", transfer_timing.mooncake_wait),
            ("rebuild", transfer_timing.rebuild),
        ] {
            m.remote_stage_duration_seconds.record(
                duration.as_secs_f64(),
                &[KeyValue::new("stage", stage), KeyValue::new("status", "ok")],
            );
        }
        Ok(result)
    }
}

impl PeerReader {
    pub(crate) fn new(
        global_index: Arc<GlobalIndex>,
        transfer: Arc<PeerTransport>,
        allocate_fn: AllocateFn,
        membership: Arc<orbitkv_catalog::MembershipView>,
    ) -> Self {
        info!(
            "Peer remote fetch enabled (advertise={})",
            membership.owner().endpoint
        );
        Self {
            global_index,
            completions: Arc::new(TransferCompletions::default()),
            membership,
            transfer,
            allocate_fn,
        }
    }

    /// Refresh peer evidence in the same request records that retain local SSD
    /// alternatives. Directory hints still require authoritative source grants.
    pub(crate) fn discover(&self, rows: &mut [crate::planning::replica::ReplicaSet]) {
        for row in rows.iter_mut() {
            row.set_remote(orbitkv_state::DiscoveryCoverage::Unavailable, Vec::new());
        }
        if !self.membership.permits(self.membership.owner()) || rows.is_empty() {
            return;
        }
        let keys: Vec<_> = rows.iter().map(|row| row.key.clone()).collect();
        let candidates = self.global_index.lookup(&keys);
        for (row, mut candidate) in rows.iter_mut().zip(candidates) {
            if candidate.key != row.key {
                continue;
            }
            candidate
                .replicas
                .retain(|replica| self.membership.permits(&replica.owner));
            row.set_remote(candidate.coverage, candidate.replicas);
        }
    }

    pub(crate) async fn fetch_plan(
        &self,
        plan: FetchPlan<'_>,
        req_id: &str,
        destination_nodes: &[NumaNode],
    ) -> FetchResult {
        let planned_blocks = plan.block_count();
        let started_at = Instant::now();
        let result = execute_fetch_plan(self, plan, req_id, destination_nodes).await;
        let metrics = core_metrics();
        metrics
            .remote_fetch_plan_segments
            .record(result.attempts as u64, &[]);
        metrics
            .remote_fetch_plan_completed_segments
            .record(result.completed_segments as u64, &[]);
        info!(
            "Peer fetch plan: req_id={req_id} attempted_segments={} completed_segments={} status={:?} planned_blocks={} fetched_blocks={} total_ms={:.2}",
            result.attempts,
            result.completed_segments,
            result.status,
            planned_blocks,
            result.blocks.len(),
            started_at.elapsed().as_secs_f64() * 1000.0
        );
        result
    }
}

/// One fetched slot: its peer-staged segments plus the NUMA node they sit on.
/// The NUMA travels with the slot so a re-served block advertises real topology.
type StagedSlot = (
    Vec<SegmentAlloc>,
    NumaNode,
    Option<Vec<crate::codec::EncodedSegment>>,
);
/// A staged block awaiting SealedBlock rebuild: its hash and per-slot allocations.
type StagedBlock = (Vec<u8>, Vec<StagedSlot>);

/// Allocate local memory, execute one peer READ batch, and rebuild blocks.
#[allow(
    clippy::too_many_arguments,
    reason = "the peer incarnation scopes cost evidence"
)]
async fn fetch_blocks_via_peer(
    transfer: &Arc<PeerTransport>,
    allocate_fn: &AllocateFn,
    namespace: &str,
    transfer_endpoint: &str,
    native_source_epoch: &str,
    blocks: &[TransferBlockInfo],
    destination_nodes: &[NumaNode],
    transfer_timeout: Duration,
    lock_guard: TransferLockGuard,
    resource: ExecutionResource,
) -> Result<(MaterializedBlocks, TransferTiming), String> {
    if blocks.is_empty() {
        return Ok((Vec::new(), TransferTiming::default()));
    }

    let mut slabs = ChunkedSlabs::new(
        allocate_fn,
        FETCH_CHUNK_BYTES,
        sum_segment_bytes_by_numa(blocks, destination_nodes)?,
    );

    // (block_hash, Vec<(slot_segments, slot_numa)>) — for building SealedBlock afterwards.
    // The per-slot NUMA is preserved so a re-served fetched block advertises real topology.
    let mut block_allocs: Vec<StagedBlock> = Vec::new();
    let mut slot_count = 0usize;
    let build_start = Instant::now();

    let (all_descs, mut timing) = {
        let mut all_descs: Vec<PeerReadSlice> = Vec::new();

        for block_info in blocks {
            if block_info.slots.len() != destination_nodes.len() {
                return Err("source grant slot count differs from destination topology".into());
            }
            slot_count += block_info.slots.len();
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

            for (slot, &numa) in block_info.slots.iter().zip(destination_nodes) {
                let mut segments = Vec::new();

                // K segment
                if slot.k_size > 0 {
                    let len = usize::try_from(slot.k_size)
                        .map_err(|_| format!("K size exceeds usize: {}", slot.k_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "K")?;
                    if slot.k_ptr == 0 {
                        return Err("remote K ptr is null".to_string());
                    }
                    all_descs.push(PeerReadSlice {
                        allocation: Arc::clone(&alloc),
                        local: local_ptr,
                        remote_address: slot.k_ptr,
                        length: len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                // V segment (split KV)
                if slot.v_size > 0 {
                    if slot.v_ptr == 0 {
                        return Err("remote V ptr is null".into());
                    }
                    let len = usize::try_from(slot.v_size)
                        .map_err(|_| format!("V size exceeds usize: {}", slot.v_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "V")?;
                    all_descs.push(PeerReadSlice {
                        allocation: Arc::clone(&alloc),
                        local: local_ptr,
                        remote_address: slot.v_ptr,
                        length: len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                let encoding = if slot.encoding.is_empty() {
                    None
                } else {
                    if slot.encoding.len() > 4096 {
                        return Err("remote codec metadata too large".into());
                    }
                    Some(
                        serde_json::from_slice(&slot.encoding)
                            .map_err(|e| format!("remote codec metadata: {e}"))?,
                    )
                };
                slot_allocs.push((segments, numa, encoding));
            }

            block_allocs.push((block_info.block_hash.clone(), slot_allocs));
        }

        if all_descs.is_empty() {
            let timing = TransferTiming {
                build_transfer_tasks: build_start.elapsed(),
                slot_count,
                numa_slab_count: slabs.chunk_count,
                ..TransferTiming::default()
            };
            return Ok((Vec::new(), timing));
        }

        let transfer_desc_count = all_descs.len();

        let timing = TransferTiming {
            build_transfer_tasks: build_start.elapsed(),
            transfer_desc_count,
            slot_count,
            numa_slab_count: slabs.chunk_count,
            ..TransferTiming::default()
        };
        (all_descs, timing)
    };

    let wait_start = Instant::now();
    let (stored_bytes, logical_bytes, representation) = if crate::cost::enabled() {
        let (logical_bytes, representation) = transfer_shape(&block_allocs);
        (
            all_descs.iter().map(|slice| slice.length as u64).sum(),
            logical_bytes,
            representation,
        )
    } else {
        (0, None, Representation::Unknown)
    };
    let mut observation = Observation::new(
        CostEstimateKey::new(
            CostObservationKind::RemoteRead,
            resource,
            representation,
            logical_bytes.unwrap_or(stored_bytes),
            all_descs.len(),
        ),
        logical_bytes,
    );
    let transfer = Arc::clone(transfer);
    let transfer_endpoint = transfer_endpoint.to_string();
    let native_source_epoch = native_source_epoch.to_string();
    let (block_allocs, transferred) = lock_guard
        .run_with_buffers(block_allocs, move || {
            observation.admitted();
            observation.submitted();
            let result = transfer.read(
                &transfer_endpoint,
                &all_descs,
                transfer_timeout,
                &native_source_epoch,
            );
            let (outcome, actual_bytes) = match &result {
                Ok(bytes) if *bytes as u64 == stored_bytes => {
                    (Outcome::Completed, Some(*bytes as u64))
                }
                Ok(bytes) => (Outcome::Failed, Some(*bytes as u64)),
                Err(PeerTransferError::Timeout) => (Outcome::TimedOut, None),
                Err(_) => (Outcome::Failed, None),
            };
            // The blocking owner outlives a cancelled caller. A timeout is only
            // recorded after native drain and never trains completed service.
            observation.finish(outcome, actual_bytes);
            result
        })
        .await
        .map_err(|error| format!("peer READ task failed: {error}"))?;
    transferred.map_err(|error| format!("peer READ failed: {error}"))?;
    let block_allocs = block_allocs.ok_or("peer READ buffers remain retained")?;
    timing.mooncake_wait = wait_start.elapsed();

    // Build SealedBlocks from allocated memory
    let rebuild_start = Instant::now();
    let mut result: MaterializedBlocks = Vec::with_capacity(block_allocs.len());
    for (hash, slot_allocs) in block_allocs {
        let key = StateKey::new(namespace.to_string(), hash);
        let slots: Vec<(RawBlock, NumaNode)> = slot_allocs
            .into_iter()
            .map(|(segs, numa, encoding)| {
                let segments: Vec<Segment> = segs
                    .into_iter()
                    .map(|sa| {
                        let ptr = NonNull::new(sa.ptr_addr as *mut u8)
                            .expect("slab segment pointer must be non-null");
                        Segment::new(ptr, sa.size, sa.alloc)
                    })
                    .collect();
                let mut raw = RawBlock::new(segments);
                raw.encoding = encoding;
                (raw, numa)
            })
            .collect();
        for (slot, _) in &slots {
            slot.validate_encoding()
                .inspect_err(|_| core_metrics().storage_codec_decode_failures.add(1, &[]))?;
        }
        let sealed = Arc::new(SealedBlock::from_slots(slots));
        result.push((key, sealed));
    }
    timing.rebuild = rebuild_start.elapsed();

    Ok((result, timing))
}

fn transfer_shape(blocks: &[StagedBlock]) -> (Option<u64>, Representation) {
    let mut logical_bytes = Some(0u64);
    let mut representation = None;
    for (_, slots) in blocks {
        for (segments, _, encoding) in slots {
            match encoding {
                Some(metadata) => {
                    if metadata.len() != segments.len() {
                        logical_bytes = None;
                    }
                    for meta in metadata {
                        logical_bytes = logical_bytes
                            .and_then(|bytes| bytes.checked_add(meta.logical_bytes as u64));
                        let next = Representation::from(meta.format);
                        representation = Some(match representation {
                            None => next,
                            Some(previous) if previous == next => previous,
                            Some(_) => Representation::Mixed,
                        });
                    }
                }
                None => {
                    logical_bytes = None;
                    representation = Some(match representation {
                        None | Some(Representation::Raw) => Representation::Raw,
                        Some(_) => Representation::Mixed,
                    });
                }
            }
        }
    }
    (
        logical_bytes,
        representation.unwrap_or(Representation::Unknown),
    )
}

/// Total staged bytes per NUMA node for one fetch batch. Used to right-size
/// the last chunk of each NUMA so small fetches don't over-allocate.
fn sum_segment_bytes_by_numa(
    blocks: &[TransferBlockInfo],
    destination_nodes: &[NumaNode],
) -> Result<HashMap<NumaNode, u64>, String> {
    let mut bytes_per_numa: HashMap<NumaNode, u64> = HashMap::new();
    for block_info in blocks {
        if block_info.slots.len() != destination_nodes.len() {
            return Err(format!(
                "remote block has {} slots, receiver registered {}",
                block_info.slots.len(),
                destination_nodes.len()
            ));
        }
        for (slot, &numa) in block_info.slots.iter().zip(destination_nodes) {
            let mut add = 0u64;
            if slot.k_size > 0 {
                add += slot.k_size;
            }
            if slot.v_size > 0 && slot.v_ptr != 0 {
                add = add
                    .checked_add(slot.v_size)
                    .ok_or_else(|| format!("segment bytes overflow on {numa}"))?;
            }
            let total = bytes_per_numa.entry(numa).or_insert(0);
            *total = total
                .checked_add(add)
                .ok_or_else(|| format!("numa bytes overflow while summing segments on {numa}"))?;
        }
    }
    Ok(bytes_per_numa)
}

/// Bump allocator over bounded pinned chunks, one active chunk per NUMA node.
/// Each staged segment holds an Arc to its own chunk, so starting a fresh
/// chunk never invalidates previously staged segments, and fetched blocks are
/// freed chunk-by-chunk on eviction instead of all-or-nothing per fetch.
///
/// A chunk is sized `min(remaining bytes on that NUMA, chunk_bytes)`: the cap
/// bounds LRU-reclaim amplification on large fetches, the remaining-bytes
/// clamp keeps small fetches from grabbing a whole `chunk_bytes` slab (which
/// would fail outright on pools smaller than the cap).
struct ChunkedSlabs<'a> {
    allocate_fn: &'a AllocateFn,
    chunk_bytes: u64,
    current: HashMap<NumaNode, NumaSlab>,
    /// Bytes of this batch not yet staged, per NUMA.
    remaining: HashMap<NumaNode, u64>,
    chunk_count: usize,
}

impl<'a> ChunkedSlabs<'a> {
    fn new(
        allocate_fn: &'a AllocateFn,
        chunk_bytes: u64,
        remaining: HashMap<NumaNode, u64>,
    ) -> Self {
        Self {
            allocate_fn,
            chunk_bytes,
            current: HashMap::new(),
            remaining,
            chunk_count: 0,
        }
    }

    fn alloc_segment(
        &mut self,
        numa: NumaNode,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::memory::pool::PinnedAllocation>), String> {
        if let Some(slab) = self.current.get_mut(&numa)
            && let Ok(seg) = slab.allocate(len, segment_kind)
        {
            self.consume_remaining(numa, len);
            return Ok(seg);
        }

        // No chunk on this NUMA yet, or the current one can't fit the segment.
        let remaining = *self
            .remaining
            .get(&numa)
            .expect("remaining bytes tracked for every NUMA in the batch");
        let chunk = remaining.min(self.chunk_bytes).max(len as u64);
        let allocation = (self.allocate_fn)(chunk, Some(numa)).ok_or_else(|| {
            format!("failed to allocate fetch chunk ({chunk} bytes) on {numa} for {segment_kind}")
        })?;
        let capacity = usize::try_from(chunk)
            .map_err(|_| format!("fetch chunk size exceeds usize: {chunk}"))?;
        self.chunk_count += 1;
        self.current.insert(
            numa,
            NumaSlab {
                allocation,
                next_offset: 0,
                capacity,
            },
        );
        self.consume_remaining(numa, len);
        self.current
            .get_mut(&numa)
            .expect("chunk just inserted")
            .allocate(len, segment_kind)
    }

    fn consume_remaining(&mut self, numa: NumaNode, len: usize) {
        let rem = self
            .remaining
            .get_mut(&numa)
            .expect("remaining bytes tracked for every NUMA in the batch");
        *rem = rem.saturating_sub(len as u64);
    }
}

struct NumaSlab {
    allocation: Arc<crate::memory::pool::PinnedAllocation>,
    next_offset: usize,
    capacity: usize,
}

impl NumaSlab {
    fn allocate(
        &mut self,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::memory::pool::PinnedAllocation>), String> {
        let end = self.next_offset.checked_add(len).ok_or_else(|| {
            format!(
                "slab offset overflow while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            )
        })?;
        if end > self.capacity {
            return Err(format!(
                "slab exhausted while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            ));
        }

        let ptr = unsafe { self.allocation.as_non_null().as_ptr().add(self.next_offset) };
        self.next_offset = end;
        let ptr = NonNull::new(ptr).ok_or_else(|| "slab pointer is null".to_string())?;
        Ok((ptr, Arc::clone(&self.allocation)))
    }
}

struct SegmentAlloc {
    ptr_addr: u64,
    alloc: Arc<crate::memory::pool::PinnedAllocation>,
    size: usize,
}

#[derive(Default)]
struct TransferTiming {
    build_transfer_tasks: Duration,
    mooncake_wait: Duration,
    rebuild: Duration,
    transfer_desc_count: usize,
    slot_count: usize,
    numa_slab_count: usize,
}

/// Compute client-side transfer timeout from server's lock timeout.
/// This is a submission budget, not proof of source lifetime: Mooncake drains
/// already-submitted work even past the deadline, retaining source reservations.
/// Orphan revocation and cross-host failure qualification remain open.
fn transfer_timeout_from_server(lock_timeout_secs: u32) -> Duration {
    let server = Duration::from_secs(lock_timeout_secs as u64);
    server
        .saturating_sub(LOCK_TIMEOUT_MARGIN)
        .max(MIN_TRANSFER_TIMEOUT)
}

#[cfg(test)]
#[path = "../../tests/unit/peer/read.rs"]
mod tests;
