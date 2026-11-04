use super::*;
use crate::block::SealedBlock;
use crate::planning::replica::ReplicaSet;
use orbitkv_state::{CacheOwner, ReplicaLocation, StateKey};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use tokio::sync::{Semaphore, mpsc};

fn metadata() -> orbitkv_state::ReplicaMetadata {
    orbitkv_state::ReplicaMetadata {
        medium: orbitkv_state::ReplicaMedium::Dram,
        representation: orbitkv_state::ReplicaRepresentation::Raw,
        stored_bytes: Some(4096),
    }
}

fn row(hash: u8, owners: &[&str]) -> ReplicaSet {
    let mut row = ReplicaSet::new(StateKey::new("ns".into(), vec![hash]));
    row.set_peers(
        owners
            .iter()
            .map(|owner| ReplicaLocation {
                owner: CacheOwner {
                    endpoint: (*owner).into(),
                    incarnation: uuid::Uuid::from_u128(1),
                },
                sequence: u64::from(hash),
                metadata: metadata(),
            })
            .collect(),
    );
    row
}

struct Fetcher {
    responses: Mutex<VecDeque<SegmentOutcome>>,
    calls: Mutex<Vec<String>>,
}
enum SegmentOutcome {
    Fetched(MaterializedBlocks),
    Rejected,
    Failed,
}

#[tonic::async_trait]
impl SegmentFetcher for Fetcher {
    type Block = Arc<SealedBlock>;
    type Grant = Result<MaterializedBlocks, ()>;

    async fn authorize_segment(
        &self,
        segment: &FetchSegment,
        _mode: AuthorizationMode,
    ) -> Result<Self::Grant, AuthorizationError> {
        self.calls
            .lock()
            .unwrap()
            .push(segment.owner.endpoint.clone());
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                SegmentOutcome::Fetched(
                    segment
                        .records
                        .iter()
                        .map(|r| (r.key.clone(), Arc::new(SealedBlock::from_slots(Vec::new()))))
                        .collect(),
                )
            });
        match response {
            SegmentOutcome::Fetched(blocks) => Ok(Ok(blocks)),
            SegmentOutcome::Rejected => Err(AuthorizationError::Rejected),
            SegmentOutcome::Failed => Ok(Err(())),
        }
    }

    async fn fetch_segment(
        &self,
        _segment: &FetchSegment,
        grant: Self::Grant,
        _req_id: &str,
        _destination_nodes: &[crate::memory::numa::NumaNode],
    ) -> Result<MaterializedBlocks, ()> {
        grant
    }
}

#[tokio::test]
async fn stale_candidate_uses_alternative_without_skipping_prefix_or_retrying_payload_failure() {
    for (response, expected) in [(SegmentOutcome::Rejected, 2), (SegmentOutcome::Failed, 0)] {
        let fetcher = Fetcher {
            responses: Mutex::new(VecDeque::from([response])),
            calls: Mutex::new(Vec::new()),
        };
        let mut rows = vec![row(1, &["a", "b"]), row(2, &["a", "b"])];
        let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
        let result = execute_fetch_plan(&fetcher, plan, "test-request", &[]).await;
        assert_eq!(result.blocks.len(), expected);
        assert_eq!(
            result.status,
            if expected == 2 {
                FetchStatus::Complete
            } else {
                FetchStatus::PayloadFailed
            }
        );
        assert_eq!(
            fetcher.calls.lock().unwrap().len(),
            if expected == 2 { 2 } else { 1 }
        );
    }
    let fetcher = Fetcher {
        responses: Mutex::new(VecDeque::from([
            SegmentOutcome::Rejected,
            SegmentOutcome::Rejected,
            SegmentOutcome::Rejected,
        ])),
        calls: Mutex::new(Vec::new()),
    };
    let mut rows = vec![row(1, &["a", "b", "c", "d"])];
    let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
    let result = execute_fetch_plan(&fetcher, plan, "test-request", &[]).await;
    assert!(result.blocks.is_empty());
    assert_eq!(result.status, FetchStatus::AuthorizationExhausted);
    assert!(result.can_replan());
    assert_eq!(fetcher.calls.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn malformed_or_short_segment_never_skips_a_gap() {
    let fetcher = Fetcher {
        responses: Mutex::new(VecDeque::from([SegmentOutcome::Fetched(vec![
            (
                StateKey::new("ns".into(), vec![1]),
                Arc::new(SealedBlock::from_slots(Vec::new())),
            ),
            (
                StateKey::new("wrong-model".into(), vec![2]),
                Arc::new(SealedBlock::from_slots(Vec::new())),
            ),
        ])])),
        calls: Mutex::new(Vec::new()),
    };
    let mut rows = vec![row(1, &["a"]), row(2, &["a"]), row(3, &["b"])];
    let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
    let result = execute_fetch_plan(&fetcher, plan, "test-request", &[]).await;
    assert_eq!(result.blocks.len(), 1);
    assert_eq!(result.status, FetchStatus::PayloadFailed);
    assert!(!result.can_replan());
    assert_eq!(*fetcher.calls.lock().unwrap(), ["a"]);
}

#[derive(Debug, PartialEq, Eq)]
enum PipelineEvent {
    Authorize(String, bool),
    Read(String),
}

struct Grant(Arc<AtomicUsize>);

impl Drop for Grant {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct PipelineFetcher {
    events: mpsc::UnboundedSender<PipelineEvent>,
    read_gate: Semaphore,
    grants: Arc<AtomicUsize>,
    peak_grants: AtomicUsize,
    fail_read: bool,
    reject_lookahead: AtomicBool,
}

#[tonic::async_trait]
impl SegmentFetcher for PipelineFetcher {
    type Block = Arc<SealedBlock>;
    type Grant = Grant;

    async fn authorize_segment(
        &self,
        segment: &FetchSegment,
        mode: AuthorizationMode,
    ) -> Result<Grant, AuthorizationError> {
        self.events
            .send(PipelineEvent::Authorize(
                segment.owner.endpoint.clone(),
                mode == AuthorizationMode::Lookahead,
            ))
            .unwrap();
        if mode == AuthorizationMode::Lookahead
            && self.reject_lookahead.swap(false, Ordering::SeqCst)
        {
            return Err(AuthorizationError::Rejected);
        }
        let held = self.grants.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_grants.fetch_max(held, Ordering::SeqCst);
        Ok(Grant(self.grants.clone()))
    }

    async fn fetch_segment(
        &self,
        segment: &FetchSegment,
        grant: Grant,
        _req_id: &str,
        _destination_nodes: &[crate::memory::numa::NumaNode],
    ) -> Result<MaterializedBlocks, ()> {
        self.events
            .send(PipelineEvent::Read(segment.owner.endpoint.clone()))
            .unwrap();
        self.read_gate.acquire().await.unwrap().forget();
        drop(grant);
        if self.fail_read {
            return Err(());
        }
        Ok(segment
            .records
            .iter()
            .map(|record| {
                (
                    record.key.clone(),
                    Arc::new(SealedBlock::from_slots(Vec::new())),
                )
            })
            .collect())
    }
}

fn pipeline(fail_read: bool) -> (Arc<PipelineFetcher>, mpsc::UnboundedReceiver<PipelineEvent>) {
    let (events, received) = mpsc::unbounded_channel();
    (
        Arc::new(PipelineFetcher {
            events,
            read_gate: Semaphore::new(0),
            grants: Arc::new(AtomicUsize::new(0)),
            peak_grants: AtomicUsize::new(0),
            fail_read,
            reject_lookahead: AtomicBool::new(false),
        }),
        received,
    )
}

async fn event(received: &mut mpsc::UnboundedReceiver<PipelineEvent>) -> PipelineEvent {
    tokio::time::timeout(std::time::Duration::from_secs(2), received.recv())
        .await
        .unwrap()
        .unwrap()
}

fn run_pipeline(
    fetcher: Arc<PipelineFetcher>,
    mut rows: Vec<ReplicaSet>,
) -> tokio::task::JoinHandle<FetchResult> {
    tokio::spawn(async move {
        let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
        execute_fetch_plan(fetcher.as_ref(), plan, "pipeline", &[]).await
    })
}

#[tokio::test]
async fn authorization_overlaps_read_with_only_one_segment_of_lookahead() {
    let (fetcher, mut events) = pipeline(false);
    // Three distinct owners force three contiguous segments.
    let task = run_pipeline(
        fetcher.clone(),
        vec![row(1, &["a"]), row(2, &["b"]), row(3, &["c"])],
    );
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("a".into(), false)
    );
    assert_eq!(event(&mut events).await, PipelineEvent::Read("a".into()));
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("b".into(), true)
    );
    assert!(
        events.try_recv().is_err(),
        "next READ and third grant must wait"
    );
    assert_eq!(fetcher.grants.load(Ordering::SeqCst), 2);
    fetcher.read_gate.add_permits(1);
    assert_eq!(event(&mut events).await, PipelineEvent::Read("b".into()));
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("c".into(), true)
    );
    assert!(events.try_recv().is_err());
    fetcher.read_gate.add_permits(1);
    assert_eq!(event(&mut events).await, PipelineEvent::Read("c".into()));
    fetcher.read_gate.add_permits(1);
    let result = task.await.unwrap();
    assert_eq!(result.status, FetchStatus::Complete);
    assert_eq!(result.attempts, 3);
    assert_eq!(result.completed_segments, 3);
    assert_eq!(fetcher.peak_grants.load(Ordering::SeqCst), 2);
    assert_eq!(fetcher.grants.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_or_cancelled_prefix_drops_unused_authorization_without_reading_it() {
    for cancel in [false, true] {
        let (fetcher, mut events) = pipeline(true);
        let task = run_pipeline(fetcher.clone(), vec![row(1, &["a"]), row(2, &["b"])]);
        for _ in 0..3 {
            event(&mut events).await;
        }
        assert_eq!(fetcher.grants.load(Ordering::SeqCst), 2);
        if cancel {
            task.abort();
            assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        } else {
            fetcher.read_gate.add_permits(1);
            let result = task.await.unwrap();
            assert_eq!(result.status, FetchStatus::PayloadFailed);
            assert!(result.blocks.is_empty());
        }
        assert!(
            events.try_recv().is_err(),
            "unused authorization must not submit a READ"
        );
        assert_eq!(fetcher.grants.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn speculative_admission_failure_retries_same_owner_after_current_read() {
    let (fetcher, mut events) = pipeline(false);
    fetcher.reject_lookahead.store(true, Ordering::SeqCst);
    let task = run_pipeline(fetcher.clone(), vec![row(1, &["x"]), row(2, &["a", "b"])]);
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("x".into(), false)
    );
    assert_eq!(event(&mut events).await, PipelineEvent::Read("x".into()));
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("a".into(), true)
    );
    fetcher.read_gate.add_permits(1);
    assert_eq!(
        event(&mut events).await,
        PipelineEvent::Authorize("a".into(), false)
    );
    assert_eq!(event(&mut events).await, PipelineEvent::Read("a".into()));
    fetcher.read_gate.add_permits(1);
    let result = task.await.unwrap();
    assert_eq!(result.status, FetchStatus::Complete);
    assert_eq!(result.attempts, 3);
    assert_eq!(fetcher.grants.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn pipelined_execution_preserves_prefix_and_owner_fallback() {
    let fetcher = Fetcher {
        responses: Mutex::new(VecDeque::from([SegmentOutcome::Rejected])),
        calls: Mutex::new(Vec::new()),
    };
    let mut rows = vec![row(1, &["a", "b"]), row(2, &["a", "b"]), row(3, &["c"])];
    let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
    let result = execute_fetch_plan(&fetcher, plan, "owner-fallback", &[]).await;
    assert_eq!(result.status, FetchStatus::Complete);
    assert_eq!(result.attempts, 3);
    assert_eq!(result.completed_segments, 2);
    assert_eq!(
        result
            .blocks
            .iter()
            .map(|(key, _)| key.hash.clone())
            .collect::<Vec<_>>(),
        [vec![1], vec![2], vec![3]],
    );
    assert_eq!(*fetcher.calls.lock().unwrap(), ["a", "b", "c"]);
}

#[tokio::test]
async fn demand_rejection_after_speculation_falls_back_without_losing_completed_prefix() {
    let blocks = |hash| {
        vec![(
            StateKey::new("ns".into(), vec![hash]),
            Arc::new(SealedBlock::from_slots(Vec::new())),
        )]
    };
    let fetcher = Fetcher {
        responses: Mutex::new(VecDeque::from([
            SegmentOutcome::Fetched(blocks(1)),
            // A release can still be awaiting acknowledgement when the next
            // segment retries on demand; ordinary bounded fallback still runs.
            SegmentOutcome::Rejected,
            SegmentOutcome::Rejected,
            SegmentOutcome::Fetched(blocks(2)),
        ])),
        calls: Mutex::new(Vec::new()),
    };
    let mut rows = vec![row(1, &["x"]), row(2, &["a", "b"])];
    let plan = FetchPlan::new(&mut rows, 1, crate::planning::peer::PeerSource::Dram).unwrap();
    let result = execute_fetch_plan(&fetcher, plan, "busy-source", &[]).await;
    assert_eq!(result.status, FetchStatus::Complete);
    assert_eq!(result.blocks.len(), 2);
    assert_eq!(result.attempts, 4);
    assert_eq!(*fetcher.calls.lock().unwrap(), ["x", "a", "a", "b"]);
}
