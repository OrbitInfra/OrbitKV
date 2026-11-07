use std::collections::{BTreeMap, HashMap, VecDeque};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use orbitkv_catalog::{DeltaApply, GlobalIndex, MembershipView};
use orbitkv_core::{InventoryReadError, ResidencyInventory};
use orbitkv_proto::proto::engine::inventory_client::InventoryClient;
use orbitkv_proto::proto::engine::inventory_server::Inventory;
use orbitkv_proto::proto::engine::{
    InventoryAck, InventoryClientFrame, InventoryDelta, InventoryFlushThrough, InventoryOpen,
    InventoryProgress, InventoryResetRequired, InventoryScopeMode, InventoryServerFrame,
    InventorySnapshotBegin, InventorySnapshotCommit, InventorySnapshotPage, inventory_client_frame,
    inventory_server_frame,
};
use orbitkv_state::{
    CacheOwner, INVENTORY_BATCH_RECORDS, INVENTORY_OPEN_MAX_BYTES, INVENTORY_STREAM_PROTOCOL,
    InventoryFence, InventoryScope,
};
use parking_lot::{Mutex, RwLock};
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming, async_trait};
use uuid::Uuid;

use super::Member;
use super::format::ClusterFormat;

const FRAME_BYTES: usize = 512 * 1024;
const MESSAGE_BYTES: usize = 1024 * 1024;
const STREAM_CREDIT_BYTES: usize = 1024 * 1024;
const AGGREGATE_CREDIT_BYTES: usize = 32 * 1024 * 1024;
const MAX_SOURCE_SESSIONS: usize = 128;
const MAX_RECEIVER_SESSIONS: usize = 128;
const SNAPSHOT_SESSIONS: usize = 2;
const OUTBOUND_BYTES_PER_SECOND: usize = 16 * 1024 * 1024;
const OUTBOUND_BURST_BYTES: usize = 1024 * 1024;
const ACK_TIMEOUT: Duration = Duration::from_secs(3);
const IDLE_PROGRESS: Duration = Duration::from_secs(1);
const RETIRE_BATCH: usize = 1024;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct InventoryRuntimeStatus {
    pub protocol: String,
    pub cluster_uuid: Uuid,
    pub source_node: String,
    pub source_node_epoch: u64,
    pub source_incarnation: Uuid,
    pub scope_digest: String,
    pub scope_kind: &'static str,
    pub scope_namespaces: usize,
    pub membership_revision: Option<i64>,
    pub source_sessions: u64,
    pub source_sessions_peak: u64,
    pub receiver_sessions: u64,
    pub receiver_sessions_peak: u64,
    pub outbound_queue_bytes: usize,
    pub outbound_queue_bytes_peak: u64,
    pub frames_sent: u64,
    pub frames_received: u64,
    pub encoded_bytes_sent: u64,
    pub encoded_bytes_received: u64,
    pub resets: u64,
}

#[derive(Default)]
struct Counters {
    source_sessions: AtomicU64,
    source_sessions_peak: AtomicU64,
    receiver_sessions: AtomicU64,
    receiver_sessions_peak: AtomicU64,
    outbound_queue_bytes_peak: AtomicU64,
    frames_sent: AtomicU64,
    frames_received: AtomicU64,
    encoded_bytes_sent: AtomicU64,
    encoded_bytes_received: AtomicU64,
    resets: AtomicU64,
}

#[derive(Clone, Copy)]
enum SessionDirection {
    Source,
    Receiver,
}

struct ActiveSession {
    shared: Arc<Shared>,
    direction: SessionDirection,
}

impl ActiveSession {
    fn new(shared: Arc<Shared>, direction: SessionDirection) -> Self {
        let (active, peak) = match direction {
            SessionDirection::Source => (
                &shared.counters.source_sessions,
                &shared.counters.source_sessions_peak,
            ),
            SessionDirection::Receiver => (
                &shared.counters.receiver_sessions,
                &shared.counters.receiver_sessions_peak,
            ),
        };
        let active = active.fetch_add(1, Ordering::Relaxed) + 1;
        peak.fetch_max(active, Ordering::Relaxed);
        Self { shared, direction }
    }
}

impl Drop for ActiveSession {
    fn drop(&mut self) {
        let active = match self.direction {
            SessionDirection::Source => &self.shared.counters.source_sessions,
            SessionDirection::Receiver => &self.shared.counters.receiver_sessions,
        };
        active.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct MemberState {
    revision: Option<i64>,
    members: BTreeMap<String, Member>,
}

struct Shared {
    format: ClusterFormat,
    own: Member,
    inventory: Arc<ResidencyInventory>,
    index: Arc<GlobalIndex>,
    membership: Arc<MembershipView>,
    scope: Arc<InventoryScope>,
    members: RwLock<MemberState>,
    member_changes: watch::Sender<u64>,
    controls: Mutex<HashMap<Uuid, (Uuid, mpsc::Sender<InventoryClientFrame>)>>,
    force_bootstrap: Mutex<std::collections::HashSet<Uuid>>,
    source_sessions: Arc<Semaphore>,
    receiver_sessions: Arc<Semaphore>,
    served_snapshots: Arc<Semaphore>,
    received_snapshots: Arc<Semaphore>,
    outbound_bytes: Arc<Semaphore>,
    outbound_pacer: Arc<tokio::sync::Mutex<tokio::time::Instant>>,
    awaiters: Arc<Semaphore>,
    counters: Counters,
}

#[derive(Clone)]
pub(crate) struct InventoryRuntime {
    shared: Arc<Shared>,
}

impl InventoryRuntime {
    pub(super) fn new(
        format: ClusterFormat,
        own: Member,
        inventory: Arc<ResidencyInventory>,
        index: Arc<GlobalIndex>,
        membership: Arc<MembershipView>,
    ) -> Self {
        let scope = index.scope();
        Self {
            shared: Arc::new(Shared {
                format,
                own,
                inventory,
                index,
                membership,
                scope,
                members: RwLock::new(MemberState::default()),
                member_changes: watch::channel(0).0,
                controls: Mutex::new(HashMap::new()),
                force_bootstrap: Mutex::new(std::collections::HashSet::new()),
                source_sessions: Arc::new(Semaphore::new(MAX_SOURCE_SESSIONS)),
                receiver_sessions: Arc::new(Semaphore::new(MAX_RECEIVER_SESSIONS)),
                served_snapshots: Arc::new(Semaphore::new(SNAPSHOT_SESSIONS)),
                received_snapshots: Arc::new(Semaphore::new(SNAPSHOT_SESSIONS)),
                outbound_bytes: Arc::new(Semaphore::new(AGGREGATE_CREDIT_BYTES)),
                outbound_pacer: Arc::new(tokio::sync::Mutex::new(
                    tokio::time::Instant::now()
                        - Duration::from_secs_f64(
                            OUTBOUND_BURST_BYTES as f64 / OUTBOUND_BYTES_PER_SECOND as f64,
                        ),
                )),
                awaiters: Arc::new(Semaphore::new(64)),
                counters: Counters::default(),
            }),
        }
    }

    pub(super) fn replace_members(&self, revision: i64, members: BTreeMap<String, Member>) {
        let previous = {
            let mut state = self.shared.members.write();
            let previous = std::mem::replace(&mut state.members, members.clone());
            state.revision = Some(revision);
            previous
        };
        let owners = members
            .values()
            .map(|member| member.owner.incarnation)
            .collect::<Vec<_>>();
        self.shared.index.set_expected_owners(revision, owners);
        for old in previous.values() {
            if !members.values().any(|member| member == old) {
                self.retire(old.owner.incarnation);
            }
        }
        self.shared.member_changes.send_replace(revision as u64);
    }

    pub(super) fn membership_unavailable(&self) {
        self.shared.member_changes.send_modify(|generation| {
            *generation = generation.saturating_add(1);
        });
    }

    fn retire(&self, owner: Uuid) {
        self.shared.index.retire_owner(owner);
        self.shared.controls.lock().remove(&owner);
        let index = Arc::clone(&self.shared.index);
        tokio::spawn(async move {
            while !index.cleanup_owner(owner, RETIRE_BATCH) {
                tokio::task::yield_now().await;
            }
        });
    }

    fn withdraw_for_rebuild(&self, owner: Uuid) {
        self.shared.index.retire_owner(owner);
        let index = Arc::clone(&self.shared.index);
        tokio::spawn(async move {
            while !index.cleanup_owner(owner, RETIRE_BATCH) {
                tokio::task::yield_now().await;
            }
        });
    }

    fn members(&self) -> BTreeMap<String, Member> {
        self.shared.members.read().members.clone()
    }

    fn member(&self, owner: Uuid) -> Option<Member> {
        let member = self
            .shared
            .members
            .read()
            .members
            .values()
            .find(|member| member.owner.incarnation == owner)
            .cloned()?;
        self.shared
            .membership
            .permits(&member.owner)
            .then_some(member)
    }

    fn exact_member(&self, member: &Member) -> bool {
        self.shared.membership.permits(&member.owner)
            && self.shared.members.read().members.get(&member.node_id) == Some(member)
    }

    pub(crate) fn service(&self) -> InventoryService {
        InventoryService {
            runtime: self.clone(),
        }
    }

    pub(crate) fn capture_fence(&self) -> Result<InventoryFence, String> {
        if self.member(self.shared.own.owner.incarnation).as_ref() != Some(&self.shared.own) {
            return Err("local inventory membership is not installed".into());
        }
        Ok(InventoryFence {
            protocol: INVENTORY_STREAM_PROTOCOL.into(),
            cluster_uuid: self.shared.format.cluster_uuid,
            source_node_epoch: self.shared.own.epoch,
            source_incarnation: self.shared.own.owner.incarnation,
            inventory_sequence: self.shared.inventory.capture_fence(),
        })
    }

    pub(crate) async fn await_fence(
        &self,
        fence: &InventoryFence,
        scope_digest: &[u8],
        timeout: Duration,
    ) -> Result<(), String> {
        if timeout.is_zero() || timeout > Duration::from_secs(30) {
            return Err("inventory await timeout must be in 1ns..=30s".into());
        }
        if scope_digest != self.shared.scope.digest() {
            return Err("requester scope does not match the installed inventory scope".into());
        }
        if fence.protocol != INVENTORY_STREAM_PROTOCOL
            || fence.cluster_uuid != self.shared.format.cluster_uuid
            || fence.source_incarnation.is_nil()
            || fence.source_node_epoch == 0
        {
            return Err("inventory fence identity differs".into());
        }
        let source = self
            .member(fence.source_incarnation)
            .ok_or("inventory fence source is not a live member")?;
        if source.epoch != fence.source_node_epoch {
            return Err("inventory fence source epoch changed".into());
        }
        let _permit = self
            .shared
            .awaiters
            .clone()
            .try_acquire_owned()
            .map_err(|_| "inventory await admission limit reached")?;
        self.request_flush(source.owner.incarnation, fence.inventory_sequence);
        tokio::time::timeout(timeout, async {
            loop {
                if !self.exact_member(&source) {
                    return Err("inventory fence source membership changed".into());
                }
                if self
                    .shared
                    .index
                    .owner_status(source.owner.incarnation)
                    .is_some_and(|status| {
                        status.fresh && status.applied_sequence >= fence.inventory_sequence
                    })
                {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .map_err(|_| "inventory fence await timed out".to_string())?
    }

    fn request_flush(&self, owner: Uuid, target: u64) {
        let sender = self
            .shared
            .controls
            .lock()
            .get(&owner)
            .map(|(_, sender)| sender.clone());
        if let Some(sender) = sender {
            let _ = sender.try_send(InventoryClientFrame {
                body: Some(inventory_client_frame::Body::FlushThrough(
                    InventoryFlushThrough {
                        target_sequence: target,
                    },
                )),
            });
        }
    }

    pub(crate) fn status(&self) -> InventoryRuntimeStatus {
        let counters = &self.shared.counters;
        InventoryRuntimeStatus {
            protocol: INVENTORY_STREAM_PROTOCOL.into(),
            cluster_uuid: self.shared.format.cluster_uuid,
            source_node: self.shared.own.node_id.clone(),
            source_node_epoch: self.shared.own.epoch,
            source_incarnation: self.shared.own.owner.incarnation,
            scope_digest: hex(&self.shared.scope.digest()),
            scope_kind: self.shared.scope.label(),
            scope_namespaces: self.shared.scope.namespaces().map_or(0, <[String]>::len),
            membership_revision: self.shared.members.read().revision,
            source_sessions: counters.source_sessions.load(Ordering::Relaxed),
            source_sessions_peak: counters.source_sessions_peak.load(Ordering::Relaxed),
            receiver_sessions: counters.receiver_sessions.load(Ordering::Relaxed),
            receiver_sessions_peak: counters.receiver_sessions_peak.load(Ordering::Relaxed),
            outbound_queue_bytes: AGGREGATE_CREDIT_BYTES
                - self.shared.outbound_bytes.available_permits(),
            outbound_queue_bytes_peak: counters.outbound_queue_bytes_peak.load(Ordering::Relaxed),
            frames_sent: counters.frames_sent.load(Ordering::Relaxed),
            frames_received: counters.frames_received.load(Ordering::Relaxed),
            encoded_bytes_sent: counters.encoded_bytes_sent.load(Ordering::Relaxed),
            encoded_bytes_received: counters.encoded_bytes_received.load(Ordering::Relaxed),
            resets: counters.resets.load(Ordering::Relaxed),
        }
    }
}

pub(crate) struct InventoryService {
    runtime: InventoryRuntime,
}

type InventoryResponseStream =
    Pin<Box<dyn Stream<Item = Result<InventoryServerFrame, Status>> + Send + 'static>>;

#[async_trait]
impl Inventory for InventoryService {
    type InventorySessionStream = InventoryResponseStream;

    async fn inventory_session(
        &self,
        request: Request<Streaming<InventoryClientFrame>>,
    ) -> Result<Response<Self::InventorySessionStream>, Status> {
        let permit = self
            .runtime
            .shared
            .source_sessions
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("source inventory session limit reached"))?;
        let mut input = request.into_inner();
        let first = tokio::time::timeout(Duration::from_secs(3), input.message())
            .await
            .map_err(|_| Status::deadline_exceeded("inventory Open timed out"))?
            .map_err(|error| Status::invalid_argument(error.to_string()))?
            .ok_or_else(|| Status::invalid_argument("inventory session lacks Open"))?;
        let Some(inventory_client_frame::Body::Open(open)) = first.body else {
            return Err(Status::invalid_argument(
                "inventory session must start with Open",
            ));
        };
        let open = *open;
        let identity = validate_open(&self.runtime, &open)?;
        let (output, response) = mpsc::channel(4);
        let runtime = self.runtime.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _active = ActiveSession::new(runtime.shared.clone(), SessionDirection::Source);
            if let Err(error) = serve_source(runtime.clone(), identity, open, input, output).await {
                log::warn!("Inventory source session stopped: {error}");
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(response))))
    }
}

struct OpenIdentity {
    source: Member,
    requester: CacheOwner,
    session_id: Uuid,
    scope: InventoryScope,
}

fn validate_open(runtime: &InventoryRuntime, open: &InventoryOpen) -> Result<OpenIdentity, Status> {
    if open.encoded_len() > INVENTORY_OPEN_MAX_BYTES {
        return Err(Status::resource_exhausted(
            "inventory Open exceeds the encoded byte limit",
        ));
    }
    let cluster_uuid = uuid_from(&open.cluster_uuid, "cluster UUID")?;
    let source_incarnation = uuid_from(&open.source_incarnation, "source incarnation")?;
    let requester_incarnation = uuid_from(&open.requester_incarnation, "requester incarnation")?;
    let session_id = uuid_from(&open.session_id, "session ID")?;
    let scope = decode_scope(open)?;
    if open.protocol != INVENTORY_STREAM_PROTOCOL
        || cluster_uuid != runtime.shared.format.cluster_uuid
        || open.scope_digest != scope.digest()
        || open.source_node != runtime.shared.own.node_id
        || open.source_epoch != runtime.shared.own.epoch
        || source_incarnation != runtime.shared.own.owner.incarnation
    {
        return Err(Status::failed_precondition(
            "inventory Open source, cluster or scope identity differs",
        ));
    }
    let requester = runtime
        .member(requester_incarnation)
        .ok_or_else(|| Status::permission_denied("requester is not a live member"))?;
    Ok(OpenIdentity {
        source: runtime.shared.own.clone(),
        requester: requester.owner,
        session_id,
        scope,
    })
}

#[derive(Clone, Default)]
struct ClientControl {
    consumed_frame_id: u64,
    installed_view_id: Option<Uuid>,
    applied_sequence: u64,
    flush_through: u64,
    invalid: Option<String>,
}

async fn serve_source(
    runtime: InventoryRuntime,
    identity: OpenIdentity,
    open: InventoryOpen,
    mut input: Streaming<InventoryClientFrame>,
    output: mpsc::Sender<Result<InventoryServerFrame, Status>>,
) -> Result<(), String> {
    let (control, mut controls) = watch::channel(ClientControl::default());
    let runtime_for_input = runtime.clone();
    let requester = identity.requester.clone();
    let input_task = tokio::spawn(async move {
        while let Some(frame) = input.message().await.map_err(|error| error.to_string())? {
            match frame.body {
                Some(inventory_client_frame::Body::Ack(ack)) => {
                    let installed = optional_uuid(&ack.installed_view_id, "installed view ID")?;
                    control.send_modify(|state| {
                        if ack.consumed_frame_id < state.consumed_frame_id
                            || ack.applied_sequence < state.applied_sequence
                        {
                            state.invalid =
                                Some("inventory acknowledgement moved backwards".into());
                        } else {
                            state.consumed_frame_id = ack.consumed_frame_id;
                            state.installed_view_id = installed;
                            state.applied_sequence = ack.applied_sequence;
                        }
                    });
                }
                Some(inventory_client_frame::Body::FlushThrough(flush)) => {
                    runtime_for_input
                        .shared
                        .inventory
                        .request_flush(flush.target_sequence);
                    control.send_modify(|state| {
                        state.flush_through = state.flush_through.max(flush.target_sequence);
                    });
                }
                Some(inventory_client_frame::Body::Open(_)) | None => {
                    control.send_modify(|state| {
                        state.invalid = Some("duplicate or empty inventory control frame".into());
                    });
                    break;
                }
            }
            if runtime_for_input.member(requester.incarnation).is_none() {
                break;
            }
        }
        Ok::<(), String>(())
    });
    let input_task = AbortOnDrop(Some(input_task));

    let mut sender = FrameSender::new(
        runtime.clone(),
        identity.session_id,
        output,
        controls.clone(),
    );
    let resume = match (
        open.resume_sequence,
        optional_uuid(&open.installed_view_id, "installed view ID")?,
    ) {
        (Some(sequence), Some(view_id)) if sequence <= runtime.shared.inventory.sequence() => {
            Some((sequence, view_id))
        }
        _ => None,
    };
    let mut after = if let Some((sequence, _)) = resume {
        match runtime.shared.inventory.coalesced_changes_scoped(
            sequence,
            runtime.shared.inventory.sequence(),
            &identity.scope,
        ) {
            Ok(_) => sequence,
            Err(InventoryReadError::HistoryGap) => {
                sender
                    .reset("source journal history is unavailable")
                    .await?;
                return Ok(());
            }
            Err(error) => return Err(format!("inventory resume: {error:?}")),
        }
    } else {
        serve_snapshot(&runtime, &mut sender, &identity.scope).await?
    };

    let mut changes = runtime.shared.inventory.changed();
    loop {
        sender.release_acknowledged()?;
        if !runtime.exact_member(&identity.source)
            || runtime.member(identity.requester.incarnation).is_none()
        {
            break;
        }
        if controls.borrow().invalid.is_some() {
            return Err(controls
                .borrow()
                .invalid
                .clone()
                .unwrap_or_else(|| "invalid client control".into()));
        }
        let head = runtime.shared.inventory.sequence();
        if head > after {
            let waited = runtime.shared.inventory.wait_to_publish(after).await;
            let head = runtime.shared.inventory.sequence();
            let delta = match bounded_delta(&runtime.shared.inventory, after, head, &identity.scope)
            {
                Ok(delta) => delta,
                Err(InventoryReadError::HistoryGap) => {
                    sender
                        .reset("source journal history is unavailable")
                        .await?;
                    break;
                }
                Err(error) => return Err(format!("inventory delta: {error:?}")),
            };
            let frame = sender.frame(inventory_server_frame::Body::Delta(InventoryDelta {
                from_exclusive: after,
                through_inclusive: delta.through,
                records: delta.records.clone().into_iter().map(Into::into).collect(),
            }))?;
            let encoded = frame.encoded_len();
            sender.send(frame).await?;
            runtime.shared.inventory.record_delta_stream(
                delta.input_records,
                delta.input_bytes,
                delta.records.len(),
                1,
                encoded,
                waited,
            );
            after = delta.through;
            continue;
        }
        tokio::select! {
            result = changes.changed() => {
                if result.is_err() {
                    break;
                }
            }
            result = controls.changed() => {
                if result.is_err() {
                    break;
                }
            }
            _ = tokio::time::sleep(IDLE_PROGRESS) => {
                let frame = sender.frame(inventory_server_frame::Body::Progress(
                    InventoryProgress {
                        through_sequence: after,
                        source_head_sequence: runtime.shared.inventory.sequence(),
                    },
                ))?;
                sender.send(frame).await?;
            }
        }
    }
    drop(input_task);
    Ok(())
}

async fn serve_snapshot(
    runtime: &InventoryRuntime,
    sender: &mut FrameSender,
    scope: &InventoryScope,
) -> Result<u64, String> {
    let _permit = runtime
        .shared
        .served_snapshots
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| "snapshot admission stopped")?;
    let snapshot_id = Uuid::new_v4();
    let view_id = Uuid::new_v4();
    let start = runtime.shared.inventory.sequence();
    let mut transcript = Sha256::new();
    let begin = sender.frame(inventory_server_frame::Body::SnapshotBegin(
        InventorySnapshotBegin {
            snapshot_id: snapshot_id.as_bytes().to_vec(),
            view_id: view_id.as_bytes().to_vec(),
            start_sequence: start,
        },
    ))?;
    transcript.update(begin.encode_to_vec());
    sender.send(begin).await?;
    let mut cursor = None;
    let mut page_number = 0;
    loop {
        let page = runtime
            .shared
            .inventory
            .scoped_page(cursor.as_ref(), scope)
            .map_err(|error| format!("inventory snapshot page: {error:?}"))?;
        cursor = page.next;
        if !page.records.is_empty() {
            let frame = InventorySnapshotPage {
                snapshot_id: snapshot_id.as_bytes().to_vec(),
                page_number,
                records: page.records.into_iter().map(Into::into).collect(),
            };
            if frame.encoded_len() > FRAME_BYTES {
                return Err("inventory snapshot page exceeds the encoded frame limit".into());
            }
            let frame = sender.frame(inventory_server_frame::Body::SnapshotPage(frame))?;
            transcript.update(frame.encode_to_vec());
            sender.send(frame).await?;
            page_number += 1;
        }
        if page.complete {
            break;
        }
        tokio::task::yield_now().await;
    }
    let end = runtime.shared.inventory.sequence();
    let mut after = start;
    while after < end {
        let delta = match bounded_delta(&runtime.shared.inventory, after, end, scope) {
            Ok(delta) => delta,
            Err(InventoryReadError::HistoryGap) => {
                sender
                    .reset("source journal overflowed during snapshot")
                    .await?;
                return Err("inventory snapshot replay history is unavailable".into());
            }
            Err(error) => return Err(format!("inventory snapshot replay: {error:?}")),
        };
        let frame = sender.frame(inventory_server_frame::Body::Delta(InventoryDelta {
            from_exclusive: after,
            through_inclusive: delta.through,
            records: delta.records.into_iter().map(Into::into).collect(),
        }))?;
        transcript.update(frame.encode_to_vec());
        sender.send(frame).await?;
        after = delta.through;
    }
    let commit = sender.frame(inventory_server_frame::Body::SnapshotCommit(
        InventorySnapshotCommit {
            snapshot_id: snapshot_id.as_bytes().to_vec(),
            view_id: view_id.as_bytes().to_vec(),
            through_sequence: end,
            transcript_digest: transcript.finalize().to_vec(),
            page_count: page_number,
        },
    ))?;
    sender.send(commit).await?;
    Ok(end)
}

struct OutstandingFrame {
    frame_id: u64,
    bytes: usize,
    _global: tokio::sync::OwnedSemaphorePermit,
}

struct AbortOnDrop<T>(Option<JoinHandle<T>>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}

struct FrameSender {
    runtime: InventoryRuntime,
    session_id: Uuid,
    output: mpsc::Sender<Result<InventoryServerFrame, Status>>,
    controls: watch::Receiver<ClientControl>,
    next_frame: u64,
    outstanding_bytes: usize,
    outstanding: VecDeque<OutstandingFrame>,
}

impl FrameSender {
    fn new(
        runtime: InventoryRuntime,
        session_id: Uuid,
        output: mpsc::Sender<Result<InventoryServerFrame, Status>>,
        controls: watch::Receiver<ClientControl>,
    ) -> Self {
        Self {
            runtime,
            session_id,
            output,
            controls,
            next_frame: 1,
            outstanding_bytes: 0,
            outstanding: VecDeque::new(),
        }
    }

    fn frame(
        &mut self,
        body: inventory_server_frame::Body,
    ) -> Result<InventoryServerFrame, String> {
        let frame = InventoryServerFrame {
            frame_id: self.next_frame,
            session_id: self.session_id.as_bytes().to_vec(),
            body: Some(body),
        };
        self.next_frame = self
            .next_frame
            .checked_add(1)
            .ok_or("inventory frame ID exhausted")?;
        if frame.encoded_len() > MESSAGE_BYTES {
            return Err("inventory frame exceeds encoded message limit".into());
        }
        Ok(frame)
    }

    async fn send(&mut self, frame: InventoryServerFrame) -> Result<(), String> {
        if !self.runtime.shared.membership.registration_valid() {
            return Err("inventory source membership is fenced".into());
        }
        let bytes = frame.encoded_len();
        while self.outstanding_bytes + bytes > STREAM_CREDIT_BYTES {
            self.release_acknowledged()?;
            if self.outstanding_bytes + bytes <= STREAM_CREDIT_BYTES {
                break;
            }
            tokio::time::timeout(ACK_TIMEOUT, self.controls.changed())
                .await
                .map_err(|_| "inventory frame credit acknowledgement timed out")?
                .map_err(|_| "inventory client control stream closed")?;
        }
        self.release_acknowledged()?;
        pace_outbound(&self.runtime.shared.outbound_pacer, bytes).await;
        let permits = u32::try_from(bytes).map_err(|_| "inventory frame size overflow")?;
        let global = tokio::time::timeout(
            ACK_TIMEOUT,
            self.runtime
                .shared
                .outbound_bytes
                .clone()
                .acquire_many_owned(permits),
        )
        .await
        .map_err(|_| "inventory aggregate byte admission timed out")?
        .map_err(|_| "inventory aggregate byte budget closed")?;
        let queued =
            AGGREGATE_CREDIT_BYTES - self.runtime.shared.outbound_bytes.available_permits();
        self.runtime
            .shared
            .counters
            .outbound_queue_bytes_peak
            .fetch_max(queued as u64, Ordering::Relaxed);
        let frame_id = frame.frame_id;
        tokio::time::timeout(ACK_TIMEOUT, self.output.send(Ok(frame)))
            .await
            .map_err(|_| "inventory response queue timed out")?
            .map_err(|_| "inventory response stream closed")?;
        self.outstanding_bytes += bytes;
        self.outstanding.push_back(OutstandingFrame {
            frame_id,
            bytes,
            _global: global,
        });
        self.runtime
            .shared
            .counters
            .frames_sent
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .shared
            .counters
            .encoded_bytes_sent
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    fn release_acknowledged(&mut self) -> Result<(), String> {
        let control = self.controls.borrow().clone();
        if let Some(error) = control.invalid {
            return Err(error);
        }
        if control.consumed_frame_id >= self.next_frame {
            return Err("inventory client acknowledged an unsent frame".into());
        }
        while self
            .outstanding
            .front()
            .is_some_and(|frame| frame.frame_id <= control.consumed_frame_id)
        {
            let frame = self
                .outstanding
                .pop_front()
                .ok_or("missing acknowledged inventory frame")?;
            self.outstanding_bytes -= frame.bytes;
        }
        Ok(())
    }

    async fn reset(&mut self, reason: &str) -> Result<(), String> {
        self.runtime
            .shared
            .counters
            .resets
            .fetch_add(1, Ordering::Relaxed);
        let frame = self.frame(inventory_server_frame::Body::ResetRequired(
            InventoryResetRequired {
                reason: reason.into(),
            },
        ))?;
        self.send(frame).await
    }
}

async fn pace_outbound(next_send: &tokio::sync::Mutex<tokio::time::Instant>, bytes: usize) {
    let now = tokio::time::Instant::now();
    let burst =
        Duration::from_secs_f64(OUTBOUND_BURST_BYTES as f64 / OUTBOUND_BYTES_PER_SECOND as f64);
    let duration = Duration::from_secs_f64(bytes as f64 / OUTBOUND_BYTES_PER_SECOND as f64);
    let ready = {
        let mut next = next_send.lock().await;
        if *next + burst < now {
            *next = now - burst;
        }
        let ready = *next;
        *next += duration;
        ready
    };
    if ready > now {
        tokio::time::sleep_until(ready).await;
    }
}

pub(super) async fn follow_members(runtime: InventoryRuntime, mut stop: watch::Receiver<bool>) {
    let mut changed = runtime.shared.member_changes.subscribe();
    let mut workers = HashMap::<Uuid, (Member, JoinHandle<()>)>::new();
    loop {
        let members = runtime.members();
        let current = members
            .values()
            .filter(|member| member.owner.incarnation != runtime.shared.own.owner.incarnation)
            .map(|member| (member.owner.incarnation, member.clone()))
            .collect::<HashMap<_, _>>();
        let stopped = workers
            .iter()
            .filter(|(owner, (member, task))| {
                task.is_finished() || current.get(owner) != Some(member)
            })
            .map(|(owner, _)| *owner)
            .collect::<Vec<_>>();
        for owner in stopped {
            if let Some((_, task)) = workers.remove(&owner) {
                task.abort();
            }
        }
        for (owner, member) in current {
            workers.entry(owner).or_insert_with(|| {
                let follower = runtime.clone();
                let task_member = member.clone();
                let task = tokio::spawn(async move {
                    follow_source(follower, task_member).await;
                });
                (member, task)
            });
        }
        tokio::select! {
            result = changed.changed() => {
                if result.is_err() {
                    break;
                }
            }
            result = stop.changed() => {
                if result.is_err() || *stop.borrow() {
                    break;
                }
            }
        }
    }
    for (_, (_, task)) in workers {
        task.abort();
    }
}

async fn follow_source(runtime: InventoryRuntime, source: Member) {
    let _permit = match runtime
        .shared
        .receiver_sessions
        .clone()
        .acquire_owned()
        .await
    {
        Ok(permit) => permit,
        Err(_) => return,
    };
    let _active = ActiveSession::new(runtime.shared.clone(), SessionDirection::Receiver);
    let mut delay = Duration::from_millis(100);
    while runtime.exact_member(&source) {
        if let Err(error) = follow_source_once(runtime.clone(), source.clone()).await {
            log::warn!(
                "Inventory receiver for node={} incarnation={} will reconnect: {error}",
                source.node_id,
                source.owner.incarnation
            );
            runtime.shared.index.mark_stale(source.owner.incarnation);
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(Duration::from_secs(3));
    }
}

async fn follow_source_once(runtime: InventoryRuntime, source: Member) -> Result<(), String> {
    let session_id = Uuid::new_v4();
    let endpoint = format!("http://{}", source.owner.endpoint);
    let channel = tonic::transport::Endpoint::from_shared(endpoint)
        .map_err(|error| error.to_string())?
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(5))
        .connect()
        .await
        .map_err(|error| error.to_string())?;
    let mut client = InventoryClient::new(channel)
        .max_decoding_message_size(MESSAGE_BYTES)
        .max_encoding_message_size(MESSAGE_BYTES);
    let (control, request) = mpsc::channel(64);
    let installed = if runtime
        .shared
        .force_bootstrap
        .lock()
        .contains(&source.owner.incarnation)
    {
        None
    } else {
        runtime
            .shared
            .index
            .owner_watermark(source.owner.incarnation)
    };
    let (scope_mode, scope_namespaces) = encode_scope(&runtime.shared.scope);
    let open = InventoryOpen {
        protocol: INVENTORY_STREAM_PROTOCOL.into(),
        cluster_uuid: runtime.shared.format.cluster_uuid.as_bytes().to_vec(),
        source_node: source.node_id.clone(),
        source_epoch: source.epoch,
        source_incarnation: source.owner.incarnation.as_bytes().to_vec(),
        requester_incarnation: runtime.shared.own.owner.incarnation.as_bytes().to_vec(),
        scope_digest: runtime.shared.scope.digest().to_vec(),
        session_id: session_id.as_bytes().to_vec(),
        resume_sequence: installed.map(|(_, sequence)| sequence),
        installed_view_id: installed
            .map(|(view_id, _)| view_id.as_bytes().to_vec())
            .unwrap_or_default(),
        scope_mode,
        scope_namespaces,
    };
    if open.encoded_len() > INVENTORY_OPEN_MAX_BYTES {
        return Err("inventory Open exceeds the encoded byte limit".into());
    }
    control
        .send(InventoryClientFrame {
            body: Some(inventory_client_frame::Body::Open(Box::new(open))),
        })
        .await
        .map_err(|_| "inventory request stream closed before Open")?;
    let mut response = client
        .inventory_session(ReceiverStream::new(request))
        .await
        .map_err(|error| error.to_string())?
        .into_inner();
    runtime
        .shared
        .controls
        .lock()
        .insert(source.owner.incarnation, (session_id, control.clone()));
    let result = receive_frames(&runtime, &source, session_id, &control, &mut response).await;
    if result
        .as_ref()
        .is_err_and(|error| error.contains("metadata budget"))
    {
        runtime.withdraw_for_rebuild(source.owner.incarnation);
    }
    if runtime
        .shared
        .controls
        .lock()
        .get(&source.owner.incarnation)
        .is_some_and(|(registered, _)| *registered == session_id)
    {
        runtime
            .shared
            .controls
            .lock()
            .remove(&source.owner.incarnation);
    }
    runtime
        .shared
        .index
        .abort_snapshot(source.owner.incarnation, session_id);
    result
}

async fn receive_frames(
    runtime: &InventoryRuntime,
    source: &Member,
    session_id: Uuid,
    control: &mpsc::Sender<InventoryClientFrame>,
    response: &mut Streaming<InventoryServerFrame>,
) -> Result<(), String> {
    let mut expected_frame = 1;
    let mut transcript = None::<Sha256>;
    let mut snapshot_id = None::<Uuid>;
    let mut snapshot_permit = None;
    while let Some(frame) = tokio::time::timeout(Duration::from_secs(3), response.message())
        .await
        .map_err(|_| "inventory response timed out")?
        .map_err(|error| error.to_string())?
    {
        if !runtime.exact_member(source) {
            return Err("source membership changed".into());
        }
        if frame.encoded_len() > MESSAGE_BYTES
            || frame.frame_id != expected_frame
            || uuid_from_string(&frame.session_id)? != session_id
        {
            return Err("inventory frame identity, order or size differs".into());
        }
        expected_frame = expected_frame
            .checked_add(1)
            .ok_or("inventory frame ID exhausted")?;
        runtime
            .shared
            .counters
            .frames_received
            .fetch_add(1, Ordering::Relaxed);
        runtime
            .shared
            .counters
            .encoded_bytes_received
            .fetch_add(frame.encoded_len() as u64, Ordering::Relaxed);
        let transcript_frame = frame.clone();
        match frame.body.ok_or("inventory response lacks a body")? {
            inventory_server_frame::Body::SnapshotBegin(begin) => {
                snapshot_permit = Some(
                    runtime
                        .shared
                        .received_snapshots
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|_| "snapshot receiver admission stopped")?,
                );
                let id = uuid_from_string(&begin.snapshot_id)?;
                let view_id = uuid_from_string(&begin.view_id)?;
                runtime.shared.index.begin_snapshot(
                    source.owner.incarnation,
                    session_id,
                    id,
                    view_id,
                    begin.start_sequence,
                )?;
                runtime.shared.index.mark_syncing(source.owner.incarnation);
                snapshot_id = Some(id);
                let mut digest = Sha256::new();
                digest.update(transcript_frame.encode_to_vec());
                transcript = Some(digest);
            }
            inventory_server_frame::Body::SnapshotPage(page) => {
                validate_records(&page.records)?;
                let id = uuid_from_string(&page.snapshot_id)?;
                if snapshot_id != Some(id) {
                    return Err("snapshot page identity changed".into());
                }
                runtime.shared.index.apply_snapshot_page(
                    source.owner.incarnation,
                    session_id,
                    id,
                    page.page_number,
                    page.records.into_iter().map(Into::into).collect(),
                )?;
                transcript
                    .as_mut()
                    .ok_or("snapshot page arrived before begin")?
                    .update(transcript_frame.encode_to_vec());
            }
            inventory_server_frame::Body::Delta(delta) => {
                validate_records(&delta.records)?;
                let records = delta.records.into_iter().map(Into::into).collect();
                if snapshot_id.is_some() {
                    runtime.shared.index.apply_snapshot_delta(
                        source.owner.incarnation,
                        session_id,
                        delta.from_exclusive,
                        delta.through_inclusive,
                        records,
                    )?;
                    transcript
                        .as_mut()
                        .ok_or("snapshot delta arrived before begin")?
                        .update(transcript_frame.encode_to_vec());
                } else {
                    let Some((view_id, _)) = runtime
                        .shared
                        .index
                        .owner_watermark(source.owner.incarnation)
                    else {
                        return Err("delta arrived without an installed owner view".into());
                    };
                    match runtime.shared.index.apply_delta(
                        source.owner.incarnation,
                        view_id,
                        delta.from_exclusive,
                        delta.through_inclusive,
                        records,
                    )? {
                        DeltaApply::Overlap { .. } => {
                            return Err("inventory delta overlaps the installed watermark".into());
                        }
                        DeltaApply::Applied | DeltaApply::Duplicate => {}
                    }
                }
            }
            inventory_server_frame::Body::SnapshotCommit(commit) => {
                let id = uuid_from_string(&commit.snapshot_id)?;
                let view_id = uuid_from_string(&commit.view_id)?;
                if snapshot_id != Some(id)
                    || transcript
                        .take()
                        .ok_or("snapshot commit arrived before begin")?
                        .finalize()
                        .as_slice()
                        != commit.transcript_digest
                {
                    return Err("snapshot transcript or identity differs".into());
                }
                let installed = runtime.shared.index.commit_snapshot(
                    source.owner.incarnation,
                    session_id,
                    id,
                    view_id,
                    commit.through_sequence,
                    commit.page_count,
                )?;
                debug_assert_eq!(installed, view_id);
                runtime
                    .shared
                    .force_bootstrap
                    .lock()
                    .remove(&source.owner.incarnation);
                snapshot_id = None;
                snapshot_permit = None;
            }
            inventory_server_frame::Body::Progress(progress) => {
                if snapshot_id.is_some() {
                    return Err("progress cannot commit a partial snapshot".into());
                }
                if progress.source_head_sequence < progress.through_sequence {
                    return Err("inventory source head precedes covered progress".into());
                }
                if let Some((view_id, applied)) = runtime
                    .shared
                    .index
                    .owner_watermark(source.owner.incarnation)
                    && progress.through_sequence > applied
                {
                    match runtime.shared.index.apply_delta(
                        source.owner.incarnation,
                        view_id,
                        applied,
                        progress.through_sequence,
                        Vec::new(),
                    )? {
                        DeltaApply::Applied | DeltaApply::Duplicate => {}
                        DeltaApply::Overlap { .. } => {
                            return Err(
                                "inventory progress overlaps the installed watermark".into()
                            );
                        }
                    }
                }
                if progress.source_head_sequence == progress.through_sequence
                    && let Some((view_id, applied)) = runtime
                        .shared
                        .index
                        .owner_watermark(source.owner.incarnation)
                    && applied == progress.through_sequence
                {
                    runtime.shared.index.confirm_progress(
                        source.owner.incarnation,
                        view_id,
                        applied,
                    )?;
                }
            }
            inventory_server_frame::Body::ResetRequired(reset) => {
                runtime
                    .shared
                    .counters
                    .resets
                    .fetch_add(1, Ordering::Relaxed);
                runtime
                    .shared
                    .force_bootstrap
                    .lock()
                    .insert(source.owner.incarnation);
                return Err(format!(
                    "source requested inventory reset: {}",
                    reset.reason
                ));
            }
        }
        let installed = runtime
            .shared
            .index
            .owner_watermark(source.owner.incarnation);
        control
            .send(InventoryClientFrame {
                body: Some(inventory_client_frame::Body::Ack(InventoryAck {
                    consumed_frame_id: frame.frame_id,
                    installed_view_id: installed
                        .map(|(view_id, _)| view_id.as_bytes().to_vec())
                        .unwrap_or_default(),
                    applied_sequence: installed.map_or(0, |(_, sequence)| sequence),
                })),
            })
            .await
            .map_err(|_| "inventory request stream closed while acknowledging")?;
    }
    drop(snapshot_permit);
    Err("inventory response stream closed".into())
}

fn validate_records(
    records: &[orbitkv_proto::proto::engine::InventoryRecord],
) -> Result<(), String> {
    if records.len() > INVENTORY_BATCH_RECORDS {
        return Err("inventory frame record count exceeds the decoded limit".into());
    }
    let bytes = records
        .iter()
        .cloned()
        .map(orbitkv_state::InventoryRecord::from)
        .try_fold(0usize, |total, record| {
            total
                .checked_add(record.estimated_size())
                .ok_or("inventory decoded byte count overflow")
        })?;
    if bytes > FRAME_BYTES {
        return Err("inventory frame exceeds the decoded byte limit".into());
    }
    Ok(())
}

fn bounded_delta(
    inventory: &ResidencyInventory,
    after: u64,
    through: u64,
    scope: &InventoryScope,
) -> Result<orbitkv_core::InventoryDelta, InventoryReadError> {
    let mut limit = through;
    loop {
        let delta = match inventory.coalesced_changes_scoped(after, limit, scope) {
            Ok(delta) => delta,
            Err(InventoryReadError::RecordTooLarge) if limit > after + 1 => {
                limit = after + (limit - after) / 2;
                continue;
            }
            Err(error) => return Err(error),
        };
        let encoded = InventoryDelta {
            from_exclusive: after,
            through_inclusive: delta.through,
            records: delta.records.iter().cloned().map(Into::into).collect(),
        }
        .encoded_len();
        if encoded <= FRAME_BYTES {
            return Ok(delta);
        }
        if delta.input_records <= 1 {
            return Err(InventoryReadError::RecordTooLarge);
        }
        limit = after + (delta.through - after) / 2;
    }
}

fn uuid_from(bytes: &[u8], label: &str) -> Result<Uuid, Status> {
    Uuid::from_slice(bytes).map_err(|_| Status::invalid_argument(format!("invalid {label}")))
}

fn uuid_from_string(bytes: &[u8]) -> Result<Uuid, String> {
    Uuid::from_slice(bytes).map_err(|_| "invalid inventory UUID".into())
}

fn optional_uuid(bytes: &[u8], label: &str) -> Result<Option<Uuid>, String> {
    if bytes.is_empty() {
        Ok(None)
    } else {
        Uuid::from_slice(bytes)
            .map(Some)
            .map_err(|_| format!("invalid {label}"))
    }
}

fn encode_scope(scope: &InventoryScope) -> (i32, Vec<Vec<u8>>) {
    match scope {
        InventoryScope::AllNamespaces => (
            InventoryScopeMode::InventoryScopeAllNamespaces as i32,
            Vec::new(),
        ),
        InventoryScope::ExactNamespaces(namespaces) => (
            InventoryScopeMode::InventoryScopeExactNamespaces as i32,
            namespaces
                .iter()
                .map(|namespace| namespace.as_bytes().to_vec())
                .collect(),
        ),
    }
}

fn decode_scope(open: &InventoryOpen) -> Result<InventoryScope, Status> {
    match InventoryScopeMode::try_from(open.scope_mode)
        .map_err(|_| Status::invalid_argument("unknown inventory scope mode"))?
    {
        InventoryScopeMode::InventoryScopeUnspecified => Err(Status::failed_precondition(
            "inventory client does not declare a scope descriptor",
        )),
        InventoryScopeMode::InventoryScopeAllNamespaces if open.scope_namespaces.is_empty() => {
            Ok(InventoryScope::AllNamespaces)
        }
        InventoryScopeMode::InventoryScopeAllNamespaces => Err(Status::invalid_argument(
            "all-namespace scope carries exact namespaces",
        )),
        InventoryScopeMode::InventoryScopeExactNamespaces => {
            let supplied = open
                .scope_namespaces
                .iter()
                .map(|namespace| {
                    std::str::from_utf8(namespace)
                        .map(str::to_owned)
                        .map_err(|_| Status::invalid_argument("inventory namespace is not UTF-8"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let scope =
                InventoryScope::exact(supplied.clone()).map_err(Status::invalid_argument)?;
            if scope.namespaces() != Some(supplied.as_slice()) {
                return Err(Status::invalid_argument(
                    "exact inventory namespaces are not sorted and unique",
                ));
            }
            Ok(scope)
        }
    }
}

#[cfg(test)]
pub(crate) fn all_namespaces_scope_digest() -> [u8; 32] {
    InventoryScope::AllNamespaces.digest()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
#[path = "../../tests/unit/cluster/inventory.rs"]
mod tests;
