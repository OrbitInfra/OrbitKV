//! Request ownership shared by all framework adapters.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use orbitkv_state::RecoveryDemand;

use crate::{
    CallOptions, CancelQueryRequest, ChannelClient, ChannelError, CompletionObservationRequest,
    PublishRequest, QueryBundleRequest, QueryBundleResponse, QueryCommand, QueryOutcomeCode,
    QueryTicket, RestoreRequest, RestoreResponse, RestoreState,
};

const MAX_WARMUPS: usize = 16;
const WARMUP_TTL: Duration = Duration::from_secs(5);
const MAX_PREPARATIONS: usize = 4;
const PREPARATION_TTL: Duration = Duration::from_millis(900);
const COMPLETION_POLL: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct QueryKey {
    instance: String,
    request: String,
    group: u32,
}

struct PendingQuery {
    ticket: QueryTicket,
    hashes: BlockHashes,
    intent: QueryIntent,
    prepared_until: Option<Instant>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryIntent {
    Lookup { wait_for_full_prefix: bool },
    Candidates,
    Shards,
    Recovery(RecoveryDemand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelCallObservation {
    pub request_id: u64,
    pub session_epoch: u64,
    pub session_token: u64,
    pub submitted_mono_ns: u64,
    pub returned_mono_ns: u64,
}

impl ChannelCallObservation {
    fn start(channel: &ChannelClient, request_id: u64) -> Self {
        Self {
            request_id,
            session_epoch: channel.session_epoch(),
            session_token: channel.session_token(),
            submitted_mono_ns: orbitkv_common::timeline::monotonic_ns(),
            returned_mono_ns: 0,
        }
    }

    fn finish(mut self) -> Self {
        self.returned_mono_ns = orbitkv_common::timeline::monotonic_ns();
        self
    }
}

/// Immutable query hashes with shared, allocation-free prefix views.
/// Reusing a batch lets pending queries compare identity in constant time.
#[derive(Clone, Debug)]
pub struct BlockHashes {
    hashes: Arc<[Vec<u8>]>,
    range: Range<usize>,
}

impl BlockHashes {
    pub fn new(hashes: Vec<Vec<u8>>) -> Self {
        let range = 0..hashes.len();
        Self {
            hashes: hashes.into(),
            range,
        }
    }

    pub fn as_slice(&self) -> &[Vec<u8>] {
        &self.hashes[self.range.clone()]
    }

    pub fn slice(&self, range: Range<usize>) -> Option<Self> {
        if range.start > range.end || range.end > self.range.len() {
            return None;
        }
        Some(Self {
            hashes: Arc::clone(&self.hashes),
            range: self.range.start + range.start..self.range.start + range.end,
        })
    }
}

impl PartialEq for BlockHashes {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.hashes, &other.hashes) && self.range == other.range)
            || self.as_slice() == other.as_slice()
    }
}
impl Eq for BlockHashes {}

#[derive(Default)]
struct Queries {
    next_operation: u64,
    pending: HashMap<QueryKey, PendingQuery>,
    warmups: HashMap<QueryKey, (QueryTicket, Instant)>,
}

impl Queries {
    fn ticket(&mut self) -> Result<QueryTicket, ChannelError> {
        self.next_operation = self
            .next_operation
            .checked_add(1)
            .ok_or(ChannelError::SessionRequiresReconnect)?;
        Ok(QueryTicket {
            operation_id: self.next_operation,
            revision: 1,
        })
    }

    fn prepare(
        &mut self,
        key: &QueryKey,
        hashes: &BlockHashes,
        intent: QueryIntent,
    ) -> Result<QueryCommand, ChannelError> {
        if let Some(query) = self.pending.get_mut(key) {
            if (query.intent == QueryIntent::Shards) != (intent == QueryIntent::Shards) {
                return Err(crate::CacheProtocolError::InvalidShardPayload.into());
            }
            if query.prepared_until.is_some() && query.hashes == *hashes && query.intent == intent {
                query.prepared_until = None;
                return Ok(QueryCommand::Claim {
                    ticket: query.ticket,
                    count_lookup: matches!(&intent, QueryIntent::Lookup { .. }),
                });
            }
            if query.hashes == *hashes && query.intent == intent {
                return Ok(QueryCommand::Poll(query.ticket));
            }
            query.ticket.revision = query
                .ticket
                .revision
                .checked_add(1)
                .ok_or(ChannelError::SessionRequiresReconnect)?;
            query.hashes = hashes.clone();
            query.intent = intent.clone();
            query.prepared_until = None;
        } else {
            let ticket = self.ticket()?;
            self.pending.insert(
                key.clone(),
                PendingQuery {
                    ticket,
                    hashes: hashes.clone(),
                    intent: intent.clone(),
                    prepared_until: None,
                },
            );
        }
        let query = &self.pending[key];
        Ok(QueryCommand::Submit(QueryBundleRequest {
            ticket: query.ticket,
            instance_id: key.instance.clone(),
            request_id: key.request.clone(),
            block_hashes: query.hashes.as_slice().to_vec(),
            group_id: key.group,
            wait_for_full_prefix: matches!(
                &intent,
                QueryIntent::Lookup {
                    wait_for_full_prefix: true
                }
            ),
            warmup: false,
            discover: intent == QueryIntent::Candidates,
            materialize: matches!(&intent, QueryIntent::Recovery(_)),
            prepare: false,
            demand: match intent {
                QueryIntent::Recovery(demand) => Some(demand),
                _ => None,
            },
        }))
    }

    fn complete(&mut self, key: &QueryKey, outcome: QueryOutcomeCode) {
        if outcome != QueryOutcomeCode::Loading {
            self.pending.remove(key);
        }
    }
}

/// A GPU restore remains owned after a wait deadline. Engine-local destinations
/// remain leased until the native executor proves its own DMA drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestoreHandle {
    pub operation_id: u64,
    pub session_epoch: u64,
    pub session_token: u64,
    owner: u64,
}

/// One selected group range within a compiled recovery contract.
pub struct RecoveryRead<'a> {
    pub contract: &'a orbitkv_state::RecoveryContract,
    pub namespace: &'a str,
    pub span: orbitkv_state::TokenRange,
    pub group: u32,
}

/// Owns query revisions, warming interests, and independent publish/restore
/// sessions. It does not allocate engine GPU pages or determine cache locality.
pub struct CacheClient {
    channel: ChannelClient,
    socket: PathBuf,
    options: CallOptions,
    publisher: Mutex<Option<Arc<ChannelClient>>>,
    closed: AtomicBool,
    requests: AtomicU64,
    queries: Mutex<Queries>,
    last_completion_poll: Mutex<Instant>,
    owner: u64,
}

impl CacheClient {
    pub fn connect(socket: impl AsRef<Path>, options: CallOptions) -> Result<Self, ChannelError> {
        static OWNERS: AtomicU64 = AtomicU64::new(1);
        let owner = next_id(&OWNERS)?;
        Ok(Self {
            channel: ChannelClient::connect(&socket, options)?,
            socket: socket.as_ref().to_path_buf(),
            options,
            publisher: Mutex::new(None),
            closed: AtomicBool::new(false),
            requests: AtomicU64::new(1),
            queries: Mutex::new(Queries::default()),
            last_completion_poll: Mutex::new(Instant::now()),
            owner,
        })
    }

    pub fn channel(&self) -> &ChannelClient {
        &self.channel
    }
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub fn publish_progress(&self) -> Result<crate::PublishProgress, ChannelError> {
        let publisher = self
            .publisher
            .try_lock()
            .map_err(|_| ChannelError::PublishObservationUnavailable)?;
        match publisher.as_ref() {
            Some(publisher) => publisher.publish_progress(),
            None => Ok(crate::PublishProgress::default()),
        }
    }

    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.channel.close();
        if let Ok(publisher) = self.publisher.lock()
            && let Some(publisher) = publisher.as_ref()
        {
            publisher.close();
        }
        if let Ok(mut queries) = self.queries.lock() {
            queries.pending.clear();
            queries.warmups.clear();
        }
    }

    pub fn query(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        group: u32,
        intent: QueryIntent,
    ) -> Result<QueryBundleResponse, ChannelError> {
        self.query_inner(instance, hashes, request, group, intent, false)
            .map(|(response, _)| response)
    }

    pub fn query_observed(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        group: u32,
        intent: QueryIntent,
    ) -> Result<(QueryBundleResponse, ChannelCallObservation), ChannelError> {
        self.query_inner(instance, hashes, request, group, intent, true)
            .and_then(|(response, observation)| {
                observation
                    .map(|observation| (response, observation))
                    .ok_or(ChannelError::SessionRequiresReconnect)
            })
    }

    fn query_inner(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        group: u32,
        intent: QueryIntent,
        observe: bool,
    ) -> Result<(QueryBundleResponse, Option<ChannelCallObservation>), ChannelError> {
        if intent == QueryIntent::Shards {
            return Err(crate::CacheProtocolError::InvalidShardPayload.into());
        }
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if let Some((ticket, _)) = queries.warmups.remove(&key) {
            self.cancel(ticket, false)?;
        }
        if queries.pending.get(&key).is_some_and(|query| {
            query
                .prepared_until
                .is_some_and(|until| Instant::now() >= until)
        }) && let Some(query) = queries.pending.remove(&key)
        {
            self.cancel(query.ticket, query.intent == QueryIntent::Shards)?;
        }
        let previous = queries.pending.get(&key).map(|query| query.ticket);
        let discover = intent == QueryIntent::Candidates;
        let command = queries.prepare(&key, hashes, intent)?;
        let request_id = next_id(&self.requests)?;
        let observation = observe.then(|| ChannelCallObservation::start(&self.channel, request_id));
        let response = self.channel.query_bundle(request_id, &command);
        match response {
            Ok(response) => {
                if (discover && response.outcome == QueryOutcomeCode::Ready)
                    || response.outcome == QueryOutcomeCode::Candidates
                        && (!discover
                            || response
                                .hit_positions
                                .iter()
                                .any(|&p| p as usize >= hashes.as_slice().len()))
                {
                    self.channel.close();
                    return Err(ChannelError::SessionRequiresReconnect);
                }
                queries.complete(&key, response.outcome);
                Ok((response, observation.map(ChannelCallObservation::finish)))
            }
            Err(error) => {
                // Failure before submission can leave the prior revision alive;
                // failure after submission can leave the replacement alive.
                if let Some(query) = queries.pending.remove(&key) {
                    let _ = self.cancel(query.ticket, query.intent == QueryIntent::Shards);
                    if let Some(previous) = previous.filter(|ticket| *ticket != query.ticket) {
                        let _ = self.cancel(previous, false);
                    }
                }
                Err(error)
            }
        }
    }

    /// Materialize exactly one group's demand, then validate actual leased
    /// coverage. Stale hints become a miss; partial leases never escape.
    pub fn read_recovery(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        read: RecoveryRead<'_>,
    ) -> Result<QueryBundleResponse, ChannelError> {
        let RecoveryRead {
            contract,
            namespace,
            span,
            group,
        } = read;
        let range = contract.read_range(namespace, span, group, hashes.as_slice().len())?;
        let selected = hashes
            .slice(range.clone())
            .ok_or(orbitkv_state::RecoveryError::InvalidSpan)?;
        let demand = contract.demand(namespace, span)?;
        let mut response = self.query(
            instance,
            &selected,
            request,
            group,
            QueryIntent::Recovery(demand),
        )?;
        if response.outcome == QueryOutcomeCode::Ready {
            let complete = response.num_hit_blocks as usize == range.len()
                && (range.is_empty() || !response.lease.is_empty())
                && (group == 0
                    || response
                        .hit_positions
                        .iter()
                        .copied()
                        .eq(0..range.len() as u32));
            if !complete {
                if !response.lease.is_empty() {
                    self.release(std::mem::take(&mut response.lease))?;
                }
                response.num_hit_blocks = 0;
                response.hit_positions.clear();
            } else {
                response.hit_positions = (range.start as u32..range.end as u32).collect();
            }
        }
        Ok(response)
    }

    /// Prepare one compiled range without transferring its lease to the caller.
    /// A later matching demand claims it; changed/expired work is retired.
    pub fn prepare_recovery(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
        read: RecoveryRead<'_>,
    ) -> Result<bool, ChannelError> {
        if hashes.as_slice().is_empty() {
            return Ok(false);
        }
        let range = read.contract.read_range(
            read.namespace,
            read.span,
            read.group,
            hashes.as_slice().len(),
        )?;
        if range.is_empty() {
            return Ok(false);
        }
        let hashes = hashes
            .slice(range)
            .ok_or(orbitkv_state::RecoveryError::InvalidSpan)?;
        let demand = read.contract.demand(read.namespace, read.span)?;
        self.prepare_query(
            instance,
            hashes,
            request,
            read.group,
            QueryIntent::Recovery(demand),
        )
    }

    /// Prepare an unselected attention prefix. A matching ordinary lookup may
    /// claim its partial prefix and counts the logical lookup exactly once.
    pub fn prepare_prefix(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
    ) -> Result<bool, ChannelError> {
        self.prepare_query(
            instance,
            hashes.clone(),
            request,
            0,
            QueryIntent::Lookup {
                wait_for_full_prefix: false,
            },
        )
    }

    fn prepare_query(
        &self,
        instance: &str,
        hashes: BlockHashes,
        request: &str,
        group: u32,
        intent: QueryIntent,
    ) -> Result<bool, ChannelError> {
        if hashes.as_slice().is_empty() {
            return Ok(false);
        }
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        let now = Instant::now();
        let expired: Vec<_> = queries
            .pending
            .iter()
            .filter(|(_, query)| query.prepared_until.is_some_and(|until| now >= until))
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            if let Some(query) = queries.pending.remove(&key) {
                self.cancel(query.ticket, query.intent == QueryIntent::Shards)?;
            }
        }
        if queries.pending.contains_key(&key)
            || queries
                .pending
                .values()
                .filter(|q| q.prepared_until.is_some())
                .count()
                >= MAX_PREPARATIONS
        {
            return Ok(false);
        }
        let ticket = queries.ticket()?;
        let response = self.channel.query_bundle(
            next_id(&self.requests)?,
            &QueryCommand::Submit(QueryBundleRequest {
                ticket,
                instance_id: instance.into(),
                request_id: request.into(),
                block_hashes: hashes.as_slice().to_vec(),
                group_id: group,
                wait_for_full_prefix: false,
                warmup: false,
                discover: false,
                materialize: matches!(&intent, QueryIntent::Recovery(_)),
                prepare: true,
                demand: match &intent {
                    QueryIntent::Recovery(demand) => Some(demand.clone()),
                    _ => None,
                },
            }),
        );
        match response {
            Ok(response) if response.outcome == QueryOutcomeCode::Loading => {
                queries.pending.insert(
                    key,
                    PendingQuery {
                        ticket,
                        hashes,
                        intent,
                        prepared_until: Some(now + PREPARATION_TTL),
                    },
                );
                Ok(true)
            }
            Ok(response) if response.outcome == QueryOutcomeCode::Busy => Ok(false),
            Ok(_) => {
                self.channel.close();
                Err(ChannelError::SessionRequiresReconnect)
            }
            Err(error) => {
                let _ = self.cancel(ticket, false);
                Err(error)
            }
        }
    }

    pub fn warm_prefix(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
    ) -> Result<bool, ChannelError> {
        if hashes.as_slice().is_empty() {
            return Ok(false);
        }
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group: 0,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        let now = Instant::now();
        let expired: Vec<_> = queries
            .warmups
            .iter()
            .filter(|(_, (_, submitted))| now.duration_since(*submitted) >= WARMUP_TTL)
            .map(|(key, _)| key.clone())
            .collect();
        for key in expired {
            if let Some((ticket, _)) = queries.warmups.remove(&key) {
                self.cancel(ticket, false)?;
            }
        }
        if queries.warmups.contains_key(&key)
            || queries.pending.contains_key(&key)
            || queries.warmups.len() >= MAX_WARMUPS
        {
            return Ok(false);
        }
        let ticket = queries.ticket()?;
        let response = self.channel.query_bundle(
            next_id(&self.requests)?,
            &QueryCommand::Submit(QueryBundleRequest {
                ticket,
                instance_id: instance.into(),
                request_id: request.into(),
                block_hashes: hashes.as_slice().to_vec(),
                group_id: 0,
                wait_for_full_prefix: false,
                warmup: true,
                discover: false,
                materialize: false,
                prepare: false,
                demand: None,
            }),
        )?;
        match response.outcome {
            QueryOutcomeCode::Loading => {
                queries.warmups.insert(key, (ticket, now));
                Ok(true)
            }
            QueryOutcomeCode::Busy => Ok(false),
            QueryOutcomeCode::Ready
                if response.num_hit_blocks == 0 && response.lease.is_empty() =>
            {
                Ok(true)
            }
            QueryOutcomeCode::Ready | QueryOutcomeCode::Candidates => {
                self.channel.close();
                Err(ChannelError::SessionRequiresReconnect)
            }
        }
    }

    pub fn cancel_query(
        &self,
        instance: &str,
        request: &str,
        group: u32,
    ) -> Result<(), ChannelError> {
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if let Some((ticket, _)) = queries.warmups.remove(&key) {
            self.cancel(ticket, false)?;
        }
        if let Some(query) = queries.pending.remove(&key) {
            self.cancel(query.ticket, query.intent == QueryIntent::Shards)?;
        }
        Ok(())
    }

    fn cancel(&self, ticket: QueryTicket, shards: bool) -> Result<(), ChannelError> {
        let request_id = next_id(&self.requests)?;
        let request = CancelQueryRequest { ticket };
        if shards {
            self.channel.cancel_shard_query(request_id, &request)
        } else {
            self.channel.cancel_query(request_id, &request)
        }
    }

    pub fn query_shards(
        &self,
        instance: &str,
        hashes: &BlockHashes,
        request: &str,
    ) -> Result<crate::ShardQueryResponse, ChannelError> {
        let key = QueryKey {
            instance: instance.into(),
            request: request.into(),
            group: 0,
        };
        let mut queries = self
            .queries
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if let Some((ticket, _)) = queries.warmups.remove(&key) {
            self.cancel(ticket, false)?;
        }
        let previous = queries.pending.get(&key).map(|query| query.ticket);
        let command = queries.prepare(&key, hashes, QueryIntent::Shards)?;
        let result = self
            .channel
            .query_shards(next_id(&self.requests)?, &command);
        match result {
            Ok(response) if response.num_hit_blocks <= hashes.as_slice().len() as u64 => {
                queries.complete(&key, response.outcome);
                Ok(response)
            }
            result => {
                if let Some(query) = queries.pending.remove(&key) {
                    let _ = self.cancel(query.ticket, true);
                    if let Some(previous) = previous.filter(|ticket| *ticket != query.ticket) {
                        let _ = self.cancel(previous, true);
                    }
                }
                match result {
                    Err(error) => Err(error),
                    Ok(_) => {
                        self.channel.close();
                        Err(crate::CacheProtocolError::InvalidShardPayload.into())
                    }
                }
            }
        }
    }

    pub fn release_shard_query(&self, control_id: Vec<u8>) -> Result<(), ChannelError> {
        self.channel
            .release_shard_query(next_id(&self.requests)?, control_id)
    }

    pub fn release(&self, lease: Vec<u8>) -> Result<(), ChannelError> {
        self.channel.release(next_id(&self.requests)?, lease)
    }

    pub fn observe_completion(
        &self,
        observation: &CompletionObservationRequest,
    ) -> Result<(), ChannelError> {
        self.channel
            .observe_completion(next_id(&self.requests)?, observation)
    }

    pub fn publish(&self, request: &PublishRequest) -> Result<(), ChannelError> {
        self.publish_inner(request, false).map(|_| ())
    }

    pub fn publish_observed(
        &self,
        request: &PublishRequest,
    ) -> Result<ChannelCallObservation, ChannelError> {
        self.publish_inner(request, true)?
            .ok_or(ChannelError::SessionRequiresReconnect)
    }

    fn publish_inner(
        &self,
        request: &PublishRequest,
        observe: bool,
    ) -> Result<Option<ChannelCallObservation>, ChannelError> {
        // A long D2H publish must not hold the query/restore descriptor slot.
        let publisher = {
            let mut publisher = self
                .publisher
                .lock()
                .map_err(|_| ChannelError::SessionRequiresReconnect)?;
            if self.closed.load(Ordering::Acquire) {
                return Err(ChannelError::SessionRequiresReconnect);
            }
            if publisher.is_none() {
                *publisher = Some(Arc::new(ChannelClient::connect(
                    &self.socket,
                    self.options,
                )?));
            }
            Arc::clone(
                publisher
                    .as_ref()
                    .ok_or(ChannelError::SessionRequiresReconnect)?,
            )
        };
        let request_id = next_id(&self.requests)?;
        let observation = observe.then(|| ChannelCallObservation::start(&publisher, request_id));
        publisher.publish(request_id, request)?;
        Ok(observation.map(ChannelCallObservation::finish))
    }

    pub fn start_restore(&self, request: &RestoreRequest) -> Result<RestoreHandle, ChannelError> {
        let operation_id = self
            .channel
            .restore_submit(next_id(&self.requests)?, request)?;
        Ok(RestoreHandle {
            operation_id,
            session_epoch: self.channel.session_epoch(),
            session_token: self.channel.session_token(),
            owner: self.owner,
        })
    }

    pub fn poll_restore(&self, handle: RestoreHandle) -> Result<RestoreResponse, ChannelError> {
        // A configured Manager epoch can repeat after restart. The native
        // issuer also fences handles against a new mapping with equal wire IDs.
        if handle.owner != self.owner
            || handle.session_token != self.channel.session_token()
            || handle.session_epoch != self.channel.session_epoch()
        {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        self.channel.restore_poll(handle.operation_id)
    }

    pub fn restore_completions(&self) -> Arc<crate::RestoreCompletions> {
        self.channel.restore_completions()
    }

    pub fn claim_local_restore(
        &self,
        handle: RestoreHandle,
    ) -> Result<Option<Vec<u8>>, ChannelError> {
        if handle.owner != self.owner
            || handle.session_token != self.channel.session_token()
            || handle.session_epoch != self.channel.session_epoch()
        {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        self.channel
            .restore_completions()
            .claim_local(handle.operation_id)
            .map_err(Into::into)
    }

    pub fn finish_local_restore(
        &self,
        handle: RestoreHandle,
        result: Result<(), String>,
        timing: Option<crate::RestoreTiming>,
    ) -> Result<bool, ChannelError> {
        if handle.owner != self.owner
            || handle.session_token != self.channel.session_token()
            || handle.session_epoch != self.channel.session_epoch()
        {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        self.channel
            .restore_completions()
            .drained(handle.operation_id, result, timing)
            .map_err(Into::into)
    }

    pub fn restore_completions_ready(&self, timeout: Duration) -> Result<bool, ChannelError> {
        if !self.channel.peer_alive()? {
            return Ok(true);
        }
        let mut last = self
            .last_completion_poll
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        let remaining = COMPLETION_POLL.saturating_sub(last.elapsed());
        if remaining.is_zero()
            || self.channel.wait_for_notification(timeout.min(remaining))?
            || last.elapsed() >= COMPLETION_POLL
        {
            *last = Instant::now();
            return Ok(true);
        }
        Ok(false)
    }

    pub fn wait_restore(
        &self,
        handle: RestoreHandle,
        timeout: Duration,
    ) -> Result<RestoreResponse, ChannelError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(ChannelError::RestoreTimeout {
                operation_id: handle.operation_id,
            })?;
        loop {
            let response = self.poll_restore(handle)?;
            if response.state != RestoreState::Pending {
                return Ok(response);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(ChannelError::RestoreTimeout {
                    operation_id: handle.operation_id,
                });
            }
            self.restore_completions_ready(remaining)?;
        }
    }
}

fn next_id(counter: &AtomicU64) -> Result<u64, ChannelError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| ChannelError::SessionRequiresReconnect)
}

#[cfg(test)]
#[path = "../tests/unit/cache_client.rs"]
mod tests;
