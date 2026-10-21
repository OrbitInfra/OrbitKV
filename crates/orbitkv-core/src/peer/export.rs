// Source allocation ownership and replay protection for peer READs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hashlink::LinkedHashMap;
use log::warn;
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use uuid::Uuid;

use crate::block::{SealedBlock, StateKey};
use crate::metrics::core_metrics;

pub(crate) const TRANSFER_WINDOW_SLOTS: usize = 64;
const MAX_TRANSFER_SESSIONS: usize = 1024;
const MAX_TRANSFER_WINDOWS: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct TransferTicket {
    pub(crate) window: Uuid,
    pub(crate) slot: usize,
    pub(crate) generation: u64,
}

impl TransferTicket {
    pub fn new(window: Uuid, slot: usize, generation: u64) -> Result<Self, PeerError> {
        if window.is_nil() || slot >= TRANSFER_WINDOW_SLOTS || generation == 0 {
            return Err(PeerError::InvalidRequest("invalid transfer ticket".into()));
        }
        Ok(Self {
            window,
            slot,
            generation,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PeerError {
    Unavailable,
    StagingFailed,
    StaleReplica,
    InvalidRequest(String),
    UnknownWindow,
    StaleTicket,
    BudgetExhausted,
}

/// Authoritative DRAM export admission and the lifetime of peer source grants.
/// Directory candidates cannot bypass this owner's incarnation/version checks.
pub struct PeerExports {
    dram: Arc<crate::storage::dram::DramStore>,
    ssd: Option<Arc<crate::storage::ssd::SsdStore>>,
    membership: Option<Arc<orbitkv_catalog::MembershipView>>,
    endpoint: Option<String>,
    locks: TransferLockManager,
    #[cfg(feature = "orbitte")]
    transport: Option<Arc<super::provider::PeerTransport>>,
}

impl PeerExports {
    pub(crate) fn new(
        dram: Arc<crate::storage::dram::DramStore>,
        ssd: Option<Arc<crate::storage::ssd::SsdStore>>,
        membership: Option<Arc<orbitkv_catalog::MembershipView>>,
        endpoint: Option<String>,
        #[cfg(feature = "orbitte")] transport: Option<Arc<super::provider::PeerTransport>>,
        timeout: Duration,
        budget_bytes: u64,
    ) -> Self {
        Self {
            dram,
            ssd,
            membership,
            endpoint,
            locks: TransferLockManager::new(timeout, budget_bytes),
            #[cfg(feature = "orbitte")]
            transport,
        }
    }

    fn validate_owner(&self, owner: Uuid) -> Result<(), PeerError> {
        if self.endpoint.is_none() {
            return Err(PeerError::Unavailable);
        }
        if self
            .membership
            .as_ref()
            .is_none_or(|view| view.owner().incarnation != owner || !view.permits(view.owner()))
        {
            return Err(PeerError::StaleReplica);
        }
        Ok(())
    }

    pub fn open(&self, owner: Uuid, requester: Uuid) -> Result<Uuid, PeerError> {
        self.validate_owner(owner)?;
        if requester.is_nil() {
            return Err(PeerError::InvalidRequest(
                "nil requester incarnation".into(),
            ));
        }
        self.locks.open(requester).ok_or(PeerError::BudgetExhausted)
    }

    pub async fn authorize(
        &self,
        owner: Uuid,
        ticket: TransferTicket,
        records: &[orbitkv_state::InventoryRecord],
    ) -> Result<Vec<(StateKey, Arc<SealedBlock>)>, PeerError> {
        self.validate_owner(owner)?;
        let first = records
            .first()
            .ok_or_else(|| PeerError::InvalidRequest("empty transfer".into()))?;
        if records.iter().any(|record| {
            record.key.namespace != first.key.namespace || !record.present || record.sequence == 0
        }) {
            return Err(PeerError::InvalidRequest(
                "invalid residency evidence".into(),
            ));
        }
        let hashes: Vec<_> = records
            .iter()
            .map(|record| record.key.hash.clone())
            .collect();
        orbitkv_state::validate_discovery_query(&first.key.namespace, &hashes)
            .map_err(|reason| PeerError::InvalidRequest(reason.into()))?;
        if let Some(blocks) = self.dram.pin_residencies(records) {
            self.locks.lock_blocks(ticket, blocks.clone())?;
            #[cfg(feature = "orbitte")]
            self.register_source(ticket, &blocks)?;
            return Ok(blocks);
        }

        let ssd = self.ssd.as_ref().ok_or(PeerError::StaleReplica)?;
        let leases = ssd
            .pin_residencies(records)
            .ok_or(PeerError::StaleReplica)?;
        let staging_bytes = ssd
            .staging_footprint(&leases)
            .ok_or(PeerError::BudgetExhausted)?;
        let reservation = self.locks.reserve(ticket, staging_bytes, leases.len())?;
        let mut materialized = ssd.read_host_batch_for_export(leases, reservation).await?;
        let mut blocks = Vec::with_capacity(records.len());
        for record in records {
            let Some(index) = materialized.iter().position(|(key, _)| key == &record.key) else {
                let _ = self.locks.release(ticket);
                return Err(PeerError::StaleReplica);
            };
            blocks.push(materialized.swap_remove(index));
        }
        if !materialized.is_empty() {
            let _ = self.locks.release(ticket);
            return Err(PeerError::StaleReplica);
        }
        #[cfg(feature = "orbitte")]
        self.register_source(ticket, &blocks)?;
        Ok(blocks)
    }

    #[cfg(feature = "orbitte")]
    fn register_source(
        &self,
        ticket: TransferTicket,
        blocks: &[(StateKey, Arc<SealedBlock>)],
    ) -> Result<(), PeerError> {
        let Some(super::provider::PeerTransport::OrbitTe(transport)) = self.transport.as_deref()
        else {
            return Ok(());
        };
        let result = transport
            .register(
                blocks.iter().flat_map(|(_, block)| block.allocations()),
                orbit_te::Permission::RemoteRead,
            )
            .map_err(|error| {
                warn!("OrbitTE source registration rejected: {error}");
                PeerError::BudgetExhausted
            })
            .and_then(|registration| {
                let mut inner = self.locks.inner.lock();
                let session = inner
                    .windows
                    .get_mut(&ticket.window)
                    .and_then(|window| window.slots.get_mut(ticket.slot))
                    .filter(|slot| slot.generation == ticket.generation)
                    .and_then(|slot| slot.transfer.as_mut())
                    .filter(|session| matches!(session.state, TransferState::Ready(_)))
                    .ok_or(PeerError::StaleTicket)?;
                session.registration = Some(registration);
                Ok(())
            });
        if result.is_err() {
            let _ = self.locks.release(ticket);
        }
        result
    }

    pub fn release(&self, ticket: TransferTicket) -> Result<usize, PeerError> {
        // Fencing prevents new grants, but terminal completion must still drain.
        self.locks.release(ticket)
    }

    pub fn lock_timeout(&self) -> Duration {
        self.locks.lock_timeout()
    }

    pub(crate) fn expire(&self) -> usize {
        self.locks.expire()
    }

    #[cfg(test)]
    pub(crate) fn transfer_accounting(&self) -> (usize, u64) {
        self.locks.accounting()
    }
}

struct TransferSession {
    state: TransferState,
    block_count: usize,
    created_at: Instant,
    reserved_bytes: u64,
    expired: bool,
    #[cfg(feature = "orbitte")]
    registration: Option<super::orbitte::RegistrationSet>,
}

enum TransferState {
    Staging { released: bool },
    Ready(Vec<(StateKey, Arc<SealedBlock>)>),
}

#[derive(Default)]
struct TransferSlot {
    generation: u64,
    transfer: Option<TransferSession>,
}

struct TransferWindow {
    requester: Uuid,
    slots: [TransferSlot; TRANSFER_WINDOW_SLOTS],
}

#[derive(Default)]
struct Transfers {
    windows: LinkedHashMap<Uuid, TransferWindow>,
    active: usize,
    reserved_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct TransferLockManager {
    inner: Arc<Mutex<Transfers>>,
    lock_timeout: Duration,
    budget_bytes: u64,
}

/// Ticket-scoped provisional admission held by the detached SSD batch owner.
/// Dropping it rolls back only its still-staging generation.
pub(crate) struct StagingReservation {
    locks: TransferLockManager,
    ticket: TransferTicket,
    active: bool,
}

/// A fully materialized source grant which is not durable until its result is
/// handed to the authorization owner. A failed handoff rolls the grant back.
pub(crate) struct PreparedTransfer {
    locks: TransferLockManager,
    ticket: TransferTicket,
    active: bool,
}

impl StagingReservation {
    pub(crate) fn commit(
        self,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> Result<PreparedTransfer, PeerError> {
        let actual_bytes = allocation_bytes(&blocks)?;
        self.commit_accounted(blocks, actual_bytes)
    }

    #[cfg(test)]
    pub(crate) fn commit_with_footprint(
        self,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
        actual_bytes: u64,
    ) -> Result<PreparedTransfer, PeerError> {
        self.commit_accounted(blocks, actual_bytes)
    }

    fn commit_accounted(
        mut self,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
        actual_bytes: u64,
    ) -> Result<PreparedTransfer, PeerError> {
        self.locks
            .commit_staging(self.ticket, blocks, actual_bytes)?;
        self.active = false;
        Ok(PreparedTransfer {
            locks: self.locks.clone(),
            ticket: self.ticket,
            active: true,
        })
    }
}

impl Drop for StagingReservation {
    fn drop(&mut self) {
        if self.active {
            self.locks.rollback(self.ticket, false);
        }
    }
}

impl PreparedTransfer {
    pub(crate) fn publish(mut self) {
        self.active = false;
    }
}

impl Drop for PreparedTransfer {
    fn drop(&mut self) {
        if self.active {
            self.locks.rollback(self.ticket, true);
        }
    }
}

impl TransferLockManager {
    pub(crate) fn new(lock_timeout: Duration, budget_bytes: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Transfers::default())),
            lock_timeout,
            budget_bytes,
        }
    }

    pub(crate) fn lock_timeout(&self) -> Duration {
        self.lock_timeout
    }

    #[cfg(test)]
    pub(crate) fn accounting(&self) -> (usize, u64) {
        let inner = self.inner.lock();
        (inner.active, inner.reserved_bytes)
    }

    /// Opening a window pins no data. Only idle windows may be evicted; an
    /// evicted UUID is never recreated by authorization or completion.
    pub(crate) fn open(&self, requester: Uuid) -> Option<Uuid> {
        let mut inner = self.inner.lock();
        if inner.windows.len() == MAX_TRANSFER_WINDOWS {
            let idle = inner.windows.iter().find_map(|(id, window)| {
                window
                    .slots
                    .iter()
                    .all(|slot| slot.transfer.is_none())
                    .then_some(*id)
            })?;
            inner.windows.remove(&idle);
        }
        let id = Uuid::new_v4();
        inner.windows.insert(
            id,
            TransferWindow {
                requester,
                slots: std::array::from_fn(|_| TransferSlot::default()),
            },
        );
        Some(id)
    }

    /// A ticket is single use. A completion arriving before authorization
    /// closes its generation, so delayed authorization cannot resurrect a hold.
    pub(crate) fn lock_blocks(
        &self,
        ticket: TransferTicket,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
    ) -> Result<(), PeerError> {
        let bytes = allocation_bytes(&blocks)?;
        let reservation = self.reserve(ticket, bytes, blocks.len())?;
        reservation.commit(blocks)?.publish();
        Ok(())
    }

    /// Reserve source session and byte budgets before staging allocates pinned
    /// memory. The ticket generation is consumed even when admission fails.
    pub(crate) fn reserve(
        &self,
        ticket: TransferTicket,
        bytes: u64,
        block_count: usize,
    ) -> Result<StagingReservation, PeerError> {
        let mut inner = self.inner.lock();
        let window = inner
            .windows
            .to_back(&ticket.window)
            .ok_or(PeerError::UnknownWindow)?;
        let slot = window
            .slots
            .get_mut(ticket.slot)
            .ok_or(PeerError::StaleTicket)?;
        if block_count == 0 || ticket.generation <= slot.generation || slot.transfer.is_some() {
            return Err(PeerError::StaleTicket);
        }
        // Consume the generation even when admission fails.
        slot.generation = ticket.generation;
        let rejection = if inner.active >= MAX_TRANSFER_SESSIONS {
            Some("sessions")
        } else if bytes > self.budget_bytes.saturating_sub(inner.reserved_bytes) {
            Some("bytes")
        } else {
            None
        };
        if let Some(reason) = rejection {
            core_metrics()
                .transfer_lock_rejections
                .add(1, &[KeyValue::new("reason", reason)]);
            return Err(PeerError::BudgetExhausted);
        }
        inner.windows[&ticket.window].slots[ticket.slot].transfer = Some(TransferSession {
            state: TransferState::Staging { released: false },
            block_count,
            created_at: Instant::now(),
            reserved_bytes: bytes,
            expired: false,
            #[cfg(feature = "orbitte")]
            registration: None,
        });
        inner.active += 1;
        inner.reserved_bytes += bytes;
        core_metrics()
            .transfer_reserved_bytes
            .add(bytes as i64, &[]);
        core_metrics()
            .transfer_lock_active
            .add(block_count as i64, &[]);
        Ok(StagingReservation {
            locks: self.clone(),
            ticket,
            active: true,
        })
    }

    fn commit_staging(
        &self,
        ticket: TransferTicket,
        blocks: Vec<(StateKey, Arc<SealedBlock>)>,
        actual_bytes: u64,
    ) -> Result<(), PeerError> {
        let mut inner = self.inner.lock();
        let previous_bytes = {
            let window = inner
                .windows
                .to_back(&ticket.window)
                .ok_or(PeerError::UnknownWindow)?;
            let slot = window
                .slots
                .get(ticket.slot)
                .ok_or(PeerError::StaleTicket)?;
            if slot.generation != ticket.generation {
                return Err(PeerError::StaleTicket);
            }
            let session = slot.transfer.as_ref().ok_or(PeerError::StaleTicket)?;
            if session.block_count != blocks.len()
                || !matches!(session.state, TransferState::Staging { released: false })
            {
                return Err(PeerError::StaleTicket);
            }
            session.reserved_bytes
        };
        let other_bytes = inner
            .reserved_bytes
            .checked_sub(previous_bytes)
            .expect("session reservation is included in transfer accounting");
        if actual_bytes > self.budget_bytes.saturating_sub(other_bytes) {
            core_metrics()
                .transfer_lock_rejections
                .add(1, &[KeyValue::new("reason", "bytes")]);
            return Err(PeerError::BudgetExhausted);
        }

        let session = inner
            .windows
            .get_mut(&ticket.window)
            .expect("validated transfer window")
            .slots
            .get_mut(ticket.slot)
            .expect("validated transfer slot")
            .transfer
            .as_mut()
            .expect("validated transfer session");
        session.reserved_bytes = actual_bytes;
        session.state = TransferState::Ready(blocks);
        inner.reserved_bytes = other_bytes + actual_bytes;
        if actual_bytes >= previous_bytes {
            core_metrics()
                .transfer_reserved_bytes
                .add((actual_bytes - previous_bytes) as i64, &[]);
        } else {
            core_metrics()
                .transfer_reserved_bytes
                .add(-((previous_bytes - actual_bytes) as i64), &[]);
        }
        Ok(())
    }

    fn rollback(&self, ticket: TransferTicket, ready: bool) {
        let mut inner = self.inner.lock();
        let Some(window) = inner.windows.get_mut(&ticket.window) else {
            return;
        };
        let Some(slot) = window.slots.get_mut(ticket.slot) else {
            return;
        };
        if slot.generation != ticket.generation
            || !slot
                .transfer
                .as_ref()
                .is_some_and(|session| matches!(session.state, TransferState::Ready(_)) == ready)
        {
            return;
        }
        let session = slot.transfer.take().expect("matched transfer session");
        release_accounting(&mut inner, &session);
    }

    /// Completion is idempotent, including unknown windows. Never close an
    /// active different generation. The peer endpoint must be cluster-isolated;
    /// possession of a window UUID is not a replacement for authentication.
    pub(crate) fn release(&self, ticket: TransferTicket) -> Result<usize, PeerError> {
        let mut inner = self.inner.lock();
        let Some(window) = inner.windows.to_back(&ticket.window) else {
            return Ok(0);
        };
        let slot = window
            .slots
            .get_mut(ticket.slot)
            .ok_or(PeerError::StaleTicket)?;
        if ticket.generation == 0 {
            return Err(PeerError::StaleTicket);
        }
        if ticket.generation < slot.generation {
            return Ok(0);
        }
        if ticket.generation > slot.generation && slot.transfer.is_some() {
            return Err(PeerError::StaleTicket);
        }
        slot.generation = ticket.generation;
        let Some(session) = slot.transfer.as_mut() else {
            return Ok(0);
        };
        if let TransferState::Staging { released } = &mut session.state {
            *released = true;
            return Ok(0);
        }
        let session = slot.transfer.take().expect("ready transfer session");
        let count = match &session.state {
            TransferState::Ready(blocks) => blocks.len(),
            TransferState::Staging { .. } => unreachable!("staging returned above"),
        };
        release_accounting(&mut inner, &session);
        Ok(count)
    }

    /// Overdue is observational: only terminal completion permits memory reuse.
    pub(crate) fn expire(&self) -> usize {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let mut expired_count = 0;
        for (id, window) in &mut inner.windows {
            for (index, slot) in window.slots.iter_mut().enumerate() {
                if let Some(session) = &mut slot.transfer
                    && !session.expired
                    && now.duration_since(session.created_at) >= self.lock_timeout
                {
                    session.expired = true;
                    expired_count += 1;
                    let block_count = match &session.state {
                        TransferState::Staging { .. } => session.block_count,
                        TransferState::Ready(blocks) => blocks.len(),
                    };
                    warn!(
                        "Transfer overdue, retaining source memory: window={} slot={} generation={} requester={} blocks={} age={:?}",
                        id,
                        index,
                        slot.generation,
                        window.requester,
                        block_count,
                        now.duration_since(session.created_at)
                    );
                }
            }
        }
        if expired_count > 0 {
            core_metrics()
                .transfer_expired_sessions
                .add(expired_count as i64, &[]);
            core_metrics()
                .transfer_lock_timeouts_total
                .add(expired_count as u64, &[]);
        }
        expired_count
    }
}

fn allocation_bytes(blocks: &[(StateKey, Arc<SealedBlock>)]) -> Result<u64, PeerError> {
    let allocations: HashMap<_, _> = blocks
        .iter()
        .flat_map(|(_, block)| block.pinned_allocations())
        .collect();
    allocations
        .values()
        .try_fold(0u64, |sum, size| sum.checked_add(*size))
        .ok_or(PeerError::BudgetExhausted)
}

fn release_accounting(inner: &mut Transfers, session: &TransferSession) {
    inner.active -= 1;
    inner.reserved_bytes -= session.reserved_bytes;
    core_metrics()
        .transfer_reserved_bytes
        .add(-(session.reserved_bytes as i64), &[]);
    if session.expired {
        core_metrics().transfer_expired_sessions.add(-1, &[]);
    }
    core_metrics()
        .transfer_lock_active
        .add(-(session.block_count as i64), &[]);
}

#[cfg(test)]
#[path = "../../tests/unit/peer/export.rs"]
mod tests;
