//! Requester-owned peer identities and membership retirement.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use orbitkv_catalog::{MemberLease, MembershipView};
use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::transfer_engine::{
    Cancellation, ErrorKind, PeerSegment, TransferEngine, TransferError, TransferOp,
    TransferOptions, TransferSlice,
};

const MAX_IDENTITIES: usize = 1024;
const RETIRE_POLL: Duration = Duration::from_millis(100);
const RETIRE_WAIT: Duration = Duration::from_millis(100);

#[derive(Clone, PartialEq, Eq, Hash)]
struct Identity {
    member: MemberLease,
    endpoint: String,
    epoch: String,
}

pub(super) struct PeerSegments {
    backend: Arc<TransferEngine>,
    membership: Arc<MembershipView>,
    entries: Mutex<HashMap<Identity, Arc<PeerIdentity>>>,
    budget: Arc<Semaphore>,
}

pub(super) struct PeerIdentity {
    identity: Identity,
    pub(super) backend: Arc<TransferEngine>,
    membership: Arc<MembershipView>,
    state: Mutex<State>,
    active: AtomicUsize,
    permit: Option<OwnedSemaphorePermit>,
}

#[derive(Default)]
struct State {
    peer: Option<Arc<PeerSegment>>,
    attempted: bool,
    retiring: bool,
    drained: bool,
    reported: bool,
}

/// Keeps the requester descriptor and its admission until the transfer returns
/// a physical drain receipt. An unknown outcome retains this admission too.
pub(crate) struct PeerOperation {
    identity: Arc<PeerIdentity>,
    peer: Arc<PeerSegment>,
}

impl PeerSegments {
    #[allow(
        clippy::mem_forget,
        reason = "a failed retirement worker retains its native owners"
    )]
    pub(super) fn new(backend: Arc<TransferEngine>, membership: Arc<MembershipView>) -> Arc<Self> {
        let segments = Arc::new(Self {
            backend,
            membership,
            entries: Mutex::new(HashMap::new()),
            budget: Arc::new(Semaphore::new(MAX_IDENTITIES)),
        });
        let worker = Arc::clone(&segments);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(RETIRE_POLL).await;
                let stopping = Arc::strong_count(&worker) == 1;
                let sweep = Arc::clone(&worker);
                // At most one blocking retirement sweep per requester. A native
                // destroy is not preemptible; it never blocks transfer admission
                // to other identities or creates additional cleanup workers.
                if tokio::task::spawn_blocking(move || sweep.retire(stopping))
                    .await
                    .is_err()
                {
                    log::error!("Peer retirement worker failed; retaining its owners");
                    std::mem::forget(worker);
                    return;
                }
                if stopping && worker.entries.lock().is_empty() {
                    return;
                }
            }
        });
        segments
    }

    pub(super) fn get(
        &self,
        member: MemberLease,
        endpoint: &str,
        epoch: &str,
    ) -> Result<Arc<PeerIdentity>, TransferError> {
        if !self.membership.permits_lease(&member) {
            return Err(fenced());
        }
        if endpoint.len() > 256 || epoch.is_empty() || epoch.len() > 64 {
            return Err(TransferError::new(
                ErrorKind::InvalidRequest,
                "invalid peer identity",
            ));
        }
        let identity = Identity {
            member,
            endpoint: endpoint.into(),
            epoch: epoch.into(),
        };
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.get(&identity) {
            return Ok(Arc::clone(entry));
        }
        let permit = Arc::clone(&self.budget).try_acquire_owned().map_err(|_| {
            TransferError::new(
                ErrorKind::ResourceExhausted,
                "peer identity budget exhausted",
            )
        })?;
        let entry = Arc::new(PeerIdentity {
            identity: identity.clone(),
            backend: Arc::clone(&self.backend),
            membership: Arc::clone(&self.membership),
            state: Mutex::new(State::default()),
            active: AtomicUsize::new(0),
            permit: Some(permit),
        });
        entries.insert(identity, Arc::clone(&entry));
        Ok(entry)
    }

    fn retire(&self, stopping: bool) {
        let expired = !self.membership.registration_valid();
        let entries: Vec<_> = self
            .entries
            .lock()
            .iter()
            .filter(|(key, _)| stopping || expired || key.member.revoked())
            .map(|(key, entry)| (key.clone(), Arc::clone(entry)))
            .collect();
        for (key, entry) in entries {
            if entry.retire() {
                self.entries.lock().remove(&key);
            }
        }
    }
}

impl PeerIdentity {
    pub(super) fn source_id(&self) -> u64 {
        crate::cost::resource_id(&(&self.identity.endpoint, &self.identity.epoch))
    }

    pub(super) fn open(
        self: &Arc<Self>,
        deadline: Instant,
    ) -> Result<PeerOperation, TransferError> {
        if !self.membership.permits_lease(&self.identity.member) {
            return Err(fenced());
        }
        let mut state = self.state.lock();
        if state.retiring || !self.membership.permits_lease(&self.identity.member) {
            return Err(fenced());
        }
        if self.active.load(Ordering::Acquire) >= MAX_IDENTITIES {
            return Err(TransferError::new(
                ErrorKind::ResourceExhausted,
                "peer operation budget exhausted",
            ));
        }
        // Grants can re-register the same source address with a fresh region
        // key. Preserve per-restore Describe; cache the identity, not stale keys.
        state.attempted = true;
        state.peer = Some(Arc::new(self.backend.open_peer(
            &self.identity.endpoint,
            &self.identity.epoch,
            deadline,
        )?));
        // A membership update can revoke authority while Describe is pending.
        // Keep the opened handle for retirement, but admit no payload with it.
        if !self.membership.permits_lease(&self.identity.member) {
            return Err(fenced());
        }
        let peer = Arc::clone(state.peer.as_ref().ok_or_else(fenced)?);
        self.active.fetch_add(1, Ordering::AcqRel);
        Ok(PeerOperation {
            identity: Arc::clone(self),
            peer,
        })
    }

    fn retire(&self) -> bool {
        let Some(mut state) = self.state.try_lock() else {
            return false;
        };
        state.retiring = true;
        if state.drained {
            return true;
        }
        let result = if state.attempted {
            self.backend.retire_peer(
                &self.identity.endpoint,
                &self.identity.epoch,
                RETIRE_WAIT,
                self.active.load(Ordering::Acquire) == 0,
            )
        } else {
            Ok(())
        };
        match result {
            Ok(()) => {
                state.drained = true;
                // Query grants may still hold this identity, but must not keep
                // a retired native descriptor charged against engine capacity.
                state.peer.take();
                true
            }
            Err(error) => {
                if !state.reported {
                    log::warn!(
                        "Retaining peer identity {} epoch {} after retirement: {error}",
                        self.identity.endpoint,
                        self.identity.epoch
                    );
                    state.reported = true;
                }
                false
            }
        }
    }
}

impl PeerOperation {
    /// # Safety
    /// The source grant, destination backing and registrations must remain held
    /// through physical drain, including retention after Unquiesced.
    #[allow(
        clippy::mem_forget,
        reason = "unknown drain retains the descriptor and admission"
    )]
    pub(crate) unsafe fn transfer(
        self,
        operation: TransferOp,
        slices: &[TransferSlice],
        options: TransferOptions,
        cancellation: &Cancellation,
    ) -> Result<usize, TransferError> {
        if !self
            .identity
            .membership
            .permits_lease(&self.identity.identity.member)
        {
            return Err(fenced());
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the caller holds byte owners; this guard retains the
            // requester descriptor and its admission through native drain.
            unsafe {
                self.identity
                    .backend
                    .transfer(operation, &self.peer, slices, options, cancellation)
            }
        }));
        match result {
            Ok(result) if !matches!(&result, Err(e) if e.kind == ErrorKind::Unquiesced) => result,
            _ => {
                std::mem::forget(self);
                Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "peer operation did not establish drain",
                ))
            }
        }
    }
}

impl Drop for PeerOperation {
    fn drop(&mut self) {
        self.identity.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Drop for PeerIdentity {
    #[allow(
        clippy::mem_forget,
        reason = "runtime teardown is not a physical drain receipt"
    )]
    fn drop(&mut self) {
        let state = self.state.get_mut();
        if state.attempted && !state.drained {
            // Runtime teardown can abort the retirement waiter. It is not a
            // native receipt and cannot release its descriptor or budget.
            std::mem::forget((
                state.peer.take(),
                Arc::clone(&self.backend),
                self.permit.take(),
            ));
        }
    }
}

fn fenced() -> TransferError {
    TransferError::new(ErrorKind::Unavailable, "peer membership identity is fenced")
}

#[cfg(test)]
#[path = "../../tests/unit/peer/segment.rs"]
pub(crate) mod tests;
