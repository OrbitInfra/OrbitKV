use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use orbitkv_state::CacheOwner;
use parking_lot::RwLock;

/// Cached membership evidence and this runtime's conservative registration deadline.
/// The control-plane adapter updates it; cache operations perform no coordinator I/O.
pub struct MembershipView {
    owner: CacheOwner,
    state: RwLock<View>,
}

#[derive(Default)]
struct View {
    members: HashMap<uuid::Uuid, MemberLease>,
    ready: bool,
    valid_until: Option<Instant>,
    fenced: bool,
}

/// Admission authority for one installed membership identity. Revocation is
/// permanent even if a later snapshot contains the same endpoint/incarnation.
#[derive(Clone)]
pub struct MemberLease(Arc<Member>);

struct Member {
    node: String,
    epoch: u64,
    owner: CacheOwner,
    revoked: AtomicBool,
}

impl MemberLease {
    pub fn owner(&self) -> &CacheOwner {
        &self.0.owner
    }

    pub fn revoked(&self) -> bool {
        self.0.revoked.load(Ordering::Acquire)
    }
}

impl PartialEq for MemberLease {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for MemberLease {}
impl Hash for MemberLease {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl View {
    fn fence(&mut self) {
        self.fenced = true;
        for member in self.members.values() {
            member.0.revoked.store(true, Ordering::Release);
        }
    }

    fn permits(&self) -> bool {
        !self.fenced && self.ready && self.valid_until.is_some_and(|until| Instant::now() < until)
    }
}

impl MembershipView {
    pub fn new(owner: CacheOwner) -> Self {
        Self {
            owner,
            state: RwLock::new(View::default()),
        }
    }

    pub fn owner(&self) -> &CacheOwner {
        &self.owner
    }

    pub fn resolve(&self, incarnation: uuid::Uuid) -> Option<CacheOwner> {
        let state = self.state.read();
        if state.fenced
            || !state.ready
            || state
                .valid_until
                .is_none_or(|until| Instant::now() >= until)
        {
            return None;
        }
        state
            .members
            .get(&incarnation)
            .filter(|member| !member.revoked())
            .map(|member| member.owner().clone())
    }

    /// A delayed acknowledgement cannot revive an expired runtime. Half the
    /// returned TTL is reserved for response delay and lease-clock uncertainty.
    pub fn renew(&self, sent_at: Instant, ttl: Duration) -> bool {
        let now = Instant::now();
        let mut state = self.state.write();
        let deadline = sent_at.checked_add(ttl / 2);
        if state.fenced
            || state.valid_until.is_some_and(|until| now >= until)
            || deadline.is_none_or(|until| now >= until)
        {
            state.fence();
            return false;
        }
        state.valid_until = deadline;
        true
    }

    pub fn replace_members(&self, members: impl IntoIterator<Item = (String, u64, CacheOwner)>) {
        let mut state = self.state.write();
        let next: HashMap<_, _> = members
            .into_iter()
            .map(|(node, epoch, owner)| {
                let member = state
                    .members
                    .get(&owner.incarnation)
                    .filter(|member| {
                        member.0.node == node
                            && member.0.epoch == epoch
                            && member.owner() == &owner
                            && !member.revoked()
                    })
                    .cloned()
                    .unwrap_or_else(|| {
                        MemberLease(Arc::new(Member {
                            node,
                            epoch,
                            owner: owner.clone(),
                            revoked: AtomicBool::new(false),
                        }))
                    });
                (owner.incarnation, member)
            })
            .collect();
        for (incarnation, member) in &state.members {
            if next.get(incarnation) != Some(member) {
                member.0.revoked.store(true, Ordering::Release);
            }
        }
        state.members = next;
        state.ready = true;
        if state.fenced
            || state
                .members
                .get(&self.owner.incarnation)
                .map(MemberLease::owner)
                != Some(&self.owner)
        {
            state.fence();
        }
    }

    /// Watch reconnect/repair suspends discovery until a complete snapshot arrives.
    pub fn invalidate_snapshot(&self) {
        let mut state = self.state.write();
        state.ready = false;
        // Preserve identities while discovery is suspended. Only an authoritative
        // replacement snapshot, or permanent local fencing, revokes them.
    }

    pub fn fence(&self) {
        self.state.write().fence();
    }

    /// Revoke an observed Watch transition before its final snapshot is installed.
    /// A delete/rejoin in one response must not revive the preceding grants.
    pub fn revoke(&self, owner: &CacheOwner) {
        if let Some(member) = self.state.read().members.get(&owner.incarnation)
            && member.owner() == owner
        {
            member.0.revoked.store(true, Ordering::Release);
        }
    }

    pub fn registration_valid(&self) -> bool {
        let state = self.state.read();
        !state.fenced
            && state
                .valid_until
                .is_some_and(|until| Instant::now() < until)
    }

    pub fn permits(&self, owner: &CacheOwner) -> bool {
        self.lease(owner).is_some()
    }

    pub fn lease(&self, owner: &CacheOwner) -> Option<MemberLease> {
        let state = self.state.read();
        if !state.permits() {
            return None;
        }
        state
            .members
            .get(&owner.incarnation)
            .filter(|member| member.owner() == owner && !member.revoked())
            .cloned()
    }

    pub fn permits_lease(&self, member: &MemberLease) -> bool {
        let state = self.state.read();
        state.permits()
            && !member.revoked()
            && state.members.get(&member.owner().incarnation) == Some(member)
    }
}

#[cfg(test)]
#[path = "../tests/unit/membership.rs"]
mod tests;
