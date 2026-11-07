use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use orbitkv_state::{
    BlockCandidates, DISCOVERY_MAX_REPLICAS_PER_MEDIUM, DiscoveryCoverage, InventoryRecord,
    InventoryScope, ReplicaLocation, ReplicaMedium, ReplicaMetadata, ReplicaRepresentation,
    StateKey,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

use crate::MembershipView;

pub const DEFAULT_INDEX_BYTES: usize = 256 * 1024 * 1024;
const OWNER_ACCOUNTING_BYTES: usize = 256;

type Residence = (StateKey, ReplicaMedium);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaApply {
    Applied,
    Duplicate,
    Overlap { applied_sequence: u64 },
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct OwnerIndexStatus {
    pub owner: Uuid,
    pub view_id: Uuid,
    pub applied_sequence: u64,
    pub installed_mono_ns: u64,
    pub fresh: bool,
    pub receipt_age_ms: u64,
    pub records: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct IndexStatus {
    pub membership_revision: Option<i64>,
    pub accounted_bytes: usize,
    pub active_bytes: usize,
    pub staging_bytes: usize,
    pub expected_owner_views: usize,
    pub installed_owner_views: usize,
    pub view_generation: u64,
    pub registration_valid: bool,
    pub coverage: DiscoveryCoverage,
    pub scope_kind: &'static str,
    pub scope_namespaces: usize,
    pub lookup_lock_wait_ns: u64,
    pub lookup_lock_hold_ns: u64,
    pub lookup_lock_samples: u64,
    pub update_lock_wait_ns: u64,
    pub update_lock_hold_ns: u64,
    pub update_lock_samples: u64,
}

#[derive(Clone)]
struct Entry {
    owner: Uuid,
    sequence: u64,
    metadata: ReplicaMetadata,
}

struct OwnerView {
    view_id: Uuid,
    applied_sequence: u64,
    installed_mono_ns: u64,
    records: BTreeMap<Residence, InventoryRecord>,
    fresh: bool,
    received_at: Instant,
    retired: bool,
}

struct StagingView {
    session_id: Uuid,
    snapshot_id: Uuid,
    view_id: Uuid,
    start_sequence: u64,
    replay_sequence: u64,
    next_page: u64,
    replay_started: bool,
    last_residence: Option<Residence>,
    max_record_sequence: u64,
    records: BTreeMap<Residence, InventoryRecord>,
    bytes: usize,
}

#[derive(Default)]
struct View {
    blocks: HashMap<StateKey, Vec<Entry>>,
    owners: HashMap<Uuid, OwnerView>,
    staging: HashMap<Uuid, StagingView>,
    expected: HashSet<Uuid>,
    accounted_bytes: usize,
    active_bytes: usize,
    staging_bytes: usize,
    membership_revision: Option<i64>,
    generation: u64,
    admitted_owner_views: usize,
    fresh_expected_owner_views: usize,
}

pub struct GlobalIndex {
    membership: Arc<MembershipView>,
    scope: Arc<InventoryScope>,
    view: RwLock<View>,
    byte_limit: usize,
    lock_metrics: LockMetrics,
}

#[derive(Default)]
struct LockMetrics {
    lookup_wait_ns: AtomicU64,
    lookup_hold_ns: AtomicU64,
    lookup_samples: AtomicU64,
    update_wait_ns: AtomicU64,
    update_hold_ns: AtomicU64,
    update_samples: AtomicU64,
}

// Fields drop in declaration order: unlock before observing the hold duration.
struct TimedRead<'a> {
    guard: RwLockReadGuard<'a, View>,
    _hold: LockHold<'a>,
}

struct LockHold<'a> {
    acquired: Instant,
    nanoseconds: &'a AtomicU64,
}

impl Drop for LockHold<'_> {
    fn drop(&mut self) {
        record_duration(self.nanoseconds, self.acquired.elapsed());
    }
}

impl Deref for TimedRead<'_> {
    type Target = View;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

struct TimedWrite<'a> {
    guard: RwLockWriteGuard<'a, View>,
    _hold: LockHold<'a>,
}

impl Deref for TimedWrite<'_> {
    type Target = View;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for TimedWrite<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl GlobalIndex {
    pub fn new(
        membership: Arc<MembershipView>,
        byte_limit: usize,
        scope: Arc<InventoryScope>,
    ) -> Self {
        Self {
            membership,
            scope,
            view: RwLock::new(View::default()),
            byte_limit,
            lock_metrics: LockMetrics::default(),
        }
    }

    pub fn scope(&self) -> Arc<InventoryScope> {
        Arc::clone(&self.scope)
    }

    fn read_for_lookup(&self) -> TimedRead<'_> {
        let waiting = Instant::now();
        let guard = self.view.read();
        record_duration(&self.lock_metrics.lookup_wait_ns, waiting.elapsed());
        self.lock_metrics
            .lookup_samples
            .fetch_add(1, Ordering::Relaxed);
        TimedRead {
            guard,
            _hold: LockHold {
                acquired: Instant::now(),
                nanoseconds: &self.lock_metrics.lookup_hold_ns,
            },
        }
    }

    fn write_view(&self) -> TimedWrite<'_> {
        let waiting = Instant::now();
        let guard = self.view.write();
        record_duration(&self.lock_metrics.update_wait_ns, waiting.elapsed());
        self.lock_metrics
            .update_samples
            .fetch_add(1, Ordering::Relaxed);
        TimedWrite {
            guard,
            _hold: LockHold {
                acquired: Instant::now(),
                nanoseconds: &self.lock_metrics.update_hold_ns,
            },
        }
    }

    pub fn reset(&self) {
        *self.write_view() = View::default();
    }

    pub fn set_expected_owners(&self, revision: i64, owners: impl IntoIterator<Item = Uuid>) {
        let local = self.membership.owner().incarnation;
        let expected = owners
            .into_iter()
            .filter(|owner| *owner != local)
            .collect::<HashSet<_>>();
        let mut view = self.write_view();
        if view
            .membership_revision
            .is_some_and(|current| revision < current)
        {
            return;
        }
        for (owner, owner_view) in &mut view.owners {
            if !expected.contains(owner) {
                owner_view.retired = true;
                owner_view.fresh = false;
            }
        }
        let retired_staging = view
            .staging
            .iter()
            .filter(|(owner, _)| !expected.contains(owner))
            .map(|(owner, staging)| (*owner, staging.bytes))
            .collect::<Vec<_>>();
        for (owner, bytes) in retired_staging {
            view.staging.remove(&owner);
            view.staging_bytes -= bytes;
            view.accounted_bytes -= bytes;
        }
        view.expected = expected;
        view.membership_revision = Some(revision);
        refresh_coverage_counts(&mut view);
        view.generation = view.generation.saturating_add(1);
    }

    pub fn mark_syncing(&self, owner: Uuid) {
        let mut view = self.write_view();
        let expected = view.expected.contains(&owner);
        let became_stale = view.owners.get_mut(&owner).is_some_and(|owner_view| {
            let became_stale = owner_view.fresh && !owner_view.retired && expected;
            owner_view.fresh = false;
            became_stale
        });
        if became_stale {
            view.fresh_expected_owner_views -= 1;
        }
        view.generation = view.generation.saturating_add(1);
    }

    pub fn mark_stale(&self, owner: Uuid) {
        self.mark_syncing(owner);
    }

    pub fn confirm_progress(
        &self,
        owner: Uuid,
        view_id: Uuid,
        applied_sequence: u64,
    ) -> Result<(), String> {
        let mut view = self.write_view();
        let expected = view.expected.contains(&owner);
        let became_fresh = {
            let owner_view = view
                .owners
                .get_mut(&owner)
                .ok_or("owner view is not installed")?;
            if owner_view.retired
                || owner_view.view_id != view_id
                || owner_view.applied_sequence != applied_sequence
            {
                return Err("owner progress does not match the installed view".into());
            }
            let became_fresh = !owner_view.fresh && expected;
            owner_view.fresh = true;
            owner_view.received_at = Instant::now();
            became_fresh
        };
        if became_fresh {
            view.fresh_expected_owner_views += 1;
        }
        view.generation = view.generation.saturating_add(1);
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "snapshot identity is deliberately explicit at the catalog boundary"
    )]
    pub fn begin_snapshot(
        &self,
        owner: Uuid,
        session_id: Uuid,
        snapshot_id: Uuid,
        view_id: Uuid,
        start_sequence: u64,
    ) -> Result<(), String> {
        if owner.is_nil() || session_id.is_nil() || snapshot_id.is_nil() || view_id.is_nil() {
            return Err("nil inventory snapshot identity".into());
        }
        let mut view = self.write_view();
        if !view.expected.contains(&owner) {
            return Err("snapshot source is not an expected member".into());
        }
        if let Some(previous) = view.staging.remove(&owner) {
            view.staging_bytes -= previous.bytes;
            view.accounted_bytes -= previous.bytes;
        }
        if view.accounted_bytes + OWNER_ACCOUNTING_BYTES > self.byte_limit {
            return Err("inventory staging exceeds metadata budget".into());
        }
        view.staging.insert(
            owner,
            StagingView {
                session_id,
                snapshot_id,
                view_id,
                start_sequence,
                replay_sequence: start_sequence,
                next_page: 0,
                replay_started: false,
                last_residence: None,
                max_record_sequence: 0,
                records: BTreeMap::new(),
                bytes: OWNER_ACCOUNTING_BYTES,
            },
        );
        view.staging_bytes += OWNER_ACCOUNTING_BYTES;
        view.accounted_bytes += OWNER_ACCOUNTING_BYTES;
        let expected = view.expected.contains(&owner);
        let became_stale = view.owners.get_mut(&owner).is_some_and(|owner_view| {
            let became_stale = owner_view.fresh && !owner_view.retired && expected;
            owner_view.fresh = false;
            became_stale
        });
        if became_stale {
            view.fresh_expected_owner_views -= 1;
        }
        view.generation = view.generation.saturating_add(1);
        Ok(())
    }

    pub fn apply_snapshot_page(
        &self,
        owner: Uuid,
        session_id: Uuid,
        snapshot_id: Uuid,
        page_number: u64,
        records: Vec<InventoryRecord>,
    ) -> Result<(), String> {
        let mut view = self.write_view();
        let (additional, updates) = {
            let staging = view
                .staging
                .get(&owner)
                .ok_or("missing inventory snapshot")?;
            if staging.session_id != session_id || staging.snapshot_id != snapshot_id {
                return Err("inventory snapshot identity changed".into());
            }
            if staging.replay_started {
                return Err("inventory snapshot page arrived after replay began".into());
            }
            if staging.next_page != page_number {
                return Err("inventory snapshot page is not contiguous".into());
            }
            validate_page(staging.last_residence.as_ref(), records, &self.scope)?
        };
        if view.accounted_bytes + additional > self.byte_limit {
            return Err("inventory staging exceeds metadata budget".into());
        }
        let staging = view
            .staging
            .get_mut(&owner)
            .ok_or("missing validated staging")?;
        for (residence, record) in updates {
            staging.max_record_sequence = staging.max_record_sequence.max(record.sequence);
            staging.last_residence = Some(residence.clone());
            staging.records.insert(residence, record);
        }
        staging.next_page += 1;
        staging.bytes += additional;
        view.staging_bytes += additional;
        view.accounted_bytes += additional;
        Ok(())
    }

    pub fn apply_snapshot_delta(
        &self,
        owner: Uuid,
        session_id: Uuid,
        from_exclusive: u64,
        through_inclusive: u64,
        records: Vec<InventoryRecord>,
    ) -> Result<(), String> {
        let mut view = self.write_view();
        let (delta, updates) = {
            let staging = view
                .staging
                .get(&owner)
                .ok_or("missing inventory snapshot")?;
            if staging.session_id != session_id || staging.replay_sequence != from_exclusive {
                return Err("inventory snapshot replay is not contiguous".into());
            }
            validate_delta(
                &staging.records,
                from_exclusive,
                through_inclusive,
                records,
                &self.scope,
            )?
        };
        if view
            .accounted_bytes
            .checked_add_signed(delta)
            .is_none_or(|bytes| bytes > self.byte_limit)
        {
            return Err("inventory staging exceeds metadata budget".into());
        }
        let staging = view
            .staging
            .get_mut(&owner)
            .ok_or("missing validated staging")?;
        staging.replay_started = true;
        apply_records(&mut staging.records, updates);
        staging.replay_sequence = through_inclusive;
        staging.bytes = staging
            .bytes
            .checked_add_signed(delta)
            .ok_or("inventory staging accounting underflow")?;
        view.staging_bytes = view
            .staging_bytes
            .checked_add_signed(delta)
            .ok_or("inventory staging accounting underflow")?;
        view.accounted_bytes = view
            .accounted_bytes
            .checked_add_signed(delta)
            .ok_or("inventory staging accounting underflow")?;
        Ok(())
    }

    pub fn commit_snapshot(
        &self,
        owner: Uuid,
        session_id: Uuid,
        snapshot_id: Uuid,
        view_id: Uuid,
        through_sequence: u64,
        page_count: u64,
    ) -> Result<Uuid, String> {
        let mut view = self.write_view();
        let staging = view
            .staging
            .remove(&owner)
            .ok_or("missing inventory snapshot")?;
        if staging.session_id != session_id
            || staging.snapshot_id != snapshot_id
            || staging.view_id != view_id
            || staging.replay_sequence != through_sequence
            || staging.next_page != page_count
            || staging.start_sequence > through_sequence
            || staging.max_record_sequence > through_sequence
        {
            view.staging.insert(owner, staging);
            return Err("inventory snapshot commit is incomplete".into());
        }
        if !view.expected.contains(&owner) || !self.membership.resolve(owner).is_some() {
            view.staging_bytes -= staging.bytes;
            view.accounted_bytes -= staging.bytes;
            return Err("inventory snapshot source is no longer admitted".into());
        }
        remove_active_owner(&mut view, owner);
        view.staging_bytes -= staging.bytes;
        let active_bytes = staging.bytes;
        for record in staging.records.values() {
            insert_block(&mut view.blocks, owner, record)?;
        }
        view.active_bytes += active_bytes;
        view.owners.insert(
            owner,
            OwnerView {
                view_id: staging.view_id,
                applied_sequence: through_sequence,
                installed_mono_ns: 0,
                records: staging.records,
                fresh: true,
                received_at: Instant::now(),
                retired: false,
            },
        );
        view.admitted_owner_views += 1;
        if view.expected.contains(&owner) {
            view.fresh_expected_owner_views += 1;
        }
        view.generation = view.generation.saturating_add(1);
        view.owners
            .get_mut(&owner)
            .ok_or("missing committed owner view")?
            .installed_mono_ns = monotonic_ns();
        Ok(view_id)
    }

    pub fn abort_snapshot(&self, owner: Uuid, session_id: Uuid) {
        let mut view = self.write_view();
        if view
            .staging
            .get(&owner)
            .is_some_and(|staging| staging.session_id == session_id)
            && let Some(staging) = view.staging.remove(&owner)
        {
            view.staging_bytes -= staging.bytes;
            view.accounted_bytes -= staging.bytes;
            view.generation = view.generation.saturating_add(1);
        }
    }

    pub fn apply_delta(
        &self,
        owner: Uuid,
        view_id: Uuid,
        from_exclusive: u64,
        through_inclusive: u64,
        records: Vec<InventoryRecord>,
    ) -> Result<DeltaApply, String> {
        let mut view = self.write_view();
        let owner_view = view
            .owners
            .get(&owner)
            .ok_or("owner view is not installed")?;
        if owner_view.view_id != view_id || owner_view.retired {
            return Err("owner view identity changed".into());
        }
        if through_inclusive <= owner_view.applied_sequence {
            return Ok(DeltaApply::Duplicate);
        }
        if from_exclusive < owner_view.applied_sequence {
            return Ok(DeltaApply::Overlap {
                applied_sequence: owner_view.applied_sequence,
            });
        }
        if from_exclusive > owner_view.applied_sequence {
            return Err("inventory delta has a coverage gap".into());
        }
        let (delta, updates) = validate_delta(
            &owner_view.records,
            from_exclusive,
            through_inclusive,
            records,
            &self.scope,
        )?;
        if view
            .accounted_bytes
            .checked_add_signed(delta)
            .is_none_or(|bytes| bytes > self.byte_limit)
        {
            return Err("complete global index exceeds metadata budget".into());
        }
        for (residence, record) in &updates {
            remove_block(&mut view.blocks, owner, residence);
            if record.present {
                insert_block(&mut view.blocks, owner, record)?;
            }
        }
        let expected = view.expected.contains(&owner);
        let became_fresh = {
            let owner_view = view
                .owners
                .get_mut(&owner)
                .ok_or("missing validated owner view")?;
            let became_fresh = !owner_view.fresh && expected;
            apply_records(&mut owner_view.records, updates);
            owner_view.applied_sequence = through_inclusive;
            owner_view.fresh = true;
            owner_view.received_at = Instant::now();
            became_fresh
        };
        if became_fresh {
            view.fresh_expected_owner_views += 1;
        }
        view.active_bytes = view
            .active_bytes
            .checked_add_signed(delta)
            .ok_or("active index accounting underflow")?;
        view.accounted_bytes = view
            .accounted_bytes
            .checked_add_signed(delta)
            .ok_or("index accounting underflow")?;
        view.generation = view.generation.saturating_add(1);
        view.owners
            .get_mut(&owner)
            .ok_or("missing applied owner view")?
            .installed_mono_ns = monotonic_ns();
        Ok(DeltaApply::Applied)
    }

    pub fn owner_watermark(&self, owner: Uuid) -> Option<(Uuid, u64)> {
        self.view.read().owners.get(&owner).and_then(|owner_view| {
            (!owner_view.retired).then_some((owner_view.view_id, owner_view.applied_sequence))
        })
    }

    pub fn owner_status(&self, owner: Uuid) -> Option<OwnerIndexStatus> {
        self.view
            .read()
            .owners
            .get(&owner)
            .map(|view| OwnerIndexStatus {
                owner,
                view_id: view.view_id,
                applied_sequence: view.applied_sequence,
                installed_mono_ns: view.installed_mono_ns,
                fresh: view.fresh && !view.retired,
                receipt_age_ms: u64::try_from(view.received_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
                records: view.records.len(),
            })
    }

    pub fn owner_statuses(&self, after: Option<Uuid>, limit: usize) -> Vec<OwnerIndexStatus> {
        let view = self.view.read();
        let mut owners = view
            .owners
            .iter()
            .filter(|(owner, _)| after.is_none_or(|after| **owner > after))
            .map(|(owner, owner_view)| OwnerIndexStatus {
                owner: *owner,
                view_id: owner_view.view_id,
                applied_sequence: owner_view.applied_sequence,
                installed_mono_ns: owner_view.installed_mono_ns,
                fresh: owner_view.fresh && !owner_view.retired,
                receipt_age_ms: u64::try_from(owner_view.received_at.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
                records: owner_view.records.len(),
            })
            .collect::<Vec<_>>();
        owners.sort_by_key(|status| status.owner);
        owners.truncate(limit.min(128));
        owners
    }

    pub fn retire_owner(&self, owner: Uuid) {
        let mut view = self.write_view();
        if let Some(staging) = view.staging.remove(&owner) {
            view.staging_bytes -= staging.bytes;
            view.accounted_bytes -= staging.bytes;
        }
        let expected = view.expected.contains(&owner);
        let (was_admitted, was_fresh) =
            view.owners
                .get(&owner)
                .map_or((false, false), |owner_view| {
                    (
                        !owner_view.retired,
                        !owner_view.retired && owner_view.fresh && expected,
                    )
                });
        if was_admitted {
            view.admitted_owner_views -= 1;
        }
        if was_fresh {
            view.fresh_expected_owner_views -= 1;
        }
        if let Some(owner_view) = view.owners.get_mut(&owner) {
            owner_view.retired = true;
            owner_view.fresh = false;
        }
        view.generation = view.generation.saturating_add(1);
    }

    pub fn cleanup_owner(&self, owner: Uuid, limit: usize) -> bool {
        let mut view = self.write_view();
        let records = match view.owners.get(&owner) {
            Some(owner_view) if owner_view.retired => owner_view
                .records
                .keys()
                .take(limit.max(1))
                .cloned()
                .collect::<Vec<_>>(),
            _ => return true,
        };
        let mut removed_bytes = 0;
        for residence in &records {
            remove_block(&mut view.blocks, owner, residence);
            if view
                .owners
                .get_mut(&owner)
                .and_then(|owner_view| owner_view.records.remove(residence))
                .is_some()
            {
                removed_bytes += entry_bytes(&residence.0);
            }
        }
        view.active_bytes -= removed_bytes;
        view.accounted_bytes -= removed_bytes;
        let done = view
            .owners
            .get(&owner)
            .is_none_or(|owner_view| owner_view.records.is_empty());
        if done && view.owners.remove(&owner).is_some() {
            view.active_bytes -= OWNER_ACCOUNTING_BYTES;
            view.accounted_bytes -= OWNER_ACCOUNTING_BYTES;
        }
        done
    }

    pub fn status(&self) -> IndexStatus {
        let view = self.view.read();
        let registration_valid = self.membership.registration_valid();
        let coverage = coverage(&view, self.membership.permits(self.membership.owner()));
        IndexStatus {
            membership_revision: view.membership_revision,
            accounted_bytes: view.accounted_bytes,
            active_bytes: view.active_bytes,
            staging_bytes: view.staging_bytes,
            expected_owner_views: view.expected.len(),
            installed_owner_views: view.fresh_expected_owner_views,
            view_generation: view.generation,
            registration_valid,
            coverage,
            scope_kind: self.scope.label(),
            scope_namespaces: self.scope.namespaces().map_or(0, <[String]>::len),
            lookup_lock_wait_ns: self.lock_metrics.lookup_wait_ns.load(Ordering::Relaxed),
            lookup_lock_hold_ns: self.lock_metrics.lookup_hold_ns.load(Ordering::Relaxed),
            lookup_lock_samples: self.lock_metrics.lookup_samples.load(Ordering::Relaxed),
            update_lock_wait_ns: self.lock_metrics.update_wait_ns.load(Ordering::Relaxed),
            update_lock_hold_ns: self.lock_metrics.update_hold_ns.load(Ordering::Relaxed),
            update_lock_samples: self.lock_metrics.update_samples.load(Ordering::Relaxed),
        }
    }

    pub fn bytes(&self) -> usize {
        self.view.read().accounted_bytes
    }

    pub fn lookup(&self, keys: &[StateKey]) -> Vec<BlockCandidates> {
        let view = self.read_for_lookup();
        let discovery_coverage = coverage(&view, self.membership.permits(self.membership.owner()));
        keys.iter()
            .map(|key| {
                if !self.scope.contains(&key.namespace) {
                    return BlockCandidates {
                        key: key.clone(),
                        replicas: Vec::new(),
                        coverage: DiscoveryCoverage::Unavailable,
                    };
                }
                let mut replicas = Vec::new();
                if discovery_coverage != DiscoveryCoverage::Unavailable
                    && let Some(entries) = view.blocks.get(key)
                {
                    for medium in [ReplicaMedium::Dram, ReplicaMedium::Ssd] {
                        let start = replicas.len();
                        for entry in entries
                            .iter()
                            .filter(|entry| entry.metadata.medium == medium)
                        {
                            if let Some(owner) = self.membership.resolve(entry.owner)
                                && owner.incarnation != self.membership.owner().incarnation
                                && view
                                    .owners
                                    .get(&entry.owner)
                                    .is_some_and(|owner_view| !owner_view.retired)
                            {
                                replicas.push(ReplicaLocation {
                                    owner,
                                    sequence: entry.sequence,
                                    metadata: entry.metadata,
                                });
                            }
                            if replicas.len() - start == DISCOVERY_MAX_REPLICAS_PER_MEDIUM {
                                break;
                            }
                        }
                    }
                }
                BlockCandidates {
                    key: key.clone(),
                    replicas,
                    coverage: discovery_coverage,
                }
            })
            .collect()
    }
}

fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid writable timespec and CLOCK_MONOTONIC has no
    // additional pointer lifetime requirements.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) } != 0 {
        return 0;
    }
    u64::try_from(now.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(now.tv_nsec).unwrap_or(0))
}

fn coverage(view: &View, registration_valid: bool) -> DiscoveryCoverage {
    if !registration_valid || view.membership_revision.is_none() {
        DiscoveryCoverage::Unavailable
    } else if view.fresh_expected_owner_views == view.expected.len() {
        DiscoveryCoverage::CompleteAtWatermarks
    } else if view.admitted_owner_views > 0 {
        DiscoveryCoverage::PartialHints
    } else {
        DiscoveryCoverage::Unavailable
    }
}

fn refresh_coverage_counts(view: &mut View) {
    view.admitted_owner_views = view
        .owners
        .values()
        .filter(|owner_view| !owner_view.retired)
        .count();
    view.fresh_expected_owner_views = view
        .expected
        .iter()
        .filter(|owner| {
            view.owners
                .get(owner)
                .is_some_and(|owner_view| !owner_view.retired && owner_view.fresh)
        })
        .count();
}

fn record_duration(counter: &AtomicU64, duration: std::time::Duration) {
    counter.fetch_add(
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

fn validate_page(
    previous: Option<&Residence>,
    records: Vec<InventoryRecord>,
    scope: &InventoryScope,
) -> Result<(usize, Vec<(Residence, InventoryRecord)>), String> {
    let mut last = previous.cloned();
    let mut bytes = 0usize;
    let mut updates = Vec::with_capacity(records.len());
    for record in records {
        if !scope.contains(&record.key.namespace) {
            return Err("inventory record is outside the installed scope".into());
        }
        let residence = validate_record(&record)?;
        if !record.present || last.as_ref().is_some_and(|last| residence <= *last) {
            return Err("inventory snapshot page is unsorted or contains a deletion".into());
        }
        bytes = bytes
            .checked_add(entry_bytes(&record.key))
            .ok_or("inventory staging byte overflow")?;
        last = Some(residence.clone());
        updates.push((residence, record));
    }
    Ok((bytes, updates))
}

fn validate_delta(
    current: &BTreeMap<Residence, InventoryRecord>,
    from_exclusive: u64,
    through_inclusive: u64,
    records: Vec<InventoryRecord>,
    scope: &InventoryScope,
) -> Result<(isize, Vec<(Residence, InventoryRecord)>), String> {
    if from_exclusive >= through_inclusive {
        return Err("inventory delta interval is empty or reversed".into());
    }
    let mut latest = BTreeMap::<Residence, InventoryRecord>::new();
    for record in records {
        if !scope.contains(&record.key.namespace) {
            return Err("inventory record is outside the installed scope".into());
        }
        if record.sequence <= from_exclusive || record.sequence > through_inclusive {
            return Err("inventory record lies outside its covered interval".into());
        }
        let residence = validate_record(&record)?;
        match latest.get(&residence) {
            Some(previous) if previous.sequence > record.sequence => continue,
            Some(previous) if previous.sequence == record.sequence && previous != &record => {
                return Err("conflicting inventory records at one generation".into());
            }
            _ => {
                latest.insert(residence, record);
            }
        }
    }
    let mut delta = 0isize;
    for (residence, record) in &latest {
        if let Some(previous) = current.get(residence) {
            if previous.sequence > record.sequence {
                continue;
            }
            if previous.sequence == record.sequence && previous != record {
                return Err("conflicting inventory generation".into());
            }
            if !record.present {
                delta -= isize::try_from(entry_bytes(&record.key)).unwrap_or(isize::MAX);
            }
        } else if record.present {
            delta += isize::try_from(entry_bytes(&record.key)).unwrap_or(isize::MAX);
        }
    }
    Ok((delta, latest.into_iter().collect()))
}

fn validate_record(record: &InventoryRecord) -> Result<Residence, String> {
    let metadata = record.metadata.ok_or("missing residency metadata")?;
    if record.sequence == 0
        || record.key.namespace.is_empty()
        || record.key.hash.is_empty()
        || !matches!(metadata.medium, ReplicaMedium::Dram | ReplicaMedium::Ssd)
        || metadata.representation == ReplicaRepresentation::Unknown
    {
        return Err("invalid global residency record".into());
    }
    Ok((record.key.clone(), metadata.medium))
}

fn apply_records(
    records: &mut BTreeMap<Residence, InventoryRecord>,
    updates: Vec<(Residence, InventoryRecord)>,
) {
    for (residence, record) in updates {
        if records
            .get(&residence)
            .is_some_and(|previous| previous.sequence > record.sequence)
        {
            continue;
        }
        if record.present {
            records.insert(residence, record);
        } else {
            records.remove(&residence);
        }
    }
}

fn insert_block(
    blocks: &mut HashMap<StateKey, Vec<Entry>>,
    owner: Uuid,
    record: &InventoryRecord,
) -> Result<(), String> {
    if !record.present {
        return Ok(());
    }
    let metadata = record.metadata.ok_or("missing record metadata")?;
    blocks.entry(record.key.clone()).or_default().push(Entry {
        owner,
        sequence: record.sequence,
        metadata,
    });
    Ok(())
}

fn remove_block(blocks: &mut HashMap<StateKey, Vec<Entry>>, owner: Uuid, residence: &Residence) {
    let remove_key = if let Some(entries) = blocks.get_mut(&residence.0) {
        entries.retain(|entry| !(entry.owner == owner && entry.metadata.medium == residence.1));
        entries.is_empty()
    } else {
        false
    };
    if remove_key {
        blocks.remove(&residence.0);
    }
}

fn remove_active_owner(view: &mut View, owner: Uuid) {
    let Some(previous) = view.owners.remove(&owner) else {
        return;
    };
    if !previous.retired {
        view.admitted_owner_views -= 1;
        if previous.fresh && view.expected.contains(&owner) {
            view.fresh_expected_owner_views -= 1;
        }
    }
    for residence in previous.records.keys() {
        remove_block(&mut view.blocks, owner, residence);
    }
    let bytes = OWNER_ACCOUNTING_BYTES
        + previous
            .records
            .keys()
            .map(|(key, _)| entry_bytes(key))
            .sum::<usize>();
    view.active_bytes -= bytes;
    view.accounted_bytes -= bytes;
}

fn entry_bytes(key: &StateKey) -> usize {
    // Charge active hash entries, the reverse-owner entry and allocation headroom.
    320 + key.namespace.len() + key.hash.len()
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
