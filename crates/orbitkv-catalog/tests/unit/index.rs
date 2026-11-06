use super::*;
use orbitkv_state::CacheOwner;
use std::time::{Duration, Instant};

fn owner(port: u16) -> CacheOwner {
    CacheOwner {
        endpoint: format!("127.0.0.1:{port}"),
        incarnation: Uuid::new_v4(),
    }
}

fn setup(limit: usize) -> (GlobalIndex, Arc<MembershipView>, CacheOwner) {
    setup_scope(limit, InventoryScope::AllNamespaces)
}

fn setup_scope(
    limit: usize,
    scope: InventoryScope,
) -> (GlobalIndex, Arc<MembershipView>, CacheOwner) {
    let local = owner(51001);
    let remote = owner(51002);
    let membership = Arc::new(MembershipView::new(local.clone()));
    assert!(membership.renew(Instant::now(), Duration::from_secs(60)));
    membership.replace_members([
        ("local".into(), 1, local),
        ("remote".into(), 1, remote.clone()),
    ]);
    let index = GlobalIndex::new(membership.clone(), limit, Arc::new(scope));
    index.set_expected_owners(10, [remote.incarnation]);
    (index, membership, remote)
}

fn record(key: u8, sequence: u64, medium: ReplicaMedium, present: bool) -> InventoryRecord {
    record_in("model", key, sequence, medium, present)
}

fn record_in(
    namespace: &str,
    key: u8,
    sequence: u64,
    medium: ReplicaMedium,
    present: bool,
) -> InventoryRecord {
    InventoryRecord {
        key: StateKey::new(namespace.into(), vec![key]),
        sequence,
        present,
        metadata: Some(ReplicaMetadata {
            medium,
            representation: ReplicaRepresentation::Raw,
            stored_bytes: Some(1024),
        }),
    }
}

#[test]
fn scoped_index_distinguishes_empty_views_and_rejects_cross_scope_rows() {
    let wanted = "orbitkv:v2:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let outside = "orbitkv:v2:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let (index, _, remote) = setup_scope(1 << 20, InventoryScope::exact([wanted.into()]).unwrap());
    let view = install(
        &index,
        &remote,
        vec![record_in(wanted, 1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    let wanted_key = record_in(wanted, 1, 1, ReplicaMedium::Dram, true).key;
    let outside_key = record_in(outside, 1, 1, ReplicaMedium::Dram, true).key;
    assert_eq!(index.lookup(&[wanted_key])[0].replicas.len(), 1);
    let outside_row = &index.lookup(&[outside_key])[0];
    assert!(outside_row.replicas.is_empty());
    assert_eq!(outside_row.coverage, DiscoveryCoverage::Unavailable);
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 1)));

    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, Uuid::new_v4(), 1)
        .unwrap();
    assert!(
        index
            .apply_snapshot_page(
                remote.incarnation,
                session,
                snapshot,
                0,
                vec![record_in(outside, 1, 2, ReplicaMedium::Dram, true)],
            )
            .is_err()
    );

    let (empty, _, empty_remote) = setup_scope(1 << 20, InventoryScope::exact(Vec::new()).unwrap());
    let empty_view = install(&empty, &empty_remote, Vec::new(), 1);
    assert_eq!(
        empty.owner_watermark(empty_remote.incarnation),
        Some((empty_view, 1))
    );
    assert_eq!(
        empty
            .owner_status(empty_remote.incarnation)
            .unwrap()
            .records,
        0
    );
    assert_eq!(
        empty.status().coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
    assert_eq!(
        empty.lookup(&[record(1, 1, ReplicaMedium::Dram, true).key])[0].coverage,
        DiscoveryCoverage::Unavailable
    );
    let status = empty.status();
    assert!(status.lookup_lock_samples > 0);
    assert!(status.update_lock_samples > 0);
}

#[test]
fn member_transitions_update_cached_coverage_without_treating_missing_as_empty() {
    let (index, membership, remote) = setup(1 << 20);
    install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    assert_eq!(
        index.status().coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
    let added = owner(51003);
    membership.replace_members([
        ("local".into(), 1, membership.owner().clone()),
        ("remote".into(), 1, remote.clone()),
        ("added".into(), 1, added.clone()),
    ]);
    index.set_expected_owners(
        11,
        [
            membership.owner().incarnation,
            remote.incarnation,
            added.incarnation,
        ],
    );
    let status = index.status();
    assert_eq!(status.expected_owner_views, 2);
    assert_eq!(status.installed_owner_views, 1);
    assert_eq!(status.coverage, DiscoveryCoverage::PartialHints);
    install(&index, &added, Vec::new(), 1);
    assert_eq!(
        index.status().coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );

    let replacement = owner(51004);
    membership.replace_members([
        ("local".into(), 1, membership.owner().clone()),
        ("added".into(), 1, added.clone()),
        ("remote".into(), 1, replacement.clone()),
    ]);
    index.set_expected_owners(
        12,
        [
            membership.owner().incarnation,
            added.incarnation,
            replacement.incarnation,
        ],
    );
    index.retire_owner(remote.incarnation);
    let status = index.status();
    assert_eq!(status.expected_owner_views, 2);
    assert_eq!(status.installed_owner_views, 1);
    assert_eq!(status.coverage, DiscoveryCoverage::PartialHints);
    install(&index, &replacement, Vec::new(), 1);
    assert_eq!(
        index.status().coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
    while !index.cleanup_owner(remote.incarnation, 1) {}
    assert_eq!(
        index.lookup(&[record(1, 1, ReplicaMedium::Dram, true).key])[0]
            .replicas
            .len(),
        0
    );
}

fn install(
    index: &GlobalIndex,
    remote: &CacheOwner,
    records: Vec<InventoryRecord>,
    through: u64,
) -> Uuid {
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    let view = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, view, through)
        .unwrap();
    index
        .apply_snapshot_page(remote.incarnation, session, snapshot, 0, records)
        .unwrap();
    index
        .commit_snapshot(remote.incarnation, session, snapshot, view, through, 1)
        .unwrap();
    view
}

#[test]
fn hidden_snapshot_installs_both_media_atomically() {
    let (index, _, remote) = setup(1 << 20);
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    let view = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, view, 2)
        .unwrap();
    index
        .apply_snapshot_page(
            remote.incarnation,
            session,
            snapshot,
            0,
            vec![
                record(1, 1, ReplicaMedium::Dram, true),
                record(1, 2, ReplicaMedium::Ssd, true),
            ],
        )
        .unwrap();
    let key = record(1, 1, ReplicaMedium::Dram, true).key;
    assert!(
        index.lookup(std::slice::from_ref(&key))[0]
            .replicas
            .is_empty()
    );
    assert_eq!(index.status().coverage, DiscoveryCoverage::Unavailable);
    index
        .commit_snapshot(remote.incarnation, session, snapshot, view, 2, 1)
        .unwrap();
    let candidates = index.lookup(&[key]);
    assert_eq!(candidates[0].replicas.len(), 2);
    assert_eq!(
        candidates[0].coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 2)));
}

#[test]
fn replay_uses_generations_and_never_resurrects_a_deleted_scan_row() {
    let (index, _, remote) = setup(1 << 20);
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    let view = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, view, 2)
        .unwrap();
    index
        .apply_snapshot_page(
            remote.incarnation,
            session,
            snapshot,
            0,
            vec![
                record(1, 3, ReplicaMedium::Dram, true),
                record(2, 2, ReplicaMedium::Dram, true),
            ],
        )
        .unwrap();
    index
        .apply_snapshot_delta(
            remote.incarnation,
            session,
            2,
            4,
            vec![
                record(1, 3, ReplicaMedium::Dram, true),
                record(2, 4, ReplicaMedium::Dram, false),
            ],
        )
        .unwrap();
    index
        .commit_snapshot(remote.incarnation, session, snapshot, view, 4, 1)
        .unwrap();
    let rows = index.lookup(&[
        record(1, 1, ReplicaMedium::Dram, true).key,
        record(2, 1, ReplicaMedium::Dram, true).key,
    ]);
    assert_eq!(rows[0].replicas[0].sequence, 3);
    assert!(rows[1].replicas.is_empty());
}

#[test]
fn deltas_are_atomic_and_reject_gaps_overlap_and_conflicts() {
    let (index, _, remote) = setup(1 << 20);
    let view = install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    let first_install = index.owner_status(remote.incarnation).unwrap();
    assert!(first_install.installed_mono_ns > 0);
    assert_eq!(
        index
            .apply_delta(
                remote.incarnation,
                view,
                1,
                3,
                vec![
                    record(1, 2, ReplicaMedium::Dram, false),
                    record(1, 3, ReplicaMedium::Dram, true),
                ],
            )
            .unwrap(),
        DeltaApply::Applied
    );
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 3)));
    let delta_install = index
        .owner_status(remote.incarnation)
        .unwrap()
        .installed_mono_ns;
    assert!(
        index
            .owner_status(remote.incarnation)
            .unwrap()
            .installed_mono_ns
            >= first_install.installed_mono_ns
    );
    assert_eq!(
        index
            .apply_delta(remote.incarnation, view, 1, 3, Vec::new())
            .unwrap(),
        DeltaApply::Duplicate
    );
    assert_eq!(
        index
            .apply_delta(
                remote.incarnation,
                view,
                2,
                4,
                vec![record(2, 4, ReplicaMedium::Dram, true)],
            )
            .unwrap(),
        DeltaApply::Overlap {
            applied_sequence: 3
        }
    );
    assert!(
        index
            .apply_delta(
                remote.incarnation,
                view,
                4,
                5,
                vec![record(2, 5, ReplicaMedium::Dram, true)],
            )
            .is_err()
    );
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 3)));
    index.confirm_progress(remote.incarnation, view, 3).unwrap();
    assert_eq!(
        index
            .owner_status(remote.incarnation)
            .unwrap()
            .installed_mono_ns,
        delta_install
    );
    index.mark_stale(remote.incarnation);
    index.confirm_progress(remote.incarnation, view, 3).unwrap();
    assert_eq!(
        index
            .owner_status(remote.incarnation)
            .unwrap()
            .installed_mono_ns,
        delta_install
    );
    index
        .apply_delta(remote.incarnation, view, 3, 4, Vec::new())
        .unwrap();
    assert!(
        index
            .owner_status(remote.incarnation)
            .unwrap()
            .installed_mono_ns
            >= delta_install
    );
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 4)));
}

#[test]
fn active_plus_staging_budget_failure_preserves_old_positive_hints() {
    let (index, _, remote) = setup(1400);
    install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, Uuid::new_v4(), 1)
        .unwrap();
    assert!(
        index
            .apply_snapshot_page(
                remote.incarnation,
                session,
                snapshot,
                0,
                vec![
                    record(2, 1, ReplicaMedium::Dram, true),
                    record(3, 1, ReplicaMedium::Dram, true),
                    record(4, 1, ReplicaMedium::Dram, true),
                ],
            )
            .is_err()
    );
    let old = record(1, 1, ReplicaMedium::Dram, true).key;
    assert_eq!(index.lookup(&[old])[0].replicas.len(), 1);
    assert_eq!(index.status().coverage, DiscoveryCoverage::PartialHints);
    index.abort_snapshot(remote.incarnation, session);
}

#[test]
fn removed_owner_is_excluded_before_bounded_reverse_cleanup() {
    let (index, membership, remote) = setup(1 << 20);
    let key = record(1, 1, ReplicaMedium::Dram, true).key;
    install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    membership.replace_members([("local".into(), 1, membership.owner().clone())]);
    index.set_expected_owners(11, [membership.owner().incarnation]);
    index.retire_owner(remote.incarnation);
    assert!(index.lookup(&[key])[0].replicas.is_empty());
    assert!(index.cleanup_owner(remote.incarnation, 1));
    assert_eq!(index.bytes(), 0);
}

#[test]
fn snapshot_commit_rejects_missing_pages_and_keeps_staging_hidden() {
    let (index, _, remote) = setup(1 << 20);
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, Uuid::new_v4(), 1)
        .unwrap();
    index
        .apply_snapshot_page(
            remote.incarnation,
            session,
            snapshot,
            0,
            vec![record(1, 1, ReplicaMedium::Dram, true)],
        )
        .unwrap();
    assert!(
        index
            .commit_snapshot(remote.incarnation, session, snapshot, Uuid::new_v4(), 1, 2,)
            .is_err()
    );
    assert!(
        index.lookup(&[record(1, 1, ReplicaMedium::Dram, true).key])[0]
            .replicas
            .is_empty()
    );
    assert!(index.status().staging_bytes > 0);
    index.abort_snapshot(remote.incarnation, session);
    assert_eq!(index.status().staging_bytes, 0);
}

#[test]
fn snapshot_commit_validates_view_before_replacing_active_rows() {
    let (index, _, remote) = setup(1 << 20);
    let old_key = record(1, 1, ReplicaMedium::Dram, true).key;
    install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    let view = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, view, 1)
        .unwrap();
    index
        .apply_snapshot_page(
            remote.incarnation,
            session,
            snapshot,
            0,
            vec![record(2, 2, ReplicaMedium::Dram, true)],
        )
        .unwrap();
    assert!(
        index
            .commit_snapshot(remote.incarnation, session, snapshot, Uuid::new_v4(), 2, 1,)
            .is_err()
    );
    assert_eq!(index.lookup(&[old_key])[0].replicas.len(), 1);
    assert_eq!(index.owner_watermark(remote.incarnation).unwrap().1, 1);
}

#[test]
fn snapshot_pages_are_rejected_after_replay_starts() {
    let (index, _, remote) = setup(1 << 20);
    let session = Uuid::new_v4();
    let snapshot = Uuid::new_v4();
    let view = Uuid::new_v4();
    index
        .begin_snapshot(remote.incarnation, session, snapshot, view, 1)
        .unwrap();
    index
        .apply_snapshot_page(
            remote.incarnation,
            session,
            snapshot,
            0,
            vec![record(1, 1, ReplicaMedium::Dram, true)],
        )
        .unwrap();
    index
        .apply_snapshot_delta(
            remote.incarnation,
            session,
            1,
            2,
            vec![record(1, 2, ReplicaMedium::Dram, false)],
        )
        .unwrap();
    assert!(
        index
            .apply_snapshot_page(
                remote.incarnation,
                session,
                snapshot,
                1,
                vec![record(1, 1, ReplicaMedium::Dram, true)],
            )
            .is_err()
    );
    index
        .commit_snapshot(remote.incarnation, session, snapshot, view, 2, 1)
        .unwrap();
    assert!(
        index.lookup(&[record(1, 1, ReplicaMedium::Dram, true).key])[0]
            .replicas
            .is_empty()
    );
}

#[test]
fn membership_refresh_does_not_revive_budget_withdrawal() {
    let (index, membership, remote) = setup(1 << 20);
    let mut old = (0..700)
        .map(|key| InventoryRecord {
            key: StateKey::new("model".into(), (key as u64).to_le_bytes().to_vec()),
            sequence: key as u64 + 1,
            present: true,
            metadata: Some(ReplicaMetadata {
                medium: ReplicaMedium::Dram,
                representation: ReplicaRepresentation::Raw,
                stored_bytes: Some(1024),
            }),
        })
        .collect::<Vec<_>>();
    old.sort_by_key(|record| (record.key.clone(), record.metadata.unwrap().medium));
    install(&index, &remote, old, 700);
    index.retire_owner(remote.incarnation);
    let before = index.bytes();
    assert!(!index.cleanup_owner(remote.incarnation, 128));
    assert!(index.bytes() < before);
    index.set_expected_owners(11, [membership.owner().incarnation, remote.incarnation]);
    assert_eq!(index.owner_watermark(remote.incarnation), None);
    assert_eq!(index.status().coverage, DiscoveryCoverage::Unavailable);
    while !index.cleanup_owner(remote.incarnation, 128) {}
    assert_eq!(index.bytes(), 0);
    let replacement = record(9, 701, ReplicaMedium::Ssd, true);
    let key = replacement.key.clone();
    let view = install(&index, &remote, vec![replacement], 701);
    assert_eq!(index.owner_watermark(remote.incarnation), Some((view, 701)));
    assert_eq!(index.lookup(&[key])[0].replicas.len(), 1);
    assert_eq!(
        index.status().coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
}

#[test]
fn interrupted_owner_view_is_a_partial_positive_hint() {
    let (index, _, remote) = setup(1 << 20);
    let key = record(1, 1, ReplicaMedium::Dram, true).key;
    install(
        &index,
        &remote,
        vec![record(1, 1, ReplicaMedium::Dram, true)],
        1,
    );
    index.mark_stale(remote.incarnation);
    let row = &index.lookup(&[key])[0];
    assert_eq!(row.coverage, DiscoveryCoverage::PartialHints);
    assert_eq!(row.replicas.len(), 1);
    let (view, sequence) = index.owner_watermark(remote.incarnation).unwrap();
    index
        .confirm_progress(remote.incarnation, view, sequence)
        .unwrap();
    assert_eq!(
        index.lookup(&[record(1, 1, ReplicaMedium::Dram, true).key])[0].coverage,
        DiscoveryCoverage::CompleteAtWatermarks
    );
}
