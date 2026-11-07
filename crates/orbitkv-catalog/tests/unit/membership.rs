use super::*;

fn owner(endpoint: &str) -> CacheOwner {
    CacheOwner {
        endpoint: endpoint.into(),
        incarnation: uuid::Uuid::new_v4(),
    }
}

#[test]
fn registration_and_complete_snapshot_are_both_required() {
    let local = owner("127.0.0.1:50055");
    let peer = owner("127.0.0.1:50056");
    let view = MembershipView::new(local.clone());
    view.replace_members([
        ("local".into(), 1, local.clone()),
        ("peer".into(), 1, peer.clone()),
    ]);
    assert!(!view.permits(&peer));
    assert!(view.renew(Instant::now(), Duration::from_secs(30)));
    assert!(view.permits(&peer));
    assert!(!view.permits(&owner(&peer.endpoint)));
    view.invalidate_snapshot();
    assert!(!view.permits(&local));
    view.replace_members([("local".into(), 1, local.clone())]);
    assert!(view.permits(&local));
    assert!(!view.permits(&peer));
}

#[test]
fn expired_or_fenced_runtime_cannot_be_revived_by_delayed_acknowledgements() {
    for cause in ["deadline", "explicit", "replacement"] {
        let local = owner("127.0.0.1:50055");
        let view = MembershipView::new(local.clone());
        view.replace_members([("local".into(), 1, local.clone())]);
        assert!(view.renew(Instant::now(), Duration::from_secs(30)));
        match cause {
            "deadline" => {
                view.state.write().valid_until = Some(Instant::now() - Duration::from_secs(1));
            }
            "explicit" => view.fence(),
            _ => view.replace_members([("local".into(), 1, owner(&local.endpoint))]),
        }
        assert!(!view.permits(&local));
        assert!(!view.renew(Instant::now(), Duration::from_secs(30)));
        view.replace_members([("local".into(), 1, local.clone())]);
        assert!(!view.permits(&local));
    }
    let view = MembershipView::new(owner("127.0.0.1:50055"));
    assert!(!view.renew(
        Instant::now() - Duration::from_secs(20),
        Duration::from_secs(30)
    ));
}

#[test]
fn authority_is_revoked_on_removal_incarnation_or_epoch_change_only() {
    let local = owner("127.0.0.1:50055");
    let peer = owner("127.0.0.1:50056");
    let view = MembershipView::new(local.clone());
    let install = |epoch, peer: CacheOwner| {
        view.replace_members([
            ("local".into(), 1, local.clone()),
            ("peer".into(), epoch, peer),
        ]);
    };
    assert!(view.renew(Instant::now(), Duration::from_secs(30)));
    install(1, peer.clone());
    let old = view.lease(&peer).unwrap();
    view.invalidate_snapshot();
    assert!(!view.permits_lease(&old));
    assert!(!old.revoked());
    install(1, peer.clone());
    assert!(
        view.permits_lease(&old),
        "unchanged snapshot preserves authority"
    );

    for change in ["epoch", "incarnation", "removal", "coalesced watch"] {
        install(1, peer.clone());
        let old = view.lease(&peer).unwrap();
        let local_lease = view.lease(&local).unwrap();
        match change {
            "epoch" => install(2, peer.clone()),
            "incarnation" => install(1, owner(&peer.endpoint)),
            "coalesced watch" => {
                view.revoke(&peer);
                assert!(!view.permits(&peer));
                assert!(view.resolve(peer.incarnation).is_none());
                install(1, peer.clone());
            }
            _ => view.replace_members([("local".into(), 1, local.clone())]),
        }
        assert!(old.revoked(), "{change}");
        assert!(!view.permits_lease(&old));
        assert!(
            view.permits_lease(&local_lease),
            "unrelated member unchanged"
        );
        install(1, peer.clone());
        assert!(view.permits(&peer));
        assert!(
            !view.permits_lease(&old),
            "rejoining cannot revive old grants"
        );
    }
    let current = view.lease(&peer).unwrap();
    view.fence();
    assert!(current.revoked());
    install(1, peer.clone());
    assert!(!view.permits(&peer));
}
