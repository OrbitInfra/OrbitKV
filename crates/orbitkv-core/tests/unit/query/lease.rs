use super::*;
use crate::SealedBlock;

#[test]
fn invalid_batch_does_not_consume_valid_lease_shares() {
    for invalid in ["missing", "instance", "expired", "duplicate"] {
        let manager = QueryLeaseManager::default();
        let source = RestoreSource::Memory(Arc::new(SealedBlock::from_slots(Vec::new())));
        let first = manager.create("a", vec![source.clone()], 2, None).unwrap();
        let second = manager
            .create(
                if invalid == "instance" { "b" } else { "a" },
                vec![source],
                1,
                None,
            )
            .unwrap();
        if invalid == "expired" {
            manager
                .inner
                .leases
                .lock()
                .unwrap()
                .get_mut(&second)
                .unwrap()
                .expires_at = Instant::now();
        }
        let tokens = [
            first,
            match invalid {
                "missing" => QueryLeaseId::fresh(),
                "duplicate" => first,
                _ => second,
            },
        ];
        let error = manager
            .consume_batch::<()>("a", &tokens, |_| {
                panic!("invalid tokens must be rejected before source validation")
            })
            .err()
            .expect("invalid batch must fail");
        assert!(matches!(
            (&error, invalid),
            (EngineError::InvalidArgument(_), "duplicate") | (EngineError::Storage(_), _)
        ));
        assert!(
            error.to_string().contains(match invalid {
                "instance" => "belongs to instance b",
                "duplicate" => "duplicate query lease",
                _ => "unknown or expired",
            }),
            "{invalid}: {error}"
        );
        let leases = manager.inner.leases.lock().unwrap();
        assert_eq!(leases[&first].remaining_consumers, 2, "{invalid}");
        assert_eq!(leases[&second].remaining_consumers, 1, "{invalid}");
    }
}

#[test]
fn source_validation_failure_preserves_the_entire_batch() {
    let manager = QueryLeaseManager::default();
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let source = RestoreSource::Memory(Arc::clone(&block));
    let first = manager.create("a", vec![source.clone()], 1, None).unwrap();
    let second = manager
        .create("a", vec![source.clone(), source], 2, None)
        .unwrap();
    let owners = Arc::strong_count(&block);
    let error = manager
        .consume_batch::<()>("a", &[first, second], |sources| {
            assert_eq!(
                sources
                    .iter()
                    .map(|blocks| blocks.len())
                    .collect::<Vec<_>>(),
                [1, 2]
            );
            Err(EngineError::InvalidArgument(
                "storage group slot mismatch".to_string(),
            ))
        })
        .err()
        .expect("source validation must reject the batch");
    assert!(
        matches!(error, EngineError::InvalidArgument(message) if message == "storage group slot mismatch")
    );
    assert_eq!(Arc::strong_count(&block), owners);
    let leases = manager.inner.leases.lock().unwrap();
    assert_eq!(leases[&first].remaining_consumers, 1);
    assert_eq!(leases[&second].remaining_consumers, 2);
    drop(leases);

    let (total_blocks, consumed, _) = manager
        .consume_batch("a", &[first, second], |sources| {
            Ok(sources.iter().map(|blocks| blocks.len()).sum::<usize>())
        })
        .unwrap();
    assert_eq!(total_blocks, 3);
    assert_eq!(consumed.len(), 3);
}

#[test]
fn batch_preserves_order_and_configured_consumer_count() {
    let manager = QueryLeaseManager::default();
    let first_block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let second_block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let first = manager
        .create(
            "a",
            vec![RestoreSource::Memory(Arc::clone(&first_block))],
            2,
            None,
        )
        .unwrap();
    let second = manager
        .create(
            "a",
            vec![RestoreSource::Memory(Arc::clone(&second_block))],
            1,
            None,
        )
        .unwrap();
    let (_, consumed, _) = manager
        .consume_batch("a", &[second, first], |_| Ok(()))
        .unwrap();
    for (source, expected) in consumed.iter().zip([&second_block, &first_block]) {
        let RestoreSource::Memory(actual) = source else {
            panic!("memory source expected");
        };
        assert!(Arc::ptr_eq(actual, expected));
    }
    assert!(manager.consume_batch("a", &[second], |_| Ok(())).is_err());
    assert_eq!(
        manager
            .consume_batch("a", &[first], |_| Ok(()))
            .unwrap()
            .1
            .len(),
        1
    );
    assert!(manager.consume_batch("a", &[first], |_| Ok(())).is_err());
    let (validated, empty, reservations) = manager
        .consume_batch("a", &[], |sources| Ok(sources.is_empty()))
        .unwrap();
    assert!(validated);
    assert!(empty.is_empty());
    assert!(reservations.is_empty());
}

#[test]
fn disconnect_releases_ready_interest_but_not_a_gpu_consumers_reservation() {
    let manager = QueryLeaseManager::default();
    let budget = crate::query::QueryBudget::new(100, 100).unwrap();
    let crate::QueryAdmission::Admitted(reservation) =
        budget.reserve("a", "ns", 100, crate::QueryMode::Demand)
    else {
        panic!("budget available");
    };
    reservation.ready(100).unwrap();
    let owner = QueryOwner {
        session: 7,
        operation: 1,
        revision: 2,
    };
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let source = Arc::downgrade(&block);
    let id = manager
        .create(
            "a",
            vec![RestoreSource::Memory(block)],
            2,
            Some((owner, reservation)),
        )
        .unwrap();
    let (_, blocks, gpu) = manager.consume_batch("a", &[id], |_| Ok(())).unwrap();
    gpu[0].restoring();
    manager
        .release_owner(|candidate| candidate.session == 7)
        .unwrap();
    assert!(source.upgrade().is_some());
    assert!(matches!(
        budget.reserve("a", "ns", 1, crate::QueryMode::Demand),
        crate::QueryAdmission::Busy
    ));
    assert!(manager.consume_batch("a", &[id], |_| Ok(())).is_err());
    drop(blocks);
    assert!(source.upgrade().is_none());
    drop(gpu);
    assert!(matches!(
        budget.reserve("a", "ns", 100, crate::QueryMode::Demand),
        crate::QueryAdmission::Admitted(_)
    ));
}

#[test]
fn registration_fence_rejects_old_leases_before_consuming_any_share() {
    let manager = QueryLeaseManager::default();
    let budget = crate::query::QueryBudget::new(4096, 4096).unwrap();
    let crate::query::QueryAdmission::Admitted(reservation) = budget.reserve(
        "instance",
        "same-layout",
        32,
        crate::query::QueryMode::Demand,
    ) else {
        panic!("reservation must fit")
    };
    let reservation = reservation.bind_registration([1; 16]);
    let token = manager
        .create(
            "instance",
            vec![RestoreSource::Memory(Arc::new(SealedBlock::from_slots(
                vec![],
            )))],
            2,
            Some((
                crate::query::QueryOwner {
                    session: 2,
                    operation: 1,
                    revision: 1,
                },
                reservation,
            )),
        )
        .unwrap();
    manager
        .validate_registration(std::iter::once(&token), [1; 16])
        .unwrap();
    assert!(
        manager
            .validate_registration(std::iter::once(&token), [2; 16])
            .is_err()
    );
    assert_eq!(
        manager.inner.leases.lock().unwrap()[&token].remaining_consumers,
        2
    );
    manager
        .validate_registration(std::iter::once(&token), [1; 16])
        .unwrap();
}

#[test]
fn poisoned_lease_lock_returns_errors_and_retains_owners() {
    let manager = QueryLeaseManager::default();
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    let weak = Arc::downgrade(&block);
    let token = manager
        .create("a", vec![RestoreSource::Memory(block)], 1, None)
        .unwrap();
    let inner = Arc::clone(&manager.inner);
    assert!(
        std::thread::spawn(move || {
            let _guard = inner.leases.lock().unwrap();
            panic!("inject poisoned lease lock");
        })
        .join()
        .is_err()
    );
    assert!(matches!(
        manager.claim(&token),
        Err(EngineError::Poisoned("query leases"))
    ));
    assert!(matches!(
        manager.release(&token),
        Err(EngineError::Poisoned("query leases"))
    ));
    assert!(matches!(
        manager.release_instance("a"),
        Err(EngineError::Poisoned("query leases"))
    ));
    assert!(matches!(
        manager.sweep_expired(),
        Err(EngineError::Poisoned("query leases"))
    ));
    assert!(matches!(
        manager.consume_batch("a", &[token], |_| Ok(())),
        Err(EngineError::Poisoned("query leases"))
    ));
    assert!(
        weak.upgrade().is_some(),
        "failed cleanup must retain the existing lease owner"
    );
}
