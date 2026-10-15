use std::sync::Arc;
use std::thread;

use rustix::event::{EventfdFlags, eventfd};

use super::*;

fn records() -> Arc<RestoreCompletions> {
    Arc::new(
        RestoreCompletions::create(17, 29, eventfd(0, EventfdFlags::NONBLOCK).unwrap()).unwrap(),
    )
}

fn independent(records: &RestoreCompletions) -> RestoreCompletions {
    RestoreCompletions::open(
        records.file().try_clone().unwrap().into(),
        records.notification_fd().try_clone().unwrap(),
        records.manager_notification_fd().try_clone().unwrap(),
        17,
        29,
    )
    .unwrap()
}

#[test]
fn only_reaped_and_acknowledged_records_recycle() {
    let records = records();
    let engine = independent(&records);
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| engine.reserve().unwrap())
        .collect();
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    let id = ids[0];
    records.claim(id).unwrap();
    assert!(
        records
            .publish_local(id, b"checked source plan", false)
            .unwrap()
    );
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
    assert_eq!(
        engine.claim_local(id).unwrap().unwrap(),
        b"checked source plan"
    );
    assert!(!records.revoke(id).unwrap());
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    engine.drained(id, Ok(()), None).unwrap();
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
    assert_eq!(
        records.manager_updates().unwrap(),
        vec![(id, GrantState::Drained)]
    );
    assert!(records.drain_succeeded(id).unwrap());
    records.release_plan(id).unwrap();
    records.reap(id).unwrap();
    assert!(matches!(records.reserve(), Err(CompletionError::Full)));
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Succeeded);
    let replacement = engine.reserve().unwrap();
    assert_eq!(
        replacement as usize % RESTORE_COMPLETION_SLOTS,
        id as usize % RESTORE_COMPLETION_SLOTS
    );
    assert!(matches!(records.claim(id), Err(CompletionError::Stale(_))));
    assert!(matches!(
        engine.drained(id, Ok(()), None),
        Err(CompletionError::Stale(_))
    ));
    assert_eq!(records.state(replacement).unwrap(), GrantState::Reserved);
}

#[test]
fn plan_bank_is_bounded_and_consumption_does_not_release_sources() {
    let records = records();
    let engine = independent(&records);
    let first = engine.reserve().unwrap();
    records.claim(first).unwrap();
    let large = vec![0x5a; RESTORE_PLAN_BYTES];
    assert!(records.publish_local(first, &large, false).unwrap());
    let second = engine.reserve().unwrap();
    records.claim(second).unwrap();
    assert!(matches!(
        records.publish_local(second, b"next", false),
        Err(CompletionError::PlanFull)
    ));
    assert!(records.release_plan(first).is_err());
    assert_eq!(engine.claim_local(first).unwrap().unwrap(), large);
    assert_eq!(
        records.manager_updates().unwrap(),
        vec![(first, GrantState::Active)]
    );
    records.release_plan(first).unwrap();
    assert!(records.publish_local(second, b"next", false).unwrap());
    assert_eq!(records.state(first).unwrap(), GrantState::Active);
    assert!(records.reap(first).is_err());
    assert_eq!(engine.claim_local(second).unwrap().unwrap(), b"next");
    engine
        .drained(second, Err("partial enqueue drained".into()), None)
        .unwrap();
    engine.drained(first, Ok(()), None).unwrap();
    for id in [first, second] {
        records.reap(id).unwrap();
    }
    assert_eq!(
        engine.poll(second).unwrap().message,
        "partial enqueue drained"
    );
    assert_eq!(engine.poll(first).unwrap().state, RestoreState::Succeeded);
}

#[test]
fn cancellation_has_a_separate_preparation_drain() {
    let records = records();
    let before = records.reserve().unwrap();
    assert!(records.cancel(before).unwrap());
    assert!(records.claim(before).is_err());
    let during = records.reserve().unwrap();
    records.claim(during).unwrap();
    assert!(!records.cancel(during).unwrap());
    assert_eq!(records.poll(during).unwrap().state, RestoreState::Pending);
    assert!(
        !records
            .publish_local(during, b"not published", false)
            .unwrap()
    );
    records.finish_cancelled(during).unwrap();
    assert_eq!(records.poll(during).unwrap().state, RestoreState::Failed);
    // A managed worker can already have submitted before route publication.
    let managed = records.reserve().unwrap();
    records.claim(managed).unwrap();
    assert!(!records.cancel(managed).unwrap());
    records.start_managed(managed).unwrap();
    assert_eq!(records.poll(managed).unwrap().state, RestoreState::Pending);
    records.complete(managed, Ok(())).unwrap();
    assert_eq!(
        records.poll(managed).unwrap().state,
        RestoreState::Succeeded
    );
}

#[test]
fn claim_and_revoke_have_exactly_one_winner_across_mappings() {
    let records = records();
    let engine = independent(&records);
    for _ in 0..128 {
        let id = engine.reserve().unwrap();
        records.claim(id).unwrap();
        records.publish_local(id, b"plan", false).unwrap();
        let gate = std::sync::Barrier::new(2);
        thread::scope(|scope| {
            let claim = scope.spawn(|| {
                gate.wait();
                engine.claim_local(id)
            });
            gate.wait();
            let revoked = records.revoke(id).unwrap();
            match claim.join().unwrap() {
                Ok(Some(plan)) => {
                    assert!(!revoked);
                    assert_eq!(plan, b"plan");
                    engine.drained(id, Ok(()), None).unwrap();
                }
                Ok(None) | Err(CompletionError::Stale(_)) => assert!(revoked),
                other => panic!("unexpected claim: {other:?}"),
            }
            assert_eq!(records.drain_succeeded(id).unwrap(), !revoked);
            records.reap(id).unwrap();
            assert_eq!(
                engine.poll(id).unwrap().state,
                if revoked {
                    RestoreState::Failed
                } else {
                    RestoreState::Succeeded
                }
            );
        });
    }
}

#[test]
fn bounded_dirty_bits_preserve_all_operations_without_queue_overflow() {
    let records = records();
    let engine = independent(&records);
    let ids: Vec<_> = (0..RESTORE_COMPLETION_SLOTS)
        .map(|_| engine.reserve().unwrap())
        .collect();
    for id in &ids {
        records.claim(*id).unwrap();
        records.publish_local(*id, b"plan", false).unwrap();
        engine.claim_local(*id).unwrap().unwrap();
        engine.drained(*id, Ok(()), None).unwrap();
    }
    let updates = records.manager_updates().unwrap();
    assert_eq!(updates.len(), RESTORE_COMPLETION_SLOTS);
    assert!(
        updates
            .iter()
            .all(|(_, state)| *state == GrantState::Drained)
    );
    assert!(records.manager_updates().unwrap().is_empty());
    for (id, _) in updates {
        records.reap(id).unwrap();
        engine.poll(id).unwrap();
    }
    assert!(records.plans.lock().unwrap().is_empty());
}

#[test]
fn stale_readers_cannot_mix_recycled_error_generations() {
    let records = records();
    let current = AtomicU64::new(0);
    let finished = std::sync::atomic::AtomicBool::new(false);
    let consumed = AtomicU64::new(0);
    thread::scope(|scope| {
        for _ in 0..2 {
            let records = &records;
            let current = &current;
            let finished = &finished;
            let consumed = &consumed;
            scope.spawn(move || {
                while !finished.load(Ordering::Acquire) {
                    let id = current.load(Ordering::Acquire);
                    if id == 0 {
                        thread::yield_now();
                        continue;
                    }
                    match records.poll(id) {
                        Ok(response) if response.state == RestoreState::Pending => {
                            thread::yield_now();
                        }
                        Ok(response) => {
                            let generation = (id - 1) / RESTORE_COMPLETION_SLOTS as u64;
                            let expected = if generation.is_multiple_of(2) {
                                "错".repeat(RESTORE_ERROR_BYTES / 3)
                            } else {
                                "é".repeat(RESTORE_ERROR_BYTES / 2)
                            };
                            assert_eq!(response.message, expected);
                            consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(CompletionError::Stale(_)) => thread::yield_now(),
                        Err(error) => panic!("{error}"),
                    }
                }
            });
        }
        for generation in 0..128u64 {
            records.word(NEXT_OPERATION_OFFSET).store(
                1 + generation * RESTORE_COMPLETION_SLOTS as u64,
                Ordering::Relaxed,
            );
            let id = records.reserve().unwrap();
            records.claim(id).unwrap();
            records.start_managed(id).unwrap();
            current.store(id, Ordering::Release);
            let message = if generation.is_multiple_of(2) {
                "错".repeat(2000)
            } else {
                "é".repeat(2000)
            };
            records.complete(id, Err(message)).unwrap();
            while records.state(id).unwrap() != GrantState::Acknowledged {
                thread::yield_now();
            }
        }
        finished.store(true, Ordering::Release);
    });
    assert_eq!(consumed.load(Ordering::Relaxed), 128);
}

#[test]
fn mapping_identity_and_operation_exhaustion_are_checked() {
    let records = records();
    assert!(matches!(
        RestoreCompletions::open(
            records.file().try_clone().unwrap().into(),
            records.notification_fd().try_clone().unwrap(),
            records.manager_notification_fd().try_clone().unwrap(),
            18,
            29
        ),
        Err(CompletionError::InvalidMapping)
    ));
    records
        .word(NEXT_OPERATION_OFFSET)
        .store(MAX_OPERATION_ID, Ordering::Relaxed);
    let last = records.reserve().unwrap();
    assert_eq!(last, MAX_OPERATION_ID);
    records.claim(last).unwrap();
    records.start_managed(last).unwrap();
    records.complete(last, Ok(())).unwrap();
    assert_eq!(records.poll(last).unwrap().state, RestoreState::Succeeded);
    assert!(matches!(records.reserve(), Err(CompletionError::Exhausted)));
    for invalid in [0, MAX_OPERATION_ID + 1, u64::MAX] {
        assert!(records.poll(invalid).is_err());
        assert!(records.claim(invalid).is_err());
        assert!(records.cancel(invalid).is_err());
    }
}

#[test]
fn admission_and_preparation_cancellation_race_without_losing_the_operation() {
    let records = records();
    let engine = independent(&records);
    let first = engine.reserve().unwrap();
    let second = records.reserve().unwrap();
    assert_ne!(first, second);
    assert!(engine.cancel(first).unwrap());
    assert!(records.cancel(second).unwrap());
    let gate = std::sync::Barrier::new(2);
    let operation = AtomicU64::new(0);
    let safe_cancel = std::sync::atomic::AtomicBool::new(false);
    thread::scope(|scope| {
        let cancel = scope.spawn(|| {
            for _ in 0..128 {
                gate.wait();
                safe_cancel.store(
                    engine.cancel(operation.load(Ordering::Acquire)).unwrap(),
                    Ordering::Release,
                );
                gate.wait();
            }
        });
        for _ in 0..128 {
            let id = engine.reserve().unwrap();
            operation.store(id, Ordering::Release);
            gate.wait();
            let admitted = records.claim(id).is_ok();
            gate.wait();
            assert_ne!(admitted, safe_cancel.load(Ordering::Acquire));
            assert!(records.claim(id).is_err(), "admission must be exactly once");
            if admitted {
                assert_eq!(records.state(id).unwrap(), GrantState::CancelRequested);
                assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
                records.finish_cancelled(id).unwrap();
                assert_eq!(engine.poll(id).unwrap().state, RestoreState::Failed);
            } else {
                assert_eq!(records.state(id).unwrap(), GrantState::Acknowledged);
            }
        }
        cancel.join().unwrap();
    });
}

#[test]
fn timing_follows_drain_generation_and_never_blocks_reclamation() {
    let records = records();
    let engine = independent(&records);
    let timing = RestoreTiming {
        readiness_ns: 10,
        dispatched_ns: 20,
        dequeued_ns: 30,
        claimed_ns: 40,
        submitted_ns: 50,
        drained_ns: 60,
    };
    let mut previous = None;
    for index in 0..=RESTORE_COMPLETION_SLOTS {
        let id = engine.reserve().unwrap();
        records.claim(id).unwrap();
        records.publish_local(id, b"plan", false).unwrap();
        engine.claim_local(id).unwrap().unwrap();
        assert!(records.drain_timing(id).is_err());
        let report = (index % 2 == 0).then_some(timing);
        engine.drained(id, Ok(()), report).unwrap();
        assert_eq!(records.drain_timing(id).unwrap(), report);
        assert!(engine.drained(id, Ok(()), Some(timing)).is_err());
        assert_eq!(records.drain_timing(id).unwrap(), report);
        // Dirty bits remain authoritative even if eventfd notification was consumed elsewhere.
        assert_eq!(
            records.manager_updates().unwrap(),
            vec![(id, GrantState::Drained)]
        );
        assert!(records.manager_updates().unwrap().is_empty());
        if let Some(stale) = previous {
            assert!(records.drain_timing(stale).is_err());
        }
        records.reap(id).unwrap();
        assert!(records.drain_timing(id).is_err());
        assert_eq!(engine.poll(id).unwrap().state, RestoreState::Succeeded);
        previous = Some(id);
    }
    for invalid in [
        RestoreTiming {
            readiness_ns: 100,
            ..timing
        },
        RestoreTiming {
            submitted_ns: 0,
            ..timing
        },
        RestoreTiming {
            drained_ns: u64::MAX,
            ..timing
        },
    ] {
        let id = engine.reserve().unwrap();
        records.claim(id).unwrap();
        records.publish_local(id, b"plan", false).unwrap();
        engine.claim_local(id).unwrap().unwrap();
        engine.drained(id, Ok(()), Some(invalid)).unwrap();
        assert_eq!(records.drain_timing(id).unwrap(), None);
        records.reap(id).unwrap();
        assert_eq!(engine.poll(id).unwrap().state, RestoreState::Succeeded);
    }
    let id = engine.reserve().unwrap();
    records.claim(id).unwrap();
    records.publish_local(id, b"plan", false).unwrap();
    engine.claim_local(id).unwrap().unwrap();
    engine
        .drained(id, Err("partial enqueue drained".into()), Some(timing))
        .unwrap();
    assert_eq!(records.drain_timing(id).unwrap(), Some(timing));
    assert!(!records.drain_succeeded(id).unwrap());
    records.reap(id).unwrap();
    assert_eq!(engine.poll(id).unwrap().state, RestoreState::Failed);
}

#[test]
fn multiple_parts_keep_one_pending_operation_until_final_drain() {
    for outcome in ["success", "failure", "revoke", "cancel"] {
        let records = records();
        let engine = independent(&records);
        let id = engine.reserve().unwrap();
        records.claim(id).unwrap();
        let first = vec![0x41; RESTORE_PLAN_BYTES];
        assert!(records.publish_local(id, &first, true).unwrap());
        assert_eq!(engine.claim_local(id).unwrap().unwrap(), first);
        assert_eq!(records.state(id).unwrap(), GrantState::ActiveMore);
        assert!(!records.revoke(id).unwrap());
        assert!(!engine.drained(id, Ok(()), None).unwrap());
        assert_eq!(records.state(id).unwrap(), GrantState::PartDrained);
        assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
        assert!(records.reap(id).is_err());
        records.release_plan(id).unwrap();
        if outcome == "revoke" {
            assert!(records.revoke(id).unwrap());
            assert!(records.continue_local(id).is_err());
        } else {
            records.continue_local(id).unwrap();
            // A duplicate dirty-bit observation must not consume another part.
            assert!(records.continue_local(id).is_err());
            if outcome == "cancel" {
                assert!(!engine.cancel(id).unwrap());
                assert!(!records.publish_local(id, b"unused", false).unwrap());
                records.finish_cancelled(id).unwrap();
                assert_eq!(engine.poll(id).unwrap().state, RestoreState::Failed);
                continue;
            }
            assert!(
                records
                    .publish_local(id, b"second", outcome == "failure")
                    .unwrap()
            );
            assert_eq!(engine.claim_local(id).unwrap().unwrap(), b"second");
            let result = if outcome == "failure" {
                Err("second part failed after drain".into())
            } else {
                Ok(())
            };
            assert!(engine.drained(id, result, None).unwrap());
            assert!(records.continue_local(id).is_err());
        }
        assert_eq!(engine.poll(id).unwrap().state, RestoreState::Pending);
        records.reap(id).unwrap();
        assert_eq!(
            engine.poll(id).unwrap().state,
            if outcome == "success" {
                RestoreState::Succeeded
            } else {
                RestoreState::Failed
            }
        );
        assert!(records.plans.lock().unwrap().is_empty());
    }
}

#[test]
fn losing_claim_cannot_drain_the_actual_winner() {
    use std::sync::mpsc;
    for more in [false, true] {
        let records = records();
        let loser = independent(&records);
        let winner = independent(&records);
        let id = records.reserve().unwrap();
        records.claim(id).unwrap();
        records
            .publish_local(id, b"winner retains this source", more)
            .unwrap();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        thread::scope(|scope| {
            let losing = scope.spawn(move || {
                crate::test_faults::pause_next_claim(entered_tx, resume_rx);
                loser.claim_local(id)
            });
            entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(
                winner.claim_local(id).unwrap().unwrap(),
                b"winner retains this source"
            );
            resume_tx.send(()).unwrap();
            assert!(
                matches!(losing.join().unwrap(), Err(CompletionError::Stale(value)) if value == id)
            );
        });
        let active = if more {
            GrantState::ActiveMore
        } else {
            GrantState::Active
        };
        assert_eq!(records.state(id).unwrap(), active);
        assert!(records.reap(id).is_err());
        assert_eq!(records.poll(id).unwrap().state, RestoreState::Pending);
        winner
            .drained(
                id,
                Err("CPU winner returned without submitting DMA".into()),
                None,
            )
            .unwrap();
        records.release_plan(id).unwrap();
        records.reap(id).unwrap();
        assert_eq!(records.poll(id).unwrap().state, RestoreState::Failed);
    }
}

#[test]
fn claimed_invalid_plan_has_no_submit_proof_before_bytes_escape() {
    for (more, offset, len) in [
        (false, 0, 0),
        (true, 0, 0),
        (false, 1, 8),
        (false, 0, RESTORE_PLAN_BYTES as u64 + 1),
    ] {
        let records = records();
        let engine = independent(&records);
        let id = records.reserve().unwrap();
        records.claim(id).unwrap();
        records
            .publish_local(id, b"not handed to an executor", more)
            .unwrap();
        let record = RestoreCompletions::offset(id).unwrap();
        records.word(record + 8).store(offset, Ordering::Relaxed);
        records.word(record + 16).store(len, Ordering::Relaxed);
        assert!(matches!(
            engine.claim_local(id),
            Err(CompletionError::InvalidPayload)
        ));
        assert_eq!(records.state(id).unwrap(), GrantState::Drained);
        assert!(!records.drain_succeeded(id).unwrap());
        records.release_plan(id).unwrap();
        records.reap(id).unwrap();
        let response = engine.poll(id).unwrap();
        assert_eq!(response.state, RestoreState::Failed);
        assert!(response.message.contains("before execution"));
    }
}
