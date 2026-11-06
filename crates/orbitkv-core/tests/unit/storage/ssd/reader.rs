use super::*;

fn staging_reservation(
    count: usize,
) -> (
    crate::peer::export::TransferLockManager,
    crate::TransferTicket,
    crate::peer::export::StagingReservation,
) {
    let locks =
        crate::peer::export::TransferLockManager::new(std::time::Duration::from_secs(30), 1);
    let ticket = crate::TransferTicket {
        window: locks.open(uuid::Uuid::new_v4()).unwrap(),
        slot: 0,
        generation: 1,
    };
    let reservation = locks.reserve(ticket, 1, count).unwrap();
    (locks, ticket, reservation)
}

#[test]
fn cancelled_batch_releases_results_and_finishes_after_every_reader() {
    let (done_tx, done_rx) = oneshot::channel();
    let observation = Observation::new(
        CostEstimateKey::new(
            CostObservationKind::LocalSsdHostReady,
            ExecutionResource::SsdStore(99),
            Representation::Raw,
            4096,
            3,
        ),
        None,
    );
    let context = Arc::new(BatchContext::new(3, done_tx, observation, Vec::new(), None));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    drop(done_rx);

    std::thread::scope(|scope| {
        for index in 0..3 {
            let context = Arc::clone(&context);
            let result = (index != 1).then(|| Arc::clone(&block));
            scope.spawn(move || {
                context.complete_one(StateKey::new("cancelled".into(), vec![index]), result);
            });
        }
    });

    assert_eq!(context.remaining.load(Ordering::Acquire), 0);
    assert!(context.failed.load(Ordering::Acquire));
    assert!(context.observation.lock().is_none());
    assert!(context.done_tx.lock().is_none());
    assert!(context.results.lock().is_empty());
    assert_eq!(Arc::strong_count(&block), 1);
}

#[tokio::test]
async fn cancelled_batch_retains_extent_until_the_last_reader_releases_its_context() {
    let (store, _queued) = super::super::tests::queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let lease = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin(crate::transfer_engine::tests::demand()))
        .collect::<Vec<_>>()
        .pop()
        .unwrap();
    let readers = Arc::clone(&lease.entry.readers);
    let (done_tx, done_rx) = oneshot::channel();
    let context = Arc::new(BatchContext::new(
        2,
        done_tx,
        Observation::disabled(),
        vec![lease],
        None,
    ));
    let submitted = Arc::clone(&context);
    drop(done_rx);
    context.complete_one(key.clone(), None);
    drop(context);
    assert_eq!(readers.load(Ordering::Acquire), 1);
    submitted.complete_one(key, None);
    assert_eq!(readers.load(Ordering::Acquire), 1);
    drop(submitted);
    assert_eq!(readers.load(Ordering::Acquire), 0);
}

#[test]
fn cancelled_export_retains_reservation_until_every_reader_drains() {
    let (locks, ticket, reservation) = staging_reservation(2);
    let (done_tx, done_rx) = oneshot::channel();
    let context = Arc::new(BatchContext::new(
        2,
        done_tx,
        Observation::disabled(),
        Vec::new(),
        Some(reservation),
    ));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));
    drop(done_rx);

    context.complete_one(
        StateKey::new("cancelled-export".into(), vec![0]),
        Some(Arc::clone(&block)),
    );
    assert_eq!(locks.accounting(), (1, 1));
    assert_eq!(locks.release(ticket), Ok(0));
    assert_eq!(locks.accounting(), (1, 1));
    context.complete_one(
        StateKey::new("cancelled-export".into(), vec![1]),
        Some(block),
    );
    assert_eq!(locks.accounting(), (0, 0));
}

#[test]
fn partial_export_failure_rolls_back_and_reports_staging_failure() {
    let (locks, _ticket, reservation) = staging_reservation(2);
    let (done_tx, done_rx) = oneshot::channel();
    let context = Arc::new(BatchContext::new(
        2,
        done_tx,
        Observation::disabled(),
        Vec::new(),
        Some(reservation),
    ));
    let block = Arc::new(SealedBlock::from_slots(Vec::new()));

    context.complete_one(StateKey::new("failed-export".into(), vec![0]), Some(block));
    context.complete_one(StateKey::new("failed-export".into(), vec![1]), None);
    assert!(matches!(
        done_rx.try_recv(),
        Ok(Err(PeerError::StagingFailed))
    ));
    assert_eq!(locks.accounting(), (0, 0));
}

#[test]
fn unknown_file_drain_keeps_export_reservation_after_logical_failure() {
    let (locks, ticket, reservation) = staging_reservation(1);
    let (done_tx, done_rx) = oneshot::channel();
    let context = Arc::new(BatchContext::new(
        1,
        done_tx,
        Observation::disabled(),
        Vec::new(),
        Some(reservation),
    ));
    // Model the file owner's unknown receipt. Production retains this Arc along
    // with the actual backing and extent; this control admits no native work.
    context.unquiesced.store(true, Ordering::Release);
    context.complete_one(StateKey::new("unknown-export".into(), vec![0]), None);
    assert!(matches!(
        done_rx.try_recv(),
        Ok(Err(PeerError::StagingFailed))
    ));
    assert!(context.reservation.lock().is_some());
    assert_eq!(locks.release(ticket), Ok(0));
    assert_eq!(locks.accounting(), (1, 1));
    drop(context);
    assert_eq!(locks.accounting(), (0, 0));
}
