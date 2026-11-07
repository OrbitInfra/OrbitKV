use super::*;

fn load_job() -> (Job, oneshot::Receiver<super::super::super::LoadOutcome>) {
    let (completion, receiver) = oneshot::channel();
    let job = Job::new(
        0,
        WorkerCommand::Load(
            LoadTask {
                plan: crate::planning::restore::RestorePlan::new(
                    0,
                    std::iter::empty::<(usize, &crate::RestoreSource)>(),
                )
                .unwrap(),
                layers: Vec::new(),
                completion,
                reservations: Vec::new(),
                peer_bytes: 0,
                codec_budget: 64 << 20,
                decode_ready_started: Instant::now(),
                decode_ready_observation: Box::new(Observation::disabled()),
                decode_admission: None,
            },
            Observation::new(
                crate::cost::CostEstimateKey::new(
                    crate::cost::CostObservationKind::GpuSsdLoad,
                    crate::cost::ExecutionResource::Gpu(0),
                    crate::cost::Representation::Raw,
                    0,
                    0,
                ),
                Some(0),
            ),
        ),
    );
    (job, receiver)
}

fn decoder_channels() -> (
    Decoder,
    std_mpsc::Receiver<DecodeCommand>,
    std_mpsc::SyncSender<DecodeReply>,
) {
    let (sender, requests) = std_mpsc::sync_channel(1);
    let (replies, receiver) = std_mpsc::sync_channel(1);
    (
        Decoder {
            sender: Some(sender),
            replies: receiver,
            thread: None,
        },
        requests,
        replies,
    )
}

#[test]
fn staging_windows_preserve_large_segments_and_bound_batches() {
    let large = 10 * 1024 * 1024 + 37;
    let (offsets, bytes) = input_window([large, 700].into_iter(), 64 << 20).unwrap();
    assert_eq!(offsets, vec![0, large.next_multiple_of(4096)]);
    assert_eq!(bytes, large.next_multiple_of(4096) + 4096);
    let (offsets, bytes) = input_window([large, 700].into_iter(), 16 << 20).unwrap();
    assert_eq!(
        offsets,
        vec![0],
        "a segment larger than a slot is never partially admitted"
    );
    assert_eq!(bytes, large.next_multiple_of(4096));
    assert!(input_window([large].into_iter(), large).is_err());
    assert!(input_window([usize::MAX].into_iter(), usize::MAX).is_err());
    assert!(input_window([0].into_iter(), 64 << 20).is_err());
    let (offsets, _) =
        input_window(std::iter::repeat_n(700, MAX_BATCH_SEGMENTS + 1), 64 << 20).unwrap();
    assert_eq!(offsets.len(), MAX_BATCH_SEGMENTS);
}

#[test]
fn incomplete_input_and_abandoned_decode_keep_completion_ownership() {
    let (mut job, consumer) = load_job();
    job.decoding = true;
    let mut jobs = VecDeque::from([job]);
    let mut active = Some(Decode {
        job: 0,
        reads: Vec::new(),
        offsets: Vec::new(),
        base: 1,
        remaining: 1,
        phase: DecodePhase::Reading,
    });
    let (decoder, requests, replies) = decoder_channels();
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(
        requests.try_recv().is_err(),
        "incomplete input must never reach a decoder"
    );
    assert!(!jobs[0].is_complete());
    active.as_mut().unwrap().remaining = 0;
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(matches!(requests.try_recv(), Ok(DecodeCommand::Decode(_))));
    drop(consumer);
    jobs[0].cancel_abandoned();
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(
        !jobs[0].is_complete(),
        "cancellation cannot release engine pages during decode"
    );
    replies.send(DecodeReply::Completed(Ok(()))).unwrap();
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(active.is_none());
    assert!(jobs[0].is_complete());
    assert!(jobs[0].error.is_some());
}

#[test]
fn runtime_decode_failure_does_not_access_or_invalidate_the_source_generation() {
    let (mut job, mut consumer) = load_job();
    job.decoding = true;
    let mut jobs = VecDeque::from([job]);
    let mut active = Some(Decode {
        job: 0,
        // No source is needed to report an operational failure. Attempting to
        // invalidate it, as for corruption, would panic instead of completing.
        reads: Vec::new(),
        offsets: Vec::new(),
        base: 1,
        remaining: 0,
        phase: DecodePhase::Decoding,
    });
    let (decoder, _requests, replies) = decoder_channels();
    replies
        .send(DecodeReply::Completed(Err(DecodeError::Runtime(
            "batch exceeds GPU codec budget".into(),
        ))))
        .unwrap();
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(active.is_none());
    assert!(jobs[0].is_complete());
    assert_eq!(jobs[0].bytes, 0);
    assert_eq!(
        jobs[0].error.as_deref(),
        Some("batch exceeds GPU codec budget")
    );
    jobs.pop_front().unwrap().finish();
    assert!(consumer.try_recv().unwrap().result.is_err());
}

#[test]
fn canceled_fragmented_input_waits_for_every_submitted_scatter() {
    let (mut job, _consumer) = load_job();
    job.decoding = true;
    job.inflight = 2;
    job.fail("canceled".into());
    let mut jobs = VecDeque::from([job]);
    let mut active = Some(Decode {
        job: 0,
        reads: Vec::new(),
        offsets: Vec::new(),
        base: 1,
        remaining: 3,
        phase: DecodePhase::Reading,
    });
    let (decoder, requests, _replies) = decoder_channels();
    for inflight in [2, 1] {
        assert_eq!(jobs[0].inflight, inflight);
        poll_decode(&decoder, &mut active, &mut jobs);
        assert!(active.is_some());
        assert!(!jobs[0].is_complete());
        jobs[0].complete(None, Ok(()));
    }
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(active.is_none());
    assert!(jobs[0].is_complete());
    assert!(requests.try_recv().is_err());
}

#[test]
fn failed_or_canceled_write_cannot_commit_a_later_successful_chunk() {
    let (mut job, _consumer) = load_job();
    job.inflight = 2;
    // A None lease makes an attempted commit fail loudly; the canceled owner
    // must never reach commit even when its final I/O completes successfully.
    job.writes.push(Write {
        lease: None,
        remaining: 2,
    });
    job.complete(
        Some(0),
        Err(TransferError::new(ErrorKind::Backend, "first chunk failed")),
    );
    job.complete(Some(0), Ok(()));
    assert_eq!(job.writes[0].remaining, 0);
    assert_eq!(job.error.as_deref(), Some("first chunk failed"));
    assert!(job.is_complete());
}

#[test]
fn cancellation_observation_waits_for_io_and_preserves_later_failure() {
    for submitted in [false, true] {
        for failed in [false, true] {
            let (mut job, consumer) = load_job();
            job.observation.admitted();
            if submitted {
                job.observation.submitted();
                job.inflight = 1;
            }
            drop(consumer);
            job.cancel_abandoned();
            assert_eq!(job.outcome, Outcome::Cancelled);
            assert_eq!(job.is_complete(), !submitted);
            if submitted {
                job.complete(
                    None,
                    if failed {
                        Err(TransferError::new(ErrorKind::Backend, "I/O failed"))
                    } else {
                        Ok(())
                    },
                );
            } else if failed {
                job.fail("preparation failed".into());
            }
            assert_eq!(
                job.outcome,
                if failed {
                    Outcome::Failed
                } else {
                    Outcome::Cancelled
                }
            );
            assert!(job.is_complete());
            // Only this safe terminal path emits the cancellation/failure sample.
            job.finish();
        }
    }
}

#[test]
fn canceled_decoder_preparation_is_not_an_io_failure() {
    let (mut job, consumer) = load_job();
    job.decoding = true;
    drop(consumer);
    job.cancel_abandoned();
    let mut jobs = VecDeque::from([job]);
    let mut active = Some(Decode {
        job: 0,
        reads: Vec::new(),
        offsets: Vec::new(),
        base: 0,
        remaining: 0,
        phase: DecodePhase::Preparing,
    });
    let (decoder, requests, replies) = decoder_channels();
    // No native registration is needed: cancellation happens before any file I/O.
    // SAFETY: with no engine this registration is bookkeeping only; the
    // fixture never submits or dereferences the synthetic address.
    let input = unsafe { CudaRegistration::new(None, 1, 4096, 0, Arc::new(())) }.unwrap();
    replies.send(DecodeReply::Prepared(Ok(input))).unwrap();
    poll_decode(&decoder, &mut active, &mut jobs);
    assert!(active.is_none());
    assert!(jobs[0].is_complete());
    assert_eq!(jobs[0].outcome, Outcome::Cancelled);
    assert!(requests.try_recv().is_err());
    jobs.pop_front().unwrap().finish();
}

#[test]
fn uncertain_file_receipt_retains_caller_completion() {
    let (mut job, mut completion) = load_job();
    job.inflight = 1;
    job.complete(
        None,
        Err(TransferError::new(
            ErrorKind::Unquiesced,
            "unknown physical drain",
        )),
    );
    assert!(job.is_complete());
    job.finish();
    assert!(matches!(
        completion.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    let (mut drained, mut completion) = load_job();
    drained.fail("timeout with physical drain".into());
    drained.finish();
    assert!(completion.try_recv().unwrap().result.is_err());
}
