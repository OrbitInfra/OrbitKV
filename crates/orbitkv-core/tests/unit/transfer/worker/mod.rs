use super::*;

fn empty_save_instance() -> Arc<InstanceContext> {
    Arc::new(InstanceContext::new("cpu-save".into(), "cpu-save".into(), 1, 1, false).unwrap())
}

#[test]
fn save_admission_retains_lost_owners_and_releases_only_completed_work() {
    let usage = Arc::new(Mutex::new(SaveUsage::default()));
    let admissions = (0..MAX_SAVE_OPERATIONS)
        .map(|_| SaveAdmission::reserve(Arc::clone(&usage), 128, 4096).unwrap())
        .collect::<Vec<_>>();
    assert!(SaveAdmission::reserve(Arc::clone(&usage), 1, 0).is_err());
    let retained = admissions[0].clone();
    retained.retain();
    retained.retain();
    assert_eq!(usage.lock().retained, 1);
    admissions[1].complete();
    admissions[1].complete();
    let replacement = SaveAdmission::reserve(Arc::clone(&usage), 128, 4096).unwrap();
    drop(replacement);
    assert_eq!(usage.lock().operations, MAX_SAVE_OPERATIONS);
    assert_eq!(usage.lock().retained, 2);
    assert!(SaveAdmission::reserve(Arc::clone(&usage), 1, 0).is_err());
    for admission in &admissions {
        admission.complete();
    }
    drop(admissions);
    drop(retained);
    let state = usage.lock();
    assert_eq!(state.operations, 1, "lost receipt must stay charged");
    assert_eq!(state.retained, 1);
    assert_eq!(state.metadata_bytes, 128);
    assert_eq!(state.logical_bytes, 4096);
}

#[test]
fn save_metadata_budget_counts_spare_capacity_and_rejects_overflow() {
    let usage = Arc::new(Mutex::new(SaveUsage::default()));
    let full = SaveAdmission::reserve(Arc::clone(&usage), MAX_SAVE_METADATA_BYTES, 0).unwrap();
    assert!(SaveAdmission::reserve(Arc::clone(&usage), 1, 0).is_err());
    assert!(SaveAdmission::reserve(Arc::clone(&usage), usize::MAX, 0).is_err());
    full.complete();
    let spare = Vec::<SaveGroup>::with_capacity(17);
    let (bytes, logical) = save_shape(&Vec::new(), &Vec::new(), &spare).unwrap();
    assert_eq!(logical, 0);
    assert_eq!(
        bytes,
        std::mem::size_of::<SaveTask>() + spare.capacity() * std::mem::size_of::<SaveGroup>()
    );
}

#[test]
fn decode_restore_admission_is_device_scoped_and_bounded() {
    let admission = device_restore_admission(i32::MAX - 20);
    let mut permits = Vec::new();
    for depth in 1..=MAX_DEVICE_RESTORES {
        let permit = admission.try_acquire().unwrap();
        assert_eq!(permit.depth, depth as u32);
        permits.push(permit);
    }
    assert!(admission.try_acquire().is_err());
    permits.pop();
    assert!(admission.try_acquire().is_ok());
}

fn empty_restore_plan(device_id: i32) -> crate::planning::restore::RestorePlan {
    crate::planning::restore::RestorePlan::new(
        device_id,
        std::iter::empty::<(usize, &crate::RestoreSource)>(),
    )
    .unwrap()
}

#[test]
fn gpu_ssd_write_admission_is_shared_per_device_and_recreated_after_release() {
    let first = device_ssd_write_admission(i32::MAX);
    let same = device_ssd_write_admission(i32::MAX);
    let other = device_ssd_write_admission(i32::MAX - 1);
    assert!(Arc::ptr_eq(&first, &same));
    assert!(!Arc::ptr_eq(&first, &other));
    let permits = (0..ssd::MAX_WRITES)
        .map(|_| Arc::clone(&first).try_acquire_owned().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(same.available_permits(), 0);
    assert!(Arc::clone(&same).try_acquire_owned().is_err());
    drop(permits);
    assert_eq!(same.available_permits(), ssd::MAX_WRITES);
    drop(first);
    drop(same);
    let replacement = device_ssd_write_admission(i32::MAX);
    assert_eq!(replacement.available_permits(), ssd::MAX_WRITES);
}

#[test]
fn cufile_worker_ownership_is_shared_per_device() {
    let first = device_cufile_worker_admission(i32::MAX - 10);
    let same = device_cufile_worker_admission(i32::MAX - 10);
    let other = device_cufile_worker_admission(i32::MAX - 11);
    assert!(Arc::ptr_eq(&first, &same));
    assert!(!Arc::ptr_eq(&first, &other));
    let owner = Arc::clone(&first).try_acquire_owned().unwrap();
    assert!(Arc::clone(&same).try_acquire_owned().is_err());
    assert!(Arc::clone(&other).try_acquire_owned().is_ok());
    drop(owner);
    assert!(Arc::clone(&same).try_acquire_owned().is_ok());
}

#[test]
fn save_task_owns_shared_admission_until_terminal_drop() {
    let admission = device_ssd_write_admission(i32::MAX - 2);
    let permit = Arc::clone(&admission).try_acquire_owned().unwrap();
    let (reply, _receiver) = oneshot::channel();
    let command = WorkerCommand::Save(
        SaveTask {
            layers: Vec::new(),
            reply,
            ssd_writes: Vec::new(),
            codec_groups: Vec::new(),
            storage: None,
            numa: NumaNode::UNKNOWN,
            ssd_admission: Some(permit),
            admission: SaveAdmission::acquire(std::mem::size_of::<SaveTask>(), 0).unwrap(),
            _instance: empty_save_instance(),
            #[cfg(feature = "tracing")]
            trace_ctx: None,
        },
        Observation::disabled(),
    );
    assert_eq!(admission.available_permits(), ssd::MAX_WRITES - 1);
    drop(command);
    assert_eq!(admission.available_permits(), ssd::MAX_WRITES);
}

#[tokio::test]
async fn shared_admission_saturation_falls_back_before_worker_submission() {
    for blocked_cufile_owner in [false, true] {
        let write_admission = Arc::new(Semaphore::new(ssd::MAX_WRITES));
        let cufile_admission = Arc::new(Semaphore::new(1));
        let write_permits = if blocked_cufile_owner {
            Vec::new()
        } else {
            (0..ssd::MAX_WRITES)
                .map(|_| Arc::clone(&write_admission).try_acquire_owned().unwrap())
                .collect::<Vec<_>>()
        };
        let cufile_permit = blocked_cufile_owner
            .then(|| Arc::clone(&cufile_admission).try_acquire_owned().unwrap());
        let (load_tx, _load_rx) = mpsc::unbounded_channel();
        let (save_tx, mut save_rx) = mpsc::unbounded_channel();
        let pool = GpuWorkerPool {
            device_id: i32::MAX - 3,
            numa_node: NumaNode::UNKNOWN,
            transfer_mode: TransferMode::Direct,
            ssd_tx: Mutex::new(None),
            ssd_host_tx: Mutex::new(None),
            codec_write_tx: Mutex::new(None),
            ssd_write_admission: Arc::clone(&write_admission),
            decode_restore_admission: device_restore_admission(i32::MAX - 3),
            cufile_worker_admission: Arc::clone(&cufile_admission),
            cufile_worker_owner: Mutex::new(None),
            load_tx,
            save_tx,
            closed: Mutex::new(false),
            drained: OnceCell::new(),
        };
        let mut saving = Box::pin(pool.batch_save(
            empty_save_instance(),
            Vec::new(),
            Vec::new(),
            vec![SaveGroup {
                key: crate::block::StateKey::new("shared-admission".into(), vec![1]),
                blocks: Vec::new(),
            }],
            None,
        ));
        assert!(futures::poll!(saving.as_mut()).is_pending());
        let Some(WorkerCommand::Save(task, _)) = save_rx.recv().await else {
            panic!("saturated GPU write must use the ordinary save lane")
        };
        assert!(task.codec_groups.is_empty());
        assert!(task.ssd_writes.is_empty());
        assert!(task.ssd_admission.is_none());
        assert!(task.reply.send(Ok(task.layers)).is_ok());
        saving.await.unwrap();
        drop(pool);
        drop(write_permits);
        drop(cufile_permit);
        assert_eq!(write_admission.available_permits(), ssd::MAX_WRITES);
        assert_eq!(cufile_admission.available_permits(), 1);
    }
}

#[test]
fn managed_restore_rejects_overlapping_targets_before_dispatch() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, _save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(0),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let mut layout = KVCacheLayout::bind(
        0x10000,
        257 * 4096,
        crate::transfer::layout::KVCacheGeometry::new(257, 4096, 0, 1, None, 1).unwrap(),
    )
    .unwrap();
    layout.storage_format = orbitkv_state::StorageFormat::Fp8FromBf16;
    for codec_budget in [4096, 64 * 1024 * 1024] {
        for indices in [vec![0, 0], (0..257).chain([0]).collect()] {
            let (completion, _) = oneshot::channel();
            let task = LoadTask {
                plan: empty_restore_plan(0),
                layers: vec![LayerTransferData {
                    layer_name: "attention".into(),
                    layout: layout.clone(),
                    blocks: indices
                        .into_iter()
                        .map(|block_idx| TransferBlock {
                            block_idx,
                            // Rejection must precede source access or GPU work.
                            block: TransferPayload::Pending,
                        })
                        .collect(),
                }],
                completion,
                reservations: vec![],
                codec_budget,
                decode_ready_started: Instant::now(),
                decode_ready_observation: Box::new(Observation::disabled()),
                decode_admission: None,
            };
            let error = pool.submit_load(task).unwrap_err();
            assert!(error.to_string().contains("overlap"));
            assert!(matches!(
                load_rx.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }
}

#[test]
fn restore_plan_must_target_the_worker_device() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, _save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(0),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let (completion, _) = oneshot::channel();
    let error = pool
        .submit_load(LoadTask {
            plan: empty_restore_plan(1),
            layers: Vec::new(),
            completion,
            reservations: Vec::new(),
            codec_budget: 0,
            decode_ready_started: Instant::now(),
            decode_ready_observation: Box::new(Observation::disabled()),
            decode_admission: None,
        })
        .unwrap_err();
    assert!(error.to_string().contains("targets device 1"));
    assert!(matches!(
        load_rx.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn drain_rejects_new_transfers_and_waits_for_all_workers() {
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let (ssd_tx, mut ssd_rx) = mpsc::unbounded_channel();
    let (ssd_host_tx, mut ssd_host_rx) = mpsc::unbounded_channel();
    let (codec_write_tx, mut codec_write_rx) = mpsc::unbounded_channel();
    let cufile_admission = Arc::new(Semaphore::new(1));
    let cufile_owner = Arc::clone(&cufile_admission).try_acquire_owned().unwrap();
    let pool = Arc::new(GpuWorkerPool {
        device_id: 0,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        ssd_tx: Mutex::new(Some(ssd_tx)),
        ssd_host_tx: Mutex::new(Some(ssd_host_tx)),
        codec_write_tx: Mutex::new(Some(codec_write_tx)),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(0),
        cufile_worker_admission: Arc::clone(&cufile_admission),
        cufile_worker_owner: Mutex::new(Some(cufile_owner)),
        load_tx,
        save_tx,
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    });
    let queued_load = || {
        let (reply, _result) = oneshot::channel();
        WorkerCommand::Load(
            LoadTask {
                plan: empty_restore_plan(0),
                layers: vec![],
                completion: reply,
                reservations: vec![],
                codec_budget: 64 * 1024 * 1024,
                decode_ready_started: Instant::now(),
                decode_ready_observation: Box::new(Observation::disabled()),
                decode_admission: None,
            },
            Observation::disabled(),
        )
    };
    pool.submit(queued_load(), false).unwrap();
    let draining = Arc::clone(&pool);
    let waiter = tokio::spawn(async move { draining.drain().await });
    assert!(matches!(
        load_rx.recv().await,
        Some(WorkerCommand::Load(..))
    ));
    let Some(WorkerCommand::Drain(load_ack)) = load_rx.recv().await else {
        panic!("missing load barrier")
    };
    let Some(WorkerCommand::Drain(save_ack)) = save_rx.recv().await else {
        panic!("missing save barrier")
    };
    let Some(WorkerCommand::Drain(ssd_ack)) = ssd_rx.recv().await else {
        panic!("missing SSD barrier")
    };
    let Some(WorkerCommand::Drain(codec_write_ack)) = codec_write_rx.recv().await else {
        panic!("missing encoded writeback barrier")
    };
    let Some(WorkerCommand::Drain(ssd_host_ack)) = ssd_host_rx.recv().await else {
        panic!("missing SSD host barrier")
    };
    let error = pool.submit(queued_load(), false).unwrap_err();
    assert!(error.to_string().contains("draining"));
    assert!(
        pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None)
            .await
            .is_err()
    );
    load_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "load completion alone must not release mappings"
    );
    save_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "SSD completion must precede mapping release"
    );
    ssd_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished());
    codec_write_ack.send(Ok(())).unwrap();
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "host reads must drain before mapping release"
    );
    ssd_host_ack.send(Ok(())).unwrap();
    waiter.await.unwrap().unwrap();
    assert_eq!(cufile_admission.available_permits(), 1);
    pool.drain().await.unwrap();
}

#[test]
fn transfer_cost_shape_uses_logical_ranges_and_actual_encoding() {
    use std::num::NonZeroU64;

    use crate::block::Segment;
    use crate::codec::EncodedSegment;
    use crate::memory::pool::PinnedAllocator;
    use orbitkv_state::StorageFormat;

    let pool = PinnedAllocator::new_global(4096, 1, false, None);
    let block = |bytes: usize| {
        let allocation = pool
            .allocate(NonZeroU64::new(bytes as u64).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), bytes, allocation))
    };
    let mut encoded = block(512);
    encoded.storage_format = StorageFormat::Ans;
    encoded.encoding = Some(vec![EncodedSegment {
        version: 1,
        format: StorageFormat::Ans,
        logical_bytes: 600,
        stored_bytes: 100,
        checksum: 0,
    }]);
    let layers = vec![LayerTransferData {
        layer_name: "split".into(),
        layout: KVCacheLayout::bind(
            0x10000,
            4096,
            crate::transfer::layout::KVCacheGeometry::new(2, 300, 2048, 2, None, 512).unwrap(),
        )
        .unwrap(),
        blocks: vec![
            TransferBlock {
                block_idx: 0,
                block: TransferPayload::Owned(encoded),
            },
            TransferBlock {
                block_idx: 1,
                block: TransferPayload::Owned(block(1024)),
            },
        ],
    }];
    assert_eq!(transfer_shape(&layers), (1200, 4));
    let (key, bytes) = transfer_key(&layers, 2, TransferMode::Direct, false, false);
    assert_eq!(bytes, 1200);
    assert_eq!(
        key,
        CostEstimateKey::new(
            CostObservationKind::GpuDecode,
            ExecutionResource::Gpu(2),
            Representation::Mixed,
            1200,
            4
        )
    );
    let (key, _) = transfer_key(&layers, 2, TransferMode::Direct, false, true);
    assert_eq!(
        key,
        CostEstimateKey::new(
            CostObservationKind::GpuSsdLoad,
            ExecutionResource::Gpu(2),
            Representation::Mixed,
            1200,
            4
        )
    );
}

#[test]
fn failed_gpu_work_is_not_reclassified_by_consumer_loss() {
    for (completed, closed, expected) in [
        (true, false, Outcome::Completed),
        (true, true, Outcome::Cancelled),
        (false, false, Outcome::Failed),
        (false, true, Outcome::Failed),
    ] {
        assert_eq!(terminal_outcome(completed, closed), expected);
    }
}

#[test]
fn raw_copy_candidates_distinguish_dma_coalescing_and_direction() {
    let mut host = [0u8; 16];
    let contiguous: Vec<_> = (0..4)
        .map(|index| CopyDesc {
            device: 0x1000 + (index * 4) as u64,
            host: host.as_mut_ptr().wrapping_add(index * 4),
            host_device: 0x2000 + (index * 4) as u64,
            size: 4,
            device_allocation: 1,
            host_registration: 2,
        })
        .collect();
    let mut fragmented = contiguous.clone();
    fragmented.swap(1, 2);
    let mut allocations = contiguous.clone();
    for (index, copy) in allocations.iter_mut().enumerate() {
        copy.host_registration = index;
    }
    for write in [false, true] {
        let (merged_keys, bytes) = raw_copy_keys(&contiguous, 3, write, usize::MAX);
        assert_eq!(bytes, 16);
        let paths = if write {
            [
                CostObservationKind::GpuSaveDirect,
                CostObservationKind::GpuSaveKernel,
            ]
        } else {
            [
                CostObservationKind::GpuLoadDirect,
                CostObservationKind::GpuLoadKernel,
            ]
        };
        for (key, path) in merged_keys.iter().zip(paths) {
            assert_eq!(
                *key,
                CostEstimateKey::new(path, ExecutionResource::Gpu(3), Representation::Raw, 16, 4)
                    .with_dma_ranges(1)
            );
        }
        for (copies, submissions) in [(&fragmented, 2), (&allocations, 4)] {
            let (keys, bytes) = raw_copy_keys(copies, 3, write, usize::MAX);
            assert_eq!(bytes, 16);
            for ((key, merged_key), path) in keys.iter().zip(merged_keys).zip(paths) {
                assert_ne!(*key, merged_key);
                assert_eq!(
                    *key,
                    CostEstimateKey::new(
                        path,
                        ExecutionResource::Gpu(3),
                        Representation::Raw,
                        16,
                        4
                    )
                    .with_dma_ranges(submissions)
                );
            }
        }
        assert_ne!(
            merged_keys,
            raw_copy_keys(&contiguous, 3, !write, usize::MAX).0
        );
    }
}

#[test]
fn raw_source_ranges_are_checked_before_appending_descriptors() {
    use std::num::NonZeroU64;

    use crate::block::Segment;
    use crate::memory::pool::PinnedAllocator;
    use crate::transfer::layout::BlockCopy;

    let pool = PinnedAllocator::new_global(4096, 1, false, None);
    let segment = |bytes| {
        let allocation = pool
            .allocate(NonZeroU64::new(bytes as u64).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        Segment::new(allocation.as_non_null(), bytes, allocation)
    };
    let contiguous = RawBlock::single_segment(segment(32));
    let split = RawBlock::two_segments(segment(16), segment(12));
    let ranges = || BlockCopies::Split {
        k: BlockCopy {
            addr: 0x1000,
            bytes: 8,
        },
        v: BlockCopy {
            addr: 0x2000,
            bytes: 8,
        },
    };
    let mut copies = Vec::new();
    for (raw, offset) in [(&contiguous, 17), (&split, 5), (&split, usize::MAX)] {
        assert!(append_copy_descs(&mut copies, 7, ranges(), raw, offset).is_err());
        assert!(
            copies.is_empty(),
            "invalid second segment must not append K"
        );
    }
    assert_eq!(
        append_copy_descs(&mut copies, 7, ranges(), &contiguous, 16).unwrap(),
        16
    );
    assert_eq!(copies[1].host, copies[0].host.wrapping_add(8));
    assert_eq!(copies[0].host_registration, copies[1].host_registration);
    copies.clear();
    assert_eq!(
        append_copy_descs(&mut copies, 7, ranges(), &split, 4).unwrap(),
        16
    );
    assert_eq!(
        copies[1].host,
        split.segment_ptr(1).unwrap().as_ptr().wrapping_add(4)
    );
    assert_eq!(copies[0].host_registration, copies[1].host_registration);
    copies.clear();
    assert!(
        append_copy_descs(
            &mut copies,
            7,
            BlockCopies::Contiguous(BlockCopy {
                addr: 0x1000,
                bytes: 33
            }),
            &contiguous,
            0,
        )
        .is_err()
    );
    assert!(copies.is_empty());
}

#[test]
fn managed_restore_retains_sources_and_reservations_through_partial_submission_drain() {
    use std::num::NonZeroU64;
    use std::time::Duration;

    use cudarc::driver::DevicePtr;

    use crate::block::Segment;
    use crate::memory::pool::PinnedAllocator;

    struct Gate {
        entered: std_mpsc::Sender<()>,
        release: std_mpsc::Receiver<()>,
    }
    unsafe extern "C" fn hold_stream(data: *mut std::ffi::c_void) {
        // SAFETY: successful launch transfers this Box to one callback.
        let gate = unsafe { Box::from_raw(data.cast::<Gate>()) };
        let _ = gate.entered.send(());
        let _ = gate.release.recv_timeout(Duration::from_secs(5));
    }
    struct PartialBackend {
        gate: Mutex<Option<Gate>>,
        submitted: std_mpsc::Sender<()>,
    }
    impl TransferBackend for PartialBackend {
        fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
            let gate = Box::into_raw(Box::new(self.gate.lock().take().unwrap()));
            // SAFETY: the callback owns the gate until it returns and does not call CUDA.
            let launched = unsafe {
                cudarc::driver::result::stream::launch_host_function(
                    stream.cu_stream(),
                    hold_stream,
                    gate.cast(),
                )
            };
            if let Err(error) = launched {
                // SAFETY: a rejected launch did not take ownership of the callback.
                unsafe { drop(Box::from_raw(gate)) };
                return Err(error.to_string());
            }
            MemcpyBackend::new(stream.context())?.h2d(copies, stream)?;
            self.submitted.send(()).unwrap();
            Err("injected failure after accepting a copy".into())
        }
        fn d2h(&self, _: &[CopyDesc], _: &Arc<CudaStream>) -> Result<(), String> {
            unreachable!("restore test")
        }
        fn name(&self) -> &'static str {
            "direct"
        }
    }

    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let target = stream.alloc_zeros::<u8>(32).unwrap();
    let device = target.device_ptr(&stream).0;
    stream.synchronize().unwrap();
    let pool = PinnedAllocator::new_global(4096, 1, false, None);
    let allocation = pool
        .allocate(NonZeroU64::new(32).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    // SAFETY: this test uniquely owns all 32 bytes before sealing them.
    unsafe { allocation.as_non_null().as_ptr().write_bytes(0x5a, 32) };
    let raw = RawBlock::single_segment(Segment::new(allocation.as_non_null(), 32, allocation));
    let source = Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)]));
    let weak = Arc::downgrade(&source);
    let budget = crate::query::QueryBudget::new(32, 32).unwrap();
    let crate::QueryAdmission::Admitted(reservation) =
        budget.reserve("raw", "ns", 32, crate::QueryMode::Demand)
    else {
        panic!("budget available")
    };
    reservation.ready(32).unwrap();
    reservation.restoring();
    let (completion, result) = oneshot::channel();
    let planned_source = crate::RestoreSource::Memory(Arc::clone(&source));
    let plan = RestorePlan::new(0, [(0, &planned_source)]).unwrap();
    drop(planned_source);
    let task = LoadTask {
        plan,
        layers: vec![LayerTransferData {
            layer_name: "raw".into(),
            layout: KVCacheLayout::bind(
                device,
                32,
                crate::transfer::layout::KVCacheGeometry::new(1, 32, 0, 1, None, 1).unwrap(),
            )
            .unwrap(),
            blocks: vec![TransferBlock {
                block_idx: 0,
                block: TransferPayload::Cached {
                    sealed: source,
                    slot_id: 0,
                    offset: 0,
                },
            }],
        }],
        completion,
        reservations: vec![reservation],
        codec_budget: 0,
        decode_ready_started: Instant::now(),
        decode_ready_observation: Box::new(Observation::disabled()),
        decode_admission: None,
    };
    let (entered_tx, entered_rx) = std_mpsc::channel();
    let (release_tx, release_rx) = std_mpsc::channel();
    let (submitted_tx, submitted_rx) = std_mpsc::channel();
    let (sender, receiver) = mpsc::unbounded_channel();
    sender
        .send(WorkerCommand::Load(task, Observation::disabled()))
        .unwrap();
    drop(sender);
    let worker = std::thread::spawn(move || {
        let mut runtime = init_worker(0, TransferMode::Direct).unwrap();
        runtime.backend = Box::new(PartialBackend {
            gate: Mutex::new(Some(Gate {
                entered: entered_tx,
                release: release_rx,
            })),
            submitted: submitted_tx,
        });
        worker_loop(0, receiver, runtime);
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    submitted_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    drop(result);
    let retained = weak.upgrade().is_some();
    let charged = matches!(
        budget.reserve("raw", "ns", 1, crate::QueryMode::Demand),
        crate::QueryAdmission::Busy
    );
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(
        retained,
        "dropping the completion receiver must not release active raw sources"
    );
    assert!(
        charged,
        "submitted DMA must retain its query byte reservation"
    );
    assert!(weak.upgrade().is_none());
    assert!(matches!(
        budget.reserve("raw", "ns", 32, crate::QueryMode::Demand),
        crate::QueryAdmission::Admitted(_)
    ));
    assert_eq!(stream.clone_dtoh(&target).unwrap(), vec![0x5a; 32]);
}

#[tokio::test]
async fn submitted_save_reply_loss_is_not_a_completed_error() {
    let (load_tx, _load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: i32::MAX - 31,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        load_tx,
        save_tx,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(i32::MAX - 31),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let owner = Arc::new(vec![37; 4096]);
    let observed_owner = Arc::downgrade(&owner);
    let saving = async {
        let _held = owner;
        pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None)
            .await
    };
    tokio::pin!(saving);
    assert!(futures::poll!(saving.as_mut()).is_pending());
    let Some(WorkerCommand::Save(task, _)) = save_rx.recv().await else {
        panic!("expected admitted SaveTask");
    };
    let admission = Arc::clone(&task.admission);
    // Real queue/oneshot boundary; no allocation, CUDA executor or DMA is created.
    drop(task.reply);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), saving.as_mut())
            .await
            .is_err(),
        "lost worker response fabricated completed error without physical drain"
    );
    assert!(
        observed_owner.upgrade().is_some(),
        "admitted caller owner was lost"
    );
    assert!(admission.retained.load(Ordering::Acquire));
    assert!(!admission.completed.load(Ordering::Acquire));
}

#[tokio::test]
async fn save_completion_receipt_releases_admission_after_queued_no_dma_work() {
    let (load_tx, _load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: i32::MAX - 32,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        load_tx,
        save_tx,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(i32::MAX - 32),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let instance = empty_save_instance();
    let observed_instance = Arc::downgrade(&instance);
    let mut saving = Box::pin(pool.batch_save(instance, Vec::new(), Vec::new(), Vec::new(), None));
    assert!(futures::poll!(saving.as_mut()).is_pending());
    let Some(WorkerCommand::Save(task, _)) = save_rx.recv().await else {
        panic!("expected SaveTask");
    };
    let admission = Arc::clone(&task.admission);
    assert!(!admission.completed.load(Ordering::Acquire));
    drop(saving);
    assert!(
        observed_instance.upgrade().is_some(),
        "SaveTask must hold registration after caller cancellation"
    );
    // Empty payload means this queue control submits no CUDA or SSD operation.
    // Production calls complete only after the respective worker drain boundary.
    task.admission.complete();
    assert!(task.reply.send(Ok(task.layers)).is_err());
    drop(task._instance);
    assert!(observed_instance.upgrade().is_none());
    assert!(admission.completed.load(Ordering::Acquire));
    *pool.closed.lock() = true;
    assert!(
        pool.batch_save(
            empty_save_instance(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None
        )
        .await
        .is_err()
    );
}

#[test]
fn save_queue_admission_is_bounded_in_a_fresh_process() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "transfer::worker::drain_tests::save_queue_admission_child",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(String::from_utf8(output.stdout).unwrap().contains(
        "save_queue_saturation=1024 receipt_repaid=true lost_owner_charged=true no_DMA=true"
    ));
}

#[tokio::test]
#[ignore = "isolated CPU queue budget control with no CUDA allocations or DMA"]
async fn save_queue_admission_child() {
    let (load_tx, _load_rx) = mpsc::unbounded_channel();
    let (save_tx, mut save_rx) = mpsc::unbounded_channel();
    let pool = GpuWorkerPool {
        device_id: i32::MAX - 33,
        numa_node: NumaNode::UNKNOWN,
        transfer_mode: TransferMode::Direct,
        load_tx,
        save_tx,
        ssd_tx: Mutex::new(None),
        ssd_host_tx: Mutex::new(None),
        codec_write_tx: Mutex::new(None),
        ssd_write_admission: Arc::new(Semaphore::new(ssd::MAX_WRITES)),
        decode_restore_admission: device_restore_admission(i32::MAX - 33),
        cufile_worker_admission: Arc::new(Semaphore::new(1)),
        cufile_worker_owner: Mutex::new(None),
        closed: Mutex::new(false),
        drained: OnceCell::new(),
    };
    let mut pending = Vec::new();
    for _ in 0..MAX_SAVE_OPERATIONS {
        let mut saving =
            Box::pin(pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None));
        assert!(futures::poll!(saving.as_mut()).is_pending());
        pending.push(saving);
    }
    assert!(
        pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None)
            .await
            .is_err()
    );
    assert_eq!(save_rx.len(), MAX_SAVE_OPERATIONS);
    let Some(WorkerCommand::Save(task, _)) = save_rx.recv().await else {
        panic!("expected SaveTask");
    };
    let admission = Arc::clone(&task.admission);
    // These empty-payload tasks have no physical device operation to drain.
    task.admission.complete();
    assert!(task.reply.send(Ok(task.layers)).is_ok());
    assert!(pending.remove(0).await.is_ok());
    let mut replacement =
        Box::pin(pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None));
    assert!(futures::poll!(replacement.as_mut()).is_pending());
    drop(replacement);
    drop(pending);
    drop(save_rx);
    assert_eq!(admission.usage.lock().operations, MAX_SAVE_OPERATIONS);
    assert_eq!(admission.usage.lock().retained, MAX_SAVE_OPERATIONS);
    assert!(
        pool.batch_save(empty_save_instance(), vec![], vec![], vec![], None)
            .await
            .is_err()
    );
    println!("save_queue_saturation=1024 receipt_repaid=true lost_owner_charged=true no_DMA=true");
}
