use super::*;
use crate::peer::restore::tests::{Control, registered, source};
use crate::transfer::layout::{KVCacheGeometry, KVCacheLayout};
use crate::transfer::worker::TransferBlock;
use orbitkv_proto::proto::engine::TransferSlotInfo;

fn layer(split: bool, page_first: bool) -> LayerTransferData {
    let mut layout = KVCacheLayout::bind(
        0x10000,
        4096,
        KVCacheGeometry::new(
            4,
            64,
            if split { 512 } else { 0 },
            if split { 2 } else { 1 },
            Some(128),
            1,
        )
        .unwrap(),
    )
    .unwrap();
    layout.owner = Some(registered(
        Arc::new(Control::default()),
        0x10000,
        4096,
        Arc::new(()),
    ));
    LayerTransferData {
        layer_name: "layer".into(),
        layout,
        blocks: vec![TransferBlock {
            block_idx: 2,
            block: TransferPayload::Peer {
                source: source(
                    Arc::new(Control::default()),
                    vec![TransferSlotInfo {
                        k_ptr: 0x20000,
                        k_size: 1024,
                        checksums: serde_json::to_vec(
                            &(1..=if split && !page_first { 32 } else { 16 })
                                .map(|i| crate::integrity::Checksum {
                                    end: i * 64,
                                    crc: 0,
                                })
                                .collect::<Vec<_>>(),
                        )
                        .unwrap(),
                        v_ptr: if split && !page_first { 0x30000 } else { 0 },
                        v_size: if split && !page_first { 1024 } else { 0 },
                        ..Default::default()
                    }],
                ),
                slot_id: 0,
                offset: 128,
            },
        }],
    }
}

#[tokio::test]
async fn raw_ranges_preserve_layer_page_split_and_sparse_destination_layouts() {
    for split in [false, true] {
        for page_first in [false, true] {
            let layer = layer(split, page_first);
            let copies = plan(std::slice::from_ref(&layer), 0).unwrap();
            assert_eq!(copies[0].remote, 0x20080);
            assert_eq!(copies[0].target, 0x10100);
            assert_eq!(copies[0].bytes, 64);
            assert_eq!(copies.len(), if split { 2 } else { 1 });
            if split {
                assert_eq!(copies[1].target, 0x10300);
                assert_eq!(copies[1].remote, if page_first { 0x200c0 } else { 0x30080 });
            }
            assert!(
                super::super::build_copy_descs(std::slice::from_ref(&layer))
                    .unwrap()
                    .0
                    .is_empty(),
                "peer GPU ranges must never reach OrbitKV H2D"
            );
        }
    }
}

#[tokio::test]
async fn reject_unowned_or_out_of_grant_destination_before_submission() {
    let mut layers = vec![layer(false, false)];
    layers[0].layout.owner = None;
    assert!(validate(&layers, 0).unwrap_err().contains("owned CUDA"));
    layers[0].layout.owner = Some(registered(
        Arc::new(Control::default()),
        0x10000,
        4096,
        Arc::new(()),
    ));
    if let TransferPayload::Peer { offset, .. } = &mut layers[0].blocks[0].block {
        *offset = 1024;
    }
    assert!(
        validate(&layers, 0)
            .unwrap_err()
            .contains("exceeds authorized")
    );
}

#[tokio::test]
async fn encoded_inputs_require_matching_representation_and_bounded_device_scratch() {
    let mut layers = vec![layer(false, false)];
    if let TransferPayload::Peer { source, offset, .. } = &mut layers[0].blocks[0].block {
        *offset = 0;
        Arc::make_mut(&mut source.slots)[0]
            .checksums
            .insert(0, crate::integrity::Checksum { end: 32, crc: 0 });
        Arc::make_mut(&mut source.slots)[0].meta.encoding = Some(vec![EncodedSegment {
            version: 1,
            format: orbitkv_state::StorageFormat::Ans,
            logical_bytes: 64,
            stored_bytes: 32,
            checksum: 0,
        }]);
    }
    layers[0].layout.storage_format = orbitkv_state::StorageFormat::Ans;
    assert!(validate(&layers, 63).unwrap_err().contains("budget"));
    assert!(plan(&layers, 64).unwrap()[0].encoded.is_some());
    let control = Arc::new(Control::default());
    let input = registered(control.clone(), 0x40000, 32, Arc::new(()));
    for _ in 0..3 {
        let mut pending = VecDeque::from(plan(&layers, 64).unwrap());
        let batch = batch(&mut pending, Some(&input)).unwrap();
        assert!(pending.is_empty());
        assert_eq!(batch.slices[0].local.as_ptr() as u64, input.address);
        assert!(Arc::ptr_eq(&batch.owners[0], &input));
    }
    assert_eq!(
        control
            .registrations
            .load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    assert_eq!(
        control.cleanups.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    layers[0].layout.storage_format = orbitkv_state::StorageFormat::Exact;
    assert!(
        validate(&layers, 64)
            .unwrap_err()
            .contains("representation")
    );
}

#[tokio::test]
async fn unknown_peer_drain_keeps_client_pending_backing_and_admission_charged() {
    use super::super::{DeviceRestoreAdmission, LoadTask, complete_peer};
    use crate::cost::Observation;
    use crate::planning::restore::RestorePlan;
    let layer = layer(false, false);
    let owner = Arc::downgrade(layer.layout.owner.as_ref().unwrap());
    let admission = Arc::new(DeviceRestoreAdmission {
        permits: Arc::new(tokio::sync::Semaphore::new(1)),
        active: std::sync::atomic::AtomicUsize::new(0),
    });
    let permit = admission.try_acquire().unwrap();
    let (completion, mut receiver) = tokio::sync::oneshot::channel();
    let task = LoadTask {
        plan: RestorePlan::new(0, []).unwrap(),
        layers: vec![layer],
        completion,
        reservations: vec![],
        peer_bytes: 0,
        codec_budget: 0,
        decode_ready_started: std::time::Instant::now(),
        decode_ready_observation: Box::new(Observation::disabled()),
        decode_admission: Some(permit),
    };
    let poisoned = std::cell::Cell::new(false);
    assert!(
        complete_peer(
            task,
            Observation::disabled(),
            &poisoned,
            Err(PeerTransferError::Unquiesced)
        )
        .is_none()
    );
    assert!(poisoned.get());
    assert!(matches!(
        receiver.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert!(owner.upgrade().is_some());
    assert_eq!(Arc::strong_count(&admission), 2);
}

// Exercise the real layout planner and batch submission with disjoint CPU backing.
async fn restores(count: usize, block_bytes: usize) {
    use std::sync::{Mutex, atomic::Ordering};
    let control = Arc::new(Control::default());
    let original: Vec<u8> = (0..block_bytes * 2).map(|n| (n % 251) as u8).collect();
    let backing = Arc::new(Mutex::new(vec![0u8; original.len() + 32]));
    let address = backing.lock().unwrap().as_mut_ptr() as u64;
    let registration = registered(
        control.clone(),
        address,
        original.len() + 32,
        backing.clone(),
    );
    for _ in 0..count {
        let source = source(
            control.clone(),
            vec![TransferSlotInfo {
                k_ptr: original.as_ptr() as u64,
                k_size: original.len() as u64,
                checksums: serde_json::to_vec(
                    &original
                        .chunks(block_bytes)
                        .enumerate()
                        .map(|(i, bytes)| crate::integrity::Checksum {
                            end: ((i + 1) * block_bytes) as u64,
                            crc: crc32fast::hash(bytes),
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
                ..Default::default()
            }],
        );
        let mut layout = KVCacheLayout::bind(
            address + 16,
            original.len(),
            KVCacheGeometry::new(2, block_bytes, 0, 1, None, 1).unwrap(),
        )
        .unwrap();
        layout.owner = Some(registration.clone());
        let layers = [LayerTransferData {
            layer_name: "layer".into(),
            layout,
            blocks: (0..2)
                .map(|i| TransferBlock {
                    block_idx: i,
                    block: TransferPayload::Peer {
                        source: source.clone(),
                        slot_id: 0,
                        offset: i * block_bytes,
                    },
                })
                .collect(),
        }];
        let mut pending = VecDeque::from(plan(&layers, 0).unwrap());
        let batch = batch(&mut pending, None).unwrap();
        assert!(pending.is_empty(), "one plan must be one native submission");
        assert_eq!(batch.slices.len(), 2);
        // SAFETY: the CPU control borrows actual disjoint bytes until synchronous completion.
        assert_eq!(
            // SAFETY: this Control backend never accesses synthetic device addresses;
            // batch owners remain retained while submission/drain counts are observed.
            unsafe { batch.source.read_gpu(&batch.slices, &batch.owners) }.unwrap(),
            original.len()
        );
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), count);
    let observed = control.options.lock().unwrap();
    assert_eq!(observed.len(), count);
    assert_eq!(
        *observed,
        *control.admissions.lock().unwrap(),
        "restores must not refresh the grant deadline"
    );

    assert_eq!(control.registrations.load(Ordering::SeqCst), 1);
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 0);
    let target = backing.lock().unwrap();
    assert_eq!(&target[16..16 + original.len()], original);
    assert_eq!(&target[..16], &[0; 16]);
    assert_eq!(&target[16 + original.len()..], &[0; 16]);
    drop(registration);
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn repeated_restores_register_the_tensor_once_until_release() {
    restores(8, 4096).await;
}
#[tokio::test]
async fn restore_larger_than_old_chunk_is_one_engine_submission() {
    restores(1, 9 << 20).await;
}

#[tokio::test]
async fn engine_limits_bound_large_and_fragmented_plans_without_reregistering() {
    let layer = layer(false, false);
    let mut copies = plan(std::slice::from_ref(&layer), 0).unwrap();
    copies[0].bytes = BATCH_BYTES + 1;
    let mut pending = VecDeque::from(copies.clone());
    assert_eq!(
        batch(&mut pending, None).unwrap().slices[0].length,
        BATCH_BYTES
    );
    assert_eq!(batch(&mut pending, None).unwrap().slices[0].length, 1);
    assert!(pending.is_empty());
    copies[0].bytes = 1;
    let mut pending = VecDeque::from(vec![copies[0].clone(); BATCH_RANGES + 1]);
    assert_eq!(
        batch(&mut pending, None).unwrap().slices.len(),
        BATCH_RANGES
    );
    assert_eq!(batch(&mut pending, None).unwrap().slices.len(), 1);
    assert!(pending.is_empty());
}

#[tokio::test]
#[ignore = "explicit CPU CUDA shim; native TCP receive is covered separately"]
async fn raw_peer_gpu_receive_checks_once_after_drain_before_readiness() {
    use cudarc::driver::{CudaContext, DevicePtr};
    let Some(metrics) = crate::integrity::tests::isolate(
        "transfer::worker::peer::tests::raw_peer_gpu_receive_checks_once_after_drain_before_readiness",
    ) else {
        return;
    };
    assert_eq!(
        std::env::var("ORBITKV_REQUIRE_LOCAL_COPY_SHIM").as_deref(),
        Ok("1")
    );
    let storage = crate::storage::Storage::new_with_config(
        16 << 20,
        false,
        crate::EngineConfig {
            transfer_transport: crate::TransportHint::Tcp,
            ..Default::default()
        },
        &[],
    )
    .unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let destination = Arc::new(stream.alloc_zeros::<u8>(512).unwrap());
    let (address, ready) = destination.device_ptr(&stream);
    drop(ready);
    stream.synchronize().unwrap();
    // SAFETY: the initialized destination/context outlive the bound stream and registration.
    let anchor = unsafe {
        CudaRegistration::new(
            storage.peer_transport().cloned(),
            address,
            512,
            0,
            destination.clone(),
        )
    }
    .unwrap();
    let runtime = super::super::init_worker(
        0,
        crate::TransferMode::Direct,
        Some(anchor.clone()),
        storage.peer_transport().cloned(),
    )
    .unwrap();
    let control = Arc::new(Control::default());
    let registered = registered(control.clone(), address, 512, destination.clone());
    for (index, corrupt) in [false, true].into_iter().enumerate() {
        let mut original = vec![0x35u8; 512];
        let expected = crc32fast::hash(&original);
        if corrupt {
            original[17] ^= 1;
        }
        let peer = source(
            control.clone(),
            vec![TransferSlotInfo {
                k_ptr: original.as_ptr() as u64,
                k_size: 512,
                checksums: serde_json::to_vec(&vec![crate::integrity::Checksum {
                    end: 512,
                    crc: expected,
                }])
                .unwrap(),
                ..Default::default()
            }],
        );
        let mut layout = KVCacheLayout::bind(
            address,
            512,
            KVCacheGeometry::new(1, 512, 0, 1, None, 1).unwrap(),
        )
        .unwrap();
        layout.owner = Some(registered.clone());
        let layers = [LayerTransferData {
            layer_name: "raw".into(),
            layout,
            blocks: vec![TransferBlock {
                block_idx: 0,
                block: TransferPayload::Peer {
                    source: peer,
                    slot_id: 0,
                    offset: 0,
                },
            }],
        }];
        let result = restore(&runtime, &layers, 8192);
        assert_eq!(result.is_err(), corrupt, "{result:?}");
        assert_eq!(
            crate::integrity::tests::bytes(&metrics, "peer_gpu"),
            (index as u64 + 1) * 512
        );
        assert_eq!(
            control.reads.load(std::sync::atomic::Ordering::SeqCst),
            index + 1
        );
    }
    registered.close().unwrap();
    runtime.drain_transfers().unwrap();
    drop(runtime);
    anchor.close().unwrap();
}
