use super::*;
use crate::peer::restore::tests::{Control, source};
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
    layout.owner = Some(Arc::new(()));
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
    layers[0].layout.owner = Some(Arc::new(()));
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
