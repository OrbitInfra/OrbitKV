use super::*;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[test]
fn transfer_authorization_requires_the_source_epoch_before_destination_use() {
    let records = [orbitkv_state::InventoryRecord {
        key: orbitkv_state::StateKey::new("source".into(), vec![1]),
        sequence: 1,
        present: true,
        metadata: None,
    }];
    let mut response = QueryBlocksForTransferResponse {
        transfer_endpoint: "127.0.0.1:15123".into(),
        blocks: vec![TransferBlockInfo {
            block_hash: vec![1],
            slots: vec![],
        }],
        ..Default::default()
    };
    for (epoch, valid) in [
        ("", false),
        ("0", false),
        ("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", false),
        ("0123456789abcdef0123456789abcdef", true),
    ] {
        response.native_source_epoch = epoch.into();
        assert_eq!(
            valid_transfer_authorization(&response, &records),
            valid,
            "epoch={epoch}"
        );
    }
    response.blocks[0].block_hash = vec![2];
    assert!(!valid_transfer_authorization(&response, &records));
}

fn test_allocate_fn(calls: Arc<AtomicUsize>) -> AllocateFn {
    let allocator = Arc::new(crate::memory::pool::PinnedAllocator::new_global(
        32 * 1024 * 1024,
        1,
        false,
        None,
    ));
    Arc::new(move |size, _numa| {
        calls.fetch_add(1, Ordering::Relaxed);
        allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN)
    })
}

fn remaining(bytes: u64) -> HashMap<NumaNode, u64> {
    HashMap::from([(NumaNode(0), bytes)])
}

#[test]
fn chunked_slabs_bump_within_chunk_then_refill() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(1536));

    let (p1, a1) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("first");
    let (p2, _a2) = slabs.alloc_segment(NumaNode(0), 512, "V").expect("second");
    assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 512);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // Third segment exceeds the current chunk: a fresh chunk is allocated
    // while earlier segments stay valid through their own chunk Arc.
    let (_p3, a3) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("third");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(slabs.chunk_count, 2);
    assert!(!Arc::ptr_eq(&a1, &a3));
}

#[test]
fn chunked_slabs_oversized_segment_gets_dedicated_chunk() {
    let calls = Arc::new(AtomicUsize::new(0));
    let allocate_fn = test_allocate_fn(Arc::clone(&calls));
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(4096));

    slabs
        .alloc_segment(NumaNode(0), 4096, "K")
        .expect("oversized segment");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(slabs.chunk_count, 1);
}

#[test]
fn chunked_slabs_allocation_failure_is_an_error() {
    let allocate_fn: AllocateFn = Arc::new(|_, _| None);
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024, remaining(512));

    let err = match slabs.alloc_segment(NumaNode(0), 512, "K") {
        Ok(_) => panic!("allocation should fail"),
        Err(err) => err,
    };
    assert!(err.contains("failed to allocate fetch chunk"));
}

#[test]
fn chunked_slabs_chunk_clamped_to_batch_remaining() {
    // A small fetch must not request the whole chunk_bytes cap — that
    // fails outright on pools smaller than the cap (jz p2p IT regression).
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&sizes);
    let inner = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    let allocate_fn: AllocateFn = Arc::new(move |size, numa| {
        recorded.lock().unwrap().push(size);
        inner(size, numa)
    });
    let mut slabs = ChunkedSlabs::new(&allocate_fn, 256 << 20, remaining(4096));

    slabs.alloc_segment(NumaNode(0), 1024, "K").expect("first");
    slabs.alloc_segment(NumaNode(0), 3072, "V").expect("second");

    // One chunk sized to the batch total, not to the 256 MiB cap.
    assert_eq!(*sizes.lock().unwrap(), vec![4096]);
    assert_eq!(slabs.chunk_count, 1);
}

#[test]
fn transfer_shape_separates_logical_encoded_bytes_from_padded_payload() {
    use crate::codec::EncodedSegment;
    use orbitkv_state::StorageFormat;

    let allocate = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    let allocation = allocate(1536, Some(NumaNode(0))).unwrap();
    let address = allocation.as_non_null().as_ptr() as u64;
    let mut blocks = vec![(
        vec![1],
        vec![(
            vec![SegmentAlloc {
                ptr_addr: address,
                alloc: Arc::clone(&allocation),
                size: 512,
            }],
            NumaNode(0),
            Some(vec![EncodedSegment {
                version: 1,
                format: StorageFormat::Ans,
                logical_bytes: 2048,
                stored_bytes: 123,
                checksum: 0,
            }]),
        )],
    )];
    assert_eq!(transfer_shape(&blocks), (Some(2048), Representation::Ans));

    blocks[0].1.push((
        vec![SegmentAlloc {
            ptr_addr: address + 512,
            alloc: allocation,
            size: 1024,
        }],
        NumaNode(0),
        None,
    ));
    assert_eq!(transfer_shape(&blocks), (None, Representation::Mixed));
}

#[test]
fn raw_payload_alignment_does_not_imply_logical_bytes() {
    let allocate = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    // A raw 300-byte state can occupy a 512-byte aligned segment. A payload
    // length alone cannot prove the engine's original logical length.
    for stored_bytes in [300, 512] {
        let allocation = allocate(512, Some(NumaNode(0))).unwrap();
        let blocks = vec![(
            vec![1],
            vec![(
                vec![SegmentAlloc {
                    ptr_addr: allocation.as_non_null().as_ptr() as u64,
                    alloc: allocation,
                    size: stored_bytes,
                }],
                NumaNode(0),
                None,
            )],
        )];
        assert_eq!(transfer_shape(&blocks), (None, Representation::Raw));
    }
}

#[test]
fn fetch_slabs_follow_receiver_slots_instead_of_source_numa() {
    use orbitkv_proto::proto::engine::TransferSlotInfo;

    let blocks = vec![TransferBlockInfo {
        block_hash: vec![1],
        slots: vec![
            TransferSlotInfo {
                numa_node: 0,
                k_ptr: 0x1000,
                k_size: 512,
                v_ptr: 0x2000,
                v_size: 1024,
                ..Default::default()
            },
            TransferSlotInfo {
                numa_node: 99,
                k_ptr: 0x3000,
                k_size: 2048,
                ..Default::default()
            },
        ],
    }];
    let nodes = [NumaNode(1), NumaNode(3)];
    let bytes = sum_segment_bytes_by_numa(&blocks, &nodes).unwrap();
    assert_eq!(bytes, HashMap::from([(nodes[0], 1536), (nodes[1], 2048)]));
    let allocations = Arc::new(Mutex::new(Vec::new()));
    let recorded = allocations.clone();
    let inner = test_allocate_fn(Arc::new(AtomicUsize::new(0)));
    let allocate: AllocateFn = Arc::new(move |size, node| {
        assert!(node.is_some_and(|node| nodes.contains(&node)));
        recorded.lock().unwrap().push((size, node));
        inner(size, node)
    });
    let mut slabs = ChunkedSlabs::new(&allocate, FETCH_CHUNK_BYTES, bytes);
    let (key, key_owner) = slabs.alloc_segment(nodes[0], 512, "K").unwrap();
    let (value, value_owner) = slabs.alloc_segment(nodes[0], 1024, "V").unwrap();
    let (_, other_owner) = slabs.alloc_segment(nodes[1], 2048, "K").unwrap();
    assert!(Arc::ptr_eq(&key_owner, &value_owner));
    assert!(!Arc::ptr_eq(&key_owner, &other_owner));
    assert_eq!(value.as_ptr() as usize - key.as_ptr() as usize, 512);
    assert_eq!(
        *allocations.lock().unwrap(),
        vec![(1536, Some(nodes[0])), (2048, Some(nodes[1]))]
    );

    for invalid in [
        &nodes[..0],
        &nodes[..1],
        &[nodes[0], nodes[1], nodes[0]][..],
    ] {
        assert!(
            sum_segment_bytes_by_numa(&blocks, invalid)
                .unwrap_err()
                .contains("receiver registered")
        );
    }
    let mut malformed = blocks;
    malformed[0].slots[0].k_size = u64::MAX;
    assert!(
        sum_segment_bytes_by_numa(&malformed, &nodes)
            .unwrap_err()
            .contains("overflow")
    );
}
