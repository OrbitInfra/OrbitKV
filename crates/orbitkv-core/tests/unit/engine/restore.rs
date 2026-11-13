use std::num::NonZeroU64;

use super::*;
use crate::block::{RawBlock, Segment};
use crate::memory::pool::PinnedAllocator;
use crate::transfer::layout::KVCacheGeometry;
use crate::transfer::local::RawRestorePart;
use crate::{NumaNode, QueryAdmission, QueryMode, QueryOwner};

#[test]
fn raw_grant_retains_allocation_and_query_charge_until_explicit_finish() {
    let pool = PinnedAllocator::new_global(512, 1, false, None).unwrap();
    let allocation = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let allocation_owner = Arc::downgrade(&allocation);
    let slot =
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), 512, allocation).unwrap());
    let block = Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)]));
    let source_owner = Arc::downgrade(&block);
    let budget = crate::query::QueryBudget::new(512, 512).unwrap();
    let QueryAdmission::Admitted(reservation) =
        budget.reserve("engine", "ns", 512, QueryMode::Demand)
    else {
        panic!("budget available")
    };
    reservation.ready(512).unwrap();
    let owner = QueryOwner {
        session: 1,
        operation: 1,
        revision: 1,
    };
    let leases = QueryLeaseManager::default();
    let lease = leases
        .create(
            "engine",
            vec![RestoreSource::Memory(block)],
            1,
            Some((owner, reservation)),
        )
        .unwrap();
    let groups = vec![RestoreGroup {
        layers: vec![RestoreLayer {
            name: "layer_99".into(),
            slot_id: 0,
            host_offset: 64,
        }],
        storage_slots: Some((0, 1)),
        targets: vec![],
    }];
    let layout = KVCacheLayout::bind(
        0x1000,
        512,
        KVCacheGeometry::new(4, 32, 256, 2, None, 1).unwrap(),
    )
    .unwrap();
    let prepared = PreparedRestore::prepare(
        &leases,
        "engine",
        0,
        groups,
        &[(lease, vec![vec![Some(2)]])],
        &[layout],
    )
    .unwrap();
    let (encoded, bytes, fragments) = prepared.raw.unwrap();
    let plan = RawRestorePart::decode(&encoded[0]).unwrap();
    assert_eq!(fragments, plan.copies.len());
    assert_eq!(plan.copies.len(), 2);
    assert_eq!(plan.copies[0].layer, "layer_99");
    assert_eq!(plan.copies[0].destination_offset, 64);
    assert_eq!(plan.copies[1].destination_offset, 320);
    assert_eq!(plan.copies[0].source.offset, 64);
    assert_eq!(plan.copies[1].source.offset, 96);
    assert_eq!(
        plan.copies[0].source.allocation_id,
        plan.copies[1].source.allocation_id
    );
    assert_eq!(bytes, 64);
    let sources = prepared
        .sources
        .into_iter()
        .map(|source| match source {
            RestoreSource::Memory(source) => source,
            _ => unreachable!(),
        })
        .collect();
    let plans = plan
        .copies
        .into_iter()
        .flat_map(|copy| {
            RawRestorePlan { copies: vec![copy] }
                .encode_parts()
                .unwrap()
        })
        .collect();
    let mut grant = RawRestoreGrant {
        plans,
        sources,
        reservations: prepared.reservations,
        bytes,
        decode_admission: None,
    };
    leases
        .release_owner(|candidate| candidate.session == 1)
        .unwrap();
    assert!(source_owner.upgrade().is_some());
    assert!(allocation_owner.upgrade().is_some());
    assert!(
        pool.allocate(NonZeroU64::new(1).unwrap(), NumaNode::UNKNOWN)
            .is_none()
    );
    assert!(matches!(
        budget.reserve("engine", "ns", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    assert!(grant.encoded_plan().1);
    assert!(grant.advance_plan());
    assert!(!grant.encoded_plan().1);
    assert!(!grant.advance_plan());
    assert!(source_owner.upgrade().is_some());
    assert!(allocation_owner.upgrade().is_some());
    assert!(matches!(
        budget.reserve("engine", "ns", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    grant.finish(true, None);
    assert!(source_owner.upgrade().is_none());
    assert!(allocation_owner.upgrade().is_none());
    assert!(
        pool.allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
            .is_some()
    );
    assert!(matches!(
        budget.reserve("engine", "ns", 512, QueryMode::Demand),
        QueryAdmission::Admitted(_)
    ));
}

#[test]
fn raw_plan_rejection_preserves_lease_before_source_bounds_or_plan_limit_failure() {
    let pool = PinnedAllocator::new_global(512, 1, false, None).unwrap();
    let allocation = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let slot =
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), 16, allocation).unwrap());
    let block = Arc::new(SealedBlock::from_slots(vec![(slot, NumaNode::UNKNOWN)]));
    let leases = QueryLeaseManager::default();
    let lease = leases
        .create("engine", vec![RestoreSource::Memory(block)], 1, None)
        .unwrap();
    let layout = KVCacheLayout::bind(
        0x1000,
        32,
        KVCacheGeometry::new(1, 32, 0, 1, None, 1).unwrap(),
    )
    .unwrap();
    for name in ["range".to_string(), "x".repeat(u16::MAX as usize + 1)] {
        let groups = vec![RestoreGroup {
            layers: vec![RestoreLayer {
                name,
                slot_id: 0,
                host_offset: 0,
            }],
            storage_slots: Some((0, 1)),
            targets: vec![],
        }];
        assert!(
            PreparedRestore::prepare(
                &leases,
                "engine",
                0,
                groups,
                &[(lease, vec![vec![Some(0)]])],
                std::slice::from_ref(&layout)
            )
            .is_err()
        );
    }
    assert!(
        leases.release(&lease).unwrap(),
        "rejected plan must leave its lease available"
    );
}

#[test]
fn raw_plan_orders_permuted_pages_and_compacts_only_matching_allocations() {
    for split in [false, true] {
        for sparse in [false, true] {
            let pool = PinnedAllocator::new_global(1024, 1, false, None).unwrap();
            let allocations: Vec<_> = (0..if split { 2 } else { 1 })
                .map(|_| {
                    pool.allocate(NonZeroU64::new(128).unwrap(), NumaNode::UNKNOWN)
                        .unwrap()
                })
                .collect();
            let order = [2, 0, 3, 1];
            let sources: Vec<_> = order
                .iter()
                .map(|&block| {
                    let segments = allocations
                        .iter()
                        .map(|allocation| {
                            // SAFETY: four 32-byte pages are inside each retained allocation.
                            let pointer = unsafe { allocation.as_non_null().add(block * 32) };
                            Segment::new(pointer, 32, Arc::clone(allocation)).unwrap()
                        })
                        .collect();
                    RestoreSource::Memory(Arc::new(SealedBlock::from_slots(vec![(
                        RawBlock::new(segments),
                        NumaNode::UNKNOWN,
                    )])))
                })
                .collect();
            let refs: Vec<_> = sources.iter().collect();
            let mut groups = vec![RestoreGroup {
                layers: vec![RestoreLayer {
                    name: "layer".into(),
                    slot_id: 0,
                    host_offset: 0,
                }],
                storage_slots: Some((0, 1)),
                targets: order
                    .iter()
                    .enumerate()
                    .map(|(source, &block)| (block * if sparse { 2 } else { 1 }, source))
                    .collect(),
            }];
            let layout = KVCacheLayout::bind(
                0x1000,
                512,
                KVCacheGeometry::new(
                    8,
                    32,
                    if split { 256 } else { 0 },
                    if split { 2 } else { 1 },
                    None,
                    1,
                )
                .unwrap(),
            )
            .unwrap();
            let (encoded, bytes, fragments) =
                PreparedRestore::raw_plan(&mut groups, &refs, &[layout])
                    .unwrap()
                    .unwrap();
            let plan = RawRestorePart::decode(&encoded[0]).unwrap();
            assert_eq!(fragments, plan.copies.len());
            let segments = if split { 2 } else { 1 };
            assert_eq!(bytes, segments * 128);
            assert_eq!(
                plan.copies.len(),
                segments as usize * if sparse { 4 } else { 1 }
            );
            for (segment, allocation) in allocations.iter().enumerate() {
                let base = RawBlock::single_segment(
                    Segment::new(allocation.as_non_null(), 128, Arc::clone(allocation)).unwrap(),
                )
                .source_range(0, 0, 128)
                .unwrap();
                let copies = &plan.copies[segment * if sparse { 4 } else { 1 }..]
                    [..if sparse { 4 } else { 1 }];
                for (page, copy) in copies.iter().enumerate() {
                    assert_eq!(copy.source.allocation_id, base.allocation_id);
                    assert_eq!(copy.source.offset, base.offset + page as u64 * 32);
                    assert_eq!(copy.source.size, if sparse { 32 } else { 128 });
                    assert_eq!(
                        copy.destination_offset,
                        segment as u64 * 256 + page as u64 * 64
                    );
                }
            }
        }
    }

    let pool = PinnedAllocator::new_global(1024, 1, false, None).unwrap();
    let sources: Vec<_> = (0..2)
        .map(|_| {
            let allocation = pool
                .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
                .unwrap();
            RestoreSource::Memory(Arc::new(SealedBlock::from_slots(vec![(
                RawBlock::single_segment(
                    Segment::new(allocation.as_non_null(), 512, allocation).unwrap(),
                ),
                NumaNode::UNKNOWN,
            )])))
        })
        .collect();
    let refs: Vec<_> = sources.iter().collect();
    let mut groups = vec![RestoreGroup {
        layers: vec![RestoreLayer {
            name: "layer".into(),
            slot_id: 0,
            host_offset: 0,
        }],
        storage_slots: Some((0, 1)),
        targets: vec![(0, 0), (1, 1)],
    }];
    let layout = KVCacheLayout::bind(
        0x1000,
        1024,
        KVCacheGeometry::new(2, 512, 0, 1, None, 1).unwrap(),
    )
    .unwrap();
    let (encoded, _, fragments) =
        PreparedRestore::raw_plan(&mut groups, &refs, std::slice::from_ref(&layout))
            .unwrap()
            .unwrap();
    let plan = RawRestorePart::decode(&encoded[0]).unwrap();
    assert_eq!(fragments, plan.copies.len());
    assert_eq!(plan.copies.len(), 2);
    assert_eq!(
        plan.copies[0].source.offset + 512,
        plan.copies[1].source.offset
    );
    assert_ne!(
        plan.copies[0].source.allocation_id,
        plan.copies[1].source.allocation_id
    );
    groups[0].targets[1].0 = 0;
    assert!(
        PreparedRestore::raw_plan(&mut groups, &refs, &[layout])
            .unwrap_err()
            .to_string()
            .contains("overlap")
    );
}

#[test]
fn large_plan_compacts_or_partitions_after_global_validation_and_preserves_rejected_leases() {
    const COUNT: usize = 32768;
    for (contiguous, layer_name, rejection) in [
        (true, "layer".into(), None),
        (false, "layer".into(), None),
        (false, "layer".into(), Some("overlap")),
        (false, "x".repeat(1024), Some("metadata limit")),
    ] {
        let size = if contiguous { COUNT * 32 } else { 32 };
        let pool = PinnedAllocator::new_global(size.max(512), 1, false, None).unwrap();
        let allocation = pool
            .allocate(NonZeroU64::new(size as u64).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        let sources = (0..COUNT)
            .map(|page| {
                // SAFETY: each selected page lies inside this retained allocation.
                let pointer = unsafe {
                    allocation
                        .as_non_null()
                        .add(if contiguous { page * 32 } else { 0 })
                };
                RestoreSource::Memory(Arc::new(SealedBlock::from_slots(vec![(
                    RawBlock::single_segment(
                        Segment::new(pointer, 32, Arc::clone(&allocation)).unwrap(),
                    ),
                    NumaNode::UNKNOWN,
                )])))
            })
            .collect();
        let leases = QueryLeaseManager::default();
        let lease = leases.create("engine", sources, 1, None).unwrap();
        let layout = KVCacheLayout::bind(
            0x1000,
            COUNT * 32,
            KVCacheGeometry::new(COUNT, 32, 0, 1, None, 1).unwrap(),
        )
        .unwrap();
        let groups = vec![RestoreGroup {
            layers: vec![RestoreLayer {
                name: layer_name,
                slot_id: 0,
                host_offset: 0,
            }],
            storage_slots: Some((0, 1)),
            targets: vec![],
        }];
        let mut targets: Vec<_> = (0..COUNT).map(Some).collect();
        if rejection == Some("overlap") {
            targets[COUNT - 1] = Some(0);
        }
        let prepared = PreparedRestore::prepare(
            &leases,
            "engine",
            0,
            groups,
            &[(lease, vec![targets])],
            &[layout],
        );
        if let Some(expected) = rejection {
            assert!(prepared.err().unwrap().to_string().contains(expected));
            assert!(
                leases.release(&lease).unwrap(),
                "rejected plan must preserve the lease"
            );
            continue;
        }
        let (encoded, bytes, fragments) = prepared.unwrap().raw.unwrap();
        assert!(
            encoded
                .iter()
                .all(|part| part.len() <= crate::transfer::local::MAX_PLAN_BYTES)
        );
        let copies: Vec<_> = encoded
            .iter()
            .flat_map(|part| RawRestorePart::decode(part).unwrap().copies)
            .collect();
        assert_eq!(fragments, copies.len());
        assert_eq!(bytes, (COUNT * 32) as u64);
        assert!(
            !leases.release(&lease).unwrap(),
            "accepted parts share one consumed lease"
        );
        if contiguous {
            assert_eq!(encoded.len(), 1);
            assert_eq!(copies.len(), 1);
            assert_eq!(copies[0].source.size, (COUNT * 32) as u64);
            assert_eq!(encoded[0].len(), 72);
        } else {
            assert!(encoded.len() > 1);
            assert_eq!(copies.len(), COUNT);
            for (index, copy) in copies.iter().enumerate() {
                assert_eq!(copy.destination_offset, (index * 32) as u64);
                assert_eq!(copy.source.size, 32);
            }
        }
    }
}
