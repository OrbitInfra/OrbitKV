use super::*;

#[test]
fn numa_largest_free_for_node_is_not_global_min() {
    let mut pools = HashMap::new();
    pools.insert(
        0,
        ShardedPinnedPool::new(4096, 1, false, NonZeroU64::new(512), NumaNode::UNKNOWN).unwrap(),
    );
    pools.insert(
        1,
        ShardedPinnedPool::new(4096, 1, false, NonZeroU64::new(512), NumaNode::UNKNOWN).unwrap(),
    );
    let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

    let pinned = match &allocator {
        PinnedAllocator::Numa(pools) => pools.pools.get(&0).unwrap(),
        _ => unreachable!(),
    };
    let _held = pinned.allocate(NonZeroU64::new(3584).unwrap()).unwrap();

    let global_min = match &allocator {
        PinnedAllocator::Numa(pools) => pools
            .pools
            .values()
            .map(|p| p.largest_free_allocation())
            .min()
            .unwrap_or(0),
        _ => unreachable!(),
    };
    let node1_largest = allocator.largest_free_allocation_for_node(NumaNode(1));

    assert_eq!(global_min, 512);
    assert_eq!(node1_largest, 4096);
}

#[test]
fn numa_largest_free_for_unknown_node_is_zero() {
    let mut pools = HashMap::new();
    pools.insert(
        0,
        ShardedPinnedPool::new(4096, 1, false, NonZeroU64::new(512), NumaNode::UNKNOWN).unwrap(),
    );
    let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

    assert_eq!(
        allocator.largest_free_allocation_for_node(NumaNode::UNKNOWN),
        0
    );
}

#[test]
fn payload_ranges_keep_arena_identity_and_change_generation_on_offset_reuse() {
    let pool = PinnedAllocator::new_global(512, 1, false, None).unwrap();
    let first = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let range = first.source_range(first.as_non_null(), 512).unwrap();
    assert_eq!(range.offset, range.allocation_offset);
    assert_eq!(range.allocation_size, 512);
    assert!(first.source_range(first.as_non_null(), 513).is_err());
    assert!(first.source_range(first.as_non_null(), 0).is_err());
    let before = NonNull::new(first.as_non_null().as_ptr().wrapping_sub(1)).unwrap();
    assert!(first.source_range(before, 1).is_err());
    let exports = pool.payload_arenas().unwrap();
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].id, range.arena_id);
    assert_eq!(exports[0].size, 512);
    drop(first);
    let second = pool
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let next = second.source_range(second.as_non_null(), 512).unwrap();
    assert_eq!(next.arena_id, range.arena_id);
    assert_eq!(next.offset, range.offset);
    assert!(next.allocation_id > range.allocation_id);
}
