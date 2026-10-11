use orbitkv_core::transfer::CopyDesc;

#[path = "../native_copy_layout/geometry.rs"]
mod geometry;
#[path = "../../crates/orbitkv-core/src/transfer/memcpy/ranges.rs"]
mod ranges;

#[test]
fn representative_shapes_match_all_observed_descriptor_and_range_pairs() {
    let layouts = geometry::layouts();
    let expected = [
        (64, 64),
        (192, 64),
        (256, 192),
        (448, 192),
        (448, 448),
        (512, 192),
        (512, 512),
        (640, 320),
        (256, 256),
    ];
    let mut model_bytes = 0;
    let mut model_descriptors = 0;
    let mut model_ranges = 0;
    for (layout, (descriptors, dma_ranges)) in layouts.iter().zip(expected) {
        let copies = layout.virtual_copies().unwrap();
        assert_eq!(copies.len(), descriptors, "{}", layout.name);
        assert_eq!(
            ranges::dma_copies(&copies, usize::MAX).count(),
            dma_ranges,
            "{}",
            layout.name
        );
        assert_eq!(layout.expected_dma_ranges, dma_ranges);
        assert!(
            copies
                .iter()
                .all(|copy| copy.size == geometry::FRAGMENT_BYTES)
        );
        model_bytes += layout.observed_batches * copies.len() * geometry::FRAGMENT_BYTES;
        model_descriptors += layout.observed_batches * descriptors;
        model_ranges += layout.observed_batches * dma_ranges;
    }
    assert_eq!(layouts.len(), expected.len());
    assert_eq!(
        (model_bytes, model_descriptors, model_ranges),
        (310_378_496, 4736, 3648)
    );
}

#[test]
fn bad_page_permutations_and_physical_bindings_are_rejected() {
    let mut layout = geometry::layouts().remove(1);
    for bad in [vec![], vec![0, 0, 1], vec![0, 1, 3], vec![0; 11]] {
        layout.host_order = bad;
        assert!(layout.ranges().is_err());
    }
    let layout = geometry::layouts().remove(0);
    assert!(layout.bind(&[], &[]).is_err());
    assert!(
        layout
            .bind(
                &vec![1; geometry::LAYERS],
                &[(std::ptr::dangling_mut::<u8>(), 2)]
            )
            .is_err()
    );
    let devices: Vec<_> = (1..=geometry::LAYERS as u64)
        .map(|id| id * 0x1000_0000)
        .collect();
    assert!(
        layout
            .bind(&devices, &[(std::ptr::dangling_mut::<u8>(), 0)])
            .is_err()
    );
    assert!(
        layout
            .bind(&devices, &[(usize::MAX as *mut u8, 2)])
            .is_err()
    );
}

#[test]
fn registration_boundaries_and_pitch_limits_survive_real_compiler_binding() {
    let layouts = geometry::layouts();
    let mut copies = layouts[1].virtual_copies().unwrap();
    assert_eq!(
        ranges::dma_copies(&copies, usize::MAX).count(),
        geometry::LAYERS
    );
    assert_eq!(
        ranges::dma_copies(&copies, geometry::FRAGMENT_BYTES).count(),
        copies.len()
    );
    for (index, copy) in copies.iter_mut().enumerate() {
        copy.host_registration = index + 1;
    }
    assert_eq!(
        ranges::dma_copies(&copies, usize::MAX).count(),
        copies.len()
    );
    let split = layouts.last().unwrap();
    assert_eq!(split.host_lengths().len(), 4);
    let copies = split.virtual_copies().unwrap();
    assert_eq!(
        ranges::dma_copies(&copies, usize::MAX).count(),
        copies.len()
    );
}
