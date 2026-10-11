use super::*;

fn row(device: u64, host: usize, size: usize) -> CopyDesc {
    CopyDesc {
        device,
        host: host as *mut u8,
        host_device: host as u64,
        size,
        device_allocation: 1,
        host_registration: 2,
    }
}

fn expanded(copies: &[CopyDesc], max_pitch: usize) -> Vec<(u64, usize)> {
    dma_copies(copies, max_pitch)
        .flat_map(|copy| {
            (0..copy.rows).flat_map(move |row| {
                (0..copy.width).map(move |byte| {
                    (
                        copy.device + (row * copy.device_pitch + byte) as u64,
                        copy.host as usize + row * copy.host_pitch + byte,
                    )
                })
            })
        })
        .collect()
}

#[test]
fn explicit_strided_rows_preserve_gaps_and_submission_order() {
    let copies = [row(100, 200, 4), row(108, 212, 4), row(116, 224, 4)];
    let compiled: Vec<_> = dma_copies(&copies, 12).collect();
    assert_eq!(compiled.len(), 1);
    let copy = &compiled[0];
    assert_eq!(
        (copy.width, copy.rows, copy.device_pitch, copy.host_pitch),
        (4, 3, 8, 12)
    );
    assert_eq!(dma_copies(&copies, 11).count(), 3);
    let mut missing = copies.to_vec();
    missing.remove(1);
    let compiled: Vec<_> = dma_copies(&missing, 24).collect();
    assert_eq!((compiled[0].rows, compiled[0].device_pitch), (2, 16));
    assert_eq!(expanded(&missing, 24).len(), 8);
}

#[test]
fn incompatible_rows_cannot_share_a_submission() {
    let first = row(100, 200, 4);
    for second in [
        row(108, 212, 3),
        row(102, 212, 4),
        row(108, 202, 4),
        row(96, 212, 4),
        row(108, 196, 4),
        CopyDesc {
            device_allocation: 9,
            ..row(108, 212, 4)
        },
        CopyDesc {
            host_registration: 9,
            ..row(108, 212, 4)
        },
    ] {
        assert_eq!(dma_copies(&[first, second], 100).count(), 2);
    }
    assert_eq!(dma_copies(&[], 100).count(), 0);
    let uneven = [first, row(108, 212, 4), row(124, 224, 4)];
    assert_eq!(
        dma_copies(&uneven, 100).map(|c| c.rows).collect::<Vec<_>>(),
        vec![2, 1]
    );
}

#[test]
fn contiguous_rows_coalesce_without_a_pitch_limit() {
    let copies = [row(100, 200, 4), row(104, 204, 6), row(110, 210, 3)];
    let compiled: Vec<_> = dma_copies(&copies, 0).collect();
    assert_eq!(compiled.len(), 1);
    assert_eq!((compiled[0].width, compiled[0].rows), (13, 1));
}

#[test]
fn overflowing_addresses_are_never_merged() {
    for copies in [
        [row(u64::MAX - 3, 200, 4), row(0, 204, 4)],
        [row(100, usize::MAX - 3, 4), row(104, 0, 4)],
        [row(u64::MAX - 11, 200, 4), row(u64::MAX - 3, 212, 4)],
        [row(100, usize::MAX - 15, 4), row(108, usize::MAX - 3, 4)],
    ] {
        assert_eq!(dma_copies(&copies, usize::MAX).count(), 2);
    }
}

#[test]
fn compilation_preserves_every_requested_byte_for_irregular_layouts() {
    let mut seed = 0x71e9_9aad_u64;
    for _ in 0..2000 {
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut copies = Vec::new();
        let mut device = 1000;
        let mut host = 2000;
        for _ in 0..64 {
            let size = 1 + (random() % 16) as usize;
            let mut copy = row(device, host, size);
            copy.device_allocation = (random() % 2) as usize;
            copy.host_registration = (random() % 2) as usize;
            copies.push(copy);
            device += random() % 24;
            host += (random() % 24) as usize;
        }
        let expected: Vec<_> = copies
            .iter()
            .flat_map(|copy| {
                (0..copy.size)
                    .map(move |byte| (copy.device + byte as u64, copy.host as usize + byte))
            })
            .collect();
        assert_eq!(expanded(&copies, 16), expected);
    }
}
