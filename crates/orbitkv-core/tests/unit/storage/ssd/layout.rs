use super::*;
use crate::transfer_engine::{FileIoElement, MAX_FILE_BYTES, MAX_FILE_ELEMENTS};

#[test]
fn extent_above_old_chunk_size_is_one_vector_with_padding() {
    let copy = CopyRange {
        file_offset: 4096 + 512,
        device: 0x10000,
        bytes: (12 << 20) + 137,
    };
    let bytes = (512 + copy.bytes).next_multiple_of(ALIGNMENT);
    for batches in [
        plan_reads(vec![copy]).unwrap(),
        plan_writes(4096, bytes as u64, vec![copy]).unwrap(),
    ] {
        assert_eq!(batches.len(), 1);
        let extent = batches.into_iter().next().unwrap().into_extent().unwrap();
        assert_eq!(extent.bytes().unwrap(), bytes);
        assert_eq!(
            extent.elements,
            vec![
                FileIoElement::Padding(512),
                FileIoElement::Range {
                    address: copy.device,
                    length: copy.bytes
                },
                FileIoElement::Padding(bytes - 512 - copy.bytes)
            ]
        );
    }
}

#[test]
fn gaps_are_padding_and_repeated_file_ranges_keep_both_destinations() {
    let copies = vec![
        CopyRange {
            file_offset: 512,
            device: 0x10000,
            bytes: 700,
        },
        CopyRange {
            file_offset: 16384,
            device: 0x20000,
            bytes: 137,
        },
    ];
    let extent = plan_reads(copies.clone())
        .unwrap()
        .remove(0)
        .into_extent()
        .unwrap();
    assert_eq!(extent.elements.len(), 5);
    assert_eq!(extent.elements[2], FileIoElement::Padding(16384 - 1212));
    let mut repeated = copies;
    repeated.push(CopyRange {
        file_offset: 512,
        device: 0x30000,
        bytes: 700,
    });
    let batches = plan_reads(repeated).unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches.iter().map(|b| b.copies.len()).sum::<usize>(), 3);
    for batch in batches {
        batch.into_extent().unwrap();
    }
}

#[test]
fn full_vector_preflight_rejects_overlap_overflow_and_budgets() {
    let copy = CopyRange {
        file_offset: 4096,
        device: 0x10000,
        bytes: 512,
    };
    for (offset, bytes, copies) in [
        (4097, 4096, vec![copy]),
        (4096, 4095, vec![copy]),
        (4096, 0, vec![copy]),
        (4096, 4096, vec![CopyRange { device: 0, ..copy }]),
        (
            4096,
            4096,
            vec![CopyRange {
                device: u64::MAX,
                ..copy
            }],
        ),
        (
            4096,
            4096,
            vec![CopyRange {
                bytes: 4097,
                ..copy
            }],
        ),
        (4096, 4096, vec![copy, copy]),
        (4096, MAX_FILE_BYTES as u64 + 4096, vec![copy]),
        (u64::MAX - 4095, 4096, vec![copy]),
    ] {
        assert!(plan_writes(offset, bytes, copies).is_err());
    }
    let copies = (0..MAX_FILE_ELEMENTS)
        .map(|i| CopyRange {
            file_offset: (i * 2) as u64,
            device: 0x10000 + i as u64,
            bytes: 1,
        })
        .collect();
    assert!(
        plan_writes(0, 4096, copies).is_err(),
        "padding counts against descriptor budget"
    );
}
