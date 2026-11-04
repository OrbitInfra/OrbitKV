use super::*;

#[test]
fn submission_rejects_ranges_outside_registered_staging() {
    let batch = |file_offset, bytes, copy| IoBatch {
        file_offset,
        bytes,
        copies: vec![copy],
    };
    let copy = CopyRange {
        file_offset: 4096,
        device: 0x10000,
        bytes: 512,
    };
    batch(4096, 4096, copy).validate().unwrap();
    for invalid in [
        batch(4096, STAGING_BYTES + ALIGNMENT, copy),
        batch(4097, 4096, copy),
        batch(4096, 4095, copy),
        batch(4096, 0, copy),
        batch(
            4096,
            4096,
            CopyRange {
                file_offset: 4095,
                ..copy
            },
        ),
        batch(
            4096,
            4096,
            CopyRange {
                bytes: 4097,
                ..copy
            },
        ),
        batch(4096, 4096, CopyRange { device: 0, ..copy }),
        batch(
            4096,
            4096,
            CopyRange {
                device: u64::MAX,
                ..copy
            },
        ),
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn writes_cover_the_padded_extent_without_reading_gpu_padding() {
    let high_address = CopyRange {
        file_offset: 4096,
        device: u64::MAX - 8,
        bytes: 8,
    };
    assert_eq!(
        plan_writes(4096, 4096, vec![high_address]).unwrap()[0].copies[0],
        high_address
    );
    let start = ALIGNMENT as u64;
    let source = CopyRange {
        file_offset: start + 512,
        device: 0x10000,
        bytes: STAGING_BYTES + 137,
    };
    let length = (source.bytes + 512).next_multiple_of(ALIGNMENT) as u64;
    let batches = plan_writes(start, length, vec![source]).unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches.iter().map(|b| b.bytes as u64).sum::<u64>(), length);
    let mut consumed = 0;
    for batch in batches {
        assert!(batch.bytes <= STAGING_BYTES);
        assert_eq!(batch.bytes % ALIGNMENT, 0);
        for copy in batch.copies {
            assert_eq!(copy.device, source.device + consumed);
            assert_eq!(copy.file_offset, source.file_offset + consumed);
            consumed += copy.bytes as u64;
        }
    }
    assert_eq!(consumed, source.bytes as u64);
    for (offset, size, copy) in [
        (start + 1, length, source),
        (start, length - 1, source),
        (start, 0, source),
        (
            start,
            length,
            CopyRange {
                device: u64::MAX,
                ..source
            },
        ),
        (
            start,
            length,
            CopyRange {
                file_offset: 0,
                ..source
            },
        ),
        (
            start,
            length,
            CopyRange {
                bytes: length as usize,
                ..source
            },
        ),
        (u64::MAX - 4095, length, source),
    ] {
        assert!(plan_writes(offset, size, vec![copy]).is_err());
    }
}

#[test]
fn aligned_batches_merge_neighbors_but_do_not_read_unrequested_gaps() {
    let copies = vec![
        CopyRange {
            file_offset: 512,
            device: 0x10000,
            bytes: 700,
        },
        CopyRange {
            file_offset: 4096,
            device: 0x20000,
            bytes: 4096,
        },
        CopyRange {
            file_offset: 16384,
            device: 0x30000,
            bytes: 1,
        },
    ];
    let batches = plan_reads(copies.clone()).unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!((batches[0].file_offset, batches[0].bytes), (0, 8192));
    assert_eq!((batches[1].file_offset, batches[1].bytes), (16384, 4096));
    assert_eq!(
        batches
            .into_iter()
            .flat_map(|batch| batch.copies)
            .collect::<Vec<_>>(),
        copies
    );
}

#[test]
fn checkpoints_larger_than_staging_split_without_losing_or_padding_gpu_bytes() {
    let copy = CopyRange {
        file_offset: 512,
        device: 0x10000,
        bytes: 3 * STAGING_BYTES + 137,
    };
    let batches = plan_reads(vec![copy]).unwrap();
    assert_eq!(batches.len(), 4);
    let mut consumed = 0;
    for batch in batches {
        assert!(batch.bytes <= STAGING_BYTES);
        assert_eq!(batch.bytes % ALIGNMENT, 0);
        assert_eq!(batch.file_offset % ALIGNMENT as u64, 0);
        for piece in batch.copies {
            assert_eq!(piece.device, copy.device + consumed as u64);
            assert_eq!(piece.file_offset, copy.file_offset + consumed as u64);
            assert!(
                piece.file_offset + piece.bytes as u64 <= batch.file_offset + batch.bytes as u64
            );
            consumed += piece.bytes;
        }
    }
    assert_eq!(consumed, copy.bytes);
}

#[test]
fn duplicate_destinations_and_overflow_are_handled_before_submission() {
    let original = CopyRange {
        file_offset: 512,
        device: 0x10000,
        bytes: STAGING_BYTES * 2,
    };
    let duplicate = CopyRange {
        device: 0x5000000,
        bytes: 100,
        ..original
    };
    let batches = plan_reads(vec![original, duplicate]).unwrap();
    assert_eq!(
        batches
            .iter()
            .flat_map(|batch| &batch.copies)
            .map(|copy| copy.bytes)
            .sum::<usize>(),
        original.bytes + 100
    );
    assert!(
        plan_reads(vec![CopyRange {
            file_offset: u64::MAX,
            ..duplicate
        }])
        .is_err()
    );
    assert!(
        plan_reads(vec![CopyRange {
            device: u64::MAX,
            ..duplicate
        }])
        .is_err()
    );
}
