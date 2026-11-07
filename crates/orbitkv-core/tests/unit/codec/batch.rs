use super::*;
use gpu::{DecodeError, DecodeInput, EncodeInput, GpuCodec, HostDecodeInput, MAX_BATCH_SEGMENTS};

#[test]
fn validate_targets_covers_batch_boundaries_and_cpu_fallback_ranges() {
    let mut ranges: Vec<_> = (0..=MAX_BATCH_SEGMENTS)
        .map(|i| (0x1000 + i as u64 * 4096, 128usize))
        .collect();
    // The request-wide validator has no batch limit and accepts unsorted ranges.
    gpu::validate_targets(ranges.iter().copied().rev()).unwrap();
    ranges[MAX_BATCH_SEGMENTS] = ranges[0];
    assert!(
        gpu::validate_targets(ranges.iter().copied())
            .unwrap_err()
            .contains("target ranges overlap")
    );
    ranges[MAX_BATCH_SEGMENTS] = (ranges[0].0 + 127, 2);
    assert!(gpu::validate_targets(ranges).is_err());

    // Tiny CPU fallbacks and raw Exact siblings larger than codec limits obey
    // the same destination contract, with adjacency permitted.
    assert_eq!(
        gpu::validate_targets([(0x1004, 4), (0x1000, 4)]).unwrap(),
        vec![(0x1000, 0x1004), (0x1004, 0x1008)]
    );
    assert!(gpu::validate_targets([(0x1000, 4), (0x1002, 4)]).is_err());
    let large = 17 * 1024 * 1024;
    gpu::validate_targets([(0x1000, large), (0x1000 + large as u64, 2)]).unwrap();
    assert!(gpu::validate_targets([(0x1000, large), (0x1002, 2)]).is_err());
}

#[test]
fn validate_targets_checks_nonzero_ranges_and_address_overflow() {
    gpu::validate_targets([]).unwrap();
    gpu::validate_targets([(u64::MAX - 7, 7)]).unwrap();
    for (case, pointer, bytes) in [
        ("null target", 0, 1),
        ("empty target", 0x1000, 0),
        ("excessive length", 1, usize::MAX),
        ("address overflow", u64::MAX - 7, 8),
    ] {
        assert!(gpu::validate_targets([(pointer, bytes)]).is_err(), "{case}");
    }
}

#[test]
fn metadata_bounds_allow_large_exact_siblings_without_weakening_codecs() {
    let large = 17 * 1024 * 1024;
    let mut meta = EncodedSegment {
        version: 1,
        format: StorageFormat::Exact,
        logical_bytes: large,
        stored_bytes: large,
        checksum: 0,
    };
    assert!(meta.validate_metadata(large).is_ok());
    assert!(meta.validate_metadata(large - 1).is_err());
    meta.logical_bytes -= 1;
    assert!(meta.validate_metadata(large).is_err());
    meta.logical_bytes = large;
    for format in [
        StorageFormat::Ans,
        StorageFormat::Ans16,
        StorageFormat::AnsFp8,
        StorageFormat::Fp8FromBf16,
    ] {
        meta.format = format;
        assert!(meta.validate_metadata(large).is_err(), "{format:?}");
    }
    meta.format = StorageFormat::Exact;
    meta.logical_bytes = usize::MAX;
    meta.stored_bytes = usize::MAX;
    assert!(meta.validate_metadata(usize::MAX).is_err());
    meta.format = StorageFormat::Ans16;
    meta.logical_bytes = 4097;
    meta.stored_bytes = 64;
    assert!(meta.validate_metadata(64).is_err());
}

// Snapshot an arena while its borrow is held. The retained device copy models
// a cuFile read buffer for restore; the host copy is only a test oracle.
fn snapshot(
    stream: &Arc<CudaStream>,
    output: &gpu::EncodedDeviceSegment,
) -> (CudaSlice<u8>, Vec<u8>, EncodedSegment) {
    let device = stream.alloc_zeros::<u8>(output.meta.stored_bytes).unwrap();
    // SAFETY: the borrowed arena output and distinct destination CudaSlice
    // remain live until the explicit stream synchronization below.
    let copied = unsafe {
        result::memcpy_dtod_async(
            device.device_ptr(stream).0,
            output.device,
            output.meta.stored_bytes,
            stream.cu_stream(),
        )
    };
    stream.synchronize().unwrap();
    copied.unwrap();
    let host = stream.clone_dtoh(&device).unwrap();
    output.meta.validate(&host).unwrap();
    assert!(output.device.is_multiple_of(4096));
    (device, host, output.meta.clone())
}

#[test]
#[ignore = "CUDA qualification: mixed batch dispatch, scratch reuse, GPU CRC failure isolation"]
fn gpu_mixed_batches_reuse_scratch_and_crc_failure_precedes_all_target_writes() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let formats = [
        StorageFormat::Fp8FromBf16,
        StorageFormat::TurboQuant {
            scalar: Scalar16::Bf16,
            role: AttentionRole::Key,
            head_dim: 32,
            bits: 3,
            seed: 41,
        },
        StorageFormat::Fp8FromFp16,
        StorageFormat::TurboQuant {
            scalar: Scalar16::Fp16,
            role: AttentionRole::Value,
            head_dim: 32,
            bits: 4,
            seed: 73,
        },
        StorageFormat::TurboQuant {
            scalar: Scalar16::Bf16,
            role: AttentionRole::PackedKeyValue,
            head_dim: 128,
            bits: 4,
            seed: 97,
        },
    ];
    let lengths = [8194, 2048, 16390, 4096, 8192];
    let hosts: Vec<Vec<u8>> = formats
        .iter()
        .zip(lengths)
        .enumerate()
        .map(|(j, (format, bytes))| {
            let bf = matches!(
                format,
                StorageFormat::Fp8FromBf16
                    | StorageFormat::TurboQuant {
                        scalar: Scalar16::Bf16,
                        ..
                    }
            );
            (0..bytes / 2)
                .flat_map(|i| {
                    let x = (i as f32 * 0.123 + j as f32).sin() * 1.7;
                    if bf {
                        bf16::from_f32(x).to_bits().to_le_bytes()
                    } else {
                        f16::from_f32(x).to_bits().to_le_bytes()
                    }
                })
                .collect()
        })
        .collect();
    let sources: Vec<_> = hosts
        .iter()
        .map(|h| stream.clone_htod(h).unwrap())
        .collect();
    let inputs: Vec<_> = sources
        .iter()
        .zip(formats)
        .map(|(src, format)| EncodeInput {
            source: src.device_ptr(&stream).0,
            bytes: src.len(),
            format,
        })
        .collect();
    let budget = 8 * 1024 * 1024;
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &inputs, budget) }.unwrap();
    assert_eq!(batch.processed, inputs.len());
    let pointers: Vec<_> = batch
        .outputs
        .iter()
        .map(|s| s.as_ref().unwrap().device)
        .collect();
    let saved: Vec<_> = batch
        .outputs
        .iter()
        .map(|s| snapshot(&stream, s.as_ref().unwrap()))
        .collect();
    drop(batch);
    let retained = codec.scratch_bytes();
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &inputs, budget) }.unwrap();
    assert_eq!(
        pointers,
        batch
            .outputs
            .iter()
            .map(|s| s.as_ref().unwrap().device)
            .collect::<Vec<_>>()
    );
    for (current, (_, host, _)) in batch.outputs.iter().zip(&saved) {
        assert_eq!(snapshot(&stream, current.as_ref().unwrap()).1, *host);
    }
    drop(batch);
    assert_eq!(codec.scratch_bytes(), retained);
    assert!(retained <= budget);

    let targets: Vec<_> = lengths
        .iter()
        .map(|&len| stream.clone_htod(&vec![0x5au8; len]).unwrap())
        .collect();
    let decode: Vec<_> = saved
        .iter()
        .zip(&targets)
        .map(|((source, _, meta), target)| DecodeInput {
            source: source.device_ptr(&stream).0,
            source_bytes: source.len(),
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta,
        })
        .collect();
    // Corrupt the last source: the earlier valid segments must not be decoded.
    let last = saved.last().unwrap();
    let original = last.1[last.2.stored_bytes - 1];
    let corrupt = [original ^ 1];
    let pointer = last.0.device_ptr(&stream).0 + (last.2.stored_bytes - 1) as u64;
    // SAFETY: the one-byte source and the final byte of the owned device
    // allocation remain live until the immediately following stream synchronization.
    unsafe { result::memcpy_htod_async(pointer, &corrupt, stream.cu_stream()) }.unwrap();
    stream.synchronize().unwrap();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    let error = unsafe { codec.decode_batch(&stream, &decode, budget) }.unwrap_err();
    assert!(
        matches!(&error, DecodeError::Corrupt(message) if message.contains("checksum mismatch")),
        "{error}"
    );
    for target in &targets {
        assert!(
            stream
                .clone_dtoh(target)
                .unwrap()
                .iter()
                .all(|&v| v == 0x5a)
        );
    }
    let repaired = [original];
    // SAFETY: the one-byte source and the final byte of the owned device
    // allocation remain live until the immediately following stream synchronization.
    unsafe { result::memcpy_htod_async(pointer, &repaired, stream.cu_stream()) }.unwrap();
    stream.synchronize().unwrap();
    // The failed batch has drained and the same scratch is immediately usable.
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    unsafe { codec.decode_batch(&stream, &decode, budget) }.unwrap();
    let device_results: Vec<_> = targets
        .iter()
        .map(|t| stream.clone_dtoh(t).unwrap())
        .collect();
    let host_inputs: Vec<_> = saved
        .iter()
        .zip(&targets)
        .map(|((_, host, meta), target)| HostDecodeInput {
            source: host,
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta,
        })
        .collect();
    assert_eq!(
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_host_batch(&stream, &host_inputs, retained, None) }.unwrap(),
        inputs.len()
    );
    for (i, target) in targets.iter().enumerate() {
        assert_eq!(stream.clone_dtoh(target).unwrap(), device_results[i]);
        if matches!(
            formats[i],
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
        ) {
            let mut expected = vec![0; lengths[i]];
            assert!(cpu::decode(
                formats[i],
                &saved[i].1[..saved[i].2.stored_bytes],
                &mut expected
            ));
            assert_eq!(device_results[i], expected);
        }
    }
    assert_eq!(codec.scratch_bytes(), retained);

    let undersized = DecodeInput {
        source_bytes: saved[0].2.stored_bytes - 1,
        ..decode[0]
    };
    assert!(matches!(
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_batch(&stream, &[undersized], budget) },
        Err(DecodeError::Runtime(_))
    ));
}

#[test]
#[ignore = "CUDA qualification: corrupt data and runtime budget failures remain distinct"]
fn gpu_decode_classifies_corruption_and_budget_without_target_writes() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let encoded = [0x30u8; 16];
    let meta = EncodedSegment {
        version: 1,
        format: StorageFormat::Fp8FromBf16,
        logical_bytes: 32,
        stored_bytes: encoded.len(),
        checksum: crc32fast::hash(&encoded),
    };
    let source = stream.clone_htod(&encoded).unwrap();
    let sentinel = vec![0x5au8; meta.logical_bytes];
    let target = stream.clone_htod(&sentinel).unwrap();
    let valid = DecodeInput {
        source: source.device_ptr(&stream).0,
        source_bytes: source.len(),
        target: target.device_ptr(&stream).0,
        target_bytes: target.len(),
        meta: &meta,
    };
    let error =
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_batch(&stream, std::slice::from_ref(&valid), 0) }.unwrap_err();
    assert!(
        matches!(&error, DecodeError::Runtime(message) if message.contains("budget")),
        "{error}"
    );
    assert_eq!(codec.scratch_bytes(), 0);
    assert_eq!(stream.clone_dtoh(&target).unwrap(), sentinel);

    let mut corrupt_meta = meta.clone();
    corrupt_meta.checksum ^= 1;
    let corrupt = DecodeInput {
        meta: &corrupt_meta,
        ..valid
    };
    let budget = 64 * 1024;
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    let error = unsafe { codec.decode_batch(&stream, &[corrupt], budget) }.unwrap_err();
    assert!(
        matches!(&error, DecodeError::Corrupt(message) if message.contains("checksum mismatch")),
        "{error}"
    );
    assert_eq!(stream.clone_dtoh(&target).unwrap(), sentinel);
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    unsafe { codec.decode_batch(&stream, &[valid], budget) }.unwrap();
    let mut expected = vec![0; meta.logical_bytes];
    assert!(cpu::decode(meta.format, &encoded, &mut expected));
    assert_eq!(stream.clone_dtoh(&target).unwrap(), expected);
}

#[test]
#[ignore = "CUDA qualification: prefix budgets and raw siblings without scratch"]
fn gpu_batch_limits_make_progress_and_raw_exact_needs_no_arena() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let small = EncodedSegment {
        version: 1,
        format: StorageFormat::Fp8FromBf16,
        logical_bytes: 2048,
        stored_bytes: 1024,
        checksum: 0,
    };
    // The previous stored_bytes + 1024 heuristic missed alignment overhead.
    assert!(!codec.host_decode_fits(&small, 8192).unwrap());
    assert!(codec.host_decode_fits(&small, 12287).unwrap());
    assert!(!codec.host_decode_fits(&small, 12286).unwrap());
    assert_eq!(codec.scratch_bytes(), 0);
    let source = stream
        .clone_htod(&vec![bf16::from_f32(0.5).to_bits(); 32768])
        .unwrap();
    let inputs: Vec<_> = (0..8)
        .map(|_| EncodeInput {
            source: source.device_ptr(&stream).0,
            bytes: source.len() * 2,
            format: StorageFormat::Fp8FromBf16,
        })
        .collect();
    let budget = 64 * 1024;
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &inputs, budget) }.unwrap();
    assert!((1..inputs.len()).contains(&batch.processed));
    assert_eq!(batch.outputs.len(), batch.processed);
    assert!(batch.outputs.iter().all(Option::is_some));
    let saved: Vec<_> = batch
        .outputs
        .iter()
        .map(|s| snapshot(&stream, s.as_ref().unwrap()))
        .collect();
    drop(batch);
    let retained = codec.scratch_bytes();
    assert!(retained <= budget);
    assert!(codec.host_decode_fits(&saved[0].2, retained).unwrap());
    assert!(!codec.host_decode_fits(&saved[0].2, 0).unwrap());
    assert_eq!(codec.scratch_bytes(), retained);
    let target = stream
        .alloc_zeros::<u8>(source.len() * 2 * inputs.len())
        .unwrap();
    let (_, host, meta) = &saved[0];
    let restores: Vec<_> = (0..inputs.len())
        .map(|i| HostDecodeInput {
            source: host,
            target: target.device_ptr(&stream).0 + (i * source.len() * 2) as u64,
            target_bytes: source.len() * 2,
            meta,
        })
        .collect();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    let count = unsafe { codec.decode_host_batch(&stream, &restores, budget, None) }.unwrap();
    assert!((1..restores.len()).contains(&count));
    assert!(codec.scratch_bytes() <= budget);
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &inputs, 0) }.unwrap();
    assert_eq!(batch.processed, 1);
    assert!(batch.outputs[0].is_none());
    drop(batch);
    assert_eq!(codec.scratch_bytes(), 0);

    let raw: Vec<_> = (0..MAX_BATCH_SEGMENTS + 3)
        .map(|_| EncodeInput {
            source: source.device_ptr(&stream).0,
            bytes: 2,
            format: StorageFormat::Exact,
        })
        .collect();
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &raw, 0) }.unwrap();
    assert_eq!(batch.processed, MAX_BATCH_SEGMENTS);
    assert!(batch.outputs.iter().all(Option::is_none));
    drop(batch);
    assert_eq!(codec.scratch_bytes(), 0);

    let host = vec![0x43; 17 * 1024 * 1024];
    let meta = EncodedSegment {
        version: 1,
        format: StorageFormat::Exact,
        logical_bytes: host.len(),
        stored_bytes: host.len(),
        checksum: crc32fast::hash(&host),
    };
    assert!(codec.host_decode_fits(&meta, 0).unwrap());
    assert_eq!(codec.scratch_bytes(), 0);
    let target = stream.alloc_zeros::<u8>(host.len()).unwrap();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    let restored = unsafe {
        codec.decode_host_batch(
            &stream,
            &[HostDecodeInput {
                source: &host,
                target: target.device_ptr(&stream).0,
                target_bytes: target.len(),
                meta: &meta,
            }],
            0,
            None,
        )
    }
    .unwrap();
    assert_eq!(restored, 1);
    assert_eq!(stream.clone_dtoh(&target).unwrap(), host);
    assert_eq!(codec.scratch_bytes(), 0);

    // An oversized Exact sibling must not consume the encoded sibling's budget.
    let encoded_target = stream.alloc_zeros::<u8>(saved[0].2.logical_bytes).unwrap();
    let mixed = [
        HostDecodeInput {
            source: &host,
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta: &meta,
        },
        HostDecodeInput {
            source: &saved[0].1,
            target: encoded_target.device_ptr(&stream).0,
            target_bytes: encoded_target.len(),
            meta: &saved[0].2,
        },
    ];
    assert_eq!(
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_host_batch(&stream, &mixed, retained, None) }.unwrap(),
        2
    );
    assert!(codec.scratch_bytes() <= retained);
    assert_eq!(stream.clone_dtoh(&target).unwrap(), host);
    let expected: Vec<_> = (0..source.len())
        .flat_map(|_| bf16::from_f32(0.5).to_bits().to_le_bytes())
        .collect();
    assert_eq!(stream.clone_dtoh(&encoded_target).unwrap(), expected);
}

#[test]
#[ignore = "CUDA/nvCOMP 5.3 qualification: heterogeneous ANS batches and equal-budget restore"]
fn gpu_ans_batches_restore_with_the_encode_reservation() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let formats = [
        StorageFormat::Ans,
        StorageFormat::Ans16,
        StorageFormat::AnsFp8,
    ];
    let mut hosts = Vec::new();
    let mut requested = Vec::new();
    for bytes in [4096, 16384, 65536] {
        for format in formats {
            hosts.push(
                (0..bytes)
                    .map(|i| if i % 7 == 0 { (i % 251) as u8 } else { 0 })
                    .collect::<Vec<_>>(),
            );
            requested.push(format);
        }
    }
    let sources: Vec<_> = hosts
        .iter()
        .map(|h| stream.clone_htod(h).unwrap())
        .collect();
    let inputs: Vec<_> = sources
        .iter()
        .zip(requested)
        .map(|(source, format)| EncodeInput {
            source: source.device_ptr(&stream).0,
            bytes: source.len(),
            format,
        })
        .collect();
    // SAFETY: the fixture owns every source CudaSlice for its declared range
    // and retains them through the synchronous codec drain and output snapshot.
    let batch = unsafe { codec.encode_batch(&stream, &inputs, 128 * 1024 * 1024) }.unwrap();
    assert_eq!(batch.processed, inputs.len());
    let saved: Vec<_> = batch
        .outputs
        .iter()
        .map(|s| snapshot(&stream, s.as_ref().unwrap()))
        .collect();
    drop(batch);
    let retained = codec.scratch_bytes();
    let targets: Vec<_> = hosts
        .iter()
        .map(|h| stream.alloc_zeros::<u8>(h.len()).unwrap())
        .collect();
    let restore: Vec<_> = saved
        .iter()
        .zip(&targets)
        .map(|((_, host, meta), target)| HostDecodeInput {
            source: host,
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta,
        })
        .collect();
    assert_eq!(
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_host_batch(&stream, &restore, retained, None) }.unwrap(),
        inputs.len()
    );
    assert_eq!(codec.scratch_bytes(), retained);
    for (target, expected) in targets.iter().zip(&hosts) {
        assert_eq!(stream.clone_dtoh(target).unwrap(), *expected);
    }
    // nvCOMP documents a recoverable per-chunk error for a data-type mismatch.
    // This exercises failure after backend submission, then reuses the arena.
    let mut wrong_type = saved[0].2.clone();
    wrong_type.format = StorageFormat::Ans16;
    let invalid = DecodeInput {
        source: saved[0].0.device_ptr(&stream).0,
        source_bytes: saved[0].0.len(),
        target: targets[0].device_ptr(&stream).0,
        target_bytes: targets[0].len(),
        meta: &wrong_type,
    };
    assert!(matches!(
        // SAFETY: the fixture retains all source/target allocations and metadata
        // through the codec drain; only completed bytes are inspected or reused afterward.
        unsafe { codec.decode_batch(&stream, &[invalid], retained) },
        Err(DecodeError::Corrupt(_))
    ));
    let device: Vec<_> = saved
        .iter()
        .zip(&targets)
        .map(|((source, _, meta), target)| DecodeInput {
            source: source.device_ptr(&stream).0,
            source_bytes: source.len(),
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta,
        })
        .collect();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    unsafe { codec.decode_batch(&stream, &device, retained) }.unwrap();
    for (target, expected) in targets.iter().zip(&hosts) {
        assert_eq!(stream.clone_dtoh(target).unwrap(), *expected);
    }
}

#[test]
#[ignore = "CUDA qualification: CRC32 matches IEEE across tile boundaries and excludes padding"]
fn gpu_crc_checks_exact_payloads_with_unaligned_lengths_and_padding() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let lengths = [
        1, 9, 255, 256, 257, 4095, 4096, 4097, 65537, 524291, 1048579,
    ];
    let hosts: Vec<Vec<u8>> = lengths
        .iter()
        .map(|&len| {
            let mut host: Vec<_> = (0..len).map(|i| (i * 123 + i / 11) as u8).collect();
            if len == 9 {
                host.copy_from_slice(b"123456789");
            }
            host.extend([0xa5; 17]);
            host
        })
        .collect();
    let metadata: Vec<_> = hosts
        .iter()
        .zip(lengths)
        .map(|(h, len)| EncodedSegment {
            version: 1,
            format: StorageFormat::Exact,
            logical_bytes: len,
            stored_bytes: len,
            checksum: if len == 9 {
                0xcbf43926
            } else {
                crc32fast::hash(&h[..len])
            },
        })
        .collect();
    let sources: Vec<_> = hosts
        .iter()
        .map(|h| stream.clone_htod(h).unwrap())
        .collect();
    let targets: Vec<_> = lengths
        .iter()
        .map(|&len| stream.clone_htod(&vec![0x39u8; len + 7]).unwrap())
        .collect();
    let inputs: Vec<_> = sources
        .iter()
        .zip(&targets)
        .zip(&metadata)
        .map(|((source, target), meta)| DecodeInput {
            source: source.device_ptr(&stream).0,
            source_bytes: source.len(),
            target: target.device_ptr(&stream).0,
            target_bytes: target.len(),
            meta,
        })
        .collect();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    unsafe { codec.decode_batch(&stream, &inputs, 1024 * 1024) }.unwrap();
    for ((target, host), len) in targets.iter().zip(&hosts).zip(lengths) {
        let actual = stream.clone_dtoh(target).unwrap();
        assert_eq!(&actual[..len], &host[..len]);
        assert!(actual[len..].iter().all(|&b| b == 0x39));
    }
}

#[test]
#[ignore = "CUDA qualification: overlapping restore ranges are rejected before target writes"]
fn gpu_decode_rejects_overlapping_targets_and_device_sources_before_writes() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = GpuCodec::new(&ctx).unwrap();
    let encoded = [0x30u8; 16];
    let meta = EncodedSegment {
        version: 1,
        format: StorageFormat::Fp8FromBf16,
        logical_bytes: 32,
        stored_bytes: encoded.len(),
        checksum: crc32fast::hash(&encoded),
    };
    let source = stream.clone_htod(&encoded).unwrap();
    let sentinel = vec![0x5au8; 96];
    let target = stream.clone_htod(&sentinel).unwrap();
    let base = target.device_ptr(&stream).0;
    let budget = 64 * 1024;
    for offsets in [[0u64, 0], [0, 16], [16, 0]] {
        let host: Vec<_> = offsets
            .iter()
            .map(|&offset| HostDecodeInput {
                source: &encoded,
                target: base + offset,
                target_bytes: 32,
                meta: &meta,
            })
            .collect();
        // SAFETY: all ranges lie inside retained real allocations. Deliberate
        // overlap is rejected by the codec preflight before any native access.
        let error = unsafe { codec.decode_host_batch(&stream, &host, budget, None) }.unwrap_err();
        assert!(error.contains("target ranges overlap"), "{error}");
        assert_eq!(stream.clone_dtoh(&target).unwrap(), sentinel);
        let device: Vec<_> = offsets
            .iter()
            .map(|&offset| DecodeInput {
                source: source.device_ptr(&stream).0,
                source_bytes: source.len(),
                target: base + offset,
                target_bytes: 32,
                meta: &meta,
            })
            .collect();
        // SAFETY: all ranges lie inside retained real allocations. Deliberate
        // overlap is rejected by the codec preflight before any native access.
        let error = unsafe { codec.decode_batch(&stream, &device, budget) }.unwrap_err();
        assert!(
            matches!(&error, DecodeError::Runtime(message) if message.contains("target ranges overlap")),
            "{error}"
        );
        assert_eq!(stream.clone_dtoh(&target).unwrap(), sentinel);
    }
    // Cover both self-overlap and a source clobbered by another segment's target.
    for source_offset in [8u64, 40] {
        let device = [
            DecodeInput {
                source: base + source_offset,
                source_bytes: 16,
                target: base,
                target_bytes: 32,
                meta: &meta,
            },
            DecodeInput {
                source: source.device_ptr(&stream).0,
                source_bytes: source.len(),
                target: base + 32,
                target_bytes: 32,
                meta: &meta,
            },
        ];
        // SAFETY: all ranges lie inside retained real allocations. Deliberate
        // overlap is rejected by the codec preflight before any native access.
        let error = unsafe { codec.decode_batch(&stream, &device, budget) }.unwrap_err();
        assert!(
            matches!(&error, DecodeError::Runtime(message) if message.contains("source and target ranges overlap")),
            "{error}"
        );
        assert_eq!(stream.clone_dtoh(&target).unwrap(), sentinel);
    }
    assert_eq!(codec.scratch_bytes(), 0);

    // Adjacent targets and shared read-only sources remain valid.
    let device: Vec<_> = [0u64, 32]
        .iter()
        .map(|&offset| DecodeInput {
            source: source.device_ptr(&stream).0,
            source_bytes: source.len(),
            target: base + offset,
            target_bytes: 32,
            meta: &meta,
        })
        .collect();
    // SAFETY: the fixture retains all source/target allocations and metadata
    // through the codec drain; only completed bytes are inspected or reused afterward.
    unsafe { codec.decode_batch(&stream, &device, budget) }.unwrap();
    let mut expected = sentinel.clone();
    for chunk in expected[..64].chunks_exact_mut(32) {
        assert!(cpu::decode(meta.format, &encoded, chunk));
    }
    assert_eq!(stream.clone_dtoh(&target).unwrap(), expected);
}
