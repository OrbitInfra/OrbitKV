use super::*;
use crate::RestoreSource;
use smallvec::smallvec;

#[test]
fn segmented_and_page_first_sources_check_exact_component_bounds() {
    let split = SlotMeta::new(smallvec![4096, 8192], crate::NumaNode::UNKNOWN);
    assert_eq!(
        segment_offset(&split, 16384, 1, 512, 700).unwrap(),
        16384 + 4096 + 512
    );
    assert!(segment_offset(&split, 0, 0, 4000, 100).is_err());
    assert!(segment_offset(&split, 0, 2, 0, 1).is_err());
    assert!(segment_offset(&split, 0, 0, usize::MAX, 2).is_err());
    let page = SlotMeta::new(smallvec![16384], crate::NumaNode::UNKNOWN);
    assert_eq!(segment_offset(&page, 4096, 0, 8192, 4096).unwrap(), 12288);
}

#[test]
fn encoded_reads_use_physical_offsets_and_full_logical_destinations() {
    use crate::transfer::layout::BlockCopy;
    let mut slot = SlotMeta::new(smallvec![1024, 2048], crate::NumaNode::UNKNOWN);
    slot.encoding = Some(vec![
        EncodedSegment {
            version: 1,
            format: StorageFormat::Fp8FromBf16,
            logical_bytes: 1400,
            stored_bytes: 700,
            checksum: 1,
        },
        EncodedSegment {
            version: 1,
            format: StorageFormat::Exact,
            logical_bytes: 1400,
            stored_bytes: 1400,
            checksum: 2,
        },
    ]);
    let copies = || BlockCopies::Split {
        k: BlockCopy {
            addr: 0x10000,
            bytes: 1400,
        },
        v: BlockCopy {
            addr: 0x20000,
            bytes: 1400,
        },
    };
    validate_slot(&slot).unwrap();
    let planned = encoded_copies(&slot, 4096, 0, copies(), StorageFormat::Fp8FromBf16).unwrap();
    assert_eq!(
        planned[0].0,
        CopyRange {
            file_offset: 4096,
            device: 0x10000,
            bytes: 700
        }
    );
    assert_eq!(
        planned[1].0,
        CopyRange {
            file_offset: 5120,
            device: 0x20000,
            bytes: 1400
        }
    );
    assert_eq!(planned[0].1.logical_bytes, 1400);
    assert!(encoded_copies(&slot, 4096, 1, copies(), StorageFormat::Fp8FromBf16).is_err());
    assert!(encoded_copies(&slot, 4096, 0, copies(), StorageFormat::Ans).is_err());
    assert!(
        encoded_copies(
            &slot,
            4096,
            0,
            BlockCopies::Contiguous(BlockCopy {
                addr: 1,
                bytes: 2800
            }),
            StorageFormat::Fp8FromBf16
        )
        .is_err()
    );
    slot.encoding.as_mut().unwrap()[0].stored_bytes = 1500;
    assert!(validate_slot(&slot).is_err());
    slot.encoding.as_mut().unwrap().pop();
    assert!(validate_slot(&slot).is_err());
}

#[test]
fn invalid_slot_sizes_and_offset_overflow_are_rejected_before_gpu_io() {
    let mut slot = SlotMeta::new(smallvec![4096], crate::NumaNode::UNKNOWN);
    slot.total_size = 8192;
    assert!(validate_slot(&slot).is_err());
    slot.segment_sizes = smallvec![u64::MAX, 1];
    assert!(validate_slot(&slot).is_err());
    assert!(segment_offset(&slot, 1, 1, 0, 1).is_err());
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn one_raw_extent_restores_through_independent_uring_and_cufile_paths() {
    restore_one_extent_through_both_paths(StorageFormat::Exact).await;
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring, libcufile and nvCOMP; run the cuFile qualification gate"]
async fn one_ans_extent_restores_through_independent_uring_and_cufile_paths() {
    restore_one_extent_through_both_paths(StorageFormat::Ans).await;
}

async fn restore_one_extent_through_both_paths(format: StorageFormat) {
    use std::num::NonZeroU64;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use cudarc::driver::{CudaContext, DevicePtr, result};
    use tokio::sync::oneshot;

    use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
    use crate::codec::gpu::{EncodeInput, GpuCodec};
    use crate::storage::Storage;
    use crate::storage::ssd::SsdReadPath;
    use crate::transfer::layout::{KVCacheGeometry, KVCacheLayout};
    use crate::transfer::worker::{GpuWorkerPool, LoadTask, TransferBlock};
    use crate::{EngineConfig, NumaNode, SsdBackend, SsdCacheConfig, StorageCodec, TransferMode};

    const BYTES: usize = 65_536;
    const CODEC_BUDGET: usize = 64 * 1024 * 1024;
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let expected: Vec<u8> = (0..BYTES)
        .map(|i| if i % 7 == 0 { (i % 251) as u8 } else { 0 })
        .collect();
    let (payload, encoding) = if format == StorageFormat::Ans {
        let input = stream.clone_htod(&expected).unwrap();
        let mut codec = GpuCodec::new(&ctx).unwrap();
        let inputs = [EncodeInput {
            source: input.device_ptr(&stream).0,
            bytes: BYTES,
            format,
        }];
        // The input and borrowed codec output remain alive through the copy.
        let batch = unsafe { codec.encode_batch(&stream, &inputs, CODEC_BUDGET) }.unwrap();
        assert_eq!(batch.processed, 1);
        let output = batch.outputs[0].as_ref().expect("real ANS encoding");
        assert_eq!(output.meta.format, StorageFormat::Ans);
        assert!(output.meta.stored_bytes < BYTES);
        let snapshot = stream.alloc_zeros::<u8>(output.meta.stored_bytes).unwrap();
        let copied = unsafe {
            result::memcpy_dtod_async(
                snapshot.device_ptr(&stream).0,
                output.device,
                output.meta.stored_bytes,
                stream.cu_stream(),
            )
        };
        stream.synchronize().unwrap();
        copied.unwrap();
        let payload = stream.clone_dtoh(&snapshot).unwrap();
        output.meta.validate(&payload).unwrap();
        (payload, Some(vec![output.meta.clone()]))
    } else {
        (expected.clone(), None)
    };

    let directory = tempfile::tempdir().unwrap();
    let storage = Storage::new_with_config(
        8 * 1024 * 1024,
        false,
        EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_paths: vec![directory.path().join("shared-extent.bin")],
                capacity_bytes: 2 * 1024 * 1024,
                backend: SsdBackend::Cufile,
                ..SsdCacheConfig::default()
            }),
            codec: if format == StorageFormat::Ans {
                StorageCodec::Ans
            } else {
                StorageCodec::None
            },
            codec_budget: CODEC_BUDGET,
            ..EngineConfig::default()
        },
        &[],
    )
    .unwrap();
    let store = storage.ssd_store.as_ref().unwrap();
    assert!(store.gpu_io.available());
    let padded = payload.len().next_multiple_of(512);
    let allocation = storage
        .allocate(NonZeroU64::new(padded as u64).unwrap(), None)
        .unwrap();
    // This allocation is exclusively owned until its sealed block is ingested.
    unsafe {
        let target = std::slice::from_raw_parts_mut(allocation.as_non_null().as_ptr(), padded);
        target.fill(0);
        target[..payload.len()].copy_from_slice(&payload);
    }
    let mut raw =
        RawBlock::single_segment(Segment::new(allocation.as_non_null(), padded, allocation));
    raw.storage_format = format;
    raw.encoding = encoding;
    let sealed = Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)]));
    let key = StateKey::new("independent-ssd-paths".into(), vec![1]);
    store.ingest_batch([(&key, &sealed)], false);
    store.flush().await;
    drop(sealed);
    drop(payload);
    assert_eq!(storage.cleanup_memory_cache().evicted_blocks, 0);

    let sources = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin(crate::transfer_engine::tests::demand()))
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 1);
    let source = Arc::clone(&sources[0]);
    drop(sources);
    let generation = (
        source.entry.shard_id,
        source.entry.begin,
        source.entry.file_offset,
        source.entry.len,
    );
    let mut target = stream.alloc_zeros::<u8>(BYTES).unwrap();
    let mut layout = KVCacheLayout::bind(
        target.device_ptr(&stream).0,
        BYTES,
        KVCacheGeometry::new(1, BYTES, 0, 1, None, 1).unwrap(),
    )
    .unwrap();
    layout.storage_format = format;
    let worker =
        GpuWorkerPool::new(0, NumaNode::UNKNOWN, TransferMode::Direct, None, None).unwrap();
    // Host staging works before any cuFile operation and again after its success.
    for (index, path) in [SsdReadPath::Uring, SsdReadPath::Cufile, SsdReadPath::Uring]
        .into_iter()
        .enumerate()
    {
        assert!(store.gpu_io.available());
        assert!(source.cufile_eligible(CODEC_BUDGET));
        stream
            .memcpy_htod(&vec![0xa5u8; BYTES], &mut target)
            .unwrap();
        stream.synchronize().unwrap();
        let (completion, result) = oneshot::channel();
        let planned_source = RestoreSource::Ssd {
            lease: Arc::clone(&source),
            path,
            allow_uring_fallback: false,
        };
        worker
            .submit_load(LoadTask {
                plan: crate::planning::restore::RestorePlan::new(0, [(0, &planned_source)])
                    .unwrap(),
                layers: vec![LayerTransferData {
                    layer_name: "attention".into(),
                    layout: layout.clone(),
                    blocks: vec![TransferBlock {
                        block_idx: 0,
                        block: TransferPayload::Ssd {
                            source: Arc::clone(&source),
                            path,
                            slot_id: 0,
                            offset: 0,
                        },
                    }],
                }],
                completion,
                reservations: vec![],
                peer_bytes: 0,
                codec_budget: CODEC_BUDGET,
                decode_ready_started: std::time::Instant::now(),
                decode_ready_observation: Box::new(crate::cost::Observation::disabled()),
                decode_admission: None,
            })
            .unwrap();
        let completed = tokio::time::timeout(Duration::from_secs(30), result).await;
        if completed.is_err() {
            // A test timeout must not free the target of submitted GPU work.
            worker.drain().await.unwrap();
        }
        completed
            .expect("SSD restore must terminate")
            .unwrap()
            .result
            .unwrap();
        assert!(worker.ssd_host_tx.lock().is_some());
        assert_eq!(worker.ssd_tx.lock().is_some(), index != 0);
        assert_eq!(stream.clone_dtoh(&target).unwrap(), expected, "{path:?}");
        assert!(store.gpu_io.available());
        assert!(source.cufile_eligible(CODEC_BUDGET));
        let current = store
            .discover(std::slice::from_ref(&key))
            .into_iter()
            .map_while(|candidate| candidate?.pin(crate::transfer_engine::tests::demand()))
            .collect::<Vec<_>>();
        assert_eq!(current.len(), 1);
        assert_eq!(
            (
                current[0].entry.shard_id,
                current[0].entry.begin,
                current[0].entry.file_offset,
                current[0].entry.len,
            ),
            generation
        );
        assert!(Arc::ptr_eq(
            &current[0].entry.readers,
            &source.entry.readers
        ));
        assert_eq!(storage.cleanup_memory_cache().evicted_blocks, 0);
    }
    worker.drain().await.unwrap();
    assert_eq!(source.entry.readers.load(Ordering::Acquire), 1);
    assert_eq!(Arc::strong_count(&source), 1);
}
