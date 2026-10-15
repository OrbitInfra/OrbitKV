//! GPU storage qualification. Run explicitly with cuFile's native or forced
//! compatibility configuration; these tests never label compatibility as GDS.
mod common;

use common::{GpuBuffer, TestEnvBuilder};
use orbitkv_core::{
    EngineConfig, LayerSave, OrbitKVEngine, QueryLeaseId, QueryMode, RestoreSource, SsdBackend,
    SsdCacheConfig, StorageCodec, TransferMode,
};

fn storage(file: std::path::PathBuf, capacity_bytes: u64) -> EngineConfig {
    EngineConfig {
        ssd_cache_config: Some(SsdCacheConfig {
            cache_paths: vec![file],
            capacity_bytes,
            backend: SsdBackend::Cufile,
            ..SsdCacheConfig::default()
        }),
        ..EngineConfig::default()
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn automatic_fallback_preserves_host_layout_and_small_block_capacity() {
    let dir = tempfile::tempdir().unwrap();
    // A driver initialized by an explicit cuFile owner has not established the
    // native-only policy required by auto, even on a native-capable mount.
    let _explicit = TestEnvBuilder::new("explicit-owner", "explicit-owner")
        .layer("attention", 1, 512)
        .pool_size(64 * 1024)
        .storage(storage(dir.path().join("explicit.bin"), 4096))
        .build();
    let env = TestEnvBuilder::new("auto-fallback", "auto-fallback")
        .layer("attention", 4, 512)
        .pool_size(64 * 1024)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_paths: vec![dir.path().join("cache.bin")],
                capacity_bytes: 4096,
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    let hashes = common::make_block_hashes(4, 71);
    env.save_all_layers_one_batch(&hashes).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    env.layers[0].data.zero_gpu();
    let result = env.query(&hashes).await;
    assert_eq!(result.blocks.len(), 4);
    assert!(
        result
            .blocks
            .iter()
            .all(|source| matches!(source, RestoreSource::Memory(_)))
    );
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, result.blocks)
        .unwrap();
    env.load_to_gpu(lease, 4).await;
    env.layers[0].data.assert_gpu_matches_expected();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn fragmented_publications_seal_before_ssd_visibility_and_restore_through_cufile() {
    let dir = tempfile::tempdir().unwrap();
    let env = TestEnvBuilder::new("cufile-partial", "gds-partial")
        .layer("attention", 3, 700)
        .split_layer("split", 3, 700, 3 * 700)
        .storage(storage(dir.path().join("cache.bin"), 1 << 20))
        .build();
    let hashes = common::make_block_hashes(3, 11);
    env.save_layer(0, &hashes).await;
    env.engine.flush_all().await;
    assert!(env.query(&hashes).await.blocks.is_empty());
    env.save_layer(1, &hashes).await;
    env.engine.flush_all().await;
    assert_eq!(
        env.engine.cleanup_memory_cache().evicted_blocks,
        hashes.len()
    );
    let result = env.query(&hashes).await;
    assert_eq!(result.blocks.len(), hashes.len());
    assert!(
        result
            .blocks
            .iter()
            .all(|b| matches!(b, RestoreSource::Ssd { .. }))
    );
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, result.blocks)
        .unwrap();
    for layer in &env.layers {
        layer.data.zero_gpu();
    }
    env.load_to_gpu(lease, hashes.len()).await;
    for layer in &env.layers {
        layer.data.assert_gpu_matches_expected();
    }
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn ssd_sources_restore_split_and_page_first_layouts_without_host_materialization() {
    for page_first in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = TestEnvBuilder::new("cufile-layout", "gds-test")
            .layer("attention", 4, 8192)
            .storage(storage(dir.path().join("cache.bin"), 1 << 20));
        if page_first {
            builder = builder.layer("compact", 4, 700).page_first();
        } else {
            builder = builder.split_layer("split", 4, 700, 4 * 700);
        }
        let env = builder.build();
        let hashes = common::make_block_hashes(4, 21);
        env.save_all_layers_one_batch(&hashes).await;
        env.engine.flush_all().await;
        assert!(env.engine.cleanup_memory_cache().evicted_blocks > 0);
        for layer in &env.layers {
            layer.data.zero_gpu();
        }
        let result = env.query(&hashes).await;
        assert_eq!(result.blocks.len(), hashes.len());
        assert!(
            result
                .blocks
                .iter()
                .all(|block| matches!(block, RestoreSource::Ssd { .. }))
        );
        assert_eq!(
            env.engine.cleanup_memory_cache().evicted_blocks,
            0,
            "query must not populate DRAM"
        );
        let lease = env
            .engine
            .create_query_lease(&env.instance_id, result.blocks)
            .unwrap();
        env.load_to_gpu(lease, hashes.len()).await;
        for layer in &env.layers {
            layer.data.assert_gpu_matches_expected();
        }

        // Mix one republished DRAM block with leased SSD sources in one restore.
        env.save_all_layers_one_batch(&hashes[..1]).await;
        let mixed = env.query(&hashes).await;
        assert!(matches!(mixed.blocks[0], RestoreSource::Memory(_)));
        assert!(
            mixed.blocks[1..]
                .iter()
                .all(|block| matches!(block, RestoreSource::Ssd { .. }))
        );
        let lease = env
            .engine
            .create_query_lease(&env.instance_id, mixed.blocks)
            .unwrap();
        for layer in &env.layers {
            layer.data.zero_gpu();
        }
        env.load_to_gpu(lease, hashes.len()).await;
        for layer in &env.layers {
            layer.data.assert_gpu_matches_expected();
        }

        let prepared = env
            .engine
            .count_prefix_hit_blocks_with_prefetch(
                &env.instance_id,
                "prepare",
                &hashes,
                QueryMode::Prepare,
            )
            .await
            .unwrap();
        assert_eq!(prepared.blocks.len(), hashes.len());
        assert!(
            prepared
                .blocks
                .iter()
                .all(|block| matches!(block, RestoreSource::Memory(_)))
        );
        drop(prepared);
        env.engine
            .unregister_instance_and_wait(&env.instance_id)
            .await
            .unwrap();
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn canceled_disk_lease_releases_ring_capacity_and_large_checkpoint_restores() {
    let dir = tempfile::tempdir().unwrap();
    let size = 10 * 1024 * 1024;
    let env = TestEnvBuilder::new("cufile-pins", "gds-pins")
        .layer("checkpoint", 1, size)
        .pool_size(64 << 20)
        .storage(storage(dir.path().join("cache.bin"), size as u64))
        .build();
    let original = env.hashes(17);
    env.save_and_wait(&original).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let pinned = env.query(&original).await;
    assert!(matches!(
        pinned.blocks.as_slice(),
        [RestoreSource::Ssd { .. }]
    ));
    let lease = env
        .engine
        .create_query_lease(&env.instance_id, pinned.blocks)
        .unwrap();
    let replacement = env.hashes(18);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    assert_eq!(
        env.query(&original).await.blocks.len(),
        1,
        "pinned source survived attempted overwrite"
    );
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(lease, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();

    let canceled = env.assert_all_hit_lease(&original).await;
    env.release(&canceled);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    assert!(env.query(&original).await.blocks.is_empty());
    let ready = env.assert_all_hit_lease(&replacement).await;
    env.release(&ready);
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn short_disk_read_fails_restore_and_releases_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cache.bin");
    // More than two slots: one failed submission must stop remaining chunks
    // while draining both already submitted reads before releasing the extent.
    let size = 10 << 20;
    let env = TestEnvBuilder::new("cufile-short", "gds-short")
        .layer("layer", 1, size)
        .pool_size(64 << 20)
        .storage(storage(file.clone(), size as u64))
        .build();
    let hashes = env.hashes(22);
    env.save_and_wait(&hashes).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let lease = env.assert_all_hit_lease(&hashes).await;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&file)
        .unwrap()
        .set_len(0)
        .unwrap();
    let completion = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &[vec!["layer"]],
            &[(lease, vec![vec![Some(0)]])],
        )
        .unwrap();
    let outcome = env.restore_outcome(completion).await;
    assert!(
        outcome.result.is_err(),
        "short reads must not become successful restores"
    );
    let replacement = env.hashes(23);
    env.save_and_wait(&replacement).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let next = env.assert_all_hit_lease(&replacement).await;
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(next, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn pinned_shard_does_not_prevent_writes_to_another_shard() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = storage(dir.path().join("cache.bin"), 8192);
    config.ssd_cache_config.as_mut().unwrap().shards = std::num::NonZeroUsize::new(2).unwrap();
    let env = TestEnvBuilder::new("cufile-shards", "gds-shards")
        .layer("layer", 1, 4096)
        .storage(config)
        .build();
    let original = env.hashes(31);
    env.save_and_wait(&original).await;
    env.engine.flush_all().await;
    env.engine.cleanup_memory_cache();
    let held = env.assert_all_hit_lease(&original).await;
    // The third write revisits shard zero, whose source is still pinned.
    for salt in [32, 33] {
        let hashes = env.hashes(salt);
        env.save_and_wait(&hashes).await;
        env.engine.flush_all().await;
        env.engine.cleanup_memory_cache();
        let found = env.assert_all_hit_lease(&hashes).await;
        env.release(&found);
    }
    env.layers[0].data.zero_gpu();
    env.load_to_gpu(held, 1).await;
    env.layers[0].data.assert_gpu_matches_expected();
    env.engine
        .unregister_instance_and_wait(&env.instance_id)
        .await
        .unwrap();
}

struct EncodedFixture {
    engine: OrbitKVEngine,
    gpu: GpuBuffer,
    expected: Vec<u8>,
    hashes: Vec<Vec<u8>>,
}

impl EncodedFixture {
    fn new(file: std::path::PathBuf, segment_bytes: usize, split: bool) -> Self {
        let blocks = 2;
        let segments = if split { 2 } else { 1 };
        let total = segment_bytes * blocks * segments;
        let mut expected = vec![0u8; total];
        for segment in 0..segments {
            for block in 0..blocks {
                // The prefix begins with a raw fallback. The next object is
                // encoded and, for split K/V, contains an exact V sibling.
                let value = if block == 0 || segment == 1 {
                    512.0f32
                } else {
                    1.0f32
                };
                let bits = ((value.to_bits() >> 16) as u16).to_le_bytes();
                let start = (segment * blocks + block) * segment_bytes;
                for pair in expected[start..start + segment_bytes].chunks_exact_mut(2) {
                    pair.copy_from_slice(&bits);
                }
            }
        }
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let gpu = GpuBuffer::alloc(ctx, total);
        gpu.copy_from_host(&expected);
        let mut config = storage(file, 64 << 20);
        config.codec = StorageCodec::Fp8;
        let engine = common::test_engine_with_pool(64 << 20, config);
        engine
            .register_context_layer_batch_strided(
                "encoded-gds",
                "encoded-gds",
                0,
                0,
                0,
                1,
                1,
                &["layer".to_owned()],
                &[gpu.as_u64()],
                &[total],
                &[blocks],
                &[segment_bytes],
                &[if split { blocks * segment_bytes } else { 0 }],
                &[segments],
                None,
                None,
                Some(&[orbitkv_state::StorageFormat::Fp8FromBf16]),
                TransferMode::Direct,
                false,
            )
            .unwrap();
        Self {
            engine,
            gpu,
            expected,
            hashes: common::make_block_hashes(blocks, 91),
        }
    }

    async fn save(&self, ids: &[usize]) {
        self.engine
            .batch_save_kv_blocks_from_ipc(
                "encoded-gds",
                self.engine
                    .registration_generation("encoded-gds")
                    .unwrap_or_default(),
                0,
                0,
                0,
                vec![LayerSave {
                    layer_name: "layer".into(),
                    block_ids: ids.to_vec(),
                    block_hashes: ids.iter().map(|&id| self.hashes[id].clone()).collect(),
                }],
            )
            .await
            .unwrap();
        self.engine.flush_all().await;
    }

    async fn query(&self, ids: &[usize]) -> orbitkv_core::QueryResult {
        self.engine
            .count_prefix_hit_blocks_with_prefetch(
                "encoded-gds",
                "gds-codec-test",
                &ids.iter()
                    .map(|&id| self.hashes[id].clone())
                    .collect::<Vec<_>>(),
                QueryMode::Demand,
            )
            .await
            .unwrap()
    }

    async fn lease(&self, ids: &[usize]) -> QueryLeaseId {
        let query = self.query(ids).await;
        assert_eq!(query.blocks.len(), ids.len());
        assert!(
            query
                .blocks
                .iter()
                .all(|source| matches!(source, RestoreSource::Ssd { .. }))
        );
        self.engine
            .create_query_lease("encoded-gds", query.blocks)
            .unwrap()
    }

    async fn restore(&self, lease: QueryLeaseId, ids: &[usize]) -> orbitkv_core::LoadOutcome {
        let completion = self
            .engine
            .restore(
                "encoded-gds",
                0,
                0,
                &[vec!["layer"]],
                &[(lease, vec![ids.iter().copied().map(Some).collect()])],
            )
            .unwrap();
        let orbitkv_core::RestoreExecution::Managed(completion) = completion else {
            panic!("encoded SSD route must execute on the Manager")
        };
        tokio::time::timeout(std::time::Duration::from_secs(30), completion)
            .await
            .unwrap()
            .unwrap()
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn encoded_gds_restores_mixed_prefix_split_siblings_and_segments_larger_than_slots() {
    // A 10 MiB logical FP8 segment stores 5 MiB: decoding its first 4 MiB
    // cuFile chunk would consume incomplete data.
    for (segment_bytes, split) in [(10 * 1024 * 1024, false), (7000, true)] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = EncodedFixture::new(dir.path().join("cache.bin"), segment_bytes, split);
        fixture.save(&[0, 1]).await;
        fixture.engine.cleanup_memory_cache();
        let query = fixture.query(&[0, 1]).await;
        assert_eq!(
            query.blocks.len(),
            2,
            "mixed representation prefix must not truncate"
        );
        assert!(
            query
                .blocks
                .iter()
                .all(|source| matches!(source, RestoreSource::Ssd { .. }))
        );
        assert!(
            query.blocks[1].memory_footprint() < query.blocks[0].memory_footprint(),
            "the second SSD object must actually be encoded"
        );
        assert_eq!(
            fixture.engine.cleanup_memory_cache().evicted_blocks,
            0,
            "encoded demand queries must not materialize host payloads"
        );
        let lease = fixture
            .engine
            .create_query_lease("encoded-gds", query.blocks)
            .unwrap();
        fixture.gpu.zero();
        fixture.restore(lease, &[0, 1]).await.result.unwrap();
        assert_eq!(fixture.gpu.copy_to_host(), fixture.expected);
        fixture
            .engine
            .unregister_instance_and_wait("encoded-gds")
            .await
            .unwrap();
    }
}

#[tokio::test]
#[ignore = "requires CUDA, io_uring and libcufile; run the cuFile qualification gate"]
async fn corrupt_encoded_gds_generation_is_hidden_pinned_and_repaired_after_drain() {
    use std::os::unix::fs::FileExt;

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cache.bin");
    let fixture = EncodedFixture::new(file.clone(), 16384, false);
    // Only the encoded block is published, making its first extent start at 0.
    fixture.save(&[1]).await;
    fixture.engine.cleanup_memory_cache();
    let failing = fixture.lease(&[1]).await;
    let held = fixture.lease(&[1]).await;
    let disk = std::fs::OpenOptions::new().write(true).open(&file).unwrap();
    disk.write_all_at(&[0xff], 0).unwrap();
    disk.sync_all().unwrap();
    fixture.gpu.zero();
    let outcome = fixture.restore(failing, &[1]).await;
    assert!(
        outcome.result.is_err(),
        "GPU CRC must reject the corrupted payload"
    );
    assert!(
        fixture.gpu.copy_to_host().iter().all(|&byte| byte == 0),
        "CRC failure must precede all decode writes"
    );
    assert!(
        fixture.query(&[1]).await.blocks.is_empty(),
        "corruption hides a still-pinned source"
    );

    fixture.gpu.copy_from_host(&fixture.expected);
    fixture.save(&[1]).await;
    fixture.engine.cleanup_memory_cache();
    assert!(
        fixture.query(&[1]).await.blocks.is_empty(),
        "repair must wait for every old lease"
    );
    assert!(fixture.engine.release_query_lease(&held));
    fixture.save(&[1]).await;
    fixture.engine.cleanup_memory_cache();
    let repaired = fixture.lease(&[1]).await;
    fixture.gpu.zero();
    fixture.restore(repaired, &[1]).await.result.unwrap();
    let restored = fixture.gpu.copy_to_host();
    assert!(restored[..16384].iter().all(|&byte| byte == 0));
    assert_eq!(&restored[16384..], &fixture.expected[16384..]);
    fixture
        .engine
        .unregister_instance_and_wait("encoded-gds")
        .await
        .unwrap();
}
