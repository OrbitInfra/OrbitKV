//! Explicit GPU byte gates. Compilation is not hardware qualification.
#![cfg(any(feature = "mooncake", feature = "orbite"))]
mod common;

use common::TestEnvBuilder;
use orbitkv_core::{EngineConfig, PeerTransferBackend, RestoreSource, SsdBackend, SsdCacheConfig};

async fn roundtrip(backend: SsdBackend, padded: bool) {
    let engine = match std::env::var("ORBITKV_SSD_TEST_ENGINE").as_deref() {
        Ok("tent") => PeerTransferBackend::Mooncake,
        Ok("orbite") => PeerTransferBackend::OrbiteTcp,
        _ => panic!("set ORBITKV_SSD_TEST_ENGINE=tent|orbite"),
    };
    let root = std::env::var_os("ORBITKV_SSD_TEST_ROOT")
        .expect("set a test-owned directory on the qualification SSD");
    let directory = tempfile::tempdir_in(root).unwrap();
    let builder = TestEnvBuilder::new("engine-file-gpu", "engine-file-gpu")
        .layer("attention", 3, 4096)
        .split_layer("split", 3, 4096, 3 * 4096)
        .pool_size(32 << 20)
        .storage(EngineConfig {
            peer_transfer_backend: engine,
            ssd_cache_config: Some(SsdCacheConfig {
                cache_paths: vec![directory.path().join("cache.bin")],
                capacity_bytes: 64 << 20,
                backend,
                ..Default::default()
            }),
            ..Default::default()
        });
    let env = if padded {
        builder
            .layer("odd-checkpoint", 3, (5 << 20) + 137)
            .split_layer("odd-split", 3, 137, 4096)
            .build()
    } else {
        builder.build()
    };
    let hashes = common::make_block_hashes(3, 83);
    env.save_all_layers_one_batch(&hashes).await;
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
            .all(|source| matches!(source, RestoreSource::Ssd { .. }))
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
#[ignore = "requires CUDA 13 and a Linux SSD; run the documented engine GPU storage gate"]
async fn cuda_staged_file_roundtrip() {
    roundtrip(SsdBackend::Auto, true).await;
}

#[tokio::test]
#[ignore = "requires CUDA 13 and native-only GDS; preserve cuFile/nvidia-fs route evidence separately"]
async fn native_gds_file_roundtrip() {
    roundtrip(SsdBackend::Cufile, false).await;
}

#[cfg(feature = "mooncake")]
#[tokio::test]
#[ignore = "requires CUDA 13 and native-only TENT GDS with padded vectors; preserve native route evidence"]
async fn tent_native_gds_padded_file_roundtrip() {
    assert_eq!(
        std::env::var("ORBITKV_SSD_TEST_ENGINE").as_deref(),
        Ok("tent")
    );
    roundtrip(SsdBackend::Cufile, true).await;
}
