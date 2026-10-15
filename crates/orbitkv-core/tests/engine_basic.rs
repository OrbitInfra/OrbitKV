//! Basic OrbitKVEngine integration tests.
//!
//! Verifies the core lifecycle: data integrity through GPU<>CPU transfers,
//! block deduplication, multi-layer completeness, and namespace isolation.

mod common;

use common::*;
use orbitkv_core::{EngineConfig, LayerSave, TransferMode};

/// Full save -> query -> load round-trip with data integrity check.
#[tokio::test]
async fn save_query_load_roundtrip() {
    let env = TestEnvBuilder::new("test-roundtrip", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Page-first round-trip: all layers of a block live in one host page slot,
/// saved in a single batch and loaded back. Each layer is given DISTINCT
/// content AND a DISTINCT (512-aligned) size, so the page offsets are an
/// uneven prefix sum (0, 512, 1536). A wrong offset or a mishandled prefix
/// sum makes a layer read/write another layer's slice and fails the assert.
#[tokio::test]
async fn page_first_multi_layer_roundtrip() {
    let mut env = TestEnvBuilder::new("test-page-first", "test-ns")
        .page_first()
        .layer("layer_a", 4, 512)
        .layer("layer_b", 4, 1024)
        .layer("layer_c", 4, 1536)
        .build();

    // Make every layer's content unique so layer offsets are actually checked.
    env.layers[1].data.overwrite(0x40);
    env.layers[2].data.overwrite(0x80);

    let hashes = make_block_hashes(4, 0xAB);

    env.save_all_layers_one_batch(&hashes).await;
    for layer in &env.layers {
        layer.data.zero_gpu();
    }

    let lease = env.assert_all_hit_lease(&hashes).await;
    env.load_to_gpu(lease, hashes.len()).await;

    for layer in &env.layers {
        layer.data.assert_gpu_matches_expected();
    }
}

/// HMA cache groups use different physical block tables for the same logical
/// request. Each layer group must therefore load into its own destination IDs.
#[tokio::test]
async fn hybrid_groups_load_to_distinct_block_ids() {
    let mut env = TestEnvBuilder::new("test-hybrid-groups", "test-ns")
        .layer("full_attention", 4, 1024)
        .layer("gdn", 4, 1024)
        .build();
    env.layers[1].data.overwrite(0x40);

    let hashes = make_block_hashes(2, 0xAC);
    let saves = vec![
        LayerSave {
            layer_name: env.layers[0].name.clone(),
            block_ids: vec![0, 1],
            block_hashes: hashes.clone(),
        },
        LayerSave {
            layer_name: env.layers[1].name.clone(),
            block_ids: vec![2, 3],
            block_hashes: hashes.clone(),
        },
    ];
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            env.engine
                .registration_generation(&env.instance_id)
                .unwrap_or_default(),
            0,
            0,
            0,
            saves,
        )
        .await
        .expect("save hybrid groups");
    env.engine.flush_saves().await;

    for layer in &env.layers {
        layer.data.zero_gpu();
    }
    let lease = env.assert_all_hit_lease(&hashes).await;
    let layer_groups = vec![vec!["full_attention"], vec![], vec!["gdn"]];
    let completion = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &layer_groups,
            &[(
                lease,
                vec![
                    vec![Some(2), Some(3)],
                    vec![None, None],
                    vec![None, Some(0)],
                ],
            )],
        )
        .expect("load hybrid groups");
    env.restore_outcome(completion)
        .await
        .result
        .expect("load hybrid groups");

    let mut full_expected = vec![0; 4 * 1024];
    full_expected[2 * 1024..3 * 1024].fill(1);
    full_expected[3 * 1024..4 * 1024].fill(2);
    env.layers[0].data.assert_gpu_matches(&full_expected);

    let mut gdn_expected = vec![0; 4 * 1024];
    gdn_expected[0..1024].fill(4 + 0x40);
    env.layers[1].data.assert_gpu_matches(&gdn_expected);
}

/// Page-first seals a block's single slot on the first save, so one save must
/// carry EVERY layer of the page. A save covering only some layers (e.g. a
/// pipeline stage holding a subset) would seal a page whose other offsets are
/// never written — stale pool memory that load would return as another block's
/// KV. The engine must reject such a partial-layer page-first save loudly.
#[tokio::test]
async fn page_first_rejects_partial_layer_save() {
    let env = TestEnvBuilder::new("test-page-first-partial", "test-ns")
        .page_first()
        .layer("layer_a", 4, 512)
        .layer("layer_b", 4, 512)
        .layer("layer_c", 4, 512)
        .build();
    let hashes = make_block_hashes(4, 0xCD);

    // Submit only layer_a — a partial page that must not seal.
    let partial = vec![LayerSave {
        layer_name: env.layers[0].name.clone(),
        block_ids: (0..hashes.len()).collect(),
        block_hashes: hashes.clone(),
    }];
    let err = env
        .engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            env.engine
                .registration_generation(&env.instance_id)
                .unwrap_or_default(),
            0,
            0,
            0,
            partial,
        )
        .await
        .expect_err("partial page-first save must be rejected");
    assert!(
        err.to_string().contains("must cover all"),
        "unexpected error: {err}"
    );
}

/// Round-trip with NUMA-aware allocation enabled.
#[tokio::test]
async fn save_query_load_roundtrip_with_numa() {
    let env = TestEnvBuilder::new("test-roundtrip-numa", "test-ns")
        .layer("layer_0", 4, 1024)
        .storage(EngineConfig {
            enable_numa_affinity: true,
            ..EngineConfig::default()
        })
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Save path deduplicates blocks already in cache.
#[tokio::test]
async fn save_deduplicates_cached_blocks() {
    let env = TestEnvBuilder::new("test-dedup", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.save_layer(0, &hashes).await; // second save — should dedup

    assert_eq!(env.count_hits_then_release(&hashes).await, 4);
}

/// Multi-layer: hash becomes hittable only after ALL layers are saved.
#[tokio::test]
async fn multi_layer_partial_save_no_hit() {
    let env = TestEnvBuilder::new("test-ml", "test-ns")
        .layer("layer_0", 4, 1024)
        .layer("layer_1", 4, 1024)
        .build();
    let hashes = make_block_hashes(4, 55);

    // Only layer 0 saved — should be 0 hits.
    env.save_layer(0, &hashes).await;
    env.wait_cached().await;
    assert_eq!(
        env.count_hits_then_release(&hashes).await,
        0,
        "partial save must not expose as hit"
    );

    // Save layer 1 — now all hit.
    env.save_layer(1, &hashes).await;
    env.wait_cached().await;
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);
}

/// Split-storage (K/V separated) round-trip with data integrity check.
#[tokio::test]
async fn save_query_load_roundtrip_split_storage() {
    // segment_size=512, kv_stride=4096 (stride > segment → split path)
    let env = TestEnvBuilder::new("test-split", "test-ns")
        .split_layer("layer_0", 4, 512, 4096)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Descriptor sorting must preserve each lease's source/destination association,
/// including skipped blocks and separate K/V regions with different bytes.
#[tokio::test]
async fn split_restore_permuted_leases_preserve_sources_and_skipped_pages() {
    const SEGMENT: usize = 512;
    const STRIDE: usize = 8192;
    let env = TestEnvBuilder::new("test-split-permutation", "test-ns")
        .split_layer("layer_0", 8, SEGMENT, STRIDE)
        .build();
    let hashes = env.hashes(0);
    let mut original = env.data().expected_bytes().to_vec();
    for byte in &mut original[STRIDE..] {
        *byte += 64;
    }
    // SAFETY: this test owns the registered GPU allocation, the source covers
    // its complete extent, and this synchronous call finishes before saving.
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyHtoD_v2(
            env.data().ptr(),
            original.as_ptr().cast(),
            original.len(),
        )
    };
    assert_eq!(result, cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS);
    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let first = env.assert_all_hit_lease(&hashes[..4]).await;
    let second = env.assert_all_hit_lease(&hashes[4..]).await;
    let destinations = [
        Some(6),
        None,
        Some(2),
        Some(0),
        Some(7),
        Some(5),
        Some(1),
        Some(3),
    ];
    let completion = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &[vec!["layer_0"]],
            &[
                (first, vec![destinations[..4].to_vec()]),
                (second, vec![destinations[4..].to_vec()]),
            ],
        )
        .expect("submit permuted leases");
    env.restore_outcome(completion)
        .await
        .result
        .expect("restore permuted leases");

    let mut expected = vec![0; original.len()];
    for (source, destination) in destinations.into_iter().enumerate() {
        if let Some(destination) = destination {
            for region in [0, STRIDE] {
                expected[region + destination * SEGMENT..region + (destination + 1) * SEGMENT]
                    .copy_from_slice(
                        &original[region + source * SEGMENT..region + (source + 1) * SEGMENT],
                    );
            }
        }
    }
    env.data().assert_gpu_matches(&expected);
}

/// Kernel backend round-trip over real mapped pinned allocations.
///
/// Split storage exercises both K and V segment descriptors, so this covers the
/// production path from `PinnedAllocation` through `RawBlock`/`LayerAlloc` into
/// `CopyDesc.host_device` for save and load.
#[tokio::test]
async fn kernel_backend_roundtrip_split_storage() {
    let env = TestEnvBuilder::new("test-kernel-split", "test-ns")
        .split_layer("layer_0", 2, 4096, 16384)
        .transfer_mode(TransferMode::Kernel)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Namespace isolation: blocks in one namespace are invisible to another.
#[tokio::test]
async fn shared_manager_reuses_only_matching_model_and_storage_identity() {
    let a = TestEnvBuilder::new("inst-a", "ns-alpha")
        .layer("layer_0", 2, 1024)
        .build();
    let hashes = make_block_hashes(2, 11);
    a.save_and_wait(&hashes).await;

    // All readers use the same manager and native hashes. Changing the GPU pool
    // capacity is safe; changing model identity or stored geometry must miss.
    for (id, namespace, block_size, num_blocks, page_first, expected) in [
        ("same", "ns-alpha", 1024, 2, false, 2),
        ("capacity", "ns-alpha", 1024, 1, false, 2),
        ("model", "ns-beta", 1024, 2, false, 0),
        ("geometry", "ns-alpha", 512, 2, false, 0),
        ("page-first", "ns-alpha", 1024, 2, true, 0),
    ] {
        register_layers(
            &a.engine,
            id,
            namespace,
            &[LayerInfo {
                name: "layer_0".into(),
                gpu_ptr: a.data().ptr(),
                total_size: a.data().total_size(),
                num_blocks,
                block_size,
                kv_stride: 0,
                segments: 1,
                group: 0,
            }],
            0,
            0,
            0,
            1,
            1,
            TransferMode::Direct,
            page_first,
        );
        let result = a
            .engine
            .count_prefix_hit_blocks_with_prefetch(id, id, &hashes, orbitkv_core::QueryMode::Demand)
            .await
            .unwrap();
        let orbitkv_core::QueryResult { blocks, .. } = result;
        assert_eq!(blocks.len(), expected, "cache identity case {id}");
    }
}

/// Local source grants must keep the same device budget as Manager-owned work.
#[tokio::test]
#[ignore = "requires exclusive GPU restore admission"]
async fn local_restore_grants_share_device_admission_until_drain() {
    let engines = [
        TestEnvBuilder::new("admission-first", "admission-ns")
            .layer("layer_0", 1, 1024)
            .build(),
        TestEnvBuilder::new("admission-second", "admission-ns")
            .layer("layer_0", 1, 1024)
            .build(),
    ];
    let hashes = engines[0].hashes(0);
    for env in &engines {
        env.save_and_wait(&hashes).await;
        env.data().zero_gpu();
    }
    let mut grants = Vec::new();
    for index in 0..128 {
        let owner = index % engines.len();
        let env = &engines[owner];
        let lease = env.assert_all_hit_lease(&hashes).await;
        let execution = env
            .engine
            .restore(
                &env.instance_id,
                0,
                0,
                &[vec!["layer_0"]],
                &[(lease, vec![vec![Some(0)]])],
            )
            .unwrap();
        let orbitkv_core::RestoreExecution::Local(grant) = execution else {
            panic!("raw DRAM must use the engine-local path");
        };
        grants.push((owner, grant));
    }
    let env = &engines[0];
    let lease = env.assert_all_hit_lease(&hashes).await;
    let error = env
        .engine
        .restore(
            &env.instance_id,
            0,
            0,
            &[vec!["layer_0"]],
            &[(lease, vec![vec![Some(0)]])],
        )
        .unwrap_err();
    assert!(error.to_string().contains("decode restore queue is full"));

    let (owner, grant) = grants.pop().unwrap();
    engines[owner]
        .restore_outcome(orbitkv_core::RestoreExecution::Local(grant))
        .await
        .result
        .unwrap();
    engines[owner].data().assert_gpu_matches_expected();
    let lease = env.assert_all_hit_lease(&hashes).await;
    env.load_to_gpu(lease, 1).await;
    env.data().assert_gpu_matches_expected();
    drop(grants);
}
