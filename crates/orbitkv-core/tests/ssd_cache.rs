//! SSD cache integration tests.
//!
//! Verifies the complete SSD lifecycle: write persistence, prefetch from SSD
//! after memory eviction, and data integrity through the full round-trip.

mod common;

use common::*;
use orbitkv_core::*;
use std::num::NonZeroUsize;

const BLOCK_SIZE: usize = 4096;
const NUM_BLOCKS: usize = 4;
/// Pool fits one batch with headroom but not two — forces eviction.
const POOL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE * 2;
const SSD_CAPACITY: u64 = 64 * 1024 * 1024;
const SMALL_SSD_CAPACITY: u64 = (BLOCK_SIZE * 2) as u64;
const OVERSIZED_SSD_CAPACITY: u64 = 512;
const PREFETCH_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Probe whether io_uring is usable in the current environment.
/// Some containers restrict the `io_uring_setup` syscall via seccomp.
fn io_uring_available() -> bool {
    // SAFETY: the zeroed 128-byte output matches Linux io_uring_params;
    // setup initializes it and a successful returned fd is closed exactly once.
    unsafe {
        let mut params = std::mem::MaybeUninit::<[u8; 128]>::zeroed();
        let fd = libc::syscall(
            libc::SYS_io_uring_setup,
            1i32,
            params.as_mut_ptr() as *mut libc::c_void,
        );
        if fd >= 0 {
            libc::close(fd as i32);
            true
        } else {
            false
        }
    }
}

macro_rules! skip_without_io_uring {
    () => {
        if !io_uring_available() {
            eprintln!("Skipping test: io_uring is not available in this environment");
            return;
        }
    };
}

fn ssd_env(instance_id: &'static str) -> (TestEnv, std::path::PathBuf, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                backend: SsdBackend::Uring,
                cache_paths: vec![cache_path.clone()],
                capacity_bytes: SSD_CAPACITY,
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    (env, cache_path, temp_dir)
}

fn ssd_split_env(instance_id: &'static str) -> (TestEnv, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .split_layer(
            "layer_0",
            NUM_BLOCKS,
            BLOCK_SIZE / 2,
            BLOCK_SIZE * NUM_BLOCKS,
        )
        .pool_size(POOL_SIZE)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                backend: SsdBackend::Uring,
                cache_paths: vec![cache_path],
                capacity_bytes: SSD_CAPACITY,
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    (env, temp_dir)
}

fn ssd_multi_path_env(
    instance_id: &'static str,
) -> (TestEnv, Vec<std::path::PathBuf>, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let path0 = temp_dir.path().join("ssd0");
    let path1 = temp_dir.path().join("ssd1");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                backend: SsdBackend::Uring,
                cache_paths: vec![path0.clone(), path1.clone()],
                capacity_bytes: SSD_CAPACITY,
                shards: NonZeroUsize::new(2).unwrap(),
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    (env, vec![path0, path1], temp_dir)
}

fn ssd_sharded_env(instance_id: &'static str) -> (TestEnv, std::path::PathBuf, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                backend: SsdBackend::Uring,
                cache_paths: vec![cache_path.clone()],
                capacity_bytes: SSD_CAPACITY,
                shards: NonZeroUsize::new(4).unwrap(),
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    (env, cache_path, temp_dir)
}

fn ssd_custom_capacity_env(
    instance_id: &'static str,
    capacity_bytes: u64,
) -> (TestEnv, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(EngineConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                backend: SsdBackend::Uring,
                cache_paths: vec![cache_path],
                capacity_bytes,
                ..SsdCacheConfig::default()
            }),
            ..EngineConfig::default()
        })
        .build();
    (env, temp_dir)
}

async fn wait_query_ready(env: &TestEnv, hashes: &[Vec<u8>]) -> (usize, usize) {
    let QueryResult { blocks, missing } =
        tokio::time::timeout(PREFETCH_WAIT_TIMEOUT, env.query(hashes))
            .await
            .expect("SSD query timed out");
    (blocks.len(), missing)
}

#[tokio::test]
async fn concurrent_queries_with_the_same_request_id_own_distinct_results() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-query-ownership");
    let first = env.hashes(91);
    let second = env.hashes(92);
    env.save_and_wait(&first).await;
    env.save_and_wait(&second).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);
    // TestEnv::query deliberately uses the same logical request ID.
    let (a, b) = tokio::join!(env.query(&first[..2]), env.query(&second));
    assert_eq!((a.blocks.len(), a.missing), (2, 0));
    assert_eq!((b.blocks.len(), b.missing), (second.len(), 0));
    let (orbitkv_core::RestoreSource::Memory(first), orbitkv_core::RestoreSource::Memory(second)) =
        (&a.blocks[0], &b.blocks[0])
    else {
        panic!("io_uring restores host blocks")
    };
    assert!(!std::sync::Arc::ptr_eq(first, second));
}

fn cleanup_resident_memory(env: &TestEnv) {
    let stats = env.engine.cleanup_memory_cache();
    assert!(
        stats.evicted_blocks > 0,
        "cleanup_memory_cache should evict resident blocks"
    );
}

#[tokio::test]
async fn selective_ssd_admission_preserves_demand_and_ignores_warming() {
    skip_without_io_uring!();
    for demand_before_eviction in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let env = TestEnvBuilder::new("selective-ssd", "selective-ns")
            .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
            .pool_size(POOL_SIZE)
            .storage(EngineConfig {
                cache_protected_percent: 80,
                ssd_cache_config: Some(SsdCacheConfig {
                    backend: SsdBackend::Uring,
                    cache_paths: vec![dir.path().join("cache.bin")],
                    capacity_bytes: SSD_CAPACITY,
                    write_policy: SsdWritePolicy::Reuse,
                    ..SsdCacheConfig::default()
                }),
                ..EngineConfig::default()
            })
            .build();
        let target = env.hashes(97);
        env.save_and_wait(&target).await;
        drop(
            env.engine
                .count_prefix_hit_blocks_with_prefetch(
                    &env.instance_id,
                    "warming",
                    &target,
                    QueryMode::Warmup,
                    std::time::Instant::now() + std::time::Duration::from_secs(30),
                )
                .await
                .unwrap(),
        );
        if demand_before_eviction {
            drop(env.query(&target).await);
        }
        env.engine.flush_all().await;
        cleanup_resident_memory(&env);
        if !demand_before_eviction {
            assert_eq!(env.query(&target).await.missing, NUM_BLOCKS);
            // The next computation republishes the same identity; bounded
            // history admits it even though its first host replica is gone.
            env.save_and_wait(&target).await;
            env.engine.flush_all().await;
            cleanup_resident_memory(&env);
        }
        let lease = env.assert_all_hit_lease(&target).await;
        env.data().zero_gpu();
        env.load_to_gpu(lease, NUM_BLOCKS).await;
        env.data().assert_gpu_matches_expected();
    }
}

/// Save blocks, flush to SSD, verify the cache file contains non-zero bytes.
#[tokio::test]
async fn ssd_write_persists_to_file() {
    skip_without_io_uring!();
    let (env, cache_path, _temp_dir) = ssd_env("test-ssd-write");

    let file_meta = std::fs::metadata(&cache_path).expect("SSD cache file should be created");
    assert_eq!(
        file_meta.len(),
        SSD_CAPACITY,
        "SSD cache file should be preallocated"
    );

    let hashes = env.hashes(44);
    env.save_and_wait(&hashes).await;
    env.engine.flush_all().await;

    // After flush_all, SSD file must contain written data.
    let mut f = std::fs::File::open(&cache_path).expect("open SSD cache file");
    let mut buf = vec![0u8; 4096];
    std::io::Read::read(&mut f, &mut buf).expect("read SSD cache file");
    assert!(
        buf.iter().any(|&b| b != 0),
        "SSD cache file should contain non-zero data after flush"
    );
}

/// Full SSD prefetch round-trip: save → flush to SSD → evict from memory →
/// query (triggers SSD prefetch) → load → verify data integrity.
#[tokio::test]
async fn ssd_prefetch_roundtrip_after_eviction() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-ssd-prefetch");

    // Phase 1: Save target blocks and ensure they're persisted to SSD.
    let target = env.hashes(1);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    // Phase 2: Evict target blocks from memory while preserving SSD data.
    cleanup_resident_memory(&env);

    // Phase 3: Query the original hashes — should trigger SSD prefetch.
    // First query may return Loading (SSD read in flight), poll until Ready.
    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);

    // Phase 4: Lease, load to GPU, and verify data integrity.
    let lease = env.assert_all_hit_lease(&target).await;
    env.data().zero_gpu();
    env.load_to_gpu(lease, target.len()).await;
    env.data().assert_gpu_matches_expected();
}

#[tokio::test]
async fn ssd_reclaim_frees_pages_independently_of_a_retained_prefix() {
    skip_without_io_uring!();
    for split in [false, true] {
        for from_ssd in [false, true] {
            let (env, _temp_dir) = if split {
                ssd_split_env("test-page-reclaim-split")
            } else {
                let (env, _, dir) = ssd_env("test-page-reclaim-contiguous");
                (env, dir)
            };
            let target = env.hashes(93);
            env.save_and_wait(&target).await;
            env.engine.flush_all().await;
            if from_ssd {
                cleanup_resident_memory(&env);
            }

            let QueryResult {
                mut blocks,
                missing,
            } = env.query(&target).await;
            assert_eq!((blocks.len(), missing), (NUM_BLOCKS, 0));
            let retained = blocks.remove(0);
            drop(blocks);
            let lease = env
                .engine
                .create_query_lease(&env.instance_id, vec![retained])
                .unwrap();
            let stats = env.engine.cleanup_memory_cache();
            assert_eq!(stats.still_referenced_blocks, 1);
            assert_eq!(
                stats.reclaimed_bytes,
                ((NUM_BLOCKS - 1) * BLOCK_SIZE) as u64,
                "one live prefix page retained its siblings: split={split}, from_ssd={from_ssd}"
            );

            env.data().zero_gpu();
            env.load_to_gpu(lease, 1).await;
            let mut expected = vec![0; env.data().total_size()];
            let segment_size = if split { BLOCK_SIZE / 2 } else { BLOCK_SIZE };
            expected[..segment_size].copy_from_slice(&env.data().expected_bytes()[..segment_size]);
            if split {
                let value_offset = BLOCK_SIZE * NUM_BLOCKS;
                expected[value_offset..value_offset + segment_size].copy_from_slice(
                    &env.data().expected_bytes()[value_offset..value_offset + segment_size],
                );
            }
            env.data().assert_gpu_matches(&expected);
        }
    }
}

#[tokio::test]
async fn ssd_sharded_prefetch_roundtrip_after_eviction() {
    skip_without_io_uring!();
    let (env, cache_path, _temp_dir) = ssd_sharded_env("test-ssd-sharded");

    assert!(
        cache_path.is_dir(),
        "sharded SSD cache path should be a directory"
    );
    for shard_id in 0..4 {
        let shard_path = cache_path.join(format!("shard-{shard_id:06}.dat"));
        let file_meta = std::fs::metadata(&shard_path).expect("SSD shard file should be created");
        assert_eq!(
            file_meta.len(),
            SSD_CAPACITY / 4,
            "SSD shard file should be preallocated"
        );
    }

    let target = env.hashes(1);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    for shard_id in 0..4 {
        let shard_path = cache_path.join(format!("shard-{shard_id:06}.dat"));
        let mut file = std::fs::File::open(&shard_path).expect("open SSD shard file");
        let mut buf = vec![0u8; BLOCK_SIZE];
        std::io::Read::read(&mut file, &mut buf).expect("read SSD shard file");
        assert!(
            buf.iter().any(|&b| b != 0),
            "round-robin write should place data in shard {shard_id}"
        );
    }

    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);
}

#[tokio::test]
async fn ssd_multi_path_prefetch_roundtrip_after_eviction() {
    skip_without_io_uring!();
    let (env, cache_paths, _temp_dir) = ssd_multi_path_env("test-ssd-multi-path");

    let shards_per_path = 2;
    for (path_id, path) in cache_paths.iter().enumerate() {
        for local_shard in 0..shards_per_path {
            let global_shard_id = path_id * shards_per_path + local_shard;
            let shard_path = path.join(format!("shard-{global_shard_id:06}.dat"));
            assert!(
                shard_path.is_file(),
                "shard {global_shard_id} should be on path {}",
                path.display()
            );
            let file_meta =
                std::fs::metadata(&shard_path).expect("SSD shard file should be created");
            assert_eq!(
                file_meta.len(),
                SSD_CAPACITY / 4,
                "SSD shard file should be preallocated"
            );
        }
    }

    let target = env.hashes(1);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);

    let lease = env.assert_all_hit_lease(&target).await;
    env.data().zero_gpu();
    env.load_to_gpu(lease, target.len()).await;
    env.data().assert_gpu_matches_expected();
}

#[tokio::test]
async fn ssd_ring_wrap_evicts_old_entries() {
    skip_without_io_uring!();
    let (env, _temp_dir) = ssd_custom_capacity_env("test-ssd-wrap-evicts", SMALL_SSD_CAPACITY);

    let old = env.hashes(31);
    env.save_and_wait(&old[..2]).await;
    env.engine.flush_all().await;
    let new = env.hashes(32);
    env.save_and_wait(&new[..2]).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &old).await;
    assert_eq!(hit, 0);
    assert_eq!(missing, old.len());
}

#[tokio::test]
async fn ssd_oversized_block_drops_disk_write_but_keeps_ram_hit() {
    skip_without_io_uring!();
    let (env, _temp_dir) =
        ssd_custom_capacity_env("test-ssd-oversized-drop", OVERSIZED_SSD_CAPACITY);

    let target = env.hashes(33);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);

    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, 0);
    assert_eq!(missing, target.len());
}

/// SSD prefetch preserves prefix semantics: if SSD only has the first part of
/// the requested prefix, the query resolves to that partial hit plus miss suffix.
#[tokio::test]
async fn ssd_prefetch_reports_partial_prefix_after_cleanup() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-ssd-partial-prefix");

    let all_hashes = env.hashes(7);
    let saved_prefix = all_hashes[..2].to_vec();
    env.save_and_wait(&saved_prefix).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &all_hashes).await;
    assert_eq!(hit, saved_prefix.len());
    assert_eq!(missing, all_hashes.len() - saved_prefix.len());
}

/// A complete SSD miss should resolve to Ready with the original miss count,
/// not stay in Loading forever.
#[tokio::test]
async fn ssd_miss_resolves_to_ready_missing_after_cleanup() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-ssd-miss");

    let persisted = env.hashes(11);
    env.save_and_wait(&persisted).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let missing_hashes = make_block_hashes(NUM_BLOCKS, 222);
    let (hit, missing) = wait_query_ready(&env, &missing_hashes).await;
    assert_eq!(hit, 0);
    assert_eq!(missing, missing_hashes.len());
}

/// A miss result for one req_id must not permanently suppress later backing
/// lookups for the same req_id once the hashes become available in SSD.
#[tokio::test]
async fn ssd_miss_does_not_poison_later_same_req_id_hit() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-ssd-miss-then-hit");

    let unrelated = env.hashes(12);
    env.save_and_wait(&unrelated).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let later_available = make_block_hashes(NUM_BLOCKS, 55);
    let (hit, missing) = wait_query_ready(&env, &later_available).await;
    assert_eq!(hit, 0);
    assert_eq!(missing, later_available.len());

    env.save_and_wait(&later_available).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &later_available).await;
    assert_eq!(hit, later_available.len());
    assert_eq!(missing, 0);
}

/// When RAM already satisfies a prefix and SSD has the suffix, the async
/// prefetch result is the complete query answer: RAM prefix plus SSD suffix.
#[tokio::test]
async fn ssd_prefetch_combines_ram_prefix_with_ssd_suffix() {
    skip_without_io_uring!();
    let (env, _cache_path, _temp_dir) = ssd_env("test-ssd-ram-prefix-ssd-suffix");

    let target = env.hashes(77);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let ram_prefix = target[..2].to_vec();
    env.save_and_wait(&ram_prefix).await;

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);

    let lease = env.assert_all_hit_lease(&target).await;
    env.data().zero_gpu();
    env.load_to_gpu(lease, target.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Split K/V storage should survive the same SSD persistence -> memory cleanup
/// -> prefetch -> load round-trip as contiguous storage.
#[tokio::test]
async fn ssd_prefetch_split_storage_roundtrip_after_cleanup() {
    skip_without_io_uring!();
    let (env, _temp_dir) = ssd_split_env("test-ssd-split-prefetch");

    let target = env.hashes(66);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;
    cleanup_resident_memory(&env);

    let (hit, missing) = wait_query_ready(&env, &target).await;
    assert_eq!(hit, target.len());
    assert_eq!(missing, 0);

    let lease = env.assert_all_hit_lease(&target).await;
    env.data().zero_gpu();
    env.load_to_gpu(lease, target.len()).await;
    env.data().assert_gpu_matches_expected();
}
