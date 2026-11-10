use super::*;
use std::fs;
use std::os::unix::fs::MetadataExt;

#[test]
fn existing_files_are_locked_before_mutation_and_preserved_on_allocation_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache");
    fs::write(&path, vec![0x5a; 4096]).unwrap();
    let files = open_cache_files(
        std::slice::from_ref(&path),
        1,
        512,
        &mut fs::OpenOptions::new(),
        true,
    )
    .unwrap();
    assert_eq!(files[0].metadata().unwrap().len(), 4096);
    assert!(
        open_cache_files(
            std::slice::from_ref(&path),
            1,
            512,
            &mut fs::OpenOptions::new(),
            false
        )
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), vec![0x5a; 4096]);
    let readonly = fs::File::open(&path).unwrap();
    assert!(reserve_cache_space(&[readonly], 8192, true).is_err());
    assert_eq!(fs::read(path).unwrap(), vec![0x5a; 4096]);
}

#[test]
fn another_file_description_blocks_recovery_even_without_a_manager_flock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data");
    fs::write(&path, vec![0x5a; 4096]).unwrap();
    let outstanding = fs::File::open(&path).unwrap();
    let open = || {
        open_cache_files(
            std::slice::from_ref(&path),
            1,
            512,
            &mut fs::OpenOptions::new(),
            true,
        )
    };
    assert!(
        open()
            .unwrap_err()
            .to_string()
            .contains("drained prior file users")
    );
    assert_eq!(fs::read(&path).unwrap(), vec![0x5a; 4096]);
    drop(outstanding);
    assert_eq!(open().unwrap()[0].metadata().unwrap().len(), 4096);
}

#[test]
fn gpu_capacity_is_physically_reserved_for_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let capacity = 1024 * 1024;
    let files = open_cache_files(
        &[dir.path().join("ssd0"), dir.path().join("ssd1")],
        2,
        capacity,
        &mut fs::OpenOptions::new(),
        false,
    )
    .unwrap();
    reserve_cache_space(&files, capacity, false).unwrap();
    for file in &files {
        let metadata = file.metadata().unwrap();
        assert_eq!(metadata.len(), capacity);
        assert!(metadata.blocks() * 512 >= capacity);
    }
    assert!(reserve_cache_space(&files, u64::MAX, false).is_err());
    assert!(aligned_shard_capacity(capacity, 0, 4096).is_err());
}

#[test]
fn failed_allocation_releases_space_reserved_by_earlier_shards() {
    let dir = tempfile::tempdir().unwrap();
    let capacity = 1024 * 1024;
    let mut files = open_cache_files(
        &[dir.path().to_path_buf()],
        2,
        capacity,
        &mut fs::OpenOptions::new(),
        false,
    )
    .unwrap();
    // A real allocation error without filling the test machine's disk.
    files[1] = fs::File::open(dir.path().join("shard-000001.dat")).unwrap();
    let error = reserve_cache_space(&files, capacity, false).unwrap_err();
    assert!(error.to_string().contains("SSD shard 1"), "{error}");
    assert_eq!(files[0].metadata().unwrap().len(), 0);
    assert_eq!(files[0].metadata().unwrap().blocks(), 0);
}

#[test]
fn test_open_cache_files_single_path_single_shard() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_path = temp_dir.path().join("cache.bin");
    let mut options = std::fs::OpenOptions::new();
    let files = open_cache_files(
        std::slice::from_ref(&cache_path),
        1,
        4096,
        &mut options,
        false,
    )
    .unwrap();
    assert_eq!(files.len(), 1);
    assert!(cache_path.is_file());
    assert_eq!(fs::metadata(&cache_path).unwrap().len(), 4096);
}

#[test]
fn test_open_cache_files_single_path_multi_shard() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cache_path = temp_dir.path().join("cache");
    let mut options = std::fs::OpenOptions::new();
    let files = open_cache_files(
        std::slice::from_ref(&cache_path),
        4,
        4096,
        &mut options,
        false,
    )
    .unwrap();
    assert_eq!(files.len(), 4);
    assert!(cache_path.is_dir());
    for shard_id in 0..4 {
        let shard = cache_path.join(format!("shard-{shard_id:06}.dat"));
        assert!(shard.is_file());
        assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
    }
}

#[test]
fn test_open_cache_files_multi_path_per_path_shards() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path0 = temp_dir.path().join("ssd0");
    let path1 = temp_dir.path().join("ssd1");
    let mut options = std::fs::OpenOptions::new();
    // 2 paths * 2 shards_per_path = 4 total shards
    let files = open_cache_files(
        &[path0.clone(), path1.clone()],
        2,
        4096,
        &mut options,
        false,
    )
    .unwrap();
    assert_eq!(files.len(), 4);
    assert!(path0.is_dir());
    assert!(path1.is_dir());
    // path0 gets shards 0,1 ; path1 gets shards 2,3
    for path_id in 0..2 {
        for local_shard in 0..2 {
            let global_shard_id = path_id * 2 + local_shard;
            let expected_path = if path_id == 0 { &path0 } else { &path1 };
            let shard = expected_path.join(format!("shard-{global_shard_id:06}.dat"));
            assert!(
                shard.is_file(),
                "shard {global_shard_id} should be at {}",
                shard.display()
            );
            assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
        }
    }
}

#[test]
fn test_open_cache_files_multi_path_single_shard_uses_all_paths() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path0 = temp_dir.path().join("ssd0");
    let path1 = temp_dir.path().join("ssd1");
    let mut options = std::fs::OpenOptions::new();
    // 2 paths * 1 shard_per_path = 2 total shards, one on each path
    let files = open_cache_files(
        &[path0.clone(), path1.clone()],
        1,
        4096,
        &mut options,
        false,
    )
    .unwrap();
    assert_eq!(files.len(), 2);
    assert!(path0.is_dir());
    assert!(path1.is_dir());
    assert!(path0.join("shard-000000.dat").is_file());
    assert!(path1.join("shard-000001.dat").is_file());
}

#[test]
fn test_open_cache_files_empty_paths_fails() {
    let mut options = std::fs::OpenOptions::new();
    let result = open_cache_files(&[], 1, 4096, &mut options, false);
    assert!(result.is_err());
}
