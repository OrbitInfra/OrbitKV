use super::*;

#[test]
fn adoption_refuses_an_existing_private_hard_link_to_a_partition_shard() {
    let root = tempfile::tempdir().unwrap();
    let namespace = format!("orbitkv:v2:{:064x}", 1);
    let dir = root.path().join(format!("{PARTITION_PREFIX}{namespace}"));
    fs::create_dir(&dir).unwrap();
    let private = root.path().join("private.dat");
    fs::write(&private, b"private cache").unwrap();
    fs::hard_link(&private, dir.join("shard-000000.dat")).unwrap();
    let err = admit(root.path(), &namespace, 1, 512, Some(1024)).unwrap_err();
    assert!(err.to_string().contains("another hard link"));
    assert_eq!(fs::read(private).unwrap(), b"private cache");
    assert!(!root.path().join(LIMIT_FILE).exists());
}

#[test]
fn installing_a_policy_preserves_existing_directory_entries() {
    let temp = tempfile::tempdir().unwrap();
    let namespace = format!("orbitkv:v2:{:064x}", 1);
    let admit = || super::admit(temp.path(), &namespace, 1, 512, Some(3072)).unwrap();
    let other = temp.path().join(".orbitkv-ssd-capacity.tmp");
    fs::write(&other, b"existing private cache").unwrap();
    let inode = fs::metadata(&other).unwrap().ino();
    let policy = temp.path().join(LIMIT_FILE);
    let mut candidate = admit();
    fs::write(&policy, b"pre-existing file").unwrap();
    assert_eq!(
        candidate.publish_limit().unwrap_err().kind(),
        ErrorKind::AlreadyExists
    );
    assert_eq!(fs::read(&policy).unwrap(), b"pre-existing file");
    drop(candidate);
    fs::remove_file(&policy).unwrap();
    admit().publish_limit().unwrap();
    assert_eq!(fs::metadata(&other).unwrap().ino(), inode);
    assert_eq!(fs::read(&other).unwrap(), b"existing private cache");
    assert_eq!(stored_limit(temp.path()).unwrap(), Some(3072));
}

#[test]
fn partially_resized_shards_keep_the_larger_of_each_old_and_requested_length() {
    let root = tempfile::tempdir().unwrap();
    let own = format!("orbitkv:v2:{:064x}", 1);
    let peer = format!("orbitkv:v2:{:064x}", 2);
    for (namespace, sizes) in [(&own, &[0, 1536][..]), (&peer, &[512][..])] {
        let dir = root.path().join(format!("{PARTITION_PREFIX}{namespace}"));
        fs::create_dir(&dir).unwrap();
        for (index, &size) in sizes.iter().enumerate() {
            File::create(dir.join(format!("shard-{index:06}.dat")))
                .unwrap()
                .set_len(size)
                .unwrap();
        }
        fs::write(dir.join("shard-000000.dat.index.tmp"), b"checkpoint").unwrap();
    }
    let admit_own = |limit| admit(root.path(), &own, 2, 1024, limit);
    assert_eq!(
        admit_own(Some(2560)).unwrap_err().kind(),
        ErrorKind::StorageFull
    );
    assert!(!root.path().join(LIMIT_FILE).exists());
    admit_own(Some(3072)).unwrap().publish_limit().unwrap();
    drop(admit_own(None).unwrap());
    assert_eq!(
        required_payload(root.path(), &own, 2, u64::MAX)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn invalid_policy_or_unaccountable_shard_fails_closed() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let own = format!("orbitkv:v2:{:064x}", 1);
    let open = || admit(root.path(), &own, 1, 512, None);
    let policy = root.path().join(LIMIT_FILE);
    let valid = frame(&512, MAX_DECIMAL_U64).unwrap();
    let mut corrupt = valid.clone();
    corrupt[4] ^= 1;
    for bytes in [
        Vec::new(),
        corrupt,
        [valid.as_slice(), &[0]].concat(),
        frame(&0, MAX_DECIMAL_U64).unwrap(),
    ] {
        fs::write(&policy, &bytes).unwrap();
        assert_eq!(open().unwrap_err().kind(), ErrorKind::InvalidData);
        assert_eq!(fs::read(&policy).unwrap(), bytes);
    }
    fs::write(&policy, &valid).unwrap();
    let dir = root.path().join(format!("{PARTITION_PREFIX}{own}"));
    fs::create_dir(&dir).unwrap();
    symlink(&policy, dir.join("shard-000000.dat")).unwrap();
    assert_eq!(open().unwrap_err().kind(), ErrorKind::InvalidData);
}
