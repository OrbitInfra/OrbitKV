use super::*;

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
    drop(admit_own(Some(3072)).unwrap());
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
