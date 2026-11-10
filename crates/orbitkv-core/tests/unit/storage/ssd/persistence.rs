use super::*;
use crate::memory::numa::NumaNode;

fn key(id: u8) -> StateKey {
    StateKey::new(
        format!("orbitkv:v2:{}", "ab".repeat(32)),
        orbitkv_state::group_hash(&[id], 0),
    )
}

fn entry(begin: u64, capacity: u64) -> SsdIndexEntry {
    SsdIndexEntry {
        shard_id: 0,
        begin,
        len: 512,
        file_offset: begin % capacity,
        slots: vec![SlotMeta::new(smallvec::smallvec![512], NumaNode::UNKNOWN)],
        encoding: Encoding::Raw,
        readers: Arc::new(AtomicUsize::new(0)),
    }
}

fn files(dir: &Path, capacity: u64) -> Vec<Arc<File>> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("data"))
        .unwrap();
    file.set_len(capacity).unwrap();
    vec![Arc::new(file)]
}

#[test]
fn durable_reservations_prevent_wrap_resurrection_and_reclaim_orphans() {
    let dir = tempfile::tempdir().unwrap();
    let files = files(dir.path(), 1024);
    let path = dir.path().join("index");
    let (mut journal, recovered) = Journal::open(path.clone(), &files, 1024, 512).unwrap();
    assert!(recovered.is_empty());
    for (id, begin) in [(1, 0), (2, 512)] {
        journal.append(&key(id), &entry(begin, 1024), None).unwrap();
        journal
            .append(&key(id), &entry(begin, 1024), Some(42))
            .unwrap();
    }
    journal.append(&key(3), &entry(1024, 1024), None).unwrap();
    // A metadata checkpoint must keep the fence even before data is submitted.
    journal.checkpoint().unwrap();
    drop(journal);
    let (mut journal, recovered) = Journal::open(path.clone(), &files, 1024, 512).unwrap();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].key, key(2));
    assert_eq!(journal.heads(), vec![1536]);
    assert_eq!(journal.report.lock().drops["orphan"], 1);
    journal.append(&key(4), &entry(1536, 1024), None).unwrap();
    journal
        .append(&key(4), &entry(1536, 1024), Some(43))
        .unwrap();
    drop(journal);
    let (_, recovered) = Journal::open(path, &files, 1024, 512).unwrap();
    assert_eq!(recovered[0].key, key(4));
    assert_eq!(recovered.len(), 1);
}

#[test]
fn malformed_metadata_cold_starts_the_whole_set() {
    for fault in [
        "torn",
        "checksum",
        "version",
        "namespace",
        "extent",
        "sequence",
        "record_limit",
        "journal_limit",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let files = files(dir.path(), 1024);
        let path = dir.path().join("index");
        let (mut journal, _) = Journal::open(path.clone(), &files, 1024, 512).unwrap();
        journal.append(&key(1), &entry(0, 1024), None).unwrap();
        journal.append(&key(1), &entry(0, 1024), Some(10)).unwrap();
        journal.checkpoint().unwrap();
        let mut header = journal.header.clone();
        let mut record = journal.records[&key(1)].clone();
        match fault {
            "version" => record.version += 1,
            "namespace" => record.key.namespace = "unbound-model".into(),
            "extent" => record.offset = 512,
            "sequence" => record.sequence = 0,
            _ => {}
        }
        // Preserve the second current entry only if the complete log is trusted.
        header.sequence = record.sequence;
        let mut bytes = frame(&header).unwrap();
        let mut body = frame(&record).unwrap();
        match fault {
            "torn" => {
                body.pop();
            }
            "checksum" => body[4] ^= 1,
            "record_limit" => body[..4].copy_from_slice(&((MAX_RECORD + 1) as u32).to_le_bytes()),
            _ => {}
        }
        bytes.extend(body);
        drop(journal);
        std::fs::write(&path, bytes).unwrap();
        if fault == "journal_limit" {
            OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(MAX_JOURNAL + 1)
                .unwrap();
        }
        let (journal, recovered) = Journal::open(path, &files, 1024, 512).unwrap();
        assert!(recovered.is_empty(), "{fault}");
        assert_eq!(
            journal.report.lock().drops.values().sum::<u64>(),
            1,
            "{fault}"
        );
    }
}

#[test]
fn capacity_shrink_rebases_only_fitting_extents_before_data_resize() {
    let dir = tempfile::tempdir().unwrap();
    let files = files(dir.path(), 2048);
    let path = dir.path().join("index");
    let (mut journal, _) = Journal::open(path.clone(), &files, 2048, 512).unwrap();
    for (id, begin) in [(1, 2048), (2, 2560), (3, 3072)] {
        journal.append(&key(id), &entry(begin, 2048), None).unwrap();
        journal
            .append(&key(id), &entry(begin, 2048), Some(17))
            .unwrap();
    }
    drop(journal);
    let (journal, recovered) = Journal::open(path.clone(), &files, 1024, 512).unwrap();
    assert_eq!(recovered.len(), 2);
    assert_eq!(journal.heads(), vec![1024]);
    assert_eq!(journal.report.lock().drops["capacity"], 1);
    assert_eq!(
        recovered.iter().map(|r| r.entry.begin).collect::<Vec<_>>(),
        [0, 512]
    );
    drop(journal);
    files[0].set_len(1024).unwrap();
    let (_, recovered) = Journal::open(path, &files, 1024, 512).unwrap();
    assert_eq!(recovered.len(), 2);
}

#[test]
fn namespace_and_representation_are_not_inferred_or_aliased() {
    let dir = tempfile::tempdir().unwrap();
    let files = files(dir.path(), 1024);
    let path = dir.path().join("index");
    let (mut journal, _) = Journal::open(path.clone(), &files, 1024, 512).unwrap();
    let mut other = key(1);
    other.namespace = format!("orbitkv:v2:{}", "cd".repeat(32));
    let mut encoded = entry(512, 1024);
    encoded.encoding = Encoding::Encoded;
    encoded.slots[0].encoding = Some(vec![crate::EncodedSegment {
        version: 1,
        format: orbitkv_state::StorageFormat::Exact,
        logical_bytes: 512,
        stored_bytes: 512,
        checksum: 3,
    }]);
    journal.append(&key(1), &entry(0, 1024), Some(1)).unwrap();
    journal.append(&other, &encoded, Some(2)).unwrap();
    drop(journal);
    let (_, recovered) = Journal::open(path, &files, 1024, 512).unwrap();
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].key, key(1));
    assert_eq!(recovered[1].key, other);
    assert!(matches!(recovered[1].entry.encoding, Encoding::Encoded));
}
