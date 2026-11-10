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
        checksum: Arc::default(),
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
fn tails_keep_the_prefix_but_valid_frames_after_corruption_cold_start() {
    for kind in ["incomplete", "crc", "zeros", "mid", "misaligned_mid"] {
        let dir = tempfile::tempdir().unwrap();
        let files = files(dir.path(), 4096);
        let path = dir.path().join("index");
        let (mut journal, _) = Journal::open(path.clone(), &files, 4096, 512).unwrap();
        journal.append(&key(1), &entry(0, 4096), Some(1)).unwrap();
        journal.checkpoint().unwrap();
        let prefix = std::fs::read(&path).unwrap();
        let mut record = journal.records[&key(1)].clone();
        record.sequence += 1;
        record.key = key(2);
        record.begin = 512;
        record.offset = 512;
        let valid = frame(&record, journal.limits.record).unwrap();
        let mut tail = valid.clone();
        match kind {
            "incomplete" => {
                tail.truncate(tail.len() / 2);
            }
            "zeros" => tail = vec![0; 16384],
            _ => tail[4] ^= 1,
        }
        if kind.ends_with("mid") {
            if kind == "misaligned_mid" {
                tail.extend_from_slice(&[0; 17]);
            }
            tail.extend_from_slice(&valid);
        }
        drop(journal);
        let mut bytes = prefix;
        bytes.extend_from_slice(&tail);
        std::fs::write(&path, bytes).unwrap();
        let (mut journal, recovered) = Journal::open(path.clone(), &files, 4096, 512).unwrap();
        if kind.ends_with("mid") {
            assert!(recovered.is_empty());
            assert_eq!(journal.report.lock().drops["metadata_mid_journal"], 1);
        } else {
            assert_eq!(recovered.len(), 1, "{kind}");
            assert_eq!(recovered[0].key, key(1));
            assert_eq!(journal.report.lock().tail_dropped_bytes, tail.len() as u64);
        }
        // New appends start at the durable clean prefix, never after garbage.
        journal
            .append(&key(3), &entry(1024, 4096), Some(3))
            .unwrap();
        drop(journal);
        let (journal, recovered) = Journal::open(path, &files, 4096, 512).unwrap();
        assert_eq!(recovered.len(), if kind.ends_with("mid") { 1 } else { 2 });
        assert!(journal.report.lock().drops.is_empty());
    }
}

#[test]
fn recovery_limits_scale_and_remain_finite() {
    let small = Limits::new(1 << 20, 1, 4096);
    let large = Limits::new(4 << 40, 4, 4096);
    assert_eq!(small.entries, 256);
    assert_eq!(large.entries, 1_048_576);
    assert_eq!(large.metadata, 1 << 30);
    assert_eq!(large.journal, 4 << 30);
    assert_eq!(large.record, 1 << 20);
    assert!(large.parse_seconds > small.parse_seconds);
    assert!(large.verify_seconds > small.verify_seconds);
    assert!(Limits::new(u64::MAX, usize::MAX, 512).verify_seconds <= 86_400);
}

#[test]
fn malformed_metadata_cold_starts_the_whole_set() {
    for fault in [
        "missing",
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
        let mut bytes = frame(&header, journal.limits.record).unwrap();
        let mut body = frame(&record, journal.limits.record).unwrap();
        match fault {
            "torn" => {
                body.pop();
            }
            "checksum" => body[4] ^= 1,
            "record_limit" => {
                body[..4].copy_from_slice(&((journal.limits.record + 1) as u32).to_le_bytes())
            }
            _ => {}
        }
        bytes.extend(body);
        if fault == "missing" {
            bytes.clear();
        }
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
        logical_bytes: 137,
        stored_bytes: 137,
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

#[test]
fn metadata_admission_rejects_without_appending_and_fences_only_durability_errors() {
    let dir = tempfile::tempdir().unwrap();
    let capacity = 20_000 * 512;
    let files = files(dir.path(), capacity);
    let (mut journal, _) = Journal::open(dir.path().join("index"), &files, capacity, 512).unwrap();
    journal
        .append(&key(1), &entry(0, capacity), Some(1))
        .unwrap();
    let template = journal.records[&key(1)].clone();
    let limit = journal.limits.entries;
    assert_eq!(limit, 20_000);
    journal.records = (0..limit - 1)
        .map(|index| {
            let mut record = template.clone();
            record.key.hash = orbitkv_state::group_hash(&index.to_le_bytes(), 0);
            record.begin = index as u64 * 512;
            record.offset = record.begin;
            (record.key.clone(), record)
        })
        .collect();
    journal.metadata_bytes = journal.records.values().map(Record::charge).sum();
    journal
        .append(
            &key(2),
            &entry((limit as u64 - 1) * 512, capacity),
            Some(22),
        )
        .unwrap();
    assert_eq!(journal.records.len(), limit);
    let length = journal.file.metadata().unwrap().len();
    let sequence = journal.header.sequence;
    assert_eq!(
        journal
            .append(&key(3), &entry(0, capacity), None)
            .unwrap_err()
            .kind(),
        io::ErrorKind::WouldBlock
    );
    assert_eq!(journal.file.metadata().unwrap().len(), length);
    assert_eq!(journal.header.sequence, sequence);
    assert!(!journal.is_fenced());
    assert!(frame(&"x".repeat(journal.limits.record), journal.limits.record).is_err());
    journal.fence();
    assert!(
        journal
            .append(&key(2), &entry(512, capacity), None)
            .is_err()
    );
    assert_eq!(journal.file.metadata().unwrap().len(), length);
}
