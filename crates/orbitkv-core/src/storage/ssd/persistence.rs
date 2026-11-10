//! Bounded metadata journal. Payloads are read and written only by OrbiTE.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::index::{Encoding, SsdIndexEntry};
use crate::{SlotMeta, StateKey};

const VERSION: u32 = 1;
pub(super) const MAX_RECORD: usize = 256 * 1024;
pub(super) const MAX_ENTRIES: usize = 16_384;
pub(super) const MAX_METADATA: usize = 16 * 1024 * 1024;
const MAX_JOURNAL: u64 = 64 * 1024 * 1024;
const PARSE_BUDGET: Duration = Duration::from_secs(30);
pub(super) const VERIFY_BUDGET: Duration = Duration::from_secs(180);
pub(super) const VERIFY_CHUNK: usize = 1024 * 1024;

#[derive(Default, Serialize)]
pub(crate) struct RecoveryReport {
    pub parsed_bytes: u64,
    pub metadata_bytes: usize,
    pub entries: usize,
    pub verified_bytes: u64,
    pub elapsed_seconds: f64,
    pub finished: bool,
    pub drops: BTreeMap<String, u64>,
}

impl RecoveryReport {
    pub(super) fn drop_entry(&mut self, reason: &str) {
        *self.drops.entry(reason.into()).or_default() += 1;
        crate::metrics::core_metrics().ssd_recovery_drops.add(
            1,
            &[opentelemetry::KeyValue::new("reason", reason.to_owned())],
        );
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shard {
    device: u64,
    inode: u64,
    capacity: u64,
    head: u64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    version: u32,
    cache: Uuid,
    sequence: u64,
    alignment: u64,
    shards: Vec<Shard>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    cache: Uuid,
    manager: Uuid,
    sequence: u64,
    key: StateKey,
    shard: usize,
    begin: u64,
    offset: u64,
    length: u64,
    slots: Vec<SlotMeta>,
    encoding: Encoding,
    // None is a durable reservation/invalidation, never a readable replica.
    checksum: Option<u32>,
}

impl Record {
    fn charge(&self) -> usize {
        std::mem::size_of::<Self>()
            + 128 // BTree node and index bookkeeping, conservatively per entry.
            + self.key.namespace.capacity()
            + self.key.hash.capacity()
            + self.slots.capacity() * std::mem::size_of::<SlotMeta>()
            + self.slots.iter().map(|slot| {
                slot.segment_sizes.capacity() * std::mem::size_of::<u64>()
                    + slot.encoding.as_ref().map_or(0, |e| {
                        e.capacity() * std::mem::size_of::<crate::EncodedSegment>()
                    })
            }).sum::<usize>()
    }

    fn entry(&self) -> SsdIndexEntry {
        SsdIndexEntry {
            shard_id: self.shard,
            begin: self.begin,
            len: self.length,
            file_offset: self.offset,
            slots: self.slots.clone(),
            encoding: self.encoding.clone(),
            readers: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn validate(&self, header: &Header) -> io::Result<()> {
        if self.version != VERSION {
            return Err(invalid("metadata_version"));
        }
        if self.cache != header.cache || self.manager.is_nil() {
            return Err(invalid("metadata_identity"));
        }
        let shard = header
            .shards
            .get(self.shard)
            .ok_or_else(|| invalid("extent_bounds"))?;
        if !orbitkv_state::is_storage_namespace(&self.key.namespace)
            || self.key.hash.len() < 17
            || self.key.hash.len() > 4096
            || &self.key.hash[..4] != b"OKS\x01"
            || self.key.hash.get(8..16) != Some(&(self.key.hash.len() as u64 - 16).to_le_bytes())
            || self.length == 0
            || !self.length.is_multiple_of(header.alignment)
            || !self.offset.is_multiple_of(header.alignment)
            || self.begin % shard.capacity != self.offset
            || self
                .offset
                .checked_add(self.length)
                .is_none_or(|end| end > shard.capacity)
            || self.begin.checked_add(self.length).is_none()
            || self.slots.is_empty()
            || self.slots.len() > 4096
        {
            return Err(invalid("extent_bounds"));
        }
        let mut payload = 0u64;
        let mut encoded = false;
        for slot in &self.slots {
            if slot.segment_sizes.is_empty()
                || slot.segment_sizes.len() > 64
                || slot.segment_sizes.contains(&0)
                || slot
                    .segment_sizes
                    .iter()
                    .try_fold(0u64, |n, &v| n.checked_add(v))
                    != Some(slot.total_size)
            {
                return Err(invalid("slot_bounds"));
            }
            payload = payload
                .checked_add(slot.total_size)
                .ok_or_else(|| invalid("slot_bounds"))?;
            if let Some(encoding) = &slot.encoding {
                encoded = true;
                if encoding.len() != slot.segment_sizes.len() {
                    return Err(invalid("encoding"));
                }
                for (meta, &bytes) in encoding.iter().zip(&slot.segment_sizes) {
                    let bytes = usize::try_from(bytes).map_err(|_| invalid("encoding"))?;
                    meta.validate_metadata(bytes)
                        .map_err(|_| invalid("encoding"))?;
                    if meta.stored_bytes != bytes {
                        return Err(invalid("encoding"));
                    }
                }
            }
        }
        if payload.checked_next_multiple_of(header.alignment) != Some(self.length)
            || encoded != matches!(self.encoding, Encoding::Encoded)
        {
            return Err(invalid("slot_bounds"));
        }
        Ok(())
    }
}

pub(super) struct Recovered {
    pub key: StateKey,
    pub entry: SsdIndexEntry,
    pub checksum: u32,
}

pub(super) struct Journal {
    header: Header,
    manager: Uuid,
    records: BTreeMap<StateKey, Record>,
    metadata_bytes: usize,
    path: PathBuf,
    file: File,
    failed: bool,
    pub report: Arc<Mutex<RecoveryReport>>,
}

fn invalid(reason: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason)
}

fn read_frame<T: for<'a> Deserialize<'a>>(file: &mut File) -> io::Result<Option<T>> {
    let mut prefix = [0; 8];
    match file.read(&mut prefix[..1])? {
        0 => return Ok(None),
        1 => {}
        _ => return Err(invalid("metadata_framing")),
    }
    file.read_exact(&mut prefix[1..])
        .map_err(|_| invalid("metadata_torn"))?;
    let length = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
    let checksum = u32::from_le_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]);
    if length == 0 || length > MAX_RECORD {
        return Err(invalid("metadata_record_limit"));
    }
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|_| invalid("metadata_torn"))?;
    if crc32c::crc32c(&bytes) != checksum {
        return Err(invalid("metadata_checksum"));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|_| invalid("metadata_schema"))
}

fn frame(value: &impl Serialize) -> io::Result<Vec<u8>> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    if bytes.is_empty() || bytes.len() > MAX_RECORD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SSD metadata record limit",
        ));
    }
    let mut framed = Vec::with_capacity(8 + bytes.len());
    framed.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    framed.extend_from_slice(&crc32c::crc32c(&bytes).to_le_bytes());
    framed.extend_from_slice(&bytes);
    Ok(framed)
}

pub(super) fn fault(name: &str) {
    #[cfg(feature = "test-hooks")]
    crate::test_faults::pause_blocking(name);
    #[cfg(not(feature = "test-hooks"))]
    let _ = name;
}

impl Journal {
    pub(super) fn open(
        path: PathBuf,
        files: &[Arc<File>],
        capacity: u64,
        alignment: u64,
    ) -> io::Result<(Self, Vec<Recovered>)> {
        let started = Instant::now();
        let shards = files
            .iter()
            .map(|file| {
                let metadata = file.metadata()?;
                Ok(Shard {
                    device: metadata.dev(),
                    inode: metadata.ino(),
                    capacity,
                    head: 0,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let report = Arc::new(Mutex::new(RecoveryReport::default()));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let mut journal = Self {
            header: Header {
                version: VERSION,
                cache: Uuid::new_v4(),
                sequence: 0,
                alignment,
                shards,
            },
            manager: Uuid::new_v4(),
            records: BTreeMap::new(),
            metadata_bytes: 0,
            path,
            file,
            failed: false,
            report,
        };
        let current = journal.header.clone();
        let length = journal.file.metadata()?.len();
        journal.report.lock().parsed_bytes = length;
        if length != 0
            && let Err(error) = journal.replay(&current, started)
        {
            if error.kind() != io::ErrorKind::InvalidData {
                return Err(error);
            }
            journal.report.lock().drop_entry(&error.to_string());
            log::warn!("SSD persistence cold start: {error}");
            journal.header = current.clone();
            journal.records.clear();
            journal.metadata_bytes = 0;
        }
        let changed_geometry = journal.header.alignment != alignment
            || journal
                .header
                .shards
                .iter()
                .any(|shard| shard.capacity != capacity);
        journal.records.retain(|_, record| {
            let reason = if record.checksum.is_none() {
                Some("orphan")
            } else if record
                .offset
                .checked_add(record.length)
                .is_none_or(|end| end > capacity)
                || !record.offset.is_multiple_of(alignment)
                || !record.length.is_multiple_of(alignment)
            {
                Some("capacity")
            } else if files
                .get(record.shard)
                .and_then(|f| f.metadata().ok())
                .is_none_or(|meta| record.offset + record.length > meta.len())
            {
                Some("data_length")
            } else {
                None
            };
            if let Some(reason) = reason {
                journal.report.lock().drop_entry(reason);
                false
            } else {
                true
            }
        });
        if changed_geometry {
            journal.header.shards = current.shards;
            journal.header.alignment = alignment;
            for record in journal.records.values_mut() {
                record.begin = record.offset;
                let shard = &mut journal.header.shards[record.shard];
                shard.head = shard.head.max(record.begin + record.length);
            }
        }
        journal.metadata_bytes = journal.records.values().map(Record::charge).sum();
        // Rewrite before any resize or new payload submission. Orphans no longer
        // consume metadata, but the durable heads retain their overwrite fences.
        journal.checkpoint()?;
        let mut recovered = Vec::with_capacity(journal.records.len());
        for record in journal.records.values() {
            if let Some(checksum) = record.checksum {
                recovered.push(Recovered {
                    key: record.key.clone(),
                    entry: record.entry(),
                    checksum,
                });
            }
        }
        recovered.sort_by_key(|r| (r.entry.shard_id, r.entry.begin));
        let mut report = journal.report.lock();
        report.entries = recovered.len();
        report.metadata_bytes = journal.metadata_bytes;
        report.elapsed_seconds = started.elapsed().as_secs_f64();
        drop(report);
        Ok((journal, recovered))
    }

    fn replay(&mut self, current: &Header, started: Instant) -> io::Result<()> {
        if self.file.metadata()?.len() > MAX_JOURNAL {
            return Err(invalid("metadata_journal_limit"));
        }
        let header: Header =
            read_frame(&mut self.file)?.ok_or_else(|| invalid("metadata_header"))?;
        if header.version != VERSION {
            return Err(invalid("metadata_version"));
        }
        if header.cache.is_nil()
            || header.shards.len() != current.shards.len()
            || !matches!(header.alignment, 512 | 4096)
            || header.shards.iter().zip(&current.shards).any(|(old, new)| {
                old.capacity == 0
                    || !old.capacity.is_multiple_of(header.alignment)
                    || old.device != new.device
                    || old.inode != new.inode
            })
        {
            return Err(invalid("metadata_identity"));
        }
        self.header = header;
        let mut last = 0;
        while let Some(record) = read_frame::<Record>(&mut self.file)? {
            if started.elapsed() > PARSE_BUDGET {
                return Err(invalid("metadata_time_limit"));
            }
            if record.sequence <= last {
                return Err(invalid("metadata_sequence"));
            }
            record.validate(&self.header)?;
            last = record.sequence;
            self.header.sequence = self.header.sequence.max(last);
            self.apply(record)?;
        }
        Ok(())
    }

    fn apply(&mut self, record: Record) -> io::Result<()> {
        let shard = &mut self.header.shards[record.shard];
        shard.head = shard.head.max(record.begin + record.length);
        let tail = shard.head.saturating_sub(shard.capacity);
        self.records
            .retain(|_, old| old.shard != record.shard || old.begin >= tail);
        // A checkpoint can contain records older than its durable head. Never
        // resurrect them even if their frame survived a later overwrite.
        if record.begin >= tail {
            self.records.insert(record.key.clone(), record);
        }
        self.metadata_bytes = self.records.values().map(Record::charge).sum();
        if self.records.len() > MAX_ENTRIES || self.metadata_bytes > MAX_METADATA {
            return Err(invalid("metadata_index_limit"));
        }
        Ok(())
    }

    pub(super) fn heads(&self) -> Vec<u64> {
        self.header.shards.iter().map(|s| s.head).collect()
    }

    pub(super) fn append(
        &mut self,
        key: &StateKey,
        entry: &SsdIndexEntry,
        checksum: Option<u32>,
    ) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other(
                "SSD persistence is fenced after a durability failure",
            ));
        }
        let sequence = self
            .header
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("metadata_sequence"))?;
        let record = Record {
            version: VERSION,
            cache: self.header.cache,
            manager: self.manager,
            sequence,
            key: key.clone(),
            shard: entry.shard_id,
            begin: entry.begin,
            offset: entry.file_offset,
            length: entry.len,
            slots: entry.slots.clone(),
            encoding: entry.encoding.clone(),
            checksum,
        };
        record.validate(&self.header)?;
        let tail =
            (entry.begin + entry.len).saturating_sub(self.header.shards[entry.shard_id].capacity);
        let mut count = 1;
        let mut bytes = record.charge();
        for (old_key, old) in &self.records {
            if old_key != key && (old.shard != entry.shard_id || old.begin >= tail) {
                count += 1;
                bytes += old.charge();
            }
        }
        if count > MAX_ENTRIES || bytes > MAX_METADATA {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "SSD persistence metadata budget exhausted",
            ));
        }
        let bytes = frame(&record)?;
        let result = (|| {
            let compact = self.file.metadata()?.len() + bytes.len() as u64 > MAX_JOURNAL;
            #[cfg(feature = "test-hooks")]
            let compact = compact || crate::test_faults::active("ssd_force_checkpoint");
            if compact {
                self.checkpoint()?;
            }
            if checksum.is_some() {
                fault("ssd_before_record");
                let mid = bytes.len() / 2;
                self.file.write_all(&bytes[..mid])?;
                fault("ssd_mid_record");
                self.file.write_all(&bytes[mid..])?;
                fault("ssd_record_written");
            } else {
                self.file.write_all(&bytes)?;
            }
            self.file.sync_data()?;
            self.header.sequence = sequence;
            self.apply(record)?;
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub(super) fn abandon(&mut self, key: &StateKey, entry: &SsdIndexEntry) {
        if self.records.get(key).is_some_and(|r| {
            r.begin == entry.begin && r.shard == entry.shard_id && r.checksum.is_none()
        }) {
            self.records.remove(key);
            self.metadata_bytes = self.records.values().map(Record::charge).sum();
        }
    }

    pub(super) fn invalidate(&mut self, key: &StateKey, entry: &SsdIndexEntry) -> io::Result<()> {
        if self
            .records
            .get(key)
            .is_some_and(|r| r.begin == entry.begin && r.shard == entry.shard_id)
        {
            self.append(key, entry, None)?;
            self.abandon(key, entry);
        }
        Ok(())
    }

    pub(super) fn fence(&mut self) {
        self.failed = true;
    }

    pub(super) fn is_fenced(&self) -> bool {
        self.failed
    }

    fn checkpoint(&mut self) -> io::Result<()> {
        let mut temporary = self.path.as_os_str().to_os_string();
        temporary.push(".tmp");
        let temporary = PathBuf::from(temporary);
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temporary)?;
            file.write_all(&frame(&self.header)?)?;
            let mut records: Vec<_> = self.records.values().collect();
            records.sort_by_key(|record| record.sequence);
            for record in records {
                let bytes = frame(record)?;
                if file.stream_position()? + bytes.len() as u64 > MAX_JOURNAL {
                    return Err(invalid("metadata_journal_limit"));
                }
                file.write_all(&bytes)?;
            }
            file.sync_all()?;
            fault("ssd_checkpoint_synced");
            std::fs::rename(&temporary, &self.path)?;
            sync_parent(&self.path)?;
            fault("ssd_checkpoint_replaced");
            self.file = OpenOptions::new().read(true).write(true).open(&self.path)?;
            self.file.seek(io::SeekFrom::End(0))?;
            Ok(())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

pub(super) fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/persistence.rs"]
mod tests;
