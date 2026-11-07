use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};
use std::time::Duration;

use orbitkv_state::{
    INVENTORY_BATCH_BYTES, INVENTORY_BATCH_RECORDS, InventoryRecord, InventoryScope, ReplicaMedium,
    ReplicaMetadata, StateKey,
};
use parking_lot::Mutex;
use tokio::sync::watch;

pub const DEFAULT_INVENTORY_JOURNAL_BYTES: usize = 16 * 1024 * 1024;

pub struct ResidencyInventory {
    state: Mutex<Inventory>,
    changed: watch::Sender<u64>,
    flush_requested: watch::Sender<u64>,
    publish_coalesce_window: Duration,
}

struct Inventory {
    residents: BTreeMap<(StateKey, ReplicaMedium), InventoryRecord>,
    journal: VecDeque<InventoryRecord>,
    sequence: u64,
    journal_bytes: usize,
    journal_bytes_peak: usize,
    byte_limit: usize,
    history_gaps: u64,
    last_change_mono_ns: u64,
    flush_through_sequence: u64,
    delta_input_records: u64,
    delta_input_bytes: u64,
    delta_output_records: u64,
    delta_frames: u64,
    delta_encoded_bytes: u64,
    coalescing_windows: u64,
    coalescing_wait_micros: u64,
    scope_filter_input_records: u64,
    scope_filter_output_records: u64,
    scope_filter_micros: u64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct InventoryStatus {
    pub sequence: u64,
    pub resident_records: usize,
    pub journal_records: usize,
    pub journal_bytes: usize,
    pub journal_bytes_peak: usize,
    pub journal_capacity_bytes: usize,
    pub history_gaps: u64,
    pub last_change_mono_ns: u64,
    pub flush_through_sequence: u64,
    pub delta_input_records: u64,
    pub delta_input_bytes: u64,
    pub delta_output_records: u64,
    pub delta_frames: u64,
    pub delta_encoded_bytes: u64,
    pub coalescing_windows: u64,
    pub coalescing_wait_micros: u64,
    pub scope_filter_input_records: u64,
    pub scope_filter_output_records: u64,
    pub scope_filter_micros: u64,
}

#[derive(Debug)]
pub struct InventoryDelta {
    pub through: u64,
    pub records: Vec<InventoryRecord>,
    pub input_records: usize,
    pub input_bytes: usize,
}

pub struct InventoryPage {
    pub records: Vec<InventoryRecord>,
    pub next: Option<(StateKey, ReplicaMedium)>,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryReadError {
    HistoryGap,
    RecordTooLarge,
    InvalidRecord,
    SequenceExhausted,
}

impl ResidencyInventory {
    pub fn new(byte_limit: usize) -> Self {
        Self {
            state: Mutex::new(Inventory {
                residents: BTreeMap::new(),
                journal: VecDeque::new(),
                sequence: 0,
                journal_bytes: 0,
                journal_bytes_peak: 0,
                byte_limit,
                history_gaps: 0,
                last_change_mono_ns: 0,
                flush_through_sequence: 0,
                delta_input_records: 0,
                delta_input_bytes: 0,
                delta_output_records: 0,
                delta_frames: 0,
                delta_encoded_bytes: 0,
                coalescing_windows: 0,
                coalescing_wait_micros: 0,
                scope_filter_input_records: 0,
                scope_filter_output_records: 0,
                scope_filter_micros: 0,
            }),
            changed: watch::channel(0).0,
            flush_requested: watch::channel(0).0,
            publish_coalesce_window: Duration::ZERO,
        }
    }

    pub fn with_publish_coalescing(
        byte_limit: usize,
        publish_coalesce_window: Duration,
    ) -> Result<Self, String> {
        if publish_coalesce_window > Duration::from_millis(5) {
            return Err("inventory publication coalescing cannot exceed 5 ms".into());
        }
        let mut inventory = Self::new(byte_limit);
        inventory.publish_coalesce_window = publish_coalesce_window;
        Ok(inventory)
    }

    pub(crate) fn change(
        &self,
        key: &StateKey,
        medium: ReplicaMedium,
        metadata: Option<ReplicaMetadata>,
    ) -> Result<(), InventoryReadError> {
        assert!(matches!(medium, ReplicaMedium::Dram | ReplicaMedium::Ssd));
        assert!(metadata.is_none_or(|value| value.medium == medium));
        let mut state = self.state.lock();
        let identity = (key.clone(), medium);
        let old = state
            .residents
            .get(&identity)
            .and_then(|record| record.metadata);
        if old == metadata {
            return Ok(());
        }
        state.sequence = state
            .sequence
            .checked_add(1)
            .ok_or(InventoryReadError::SequenceExhausted)?;
        let record = InventoryRecord {
            key: key.clone(),
            sequence: state.sequence,
            present: metadata.is_some(),
            metadata: metadata.or(old),
        };
        if record.present {
            state.residents.insert(identity, record.clone());
        } else {
            state.residents.remove(&identity);
        }
        state.journal_bytes += record.estimated_size();
        state.journal.push_back(record);
        while state.journal_bytes > state.byte_limit {
            if let Some(record) = state.journal.pop_front() {
                state.journal_bytes -= record.estimated_size();
            }
        }
        state.journal_bytes_peak = state.journal_bytes_peak.max(state.journal_bytes);
        state.last_change_mono_ns = monotonic_ns();
        self.changed.send_replace(state.sequence);
        Ok(())
    }

    #[cfg(feature = "test-hooks")]
    pub fn test_change(
        &self,
        key: &StateKey,
        medium: ReplicaMedium,
        metadata: Option<ReplicaMetadata>,
    ) -> Result<(), InventoryReadError> {
        self.change(key, medium, metadata)
    }

    pub fn sequence(&self) -> u64 {
        self.state.lock().sequence
    }

    pub fn status(&self) -> InventoryStatus {
        let state = self.state.lock();
        InventoryStatus {
            sequence: state.sequence,
            resident_records: state.residents.len(),
            journal_records: state.journal.len(),
            journal_bytes: state.journal_bytes,
            journal_bytes_peak: state.journal_bytes_peak,
            journal_capacity_bytes: state.byte_limit,
            history_gaps: state.history_gaps,
            last_change_mono_ns: state.last_change_mono_ns,
            flush_through_sequence: state.flush_through_sequence,
            delta_input_records: state.delta_input_records,
            delta_input_bytes: state.delta_input_bytes,
            delta_output_records: state.delta_output_records,
            delta_frames: state.delta_frames,
            delta_encoded_bytes: state.delta_encoded_bytes,
            coalescing_windows: state.coalescing_windows,
            coalescing_wait_micros: state.coalescing_wait_micros,
            scope_filter_input_records: state.scope_filter_input_records,
            scope_filter_output_records: state.scope_filter_output_records,
            scope_filter_micros: state.scope_filter_micros,
        }
    }

    pub fn changed(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    pub async fn wait_to_publish(&self, after: u64) -> Duration {
        let window = self.publish_coalesce_window;
        let mut change = self.changed.subscribe();
        let mut flush = self.flush_requested.subscribe();
        if window.is_zero() || *flush.borrow() > after {
            return Duration::ZERO;
        }
        let started = tokio::time::Instant::now();
        let maximum = started + Duration::from_millis(5);
        let mut quiet = (started + window).min(maximum);
        loop {
            if *flush.borrow() > after {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep_until(quiet) => break,
                _ = tokio::time::sleep_until(maximum) => break,
                result = flush.changed() => {
                    if result.is_err() || *flush.borrow() > after {
                        break;
                    }
                },
                result = change.changed() => {
                    if result.is_err() {
                        break;
                    }
                    quiet = (tokio::time::Instant::now() + window).min(maximum);
                },
            }
        }
        started.elapsed()
    }

    pub fn page(
        &self,
        after: Option<&(StateKey, ReplicaMedium)>,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        self.scoped_page(after, &InventoryScope::AllNamespaces)
            .map(|page| page.records)
    }

    pub fn scoped_page(
        &self,
        after: Option<&(StateKey, ReplicaMedium)>,
        scope: &InventoryScope,
    ) -> Result<InventoryPage, InventoryReadError> {
        let started = std::time::Instant::now();
        let mut state = self.state.lock();
        if state.sequence == u64::MAX {
            return Err(InventoryReadError::SequenceExhausted);
        }
        let (records, next, complete, input_records) = {
            let mut scanned = 0usize;
            let mut scanned_bytes = 0usize;
            let mut records = Vec::new();
            let mut next = None;
            let mut source = state
                .residents
                .range((after.map_or(Unbounded, Excluded), Unbounded))
                .peekable();
            while let Some((identity, record)) = source.peek() {
                let bytes = record.estimated_size();
                if scanned == INVENTORY_BATCH_RECORDS
                    || (scanned > 0 && scanned_bytes + bytes > INVENTORY_BATCH_BYTES)
                {
                    break;
                }
                if scanned == 0 && bytes > INVENTORY_BATCH_BYTES {
                    return Err(InventoryReadError::RecordTooLarge);
                }
                let identity = (*identity).clone();
                let record = (*record).clone();
                source.next();
                scanned += 1;
                scanned_bytes += bytes;
                next = Some(identity);
                if scope.contains(&record.key.namespace) {
                    records.push(record);
                }
            }
            (records, next, source.peek().is_none(), scanned)
        };
        state.scope_filter_input_records = state
            .scope_filter_input_records
            .saturating_add(u64::try_from(input_records).unwrap_or(u64::MAX));
        state.scope_filter_output_records = state
            .scope_filter_output_records
            .saturating_add(u64::try_from(records.len()).unwrap_or(u64::MAX));
        state.scope_filter_micros = state
            .scope_filter_micros
            .saturating_add(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
        Ok(InventoryPage {
            records,
            next,
            complete,
        })
    }

    pub fn changes(
        &self,
        after: u64,
        through: u64,
    ) -> Result<Vec<InventoryRecord>, InventoryReadError> {
        let mut state = self.state.lock();
        if state.sequence == u64::MAX {
            return Err(InventoryReadError::SequenceExhausted);
        }
        if after > through
            || through > state.sequence
            || (after != state.sequence
                && state
                    .journal
                    .front()
                    .is_none_or(|record| record.sequence > after + 1))
        {
            state.history_gaps = state.history_gaps.saturating_add(1);
            return Err(InventoryReadError::HistoryGap);
        }
        if after == through {
            return Ok(Vec::new());
        }
        let first = state
            .journal
            .front()
            .ok_or(InventoryReadError::HistoryGap)?
            .sequence;
        let start =
            usize::try_from(after + 1 - first).map_err(|_| InventoryReadError::HistoryGap)?;
        let end =
            usize::try_from(through - first + 1).map_err(|_| InventoryReadError::HistoryGap)?;
        bounded_records(state.journal.range(start..end).cloned())
    }

    pub fn coalesced_changes(
        &self,
        after: u64,
        through: u64,
    ) -> Result<InventoryDelta, InventoryReadError> {
        self.coalesced_changes_scoped(after, through, &InventoryScope::AllNamespaces)
    }

    pub fn coalesced_changes_scoped(
        &self,
        after: u64,
        through: u64,
        scope: &InventoryScope,
    ) -> Result<InventoryDelta, InventoryReadError> {
        let started = std::time::Instant::now();
        let input = self.changes(after, through)?;
        let through = input.last().map_or(after, |record| record.sequence);
        let input_records = input.len();
        let input_bytes = input.iter().map(InventoryRecord::estimated_size).sum();
        let mut records = BTreeMap::new();
        for record in input {
            let Some(metadata) = record.metadata else {
                return Err(InventoryReadError::InvalidRecord);
            };
            if scope.contains(&record.key.namespace) {
                records.insert((record.key.clone(), metadata.medium), record);
            }
        }
        let output_records = records.len();
        let mut state = self.state.lock();
        state.scope_filter_input_records = state
            .scope_filter_input_records
            .saturating_add(u64::try_from(input_records).unwrap_or(u64::MAX));
        state.scope_filter_output_records = state
            .scope_filter_output_records
            .saturating_add(u64::try_from(output_records).unwrap_or(u64::MAX));
        state.scope_filter_micros = state
            .scope_filter_micros
            .saturating_add(u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX));
        drop(state);
        Ok(InventoryDelta {
            through,
            records: records.into_values().collect(),
            input_records,
            input_bytes,
        })
    }

    pub fn record_delta_stream(
        &self,
        input_records: usize,
        input_bytes: usize,
        output_records: usize,
        frames: usize,
        encoded_bytes: usize,
        coalescing_wait: Duration,
    ) {
        let mut state = self.state.lock();
        let input_records = u64::try_from(input_records).unwrap_or(u64::MAX);
        let input_bytes = u64::try_from(input_bytes).unwrap_or(u64::MAX);
        let output_records = u64::try_from(output_records).unwrap_or(u64::MAX);
        let frames = u64::try_from(frames).unwrap_or(u64::MAX);
        let encoded_bytes = u64::try_from(encoded_bytes).unwrap_or(u64::MAX);
        let coalescing_wait = u64::try_from(coalescing_wait.as_micros()).unwrap_or(u64::MAX);
        state.delta_input_records = state.delta_input_records.saturating_add(input_records);
        state.delta_input_bytes = state.delta_input_bytes.saturating_add(input_bytes);
        state.delta_output_records = state.delta_output_records.saturating_add(output_records);
        state.delta_frames = state.delta_frames.saturating_add(frames);
        state.delta_encoded_bytes = state.delta_encoded_bytes.saturating_add(encoded_bytes);
        state.coalescing_windows = state.coalescing_windows.saturating_add(1);
        state.coalescing_wait_micros = state.coalescing_wait_micros.saturating_add(coalescing_wait);
    }

    pub(crate) fn contains_record(&self, record: &InventoryRecord, medium: ReplicaMedium) -> bool {
        record.present
            && self
                .state
                .lock()
                .residents
                .get(&(record.key.clone(), medium))
                .is_some_and(|resident| resident.sequence == record.sequence)
    }

    pub fn request_flush(&self, target: u64) {
        let target = {
            let mut state = self.state.lock();
            state.flush_through_sequence = state.flush_through_sequence.max(target);
            state.flush_through_sequence
        };
        self.flush_requested.send_replace(target);
        self.changed.send_replace(self.sequence());
    }

    pub fn capture_fence(&self) -> u64 {
        let target = self.sequence();
        self.request_flush(target);
        target
    }
}

fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid writable timespec and CLOCK_MONOTONIC has no
    // additional pointer lifetime requirements.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) } != 0 {
        return 0;
    }
    u64::try_from(now.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(now.tv_nsec).unwrap_or(0))
}

fn bounded_records(
    records: impl Iterator<Item = InventoryRecord>,
) -> Result<Vec<InventoryRecord>, InventoryReadError> {
    let mut result = Vec::new();
    let mut bytes = 0;
    for record in records {
        let size = record.estimated_size();
        if size > INVENTORY_BATCH_BYTES {
            return Err(InventoryReadError::RecordTooLarge);
        }
        if result.len() == INVENTORY_BATCH_RECORDS || bytes + size > INVENTORY_BATCH_BYTES {
            break;
        }
        bytes += size;
        result.push(record);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../../tests/unit/storage/inventory.rs"]
mod tests;
