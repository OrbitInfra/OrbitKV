use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::display::Base64Display;
use base64::engine::general_purpose::STANDARD;
use serde::Serialize;

use super::{CopyDesc, LocalTensor, RawCopy};

static NEXT_TRACE: AtomicU64 = AtomicU64::new(1);

const COPY_FIELDS: [&str; 15] = [
    "device",
    "host",
    "host_device",
    "size",
    "device_allocation",
    "host_registration",
    "registered_device_start",
    "registered_device_bytes",
    "destination_offset",
    "arena_id",
    "arena_bytes",
    "source_allocation_id",
    "source_allocation_offset",
    "source_allocation_bytes",
    "source_offset",
];

#[derive(Clone)]
struct Geometry {
    device: u64,
    host: u64,
    host_device: u64,
    size: usize,
    device_allocation: usize,
    host_registration: usize,
    registered_device_start: u64,
    registered_device_bytes: usize,
    destination_offset: u64,
    arena_id: u64,
    arena_bytes: usize,
    source_allocation_id: u64,
    source_allocation_offset: u64,
    source_allocation_bytes: u64,
    source_offset: u64,
}

impl Geometry {
    fn values(&self) -> [u64; 15] {
        [
            self.device,
            self.host,
            self.host_device,
            self.size as u64,
            self.device_allocation as u64,
            self.host_registration as u64,
            self.registered_device_start,
            self.registered_device_bytes as u64,
            self.destination_offset,
            self.arena_id,
            self.arena_bytes as u64,
            self.source_allocation_id,
            self.source_allocation_offset,
            self.source_allocation_bytes,
            self.source_offset,
        ]
    }
}

fn serialize_geometry<S: serde::Serializer>(
    geometry: &[Geometry],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let Some(capacity) = geometry.len().checked_mul(COPY_FIELDS.len() * 10) else {
        return Err(serde::ser::Error::custom("geometry frame is too large"));
    };
    let mut data = Vec::new();
    data.try_reserve_exact(capacity)
        .map_err(serde::ser::Error::custom)?;
    for row in geometry {
        for mut value in row.values() {
            while value >= 128 {
                data.push((value as u8 & 127) | 128);
                value >>= 7;
            }
            data.push(value as u8);
        }
    }
    serializer.collect_str(&Base64Display::new(&data, &STANDARD))
}

struct Batch {
    geometry: Vec<Geometry>,
    shape: orbitkv_common::timeline::RawCopyTimelineFields,
    tid: u64,
    begin_ns: u64,
    submitted_ns: u64,
}

struct Part {
    planned_descriptors: usize,
    final_part: bool,
    batches: Vec<Batch>,
}

#[derive(Clone, Copy, Serialize)]
struct TraceIdentity {
    schema: u8,
    pid: u32,
    trace_id: u64,
    device: usize,
    sequence: usize,
    clock: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "stage")]
enum TraceRecord<'a> {
    #[serde(rename = "local_raw_copy_open")]
    Open {
        #[serde(flatten)]
        identity: TraceIdentity,
        max_pitch: usize,
        limit: usize,
        records_before: usize,
        copy_fields: &'static [&'static str; 15],
    },
    #[serde(rename = "local_raw_copy_batch")]
    Batch {
        #[serde(flatten)]
        identity: TraceIdentity,
        part_id: usize,
        batch_id: usize,
        backend: &'static str,
        direction: &'static str,
        bytes: u64,
        descriptors: u64,
        dma_ranges: u64,
        zero_descriptors: u64,
        min_bytes: u64,
        max_bytes: u64,
        size_histogram: &'a [u64; 8],
        cpu_submit_ns: u64,
        tid: u64,
        begin_ns: u64,
        submitted_ns: u64,
        success: bool,
        copies_encoding: &'static str,
        copy_rows: usize,
        #[serde(serialize_with = "serialize_geometry")]
        copies: &'a [Geometry],
    },
    #[serde(rename = "local_raw_copy_part")]
    Part {
        #[serde(flatten)]
        identity: TraceIdentity,
        part_id: usize,
        final_part: bool,
        planned_descriptors: usize,
        submitted_descriptors: usize,
        batches: usize,
        host_drain_ns: u64,
        drained_ns: u64,
        success: bool,
    },
    #[serde(rename = "local_raw_copy_complete")]
    Complete {
        #[serde(flatten)]
        identity: TraceIdentity,
        parts: usize,
        batches: usize,
        descriptors: usize,
        records_reserved: usize,
        invalid_reason: Option<&'static str>,
        complete: bool,
    },
}

/// Publishes each physically drained part under one cumulative executor budget.
/// Opening records precede submission; geometry serialization follows drain.
pub(super) struct LocalCopyTrace {
    id: u64,
    device: usize,
    max_pitch: usize,
    limit: usize,
    used: usize,
    invalid: Option<&'static str>,
    parts: usize,
    batches: usize,
    descriptors: usize,
    closed: bool,
    part: Option<Part>,
    geometry: HashMap<u64, Geometry>,
}

impl LocalCopyTrace {
    pub(super) fn new(device: usize, max_pitch: Result<usize, String>, limit: usize) -> Self {
        let (max_pitch, invalid) = match max_pitch {
            Ok(pitch) => (pitch, None),
            Err(_) => (0, Some("maximum_pitch_unavailable")),
        };
        Self {
            id: NEXT_TRACE.fetch_add(1, Ordering::Relaxed),
            device,
            max_pitch,
            limit,
            used: 0,
            invalid,
            parts: 0,
            batches: 0,
            descriptors: 0,
            closed: false,
            part: None,
            geometry: HashMap::new(),
        }
    }

    fn admit(&mut self, records: usize) -> bool {
        if self.invalid.is_some() {
            return false;
        }
        if self
            .used
            .checked_add(records)
            .is_none_or(|count| count > self.limit)
        {
            self.invalid = Some("capacity_exceeded");
            return false;
        }
        self.used += records;
        true
    }

    pub(super) fn begin_part(&mut self, planned_descriptors: usize, final_part: bool) {
        self.begin_part_with(planned_descriptors, final_part, Self::write);
    }

    fn identity(&self) -> TraceIdentity {
        TraceIdentity {
            schema: 4,
            pid: std::process::id(),
            trace_id: self.id,
            device: self.device,
            sequence: self.parts,
            clock: "CLOCK_MONOTONIC",
        }
    }

    fn begin_part_with(
        &mut self,
        planned_descriptors: usize,
        final_part: bool,
        mut write: impl FnMut(TraceRecord<'_>),
    ) {
        if self.closed {
            return;
        }
        if self.part.is_some() {
            self.invalid = Some("unfinished_previous_part");
            return;
        }
        write(TraceRecord::Open {
            identity: self.identity(),
            max_pitch: self.max_pitch,
            limit: self.limit,
            records_before: self.used,
            copy_fields: &COPY_FIELDS,
        });
        self.geometry.clear();
        if self.admit(planned_descriptors.saturating_mul(2).saturating_add(1))
            && self.geometry.try_reserve(planned_descriptors).is_err()
        {
            self.invalid = Some("capture_allocation_failed");
        }
        self.part = Some(Part {
            planned_descriptors,
            final_part,
            batches: Vec::new(),
        });
    }

    pub(super) fn record_geometry(
        &mut self,
        raw: &RawCopy,
        copy: &CopyDesc,
        tensor: &LocalTensor,
        arena_bytes: usize,
    ) {
        if self.invalid.is_some() {
            return;
        }
        self.geometry.insert(
            copy.device,
            Geometry {
                device: copy.device,
                host: copy.host as u64,
                host_device: copy.host_device,
                size: copy.size,
                device_allocation: copy.device_allocation,
                host_registration: copy.host_registration,
                registered_device_start: tensor.address,
                registered_device_bytes: tensor.bytes,
                destination_offset: raw.destination_offset,
                arena_id: raw.source.arena_id,
                arena_bytes,
                source_allocation_id: raw.source.allocation_id,
                source_allocation_offset: raw.source.allocation_offset,
                source_allocation_bytes: raw.source.allocation_size,
                source_offset: raw.source.offset,
            },
        );
    }

    pub(super) fn begin_batch(
        &mut self,
        copies: &[CopyDesc],
        backend: &'static str,
    ) -> Option<usize> {
        if self.invalid.is_some() || copies.is_empty() {
            return None;
        }
        let mut geometry = Vec::new();
        if geometry.try_reserve_exact(copies.len()).is_err() {
            self.invalid = Some("capture_allocation_failed");
            return None;
        }
        for copy in copies {
            let Some(bounds) = self.geometry.get(&copy.device) else {
                self.invalid = Some("submission_geometry_missing");
                return None;
            };
            geometry.push(bounds.clone());
        }
        let mut shape = crate::transfer::raw_copy_shape(copies, self.max_pitch);
        shape.device = self.device as u64;
        shape.backend = backend;
        shape.direction = "h2d";
        let Some(part) = self.part.as_mut() else {
            self.invalid = Some("submission_part_missing");
            return None;
        };
        if part.batches.try_reserve(1).is_err() {
            self.invalid = Some("capture_allocation_failed");
            return None;
        }
        let index = part.batches.len();
        part.batches.push(Batch {
            geometry,
            shape,
            // SAFETY: gettid only observes the current Linux thread identity.
            tid: unsafe { libc::syscall(libc::SYS_gettid) } as u64,
            begin_ns: orbitkv_common::timeline::monotonic_ns(),
            submitted_ns: 0,
        });
        Some(index)
    }

    pub(super) fn finish_batch(&mut self, index: usize, cpu_submit_ns: u64, success: bool) {
        if let Some(batch) = self
            .part
            .as_mut()
            .and_then(|part| part.batches.get_mut(index))
        {
            batch.submitted_ns = orbitkv_common::timeline::monotonic_ns();
            batch.shape.cpu_submit_ns = cpu_submit_ns;
            batch.shape.success = success;
        } else {
            self.invalid = Some("submission_part_missing");
        }
    }

    pub(super) fn finish_part(&mut self, host_drain_ns: u64, success: bool) {
        if self.part.is_none() {
            return;
        }
        self.finish_part_with(host_drain_ns, success, Self::write);
        log::logger().flush();
    }

    fn finish_part_with(
        &mut self,
        host_drain_ns: u64,
        success: bool,
        mut write: impl FnMut(TraceRecord<'_>),
    ) {
        let Some(part) = self.part.take() else {
            return;
        };
        let drained_ns = orbitkv_common::timeline::monotonic_ns();
        let identity = self.identity();
        let part_id = self.parts;
        let mut successful = success;
        {
            let submitted: usize = part.batches.iter().map(|batch| batch.geometry.len()).sum();
            for (batch_id, batch) in part.batches.iter().enumerate() {
                let shape = &batch.shape;
                write(TraceRecord::Batch {
                    identity,
                    part_id,
                    batch_id,
                    backend: shape.backend,
                    direction: shape.direction,
                    bytes: shape.bytes,
                    descriptors: shape.descriptors,
                    dma_ranges: shape.dma_ranges,
                    zero_descriptors: shape.zero_descriptors,
                    min_bytes: shape.min_bytes,
                    max_bytes: shape.max_bytes,
                    size_histogram: &shape.size_histogram,
                    cpu_submit_ns: shape.cpu_submit_ns,
                    tid: batch.tid,
                    begin_ns: batch.begin_ns,
                    submitted_ns: batch.submitted_ns,
                    success: shape.success,
                    copies_encoding: "uvarint-base64-v1",
                    copy_rows: batch.geometry.len(),
                    copies: &batch.geometry,
                });
                successful &= shape.success;
                self.batches += 1;
                self.descriptors += batch.geometry.len();
            }
            write(TraceRecord::Part {
                identity,
                part_id,
                final_part: part.final_part,
                planned_descriptors: part.planned_descriptors,
                submitted_descriptors: submitted,
                batches: part.batches.len(),
                host_drain_ns,
                drained_ns,
                success,
            });
            successful &= submitted == part.planned_descriptors;
        }
        if !successful && self.invalid.is_none() {
            self.invalid = Some("incomplete_or_failed_part");
        }
        write(TraceRecord::Complete {
            identity,
            parts: self.parts + 1,
            batches: self.batches,
            descriptors: self.descriptors,
            records_reserved: self.used,
            invalid_reason: self.invalid,
            complete: self.invalid.is_none() && successful,
        });
        self.parts += 1;
        self.closed = self.invalid.is_some();
        self.geometry.clear();
    }

    fn write(record: TraceRecord<'_>) {
        match serde_json::to_string(&record) {
            Ok(fields) => log::info!("local_raw_copy {fields}"),
            Err(error) => log::error!("Cannot serialize local raw-copy capture: {error}"),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/local/trace.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../benches/native_copy_trace.rs"]
mod profile;
