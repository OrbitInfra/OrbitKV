use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use super::{CopyDesc, LocalTensor, RawCopy};

static NEXT_TRACE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Serialize)]
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

    fn base(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": 2, "pid": std::process::id(), "trace_id": self.id,
            "device": self.device, "sequence": self.parts,
            "clock": "CLOCK_MONOTONIC",
        })
    }

    fn begin_part_with(
        &mut self,
        planned_descriptors: usize,
        final_part: bool,
        mut write: impl FnMut(serde_json::Value),
    ) {
        if self.closed {
            return;
        }
        if self.part.is_some() {
            self.invalid = Some("unfinished_previous_part");
            return;
        }
        let mut header = self.base();
        header["stage"] = "local_raw_copy_open".into();
        header["max_pitch"] = self.max_pitch.into();
        header["limit"] = self.limit.into();
        header["records_before"] = self.used.into();
        write(header);
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
        mut write: impl FnMut(serde_json::Value),
    ) {
        let Some(part) = self.part.take() else {
            return;
        };
        let drained_ns = orbitkv_common::timeline::monotonic_ns();
        let base = self.base();
        let part_id = self.parts;
        let mut successful = success;
        {
            let submitted: usize = part.batches.iter().map(|batch| batch.geometry.len()).sum();
            for (batch_id, batch) in part.batches.iter().enumerate() {
                let shape = &batch.shape;
                let mut event = base.clone();
                event["stage"] = "local_raw_copy_batch".into();
                event["part_id"] = part_id.into();
                event["batch_id"] = batch_id.into();
                event["backend"] = shape.backend.into();
                event["direction"] = shape.direction.into();
                event["bytes"] = shape.bytes.into();
                event["descriptors"] = shape.descriptors.into();
                event["dma_ranges"] = shape.dma_ranges.into();
                event["zero_descriptors"] = shape.zero_descriptors.into();
                event["min_bytes"] = shape.min_bytes.into();
                event["max_bytes"] = shape.max_bytes.into();
                event["size_histogram"] = serde_json::json!(shape.size_histogram);
                event["cpu_submit_ns"] = shape.cpu_submit_ns.into();
                event["tid"] = batch.tid.into();
                event["begin_ns"] = batch.begin_ns.into();
                event["submitted_ns"] = batch.submitted_ns.into();
                event["success"] = shape.success.into();
                event["copies"] = serde_json::json!(batch.geometry);
                write(event);
                successful &= shape.success;
                self.batches += 1;
                self.descriptors += batch.geometry.len();
            }
            let mut event = base.clone();
            event["stage"] = "local_raw_copy_part".into();
            event["part_id"] = part_id.into();
            event["final_part"] = part.final_part.into();
            event["planned_descriptors"] = part.planned_descriptors.into();
            event["submitted_descriptors"] = submitted.into();
            event["batches"] = part.batches.len().into();
            event["host_drain_ns"] = host_drain_ns.into();
            event["drained_ns"] = drained_ns.into();
            event["success"] = success.into();
            write(event);
            successful &= submitted == part.planned_descriptors;
        }
        if !successful && self.invalid.is_none() {
            self.invalid = Some("incomplete_or_failed_part");
        }
        let mut complete = base;
        complete["stage"] = "local_raw_copy_complete".into();
        complete["parts"] = (self.parts + 1).into();
        complete["batches"] = self.batches.into();
        complete["descriptors"] = self.descriptors.into();
        complete["records_reserved"] = self.used.into();
        complete["invalid_reason"] = self.invalid.into();
        complete["complete"] = (self.invalid.is_none() && successful).into();
        write(complete);
        self.parts += 1;
        self.closed = self.invalid.is_some();
        self.geometry.clear();
    }

    fn write(fields: serde_json::Value) {
        log::info!("local_raw_copy {fields}");
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/local/trace.rs"]
mod tests;
