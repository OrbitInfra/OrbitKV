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
}

struct Part {
    planned_descriptors: usize,
    final_part: bool,
    batches: Vec<Batch>,
    host_drain_ns: Option<u64>,
    success: bool,
}

/// Retains bounded scalar observations until the owning executor has drained.
/// JSON and logging happen when that executor is dropped, outside submission.
pub(super) struct LocalCopyTrace {
    id: u64,
    device: usize,
    max_pitch: usize,
    limit: usize,
    used: usize,
    invalid: Option<&'static str>,
    parts: Vec<Part>,
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
            parts: Vec::new(),
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
        if self
            .parts
            .last()
            .is_some_and(|part| part.host_drain_ns.is_none())
        {
            self.invalid = Some("unfinished_previous_part");
        }
        if !self.admit(planned_descriptors.saturating_mul(2).saturating_add(1)) {
            return;
        }
        self.geometry.clear();
        if self.parts.try_reserve(1).is_err()
            || self.geometry.try_reserve(planned_descriptors).is_err()
        {
            self.invalid = Some("capture_allocation_failed");
            return;
        }
        self.parts.push(Part {
            planned_descriptors,
            final_part,
            batches: Vec::new(),
            host_drain_ns: None,
            success: false,
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
        let Some(part) = self.parts.last_mut() else {
            self.invalid = Some("submission_part_missing");
            return None;
        };
        if part.batches.try_reserve(1).is_err() {
            self.invalid = Some("capture_allocation_failed");
            return None;
        }
        let index = part.batches.len();
        part.batches.push(Batch { geometry, shape });
        Some(index)
    }

    pub(super) fn finish_batch(&mut self, index: usize, cpu_submit_ns: u64, success: bool) {
        if let Some(batch) = self
            .parts
            .last_mut()
            .and_then(|part| part.batches.get_mut(index))
        {
            batch.shape.cpu_submit_ns = cpu_submit_ns;
            batch.shape.success = success;
        } else {
            self.invalid = Some("submission_part_missing");
        }
    }

    pub(super) fn finish_part(&mut self, host_drain_ns: u64, success: bool) {
        self.geometry.clear();
        if self.invalid.is_some() {
            return;
        }
        if let Some(part) = self.parts.last_mut() {
            part.host_drain_ns = Some(host_drain_ns);
            part.success = success;
        } else {
            self.invalid = Some("drain_part_missing");
        }
    }

    fn flush_with(&self, mut write: impl FnMut(serde_json::Value)) {
        let pid = std::process::id();
        let base = serde_json::json!({
            "schema": 1, "pid": pid, "trace_id": self.id, "device": self.device,
        });
        let mut header = base.clone();
        header["stage"] = "local_raw_copy_open".into();
        header["max_pitch"] = self.max_pitch.into();
        header["limit"] = self.limit.into();
        write(header);
        let mut batches = 0;
        let mut descriptors = 0;
        let mut successful = true;
        for (part_id, part) in self.parts.iter().enumerate() {
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
                event["success"] = shape.success.into();
                event["copies"] = serde_json::json!(batch.geometry);
                write(event);
                successful &= shape.success;
                batches += 1;
                descriptors += batch.geometry.len();
            }
            let mut event = base.clone();
            event["stage"] = "local_raw_copy_part".into();
            event["part_id"] = part_id.into();
            event["final_part"] = part.final_part.into();
            event["planned_descriptors"] = part.planned_descriptors.into();
            event["submitted_descriptors"] = submitted.into();
            event["batches"] = part.batches.len().into();
            event["host_drain_ns"] = part.host_drain_ns.into();
            event["success"] = part.success.into();
            write(event);
            successful &= part.success
                && part.host_drain_ns.is_some()
                && submitted == part.planned_descriptors;
        }
        let mut complete = base;
        complete["stage"] = "local_raw_copy_complete".into();
        complete["parts"] = self.parts.len().into();
        complete["batches"] = batches.into();
        complete["descriptors"] = descriptors.into();
        complete["records_reserved"] = self.used.into();
        complete["invalid_reason"] = self.invalid.into();
        complete["complete"] = (self.invalid.is_none() && successful).into();
        write(complete);
    }

    pub(super) fn flush(&self) {
        self.flush_with(|fields| log::info!("local_raw_copy {fields}"));
        log::logger().flush();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/local/trace.rs"]
mod tests;
