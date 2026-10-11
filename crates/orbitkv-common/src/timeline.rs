//! Opt-in stage observations; durations are measured within one process.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock};

const MAX_DIAGNOSTIC_EVENTS: u64 = 65_536;

pub static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("ORBITKV_TRACE_TRANSFERS").is_ok_and(|value| value == "1"));

static DIAGNOSTIC_LIMIT: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
        .min(MAX_DIAGNOSTIC_EVENTS)
});
static BUFFER_START_FAILED: AtomicBool = AtomicBool::new(false);
static BUFFER_START_FAILURE_REPORTED: AtomicBool = AtomicBool::new(false);

const TIMELINE_SEALED: u64 = 1 << 63;
const TIMELINE_COUNT: u64 = !TIMELINE_SEALED;

const HAS_BLOCKS: u16 = 1 << 0;
const HAS_BYTES: u16 = 1 << 1;
const HAS_GROUP: u16 = 1 << 2;
const HAS_PENDING_BLOCKS: u16 = 1 << 3;
const HAS_INFLIGHT_WRITES: u16 = 1 << 4;
const HAS_MAX_INFLIGHT_WRITES: u16 = 1 << 5;
const HAS_ELAPSED_US: u16 = 1 << 6;

#[derive(Clone, Copy, Debug, Default)]
pub struct DiagnosticFields {
    request_id: u64,
    session_epoch: u64,
    session_token: u64,
    blocks: u64,
    bytes: u64,
    group: u64,
    pending_blocks: u64,
    inflight_writes: u64,
    max_inflight_writes: u64,
    elapsed_us: u64,
    present: u16,
    success: Option<bool>,
    outcome: Option<&'static str>,
}

impl DiagnosticFields {
    pub const fn operation(request_id: u64, session_epoch: u64, session_token: u64) -> Self {
        Self {
            request_id,
            session_epoch,
            session_token,
            blocks: 0,
            bytes: 0,
            group: 0,
            pending_blocks: 0,
            inflight_writes: 0,
            max_inflight_writes: 0,
            elapsed_us: 0,
            present: 0,
            success: None,
            outcome: None,
        }
    }

    pub const fn with_operation(
        mut self,
        request_id: u64,
        session_epoch: u64,
        session_token: u64,
    ) -> Self {
        self.request_id = request_id;
        self.session_epoch = session_epoch;
        self.session_token = session_token;
        self
    }

    pub const fn blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks as u64;
        self.present |= HAS_BLOCKS;
        self
    }

    pub const fn bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self.present |= HAS_BYTES;
        self
    }

    pub const fn group(mut self, group: u32) -> Self {
        self.group = group as u64;
        self.present |= HAS_GROUP;
        self
    }

    pub const fn pending_blocks(mut self, pending_blocks: usize) -> Self {
        self.pending_blocks = pending_blocks as u64;
        self.present |= HAS_PENDING_BLOCKS;
        self
    }

    pub const fn inflight_writes(mut self, inflight_writes: usize) -> Self {
        self.inflight_writes = inflight_writes as u64;
        self.present |= HAS_INFLIGHT_WRITES;
        self
    }

    pub const fn max_inflight_writes(mut self, max_inflight_writes: usize) -> Self {
        self.max_inflight_writes = max_inflight_writes as u64;
        self.present |= HAS_MAX_INFLIGHT_WRITES;
        self
    }

    pub const fn elapsed_us(mut self, elapsed_us: u64) -> Self {
        self.elapsed_us = elapsed_us;
        self.present |= HAS_ELAPSED_US;
        self
    }

    pub const fn success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    pub const fn outcome(mut self, outcome: &'static str) -> Self {
        self.outcome = Some(outcome);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RestoreTimelineFields {
    manager_epoch: u64,
    client_token: u64,
    operation_id: u64,
    elapsed_us: Option<u64>,
    elapsed_ns: Option<u64>,
    success: Option<bool>,
    drain_timing: Option<[u64; 6]>,
}

impl RestoreTimelineFields {
    pub const fn operation(manager_epoch: u64, client_token: u64, operation_id: u64) -> Self {
        Self {
            manager_epoch,
            client_token,
            operation_id,
            elapsed_us: None,
            elapsed_ns: None,
            success: None,
            drain_timing: None,
        }
    }

    pub const fn elapsed_us(mut self, elapsed_us: u64) -> Self {
        self.elapsed_us = Some(elapsed_us);
        self
    }

    pub const fn elapsed_ns(mut self, elapsed_ns: u64) -> Self {
        self.elapsed_ns = Some(elapsed_ns);
        self
    }

    pub const fn success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }

    pub const fn drain_timing(
        mut self,
        readiness_ns: u64,
        dispatched_ns: u64,
        dequeued_ns: u64,
        claimed_ns: u64,
        submitted_ns: u64,
        drained_ns: u64,
    ) -> Self {
        self.drain_timing = Some([
            readiness_ns,
            dispatched_ns,
            dequeued_ns,
            claimed_ns,
            submitted_ns,
            drained_ns,
        ]);
        self
    }
}

struct DiagnosticEvent {
    index: u64,
    stage: &'static str,
    pid: u32,
    monotonic_ns: u64,
    fields: DiagnosticFields,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RawCopyTimelineFields {
    pub device: u64,
    pub backend: &'static str,
    pub direction: &'static str,
    pub bytes: u64,
    pub descriptors: u64,
    pub dma_ranges: u64,
    pub zero_descriptors: u64,
    pub min_bytes: u64,
    pub max_bytes: u64,
    /// Counts for <=256 B, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, larger.
    pub size_histogram: [u64; 8],
    pub cpu_submit_ns: u64,
    pub host_drain_ns: u64,
    pub success: bool,
}

struct RawCopyEvent {
    index: u64,
    pid: u32,
    monotonic_ns: u64,
    fields: RawCopyTimelineFields,
}

const FIXED_TEXT_BYTES: usize = 96;

struct FixedText {
    bytes: [u8; FIXED_TEXT_BYTES],
    len: u8,
    truncated: bool,
}

impl FixedText {
    fn new(value: &str) -> Self {
        let mut bytes = [0; FIXED_TEXT_BYTES];
        let len = value.len().min(FIXED_TEXT_BYTES);
        bytes[..len].copy_from_slice(&value.as_bytes()[..len]);
        Self {
            bytes,
            len: len as u8,
            truncated: value.len() > len,
        }
    }

    fn as_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.bytes[..self.len as usize])
    }
}

struct QueryPathEvent {
    stage: &'static str,
    pid: u32,
    monotonic_ns: u64,
    request_id: FixedText,
    instance_id: FixedText,
    group_id: u32,
    warmup: bool,
    prepare: bool,
    elapsed_us: u64,
    hit_blocks: usize,
}

struct RestoreEvent {
    stage: &'static str,
    pid: u32,
    at_unix_ns: u64,
    monotonic_ns: u64,
    fields: RestoreTimelineFields,
}

struct QueryControlEvent {
    stage: &'static str,
    pid: u32,
    at_unix_ns: u64,
    monotonic_ns: u64,
    request_id: String,
    operation_id: u64,
    revision: u64,
}

enum BufferedEvent {
    Diagnostic(DiagnosticEvent),
    QueryPath(QueryPathEvent),
    Restore(RestoreEvent),
    QueryControl(QueryControlEvent),
    RawCopy(RawCopyEvent),
}

struct BufferedTimeline {
    slots: Box<[OnceLock<BufferedEvent>]>,
    state: AtomicU64,
    overflow: AtomicBool,
    late_records: AtomicU64,
}

#[derive(Debug, PartialEq, Eq)]
struct FlushSummary {
    reserved: u64,
    ready: u64,
    overflow: bool,
    late_records: u64,
}

impl BufferedTimeline {
    fn start(limit: u64) -> Option<Self> {
        let capacity = usize::try_from(limit).ok()?;
        if capacity == 0 {
            return None;
        }
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity).ok()?;
        slots.resize_with(capacity, OnceLock::new);
        Some(Self {
            slots: slots.into_boxed_slice(),
            state: AtomicU64::new(0),
            overflow: AtomicBool::new(false),
            late_records: AtomicU64::new(0),
        })
    }

    fn reserve(&self) -> Option<u64> {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & TIMELINE_SEALED != 0 {
                self.late_records.fetch_add(1, Ordering::Release);
                return None;
            }
            let index = state & TIMELINE_COUNT;
            if index >= self.slots.len() as u64 {
                self.overflow.store(true, Ordering::Relaxed);
                return None;
            }
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(index),
                Err(observed) => state = observed,
            }
        }
    }

    fn write(&self, index: u64, event: BufferedEvent) {
        if self.slots[index as usize].set(event).is_err() {
            self.overflow.store(true, Ordering::Relaxed);
        }
    }

    fn record_diagnostic(&self, stage: &'static str, fields: DiagnosticFields) {
        if let Some(index) = self.reserve() {
            self.write(
                index,
                BufferedEvent::Diagnostic(DiagnosticEvent {
                    index,
                    stage,
                    pid: std::process::id(),
                    monotonic_ns: monotonic_ns(),
                    fields,
                }),
            );
        }
    }

    fn record_restore(&self, stage: &'static str, fields: RestoreTimelineFields) {
        if let Some(index) = self.reserve() {
            self.write(
                index,
                BufferedEvent::Restore(RestoreEvent {
                    stage,
                    pid: std::process::id(),
                    at_unix_ns: unix_ns(),
                    monotonic_ns: monotonic_ns(),
                    fields,
                }),
            );
        }
    }

    fn record_raw_copy(&self, fields: RawCopyTimelineFields) {
        if let Some(index) = self.reserve() {
            self.write(
                index,
                BufferedEvent::RawCopy(RawCopyEvent {
                    index,
                    pid: std::process::id(),
                    monotonic_ns: monotonic_ns(),
                    fields,
                }),
            );
        }
    }

    fn record_query_control(
        &self,
        stage: &'static str,
        request_id: String,
        operation_id: u64,
        revision: u64,
    ) {
        if let Some(index) = self.reserve() {
            self.write(
                index,
                BufferedEvent::QueryControl(QueryControlEvent {
                    stage,
                    pid: std::process::id(),
                    at_unix_ns: unix_ns(),
                    monotonic_ns: monotonic_ns(),
                    request_id,
                    operation_id,
                    revision,
                }),
            );
        }
    }

    #[allow(clippy::too_many_arguments, reason = "fixed diagnostic event schema")]
    fn record_query_path(
        &self,
        stage: &'static str,
        request_id: &str,
        instance_id: &str,
        group_id: u32,
        warmup: bool,
        prepare: bool,
        elapsed_us: u64,
        hit_blocks: usize,
    ) {
        if let Some(index) = self.reserve() {
            self.write(
                index,
                BufferedEvent::QueryPath(QueryPathEvent {
                    stage,
                    pid: std::process::id(),
                    monotonic_ns: monotonic_ns(),
                    request_id: FixedText::new(request_id),
                    instance_id: FixedText::new(instance_id),
                    group_id,
                    warmup,
                    prepare,
                    elapsed_us,
                    hit_blocks,
                }),
            );
        }
    }

    fn flush_with(&self, mut write: impl FnMut(serde_json::Value)) -> Option<FlushSummary> {
        let state = self.state.fetch_or(TIMELINE_SEALED, Ordering::AcqRel);
        let first_flush = state & TIMELINE_SEALED == 0;
        let reserved = state & TIMELINE_COUNT;
        let mut ready = reserved;
        if first_flush {
            ready = 0;
            for slot in &self.slots[..reserved as usize] {
                if let Some(event) = slot.get() {
                    ready += 1;
                    write(buffered_event_json(event));
                }
            }
        }
        let overflow = first_flush && self.overflow.swap(false, Ordering::AcqRel);
        let late_records = self.late_records.swap(0, Ordering::AcqRel);
        if !first_flush && late_records == 0 {
            return None;
        }
        if overflow || ready != reserved || late_records != 0 {
            let reason = match (overflow, ready != reserved, late_records != 0) {
                (_, true, _) => "slot_not_ready",
                (true, false, _) => "capacity_exceeded",
                (false, false, true) => "post_seal_record",
                (false, false, false) => unreachable!(),
            };
            write(diagnostic_limit_event(
                reason,
                self.slots.len() as u64,
                reserved,
                ready,
                late_records,
            ));
        }
        Some(FlushSummary {
            reserved,
            ready,
            overflow,
            late_records,
        })
    }

    fn flush(&self) {
        if let Some(summary) = self.flush_with(|fields| log::info!("cache_timeline {fields}")) {
            if !summary.overflow && summary.ready == summary.reserved && summary.late_records == 0 {
                log::info!("cache_timeline {}", diagnostic_complete_event(&summary));
            }
            log::logger().flush();
        }
    }
}

static BUFFERED_TIMELINE: LazyLock<Option<BufferedTimeline>> = LazyLock::new(|| {
    if !*ENABLED || *DIAGNOSTIC_LIMIT == 0 {
        return None;
    }
    let timeline = BufferedTimeline::start(*DIAGNOSTIC_LIMIT);
    if timeline.is_none() {
        BUFFER_START_FAILED.store(true, Ordering::Relaxed);
    }
    timeline
});

fn report_buffer_start_failure() {
    if BUFFER_START_FAILED.load(Ordering::Relaxed)
        && !BUFFER_START_FAILURE_REPORTED.swap(true, Ordering::Relaxed)
    {
        let event = diagnostic_limit_event("buffer_allocation_failed", *DIAGNOSTIC_LIMIT, 0, 0, 0);
        log::info!("cache_timeline {event}");
    }
}

fn buffered_event_json(event: &BufferedEvent) -> serde_json::Value {
    match event {
        BufferedEvent::Diagnostic(event) => diagnostic_event_json(event),
        BufferedEvent::QueryPath(event) => query_path_event_json(event),
        BufferedEvent::Restore(event) => restore_event_json(event),
        BufferedEvent::QueryControl(event) => query_control_event_json(event),
        BufferedEvent::RawCopy(event) => raw_copy_event_json(event),
    }
}

fn raw_copy_event_json(event: &RawCopyEvent) -> serde_json::Value {
    let fields = &event.fields;
    serde_json::json!({
        "stage": "manager_raw_copy_complete",
        "diagnostic_event": event.index,
        "pid": event.pid,
        "monotonic_ns": event.monotonic_ns,
        "device": fields.device,
        "backend": fields.backend,
        "direction": fields.direction,
        "bytes": fields.bytes,
        "descriptors": fields.descriptors,
        "dma_ranges": fields.dma_ranges,
        "zero_descriptors": fields.zero_descriptors,
        "min_bytes": fields.min_bytes,
        "max_bytes": fields.max_bytes,
        "size_histogram": fields.size_histogram,
        "cpu_submit_ns": fields.cpu_submit_ns,
        "host_drain_ns": fields.host_drain_ns,
        "success": fields.success,
    })
}

fn diagnostic_complete_event(summary: &FlushSummary) -> serde_json::Value {
    serde_json::json!({
        "stage": "diagnostic_timeline_complete",
        "pid": std::process::id(),
        "monotonic_ns": monotonic_ns(),
        "reserved": summary.reserved,
        "ready": summary.ready,
    })
}

fn diagnostic_event_json(event: &DiagnosticEvent) -> serde_json::Value {
    let mut value = serde_json::json!({
        "diagnostic_event": event.index,
        "stage": event.stage,
        "pid": event.pid,
        "monotonic_ns": event.monotonic_ns,
        "request_id": event.fields.request_id,
        "session_epoch": event.fields.session_epoch,
        "session_token": event.fields.session_token,
    });
    let optional = [
        (HAS_BLOCKS, "blocks", event.fields.blocks),
        (HAS_BYTES, "bytes", event.fields.bytes),
        (HAS_GROUP, "group", event.fields.group),
        (
            HAS_PENDING_BLOCKS,
            "pending_blocks",
            event.fields.pending_blocks,
        ),
        (
            HAS_INFLIGHT_WRITES,
            "inflight_writes",
            event.fields.inflight_writes,
        ),
        (
            HAS_MAX_INFLIGHT_WRITES,
            "max_inflight_writes",
            event.fields.max_inflight_writes,
        ),
        (HAS_ELAPSED_US, "elapsed_us", event.fields.elapsed_us),
    ];
    for (mask, name, field) in optional {
        if event.fields.present & mask != 0 {
            value[name] = field.into();
        }
    }
    if let Some(success) = event.fields.success {
        value["success"] = success.into();
    }
    if let Some(outcome) = event.fields.outcome {
        value["outcome"] = outcome.into();
    }
    value
}

fn query_path_event_json(event: &QueryPathEvent) -> serde_json::Value {
    serde_json::json!({
        "stage": event.stage,
        "pid": event.pid,
        "monotonic_ns": event.monotonic_ns,
        "request_id": event.request_id.as_str(),
        "request_id_truncated": event.request_id.truncated,
        "instance_id": event.instance_id.as_str(),
        "instance_id_truncated": event.instance_id.truncated,
        "group_id": event.group_id,
        "warmup": event.warmup,
        "prepare": event.prepare,
        "elapsed_us": event.elapsed_us,
        "hit_blocks": event.hit_blocks,
    })
}

fn restore_event_json(event: &RestoreEvent) -> serde_json::Value {
    let mut value = serde_json::json!({
        "stage": event.stage,
        "pid": event.pid,
        "at_unix_ns": event.at_unix_ns,
        "monotonic_ns": event.monotonic_ns,
        "restore_key": format!(
            "manager:{}:{}:{}",
            event.fields.manager_epoch, event.fields.client_token, event.fields.operation_id
        ),
    });
    if let Some(elapsed_us) = event.fields.elapsed_us {
        value["elapsed_us"] = elapsed_us.into();
    }
    if let Some(elapsed_ns) = event.fields.elapsed_ns {
        value["elapsed_ns"] = elapsed_ns.into();
    }
    if let Some(success) = event.fields.success {
        value["success"] = success.into();
    }
    if let Some(
        [
            readiness_ns,
            dispatched_ns,
            dequeued_ns,
            claimed_ns,
            submitted_ns,
            drained_ns,
        ],
    ) = event.fields.drain_timing
    {
        value["readiness_ns"] = readiness_ns.into();
        value["dispatched_ns"] = dispatched_ns.into();
        value["dequeued_ns"] = dequeued_ns.into();
        value["claimed_ns"] = claimed_ns.into();
        value["submitted_ns"] = submitted_ns.into();
        value["drained_ns"] = drained_ns.into();
    }
    value
}

fn query_control_event_json(event: &QueryControlEvent) -> serde_json::Value {
    serde_json::json!({
        "stage": event.stage,
        "pid": event.pid,
        "at_unix_ns": event.at_unix_ns,
        "monotonic_ns": event.monotonic_ns,
        "request_id": event.request_id,
        "operation_id": event.operation_id,
        "revision": event.revision,
    })
}

fn diagnostic_limit_event(
    reason: &'static str,
    limit: u64,
    reserved: u64,
    ready: u64,
    late_records: u64,
) -> serde_json::Value {
    serde_json::json!({
        "stage": "diagnostic_timeline_limit",
        "pid": std::process::id(),
        "monotonic_ns": monotonic_ns(),
        "reason": reason,
        "limit": limit,
        "reserved": reserved,
        "ready": ready,
        "late_records": late_records,
    })
}

pub fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(now.tv_sec)
        .unwrap_or(0)
        .saturating_mul(1_000_000_000)
        .saturating_add(u64::try_from(now.tv_nsec).unwrap_or(0))
}

fn unix_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

pub fn diagnostic_enabled() -> bool {
    *ENABLED && *DIAGNOSTIC_LIMIT > 0
}

pub fn record_restore(stage: &'static str, fields: RestoreTimelineFields) {
    if !*ENABLED {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_restore(stage, fields);
    } else if *DIAGNOSTIC_LIMIT == 0 {
        let event = RestoreEvent {
            stage,
            pid: std::process::id(),
            at_unix_ns: unix_ns(),
            monotonic_ns: monotonic_ns(),
            fields,
        };
        log::info!("cache_timeline {}", restore_event_json(&event));
    }
}

pub fn record_query_control(
    stage: &'static str,
    request_id: String,
    operation_id: u64,
    revision: u64,
) {
    if !*ENABLED {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_query_control(stage, request_id, operation_id, revision);
    } else if *DIAGNOSTIC_LIMIT == 0 {
        let event = QueryControlEvent {
            stage,
            pid: std::process::id(),
            at_unix_ns: unix_ns(),
            monotonic_ns: monotonic_ns(),
            request_id,
            operation_id,
            revision,
        };
        log::info!("cache_timeline {}", query_control_event_json(&event));
    }
}

/// Drain diagnostic events after request, storage and lifecycle owners stop.
pub fn flush_diagnostic() {
    if !diagnostic_enabled() {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.flush();
    } else if *DIAGNOSTIC_LIMIT > 0 {
        report_buffer_start_failure();
    }
}

pub fn record_diagnostic(stage: &'static str, fields: DiagnosticFields) {
    if !diagnostic_enabled() {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_diagnostic(stage, fields);
    }
}

pub fn record_raw_copy(fields: RawCopyTimelineFields) {
    if !diagnostic_enabled() {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_raw_copy(fields);
    }
}

#[allow(clippy::too_many_arguments, reason = "existing query timeline schema")]
pub fn record_query_path(
    stage: &'static str,
    request_id: &str,
    instance_id: &str,
    group_id: u32,
    warmup: bool,
    prepare: bool,
    elapsed_us: u64,
    hit_blocks: usize,
) {
    if !*ENABLED {
        return;
    }
    if let Some(timeline) = BUFFERED_TIMELINE.as_ref() {
        timeline.record_query_path(
            stage,
            request_id,
            instance_id,
            group_id,
            warmup,
            prepare,
            elapsed_us,
            hit_blocks,
        );
    } else if *DIAGNOSTIC_LIMIT == 0 {
        let fields = serde_json::json!({
            "stage": stage,
            "pid": std::process::id(),
            "at_unix_ns": unix_ns(),
            "monotonic_ns": monotonic_ns(),
            "request_id": request_id,
            "instance_id": instance_id,
            "group_id": group_id,
            "warmup": warmup,
            "prepare": prepare,
            "elapsed_us": elapsed_us,
            "hit_blocks": hit_blocks,
        });
        log::info!("cache_timeline {fields}");
    }
}

#[cfg(test)]
#[path = "../tests/unit/timeline.rs"]
mod tests;
