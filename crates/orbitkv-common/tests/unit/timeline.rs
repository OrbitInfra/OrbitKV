use std::sync::Arc;

use super::{
    BufferedTimeline, DiagnosticEvent, DiagnosticFields, FlushSummary, QueryControlEvent,
    RawCopyEvent, RawCopyTimelineFields, RestoreEvent, RestoreTimelineFields,
};

#[test]
fn raw_copy_event_is_fixed_bounded_and_preserves_shape_and_cpu_boundaries() {
    assert!(std::mem::size_of::<RawCopyEvent>() <= std::mem::size_of::<super::QueryPathEvent>());
    let event = RawCopyEvent {
        index: 0,
        pid: 11,
        monotonic_ns: 13,
        fields: RawCopyTimelineFields {
            device: 7,
            backend: "direct",
            direction: "d2h",
            bytes: 8192,
            descriptors: 2,
            dma_ranges: 1,
            min_bytes: 4096,
            max_bytes: 4096,
            size_histogram: [0, 2, 0, 0, 0, 0, 0, 0],
            cpu_submit_ns: 17,
            host_drain_ns: 19,
            success: true,
            ..Default::default()
        },
    };
    let expected = serde_json::json!({
        "stage": "manager_raw_copy_complete", "diagnostic_event": 0,
        "pid": 11, "monotonic_ns": 13, "device": 7,
        "backend": "direct", "direction": "d2h", "bytes": 8192,
        "descriptors": 2, "dma_ranges": 1, "zero_descriptors": 0,
        "min_bytes": 4096, "max_bytes": 4096,
        "size_histogram": [0, 2, 0, 0, 0, 0, 0, 0],
        "cpu_submit_ns": 17, "host_drain_ns": 19, "success": true,
    });
    assert_eq!(super::raw_copy_event_json(&event), expected);
    let timeline = BufferedTimeline::start(1).unwrap();
    timeline.record_raw_copy(event.fields);
    timeline.record_diagnostic("overflow", DiagnosticFields::default());
    let mut records = Vec::new();
    let summary = timeline.flush_with(|event| records.push(event)).unwrap();
    assert!(summary.overflow);
    assert_eq!(records[0]["stage"], "manager_raw_copy_complete");
    assert_eq!(records[1]["stage"], "diagnostic_timeline_limit");
    assert_eq!(super::diagnostic_complete_event(&summary)["reserved"], 1);
}

#[test]
fn monotonic_clock_advances() {
    let before = super::monotonic_ns();
    let after = super::monotonic_ns();
    assert!(before > 0);
    assert!(after >= before);
}

#[test]
fn fixed_diagnostic_event_preserves_stage_correlation_and_queue_fields() {
    let event = DiagnosticEvent {
        index: 7,
        stage: "publish_ssd_dequeue",
        pid: 11,
        monotonic_ns: 13,
        fields: DiagnosticFields::operation(17, 19, 23)
            .blocks(8)
            .pending_blocks(3)
            .inflight_writes(2)
            .max_inflight_writes(4),
    };
    assert_eq!(
        super::diagnostic_event_json(&event),
        serde_json::json!({
            "diagnostic_event": 7,
            "stage": "publish_ssd_dequeue",
            "pid": 11,
            "monotonic_ns": 13,
            "request_id": 17,
            "session_epoch": 19,
            "session_token": 23,
            "blocks": 8,
            "pending_blocks": 3,
            "inflight_writes": 2,
            "max_inflight_writes": 4,
        })
    );
}

#[test]
fn fixed_restore_event_preserves_existing_timing_schema() {
    let event = RestoreEvent {
        stage: "local_restore_complete",
        pid: 11,
        at_unix_ns: 12,
        monotonic_ns: 13,
        fields: RestoreTimelineFields::operation(17, 19, 23)
            .success(true)
            .drain_timing(29, 31, 37, 41, 43, 47),
    };
    assert_eq!(
        super::restore_event_json(&event),
        serde_json::json!({
            "stage": "local_restore_complete",
            "pid": 11,
            "at_unix_ns": 12,
            "monotonic_ns": 13,
            "restore_key": "manager:17:19:23",
            "success": true,
            "readiness_ns": 29,
            "dispatched_ns": 31,
            "dequeued_ns": 37,
            "claimed_ns": 41,
            "submitted_ns": 43,
            "drained_ns": 47,
        })
    );

    let observed = RestoreEvent {
        stage: "local_restore_observed",
        pid: 53,
        at_unix_ns: 59,
        monotonic_ns: 61,
        fields: RestoreTimelineFields::operation(67, 71, 73)
            .elapsed_ns(79)
            .success(false),
    };
    assert_eq!(
        super::restore_event_json(&observed),
        serde_json::json!({
            "stage": "local_restore_observed",
            "pid": 53,
            "at_unix_ns": 59,
            "monotonic_ns": 61,
            "restore_key": "manager:67:71:73",
            "elapsed_ns": 79,
            "success": false,
        })
    );
}

#[test]
fn zero_limit_does_not_allocate_buffer() {
    assert!(BufferedTimeline::start(0).is_none());
}

#[test]
fn concurrent_writers_claim_each_slot_once_and_flush_once() {
    const THREADS: u64 = 8;
    const EVENTS_PER_THREAD: u64 = 128;
    let timeline = Arc::new(BufferedTimeline::start(THREADS * EVENTS_PER_THREAD).unwrap());
    let mut writers = Vec::new();
    for thread in 0..THREADS {
        let timeline = Arc::clone(&timeline);
        writers.push(std::thread::spawn(move || {
            for event in 0..EVENTS_PER_THREAD {
                timeline.record_diagnostic(
                    "concurrent",
                    DiagnosticFields::operation(thread * EVENTS_PER_THREAD + event, thread, event),
                );
            }
        }));
    }
    for writer in writers {
        writer.join().unwrap();
    }

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: THREADS * EVENTS_PER_THREAD,
            ready: THREADS * EVENTS_PER_THREAD,
            overflow: false,
            late_records: 0,
        })
    );
    let mut indices = events
        .iter()
        .map(|event| event["diagnostic_event"].as_u64().unwrap())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    assert_eq!(
        indices,
        (0..THREADS * EVENTS_PER_THREAD).collect::<Vec<_>>()
    );

    assert_eq!(timeline.flush_with(|event| events.push(event)), None);
    assert_eq!(events.len() as u64, THREADS * EVENTS_PER_THREAD);
}

#[test]
fn overflow_emits_invalidating_limit_event() {
    let timeline = BufferedTimeline::start(1).unwrap();
    timeline.record_diagnostic("first", DiagnosticFields::default());
    timeline.record_diagnostic("overflow", DiagnosticFields::default());

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 1,
            ready: 1,
            overflow: true,
            late_records: 0,
        })
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["stage"], "diagnostic_timeline_limit");
    assert_eq!(events[1]["reason"], "capacity_exceeded");
}

#[test]
fn incomplete_slot_emits_invalidating_limit_event() {
    let timeline = BufferedTimeline::start(2).unwrap();
    assert_eq!(timeline.reserve(), Some(0));
    timeline.record_diagnostic("ready", DiagnosticFields::default());

    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 2,
            ready: 1,
            overflow: false,
            late_records: 0,
        })
    );
    assert_eq!(events.len(), 2);
    assert_eq!(events[1]["stage"], "diagnostic_timeline_limit");
    assert_eq!(events[1]["reason"], "slot_not_ready");
    assert_eq!(events[1]["reserved"], 2);
    assert_eq!(events[1]["ready"], 1);
}

#[test]
fn post_seal_record_is_reported_by_followup_flush() {
    let timeline = BufferedTimeline::start(1).unwrap();
    let mut events = Vec::new();
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 0,
            ready: 0,
            overflow: false,
            late_records: 0,
        })
    );
    timeline.record_diagnostic("late", DiagnosticFields::default());
    assert_eq!(
        timeline.flush_with(|event| events.push(event)),
        Some(FlushSummary {
            reserved: 0,
            ready: 0,
            overflow: false,
            late_records: 1,
        })
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["stage"], "diagnostic_timeline_limit");
    assert_eq!(events[0]["reason"], "post_seal_record");
    assert_eq!(events[0]["late_records"], 1);
}

#[test]
fn query_control_preserves_complete_request_id() {
    let request_id = "r".repeat(256);
    let event = QueryControlEvent {
        stage: "query_cancel",
        pid: 11,
        at_unix_ns: 12,
        monotonic_ns: 13,
        request_id: request_id.clone(),
        operation_id: 17,
        revision: 19,
    };
    let value = super::query_control_event_json(&event);
    assert_eq!(value["request_id"], request_id);
    assert!(value.get("request_id_truncated").is_none());
}
