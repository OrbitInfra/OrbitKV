use super::*;
use crate::transfer::local::SourceRange;

#[test]
fn completed_part_is_published_before_process_exit() {
    const CHILD_LOG: &str = "ORBITKV_TEST_LOCAL_COPY_CHILD_LOG";
    if let Some(path) = std::env::var_os(CHILD_LOG) {
        use std::io::Write;

        struct CaptureLogger(std::sync::Mutex<std::io::BufWriter<std::fs::File>>);

        impl log::Log for CaptureLogger {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }

            fn log(&self, record: &log::Record<'_>) {
                writeln!(self.0.lock().unwrap(), "{}", record.args()).unwrap();
            }

            fn flush(&self) {
                self.0.lock().unwrap().flush().unwrap();
            }
        }

        let file = std::fs::File::create(path).unwrap();
        log::set_logger(Box::leak(Box::new(CaptureLogger(std::sync::Mutex::new(
            std::io::BufWriter::with_capacity(1024 * 1024, file),
        )))))
        .unwrap();
        log::set_max_level(log::LevelFilter::Info);
        let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 100);
        let copies = copies();
        trace.begin_part(2, true);
        bindings(&mut trace, &copies);
        for copy in &copies {
            let index = trace.begin_batch(&[*copy], "direct").unwrap();
            trace.finish_batch(index, 17, true);
        }
        trace.finish_part(31, true);
        // A released engine can end the process without running destructors.
        std::process::exit(0);
    }

    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("capture.log");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "transfer::local::trace::tests::completed_part_is_published_before_process_exit",
        ])
        .env(CHILD_LOG, &path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let events: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line.strip_prefix("local_raw_copy ").unwrap()).unwrap())
        .collect();
    assert_eq!(events.len(), 5);
    assert_eq!(events[0]["stage"], "local_raw_copy_open");
    assert_eq!(events[4]["stage"], "local_raw_copy_complete");
    assert_eq!(events[4]["complete"], true);
    assert_eq!(events[4]["descriptors"], 2);
}

fn bindings(trace: &mut LocalCopyTrace, copies: &[CopyDesc]) {
    let tensor = LocalTensor {
        name: "layer".into(),
        address: 0x10000,
        bytes: 65536,
        allocation: 7,
    };
    for copy in copies {
        let raw = RawCopy {
            source: SourceRange {
                arena_id: 2,
                allocation_id: 3,
                allocation_offset: 256,
                allocation_size: 16384,
                offset: copy.host as u64 - 0x100000,
                size: copy.size as u64,
            },
            layer: "layer".into(),
            destination_offset: copy.device - tensor.address,
        };
        trace.record_geometry(&raw, copy, &tensor, 65536);
    }
}

fn copies() -> [CopyDesc; 2] {
    [0, 4096].map(|offset| CopyDesc {
        device: 0x10000 + offset,
        host: (0x100100 + offset) as *mut u8,
        host_device: 0x200100 + offset,
        size: 4096,
        device_allocation: 7,
        host_registration: 0x100000,
    })
}

#[test]
fn captures_actual_submitted_order_and_one_shared_part_drain() {
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 100);
    let mut values = Vec::new();
    let copies = copies();
    trace.begin_part_with(2, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    bindings(&mut trace, &copies);
    for copy in copies.iter().rev() {
        let index = trace.begin_batch(&[*copy], "direct").unwrap();
        trace.finish_batch(index, 17, true);
    }
    trace.finish_part_with(31, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    assert_eq!(values.len(), 5);
    assert_eq!(values[1]["copies"][0][8], 4096);
    assert_eq!(values[2]["copies"][0][14], 256);
    assert_eq!(values[1]["copies"][0][13], 16384);
    assert_eq!(values[1]["copies"][0][7], 65536);
    assert_eq!(values[1]["dma_ranges"], 1);
    assert_eq!(
        values[1],
        serde_json::json!({
            "schema": 3,
            "pid": std::process::id(),
            "trace_id": trace.id,
            "device": 0,
            "sequence": 0,
            "clock": "CLOCK_MONOTONIC",
            "stage": "local_raw_copy_batch",
            "part_id": 0,
            "batch_id": 0,
            "backend": "direct",
            "direction": "h2d",
            "bytes": 4096,
            "descriptors": 1,
            "dma_ranges": 1,
            "zero_descriptors": 0,
            "min_bytes": 4096,
            "max_bytes": 4096,
            "size_histogram": [0, 1, 0, 0, 0, 0, 0, 0],
            "cpu_submit_ns": 17,
            "tid": values[1]["tid"],
            "begin_ns": values[1]["begin_ns"],
            "submitted_ns": values[1]["submitted_ns"],
            "success": true,
            "copies": [[
                0x11000, 0x101100, 0x201100, 4096, 7, 0x100000,
                0x10000, 65536, 4096, 2, 65536, 3, 256, 16384, 4352,
            ]],
        })
    );
    assert!(values[4].get("invalid_reason").unwrap().is_null());
    assert_eq!(values[0]["clock"], "CLOCK_MONOTONIC");
    assert_eq!(values[1]["tid"], values[2]["tid"]);
    for batch in &values[1..3] {
        let begin = batch["begin_ns"].as_u64().unwrap();
        let submitted = batch["submitted_ns"].as_u64().unwrap();
        let drained = values[3]["drained_ns"].as_u64().unwrap();
        assert!(batch["tid"].as_u64().unwrap() > 0);
        assert!(begin > 0 && begin <= submitted && submitted <= drained);
    }
    assert!(values[1].get("host_drain_ns").is_none());
    assert_eq!(values[3]["host_drain_ns"], 31);
    assert_eq!(values[4]["descriptors"], 2);
    assert_eq!(values[4]["complete"], true);
}

#[test]
fn partial_failure_overflow_missing_submission_and_missing_drain_are_invalid() {
    for mode in ["failure", "overflow", "lost_batch", "no_drain", "no_pitch"] {
        let pitch = if mode == "no_pitch" {
            Err("driver".into())
        } else {
            Ok(usize::MAX)
        };
        let mut trace = LocalCopyTrace::new(0, pitch, if mode == "overflow" { 4 } else { 100 });
        let mut values = Vec::new();
        let copies = copies();
        trace.begin_part_with(2, true, |event| {
            values.push(serde_json::to_value(event).unwrap());
        });
        bindings(&mut trace, &copies);
        if let Some(index) = trace.begin_batch(&copies[..1], "direct") {
            trace.finish_batch(index, 17, mode != "failure");
        }
        if mode != "lost_batch"
            && let Some(index) = trace.begin_batch(&copies[1..], "direct")
        {
            trace.finish_batch(index, 17, true);
        }
        if mode != "no_drain" {
            trace.finish_part_with(31, mode != "failure", |event| {
                values.push(serde_json::to_value(event).unwrap());
            });
            assert_eq!(values.last().unwrap()["complete"], false, "{mode}");
        } else {
            assert_eq!(values.len(), 1);
            assert_eq!(values[0]["stage"], "local_raw_copy_open");
        }
    }
}

#[test]
fn empty_managed_parts_are_preserved_without_inventing_h2d_bytes() {
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 2);
    let mut values = Vec::new();
    trace.begin_part_with(0, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    trace.finish_part_with(19, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    assert_eq!(values.len(), 3);
    assert_eq!(values[1]["planned_descriptors"], 0);
    assert_eq!(values[2]["batches"], 0);
    assert_eq!(values[2]["complete"], true);
}

#[test]
fn publishing_a_part_does_not_replenish_the_executor_budget() {
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 10);
    let mut values = Vec::new();
    let copies = copies();
    for sequence in 0..3 {
        let start = values.len();
        trace.begin_part_with(2, true, |event| {
            values.push(serde_json::to_value(event).unwrap());
        });
        bindings(&mut trace, &copies);
        if let Some(index) = trace.begin_batch(&copies, "direct") {
            trace.finish_batch(index, 17, true);
        }
        trace.finish_part_with(31, true, |event| {
            values.push(serde_json::to_value(event).unwrap());
        });
        assert_eq!(values[start]["sequence"], sequence);
        assert_eq!(values[start]["records_before"], (sequence * 5).min(10));
        assert_eq!(values.last().unwrap()["complete"], sequence < 2);
        assert!(trace.part.is_none());
    }
    assert_eq!(
        values.last().unwrap()["invalid_reason"],
        "capacity_exceeded"
    );
    let count = values.len();
    trace.begin_part_with(0, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    trace.finish_part_with(31, true, |event| {
        values.push(serde_json::to_value(event).unwrap());
    });
    assert_eq!(values.len(), count);
}

#[test]
fn compact_geometry_preserves_unsigned_values_and_declares_every_field() {
    let geometry = Geometry {
        device: u64::MAX,
        host: u64::MAX - 1,
        host_device: u64::MAX - 2,
        size: usize::MAX,
        device_allocation: usize::MAX - 1,
        host_registration: usize::MAX - 2,
        registered_device_start: u64::MAX - 3,
        registered_device_bytes: usize::MAX - 3,
        destination_offset: u64::MAX - 4,
        arena_id: u64::MAX - 5,
        arena_bytes: usize::MAX - 4,
        source_allocation_id: u64::MAX - 6,
        source_allocation_offset: u64::MAX - 7,
        source_allocation_bytes: u64::MAX - 8,
        source_offset: u64::MAX - 9,
    };
    let encoded = serde_json::to_string(&geometry).unwrap();
    let decoded: Vec<u64> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        decoded,
        [
            u64::MAX,
            u64::MAX - 1,
            u64::MAX - 2,
            usize::MAX as u64,
            (usize::MAX - 1) as u64,
            (usize::MAX - 2) as u64,
            u64::MAX - 3,
            (usize::MAX - 3) as u64,
            u64::MAX - 4,
            u64::MAX - 5,
            (usize::MAX - 4) as u64,
            u64::MAX - 6,
            u64::MAX - 7,
            u64::MAX - 8,
            u64::MAX - 9,
        ]
    );
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 100);
    trace.begin_part_with(0, true, |event| {
        let header = serde_json::to_value(event).unwrap();
        assert_eq!(header["schema"], 3);
        assert_eq!(
            header["copy_fields"],
            serde_json::json!([
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
            ])
        );
    });
}
