use base64::Engine;

use super::*;
use crate::transfer::local::SourceRange;

fn decoded_geometry(encoded: &serde_json::Value, rows: usize) -> serde_json::Value {
    fn integer(data: &[u8], cursor: &mut usize) -> u64 {
        let mut value = 0_u64;
        for digit in 0..10 {
            let byte = data[*cursor];
            *cursor += 1;
            assert!(digit != 9 || byte <= 1);
            value |= (byte as u64 & 127) << (digit * 7);
            if byte < 128 {
                assert!(digit == 0 || byte != 0);
                return value;
            }
        }
        panic!("unterminated integer");
    }

    let data = STANDARD.decode(encoded.as_str().unwrap()).unwrap();
    let mut cursor = 0;
    let mut decoded = vec![[0_u64; COPY_FIELDS.len()]; rows];
    for column in 0..COPY_FIELDS.len() {
        let mut next = 0;
        let mut previous = None;
        while next < rows {
            let count = usize::try_from(integer(&data, &mut cursor)).unwrap();
            assert!(count > 0 && count <= rows - next);
            let value = integer(&data, &mut cursor);
            assert_ne!(previous, Some(value));
            for row in &mut decoded[next..next + count] {
                row[column] = value;
            }
            next += count;
            previous = Some(value);
        }
    }
    assert_eq!(cursor, data.len());
    serde_json::json!(decoded)
}

fn record(event: TraceRecord<'_>) -> serde_json::Value {
    let mut value = serde_json::to_value(event).unwrap();
    if value["stage"] == "local_raw_copy_batch" {
        assert_eq!(value["copies_encoding"], "uvarint-rle-base64-v1");
        assert_eq!(value["copy_rows"], value["descriptors"]);
        value["copies"] = decoded_geometry(
            &value["copies"],
            value["copy_rows"].as_u64().unwrap() as usize,
        );
    }
    value
}

#[test]
fn completed_part_is_published_before_process_exit() {
    const CHILD_LOG: &str = "ORBITKV_TEST_LOCAL_COPY_CHILD_LOG";
    const CHILD_LOGGER: &str = "ORBITKV_TEST_LOCAL_COPY_CHILD_LOGGER";
    if let Some(path) = std::env::var_os(CHILD_LOG) {
        use std::io::Write;

        struct CaptureLogger {
            output: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
            messages: std::sync::Mutex<Vec<(usize, String, String)>>,
        }

        impl log::Log for CaptureLogger {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool {
                true
            }

            fn log(&self, record: &log::Record<'_>) {
                let fields = record.args().to_string();
                self.messages.lock().unwrap().push((
                    fields.len(),
                    record.target().into(),
                    record.level().to_string(),
                ));
                writeln!(self.output.lock().unwrap(), "{fields}").unwrap();
            }

            fn flush(&self) {
                self.output.lock().unwrap().flush().unwrap();
            }
        }

        let path = std::path::PathBuf::from(path);
        let logger = if std::env::var(CHILD_LOGGER).as_deref() == Ok("default") {
            orbitkv_common::logging::init_stderr("info");
            None
        } else {
            let file = std::fs::File::create(&path).unwrap();
            let logger: &'static CaptureLogger = Box::leak(Box::new(CaptureLogger {
                output: std::sync::Mutex::new(std::io::BufWriter::with_capacity(1024 * 1024, file)),
                messages: std::sync::Mutex::new(Vec::new()),
            }));
            log::set_logger(logger).unwrap();
            log::set_max_level(log::LevelFilter::Info);
            Some(logger)
        };
        let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 16384);
        let large = |count: u64, size: u64| {
            (0..count)
                .map(|index| CopyDesc {
                    device: 0x10000 + index * size,
                    host: (0x100100 + index * size) as *mut u8,
                    host_device: 0x200100 + index * size,
                    size: size as usize,
                    device_allocation: 7,
                    host_registration: 0x100000,
                })
                .collect::<Vec<_>>()
        };
        let parts = [
            large(256, 64),
            Vec::new(),
            copies().to_vec(),
            large(4096, 4),
        ];
        for (sequence, copies) in parts.iter().enumerate() {
            trace.begin_part(copies.len(), true);
            bindings(&mut trace, copies);
            if sequence == 3 {
                for batch in [&copies[..1], &copies[1..2], &copies[2..]] {
                    let index = trace.begin_batch(batch, "direct").unwrap();
                    trace.finish_batch(index, 17, true);
                }
            } else {
                for copy in copies {
                    let index = trace.begin_batch(&[*copy], "direct").unwrap();
                    trace.finish_batch(index, 17, true);
                }
            }
            trace.finish_part(31, true);
            assert_eq!(trace.publication_len, 0);
            assert_eq!(trace.publication.as_ref().unwrap().len(), PUBLICATION_BYTES);
            assert!(trace.invalid.is_none());
        }
        // An invalid final part must also publish its terminal tail.
        trace.begin_part(16384, true);
        trace.finish_part(31, true);
        assert_eq!(trace.publication_len, 0);
        assert!(trace.closed);
        if let Some(logger) = logger {
            let messages = serde_json::to_vec(&*logger.messages.lock().unwrap()).unwrap();
            std::fs::write(path.with_extension("messages.json"), messages).unwrap();
        }
        // A released engine can end the process without running destructors.
        std::process::exit(0);
    }

    for mode in ["buffered", "default"] {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(format!("{mode}.log"));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transfer::local::trace::tests::completed_part_is_published_before_process_exit",
            ])
            .env(CHILD_LOG, &path)
            .env(CHILD_LOGGER, mode)
            .env("RUST_LOG", "info")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        if mode == "default" {
            std::fs::write(&path, &output.stderr).unwrap();
        }
        let events: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .filter_map(|line| line.split_once(RECORD_TAG))
            .map(|(_, fields)| serde_json::from_str(fields).unwrap())
            .collect();
        if mode == "buffered" {
            let messages: Vec<(usize, String, String)> = serde_json::from_slice(
                &std::fs::read(path.with_extension("messages.json")).unwrap(),
            )
            .unwrap();
            assert!(messages.len() < events.len());
            assert_eq!(
                messages
                    .iter()
                    .filter(|message| message.0 > PUBLICATION_BYTES)
                    .count(),
                1
            );
            assert!(messages.iter().all(|(_, target, level)| target
                == "orbitkv_core::transfer::local::trace"
                && level == "INFO"));
        }
        let mut start = 0;
        for (sequence, (descriptors, batches, reserved)) in [
            (256, 256, 513),
            (256, 256, 514),
            (258, 258, 519),
            (4354, 261, 8712),
            (4354, 261, 8712),
        ]
        .into_iter()
        .enumerate()
        {
            let end = events[start..]
                .iter()
                .position(|event| event["stage"] == "local_raw_copy_complete")
                .unwrap()
                + start;
            let part = &events[start..=end];
            assert_eq!(part[0]["stage"], "local_raw_copy_open");
            assert_eq!(part[0]["sequence"], sequence);
            assert_eq!(part[part.len() - 2]["stage"], "local_raw_copy_part");
            let footer = part.last().unwrap();
            assert_eq!(footer["parts"], sequence + 1);
            assert_eq!(footer["descriptors"], descriptors);
            assert_eq!(footer["batches"], batches);
            assert_eq!(footer["records_reserved"], reserved);
            assert_eq!(footer["complete"], sequence < 4);
            if sequence == 4 {
                assert_eq!(footer["invalid_reason"], "capacity_exceeded");
            }
            for (index, batch) in part[1..part.len() - 2].iter().enumerate() {
                assert_eq!(batch["batch_id"], index);
                let rows = batch["copy_rows"].as_u64().unwrap() as usize;
                let geometry = decoded_geometry(&batch["copies"], rows);
                for (offset, row) in geometry.as_array().unwrap().iter().enumerate() {
                    let (copy_index, size) = match sequence {
                        0 => (index, 64),
                        2 => (index * 1024, 4),
                        3 => (if index == 2 { offset + 2 } else { index }, 4),
                        _ => unreachable!(),
                    };
                    let stride = copy_index * size;
                    assert_eq!(
                        row,
                        &serde_json::json!([
                            0x10000 + stride,
                            0x100100 + stride,
                            0x200100 + stride,
                            if sequence == 2 { 4096 } else { size },
                            7,
                            0x100000,
                            0x10000,
                            65536,
                            stride,
                            2,
                            65536,
                            3,
                            256,
                            16384,
                            256 + stride,
                        ])
                    );
                }
            }
            start = end + 1;
        }
        assert_eq!(start, events.len());
        assert_eq!(events.len(), 276);
        assert_eq!(events[270]["copy_rows"], 4094);
    }
}

#[test]
fn publication_capacity_exact_fit_overflow_and_terminal_tails() {
    for frame_bytes in [
        PUBLICATION_BYTES - 1,
        PUBLICATION_BYTES,
        PUBLICATION_BYTES + 1,
    ] {
        let fields = format!("\"{}\"", "x".repeat(frame_bytes - RECORD_TAG.len() - 2));
        let mut publication = None;
        let mut used = 0;
        assert!(LocalCopyTrace::publish_fields(
            &fields,
            false,
            &mut publication,
            &mut used
        ));
        assert_eq!(
            used,
            if frame_bytes > PUBLICATION_BYTES {
                0
            } else {
                frame_bytes
            }
        );
        assert!(LocalCopyTrace::publish_fields(
            "{}",
            false,
            &mut publication,
            &mut used
        ));
        assert_eq!(used, RECORD_TAG.len() + 2);
        let buffer = publication.as_ref().unwrap();
        assert_eq!(buffer.len(), PUBLICATION_BYTES);
        assert_eq!(&buffer[..used], b"local_raw_copy {}");
        assert!(LocalCopyTrace::publish_fields(
            "{}",
            true,
            &mut publication,
            &mut used
        ));
        assert_eq!(used, 0);
        assert_eq!(publication.as_ref().unwrap().len(), PUBLICATION_BYTES);
        publication.as_mut().unwrap()[0] = 255;
        used = 1;
        assert!(!LocalCopyTrace::publish_pending(&publication, &mut used));
        assert_eq!(used, 0);
    }
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
        values.push(record(event));
        true
    });
    bindings(&mut trace, &copies);
    for copy in copies.iter().rev() {
        let index = trace.begin_batch(&[*copy], "direct").unwrap();
        trace.finish_batch(index, 17, true);
    }
    trace.finish_part_with(31, true, |event| {
        values.push(record(event));
        true
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
            "schema": 4,
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
            "copies_encoding": "uvarint-rle-base64-v1",
            "copy_rows": 1,
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
    for mode in [
        "failure",
        "overflow",
        "lost_batch",
        "no_drain",
        "no_pitch",
        "publication",
    ] {
        let pitch = if mode == "no_pitch" {
            Err("driver".into())
        } else {
            Ok(usize::MAX)
        };
        let mut trace = LocalCopyTrace::new(0, pitch, if mode == "overflow" { 4 } else { 100 });
        let mut values = Vec::new();
        let copies = copies();
        trace.begin_part_with(2, true, |event| {
            values.push(record(event));
            true
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
                let published =
                    mode != "publication" || !matches!(&event, TraceRecord::Batch { .. });
                values.push(record(event));
                published
            });
            assert_eq!(values.last().unwrap()["complete"], false, "{mode}");
            if mode == "publication" {
                assert_eq!(
                    values.last().unwrap()["invalid_reason"],
                    "capture_publication_failed"
                );
                assert!(trace.closed);
            }
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
        values.push(record(event));
        true
    });
    trace.finish_part_with(19, true, |event| {
        values.push(record(event));
        true
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
            values.push(record(event));
            true
        });
        bindings(&mut trace, &copies);
        if let Some(index) = trace.begin_batch(&copies, "direct") {
            trace.finish_batch(index, 17, true);
        }
        trace.finish_part_with(31, true, |event| {
            values.push(record(event));
            true
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
        values.push(record(event));
        true
    });
    trace.finish_part_with(31, true, |event| {
        values.push(record(event));
        true
    });
    assert_eq!(values.len(), count);
}

#[test]
fn compact_geometry_preserves_unsigned_values_and_declares_every_field() {
    let geometry = [
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
    ];
    let mut encoded = Vec::new();
    serialize_geometry(&[geometry], &mut serde_json::Serializer::new(&mut encoded)).unwrap();
    let wire: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    let rows = decoded_geometry(&wire, 1);
    let decoded: Vec<u64> = serde_json::from_value(rows[0].clone()).unwrap();
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
        assert_eq!(header["schema"], 4);
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
        true
    });
}

#[test]
fn geometry_encoding_preserves_varint_boundaries_and_empty_rows() {
    let values = [
        0,
        1,
        127,
        128,
        255,
        256,
        u32::MAX as u64,
        1_u64 << 32,
        (1_u64 << 63) - 1,
        1_u64 << 63,
        u64::MAX,
    ];
    let geometry: Vec<_> = values.iter().map(|value| [*value; 15]).collect();
    let repeated = vec![[u64::MAX; 15]; 129];
    let mixed: Vec<_> = geometry
        .iter()
        .copied()
        .chain(repeated.iter().copied())
        .collect();
    for rows in [&geometry[..], &repeated, &mixed, &[]] {
        let mut encoded = Vec::new();
        serialize_geometry(rows, &mut serde_json::Serializer::new(&mut encoded)).unwrap();
        let wire = serde_json::from_slice(&encoded).unwrap();
        let decoded = decoded_geometry(&wire, rows.len());
        assert_eq!(decoded, serde_json::json!(rows));
    }
}
