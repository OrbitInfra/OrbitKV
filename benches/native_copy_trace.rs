//! Explicit CPU diagnostic of the real private trace owner on captured geometry.

use std::hint::black_box;
use std::time::Instant;

use serde_json::{Value, json};

use super::*;
use crate::transfer::local::SourceRange;

struct BoundCopy {
    raw: RawCopy,
    copy: CopyDesc,
    tensor: LocalTensor,
    arena_bytes: usize,
}

struct InputPart {
    records: Vec<Value>,
    batches: Vec<Vec<BoundCopy>>,
}

fn number(value: &Value, field: &str) -> u64 {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("invalid {field}"))
}

fn stable(mut value: Value) -> Value {
    for field in [
        "pid",
        "trace_id",
        "tid",
        "begin_ns",
        "submitted_ns",
        "drained_ns",
    ] {
        value.as_object_mut().unwrap().remove(field);
    }
    value
}

fn load(path: &str) -> Vec<InputPart> {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.len() <= 64 * 1024 * 1024);
    let mut parts = Vec::new();
    let mut records = Vec::new();
    for line in text.lines() {
        let Some((_, fields)) = line.split_once("local_raw_copy ") else {
            continue;
        };
        let value: Value = serde_json::from_str(fields).unwrap();
        assert_eq!(value["schema"], 2);
        assert_eq!(value["clock"], "CLOCK_MONOTONIC");
        let complete = value["stage"] == "local_raw_copy_complete";
        records.push(value);
        if complete {
            assert_eq!(records.first().unwrap()["stage"], "local_raw_copy_open");
            assert_eq!(records[records.len() - 2]["stage"], "local_raw_copy_part");
            assert_eq!(records.last().unwrap()["complete"], true);
            let batches = records[1..records.len() - 2]
                .iter()
                .enumerate()
                .map(|(index, batch)| {
                    assert_eq!(batch["stage"], "local_raw_copy_batch");
                    assert_eq!(number(batch, "batch_id"), index as u64);
                    assert_eq!(batch["backend"], "direct");
                    assert_eq!(batch["direction"], "h2d");
                    assert_eq!(batch["success"], true);
                    batch["copies"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|geometry| BoundCopy {
                            raw: RawCopy {
                                source: SourceRange {
                                    arena_id: number(geometry, "arena_id"),
                                    allocation_id: number(geometry, "source_allocation_id"),
                                    allocation_offset: number(geometry, "source_allocation_offset"),
                                    allocation_size: number(geometry, "source_allocation_bytes"),
                                    offset: number(geometry, "source_offset"),
                                    size: number(geometry, "size"),
                                },
                                layer: "captured".into(),
                                destination_offset: number(geometry, "destination_offset"),
                            },
                            copy: CopyDesc {
                                device: number(geometry, "device"),
                                host: number(geometry, "host") as *mut u8,
                                host_device: number(geometry, "host_device"),
                                size: number(geometry, "size") as usize,
                                device_allocation: number(geometry, "device_allocation") as usize,
                                host_registration: number(geometry, "host_registration") as usize,
                            },
                            tensor: LocalTensor {
                                name: "captured".into(),
                                address: number(geometry, "registered_device_start"),
                                bytes: number(geometry, "registered_device_bytes") as usize,
                                allocation: number(geometry, "device_allocation") as usize,
                            },
                            arena_bytes: number(geometry, "arena_bytes") as usize,
                        })
                        .collect()
                })
                .collect();
            parts.push(InputPart {
                records: std::mem::take(&mut records),
                batches,
            });
        }
    }
    assert!(records.is_empty() && !parts.is_empty());
    parts
}

#[test]
#[ignore = "explicit CPU profile; requires captured trace input and fresh output paths"]
fn captured_geometry_profile() {
    let input = std::env::var("ORBITKV_TRACE_PROFILE_INPUT").unwrap();
    let output = std::env::var("ORBITKV_TRACE_PROFILE_OUTPUT").unwrap();
    let mode = std::env::var("ORBITKV_TRACE_PROFILE_MODE").unwrap();
    assert!(matches!(mode.as_str(), "staged" | "whole"));
    let parts = load(&input);
    let descriptors: Vec<Vec<Vec<CopyDesc>>> = parts
        .iter()
        .map(|part| {
            part.batches
                .iter()
                .map(|batch| batch.iter().map(|bound| bound.copy).collect())
                .collect()
        })
        .collect();
    let mut clocks: Vec<_> = (0..10_000)
        .map(|_| {
            let started = Instant::now();
            black_box(started.elapsed().as_nanos() as u64)
        })
        .collect();
    clocks.sort_unstable();
    orbitkv_common::logging::init_stderr("info");
    let mut samples = Vec::new();
    for sequence in 0..12 {
        let mut trace = LocalCopyTrace::new(
            number(&parts[0].records[0], "device") as usize,
            Ok(number(&parts[0].records[0], "max_pitch") as usize),
            number(&parts[0].records[0], "limit") as usize,
        );
        for (index, part) in parts.iter().enumerate() {
            let footer = &part.records[part.records.len() - 2];
            let planned = number(footer, "planned_descriptors") as usize;
            let final_part = footer["final_part"].as_bool().unwrap();
            let mut encoded = Vec::with_capacity(part.records.len());
            let mut encoding_ns = 0_u64;
            let mut encode = |record: TraceRecord<'_>| {
                let started = Instant::now();
                let fields = serde_json::to_string(&record).unwrap();
                encoding_ns += started.elapsed().as_nanos() as u64;
                encoded.push(fields);
            };
            let total_started = Instant::now();
            let started = Instant::now();
            if mode == "whole" {
                trace.begin_part(planned, final_part);
            } else {
                trace.begin_part_with(planned, final_part, &mut encode);
            }
            let open_ns = started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            for batch in &part.batches {
                for bound in batch {
                    trace.record_geometry(
                        &bound.raw,
                        &bound.copy,
                        &bound.tensor,
                        bound.arena_bytes,
                    );
                }
            }
            let geometry_ns = started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            for (batch_id, copies) in descriptors[index].iter().enumerate() {
                let batch = trace.begin_batch(copies, "direct").unwrap();
                trace.finish_batch(
                    batch,
                    number(&part.records[batch_id + 1], "cpu_submit_ns"),
                    true,
                );
            }
            let batch_capture_ns = started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            if mode == "whole" {
                trace.finish_part(number(footer, "host_drain_ns"), true);
            } else {
                trace.finish_part_with(number(footer, "host_drain_ns"), true, &mut encode);
            }
            let finish_ns = started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            if mode == "staged" {
                for fields in &encoded {
                    log::info!(target: "orbitkv_core::transfer::local::trace", "local_raw_copy {fields}");
                }
            }
            let publish_ns = started.elapsed().as_nanos() as u64;
            let started = Instant::now();
            if mode == "staged" {
                log::logger().flush();
            }
            let flush_ns = started.elapsed().as_nanos() as u64;
            let total_ns = total_started.elapsed().as_nanos() as u64;
            assert!(trace.invalid.is_none());
            if mode == "staged" {
                assert_eq!(encoded.len(), part.records.len());
                for (fields, expected) in encoded.iter().zip(&part.records) {
                    assert_eq!(
                        stable(serde_json::from_str(fields).unwrap()),
                        stable(expected.clone())
                    );
                }
            }
            samples.push(json!({
                "sequence": sequence, "warmup": sequence < 4, "part": index,
                "open_ns": open_ns, "geometry_ns": geometry_ns,
                "batch_capture_ns": batch_capture_ns, "finish_ns": finish_ns,
                "encoding_ns": encoding_ns, "publish_ns": publish_ns, "flush_ns": flush_ns,
                "total_ns": total_ns, "records": part.records.len(), "descriptors": planned,
                "encoded_bytes": encoded.iter().map(String::len).sum::<usize>(),
            }));
        }
    }
    let result = json!({
        "schema": 1, "mode": mode, "input": input,
        "clock_control_median_ns": clocks[clocks.len()/2],
        "clock_control_p99_ns": clocks[clocks.len()*99/100],
        "samples": samples,
        "scope": "CPU geometry replay; no CUDA, DMA, ownership or serving qualification",
    });
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, &result).unwrap();
    writeln!(file).unwrap();
    file.flush().unwrap();
}
