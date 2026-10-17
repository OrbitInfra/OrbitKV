//! Explicit CPU diagnostic of the real private trace owner on captured geometry.

use std::hint::black_box;
use std::io;
use std::mem::MaybeUninit;
use std::time::Instant;

use serde_json::{Value, json};

use super::*;
use crate::transfer::local::SourceRange;

fn invalid_probe(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn timespec_ns(value: libc::timespec) -> io::Result<u64> {
    let seconds = u64::try_from(value.tv_sec).map_err(|_| invalid_probe("negative CPU seconds"))?;
    let nanos =
        u64::try_from(value.tv_nsec).map_err(|_| invalid_probe("negative CPU nanoseconds"))?;
    if nanos >= 1_000_000_000 {
        return Err(invalid_probe("CPU nanoseconds out of range"));
    }
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|base| base.checked_add(nanos))
        .ok_or_else(|| invalid_probe("CPU time overflow"))
}

fn thread_clock(resolution: bool) -> io::Result<u64> {
    let mut value = MaybeUninit::uninit();
    // SAFETY: the syscall writes one live timespec; it does not touch captured addresses.
    let status = unsafe {
        if resolution {
            libc::clock_getres(libc::CLOCK_THREAD_CPUTIME_ID, value.as_mut_ptr())
        } else {
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, value.as_mut_ptr())
        }
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a successful clock call initialized the timespec.
    timespec_ns(unsafe { value.assume_init() })
}

#[derive(Clone, Copy, serde::Serialize)]
struct ThreadStamp {
    wall_before_ns: u64,
    cpu_ns: u64,
    wall_after_ns: u64,
}

impl ThreadStamp {
    fn read(epoch: Instant) -> io::Result<Self> {
        let wall_before_ns = epoch
            .elapsed()
            .as_nanos()
            .try_into()
            .map_err(|_| invalid_probe("wall time overflow"))?;
        let cpu_ns = thread_clock(false)?;
        let wall_after_ns = epoch
            .elapsed()
            .as_nanos()
            .try_into()
            .map_err(|_| invalid_probe("wall time overflow"))?;
        Ok(Self {
            wall_before_ns,
            cpu_ns,
            wall_after_ns,
        })
    }

    fn interval(self, end: Self) -> io::Result<Value> {
        if self.wall_before_ns > self.wall_after_ns || end.wall_before_ns > end.wall_after_ns {
            return Err(invalid_probe("reversed clock bracket"));
        }
        let cpu_ns = end
            .cpu_ns
            .checked_sub(self.cpu_ns)
            .ok_or_else(|| invalid_probe("CPU clock regressed"))?;
        let lower = end
            .wall_before_ns
            .checked_sub(self.wall_after_ns)
            .ok_or_else(|| invalid_probe("overlapping clock brackets"))?;
        let upper = end
            .wall_after_ns
            .checked_sub(self.wall_before_ns)
            .ok_or_else(|| invalid_probe("wall clock regressed"))?;
        Ok(json!({
            "cpu_ns": cpu_ns,
            "wall_lower_ns": lower, "wall_upper_ns": upper,
            "residual_lower_ns": i128::from(lower) - i128::from(cpu_ns),
            "residual_upper_ns": i128::from(upper) - i128::from(cpu_ns),
        }))
    }
}

#[derive(Clone, Copy, serde::Serialize)]
struct ThreadUsage {
    minor_faults: u64,
    major_faults: u64,
    voluntary_switches: u64,
    involuntary_switches: u64,
}

impl ThreadUsage {
    fn read() -> io::Result<Self> {
        let mut usage = MaybeUninit::uninit();
        // SAFETY: getrusage writes one live rusage belonging to the calling thread.
        if unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: successful getrusage initialized the output structure.
        let usage = unsafe { usage.assume_init() };
        let counter =
            |value| u64::try_from(value).map_err(|_| invalid_probe("negative thread counter"));
        Ok(Self {
            minor_faults: counter(usage.ru_minflt)?,
            major_faults: counter(usage.ru_majflt)?,
            voluntary_switches: counter(usage.ru_nvcsw)?,
            involuntary_switches: counter(usage.ru_nivcsw)?,
        })
    }

    fn delta(self, end: Self) -> io::Result<Self> {
        let delta = |before: u64, after: u64| {
            after
                .checked_sub(before)
                .ok_or_else(|| invalid_probe("thread counter regressed"))
        };
        Ok(Self {
            minor_faults: delta(self.minor_faults, end.minor_faults)?,
            major_faults: delta(self.major_faults, end.major_faults)?,
            voluntary_switches: delta(self.voluntary_switches, end.voluntary_switches)?,
            involuntary_switches: delta(self.involuntary_switches, end.involuntary_switches)?,
        })
    }
}

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

fn load(path: &str) -> Vec<InputPart> {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(text.len() <= 64 * 1024 * 1024);
    let mut parts = Vec::new();
    let mut records = Vec::new();
    for line in text.lines() {
        let Some((_, fields)) = line.split_once("local_raw_copy ") else {
            continue;
        };
        let mut value: Value = serde_json::from_str(fields).unwrap();
        assert!(value["schema"] == 2 || value["schema"] == 3);
        if value["schema"] == 3 {
            if value["stage"] == "local_raw_copy_open" {
                assert_eq!(value["copy_fields"], json!(COPY_FIELDS));
            } else if value["stage"] == "local_raw_copy_batch" {
                let copies = value["copies"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        let row = row.as_array().unwrap();
                        assert_eq!(row.len(), COPY_FIELDS.len());
                        Value::Object(
                            COPY_FIELDS
                                .iter()
                                .zip(row)
                                .map(|(field, value)| {
                                    assert!(value.as_u64().is_some());
                                    ((*field).to_owned(), value.clone())
                                })
                                .collect(),
                        )
                    })
                    .collect();
                value["copies"] = Value::Array(copies);
            }
        }
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

fn profile_schedule(
    probe: &str,
    contrast: Option<&str>,
    order: Option<&str>,
) -> io::Result<Vec<(Option<char>, bool)>> {
    if !matches!(probe, "off" | "thread") {
        return Err(invalid_probe("unknown probe mode"));
    }
    match (contrast, order) {
        (None, None) => Ok(vec![(None, probe == "thread"); 12]),
        (Some(contrast @ ("aa" | "ab")), Some(order @ ("ABBABAAB" | "BAABABBA")))
            if probe == "off" =>
        {
            Ok(order
                .chars()
                .take(4)
                .chain(order.chars())
                .map(|arm| (Some(arm), contrast == "ab" && arm == 'B'))
                .collect())
        }
        _ => Err(invalid_probe(
            "interleaved contrast/order must be paired; fixed probe must be off",
        )),
    }
}

#[test]
#[ignore = "explicit CPU profile; requires captured trace input and fresh output paths"]
fn captured_geometry_profile() {
    let input = std::env::var("ORBITKV_TRACE_PROFILE_INPUT").unwrap();
    let output = std::env::var("ORBITKV_TRACE_PROFILE_OUTPUT").unwrap();
    let mode = std::env::var("ORBITKV_TRACE_PROFILE_MODE").unwrap();
    assert!(matches!(mode.as_str(), "staged" | "whole"));
    let probe = match std::env::var("ORBITKV_TRACE_PROFILE_PROBE") {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => "off".to_owned(),
        Err(error) => panic!("invalid probe mode: {error}"),
    };
    let optional = |name| match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("invalid profile setting: {error}"),
    };
    let contrast = optional("ORBITKV_TRACE_PROFILE_CONTRAST");
    let order = optional("ORBITKV_TRACE_PROFILE_ORDER");
    let schedule = profile_schedule(&probe, contrast.as_deref(), order.as_deref()).unwrap();
    let cpu_resolution_ns = schedule
        .iter()
        .any(|(_, enabled)| *enabled)
        .then(|| thread_clock(true).unwrap());
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
    let profile_epoch = Instant::now();
    // SAFETY: gettid has no pointer arguments and returns the calling profile thread.
    let profile_tid = unsafe { libc::syscall(libc::SYS_gettid) };
    assert!(profile_tid > 0);
    let mut samples = Vec::new();
    for (sequence, (arm, probe_enabled)) in schedule.into_iter().enumerate() {
        let mut trace = LocalCopyTrace::new(
            number(&parts[0].records[0], "device") as usize,
            Ok(number(&parts[0].records[0], "max_pitch") as usize),
            number(&parts[0].records[0], "limit") as usize,
        );
        let mut publication = None;
        let mut publication_len = 0;
        for (index, part) in parts.iter().enumerate() {
            let footer = &part.records[part.records.len() - 2];
            let planned = number(footer, "planned_descriptors") as usize;
            let final_part = footer["final_part"].as_bool().unwrap();
            let mut encoding_ns = 0_u64;
            let mut publish_ns = 0_u64;
            let mut encoded_bytes = 0_usize;
            let mut encode = |record: TraceRecord<'_>| {
                let open = matches!(&record, TraceRecord::Open { .. });
                let complete = matches!(&record, TraceRecord::Complete { .. });
                let started = Instant::now();
                let fields = serde_json::to_string(&record).unwrap();
                encoding_ns += started.elapsed().as_nanos() as u64;
                encoded_bytes += fields.len();
                let started = Instant::now();
                let success = if open {
                    log::info!(target: "orbitkv_core::transfer::local::trace", "local_raw_copy {fields}");
                    true
                } else {
                    LocalCopyTrace::publish_fields(
                        &fields,
                        complete,
                        &mut publication,
                        &mut publication_len,
                    )
                };
                drop(fields);
                publish_ns += started.elapsed().as_nanos() as u64;
                success
            };
            let mut stamps = [None; 6];
            let observed_begin_ns = u64::try_from(profile_epoch.elapsed().as_nanos()).unwrap();
            let usage_before = probe_enabled.then(|| ThreadUsage::read().unwrap());
            let total_started = Instant::now();
            stamps[0] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
            let started = Instant::now();
            if mode == "whole" {
                trace.begin_part(planned, final_part);
            } else {
                trace.begin_part_with(planned, final_part, &mut encode);
            }
            let open_ns = started.elapsed().as_nanos() as u64;
            stamps[1] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
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
            stamps[2] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
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
            stamps[3] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
            let started = Instant::now();
            if mode == "whole" {
                trace.finish_part(number(footer, "host_drain_ns"), true);
            } else {
                trace.finish_part_with(number(footer, "host_drain_ns"), true, &mut encode);
            }
            let finish_ns = started.elapsed().as_nanos() as u64;
            stamps[4] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
            let started = Instant::now();
            if mode == "staged" {
                log::logger().flush();
            }
            let flush_ns = started.elapsed().as_nanos() as u64;
            stamps[5] = probe_enabled.then(|| ThreadStamp::read(total_started).unwrap());
            let total_finished = Instant::now();
            let total_ns = u64::try_from(
                total_finished
                    .checked_duration_since(total_started)
                    .unwrap()
                    .as_nanos(),
            )
            .unwrap();
            let usage_after = probe_enabled.then(|| ThreadUsage::read().unwrap());
            let observed_end_ns = u64::try_from(profile_epoch.elapsed().as_nanos()).unwrap();
            let observed_ns = observed_end_ns.checked_sub(observed_begin_ns).unwrap();
            let total_begin_ns = u64::try_from(
                total_started
                    .checked_duration_since(profile_epoch)
                    .unwrap()
                    .as_nanos(),
            )
            .unwrap();
            let thread = if probe_enabled {
                let stamps = stamps.map(Option::unwrap);
                let before = usage_before.unwrap();
                let after = usage_after.unwrap();
                Some(json!({
                    "stamps": stamps,
                    "intervals": stamps.windows(2).map(|pair| pair[0].interval(pair[1]).unwrap()).collect::<Vec<_>>(),
                    "whole": stamps[0].interval(stamps[5]).unwrap(),
                    "usage_before": before, "usage_after": after,
                    "usage_delta": before.delta(after).unwrap(),
                }))
            } else {
                None
            };
            assert!(trace.invalid.is_none());
            assert_eq!(trace.publication_len, 0);
            assert_eq!(publication_len, 0);
            let publication_bytes = if mode == "whole" {
                trace.publication.as_ref().map_or(0, |buffer| buffer.len())
            } else {
                publication.as_ref().map_or(0, |buffer| buffer.len())
            };
            assert_eq!(publication_bytes, PUBLICATION_BYTES);
            let mut sample = json!({
                "sequence": sequence, "warmup": sequence < 4, "part": index,
                "open_ns": open_ns, "geometry_ns": geometry_ns,
                "batch_capture_ns": batch_capture_ns, "finish_ns": finish_ns,
                "encoding_ns": encoding_ns, "publish_ns": publish_ns, "flush_ns": flush_ns,
                "total_ns": total_ns, "observed_ns": observed_ns,
                "observed_begin_ns": observed_begin_ns, "observed_end_ns": observed_end_ns,
                "total_begin_ns": total_begin_ns, "thread": thread, "records": part.records.len(), "descriptors": planned,
                "encoded_bytes": encoded_bytes, "publication_bytes": publication_bytes,
            });
            if let Some(arm) = arm {
                sample["arm"] = json!(arm);
                sample["probe"] = json!(if probe_enabled { "thread" } else { "off" });
            }
            samples.push(sample);
        }
    }
    let mut result = json!({
        "schema": if contrast.is_some() { 3 } else { 2 }, "mode": mode, "input": input,
        "probe": if contrast.is_some() { "per-sequence" } else { &probe },
        "thread_cpu_resolution_ns": cpu_resolution_ns, "profile_tid": profile_tid,
        "wall_epoch": "Instant profile epoch; stamp wall offsets are relative to total_begin_ns",
        "thread_probe_scope": "Six CLOCK_THREAD_CPUTIME_ID reads bracketed by Instant wall reads relative to total start; adjacent intervals include timer bookkeeping. Signed residual is not proven scheduler wait. RUSAGE_THREAD encloses the part outside total_ns; observed_ns includes both calls. Result serialization and trace constructor/destructor remain excluded.",
        "clock_control_median_ns": clocks[clocks.len()/2],
        "clock_control_p99_ns": clocks[clocks.len()*99/100],
        "samples": samples,
        "scope": "CPU geometry replay; no CUDA, DMA, ownership or serving qualification",
    });
    if let Some(contrast) = contrast {
        result["contrast"] = json!(contrast);
        result["order"] = json!(order.unwrap());
    }
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

#[cfg(test)]
#[path = "../crates/orbitkv-core/tests/unit/transfer/local/trace_profile.rs"]
mod tests;
