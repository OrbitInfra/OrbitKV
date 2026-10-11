//! Matched byte controls and submission-to-drain timings of native copy owners.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use orbitkv_core::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde_json::json;

const GUARD: usize = 32;

fn check(result: sys::CUresult) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
}

struct Memory {
    host: *mut u8,
    mapped: u64,
    device: u64,
    len: usize,
}

impl Memory {
    fn new(len: usize) -> Self {
        let mut host = std::ptr::null_mut();
        let mut mapped = 0;
        let mut device = 0;
        unsafe {
            check(sys::cuMemHostAlloc(
                &mut host,
                len,
                sys::CU_MEMHOSTALLOC_DEVICEMAP,
            ));
            check(sys::cuMemHostGetDevicePointer_v2(&mut mapped, host, 0));
            check(sys::cuMemAlloc_v2(&mut device, len));
        }
        Self {
            host: host.cast(),
            mapped,
            device,
            len,
        }
    }

    fn host_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.len) }
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe {
            check(sys::cuMemFree_v2(self.device));
            check(sys::cuMemFreeHost(self.host.cast()));
        }
    }
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    sizes: Vec<usize>,
    host_offset: usize,
    device_offset: usize,
    reverse: bool,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (name, sizes, host_offset, device_offset, reverse) in [
        ("empty", vec![], 0, 0, false),
        ("zero", vec![0, 0, 0], 0, 0, false),
        (
            "tails",
            vec![1, 7, 15, 16, 17, 31, 4097, 65543],
            0,
            0,
            false,
        ),
        ("unaligned-host", vec![17, 4097, 1048583], 1, 0, false),
        ("unaligned-device", vec![17, 4097, 1048583], 0, 1, false),
        ("unaligned-both", vec![17, 4097, 1048583], 1, 3, true),
        (
            "mixed-reordered",
            vec![0, 17, 4096, 65543, 1048583],
            0,
            0,
            true,
        ),
        ("grid-stride", vec![1; 65536], 0, 0, true),
        ("one-4mib", vec![4194304], 0, 0, false),
        ("large-1", vec![1048576], 0, 0, false),
        ("large-2", vec![1048576; 2], 0, 0, false),
        ("large-8", vec![1048576; 8], 0, 0, true),
        ("large-32", vec![1048576; 32], 0, 0, true),
        ("ascending-4k-1024", vec![4096; 1024], 0, 0, false),
        ("ascending-4k-4096", vec![4096; 4096], 0, 0, false),
        ("fragmented-4k-1024", vec![4096; 1024], 0, 0, true),
        ("fragmented-4k-4096", vec![4096; 4096], 0, 0, true),
    ] {
        cases.push(Case {
            name,
            sizes,
            host_offset,
            device_offset,
            reverse,
        });
    }
    cases
}

fn submit(
    backend: &dyn TransferBackend,
    copies: &[CopyDesc],
    stream: &Arc<CudaStream>,
    h2d: bool,
) -> (f64, f64) {
    let started = Instant::now();
    let result = if h2d {
        backend.h2d(copies, stream)
    } else {
        backend.d2h(copies, stream)
    };
    let submitted = Instant::now();
    // Even a partial submission error cannot release either allocation early.
    stream.synchronize().expect("physical stream drain");
    let drained = Instant::now();
    result.expect("native submission");
    (
        submitted.duration_since(started).as_secs_f64() * 1000.0,
        drained.duration_since(submitted).as_secs_f64() * 1000.0,
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut output: Option<PathBuf> = None;
    let mut seed = 20261007;
    let mut samples: usize = 30;
    let mut fragment_bytes: Option<usize> = None;
    let mut blocks: Option<usize> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => output = Some(args.next().expect("output path").into()),
            "--seed" => seed = args.next().expect("seed").parse().expect("numeric seed"),
            "--samples" => {
                samples = args
                    .next()
                    .expect("samples")
                    .parse()
                    .expect("numeric samples");
            }
            "--fragment-bytes" => {
                fragment_bytes = Some(
                    args.next()
                        .expect("fragment bytes")
                        .parse()
                        .expect("numeric fragment bytes"),
                );
            }
            "--blocks" => {
                blocks = Some(
                    args.next()
                        .expect("blocks")
                        .parse()
                        .expect("numeric blocks"),
                );
            }
            "--bench" => {}
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(samples > 0);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.expect("--output requires a new external JSONL path"))
        .expect("new output");
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let kernel = KernelBackend::new(&context).expect("native kernel");
    let dma = MemcpyBackend::new(&context).expect("native DMA");
    let backends: [&dyn TransferBackend; 2] = [&kernel, &dma];
    let mut selected = match (fragment_bytes, blocks) {
        (None, None) => cases(),
        (Some(size), Some(count)) => {
            assert!(size > 0 && count > 0);
            size.checked_add(2 * GUARD + 16)
                .and_then(|width| width.checked_mul(count))
                .expect("custom fragment allocation overflow");
            vec![Case {
                name: "custom-fragments",
                sizes: vec![size; count],
                host_offset: 0,
                device_offset: 0,
                reverse: true,
            }]
        }
        _ => panic!("--fragment-bytes and --blocks must be specified together"),
    };
    selected.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    for case in selected {
        let widths: Vec<_> = case
            .sizes
            .iter()
            .map(|n| n.div_ceil(16) * 16 + 2 * GUARD)
            .collect();
        let total: usize = widths.iter().sum::<usize>() + 2 * GUARD + 16;
        let mut memory = Memory::new(total);
        let mut copies = Vec::new();
        let mut offset = GUARD;
        let mut pattern = vec![0xA5; total];
        let mut expected = vec![0xCD; total];
        for (id, (&size, &width)) in case.sizes.iter().zip(&widths).enumerate() {
            let source_offset = offset + case.host_offset;
            let device_offset = offset + case.device_offset;
            for j in 0..size {
                let byte = ((j * 31) ^ (j >> 8) ^ (id * 17)) as u8;
                pattern[source_offset + j] = byte;
                expected[device_offset + j] = byte;
            }
            copies.push(CopyDesc {
                device: memory.device + device_offset as u64,
                host: unsafe { memory.host.add(source_offset) },
                host_device: memory.mapped + source_offset as u64,
                size,
                device_allocation: 0,
                host_registration: 0,
            });
            offset += width;
        }
        if case.reverse {
            copies.reverse();
        }
        for h2d in [true, false] {
            let mut timings = [Vec::new(), Vec::new()];
            let mut submissions = [Vec::new(), Vec::new()];
            let mut waits = [Vec::new(), Vec::new()];
            for (id, backend) in backends.iter().enumerate() {
                let (source, destination, result) = if h2d {
                    (&pattern, vec![0xCD; total], &expected)
                } else {
                    (&expected, vec![0xA5; total], &pattern)
                };
                memory
                    .host_slice()
                    .copy_from_slice(if h2d { source } else { &destination });
                let device_source = if h2d { &destination } else { source };
                unsafe {
                    check(sys::cuMemcpyHtoD_v2(
                        memory.device,
                        device_source.as_ptr().cast(),
                        total,
                    ));
                }
                submit(*backend, &copies, &stream, h2d);
                let mut device = vec![0; total];
                unsafe {
                    check(sys::cuMemcpyDtoH_v2(
                        device.as_mut_ptr().cast(),
                        memory.device,
                        total,
                    ));
                }
                assert_eq!(
                    device,
                    if h2d { result } else { source }.as_slice(),
                    "device {} {}",
                    case.name,
                    backend.name()
                );
                assert_eq!(
                    memory.host_slice(),
                    if h2d { source } else { result }.as_slice(),
                    "host {} {}",
                    case.name,
                    backend.name()
                );
                for _ in 0..5 {
                    submit(*backend, &copies, &stream, h2d);
                }
                timings[id].reserve(samples);
                submissions[id].reserve(samples);
                waits[id].reserve(samples);
            }
            for sample in 0..samples {
                let order = if sample % 4 < 2 { [0, 1] } else { [1, 0] };
                for id in order {
                    let started = Instant::now();
                    let (submission, wait) = submit(backends[id], &copies, &stream, h2d);
                    timings[id].push(started.elapsed().as_secs_f64() * 1000.0);
                    submissions[id].push(submission);
                    waits[id].push(wait);
                }
            }
            let row = json!({
                "case": case.name, "direction": if h2d { "h2d" } else { "d2h" },
                "sizes": case.sizes, "descriptors": copies.len(), "seed": seed,
                "warmup": 5, "samples": samples, "bytes": case.sizes.iter().sum::<usize>(),
                "byte_source_guard_oracle": "passed", "kernel_submit_to_drain_ms": timings[0],
                "dma_submit_to_drain_ms": timings[1],
                "kernel_cpu_submission_ms": submissions[0], "kernel_stream_wait_ms": waits[0],
                "dma_cpu_submission_ms": submissions[1], "dma_stream_wait_ms": waits[1],
                "scope": "Native batch CPU wall time with attribution clocks; no inference latency claim",
                "attribution_scope": "CPU submission includes packing, upload and driver calls; stream wait is host synchronization waiting, not isolated GPU duration. Total retains the outer call-to-return boundary and includes clock overhead."
            });
            writeln!(file, "{row}").expect("write evidence");
            file.flush().expect("flush evidence");
            println!("{} {} passed", case.name, if h2d { "h2d" } else { "d2h" });
        }
    }
}
