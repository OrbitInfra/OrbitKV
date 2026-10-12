//! Native copy comparison with explicit physical owners and disclosed geometry.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, sys};
use orbitkv_core::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use serde_json::json;
use uuid::Uuid;

#[path = "native_copy_layout/capture.rs"]
mod capture;
#[path = "native_copy_layout/geometry.rs"]
mod geometry;
#[path = "../crates/orbitkv-core/src/transfer/memcpy/ranges.rs"]
mod ranges;

use geometry::{FRAGMENT_BYTES, LAYERS, Layout};

const EVENT_SCOPE: &str = "Stream event interval includes descriptor upload and possible host-feed/scheduling gaps; not isolated GPU execution or bandwidth";

fn check(result: sys::CUresult) {
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
}

struct HostAllocation {
    address: NonNull<u8>,
    mapped: u64,
    len: usize,
}

impl HostAllocation {
    fn new(len: usize) -> Self {
        let mut address = std::ptr::null_mut();
        let mut mapped = 0;
        // SAFETY: CUDA allocates a new mapped pinned region of the requested size.
        unsafe {
            check(sys::cuMemHostAlloc(
                &mut address,
                len,
                sys::CU_MEMHOSTALLOC_DEVICEMAP,
            ));
            let result = sys::cuMemHostGetDevicePointer_v2(&mut mapped, address, 0);
            if result != sys::CUresult::CUDA_SUCCESS {
                check(sys::cuMemFreeHost(address));
                check(result);
            }
        }
        Self {
            address: NonNull::new(address.cast()).expect("non-null pinned allocation"),
            mapped,
            len,
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the fixture retains this allocation and drains before inspection.
        unsafe { std::slice::from_raw_parts(self.address.as_ptr(), self.len) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: all prior GPU work is drained and the fixture has exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.address.as_ptr(), self.len) }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        // SAFETY: every submitted stream is physically drained before owner release.
        unsafe { check(sys::cuMemFreeHost(self.address.as_ptr().cast())) };
    }
}

struct DeviceAllocation {
    address: u64,
    len: usize,
}

impl DeviceAllocation {
    fn new(len: usize) -> Self {
        let mut address = 0;
        // SAFETY: CUDA allocates a new device region of the requested size.
        unsafe { check(sys::cuMemAlloc_v2(&mut address, len)) };
        Self { address, len }
    }

    fn write(&self, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.len);
        // SAFETY: source and destination span len bytes and no stream is in flight.
        unsafe {
            check(sys::cuMemcpyHtoD_v2(
                self.address,
                bytes.as_ptr().cast(),
                self.len,
            ));
        }
    }

    fn read(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.len];
        // SAFETY: destination spans len bytes and prior stream work is drained.
        unsafe {
            check(sys::cuMemcpyDtoH_v2(
                bytes.as_mut_ptr().cast(),
                self.address,
                self.len,
            ));
        }
        bytes
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        // SAFETY: the fixture retains all owners until physical stream completion.
        unsafe { check(sys::cuMemFree_v2(self.address)) };
    }
}

struct Memory {
    hosts: Vec<HostAllocation>,
    devices: Vec<DeviceAllocation>,
    host_source: Vec<Vec<u8>>,
    device_source: Vec<Vec<u8>>,
    host_result: Vec<Vec<u8>>,
    device_result: Vec<Vec<u8>>,
}

impl Memory {
    fn new(layout: &Layout) -> Self {
        let hosts: Vec<_> = layout
            .host_lengths()
            .into_iter()
            .map(HostAllocation::new)
            .collect();
        let devices: Vec<_> = layout
            .device_lengths()
            .into_iter()
            .map(DeviceAllocation::new)
            .collect();
        let mut host_source: Vec<_> = hosts.iter().map(|host| vec![0xA5; host.len]).collect();
        let mut device_source: Vec<_> = devices
            .iter()
            .map(|device| vec![0x5A; device.len])
            .collect();
        let mut host_result: Vec<_> = hosts.iter().map(|host| vec![0xCD; host.len]).collect();
        let mut device_result: Vec<_> = devices
            .iter()
            .map(|device| vec![0xCD; device.len])
            .collect();
        for range in layout.ranges().expect("validated layout") {
            let id =
                (range.layer * layout.pages() + range.device_offset / geometry::SLOT_BYTES) as u64;
            let mut payload = vec![0; range.bytes];
            for (byte, value) in payload.iter_mut().enumerate() {
                *value = ((byte * 31) ^ (byte >> 8) ^ (id as usize * 17)) as u8;
            }
            payload[..8].copy_from_slice(&id.to_le_bytes());
            let host_range = range.host_offset..range.host_offset + range.bytes;
            let device_range = range.device_offset..range.device_offset + range.bytes;
            host_source[range.host_registration][host_range.clone()].copy_from_slice(&payload);
            host_result[range.host_registration][host_range].copy_from_slice(&payload);
            device_source[range.layer][device_range.clone()].copy_from_slice(&payload);
            device_result[range.layer][device_range].copy_from_slice(&payload);
        }
        Self {
            hosts,
            devices,
            host_source,
            device_source,
            host_result,
            device_result,
        }
    }

    fn copies(&self, layout: &Layout) -> Vec<CopyDesc> {
        layout
            .bind(
                &self
                    .devices
                    .iter()
                    .map(|device| device.address)
                    .collect::<Vec<_>>(),
                &self
                    .hosts
                    .iter()
                    .map(|host| (host.address.as_ptr(), host.mapped))
                    .collect::<Vec<_>>(),
            )
            .expect("physical layout binding")
    }

    fn prepare(&mut self, h2d: bool) {
        for (index, host) in self.hosts.iter_mut().enumerate() {
            if h2d {
                host.bytes_mut().copy_from_slice(&self.host_source[index]);
            } else {
                host.bytes_mut().fill(0xCD);
            }
        }
        for (index, device) in self.devices.iter().enumerate() {
            if h2d {
                device.write(&vec![0xCD; device.len]);
            } else {
                device.write(&self.device_source[index]);
            }
        }
    }

    fn validate(&self, h2d: bool) {
        for (index, host) in self.hosts.iter().enumerate() {
            assert_eq!(
                host.bytes(),
                if h2d {
                    &self.host_source[index]
                } else {
                    &self.host_result[index]
                },
                "host output/source/guard, registration {index}"
            );
        }
        for (index, device) in self.devices.iter().enumerate() {
            assert_eq!(
                device.read(),
                if h2d {
                    &self.device_result[index]
                } else {
                    &self.device_source[index]
                }
                .as_slice(),
                "device output/source/guard, allocation {index}"
            );
        }
    }
}

struct Events {
    start: CudaEvent,
    end: CudaEvent,
}

impl Events {
    fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            start: context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .expect("start event"),
            end: context
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .expect("end event"),
        }
    }
}

#[derive(Serialize)]
struct Measurements {
    call_to_return_ms: Vec<f64>,
    cpu_submission_ms: Vec<f64>,
    host_wait_ms: Vec<f64>,
    stream_event_ms: Option<Vec<f32>>,
}

impl Measurements {
    fn new(samples: usize, events: bool) -> Self {
        Self {
            call_to_return_ms: Vec::with_capacity(samples),
            cpu_submission_ms: Vec::with_capacity(samples),
            host_wait_ms: Vec::with_capacity(samples),
            stream_event_ms: events.then(|| Vec::with_capacity(samples)),
        }
    }

    fn push(&mut self, sample: Sample) {
        self.call_to_return_ms.push(sample.total);
        self.cpu_submission_ms.push(sample.submission);
        self.host_wait_ms.push(sample.wait);
        if let Some(events) = &mut self.stream_event_ms {
            events.push(sample.event.expect("event sample"));
        } else {
            assert!(sample.event.is_none());
        }
    }
}

struct Sample {
    total: f64,
    submission: f64,
    wait: f64,
    event: Option<f32>,
}

fn submit(
    backend: &dyn TransferBackend,
    copies: &[CopyDesc],
    stream: &Arc<CudaStream>,
    h2d: bool,
    events: Option<&Events>,
) -> Sample {
    let outer = Instant::now();
    if let Some(events) = events {
        events.start.record(stream).expect("record start event");
    }
    let started = Instant::now();
    let result = if h2d {
        backend.h2d(copies, stream)
    } else {
        backend.d2h(copies, stream)
    };
    let submitted = Instant::now();
    let event_result = events.map(|events| events.end.record(stream));
    if let Err(error) = stream.synchronize() {
        eprintln!("physical native drain unavailable: {error}");
        std::process::abort();
    }
    let drained = Instant::now();
    result.expect("native backend submission after physical drain");
    if let Some(result) = event_result {
        result.expect("record end event after physical drain");
    }
    let event = events.map(|events| {
        events
            .start
            .elapsed_ms(&events.end)
            .expect("event interval")
    });
    Sample {
        total: outer.elapsed().as_secs_f64() * 1000.0,
        submission: submitted.duration_since(started).as_secs_f64() * 1000.0,
        wait: drained.duration_since(submitted).as_secs_f64() * 1000.0,
        event,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut output: Option<PathBuf> = None;
    let mut seed = 2026101111;
    let mut samples: usize = 100;
    let mut validate_only = false;
    let mut local_geometry: Option<PathBuf> = None;
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
            "--validate-layout-only" => validate_only = true,
            "--validate-local-geometry" => {
                local_geometry = Some(args.next().expect("geometry path").into());
            }
            "--bench" => {}
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(
        samples >= 4 && samples.is_multiple_of(4),
        "samples must balance all four modes"
    );
    let output = output.expect("new external --output required");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .expect("new output");
    if let Some(path) = local_geometry {
        assert!(!validate_only, "select one CPU validation mode");
        let summary =
            serde_json::from_reader(std::fs::File::open(path).expect("captured geometry input"))
                .expect("captured geometry JSON");
        let receipt = capture::validate(&summary).expect("production compiler verification");
        writeln!(file, "{receipt}").expect("write compiler receipt");
        file.flush().expect("flush compiler receipt");
        return;
    }
    let mut layouts = geometry::layouts();
    for layout in &layouts {
        let copies = layout.virtual_copies().expect("valid pointer-free layout");
        assert_eq!(
            ranges::dma_copies(&copies, usize::MAX).count(),
            layout.expected_dma_ranges
        );
    }
    if validate_only {
        for layout in layouts {
            writeln!(file, "{}", json!({
                "case": layout.name, "layers": LAYERS, "fragment_bytes": FRAGMENT_BYTES,
                "descriptors": LAYERS * layout.pages(), "dma_ranges": layout.expected_dma_ranges,
                "host_registrations": layout.host_lengths().len(), "device_allocations": LAYERS,
                "host_order": layout.host_order, "host_lengths": layout.host_lengths(),
                "device_lengths": layout.device_lengths(), "normalized_ranges": layout.ranges().expect("valid ranges"),
                "scope": "CPU layout validation with virtual addresses, no CUDA initialization"
            })).expect("write CPU layout receipt");
        }
        file.flush().expect("flush CPU validation");
        println!("9 layouts passed before CUDA initialization");
        return;
    }
    let context = CudaContext::new(0).expect("CUDA context");
    let device_uuid = format!(
        "GPU-{}",
        Uuid::from_bytes(context.uuid().expect("CUDA UUID").bytes.map(|v| v as u8))
    );
    let mut device_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("device.json"))
        .expect("new device identity");
    writeln!(
        device_file,
        "{}",
        json!({"schema": 1, "cuda_device_ordinal": 0,
        "cuda_device_uuid": device_uuid, "native_pid": std::process::id()})
    )
    .expect("write device identity");
    device_file.flush().expect("flush device identity");
    let stream = context.new_stream().expect("dedicated CUDA stream");
    let kernel = KernelBackend::new(&context).expect("production kernel backend");
    let dma = MemcpyBackend::new(&context).expect("production DMA backend");
    let backends: [&dyn TransferBackend; 2] = [&kernel, &dma];
    let events = Events::new(&context);
    let max_pitch = usize::try_from(
        context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PITCH)
            .expect("CUDA maximum pitch"),
    )
    .expect("positive pitch");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    layouts.shuffle(&mut rng);
    for layout in layouts {
        let mut memory = Memory::new(&layout);
        let copies = memory.copies(&layout);
        let dma_ranges = ranges::dma_copies(&copies, max_pitch).count();
        assert_eq!(
            dma_ranges, layout.expected_dma_ranges,
            "actual physical compiler ranges"
        );
        let mut directions = [true, false];
        directions.shuffle(&mut rng);
        for h2d in directions {
            let mut data: [[Measurements; 2]; 2] = std::array::from_fn(|_| {
                std::array::from_fn(|event| Measurements::new(samples, event == 1))
            });
            for (backend_id, backend) in backends.iter().enumerate() {
                for (event_id, measurements) in data[backend_id].iter().enumerate() {
                    let selected_events = (event_id == 1).then_some(&events);
                    memory.prepare(h2d);
                    submit(*backend, &copies, &stream, h2d, selected_events);
                    memory.validate(h2d);
                    for _ in 0..5 {
                        submit(*backend, &copies, &stream, h2d, selected_events);
                    }
                    assert!(measurements.call_to_return_ms.is_empty());
                }
            }
            let mut sample_order = Vec::with_capacity(samples);
            const ORDERS: [[usize; 4]; 4] =
                [[0, 2, 3, 1], [2, 0, 1, 3], [1, 3, 2, 0], [3, 1, 0, 2]];
            for sample in 0..samples {
                let order = ORDERS[sample % ORDERS.len()];
                sample_order.push(order);
                for mode in order {
                    let backend = mode / 2;
                    let event = mode % 2;
                    data[backend][event].push(submit(
                        backends[backend],
                        &copies,
                        &stream,
                        h2d,
                        (event == 1).then_some(&events),
                    ));
                }
            }
            memory.validate(h2d);
            let row = json!({
                "schema": 1, "case": layout.name, "direction": if h2d { "h2d" } else { "d2h" },
                "cuda_device_uuid": device_uuid, "native_pid": std::process::id(), "seed": seed,
                "samples": samples, "warmup": 5, "descriptors": copies.len(),
                "fragment_bytes": FRAGMENT_BYTES, "bytes": copies.len() * FRAGMENT_BYTES,
                "dma_ranges": dma_ranges, "max_pitch": max_pitch,
                "layers": LAYERS, "host_order": layout.host_order, "split_host": layout.split_host,
                "observed_model_batch_weight": layout.observed_batches,
                "host_allocations": memory.hosts.iter().map(|host| json!({"address":host.address.as_ptr() as usize,
                    "mapped":host.mapped,"bytes":host.len})).collect::<Vec<_>>(),
                "device_allocations": memory.devices.iter().map(|device|json!({"address":device.address,"bytes":device.len})).collect::<Vec<_>>(),
                "normalized_ranges": layout.ranges().expect("valid ranges"),
                "sample_order": sample_order, "kernel": {"event_off":data[0][0],"event_on":data[0][1]},
                "dma": {"event_off":data[1][0],"event_on":data[1][1]},
                "byte_source_guard_oracle": "passed before each backend/event warmup and after final measured submission",
                "layout_scope": "Range-count-matched representative family with real physical owners and disclosed guard padding; not captured model addresses/pitch replay. H2D is a synthetic control.",
                "event_scope": EVENT_SCOPE,
                "timing_scope": "Event-off and event-on native call/submission/drain clocks include timer overhead; no inference latency, normal-path overhead qualification or confidence claim"
            });
            writeln!(file, "{row}").expect("write native row");
            file.flush().expect("flush native row");
            println!("{} {} passed", layout.name, if h2d { "h2d" } else { "d2h" });
        }
    }
}
