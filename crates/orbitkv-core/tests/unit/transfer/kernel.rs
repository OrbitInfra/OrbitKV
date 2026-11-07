use super::*;
use crate::memory::numa::NumaNode;
use crate::memory::pinned::{PagePolicy, PinnedMemory};
use crate::transfer::MemcpyBackend;
use cudarc::driver::sys;

#[derive(Clone, Copy)]
struct MappedHost {
    host: *mut u8,
    device: u64,
}

fn alloc_device(len: usize) -> u64 {
    let mut d: sys::CUdeviceptr = 0;
    // SAFETY: the test bound a live CUDA context; d is valid writable output
    // and the returned allocation is retained until all test copies drain.
    let r = unsafe { sys::cuMemAlloc_v2(&mut d, len) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
    d
}

/// Build `n` descriptors of `seg` bytes each over contiguous device/host
/// regions. `host_base` is reinterpreted per direction by the backend.
fn descs(device_base: u64, host_base: MappedHost, n: usize, seg: usize) -> Vec<CopyDesc> {
    (0..n)
        .map(|k| CopyDesc {
            device: device_base + (k * seg) as u64,
            // SAFETY: within the [host_base, host_base+n*seg) allocation.
            host: unsafe { host_base.host.add(k * seg) },
            host_device: host_base.device + (k * seg) as u64,
            size: seg,
            device_allocation: 0,
            host_registration: 0,
        })
        .collect()
}

/// The kernel backend must move bytes identically to the direct backend, in
/// both directions, over mapped pinned host memory.
#[test]
#[ignore = "requires a CUDA GPU"]
fn kernel_matches_direct_both_directions() {
    const N: usize = 257;
    const SEG: usize = 4096 + 16; // non-power-of-two, 16B-aligned
    const TAIL: usize = 7; // exercise the vectorized kernel's scalar tail
    let total = N * SEG + TAIL;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernel = KernelBackend::new(&ctx).expect("kernel backend");
    let memcpy = MemcpyBackend::new(&ctx).expect("DMA backend");

    let memory = PinnedMemory::allocate(total, PagePolicy::Regular, NumaNode::UNKNOWN)
        .expect("shared pinned payload backing");
    let host = MappedHost {
        host: memory.as_non_null().as_ptr(),
        device: memory.device_ptr().as_ptr() as u64,
    };
    let device = alloc_device(total);

    let mut pattern = vec![0u8; total];
    let mut out = vec![0u8; total];
    let zeros = vec![0u8; total];
    // SAFETY: memory owns the entire mapped host region; the test accesses
    // this slice only before submission or after stream synchronization.
    let host_slice = unsafe { std::slice::from_raw_parts_mut(host.host, total) };

    let read_device = |out: &mut [u8]| {
        // SAFETY: out has total writable bytes, device owns total initialized
        // bytes, and each test submission has synchronized before this oracle copy.
        let r = unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, device, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "DtoH");
    };
    let clear_device = || {
        // SAFETY: zeros owns total source bytes and device owns total destination
        // bytes; this synchronous copy ends before the next test submission.
        let r = unsafe { sys::cuMemcpyHtoD_v2(device, zeros.as_ptr() as *const _, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD clear");
    };

    let mut contiguous = descs(device, host, N, SEG);
    contiguous.last_mut().unwrap().size += TAIL;
    // 73 and N are coprime: every range is visited once, with no adjacent pair
    // in physical order. Bytes and descriptor count stay unchanged.
    let reordered: Vec<_> = (0..N).map(|index| contiguous[index * 73 % N]).collect();
    let mut allocation_boundaries = contiguous.clone();
    for (index, copy) in allocation_boundaries.iter_mut().enumerate() {
        // Model separate registrations and GPU allocations. Either physical
        // registration boundary must prevent direct-copy coalescing.
        copy.host_registration = index / 3;
        copy.device_allocation = index / 5;
    }

    for (shape, copies) in [
        ("contiguous", contiguous),
        ("reordered", reordered),
        ("allocation_boundaries", allocation_boundaries),
    ] {
        assert_eq!(copies.len(), N);
        assert_eq!(copies.iter().map(|copy| copy.size).sum::<usize>(), total);
        for round in 0..4 {
            let backend: &dyn TentCopy = if round % 2 == 0 { &kernel } else { &memcpy };
            // Include descriptor identity and the round so stale data or
            // misrouted ranges cannot hide behind a short repeating byte pattern.
            for (index, byte) in pattern.iter_mut().enumerate() {
                *byte = ((index * 31) ^ (index >> 8) ^ ((index / SEG) * 17) ^ (round * 67)) as u8;
            }
            host_slice.copy_from_slice(&pattern);
            clear_device();
            backend.h2d(&copies, &stream).expect("h2d");
            stream.synchronize().expect("sync");
            read_device(&mut out);
            assert_eq!(
                out,
                pattern,
                "h2d mismatch: shape={shape}, round={round}, backend={}",
                backend.name()
            );

            for byte in &mut pattern {
                *byte = byte.wrapping_add(97);
            }
            // SAFETY: pattern and the device allocation both cover total bytes; the
            // previous case drained before this synchronous reinitialization.
            let r = unsafe { sys::cuMemcpyHtoD_v2(device, pattern.as_ptr() as *const _, total) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD seed");
            host_slice.fill(0);
            backend.d2h(&copies, &stream).expect("d2h");
            stream.synchronize().expect("sync");
            assert_eq!(
                host_slice,
                &pattern[..],
                "d2h mismatch: shape={shape}, round={round}, backend={}",
                backend.name()
            );
        }
    }

    // SAFETY: all test submissions and oracle reads completed; device is
    // the exact allocation returned by alloc_device and is freed only once.
    unsafe {
        sys::cuMemFree_v2(device);
    }
}
