use super::*;

#[test]
#[ignore = "requires a CUDA GPU"]
fn strided_dma_preserves_unrequested_bytes_both_directions() {
    use crate::memory::numa::NumaNode;
    use crate::memory::pinned::{PagePolicy, PinnedMemory};
    use cudarc::driver::DevicePtr;

    const ROWS: usize = 33;
    const WIDTH: usize = 129;
    const HOST_PITCH: usize = 193;
    const DEVICE_PITCH: usize = 257;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let backend = MemcpyBackend::new(&context).unwrap();
    let memory =
        PinnedMemory::allocate(ROWS * HOST_PITCH, PagePolicy::Regular, NumaNode::UNKNOWN).unwrap();
    let target = stream.alloc_zeros::<u8>(ROWS * DEVICE_PITCH).unwrap();
    let device = target.device_ptr(&stream).0;
    stream.synchronize().unwrap();
    // SAFETY: this test exclusively owns the pinned region and synchronizes
    // every transfer before accessing it or freeing either allocation.
    let host =
        unsafe { std::slice::from_raw_parts_mut(memory.as_ptr().cast_mut(), ROWS * HOST_PITCH) };
    let copies: Vec<_> = (0..ROWS)
        .map(|index| CopyDesc {
            device: device + (index * DEVICE_PITCH) as u64,
            host: host.as_mut_ptr().wrapping_add(index * HOST_PITCH),
            host_device: memory.device_ptr().as_ptr() as u64 + (index * HOST_PITCH) as u64,
            size: WIDTH,
            device_allocation: 1,
            host_registration: memory.as_ptr() as usize,
        })
        .collect();
    assert_eq!(dma_copies(&copies, backend.max_pitch).count(), 1);
    for missing_row in [None, Some(13)] {
        let selected: Vec<_> = copies
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != missing_row)
            .map(|(_, copy)| *copy)
            .collect();
        host.fill(0x7b);
        let mut expected_device = vec![0xa6; ROWS * DEVICE_PITCH];
        // SAFETY: the target is owned above and no transfer is in flight.
        unsafe { sys::cuMemsetD8_v2(device, 0xa6, expected_device.len()).result() }.unwrap();
        for index in 0..ROWS {
            if Some(index) == missing_row {
                continue;
            }
            for byte in 0..WIDTH {
                let value = ((index * 71 + byte * 13) % 251) as u8;
                host[index * HOST_PITCH + byte] = value;
                expected_device[index * DEVICE_PITCH + byte] = value;
            }
        }
        backend.h2d(&selected, &stream).unwrap();
        stream.synchronize().unwrap();
        let actual = stream.clone_dtoh(&target).unwrap();
        assert_eq!(
            actual, expected_device,
            "H2D touched a gap or copied stale rows"
        );
        let mut expected_host = vec![0x3d; host.len()];
        for index in 0..ROWS {
            if Some(index) != missing_row {
                expected_host[index * HOST_PITCH..index * HOST_PITCH + WIDTH]
                    .copy_from_slice(&host[index * HOST_PITCH..index * HOST_PITCH + WIDTH]);
            }
        }
        host.fill(0x3d);
        backend.d2h(&selected, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(
            host, expected_host,
            "D2H touched a gap or copied stale rows"
        );
    }
}
