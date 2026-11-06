use super::*;
use crate::memory::{numa::NumaNode, pool::PinnedAllocator};
use crate::transfer_engine::{CudaRegistration, TransferBackend};
use cudarc::driver::{CudaContext, result};
use std::num::NonZeroU64;

struct Device {
    address: u64,
    length: usize,
    _context: Arc<CudaContext>,
}
impl Device {
    fn new(context: &Arc<CudaContext>, length: usize) -> Arc<Self> {
        // SAFETY: the fixture provides a real CPU allocation behind this driver ABI.
        let address = unsafe { result::malloc_sync(length) }.unwrap();
        Arc::new(Self {
            address,
            length,
            _context: context.clone(),
        })
    }
    fn bytes(&self) -> Vec<u8> {
        // SAFETY: called only in the required CPU shim after physical drain.
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.length) }.to_vec()
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: final allocation owner is released only after engine loans drain.
        unsafe { result::free_sync(self.address) }.unwrap();
    }
}
struct Shim(libloading::Library);
impl Shim {
    fn load() -> Self {
        assert_eq!(
            std::env::var("ORBITKV_REQUIRE_LOCAL_COPY_SHIM").as_deref(),
            Ok("1"),
            "run with the explicit CPU CUDA fixture; never use an installed GPU driver"
        );
        // SAFETY: only the selected test fixture exports the control ABI.
        Self(unsafe { libloading::Library::new("libcuda.so.1") }.unwrap())
    }
    fn set(&self, name: &[u8], value: i32) {
        // SAFETY: fixture functions have this exact ABI; the library stays loaded.
        unsafe { self.0.get::<unsafe extern "C" fn(i32)>(name).unwrap()(value) };
    }
    fn count(&self, index: u32) -> u64 {
        // SAFETY: the fixture's counter ABI matches c_ulong on both Linux targets.
        unsafe {
            self.0
                .get::<unsafe extern "C" fn(u32) -> libc::c_ulong>(b"orbitkv_test_cuda_count")
                .unwrap()(index)
        }
    }
}

fn engine(bytes: usize) -> Arc<OrbitTeTransport> {
    Arc::new(
        OrbitTeTransport::new(
            bytes as u64,
            "127.0.0.1:0",
            orbit_te::Transport::Tcp,
            None,
            #[cfg(feature = "orbitte-rdma")]
            None,
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires the explicit CPU CUDA-driver fixture"]
fn local_copy_shim_bytes_stream_binding_and_pin_owners() {
    let shim = Shim::load();
    for mode in [TransferMode::Direct, TransferMode::Kernel] {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let device = Device::new(&context, 65536);
        let allocator = PinnedAllocator::new_global(65536, 1, false, None);
        let host = allocator
            .allocate(NonZeroU64::new(65536).unwrap(), NumaNode::UNKNOWN)
            .unwrap();
        let address = host.as_non_null().as_ptr();
        // SAFETY: the fresh pinned slot is exclusive and no copies are admitted.
        unsafe {
            for i in 0..65536 {
                address.add(i).write((i % 251) as u8);
            }
        }
        let engine = engine(65536);
        let backend: Arc<dyn TransferBackend> = engine.clone();
        assert!(
            crate::transfer::worker::GpuWorkerPool::new(
                0,
                NumaNode::UNKNOWN,
                mode,
                None,
                Some(backend.clone())
            )
            .is_err(),
            "a missing OrbiTE registration must not select TENT local copies"
        );
        // SAFETY: the actual device/context owner is retained in the registration.
        let registered = unsafe {
            CudaRegistration::new(
                Some(backend.clone()),
                device.address,
                device.length,
                0,
                device.clone(),
            )
        }
        .unwrap();
        let binds = shim.count(0);
        let kernels = shim.count(4);
        let worker = backend
            .bind_local(stream.clone(), mode, device.address)
            .unwrap();
        let hosts = vec![LocalHost::Pinned(host.clone())];
        let copies: Vec<_> = (0..6)
            .map(|row| CopyDesc {
                device: device.address + row * 80,
                // SAFETY: checked row offsets remain inside the live pinned slot.
                host: unsafe { address.add(row as usize * 64) },
                host_device: address as u64 + row * 64,
                size: 32,
                device_allocation: 1,
                host_registration: host.registration_id(),
            })
            .collect();
        for _ in 0..3 {
            // SAFETY: sources immutable and destination exclusive until completion.
            let mut completion = unsafe { worker.submit(&copies, &hosts, true) }.unwrap();
            completion.wait().unwrap();
        }
        assert_eq!(shim.count(0) - binds, 1, "one bind per worker stream");
        assert_eq!(
            shim.count(4) - kernels,
            if mode == TransferMode::Kernel { 3 } else { 0 }
        );
        let bytes = device.bytes();
        for row in 0..6 {
            assert_eq!(
                &bytes[row * 80..row * 80 + 32],
                &(row * 64..row * 64 + 32)
                    .map(|i| (i % 251) as u8)
                    .collect::<Vec<_>>()
            );
            assert!(
                bytes[row * 80 + 32..(row + 1) * 80].iter().all(|&b| b == 0),
                "strided gaps untouched"
            );
        }
        // SAFETY: all earlier copies drained; clear only the destination host rows.
        unsafe {
            for copy in &copies {
                std::ptr::write_bytes(copy.host, 0, copy.size);
            }
        }
        shim.set(b"orbitkv_test_cuda_hold", 1);
        // SAFETY: the same owners cover this exclusive D2H batch.
        let mut completion = unsafe { worker.submit(&copies, &hosts, false) }.unwrap();
        assert!(!completion.poll().unwrap());
        let weak = Arc::downgrade(&host);
        drop((host, hosts, allocator));
        assert!(
            weak.upgrade().is_some(),
            "completion holds the exact allocator slot and physical pin"
        );
        let retained = weak.upgrade().unwrap();
        shim.set(b"orbitkv_test_cuda_hold", 0);
        completion.wait().unwrap();
        for (row, copy) in copies.iter().enumerate() {
            // SAFETY: D2H drained, and retained still owns these pinned rows.
            let actual = unsafe { std::slice::from_raw_parts(copy.host, copy.size) };
            let expected: Vec<_> = (row * 64..row * 64 + 32)
                .map(|index| (index % 251) as u8)
                .collect();
            assert_eq!(actual, expected, "D2H bytes in row {row}");
        }
        drop(retained);
        assert!(
            weak.upgrade().is_none(),
            "drained completion releases the slot"
        );
        registered.close().unwrap();
        worker.close().unwrap();
    }
}

#[test]
#[ignore = "requires the explicit CPU CUDA-driver fixture"]
fn local_copy_shim_large_batch_and_partial_submission_drain() {
    let shim = Shim::load();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let device = Device::new(&context, 65536);
    let allocator = PinnedAllocator::new_global(65536, 1, false, None);
    let host = allocator
        .allocate(NonZeroU64::new(65536).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let address = host.as_non_null().as_ptr();
    // SAFETY: exclusive unsubmitted pinned allocation.
    unsafe {
        address.write_bytes(0x5a, 65536);
    }
    let engine = engine(65536);
    let backend: Arc<dyn TransferBackend> = engine.clone();
    // SAFETY: retained actual device/context backing, with no concurrent accesses.
    let registered = unsafe {
        CudaRegistration::new(
            Some(backend.clone()),
            device.address,
            device.length,
            0,
            device.clone(),
        )
    }
    .unwrap();
    let worker = backend
        .bind_local(stream, TransferMode::Direct, device.address)
        .unwrap();
    let hosts = [LocalHost::Pinned(host)];
    let copies: Vec<_> = (0..DESCRIPTORS + 3)
        .map(|i| CopyDesc {
            device: device.address + (i * 4) as u64,
            // SAFETY: the entire bounded descriptor vector fits the allocation.
            host: unsafe { address.add(i * 4) },
            host_device: address as u64 + (i * 4) as u64,
            size: 2,
            device_allocation: 1,
            host_registration: 1,
        })
        .collect();
    // SAFETY: both backing owners stay retained until all windows drain.
    let mut completion = unsafe { worker.submit(&copies, &hosts, true) }.unwrap();
    completion.wait().unwrap();
    let bytes = device.bytes();
    for i in 0..copies.len() {
        assert_eq!(&bytes[i * 4..i * 4 + 4], &[0x5a, 0x5a, 0, 0]);
    }
    let mut invalid = copies.clone();
    invalid[DESCRIPTORS + 1].device = 1;
    // SAFETY: invalid destination is rejected by the production registry; the
    // already admitted first window must drain before this error returns.
    assert!(unsafe { worker.submit(&invalid, &hosts, true) }.is_err());
    // Two strided rows form one 2D command; a differently sized sibling is a
    // second command. Reject it after the first enqueue, then prove its prefix.
    let mut partial = copies[..3].to_vec();
    partial[2].size = 1;
    // SAFETY: earlier work drained; this is CPU fixture storage, never a GPU pointer.
    unsafe {
        (device.address as *mut u8).write_bytes(0, device.length);
    }
    shim.set(b"orbitkv_test_cuda_fail_after", 1);
    // SAFETY: unchanged owners remain held while the first command drains.
    let mut failed = unsafe { worker.submit(&partial, &hosts, true) }.unwrap();
    let error = failed.wait().unwrap_err();
    assert_eq!(error.kind, ErrorKind::Backend);
    assert_eq!(
        &device.bytes()[..12],
        &[0x5a, 0x5a, 0, 0, 0x5a, 0x5a, 0, 0, 0, 0, 0, 0]
    );
    registered.close().unwrap();
    worker.close().unwrap();
}

#[test]
#[ignore = "requires the explicit CPU CUDA-driver fixture"]
fn local_copy_shim_client_restore_layers_and_context() {
    use crate::transfer::local::{LocalRestoreExecutor, LocalTensor, RawCopy, RawRestorePart};
    let shim = Shim::load();
    let context = CudaContext::new(0).unwrap();
    let device = Device::new(&context, 4096);
    let allocator = PinnedAllocator::new_global(4096, 1, false, None);
    let host = allocator
        .allocate(NonZeroU64::new(512).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    // SAFETY: the slot is exclusive before the grant starts.
    unsafe {
        host.as_non_null().as_ptr().write_bytes(0x93, 512);
    }
    let tensors = (0..13)
        .map(|index| {
            LocalTensor::new(
                format!("layer{index}"),
                device.address + index * 128,
                4096 - index as usize * 128,
                device.address as usize,
                2,
                32,
                0,
                1,
            )
            .unwrap()
        })
        .collect();
    let source = host.source_range(host.as_non_null(), 32).unwrap();
    let prior = 0x9876_usize as cudarc::driver::sys::CUcontext;
    // SAFETY: the CPU fixture recognizes borrowed context identities without dereference.
    unsafe { result::ctx::set_current(prior) }.unwrap();
    let mut executor = LocalRestoreExecutor::with_engine(
        0,
        tensors,
        allocator.payload_arenas().unwrap(),
        TransferMode::Kernel,
        crate::PeerTransferBackend::OrbitTeTcp,
        device.clone(),
    )
    .unwrap();
    assert_eq!(result::ctx::get_current().unwrap(), Some(prior));
    let binds = shim.count(0);
    let plan = RawRestorePart {
        copies: (0..13)
            .map(|index| RawCopy {
                source: crate::transfer::local::SourceRange {
                    offset: source.offset + index * 32,
                    ..source.clone()
                },
                layer: format!("layer{index}"),
                destination_offset: 32,
            })
            .collect(),
        completed_layers: (0..13).map(|index| format!("layer{index}")).collect(),
    };
    for _ in 0..3 {
        let native_events: Vec<_> = (0..13).map(|_| context.new_event(None).unwrap()).collect();
        let mut events = native_events
            .iter()
            .enumerate()
            .map(|(index, event)| (format!("layer{index}"), event.cu_event() as u64))
            .collect();
        let enqueued = std::cell::Cell::new(false);
        // More layers than completion slots, with delayed native progress.
        shim.set(b"orbitkv_test_cuda_query_delay", 100);
        executor
            .execute(&plan, &mut events, true, || enqueued.set(true), None)
            .unwrap();
        assert!(enqueued.get() && events.is_empty());
        for (index, event) in native_events.iter().enumerate() {
            event.synchronize().unwrap();
            assert_eq!(
                &device.bytes()[index * 128 + 32..index * 128 + 64],
                &[0x93; 32]
            );
        }
    }
    assert_eq!(shim.count(0), binds, "client copies reuse their binding");
}

#[test]
#[ignore = "requires the explicit CPU CUDA-driver fixture"]
fn local_copy_shim_unknown_completion_retains_backing() {
    let shim = Shim::load();
    if std::env::var_os("ORBITKV_LOCAL_COPY_UNCERTAIN_CHILD").is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "peer::orbitte::local::tests::local_copy_shim_unknown_completion_retains_backing",
                "--ignored",
                "--test-threads=1",
            ])
            .env("ORBITKV_LOCAL_COPY_UNCERTAIN_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let device = Device::new(&context, 4096);
    let allocator = PinnedAllocator::new_global(4096, 1, false, None);
    let host = allocator
        .allocate(NonZeroU64::new(4096).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let backend: Arc<dyn TransferBackend> = engine(4096);
    // SAFETY: actual device allocation and context remain held through this child.
    let registered = unsafe {
        CudaRegistration::new(
            Some(backend.clone()),
            device.address,
            4096,
            0,
            device.clone(),
        )
    }
    .unwrap();
    let worker = backend
        .bind_local(stream, TransferMode::Direct, device.address)
        .unwrap();
    let copy = CopyDesc {
        device: device.address,
        host: host.as_non_null().as_ptr(),
        host_device: host.as_non_null().as_ptr() as u64,
        size: 32,
        device_allocation: 1,
        host_registration: host.registration_id(),
    };
    let weak = Arc::downgrade(&host);
    // SAFETY: exclusive source/destination are held until completion or process exit.
    let mut completion =
        unsafe { worker.submit(&[copy], &[LocalHost::Pinned(host.clone())], true) }.unwrap();
    drop((host, allocator));
    shim.set(b"orbitkv_test_cuda_query_error", 999);
    assert_eq!(completion.poll().unwrap_err().kind, ErrorKind::Unquiesced);
    assert!(weak.upgrade().is_some());
    assert!(worker.close().is_err());
    assert!(weak.upgrade().is_some());
    // Do not drop uncertain owners, retry cleanup, or turn the error into a
    // release receipt. The isolated process ends with its quarantine still held.
    let _retained = (completion, registered, worker, device);
    std::process::exit(0);
}
