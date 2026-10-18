use super::*;
use cudarc::driver::CudaContext;
use std::sync::mpsc as std_mpsc;

#[test]
fn raw_copy_shape_uses_physical_registration_and_actual_dma_compiler() {
    let mut copies = [0, 4096, 8192].map(|offset| CopyDesc {
        device: 0x10000 + offset,
        host: (0x100000 + offset) as *mut u8,
        host_device: 0x200000 + offset,
        size: 4096,
        device_allocation: 1,
        host_registration: 2,
    });
    let shape = raw_copy_shape(&copies, usize::MAX);
    assert_eq!(shape.bytes, 12288);
    assert_eq!(shape.descriptors, 3);
    assert_eq!(shape.dma_ranges, 1);
    assert_eq!(shape.size_histogram, [0, 3, 0, 0, 0, 0, 0, 0]);
    assert_eq!((shape.min_bytes, shape.max_bytes), (4096, 4096));
    copies[1].host_registration = 3;
    assert_eq!(raw_copy_shape(&copies, usize::MAX).dma_ranges, 3);
    copies.reverse();
    assert_eq!(raw_copy_shape(&copies, usize::MAX).dma_ranges, 3);
}

#[test]
fn raw_copy_shape_histogram_covers_zero_boundaries_and_large_fragments() {
    let sizes = [
        0, 256, 257, 4096, 4097, 16384, 65536, 262144, 1048576, 4194304, 4194305,
    ];
    let copies: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(index, &size)| CopyDesc {
            device: 0x10000 + index as u64 * 0x1000000,
            host: (0x100000 + index * 0x1000000) as *mut u8,
            host_device: 0,
            size,
            device_allocation: index,
            host_registration: index,
        })
        .collect();
    let shape = raw_copy_shape(&copies, usize::MAX);
    assert_eq!(shape.bytes, sizes.iter().sum::<usize>() as u64);
    assert_eq!(shape.descriptors, sizes.len() as u64);
    assert_eq!(shape.zero_descriptors, 1);
    assert_eq!(shape.size_histogram, [2, 2, 2, 1, 1, 1, 1, 1]);
    assert_eq!((shape.min_bytes, shape.max_bytes), (0, 4194305));
    let empty = raw_copy_shape(&[], usize::MAX);
    assert_eq!(empty.bytes, 0);
    assert_eq!(empty.dma_ranges, 0);
    assert_eq!(empty.size_histogram, [0; 8]);
}

#[test]
fn failed_submission_waits_for_already_enqueued_stream_work() {
    struct Gate {
        entered: std_mpsc::Sender<()>,
        release: std_mpsc::Receiver<()>,
    }
    unsafe extern "C" fn hold_stream(data: *mut std::ffi::c_void) {
        // SAFETY: the successful launch transfers this Box to one callback.
        let gate = unsafe { Box::from_raw(data.cast::<Gate>()) };
        let _ = gate.entered.send(());
        let _ = gate.release.recv_timeout(std::time::Duration::from_secs(5));
    }

    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let (entered_tx, entered_rx) = std_mpsc::channel();
    let (release_tx, release_rx) = std_mpsc::channel();
    let gate = Box::into_raw(Box::new(Gate {
        entered: entered_tx,
        release: release_rx,
    }));
    // SAFETY: stream and callback data stay alive through synchronization;
    // the callback does not call CUDA or unwind across the C ABI.
    let launched = unsafe {
        cudarc::driver::result::stream::launch_host_function(
            stream.cu_stream(),
            hold_stream,
            gate.cast(),
        )
    };
    if let Err(error) = launched {
        // SAFETY: a failed launch did not take ownership of the callback.
        unsafe { drop(Box::from_raw(gate)) };
        panic!("host callback launch failed: {error}");
    }
    let (finished_tx, finished_rx) = std_mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = finish_gpu_transfer(&stream, Err("partial submission".to_string()));
        let _ = finished_tx.send(result);
    });
    entered_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    let premature = finished_rx.recv_timeout(std::time::Duration::from_millis(50));
    let _ = release_tx.send(());
    worker.join().unwrap();
    assert!(matches!(
        premature,
        Err(std_mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(
        finished_rx
            .recv()
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("partial submission")
    );
}

#[test]
fn gpu_unwind_retains_borrowed_owner_and_admission_in_an_isolated_process() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "transfer::tests::gpu_unwind_retention_child",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("caller_owner_retained=true admission_charged=true no_DMA=true"));
}

#[test]
#[ignore = "isolated no-DMA CPU ownership fixture"]
fn gpu_unwind_retention_child() {
    use std::sync::{Arc, mpsc};
    use std::time::Duration;
    let owner = Arc::new(vec![37; 4096]);
    let observed = Arc::downgrade(&owner);
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let permit = Arc::clone(&admission).try_acquire_owned().unwrap();
    let (reached, entered) = mpsc::channel();
    let (returned, result) = mpsc::channel();
    std::thread::spawn(move || {
        let _permit = permit;
        contain_gpu_unwind(|| {
            assert_eq!(owner.as_slice(), &[37; 4096]);
            reached.send(()).unwrap();
            fn controlled_unwind() {
                panic!("controlled worker execution unwind");
            }
            controlled_unwind();
        });
        drop(owner);
        returned.send(()).unwrap();
    });
    entered.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        result.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(observed.upgrade().is_some());
    assert!(Arc::clone(&admission).try_acquire_owned().is_err());
    println!("caller_owner_retained=true admission_charged=true no_DMA=true");
    // Only an isolated CPU fixture process exits with its deliberately parked thread.
}
