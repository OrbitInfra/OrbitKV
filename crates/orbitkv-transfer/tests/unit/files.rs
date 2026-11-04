use super::*;
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::AtomicUsize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Reject,
    Short,
    Pending,
    Release,
    Close,
}
struct Control {
    fault: Fault,
    submits: AtomicUsize,
    ranges: AtomicUsize,
    closes: AtomicUsize,
    releases: AtomicUsize,
    destroyed: AtomicBool,
    owner: Mutex<Option<(usize, unsafe extern "C" fn(*mut c_void))>>,
}
unsafe fn control<'a>(engine: native::NativeEngine) -> &'a Control {
    // SAFETY: each fixture installs an Arc<Control> as its native engine handle.
    unsafe { &*engine.cast::<Control>() }
}
unsafe extern "C" fn create(_: *const ffi::FileOptions, _: *mut native::NativeEngine) -> c_int {
    3
}
unsafe extern "C" fn open(
    _: native::NativeEngine,
    _: *const c_char,
    _: c_int,
    _: c_int,
    id: *mut u64,
    size: *mut u64,
) -> c_int {
    // SAFETY: the production wrapper provides these writable outputs.
    unsafe {
        *id = 1;
        *size = 8192;
    }
    0
}
unsafe extern "C" fn submit(
    engine: native::NativeEngine,
    _: u64,
    _: c_int,
    _: u64,
    _: *const ffi::FileRange,
    count: usize,
    owner: *const ffi::FileOwner,
    batch: *mut u64,
) -> c_int {
    // SAFETY: all pointers were supplied by the production wrapper.
    unsafe {
        let c = control(engine);
        c.submits.fetch_add(1, Ordering::Relaxed);
        c.ranges.store(count, Ordering::Relaxed);
        if c.fault == Fault::Reject {
            return 9;
        }
        ((*owner).retain)((*owner).context);
        *c.owner.lock().unwrap() = Some(((*owner).context as usize, (*owner).release));
        *batch = 1;
    }
    0
}
unsafe extern "C" fn drain(
    engine: native::NativeEngine,
    _: u64,
    _: u64,
    _: c_int,
    result: *mut ffi::FileResult,
) -> c_int {
    // SAFETY: engine/output and callback references are retained by the caller.
    unsafe {
        let c = control(engine);
        if c.fault == Fault::Pending {
            return ffi::UNQUIESCED;
        }
        if let Some((owner, release)) = c.owner.lock().unwrap().take() {
            release(owner as *mut c_void);
        }
        *result = ffi::FileResult {
            outcome: if c.fault == Fault::Short { 5 } else { 0 },
            drained: 1,
            transferred_bytes: if c.fault == Fault::Short { 512 } else { 1024 },
        };
        (*result).outcome
    }
}
unsafe extern "C" fn release_batch(engine: native::NativeEngine, _: u64) -> c_int {
    // SAFETY: this fixture's native owner remains live through receipt release.
    let c = unsafe { control(engine) };
    c.releases.fetch_add(1, Ordering::Relaxed);
    if c.fault == Fault::Release { 4 } else { 0 }
}
unsafe extern "C" fn close(
    engine: native::NativeEngine,
    _: u64,
    _: u64,
    result: *mut ffi::FileResult,
) -> c_int {
    // SAFETY: this fixture's engine and writable result remain live.
    unsafe {
        let c = control(engine);
        c.closes.fetch_add(1, Ordering::Relaxed);
        if c.fault == Fault::Close {
            return ffi::UNQUIESCED;
        }
        *result = ffi::FileResult {
            outcome: 0,
            drained: 1,
            transferred_bytes: 0,
        };
    }
    0
}
unsafe extern "C" fn destroy(engine: native::NativeEngine) {
    // SAFETY: consumes the one raw Arc installed by fixture().
    unsafe { Arc::from_raw(engine.cast::<Control>()) }
        .destroyed
        .store(true, Ordering::Release);
}
fn fixture(fault: Fault) -> (Arc<FileEngine>, Arc<FileSegment>, Arc<Control>) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let c = Arc::new(Control {
        fault,
        submits: AtomicUsize::new(0),
        ranges: AtomicUsize::new(0),
        closes: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
        destroyed: AtomicBool::new(false),
        owner: Mutex::new(None),
    });
    let engine = Arc::new(FileEngine {
        native: Arc::into_raw(Arc::clone(&c)).cast_mut().cast(),
        api: ffi::FileApi {
            create,
            open,
            submit,
            drain,
            release: release_batch,
            close,
            destroy,
        },
        gds: true,
        retained: AtomicBool::new(false),
        files: Mutex::new(Files::default()),
    });
    let path = std::env::temp_dir().join(format!(
        "tent-owner-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    File::create(&path).unwrap().set_len(8192).unwrap();
    let file = engine.open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    (engine, file, c)
}

#[test]
fn file_receipts_release_only_physically_drained_backing_and_never_replay() {
    // CPU callbacks drive the actual Rust owners; this is not storage/GPU evidence.
    for fault in [
        Fault::None,
        Fault::Reject,
        Fault::Short,
        Fault::Pending,
        Fault::Release,
    ] {
        let (engine, file, c) = fixture(fault);
        let owner = Arc::new(vec![0x5au8; 1024]);
        let weak = Arc::downgrade(&owner);
        let slices = [0, 512].map(|offset| TransferSlice {
            local: std::ptr::NonNull::new(owner.as_ptr().wrapping_add(offset).cast_mut()).unwrap(),
            remote_address: offset as u64,
            length: 512,
        });
        // SAFETY: the entire source remains immutable in the transferred Arc.
        let result = unsafe {
            file.transfer(
                TransferOp::Write,
                FileTransport::IoUring,
                &slices,
                owner,
                Instant::now() + Duration::from_millis(10),
                &AtomicBool::new(false),
            )
        };
        assert_eq!(result.is_ok(), fault == Fault::None, "{fault:?}");
        assert_eq!(c.submits.load(Ordering::Relaxed), 1, "{fault:?}");
        assert_eq!(c.ranges.load(Ordering::Relaxed), 2);
        let retained = matches!(fault, Fault::Pending | Fault::Release);
        assert_eq!(weak.upgrade().is_some(), retained, "{fault:?}");
        assert_eq!(
            engine
                .shutdown(Instant::now() + Duration::from_secs(1))
                .is_err(),
            retained
        );
        drop(file);
        drop(engine);
        assert_eq!(c.destroyed.load(Ordering::Acquire), !retained);
    }
}

#[test]
fn unknown_file_close_is_not_retried_or_reported_as_shutdown() {
    let (engine, file, c) = fixture(Fault::Close);
    let context = Arc::new(());
    let context_weak = Arc::downgrade(&context);
    file.handle(FileTransport::Gds(context)).unwrap();
    assert!(file.close(Instant::now()).is_err());
    drop(file);
    assert!(engine.shutdown(Instant::now()).is_err());
    drop(engine);
    assert_eq!(c.closes.load(Ordering::Relaxed), 1);
    assert!(!c.destroyed.load(Ordering::Acquire));
    assert!(context_weak.upgrade().is_some());
}

#[test]
fn preadmission_cancel_and_deadline_never_submit_or_retain() {
    let (engine, file, c) = fixture(Fault::None);
    for canceled in [true, false] {
        let owner = Arc::new(vec![0u8; 1024]);
        let weak = Arc::downgrade(&owner);
        let slice = TransferSlice {
            local: std::ptr::NonNull::new(owner.as_ptr().cast_mut()).unwrap(),
            remote_address: 0,
            length: 1024,
        };
        // SAFETY: destination and its backing are exclusive until return.
        let result = unsafe {
            file.transfer(
                TransferOp::Read,
                FileTransport::IoUring,
                &[slice],
                owner,
                Instant::now(),
                &AtomicBool::new(canceled),
            )
        };
        assert!(matches!(
            result,
            Err(TransferError::Timeout | TransferError::Cancelled)
        ));
        assert!(weak.upgrade().is_none());
    }
    assert_eq!(c.submits.load(Ordering::Relaxed), 0);
    drop(file);
    drop(engine);
    assert!(c.destroyed.load(Ordering::Acquire));
}
