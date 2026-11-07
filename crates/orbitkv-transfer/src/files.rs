//! A separate TENT engine owns file transports and their physical receipts.

use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::mem::ManuallyDrop;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use orbitkv_mooncake_sys::{self as native, files as ffi};

use crate::{Result, TransferError, TransferOp};

const MAX_FILES: usize = 256;
const MAX_ELEMENTS: usize = 16_384;
const MAX_BYTES: usize = 64 << 20;

fn error(operation: &'static str, status: i32, transferred_bytes: u64) -> TransferError {
    TransferError::File {
        operation,
        status,
        transferred_bytes,
    }
}

/// Consecutive bytes of one file extent. Padding is generated/discarded by TENT.
#[derive(Clone, Copy)]
pub enum FileElement {
    Range { address: u64, length: usize },
    Padding(usize),
}

pub struct FileExtent<'a> {
    pub offset: u64,
    pub elements: &'a [FileElement],
}

pub enum FileTransport {
    IoUring,
    /// The caller binds this exact CUDA context before submitting. Keep its
    /// owner through cuFile handle deregistration, not just through the batch.
    Gds(Arc<dyn Send + Sync>),
}

#[derive(Default)]
struct Files {
    closed: bool,
    opening: usize,
    handles: Vec<Weak<FileSegment>>,
}

struct Opening<'a>(&'a Mutex<Files>);
impl Drop for Opening<'_> {
    fn drop(&mut self) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).opening -= 1;
    }
}

pub struct FileEngine {
    native: native::NativeEngine,
    api: ffi::FileApi,
    gds: bool,
    retained: AtomicBool,
    files: Mutex<Files>,
}

// SAFETY: the native file API synchronizes its registry; Rust serializes its
// admission and keeps every native owner alive through an explicit receipt.
unsafe impl Send for FileEngine {}
// SAFETY: the pinned file API synchronizes native state; Rust handle/admission
// state is mutex-protected and retained prevents destruction after uncertain drain.
unsafe impl Sync for FileEngine {}

impl FileEngine {
    pub fn new(native_gds: bool) -> Result<Arc<Self>> {
        let api = ffi::file_api().map_err(TransferError::NativeRuntime)?;
        let _creation = crate::engine::ENGINE_CREATE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let options = ffi::FileOptions {
            struct_size: size_of::<ffi::FileOptions>() as u32,
            io_uring_enabled: 1,
            gds_mode: if native_gds { 2 } else { 0 },
            io_uring_queue_depth: 32,
            gds_queue_depth: 32,
        };
        let mut engine = std::ptr::null_mut();
        // SAFETY: creation is serialized, options match the pinned header, and
        // file transports are installed on this new engine only.
        let status = unsafe {
            ffi::configure_file_engine();
            (api.create)(&options, &mut engine)
        };
        if status != 0 || engine.is_null() {
            return Err(error("create", if status == 0 { 3 } else { status }, 0));
        }
        Ok(Arc::new(Self {
            native: engine,
            api,
            gds: native_gds,
            retained: AtomicBool::new(false),
            files: Mutex::new(Files::default()),
        }))
    }

    pub fn open(self: &Arc<Self>, path: &Path) -> Result<Arc<FileSegment>> {
        {
            let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
            files.handles.retain(|weak| weak.strong_count() != 0);
            let maximum = if self.gds { MAX_FILES / 2 } else { MAX_FILES };
            if files.closed || files.handles.len() + files.opening >= maximum {
                return Err(error("open admission", 9, 0));
            }
            files.opening += 1;
        }
        let _opening = Opening(&self.files);
        // This fd pins the inode for the later, context-bound GDS open too.
        let file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|_| error("open inode", 4, 0))?,
        );
        let uri = CString::new(format!("file:///proc/self/fd/{}", file.as_raw_fd()))?;
        let (id, length) = self.open_native(&uri, ffi::IOURING)?;
        let file = Arc::new(FileSegment {
            engine: Arc::clone(self),
            file,
            uri,
            length,
            closed: AtomicBool::new(false),
            handles: Mutex::new(Handles {
                uring: Some(id),
                gds: None,
                retained: false,
            }),
        });
        self.files
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .handles
            .push(Arc::downgrade(&file));
        Ok(file)
    }

    fn open_native(&self, uri: &CString, transport: i32) -> Result<(u64, u64)> {
        let (mut id, mut length) = (0, 0);
        // SAFETY: engine and URI are live; outputs belong to this call.
        let status = unsafe {
            (self.api.open)(
                self.native,
                uri.as_ptr(),
                ffi::READ_WRITE,
                transport,
                &mut id,
                &mut length,
            )
        };
        if status != 0 || id == 0 {
            if status == 0 || id != 0 {
                self.retained.store(true, Ordering::Release);
                return Err(error("open receipt", ffi::UNQUIESCED, 0));
            }
            return Err(error("open", status, 0));
        }
        Ok((id, length))
    }

    pub fn shutdown(&self, deadline: Instant) -> Result<()> {
        let handles = {
            let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
            files.closed = true;
            if files.opening != 0 || self.retained.load(Ordering::Acquire) {
                return Err(error("shutdown during open", ffi::UNQUIESCED, 0));
            }
            files
                .handles
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>()
        };
        for file in handles {
            file.close(deadline)?;
        }
        Ok(())
    }
}

impl Drop for FileEngine {
    fn drop(&mut self) {
        // SAFETY: file/batch owners hold an Arc, including retained failures.
        // The C destructor independently refuses live file handles/receipts.
        if !self.retained.load(Ordering::Acquire) {
            // SAFETY: all file handles and batches have drained and this is the last
            // engine owner; retained engines bypass this sole destroy attempt.
            unsafe { (self.api.destroy)(self.native) };
        }
    }
}

struct Handles {
    uring: Option<u64>,
    gds: Option<(u64, Arc<dyn Send + Sync>)>,
    retained: bool,
}

pub struct FileSegment {
    engine: Arc<FileEngine>,
    file: Arc<File>,
    uri: CString,
    pub length: u64,
    closed: AtomicBool,
    handles: Mutex<Handles>,
}

impl FileSegment {
    fn handle(&self, transport: FileTransport) -> Result<u64> {
        if self
            .engine
            .files
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closed
        {
            return Err(error("closed engine", 2, 0));
        }
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        if self.closed.load(Ordering::Acquire) || handles.retained {
            return Err(error("closed file", 2, 0));
        }
        match transport {
            FileTransport::IoUring => handles.uring.ok_or_else(|| error("closed file", 2, 0)),
            FileTransport::Gds(context) => {
                if !self.engine.gds {
                    return Err(error("GDS disabled", 2, 0));
                }
                if handles.gds.is_none() {
                    let (id, length) = self.engine.open_native(&self.uri, ffi::GDS)?;
                    handles.gds = Some((id, context));
                    if length != self.length {
                        self.closed.store(true, Ordering::Release);
                        return Err(error("changed file length", 1, 0));
                    }
                }
                handles
                    .gds
                    .as_ref()
                    .map(|(id, _)| *id)
                    .ok_or_else(|| error("GDS open", 3, 0))
            }
        }
    }

    /// # Safety
    /// `owner` retains actual backing, native registrations and extent leases.
    /// Sources are ready and immutable; READ destinations are exclusive. The
    /// GDS route carries the caller's bound CUDA context. No WRITE is replayed.
    /// Wait for physical drain before a dependent or overlapping file access.
    pub unsafe fn transfer<T: Send + Sync + 'static>(
        self: &Arc<Self>,
        operation: TransferOp,
        transport: FileTransport,
        extent: FileExtent<'_>,
        owner: Arc<T>,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<usize> {
        let FileExtent { offset, elements } = extent;
        if elements.is_empty() || elements.len() > MAX_ELEMENTS {
            return Err(error("element budget", 1, 0));
        }
        let mut expected = 0u64;
        let mut native = Vec::with_capacity(elements.len());
        for element in elements {
            let (kind, address, length) = match *element {
                FileElement::Range { address, length } => {
                    if address == 0 || address.checked_add(length as u64).is_none() {
                        return Err(error("invalid range", 1, 0));
                    }
                    (ffi::RANGE, address as *mut _, length)
                }
                FileElement::Padding(length) => (ffi::PADDING, std::ptr::null_mut(), length),
            };
            if length == 0 {
                return Err(error("empty element", 1, 0));
            }
            expected = expected
                .checked_add(length as u64)
                .filter(|bytes| *bytes <= MAX_BYTES as u64)
                .ok_or_else(|| error("element byte budget", 1, 0))?;
            native.push(ffi::FileElement {
                kind,
                address,
                length: length as u64,
            });
        }
        offset
            .checked_add(expected)
            .filter(|end| *end <= self.length && i64::try_from(*end).is_ok())
            .ok_or_else(|| error("file extent", 1, 0))?;
        if cancel.load(Ordering::Acquire) {
            return Err(TransferError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(TransferError::Timeout);
        }
        let file = self.handle(transport)?;
        // Keep a Rust reference until the receipt, so native release is never
        // the last backing/context/extent reference and cannot reenter the API.
        let mut owners = ManuallyDrop::new((Arc::clone(self), owner));
        let token = ffi::FileOwner {
            context: Arc::as_ptr(&owners.1).cast_mut().cast(),
            retain: retain::<T>,
            release: release::<T>,
        };
        let mut batch = 0;
        // SAFETY: range descriptors are copied at admission, and retain takes a
        // native Arc reference before any worker can access the bytes.
        let status = unsafe {
            (self.engine.api.submitv)(
                self.engine.native,
                file,
                operation.code(),
                offset,
                native.as_ptr(),
                native.len(),
                &token,
                &mut batch,
            )
        };
        if status != 0 && batch == 0 {
            // SAFETY: rejection promises no callback and no admitted I/O.
            unsafe { ManuallyDrop::drop(&mut owners) };
            return Err(error("submit", status, 0));
        }
        if status != 0 || batch == 0 {
            self.engine.retained.store(true, Ordering::Release);
            return Err(error("submit receipt", ffi::UNQUIESCED, 0));
        }
        loop {
            let canceled = cancel.load(Ordering::Acquire);
            let expired = Instant::now() >= deadline;
            let mut receipt = ffi::FileResult::default();
            // Polling with zero does not impose a fresh native deadline. At our
            // deadline/cancel, close further dispatch and retain on unknown drain.
            // SAFETY: the batch and every owner remain live above.
            let status = unsafe {
                (self.engine.api.drain)(
                    self.engine.native,
                    batch,
                    u64::from(canceled || expired),
                    i32::from(canceled || expired),
                    &mut receipt,
                )
            };
            if receipt.drained == 1 {
                // SAFETY: native I/O, staging and the owner's release callback
                // finished. The logical outcome alone never reaches this call.
                let released = unsafe { (self.engine.api.release)(self.engine.native, batch) };
                if released != 0 {
                    self.engine.retained.store(true, Ordering::Release);
                    return Err(error("release", ffi::UNQUIESCED, receipt.transferred_bytes));
                }
                // SAFETY: both physical drain and native receipt cleanup succeeded.
                unsafe { ManuallyDrop::drop(&mut owners) };
                if status != receipt.outcome {
                    return Err(error("inconsistent receipt", 4, receipt.transferred_bytes));
                }
                if status != 0 {
                    return Err(error("drain", status, receipt.transferred_bytes));
                }
                if receipt.transferred_bytes != expected {
                    return Err(error("short transfer", 5, receipt.transferred_bytes));
                }
                if canceled {
                    return Err(TransferError::Cancelled);
                }
                if expired {
                    return Err(TransferError::Timeout);
                }
                return Ok(expected as usize);
            }
            if status != ffi::UNQUIESCED || canceled || expired || receipt.outcome == 4 {
                self.engine.retained.store(true, Ordering::Release);
                return Err(error("drain", ffi::UNQUIESCED, receipt.transferred_bytes));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn close(&self, deadline: Instant) -> Result<()> {
        let mut handles = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        self.closed.store(true, Ordering::Release);
        if handles.retained {
            return Err(error("retained close", ffi::UNQUIESCED, 0));
        }
        for id in [handles.uring, handles.gds.as_ref().map(|(id, _)| *id)]
            .into_iter()
            .flatten()
        {
            let mut receipt = ffi::FileResult::default();
            let millis = deadline
                .saturating_duration_since(Instant::now())
                .as_millis()
                .min(86_400_000) as u64;
            // SAFETY: handles stay owned and the C close fences admission before drain.
            let status =
                unsafe { (self.engine.api.close)(self.engine.native, id, millis, &mut receipt) };
            if status != 0 || receipt.drained != 1 {
                handles.retained = true;
                self.engine.retained.store(true, Ordering::Release);
                return Err(error("close", ffi::UNQUIESCED, receipt.transferred_bytes));
            }
            if handles.uring == Some(id) {
                handles.uring = None;
            } else {
                handles.gds = None;
            }
        }
        Ok(())
    }
}

unsafe extern "C" fn retain<T>(context: *mut std::ffi::c_void) {
    // SAFETY: context is a live Arc<T> pointer, held by the submitting Rust owner.
    unsafe { Arc::increment_strong_count(context.cast::<T>()) };
}
unsafe extern "C" fn release<T>(context: *mut std::ffi::c_void) {
    // SAFETY: consumes only native retain's reference. The Rust reference remains
    // held through the receipt, so this cannot run backing destructors or reenter.
    unsafe { Arc::decrement_strong_count(context.cast::<T>()) };
}

impl Drop for FileSegment {
    #[allow(
        clippy::mem_forget,
        reason = "unknown file close retains its exact inode, context and engine"
    )]
    fn drop(&mut self) {
        if let Err(error) = self.close(Instant::now() + Duration::from_secs(5)) {
            log::error!("Retaining TENT file owner: {error}");
            let context = self
                .handles
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .gds
                .take();
            std::mem::forget((Arc::clone(&self.engine), Arc::clone(&self.file), context));
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/files.rs"]
mod tests;
