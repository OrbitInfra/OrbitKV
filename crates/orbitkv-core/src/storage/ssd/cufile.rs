//! Optional cuFile reads and writes through bounded, registered GPU staging.
//!
//! The library is process-owned; ordinary io_uring deployments never load it.
//! cuFile may use its CPU compatibility path. Selecting this backend alone is
//! not evidence of native GDS; qualification must disable compatibility mode.

use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUstream;
use libloading::Library;

use super::GpuIo;
use crate::cost::ExecutionResource;

pub(crate) const ALIGNMENT: usize = 4096;
pub(crate) const STAGING_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const STAGING_SLOTS: usize = 2;

mod slot;
pub(crate) use slot::GpuSlot;

#[repr(C)]
#[derive(Clone, Copy)]
struct Status {
    error: i32,
    cuda_error: i32,
}

impl Status {
    fn check(self, operation: &str) -> Result<(), String> {
        if self.error == 0 {
            Ok(())
        } else {
            Err(format!(
                "{operation}: cuFile error {}, CUDA error {}",
                self.error, self.cuda_error
            ))
        }
    }
}

#[repr(C)]
union FileHandle {
    fd: i32,
    handle: *mut c_void,
}

#[repr(C)]
struct Descriptor {
    kind: i32,
    handle: FileHandle,
    fs_ops: *const c_void,
}

type RegisterFile = unsafe extern "C" fn(*mut *mut c_void, *mut Descriptor) -> Status;
type DeregisterFile = unsafe extern "C" fn(*mut c_void);
type RegisterBuffer = unsafe extern "C" fn(*const c_void, usize, i32) -> Status;
type DeregisterBuffer = unsafe extern "C" fn(*const c_void) -> Status;
type StreamRegister = unsafe extern "C" fn(CUstream, u32) -> Status;
type StreamDeregister = unsafe extern "C" fn(CUstream) -> Status;
// Match the installed cufile.h ABI: off_t and ssize_t are 64-bit on Linux.
type AsyncIo = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut usize,
    *mut i64,
    *mut i64,
    *mut isize,
    CUstream,
) -> Status;

pub(crate) struct Cufile {
    _library: Library,
    native_only: bool,
    register_file: RegisterFile,
    deregister_file: DeregisterFile,
    register_buffer: RegisterBuffer,
    deregister_buffer: DeregisterBuffer,
    register_stream: StreamRegister,
    deregister_stream: StreamDeregister,
    read: AsyncIo,
    write: AsyncIo,
}

impl Cufile {
    fn get(native_only: bool) -> Result<Arc<Self>, String> {
        static DRIVER: OnceLock<Result<Arc<Cufile>, String>> = OnceLock::new();
        let driver = DRIVER
            .get_or_init(|| Self::load(native_only).map(Arc::new))
            .clone()?;
        if native_only && !driver.native_only {
            return Err("cuFile was already initialized without native-only configuration".into());
        }
        Ok(driver)
    }

    fn load(native_only: bool) -> Result<Self, String> {
        // SAFETY: public cuFile C ABI; the library outlives every copied symbol.
        unsafe {
            let library = Library::new("libcufile.so.0")
                .or_else(|_| Library::new("libcufile.so"))
                .map_err(|e| {
                    format!("cuFile unavailable: {e}; install NVIDIA GDS or select uring")
                })?;
            let open = *library
                .get::<unsafe extern "C" fn() -> Status>(b"cuFileDriverOpen\0")
                .map_err(|e| e.to_string())?;
            if native_only {
                // cuFile parameters are staged before DriverOpen. Avoid mutating
                // process environment after CUDA/Python worker threads exist.
                let set = *library
                    .get::<unsafe extern "C" fn(i32, bool) -> Status>(b"cuFileSetParameterBool\0")
                    .map_err(|e| format!("cuFile native-only configuration unavailable: {e}"))?;
                // CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE / FORCE_COMPAT_MODE.
                set(1, false).check("disable cuFile compatibility")?;
                set(2, false).check("disable forced cuFile compatibility")?;
            }
            let driver = Self {
                native_only,
                register_file: *library
                    .get(b"cuFileHandleRegister\0")
                    .map_err(|e| e.to_string())?,
                deregister_file: *library
                    .get(b"cuFileHandleDeregister\0")
                    .map_err(|e| e.to_string())?,
                register_buffer: *library
                    .get(b"cuFileBufRegister\0")
                    .map_err(|e| e.to_string())?,
                deregister_buffer: *library
                    .get(b"cuFileBufDeregister\0")
                    .map_err(|e| e.to_string())?,
                register_stream: *library
                    .get(b"cuFileStreamRegister\0")
                    .map_err(|e| e.to_string())?,
                deregister_stream: *library
                    .get(b"cuFileStreamDeregister\0")
                    .map_err(|e| e.to_string())?,
                read: *library
                    .get(b"cuFileReadAsync\0")
                    .map_err(|e| e.to_string())?,
                write: *library
                    .get(b"cuFileWriteAsync\0")
                    .map_err(|e| e.to_string())?,
                _library: library,
            };
            open().check("cuFileDriverOpen")?;
            log::info!(
                "SSD cuFile backend loaded (native_only={native_only}); qualify the hardware I/O path separately"
            );
            Ok(driver)
        }
    }
}

pub(crate) struct CufileFile {
    driver: Arc<Cufile>,
    handle: NonNull<c_void>,
    _file: File,
    pub(crate) gpu_io: Arc<GpuIo>,
    cost_resource: ExecutionResource,
}

// SAFETY: cuFile's file APIs are thread-safe. The handle and fd stay live until
// every SSD query/worker reference has been released.
unsafe impl Send for CufileFile {}
unsafe impl Sync for CufileFile {}

impl CufileFile {
    pub(crate) fn new(file: File, gpu_io: Arc<GpuIo>) -> Result<Self, String> {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(1);
        let cost_resource = if crate::cost::enabled() {
            let incarnation = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            match file.metadata() {
                Ok(metadata) => {
                    crate::cost::resource_id(&(incarnation, metadata.dev(), metadata.ino()))
                }
                Err(_) => crate::cost::resource_id(&(incarnation, file.as_raw_fd())),
            }
        } else {
            0
        };
        if gpu_io.automatic {
            let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
            // SAFETY: live descriptor and writable statfs output.
            if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let stat = unsafe { stat.assume_init() };
            if !matches!(stat.f_type, libc::EXT4_SUPER_MAGIC | libc::XFS_SUPER_MAGIC) {
                return Err("automatic cuFile selection requires an ext4 or XFS mount".into());
            }
        }
        let driver = Cufile::get(gpu_io.automatic)?;
        // Initialize the complete C descriptor, including unused union bytes.
        let mut descriptor: Descriptor = unsafe { std::mem::zeroed() };
        descriptor.kind = 1;
        descriptor.handle.fd = file.as_raw_fd();
        let mut handle = std::ptr::null_mut();
        // SAFETY: descriptor refers to our owned file; outputs are valid stack storage.
        unsafe { (driver.register_file)(&raw mut handle, &raw mut descriptor) }
            .check("cuFileHandleRegister")?;
        let handle = NonNull::new(handle).ok_or("cuFile returned a null file handle")?;
        Ok(Self {
            driver,
            handle,
            _file: file,
            gpu_io,
            cost_resource: ExecutionResource::SsdFile(cost_resource),
        })
    }
}

impl Drop for CufileFile {
    fn drop(&mut self) {
        // SAFETY: every submission owns an Arc until I/O and GPU copies finish.
        unsafe { (self.driver.deregister_file)(self.handle.as_ptr()) };
    }
}

/// One file interval and the corresponding engine-owned GPU range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRange {
    pub file_offset: u64,
    pub device: u64,
    pub bytes: usize,
}

#[derive(Debug)]
pub(crate) struct IoBatch {
    pub file_offset: u64,
    pub bytes: usize,
    pub copies: Vec<CopyRange>,
}

impl IoBatch {
    pub(super) fn validate(&self) -> Result<(), String> {
        let end = self
            .file_offset
            .checked_add(self.bytes as u64)
            .filter(|&end| i64::try_from(end).is_ok())
            .ok_or("SSD batch offset overflow")?;
        if self.bytes == 0
            || self.bytes > STAGING_BYTES
            || !self.bytes.is_multiple_of(ALIGNMENT)
            || !self.file_offset.is_multiple_of(ALIGNMENT as u64)
        {
            return Err("SSD batch exceeds aligned GPU staging".into());
        }
        for copy in &self.copies {
            if copy.device == 0
                || copy.file_offset < self.file_offset
                || copy
                    .file_offset
                    .checked_add(copy.bytes as u64)
                    .is_none_or(|limit| limit > end)
                || copy.device.checked_add(copy.bytes as u64).is_none()
            {
                return Err("SSD batch copy exceeds its owned ranges".into());
            }
        }
        Ok(())
    }
}

/// Merge adjacent aligned intervals, splitting large state checkpoints to keep
/// staging bounded. Read amplification is restricted to the aligned edges.
pub(crate) fn plan_reads(mut copies: Vec<CopyRange>) -> Result<Vec<IoBatch>, String> {
    copies.sort_unstable_by_key(|copy| copy.file_offset);
    let mut batches: Vec<IoBatch> = Vec::new();
    for mut copy in copies {
        copy.file_offset
            .checked_add(copy.bytes as u64)
            .filter(|&end| i64::try_from(end).is_ok())
            .ok_or("SSD read offset overflow")?;
        copy.device
            .checked_add(copy.bytes as u64)
            .ok_or("GPU destination overflow")?;
        while copy.bytes != 0 {
            let aligned = copy.file_offset / ALIGNMENT as u64 * ALIGNMENT as u64;
            let append = batches.last().is_some_and(|batch| {
                copy.file_offset >= batch.file_offset
                    && aligned <= batch.file_offset + batch.bytes as u64
                    && copy.file_offset < batch.file_offset + STAGING_BYTES as u64
            });
            if !append {
                batches.push(IoBatch {
                    file_offset: aligned,
                    bytes: 0,
                    copies: Vec::new(),
                });
            }
            let batch = batches.last_mut().expect("batch inserted");
            let offset = (copy.file_offset - batch.file_offset) as usize;
            let bytes = copy.bytes.min(STAGING_BYTES - offset);
            batch.bytes = batch
                .bytes
                .max((offset + bytes).next_multiple_of(ALIGNMENT));
            batch.copies.push(CopyRange { bytes, ..copy });
            copy.file_offset += bytes as u64;
            copy.device += bytes as u64;
            copy.bytes -= bytes;
        }
    }
    Ok(batches)
}

/// A complete object owns its aligned extent, including padding. Gather only
/// its valid GPU bytes and zero padding before any chunk becomes visible.
pub(crate) fn plan_writes(
    file_offset: u64,
    bytes: u64,
    copies: Vec<CopyRange>,
) -> Result<Vec<IoBatch>, String> {
    let end = file_offset
        .checked_add(bytes)
        .filter(|&end| i64::try_from(end).is_ok())
        .ok_or("SSD write offset overflow")?;
    if !file_offset.is_multiple_of(ALIGNMENT as u64)
        || !bytes.is_multiple_of(ALIGNMENT as u64)
        || bytes == 0
    {
        return Err("SSD writes require an aligned, nonempty reservation".into());
    }
    for copy in &copies {
        if copy.file_offset < file_offset
            || copy
                .file_offset
                .checked_add(copy.bytes as u64)
                .is_none_or(|limit| limit > end)
            || copy.device.checked_add(copy.bytes as u64).is_none()
        {
            return Err("GPU write range exceeds its reservation".into());
        }
    }
    let mut batches = Vec::new();
    let mut start = file_offset;
    while start < end {
        let limit = end.min(start + STAGING_BYTES as u64);
        let pieces = copies
            .iter()
            .filter_map(|copy| {
                let begin = copy.file_offset.max(start);
                let finish = (copy.file_offset + copy.bytes as u64).min(limit);
                (begin < finish).then(|| CopyRange {
                    file_offset: begin,
                    device: copy.device + (begin - copy.file_offset),
                    bytes: (finish - begin) as usize,
                })
            })
            .collect();
        batches.push(IoBatch {
            file_offset: start,
            bytes: (limit - start) as usize,
            copies: pieces,
        });
        start = limit;
    }
    Ok(batches)
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/cufile.rs"]
mod tests;
