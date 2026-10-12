//! Shared payload memory pinned independently in each CUDA process.
//!
//! Every pool shard owns a size-sealed memfd mapped with `MAP_SHARED`. Regular
//! and reserved huge pages use the same backing and registration path. Manager
//! allocation binds the mapping to the requested NUMA node before first-touch,
//! CUDA registration or export to another process.
//!
//! Owners must drain GPU access before dropping a mapping. Another process can
//! keep the backing alive with its own mapping and CUDA registration after the
//! allocating process exits.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::OnceLock;

use crate::memory::numa::{NumaNode, pin_thread_to_numa_node};
use cudarc::runtime::sys as rt;

/// Cached huge page size from /proc/meminfo
static HUGE_PAGE_SIZE: OnceLock<Option<usize>> = OnceLock::new();

/// Read the system's default huge page size from /proc/meminfo.
/// Returns None if reading or parsing fails.
fn get_huge_page_size() -> Option<usize> {
    *HUGE_PAGE_SIZE.get_or_init(read_hugepage_size_from_proc)
}

/// Parse Hugepagesize from /proc/meminfo (in kB, convert to bytes)
fn read_hugepage_size_from_proc() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        // Format: "Hugepagesize:       2048 kB"
        if line.starts_with("Hugepagesize:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // parts = ["Hugepagesize:", "2048", "kB"]
            if parts.len() == 3 && parts[2] == "kB" {
                let kb: usize = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// Error type for pinned memory allocation.
#[derive(Debug)]
pub(crate) enum PinnedMemError {
    /// memfd creation, sizing, or sealing failed.
    BackingFailed(io::Error),
    /// mmap failed
    MmapFailed(io::Error),
    /// Explicit NUMA placement of a mapped payload failed.
    NumaBindFailed(NumaNode, io::Error),
    /// The requested or rounded size cannot be represented by mmap/ftruncate.
    SizeOverflow,
    /// cudaHostRegister failed
    CudaRegisterFailed(rt::cudaError),
    /// cudaHostGetDevicePointer failed
    CudaGetDevicePointerFailed(rt::cudaError),
    /// Size must be greater than zero
    ZeroSize,
    /// Failed to determine huge page size from /proc/meminfo
    HugePageSizeUnavailable,
}

impl std::fmt::Display for PinnedMemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MmapFailed(e) => write!(f, "mmap failed: {}", e),
            Self::NumaBindFailed(node, e) => write!(f, "shared payload bind to {node} failed: {e}"),
            Self::BackingFailed(e) => write!(f, "shared payload backing failed: {}", e),
            Self::SizeOverflow => write!(f, "shared payload size exceeds the mapping limit"),
            Self::CudaRegisterFailed(e) => write!(f, "cudaHostRegister failed: {:?}", e),
            Self::CudaGetDevicePointerFailed(e) => {
                write!(f, "cudaHostGetDevicePointer failed: {:?}", e)
            }
            Self::ZeroSize => write!(f, "size must be greater than zero"),
            Self::HugePageSizeUnavailable => write!(
                f,
                "cannot determine huge page size: Hugepagesize not found in /proc/meminfo"
            ),
        }
    }
}

impl std::error::Error for PinnedMemError {}

/// Page allocation policy for a shared payload memfd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PagePolicy {
    Regular,
    /// Requires reserved default-size huge pages on the host.
    HugePages,
}

/// RAII wrapper for CUDA pinned memory.
///
/// Memory is automatically freed/unmapped and unregistered when dropped.
pub(crate) struct PinnedMemory {
    ptr: NonNull<u8>,
    device_ptr: NonNull<u8>,
    size: usize,
    fd: OwnedFd,
}

impl std::fmt::Debug for PinnedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedMemory")
            .field("ptr", &format!("{:p}", self.ptr.as_ptr()))
            .field("device_ptr", &format!("{:p}", self.device_ptr.as_ptr()))
            .field("size", &self.size)
            .field("fd", &self.fd.as_raw_fd())
            .finish()
    }
}

// SAFETY: PinnedMemory owns a pinned memory region that is:
// - Fixed in physical memory (pinned by CUDA)
// - Safe to access from any thread
// The pointer is valid for the lifetime of this struct.
// Sync is safe because PinnedMemory exposes only an immutable pointer via
// as_ptr() and has no interior mutability.
unsafe impl Send for PinnedMemory {}
unsafe impl Sync for PinnedMemory {}

impl PinnedMemory {
    /// Allocate shared payload pages and register this process's mapping with CUDA.
    pub(crate) fn allocate(
        size: usize,
        pages: PagePolicy,
        node: NumaNode,
    ) -> Result<Self, PinnedMemError> {
        let (fd, size) = create_backing(size, pages)?;
        // SAFETY: fd owns a size-sealed file of exactly size bytes. The mapping
        // is writable and shared, and its result is checked before use.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(PinnedMemError::MmapFailed(io::Error::last_os_error()));
        }

        if node.is_valid()
            && let Err(error) = bind_payload_mapping(ptr, size, node)
        {
            // SAFETY: ptr was successfully mmap'd above, but never exported.
            unsafe { libc::munmap(ptr, size) };
            return Err(PinnedMemError::NumaBindFailed(node, error));
        }
        parallel_pre_touch(ptr.cast::<u8>(), size, node);

        // SAFETY: ptr is a valid mapping of size bytes (checked above).
        let result = unsafe { rt::cudaHostRegister(ptr, size, rt::cudaHostRegisterMapped) };
        if result != rt::cudaError::cudaSuccess {
            // SAFETY: ptr was successfully mmap'd above.
            unsafe { libc::munmap(ptr, size) };
            return Err(PinnedMemError::CudaRegisterFailed(result));
        }

        let ptr = NonNull::new(ptr.cast::<u8>()).expect("mmap returned null");
        let device_ptr = match mapped_device_pointer(ptr) {
            Ok(device_ptr) => device_ptr,
            Err(err) => {
                // SAFETY: ptr was successfully registered and mmap'd above.
                unsafe {
                    rt::cudaHostUnregister(ptr.as_ptr().cast());
                    libc::munmap(ptr.as_ptr().cast(), size);
                }
                return Err(err);
            }
        };
        Ok(Self {
            ptr,
            device_ptr,
            size,
            fd,
        })
    }

    pub(crate) fn export_fd(&self) -> io::Result<OwnedFd> {
        self.fd.try_clone()
    }

    /// Get a raw pointer to the allocated memory.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn device_ptr(&self) -> NonNull<u8> {
        self.device_ptr
    }

    /// Get the size of the allocation in bytes.
    ///
    /// This is the aligned size, which may be larger than the requested size.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.size
    }
}

fn numa_binding_mask(node: NumaNode) -> (Vec<libc::c_ulong>, usize) {
    let bits = libc::c_ulong::BITS as usize;
    let mut mask = vec![0 as libc::c_ulong; (node.0 as usize + 1).div_ceil(bits)];
    mask[node.0 as usize / bits] |= 1 << (node.0 as usize % bits);
    let maxnode = mask.len() * bits + 1;
    (mask, maxnode)
}

fn bind_payload_mapping(ptr: *mut libc::c_void, size: usize, node: NumaNode) -> io::Result<()> {
    let (mask, maxnode) = numa_binding_mask(node);
    // SAFETY: caller owns an unfaulted mapping of size bytes. mask holds the
    // kernel's maxnode-1 bits and remains live for mbind. Shared policy overrides
    // inherited interleave before any CUDA registration.
    let bound = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            ptr,
            size,
            libc::MPOL_BIND | libc::MPOL_F_STATIC_NODES,
            mask.as_ptr(),
            maxnode,
            0 as libc::c_uint,
        )
    };
    if bound != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn create_backing(size: usize, pages: PagePolicy) -> Result<(OwnedFd, usize), PinnedMemError> {
    if size == 0 {
        return Err(PinnedMemError::ZeroSize);
    }
    let size = match pages {
        PagePolicy::Regular => size,
        PagePolicy::HugePages => {
            let page = get_huge_page_size().ok_or(PinnedMemError::HugePageSizeUnavailable)?;
            size.checked_add(page - 1)
                .map(|rounded| rounded / page * page)
                .ok_or(PinnedMemError::SizeOverflow)?
        }
    };
    if size > isize::MAX as usize || i64::try_from(size).is_err() {
        return Err(PinnedMemError::SizeOverflow);
    }
    let flags = libc::MFD_CLOEXEC
        | libc::MFD_ALLOW_SEALING
        | if pages == PagePolicy::HugePages {
            libc::MFD_HUGETLB
        } else {
            0
        };
    // SAFETY: the name is NUL-terminated and flags select a shared memory file.
    let raw_fd = unsafe { libc::memfd_create(c"orbitkv-payload".as_ptr(), flags) };
    if raw_fd == -1 {
        return Err(PinnedMemError::BackingFailed(io::Error::last_os_error()));
    }
    // SAFETY: memfd_create returned a new descriptor owned by this function.
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    // SAFETY: size fits off_t and fd identifies a newly created writable memfd.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) } == -1 {
        return Err(PinnedMemError::BackingFailed(io::Error::last_os_error()));
    }
    // Preserve writable shared mappings while preventing truncation and resize.
    // SAFETY: fd is a sealing-enabled memfd and the seal bitmask is valid.
    if unsafe {
        libc::fcntl(
            fd.as_raw_fd(),
            libc::F_ADD_SEALS,
            libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
        )
    } == -1
    {
        return Err(PinnedMemError::BackingFailed(io::Error::last_os_error()));
    }
    Ok((fd, size))
}

fn mapped_device_pointer(host_ptr: NonNull<u8>) -> Result<NonNull<u8>, PinnedMemError> {
    let mut device_ptr: *mut libc::c_void = std::ptr::null_mut();
    // SAFETY: host_ptr is mapped pinned memory allocated/registered by CUDA.
    let result = unsafe {
        rt::cudaHostGetDevicePointer(&mut device_ptr, host_ptr.as_ptr() as *mut libc::c_void, 0)
    };
    if result != rt::cudaError::cudaSuccess {
        return Err(PinnedMemError::CudaGetDevicePointerFailed(result));
    }
    Ok(NonNull::new(device_ptr as *mut u8).expect("cudaHostGetDevicePointer returned null"))
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        // SAFETY: this mapping was registered in this process. Its owner must
        // have drained CUDA work before releasing the final PinnedAllocation.
        let result = unsafe { rt::cudaHostUnregister(self.ptr.as_ptr().cast()) };
        if result != rt::cudaError::cudaSuccess && result != rt::cudaError::cudaErrorCudartUnloading
        {
            eprintln!("Warning: cudaHostUnregister failed: {:?}", result);
        }
        // SAFETY: ptr was mapped with this size; other processes own independent
        // mappings and registrations of the same file.
        if unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.size) } == -1 {
            eprintln!("Warning: munmap failed: {}", io::Error::last_os_error());
        }
    }
}

/// Fault in every page of `[ptr, ptr+size)` across worker threads pinned to
/// `node`. Each thread owns a disjoint chunk and writes one byte per page
/// (plus a tail byte) to force materialization. First-touch on a pinned
/// thread places each page on `node`'s local memory.
///
/// If `node.is_unknown()`, threads run with the calling thread's existing
/// affinity (typically set by `run_on_numa` at the call site).
fn parallel_pre_touch(ptr: *mut u8, size: usize, node: NumaNode) {
    let page = page_size();
    let threads = touch_threads(size);
    let chunk = size.div_ceil(threads);
    // *mut u8 is not Send; smuggle as usize and reconstruct inside the scope.
    // Lifetimes are bounded by thread::scope.
    let base = ptr as usize;

    std::thread::scope(|s| {
        for i in 0..threads {
            let off = i * chunk;
            if off >= size {
                break;
            }
            let len = chunk.min(size - off);
            s.spawn(move || {
                if node.is_valid()
                    && let Err(e) = pin_thread_to_numa_node(node)
                {
                    log::warn!("pre-touch NUMA pin to {} failed: {}", node, e);
                }
                let p = (base + off) as *mut u8;
                let mut off = 0usize;
                while off < len {
                    // SAFETY: chunk is a disjoint sub-range of the caller's mapping.
                    unsafe { p.add(off).write_volatile(0u8) };
                    off += page;
                }
                // SAFETY: len > 0 is enforced by the chunk bookkeeping above.
                unsafe { p.add(len - 1).write_volatile(0u8) };
            });
        }
    });
}

/// Pick worker count for pre-touch.
///
/// Spinning up many threads for a small allocation costs more than it saves —
/// each thread must at least pay a NUMA pin + page-walk. Grant each worker a
/// minimum chunk so tiny allocations stay single-threaded and large ones still
/// scale up to the CPU count.
fn touch_threads(size: usize) -> usize {
    const MIN_BYTES_PER_THREAD: usize = 1 << 30; // 1 GiB
    let by_size = size.div_ceil(MIN_BYTES_PER_THREAD).max(1);
    let by_cpu = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .max(1);
    by_size.min(by_cpu)
}

fn page_size() -> usize {
    // SAFETY: sysconf with a valid name; non-positive result is fallback-handled.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

#[cfg(test)]
#[path = "../../tests/unit/memory/pinned.rs"]
mod tests;
