use super::*;

use std::io::{Read, Write};
use std::os::fd::RawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

const GPU_CHILD_SOCKET: &str = "ORBITKV_PINNED_GPU_CHILD_SOCKET";
const CHILD_TEST: &str = "memory::pinned::tests::shared_backing_gpu_child";
const CHILD_TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn shared_backing_is_size_sealed() {
    let (fd, size) = create_backing(4096, PagePolicy::Regular).unwrap();
    // SAFETY: fcntl only inspects the owned descriptor's seals.
    let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
    assert_eq!(
        seals,
        libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL
    );
    for changed_size in [0, size * 2] {
        // SAFETY: the owned memfd remains live throughout the call.
        assert_eq!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), changed_size as libc::off_t) },
            -1
        );
        assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EPERM));
    }
    assert_eq!(
        std::fs::File::from(fd).metadata().unwrap().len(),
        size as u64
    );
}

#[test]
fn invalid_sizes_fail_before_allocation() {
    for pages in [PagePolicy::Regular, PagePolicy::HugePages] {
        assert!(matches!(
            PinnedMemory::allocate(0, pages, NumaNode::UNKNOWN),
            Err(PinnedMemError::ZeroSize)
        ));
        assert!(matches!(
            PinnedMemory::allocate(usize::MAX, pages, NumaNode::UNKNOWN),
            Err(PinnedMemError::SizeOverflow)
        ));
    }
}

#[test]
fn numa_binding_preserves_the_selected_bit_at_kernel_word_boundaries() {
    let bits = libc::c_ulong::BITS as usize;
    for node in [0, 1, bits - 1, bits, bits + 1, bits * 2 - 1] {
        let (mask, maxnode) = numa_binding_mask(NumaNode(node as u32));
        // Linux get_nodes consumes maxnode - 1 bits, including across words.
        let selected: Vec<_> = (0..maxnode - 1)
            .filter(|bit| mask[bit / bits] & (1 << (bit % bits)) != 0)
            .collect();
        assert_eq!(selected, [node]);
    }
}

#[test]
#[ignore = "requires a NUMA host allowing mbind and get_mempolicy; CPU-only allocation gate"]
fn payload_binding_installs_a_real_shared_memory_policy() {
    const MPOL_F_ADDR: libc::c_int = 1 << 1;
    let node = NumaNode(0);
    let (fd, size) = create_backing(4096, PagePolicy::Regular).unwrap();
    // SAFETY: fd owns a size-sealed file and the mapping result is checked.
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
    assert_ne!(ptr, libc::MAP_FAILED);
    let binding = bind_payload_mapping(ptr, size, node);
    let (mut mask, maxnode) = numa_binding_mask(node);
    mask.fill(0);
    let mut mode = 0 as libc::c_int;
    // SAFETY: ptr is mapped; mode and mask are writable through get_mempolicy.
    let query = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode,
            mask.as_mut_ptr(),
            maxnode,
            ptr,
            MPOL_F_ADDR,
        )
    };
    let query_error = io::Error::last_os_error();
    // SAFETY: ptr was mapped above and no CUDA registration took place.
    assert_eq!(unsafe { libc::munmap(ptr, size) }, 0);
    binding.unwrap();
    assert_eq!(query, 0, "{query_error}");
    assert_eq!(mode, libc::MPOL_BIND | libc::MPOL_F_STATIC_NODES);
    assert_eq!(mask, [1]);
}

#[test]
fn unavailable_numa_node_fails_before_cuda_registration() {
    let Ok(nodes) = std::fs::read_dir("/sys/devices/system/node") else {
        return;
    };
    let Some(node) = nodes
        .filter_map(|entry| {
            entry
                .ok()?
                .file_name()
                .to_str()?
                .strip_prefix("node")?
                .parse::<u32>()
                .ok()
        })
        .max()
        .and_then(|node| node.checked_add(1))
    else {
        return;
    };
    assert!(matches!(
        PinnedMemory::allocate(4096, PagePolicy::Regular, NumaNode(node)),
        Err(PinnedMemError::NumaBindFailed(NumaNode(actual), _)) if actual == node
    ));
}

#[test]
fn test_read_hugepage_size() {
    let size = read_hugepage_size_from_proc().expect("Hugepagesize missing from /proc/meminfo");
    assert!(size >= 2 * 1024 * 1024);
    assert!(size.is_power_of_two());
}

#[test]
#[ignore = "requires a CUDA GPU; qualifies shared payload pool lifetime across exec processes"]
fn shared_backing_survives_parent_cuda_unregister() {
    // SAFETY: select the first visible GPU before allocating or registering memory.
    cuda_ok(unsafe { rt::cudaSetDevice(0) });
    let memory = PinnedMemory::allocate(256 * 1024, PagePolicy::Regular, NumaNode::UNKNOWN)
        .expect("allocate Manager payload backing");
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("payload.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", CHILD_TEST, "--nocapture"])
            .env(GPU_CHILD_SOCKET, &socket)
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + CHILD_TIMEOUT;
    let mut connection = loop {
        match listener.accept() {
            Ok((connection, _)) => break connection,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                assert!(
                    child.0.try_wait().unwrap().is_none(),
                    "GPU child exited before attach"
                );
                assert!(Instant::now() < deadline, "GPU child attach timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("accept GPU child: {error}"),
        }
    };
    set_timeouts(&connection);
    send_backing(&connection, &memory.fd, memory.size());
    let mut ready = [0u8];
    connection.read_exact(&mut ready).unwrap();
    assert_eq!(ready, [b'R']);

    // The child has already mapped and registered the same physical payload.
    // Write after its ready message to catch an accidental private/copy backing.
    for offset in 0..memory.size() {
        // SAFETY: the Manager exclusively writes the live mapping until 'D'.
        unsafe { memory.ptr.as_ptr().add(offset).write(payload_byte(offset)) };
    }
    drop(memory);
    connection.write_all(b"D").unwrap();
    let mut completed = [0u8];
    connection.read_exact(&mut completed).unwrap();
    assert_eq!(completed, [b'C']);
    child.wait_success();
}

#[test]
#[ignore = "private exec child of shared_backing_survives_parent_cuda_unregister"]
fn shared_backing_gpu_child() {
    let Some(socket) = std::env::var_os(GPU_CHILD_SOCKET) else {
        return;
    };
    let mut connection = UnixStream::connect(socket).unwrap();
    set_timeouts(&connection);
    let (fd, size) = receive_backing(&connection);
    // SAFETY: this is an exec process with its own CUDA runtime/context.
    cuda_ok(unsafe { rt::cudaSetDevice(0) });
    let memory = import_backing(fd, size);
    connection.write_all(b"R").unwrap();
    let mut dropped = [0u8];
    connection.read_exact(&mut dropped).unwrap();
    assert_eq!(dropped, [b'D']);

    // Manager mapping, CUDA registration and FD are all gone. The child still
    // owns its mapping and independent registration through both DMA directions.
    let mut device = std::ptr::null_mut();
    // SAFETY: device is a valid output location and size is the received backing size.
    cuda_ok(unsafe { rt::cudaMalloc(&mut device, size) });
    let device = DeviceBuffer(device);
    // SAFETY: both allocations span size bytes. cudaMemcpy completes before the
    // host buffer is overwritten or inspected by the next step.
    unsafe {
        cuda_ok(rt::cudaMemcpy(
            device.0,
            memory.ptr.as_ptr().cast(),
            size,
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ));
        memory.ptr.as_ptr().write_bytes(0, size);
        cuda_ok(rt::cudaMemcpy(
            memory.ptr.as_ptr().cast(),
            device.0,
            size,
            rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ));
        for (offset, actual) in std::slice::from_raw_parts(memory.ptr.as_ptr(), size)
            .iter()
            .copied()
            .enumerate()
        {
            assert_eq!(actual, payload_byte(offset), "payload byte {offset}");
        }
    }
    drop(device);
    drop(memory);
    connection.write_all(b"C").unwrap();
}

// Import is deliberately a private fixture until the executor owns a production
// mapping protocol. In particular, it must not run the allocator's pre-touch and
// overwrite live Manager payload bytes.
fn import_backing(fd: OwnedFd, size: usize) -> PinnedMemory {
    // SAFETY: the fixture receives a size-sealed memfd and its exact size.
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
    assert_ne!(ptr, libc::MAP_FAILED);
    // SAFETY: this process owns the successful mapping above.
    cuda_ok(unsafe { rt::cudaHostRegister(ptr, size, rt::cudaHostRegisterMapped) });
    let ptr = NonNull::new(ptr.cast::<u8>()).unwrap();
    PinnedMemory {
        ptr,
        device_ptr: mapped_device_pointer(ptr).unwrap(),
        size,
        fd,
    }
}

fn payload_byte(offset: usize) -> u8 {
    ((offset * 17 + offset / 251) % 251) as u8
}

fn cuda_ok(result: rt::cudaError) {
    assert_eq!(result, rt::cudaError::cudaSuccess);
}

fn set_timeouts(connection: &UnixStream) {
    connection.set_read_timeout(Some(CHILD_TIMEOUT)).unwrap();
    connection.set_write_timeout(Some(CHILD_TIMEOUT)).unwrap();
}

fn send_backing(connection: &UnixStream, fd: &OwnedFd, size: usize) {
    let mut payload = (size as u64).to_le_bytes();
    let mut iov = libc::iovec {
        iov_base: payload.as_mut_ptr().cast(),
        iov_len: payload.len(),
    };
    let mut control = [0usize; 8];
    // SAFETY: all pointer/length pairs below identify live, correctly aligned
    // stack storage; the ancillary message contains exactly one owned descriptor.
    let result = unsafe {
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = libc::CMSG_SPACE(size_of::<RawFd>() as _) as _;
        let header = libc::CMSG_FIRSTHDR(&message);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(size_of::<RawFd>() as _) as _;
        libc::CMSG_DATA(header)
            .cast::<RawFd>()
            .write(fd.as_raw_fd());
        libc::sendmsg(connection.as_raw_fd(), &message, libc::MSG_NOSIGNAL)
    };
    assert_eq!(
        result,
        payload.len() as isize,
        "send memfd: {}",
        io::Error::last_os_error()
    );
}

fn receive_backing(connection: &UnixStream) -> (OwnedFd, usize) {
    let mut payload = [0u8; 8];
    let mut iov = libc::iovec {
        iov_base: payload.as_mut_ptr().cast(),
        iov_len: payload.len(),
    };
    let mut control = [0usize; 8];
    // SAFETY: the buffers are aligned and live through recvmsg. The parent sends
    // a single SCM_RIGHTS descriptor, which becomes owned by this child.
    let fd = unsafe {
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = size_of_val(&control);
        let received = libc::recvmsg(
            connection.as_raw_fd(),
            &mut message,
            libc::MSG_CMSG_CLOEXEC | libc::MSG_WAITALL,
        );
        assert_eq!(received, payload.len() as isize, "receive memfd");
        assert_eq!(message.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC), 0);
        let header = libc::CMSG_FIRSTHDR(&message);
        assert!(!header.is_null());
        assert_eq!((*header).cmsg_level, libc::SOL_SOCKET);
        assert_eq!((*header).cmsg_type, libc::SCM_RIGHTS);
        assert_eq!(
            (*header).cmsg_len,
            libc::CMSG_LEN(size_of::<RawFd>() as _) as usize
        );
        OwnedFd::from_raw_fd(libc::CMSG_DATA(header).cast::<RawFd>().read())
    };
    (fd, usize::try_from(u64::from_le_bytes(payload)).unwrap())
}

struct DeviceBuffer(*mut libc::c_void);

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: the fixture completed all copies before releasing its buffer.
        unsafe { rt::cudaFree(self.0) };
    }
}

struct ChildGuard(Child);

impl ChildGuard {
    fn wait_success(&mut self) {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success(), "GPU child failed: {status}");
                return;
            }
            assert!(Instant::now() < deadline, "GPU child exit timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
