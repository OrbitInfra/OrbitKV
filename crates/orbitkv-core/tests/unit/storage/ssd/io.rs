use super::*;

#[test]
fn unknown_drain_retains_actual_owners_and_exhausts_admission() {
    let admission = Arc::new(Semaphore::new(1));
    let backing = Arc::new(vec![0x5a; 512]);
    let weak = Arc::downgrade(&backing);
    let permit = admit(&admission).unwrap();
    drop(DrainOwner::new((backing, permit)));
    assert!(weak.upgrade().is_some());
    assert_eq!(
        admit(&admission).unwrap_err().kind,
        ErrorKind::ResourceExhausted
    );

    let admission = Arc::new(Semaphore::new(1));
    let backing = Arc::new(vec![0; 512]);
    let weak = Arc::downgrade(&backing);
    DrainOwner::new((backing, admit(&admission).unwrap())).drain();
    assert!(weak.upgrade().is_none());
    assert!(admit(&admission).is_ok());
}

pub(in crate::storage::ssd) fn queued_io() -> SsdIo {
    // Metadata/queue controls never start a reader or submit byte I/O.
    SsdIo {
        files: Vec::new(),
        cost_resource: ExecutionResource::SsdStore(resource_id(&"queued SSD")),
    }
}

#[cfg(any(feature = "mooncake", feature = "orbite"))]
mod engine {
    use super::*;
    use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
    use crate::memory::numa::NumaNode;
    use crate::storage::Storage;
    use crate::storage::ssd::{SsdBackend, SsdCacheConfig};
    use crate::transfer_engine::{Permission, Registration, TransferSlice};
    use std::num::{NonZeroU64, NonZeroUsize};
    use std::os::unix::fs::FileExt;

    fn file_extent(slices: Vec<TransferSlice>) -> FileExtent {
        FileExtent::from_slices(
            slices[0].remote_address,
            slices.iter().map(|s| s.length).sum(),
            slices,
        )
        .unwrap()
    }

    fn directory() -> tempfile::TempDir {
        let root = std::env::var_os("ORBITKV_SSD_TEST_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let directory = tempfile::tempdir_in(root).unwrap();
        let fd = File::open(directory.path()).unwrap();
        let mut info = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: fd is live and info has space for the kernel's statfs output.
        assert_eq!(
            unsafe { libc::fstatfs(fd.as_raw_fd(), info.as_mut_ptr()) },
            0
        );
        // SAFETY: successful fstatfs initialized info.
        let magic = unsafe { info.assume_init() }.f_type as u64;
        let expected = match std::env::var("ORBITKV_SSD_TEST_FS").as_deref() {
            Ok("ext4") => 0xef53,
            Ok("tmpfs") => 0x0102_1994,
            _ => panic!(
                "select ORBITKV_SSD_TEST_FS=ext4|tmpfs; do not silently substitute a filesystem"
            ),
        };
        assert_eq!(magic, expected, "wrong real-file test filesystem");
        directory
    }

    fn buffered(file: &File) -> File {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("/proc/self/fd/{}", file.as_raw_fd()))
            .unwrap()
    }

    fn assert_engine_direct(file: &File) {
        use std::os::unix::fs::MetadataExt;
        if std::env::var("ORBITKV_SSD_TEST_FS").as_deref() != Ok("ext4") {
            return; // tmpfs is the separate memory-filesystem profile.
        }
        let inode = file.metadata().unwrap();
        let direct = std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| {
                let Some(fd) = entry
                    .file_name()
                    .to_str()
                    .and_then(|s| s.parse::<i32>().ok())
                else {
                    return false;
                };
                if fd == file.as_raw_fd() {
                    return false;
                }
                let Ok(metadata) = std::fs::metadata(entry.path()) else {
                    return false;
                };
                // SAFETY: F_GETFL only observes this fd; a concurrently closed fd returns -1.
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                metadata.dev() == inode.dev()
                    && metadata.ino() == inode.ino()
                    && flags >= 0
                    && flags & libc::O_DIRECT != 0
            });
        assert!(direct, "engine must retain its own O_DIRECT fd on ext4");
    }

    fn selected_backend() -> crate::PeerTransferBackend {
        match std::env::var("ORBITKV_SSD_TEST_ENGINE").as_deref() {
            Ok("tent") => crate::PeerTransferBackend::Mooncake,
            Ok("orbite") => crate::PeerTransferBackend::OrbiteTcp,
            _ => panic!("select the tested engine with ORBITKV_SSD_TEST_ENGINE=tent|orbite"),
        }
    }

    fn storage(path: PathBuf, capacity: u64, shards: usize) -> Arc<Storage> {
        assert_eq!(
            std::env::var("ORBITKV_REQUIRE_ENGINE_SSD").as_deref(),
            Ok("1"),
            "run this real-file gate explicitly with ORBITKV_REQUIRE_ENGINE_SSD=1 and the CPU host-registration fixture"
        );
        if selected_backend() == crate::PeerTransferBackend::Mooncake {
            assert_eq!(
                std::env::var("MC_FORCE_TCP").as_deref(),
                Ok("1"),
                "the CPU/TCP source-drain control requires MC_FORCE_TCP=1 even on runners with RDMA devices"
            );
        }
        Storage::new_with_config(
            16 << 20,
            false,
            crate::EngineConfig {
                peer_transfer_backend: selected_backend(),
                ssd_cache_config: Some(SsdCacheConfig {
                    cache_paths: vec![path],
                    capacity_bytes: capacity,
                    shards: NonZeroUsize::new(shards).unwrap(),
                    backend: SsdBackend::Uring,
                    ..Default::default()
                }),
                ..Default::default()
            },
            &[],
        )
        .unwrap()
    }

    fn allocation(storage: &Storage, bytes: usize, value: u8) -> Arc<PinnedAllocation> {
        let allocation = storage
            .allocate(NonZeroU64::new(bytes as u64).unwrap(), None)
            .unwrap();
        // SAFETY: this fresh allocation has no other reader or writer.
        unsafe { std::ptr::write_bytes(allocation.mapped_ptr().host().as_ptr(), value, bytes) };
        allocation
    }

    fn block(storage: &Storage, encoded: bool, value: u8) -> Arc<SealedBlock> {
        let mut slots = Vec::new();
        for value in [value, value + 1] {
            let a = allocation(storage, 512, value);
            let mut raw = RawBlock::single_segment(Segment::new(a.mapped_ptr().host(), 512, a));
            if encoded {
                raw.encoding = Some(vec![crate::codec::EncodedSegment {
                    version: 1,
                    format: orbitkv_state::StorageFormat::Exact,
                    logical_bytes: 512,
                    stored_bytes: 512,
                    checksum: crc32fast::hash(&[value; 512]),
                }]);
            }
            slots.push((raw, NumaNode::UNKNOWN));
        }
        Arc::new(SealedBlock::from_slots(slots))
    }

    #[tokio::test]
    #[ignore = "real Linux file I/O plus CPU host-registration fixture; CI requires this gate"]
    async fn engine_files_preserve_bytes_metadata_shards_and_leased_generations() {
        let dir = directory();
        let storage = storage(dir.path().to_path_buf(), 2048, 2);
        let store = storage.ssd_store.as_ref().unwrap();
        assert!(
            store.io.engine_file(0).is_some(),
            "standalone SSD must create the selected engine without a global index"
        );
        let mut held = Vec::new();
        for (encoded, value) in [(false, 0x31), (true, 0x72)] {
            let key = StateKey::new("engine-file".into(), vec![value]);
            let source = block(&storage, encoded, value);
            store.ingest_batch(std::iter::once((&key, &source)), false);
            store.flush().await;
            let lease = store
                .discover(std::slice::from_ref(&key))
                .remove(0)
                .unwrap()
                .pin()
                .unwrap();
            assert_eq!(lease.entry.slots.len(), 2);
            assert_eq!(lease.entry.slots[0].encoding.is_some(), encoded);
            let mut disk = vec![0; 1024];
            buffered(&store._files[lease.entry.shard_id])
                .read_exact_at(&mut disk, lease.entry.file_offset)
                .unwrap();
            assert_eq!(&disk[..512], &[value; 512]);
            assert_eq!(&disk[512..], &[value + 1; 512]);
            drop(source);
            let restored = store
                .read_host_batch(vec![Arc::clone(&lease)])
                .await
                .unwrap()
                .remove(0)
                .1;
            for (index, slot) in restored.slots().iter().enumerate() {
                // SAFETY: the read receipt initialized and drained this complete allocation.
                let bytes = unsafe {
                    std::slice::from_raw_parts(slot.segment_ptr(0).unwrap().as_ptr(), 512)
                };
                assert_eq!(bytes, &[value + index as u8; 512]);
                assert_eq!(slot.encoding.is_some(), encoded);
            }
            held.push(lease);
        }
        assert_ne!(held[0].entry.shard_id, held[1].entry.shard_id);
        let next = StateKey::new("engine-file".into(), vec![0xff]);
        let slots = held[0].entry.slots.clone();
        assert!(
            store
                .inner
                .lock()
                .ring
                .reserve(
                    &next,
                    slots.clone(),
                    crate::storage::ssd::index::Encoding::Raw
                )
                .is_none()
        );
        drop(held);
        assert!(
            store
                .inner
                .lock()
                .ring
                .reserve(&next, slots, crate::storage::ssd::index::Encoding::Raw)
                .is_some()
        );
        store.commit_write(&next, false);
    }

    #[tokio::test]
    #[ignore = "real Linux file I/O plus CPU host-registration fixture; CI requires this gate"]
    async fn engine_files_reject_misalignment_and_truncated_files() {
        let dir = directory();
        let storage = storage(dir.path().join("cache.bin"), 4096, 1);
        let store = storage.ssd_store.as_ref().unwrap();
        let file = store.io.engine_file(0).unwrap();
        let source = allocation(&storage, 512, 0x47);
        let slice = TransferSlice {
            local: source.mapped_ptr().host(),
            remote_address: 0,
            length: 512,
        };
        let transfer = |operation, slices, memory| {
            file.submit(
                operation,
                FileTransport::IoUring,
                file_extent(slices),
                FileMemory::Pinned(vec![memory]),
                Arc::new(()),
            )
        };
        assert_eq!(
            transfer(TransferOp::Write, vec![slice], Arc::clone(&source))
                .unwrap()
                .await
                .unwrap()
                .unwrap(),
            512
        );
        for bad in [
            TransferSlice {
                remote_address: 1,
                ..slice
            },
            TransferSlice {
                length: 511,
                ..slice
            },
        ] {
            assert_eq!(
                transfer(TransferOp::Write, vec![bad], Arc::clone(&source))
                    .unwrap_err()
                    .kind,
                ErrorKind::InvalidRequest
            );
        }
        let out_of_range = TransferSlice {
            remote_address: 4096,
            ..slice
        };
        assert!(
            transfer(TransferOp::Write, vec![out_of_range], source)
                .unwrap()
                .await
                .unwrap()
                .is_err()
        );
        assert_eq!(store._files[0].metadata().unwrap().len(), 4096);
        store._files[0].set_len(512).unwrap();
        let destination = allocation(&storage, 2048, 0xa5);
        let read = TransferSlice {
            local: destination.mapped_ptr().host(),
            remote_address: 0,
            length: 1024,
        };
        let error = transfer(TransferOp::Read, vec![read], Arc::clone(&destination))
            .unwrap()
            .await
            .unwrap()
            .unwrap_err();
        // The native file owner pins its admitted length; OrbiTE refreshes it.
        let bytes =
            unsafe { std::slice::from_raw_parts(destination.mapped_ptr().host().as_ptr(), 2048) };
        if selected_backend() == crate::PeerTransferBackend::Mooncake {
            assert_eq!(error.kind, ErrorKind::Backend);
            assert_eq!(&bytes[..512], &[0x47; 512]);
            assert!(error.message.contains("512 bytes"), "{error}");
            // submitv scatters only the completed prefix, including on O_DIRECT.
            assert_eq!(&bytes[512..], &[0xa5; 1536]);
        } else {
            assert_eq!(error.kind, ErrorKind::InvalidRequest);
            assert_eq!(bytes, &[0xa5; 2048]);
        }
        // EOF falls inside padding, then inside a data range. The TENT vector
        // receipt counts the file prefix (including padding); only that prefix
        // may be scattered. OrbiTE rejects the shortened file at preflight.
        for range_bytes in [128, 512] {
            // SAFETY: the previous READ physically drained before reinitialization.
            unsafe { std::ptr::write_bytes(destination.as_non_null().as_ptr(), 0xa5, 2048) };
            let error = file
                .submit(
                    TransferOp::Read,
                    FileTransport::IoUring,
                    FileExtent {
                        offset: 0,
                        elements: vec![
                            FileIoElement::Padding(256),
                            FileIoElement::Range {
                                address: destination.as_non_null().as_ptr() as u64,
                                length: range_bytes,
                            },
                            FileIoElement::Padding(896 - 256 - range_bytes),
                            FileIoElement::Range {
                                address: destination.as_non_null().as_ptr() as u64 + 1024,
                                length: 128,
                            },
                        ],
                    },
                    FileMemory::Pinned(vec![Arc::clone(&destination)]),
                    Arc::new(()),
                )
                .unwrap()
                .await
                .unwrap()
                .unwrap_err();
            // SAFETY: the failed request has an independent physical drain receipt.
            let bytes =
                unsafe { std::slice::from_raw_parts(destination.as_non_null().as_ptr(), 2048) };
            let copied = if selected_backend() == crate::PeerTransferBackend::Mooncake {
                assert_eq!(error.kind, ErrorKind::Backend);
                assert!(error.message.contains("512 bytes"), "{error}");
                range_bytes.min(256)
            } else {
                assert_eq!(error.kind, ErrorKind::InvalidRequest);
                0
            };
            assert!(bytes[..copied].iter().all(|&b| b == 0x47));
            assert!(
                bytes[copied..].iter().all(|&b| b == 0xa5),
                "short READ overwrote its suffix"
            );
        }
        assert_eq!(file.admission.available_permits(), MAX_OPERATIONS);
    }

    #[tokio::test]
    #[ignore = "real Linux file I/O plus CPU host-registration fixture; CI requires this gate"]
    async fn engine_files_cancel_before_submit_and_drain_detached_receivers() {
        let dir = directory();
        let storage = storage(dir.path().join("cache.bin"), 4096, 1);
        let store = storage.ssd_store.as_ref().unwrap();
        let file = store.io.engine_file(0).unwrap();
        let source = allocation(&storage, 512, 0x69);
        let slices = [TransferSlice {
            local: source.mapped_ptr().host(),
            remote_address: 0,
            length: 512,
        }];
        let cancel = Cancellation::default();
        cancel.cancel();
        for (deadline, cancellation, expected) in [
            (
                Instant::now() + Duration::from_secs(1),
                &cancel,
                ErrorKind::Cancelled,
            ),
            (Instant::now(), &Cancellation::default(), ErrorKind::Timeout),
        ] {
            // SAFETY: registration and backing are held, and this destination is exclusively owned.
            let error = unsafe {
                Arc::clone(&file.backend).transfer_file(
                    TransferOp::Write,
                    &file.segment,
                    FileTransport::IoUring,
                    Arc::new(FileTransfer {
                        extent: file_extent(slices.to_vec()),
                        memory: FileMemory::Pinned(vec![Arc::clone(&source)]),
                        _lease: Arc::new(Arc::clone(&file)),
                    }),
                    deadline,
                    cancellation,
                )
            }
            .unwrap_err();
            assert_eq!(error.kind, expected);
        }
        let mut disk = [0xff; 512];
        buffered(&store._files[0])
            .read_exact_at(&mut disk, 0)
            .unwrap();
        assert_eq!(disk, [0; 512]);
        // An engine segment must remain bound to the opened inode after rename.
        let path = dir.path().join("cache.bin");
        std::fs::rename(&path, dir.path().join("original.bin")).unwrap();
        let replacement = File::create(&path).unwrap();
        replacement.set_len(4096).unwrap();
        let extent = Arc::new(());
        let weak = Arc::downgrade(&extent);
        let receiver = file
            .submit(
                TransferOp::Write,
                FileTransport::IoUring,
                file_extent(slices.to_vec()),
                FileMemory::Pinned(vec![source]),
                extent,
            )
            .unwrap();
        drop(receiver);
        tokio::time::timeout(Duration::from_secs(5), async {
            while weak.upgrade().is_some() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        buffered(&store._files[0])
            .read_exact_at(&mut disk, 0)
            .unwrap();
        assert_eq!(disk, [0x69; 512]);
        File::open(path)
            .unwrap()
            .read_exact_at(&mut disk, 0)
            .unwrap();
        assert_eq!(
            disk, [0; 512],
            "path replacement must not redirect file I/O"
        );
        assert_eq!(file.admission.available_permits(), MAX_OPERATIONS);
        if selected_backend() == crate::PeerTransferBackend::Mooncake {
            file.backend
                .shutdown(Instant::now() + Duration::from_secs(1))
                .unwrap();
        }
    }

    #[tokio::test]
    #[ignore = "real Linux vectored file I/O; CI requires both engines and filesystems"]
    async fn engine_files_vectors_zero_padding_discard_reads_and_submit_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = directory();
        let size: usize = (5 << 20) + 137;
        let total = (4096 + size).next_multiple_of(4096);
        let storage = storage(dir.path().join("vectors.bin"), total as u64, 1);
        let store = storage.ssd_store.as_ref().unwrap();
        let real = store.io.engine_file(0).unwrap();
        let backend = Arc::new(Fault {
            real: Arc::clone(&real.backend),
            kind: None,
            short: false,
            calls: AtomicUsize::new(0),
        });
        let file = Arc::new(EngineFile {
            backend: backend.clone(),
            segment: FileSegment {
                owner_epoch: real.segment.owner_epoch.clone(),
                length: real.segment.length,
                inner: real.segment.inner.clone(),
            },
            file: Arc::clone(&real.file),
            admission: Arc::new(Semaphore::new(1)),
            runtime: tokio::runtime::Handle::current(),
            cost_resource: real.cost_resource,
        });
        let first = allocation(&storage, 512, 0x71);
        let second = allocation(&storage, size.next_multiple_of(512), 0x82);
        let vector = || FileExtent {
            offset: 0,
            elements: vec![
                FileIoElement::Padding(512),
                FileIoElement::Range {
                    address: first.as_non_null().as_ptr() as u64,
                    length: 137,
                },
                FileIoElement::Padding(4096 - 512 - 137),
                FileIoElement::Range {
                    address: second.as_non_null().as_ptr() as u64,
                    length: size,
                },
                FileIoElement::Padding(total - 4096 - size),
            ],
        };
        let owners = || FileMemory::Pinned(vec![Arc::clone(&first), Arc::clone(&second)]);
        // A skipped padding WRITE must fail even on a freshly allocated shard.
        buffered(&store._files[0])
            .write_all_at(&vec![0xcc; total], 0)
            .unwrap();
        assert_eq!(
            file.submit(
                TransferOp::Write,
                FileTransport::IoUring,
                vector(),
                owners(),
                Arc::new(())
            )
            .unwrap()
            .await
            .unwrap()
            .unwrap(),
            total
        );
        assert_engine_direct(&store._files[0]);
        assert_eq!(
            backend.calls.load(Ordering::Relaxed),
            1,
            "one extent exceeds the old 4 MiB chunk size"
        );
        let mut disk = vec![0xff; total];
        buffered(&store._files[0])
            .read_exact_at(&mut disk, 0)
            .unwrap();
        let mut expected = vec![0; total];
        expected[512..649].fill(0x71);
        expected[4096..4096 + size].fill(0x82);
        assert_eq!(disk, expected, "padding is engine-generated zero data");
        for allocation in [&first, &second] {
            // SAFETY: the WRITE receipt has drained all readers.
            unsafe {
                std::ptr::write_bytes(
                    allocation.as_non_null().as_ptr(),
                    0xa5,
                    allocation.size_bytes() as usize,
                );
            };
        }
        assert_eq!(
            file.submit(
                TransferOp::Read,
                FileTransport::IoUring,
                vector(),
                owners(),
                Arc::new(())
            )
            .unwrap()
            .await
            .unwrap()
            .unwrap(),
            total
        );
        assert_eq!(backend.calls.load(Ordering::Relaxed), 2);
        for (allocation, length, value) in [(&first, 137, 0x71), (&second, size, 0x82)] {
            // SAFETY: read receipt proves initialized payload and physical drain.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    allocation.as_non_null().as_ptr(),
                    allocation.size_bytes() as usize,
                )
            };
            assert!(bytes[..length].iter().all(|&b| b == value));
            assert!(
                bytes[length..].iter().all(|&b| b == 0xa5),
                "padding must not touch memory outside the described ranges"
            );
        }
        let error = file
            .submit(
                TransferOp::Write,
                FileTransport::Gds,
                vector(),
                owners(),
                Arc::new(()),
            )
            .unwrap()
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::Unavailable,
            "unconfigured native GDS must reject before touching host bytes"
        );
        buffered(&store._files[0])
            .read_exact_at(&mut disk, 0)
            .unwrap();
        assert_eq!(disk, expected);
        for op in [TransferOp::Write, TransferOp::Read] {
            assert_eq!(
                file.submit(
                    op,
                    FileTransport::IoUring,
                    FileExtent {
                        offset: 0,
                        elements: vec![FileIoElement::Padding(total)]
                    },
                    FileMemory::Pinned(Vec::new()),
                    Arc::new(())
                )
                .unwrap()
                .await
                .unwrap()
                .unwrap(),
                total
            );
        }
        buffered(&store._files[0])
            .read_exact_at(&mut disk, 0)
            .unwrap();
        assert!(disk.iter().all(|&b| b == 0));
    }

    struct Fault {
        real: Arc<dyn TransferBackend>,
        kind: Option<ErrorKind>,
        short: bool,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl TransferBackend for Fault {
        fn transfer_endpoint(&self) -> &str {
            self.real.transfer_endpoint()
        }
        fn source_epoch(&self) -> &str {
            self.real.source_epoch()
        }
        fn accepts_endpoint(&self, endpoint: &str) -> bool {
            self.real.accepts_endpoint(endpoint)
        }
        fn register_pinned(
            self: Arc<Self>,
            allocations: &[Arc<PinnedAllocation>],
            permission: Permission,
        ) -> Result<Registration, TransferError> {
            Arc::clone(&self.real).register_pinned(allocations, permission)
        }
        unsafe fn register_cuda(
            self: Arc<Self>,
            _: u64,
            _: usize,
            _: i32,
            _: Arc<dyn Send + Sync>,
            _: Permission,
        ) -> Result<Registration, TransferError> {
            Err(TransferError::new(
                ErrorKind::Unavailable,
                "CPU-only fault control",
            ))
        }
        unsafe fn unregister(&self, registration: &mut Registration) -> Result<(), TransferError> {
            // SAFETY: the caller established drain, and this delegates the actual registration.
            unsafe { self.real.unregister(registration) }
        }
        fn open_peer(
            &self,
            _: &str,
            _: &str,
            _: Instant,
        ) -> Result<crate::transfer_engine::PeerSegment, TransferError> {
            Err(TransferError::new(
                ErrorKind::Unavailable,
                "file-only fault control",
            ))
        }
        unsafe fn transfer(
            &self,
            _: TransferOp,
            _: &crate::transfer_engine::PeerSegment,
            _: &[TransferSlice],
            _: Instant,
            _: &Cancellation,
        ) -> Result<usize, TransferError> {
            Err(TransferError::new(
                ErrorKind::Unavailable,
                "file-only fault control",
            ))
        }
        fn shutdown(&self, deadline: Instant) -> Result<(), TransferError> {
            self.real.shutdown(deadline)
        }
        fn invalidate_segment(&self, _: &str) {}
        fn record_nic_load(&self) {}
        fn close_file(&self, file: &FileSegment, deadline: Instant) -> Result<(), TransferError> {
            self.real.close_file(file, deadline)
        }
        unsafe fn transfer_file(
            self: Arc<Self>,
            operation: TransferOp,
            file: &FileSegment,
            transport: FileTransport,
            transfer: Arc<FileTransfer>,
            deadline: Instant,
            cancel: &Cancellation,
        ) -> Result<usize, TransferError> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            match (self.short, self.kind) {
                (false, Some(kind)) => Err(TransferError::new(kind, "injected transfer receipt")),
                _ => {
                    // SAFETY: this forwards the unchanged, registered descriptors and owners.
                    unsafe {
                        Arc::clone(&self.real)
                            .transfer_file(operation, file, transport, transfer, deadline, cancel)
                    }
                    .map(|bytes| if self.short { bytes / 2 } else { bytes })
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "isolated retained-owner control; intentionally retains bounded owners until process exit"]
    async fn engine_files_unknown_drain_keeps_real_extents_and_backing() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        for (kind, short, retained) in [
            (ErrorKind::Timeout, false, false),
            (ErrorKind::Unquiesced, false, true),
            (ErrorKind::Backend, true, false),
        ] {
            let dir = directory();
            let storage = storage(dir.path().join("cache.bin"), 512, 1);
            let store = storage.ssd_store.as_ref().unwrap();
            let real = store.io.engine_file(0).unwrap();
            let backend = Arc::new(Fault {
                real: Arc::clone(&real.backend),
                kind: Some(kind),
                short,
                calls: AtomicUsize::new(0),
            });
            let file = Arc::new(EngineFile {
                backend: backend.clone(),
                segment: FileSegment {
                    owner_epoch: real.segment.owner_epoch.clone(),
                    length: real.segment.length,
                    inner: real.segment.inner.clone(),
                },
                file: Arc::clone(&real.file),
                admission: Arc::new(Semaphore::new(1)),
                runtime: tokio::runtime::Handle::current(),
                cost_resource: real.cost_resource,
            });
            let key = StateKey::new("uncertain-file".into(), vec![1]);
            let slots = vec![crate::SlotMeta::new(
                smallvec::smallvec![512],
                NumaNode::UNKNOWN,
            )];
            let entry = {
                let mut inner = store.inner.lock();
                inner.pending_writes.insert(key.clone());
                inner
                    .ring
                    .reserve(
                        &key,
                        slots.clone(),
                        crate::storage::ssd::index::Encoding::Raw,
                    )
                    .unwrap()
            };
            let lease = Arc::new(crate::storage::ssd::WriteLease {
                entry,
                key: key.clone(),
                committed: AtomicBool::new(false),
                gpu: false,
                store: Arc::clone(store),
            });
            let lease_weak = Arc::downgrade(&lease);
            let source = allocation(&storage, 512, 0x52);
            let source_weak = Arc::downgrade(&source);
            let slices = vec![TransferSlice {
                local: source.mapped_ptr().host(),
                remote_address: 0,
                length: 512,
            }];
            let error = file
                .submit(
                    TransferOp::Write,
                    FileTransport::IoUring,
                    file_extent(slices),
                    FileMemory::Pinned(vec![source]),
                    lease,
                )
                .unwrap()
                .await
                .unwrap()
                .unwrap_err();
            assert_eq!(error.kind == ErrorKind::Unquiesced, retained);
            assert_eq!(
                backend.calls.load(Ordering::Relaxed),
                1,
                "uncertain writes must never replay"
            );
            assert_eq!(lease_weak.upgrade().is_some(), retained);
            assert_eq!(source_weak.upgrade().is_some(), retained);
            assert_eq!(file.admission.available_permits(), usize::from(!retained));
            assert!(
                store.discover(std::slice::from_ref(&key))[0].is_none(),
                "failed writes must remain unpublished"
            );
            let replacement = StateKey::new("uncertain-file".into(), vec![2]);
            let reserved = store.inner.lock().ring.reserve(
                &replacement,
                slots,
                crate::storage::ssd::index::Encoding::Raw,
            );
            assert_eq!(
                reserved.is_none(),
                retained,
                "unknown drain must prevent extent reuse"
            );
            if reserved.is_some() {
                store.commit_write(&replacement, false);
            }
        }
    }
}
