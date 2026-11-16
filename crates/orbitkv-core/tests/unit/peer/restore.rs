use super::*;
use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::backend::{RegistrationInner, Segment};
use crate::transfer_engine::{PeerSegment, Permission, Registration, TransferError};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

#[derive(Default)]
pub(crate) struct Control {
    engine: Mutex<std::sync::Weak<TransferEngine>>,
    pub(crate) result: Mutex<Option<ErrorKind>>,
    pub(crate) registrations: AtomicUsize,
    pub(crate) reads: AtomicUsize,
    pub(crate) options: Mutex<Vec<crate::transfer_engine::TransferOptions>>,
    pub(crate) admissions: Mutex<Vec<crate::transfer_engine::TransferOptions>>,
    pub(crate) cleanups: AtomicUsize,
    cleanup_failed: std::sync::atomic::AtomicBool,
    failure: Mutex<Option<Failure>>,
}

#[derive(Clone, Copy, Debug)]
enum Failure {
    RegistrationPanic,
    RegistrationUnquiesced,
    TransferPanic,
    CleanupPanic,
}

impl Control {
    pub(crate) fn engine(self: &Arc<Self>) -> Arc<TransferEngine> {
        let mut cached = self.engine.lock().unwrap();
        if let Some(engine) = cached.upgrade() {
            return engine;
        }
        let mut engine = TransferEngine::new(
            4096,
            "127.0.0.1:0",
            orbite::Transport::Tcp,
            None,
            #[cfg(feature = "orbite-rdma")]
            None,
        )
        .unwrap();
        engine.control = Some(self.clone());
        let engine = Arc::new(engine);
        *cached = Arc::downgrade(&engine);
        engine
    }

    pub(crate) fn transfer_endpoint(&self) -> &str {
        "control"
    }
    pub(crate) fn source_epoch(&self) -> &str {
        "epoch"
    }
    pub(crate) fn accepts_endpoint(&self, _: &str) -> bool {
        true
    }
    pub(crate) fn register_pinned(
        self: Arc<Self>,
        _: &[Arc<PinnedAllocation>],
        _: Permission,
    ) -> Result<Registration, TransferError> {
        panic!("GPU restore must not allocate/register pinned payloads")
    }
    pub(crate) unsafe fn register_cuda(
        self: Arc<Self>,
        _: u64,
        _: usize,
        _: i32,
        owner: Arc<dyn Send + Sync>,
        _: Permission,
    ) -> Result<Registration, TransferError> {
        self.registrations.fetch_add(1, Ordering::SeqCst);
        match *self.failure.lock().unwrap() {
            Some(Failure::RegistrationPanic) => panic!("registration panicked"),
            Some(Failure::RegistrationUnquiesced) => {
                return Err(TransferError::new(
                    ErrorKind::Unquiesced,
                    "registration failed",
                ));
            }
            _ => {}
        }
        Ok(Registration {
            inner: RegistrationInner::Control(owner),
        })
    }
    pub(crate) unsafe fn unregister(
        &self,
        registration: &mut Registration,
    ) -> Result<(), TransferError> {
        assert!(matches!(&registration.inner, RegistrationInner::Control(_)));
        self.cleanups.fetch_add(1, Ordering::SeqCst);
        if matches!(*self.failure.lock().unwrap(), Some(Failure::CleanupPanic)) {
            panic!("cleanup panicked");
        }
        if self.cleanup_failed.load(Ordering::SeqCst) {
            Err(TransferError::new(ErrorKind::Backend, "cleanup failed"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn open_peer(
        &self,
        _: &str,
        epoch: &str,
        deadline: Instant,
    ) -> Result<PeerSegment, TransferError> {
        if epoch != "epoch" {
            return Err(TransferError::new(ErrorKind::EpochMismatch, "stale"));
        }
        if deadline <= Instant::now() {
            return Err(TransferError::new(ErrorKind::Timeout, "expired"));
        }
        Ok(PeerSegment {
            owner_epoch: epoch.into(),
            inner: Segment::Control,
        })
    }
    pub(crate) unsafe fn transfer(
        &self,
        op: TransferOp,
        segment: &PeerSegment,
        slices: &[TransferSlice],
        options: crate::transfer_engine::TransferOptions,
        _: &Cancellation,
    ) -> Result<usize, TransferError> {
        assert_eq!(segment.owner_epoch, "epoch");
        assert!(matches!(segment.inner, Segment::Control));
        assert_eq!(op, TransferOp::Read);
        assert_eq!(
            options.intent,
            crate::transfer_engine::TransferIntent::ForegroundGet
        );
        self.options.lock().unwrap().push(options);
        self.reads.fetch_add(1, Ordering::SeqCst);
        if matches!(*self.failure.lock().unwrap(), Some(Failure::TransferPanic)) {
            panic!("transfer panicked");
        }
        if let Some(kind) = *self.result.lock().unwrap() {
            return Err(TransferError::new(kind, "injected"));
        }
        for slice in slices {
            // SAFETY: CPU control borrows actual disjoint host buffers. This tests
            // the production descriptor/owner boundary, not CUDA or RDMA hardware.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.remote_address as *const u8,
                    slice.local.as_ptr(),
                    slice.length,
                );
            }
        }
        Ok(slices.iter().map(|slice| slice.length).sum())
    }
    pub(crate) fn retire_peer(
        &self,
        _: &str,
        _: &str,
        _: std::time::Duration,
        drained: bool,
    ) -> Result<(), TransferError> {
        if drained {
            Ok(())
        } else {
            Err(TransferError::new(
                ErrorKind::Unquiesced,
                "control still active",
            ))
        }
    }
}

pub(crate) fn source(
    control: Arc<Control>,
    info: Vec<orbitkv_proto::proto::engine::TransferSlotInfo>,
) -> PeerRestoreBlock {
    let source_owner = orbitkv_state::CacheOwner {
        endpoint: "control".into(),
        incarnation: uuid::Uuid::new_v4(),
    };
    let membership = Arc::new(orbitkv_catalog::MembershipView::new(source_owner.clone()));
    membership.renew(Instant::now(), std::time::Duration::from_secs(120));
    membership.replace_members([("control".into(), 1, source_owner.clone())]);
    let options = crate::transfer_engine::tests::demand();
    control.admissions.lock().unwrap().push(options);
    let segments = PeerSegments::new(control.engine(), membership.clone());
    let peer = segments
        .get(membership.lease(&source_owner).unwrap(), "control", "epoch")
        .unwrap();
    PeerRestoreBlock {
        expiration: None,
        hash: vec![1],
        grant: Arc::new(Grant {
            _lock: TransferLockGuard::test_detached(),
            _segments: segments,
            peer,
            options,
        }),
        slots: validate_slots(&info, info.len()).unwrap().into(),
    }
}

pub(crate) fn registered(
    control: Arc<Control>,
    address: u64,
    length: usize,
    owner: Arc<dyn Send + Sync>,
) -> Arc<CudaRegistration> {
    // SAFETY: CPU control retains host backing or uses only non-dereferencing fixtures.
    unsafe { CudaRegistration::new(control.engine(), address, length, 0, owner) }.unwrap()
}

#[test]
fn descriptor_validation_rejects_missing_overflowing_and_malformed_grants() {
    use orbitkv_proto::proto::engine::TransferSlotInfo;
    let mut info = TransferSlotInfo {
        k_ptr: 4096,
        k_size: 8192,
        checksums: serde_json::to_vec(&vec![crate::integrity::Checksum { end: 8192, crc: 0 }])
            .unwrap(),
        ..Default::default()
    };
    let parsed = validate_slots(std::slice::from_ref(&info), 1).unwrap();
    assert_eq!(parsed[0].range(0, 4096, 4096).unwrap(), 8192);
    for (segment, offset, size) in [(0, 8191, 2), (1, 0, 1), (0, usize::MAX, 1)] {
        assert!(parsed[0].range(segment, offset, size).is_err());
    }
    assert!(validate_slots(&[], 1).is_err());
    for address in [0, u64::MAX] {
        info.k_ptr = address;
        assert!(validate_slots(std::slice::from_ref(&info), 1).is_err());
    }
    info.k_ptr = 4096;
    for encoding in [b"[]".to_vec(), vec![0; 4097]] {
        info.encoding = encoding;
        assert!(validate_slots(std::slice::from_ref(&info), 1).is_err());
    }
}

#[tokio::test]
async fn uncertainty_retains_actual_backing_and_never_retries_cleanup_or_transfer() {
    for failure in [
        None,
        Some(Failure::RegistrationPanic),
        Some(Failure::RegistrationUnquiesced),
        Some(Failure::TransferPanic),
        Some(Failure::CleanupPanic),
    ] {
        let control = Arc::new(Control::default());
        *control.failure.lock().unwrap() = failure;
        let owner = Arc::new(());
        let backing = Arc::downgrade(&owner);
        // SAFETY: these controls fail without dereferencing fixture addresses.
        let registration = unsafe { CudaRegistration::new(control.engine(), 8192, 16, 0, owner) };
        if let Ok(registration) = registration {
            if matches!(failure, Some(Failure::CleanupPanic)) {
                assert!(registration.close().is_err());
                assert!(registration.close().is_err());
                assert_eq!(control.cleanups.load(Ordering::SeqCst), 1);
            } else {
                *control.result.lock().unwrap() = Some(ErrorKind::Unquiesced);
                let source = source(control.clone(), vec![]);
                let grant = Arc::downgrade(&source.grant);
                let slice = TransferSlice {
                    local: std::ptr::NonNull::new(8192 as *mut u8).unwrap(),
                    remote_address: 4096,
                    length: 16,
                };
                // SAFETY: injected failures never dereference these addresses.
                assert!(matches!(
                    // SAFETY: the Control backend never dereferences the synthetic address;
                    // the registration/owner remain live to test admission and drain accounting.
                    unsafe { source.read_gpu(&[slice], std::slice::from_ref(&registration)) },
                    Err(PeerTransferError::Unquiesced)
                ));
                drop(source);
                assert!(grant.upgrade().is_some());
                assert_eq!(control.reads.load(Ordering::SeqCst), 1);
                assert!(registration.close().is_err());
            }
            drop(registration);
        } else {
            assert_eq!(registration.unwrap_err().kind, ErrorKind::Unquiesced);
        }
        assert_eq!(
            control.cleanups.load(Ordering::SeqCst),
            usize::from(matches!(failure, Some(Failure::CleanupPanic)))
        );
        assert!(backing.upgrade().is_some(), "{failure:?}");
        assert_eq!(control.registrations.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn epoch_deadline_and_membership_fences_reject_before_submission() {
    let control = Arc::new(Control::default());
    let owner = registered(control.clone(), 8192, 16, Arc::new(()));
    let slice = TransferSlice {
        local: std::ptr::NonNull::new(8192 as *mut u8).unwrap(),
        remote_address: 4096,
        length: 16,
    };
    for fence in 0..3 {
        let mut source = source(control.clone(), vec![]);
        let grant = Arc::get_mut(&mut source.grant).unwrap();
        match fence {
            0 => {
                grant.peer =
                    crate::peer::segment::tests::with_epoch(&grant._segments, &grant.peer, "stale");
            }
            1 => grant.options.deadline = Instant::now(),
            _ => crate::peer::segment::tests::fence(&grant._segments),
        }
        // SAFETY: preflight fences reject before any payload use.
        assert!(unsafe { source.read_gpu(&[slice], std::slice::from_ref(&owner)) }.is_err());
    }
    assert_eq!(control.reads.load(Ordering::SeqCst), 0);
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 0);
    owner.close().unwrap();
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 1);
}

#[test]
fn release_waits_for_active_loans_and_failed_unregister_retains_backing() {
    let control = Arc::new(Control::default());
    let backing = Arc::new(());
    let weak = Arc::downgrade(&backing);
    let owner = registered(control.clone(), 8192, 16, backing);
    let backend = control.engine();
    let loan = owner.loan(&backend).unwrap();
    let other = owner.clone();
    control.cleanup_failed.store(true, Ordering::SeqCst);
    let (tx, rx) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || tx.send(other.close()).unwrap());
    assert!(
        rx.recv_timeout(std::time::Duration::from_millis(20))
            .is_err()
    );
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 0);
    drop(loan);
    assert_eq!(rx.recv().unwrap().unwrap_err().kind, ErrorKind::Unquiesced);
    thread.join().unwrap();
    drop(owner);
    assert_eq!(control.cleanups.load(Ordering::SeqCst), 1);
    assert!(weak.upgrade().is_some());
}

#[tokio::test]
async fn direct_restore_preserves_expiry_and_rejects_expired_admission() {
    use orbitkv_proto::proto::engine::{
        CacheExpiry, QueryBlocksForTransferResponse, TransferBlockInfo, TransferSlotInfo,
    };
    let control = Arc::new(Control::default());
    let owner = orbitkv_state::CacheOwner {
        endpoint: "control".into(),
        incarnation: uuid::Uuid::new_v4(),
    };
    let membership = Arc::new(orbitkv_catalog::MembershipView::new(owner.clone()));
    membership.renew(Instant::now(), std::time::Duration::from_secs(120));
    membership.replace_members([("control".into(), 1, owner.clone())]);
    let index = Arc::new(orbitkv_catalog::GlobalIndex::new(
        membership.clone(),
        16 * 1024,
        Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
    ));
    let reader = PeerReader::new(
        index,
        control.engine(),
        Arc::new(|_, _| None),
        membership.clone(),
    );
    let segments = PeerSegments::new(control.engine(), membership.clone());
    let peer = segments
        .get(membership.lease(&owner).unwrap(), "control", "epoch")
        .unwrap();
    let segment = FetchSegment {
        owner,
        source: crate::planning::peer::PeerSource::Dram,
        records: vec![orbitkv_state::InventoryRecord {
            key: StateKey::new("ns".into(), vec![1]),
            sequence: 1,
            present: true,
            metadata: None,
        }],
    };
    let expiry = crate::storage::expiry::Expiration::new(Some(std::time::Duration::from_secs(60)))
        .unwrap()
        .unwrap()
        .record;
    let response = |cache_expiry| QueryBlocksForTransferResponse {
        cache_protocol: orbitkv_state::INVENTORY_STREAM_PROTOCOL.into(),
        blocks: vec![TransferBlockInfo {
            cache_expiry,
            block_hash: vec![1],
            slots: vec![TransferSlotInfo {
                k_ptr: 4096,
                k_size: 512,
                checksums: serde_json::to_vec(&vec![crate::integrity::Checksum {
                    end: 512,
                    crc: 0,
                }])
                .unwrap(),
                ..Default::default()
            }],
        }],
        ..Default::default()
    };
    let grant = AuthorizedSegment::for_restore_test(
        segments.clone(),
        peer.clone(),
        response(Some(CacheExpiry {
            published_unix_millis: expiry.published_unix_millis(),
            expires_unix_millis: expiry.expires_unix_millis(),
        })),
    );
    let restored = RestoreReader(&reader)
        .fetch_segment(&segment, grant, "ttl", &[NumaNode::UNKNOWN])
        .await
        .unwrap();
    let block = &restored[0].1;
    let held = Arc::downgrade(&block.grant);
    assert_eq!(block.expiration.as_ref().unwrap().record, expiry);
    assert!(!block.is_expired());
    block.expiration.as_ref().unwrap().expire_for_test();
    assert!(block.is_expired());
    assert!(held.upgrade().is_some());
    assert_eq!(block.slots[0].range(0, 0, 512).unwrap(), 4096);
    assert_eq!(control.reads.load(Ordering::Acquire), 0);
    drop(restored);
    assert!(held.upgrade().is_none());
    for (birth, deadline) in [(1, 2), (0, 2), (2, 1)] {
        let grant = AuthorizedSegment::for_restore_test(
            segments.clone(),
            peer.clone(),
            response(Some(CacheExpiry {
                published_unix_millis: birth,
                expires_unix_millis: deadline,
            })),
        );
        assert!(
            RestoreReader(&reader)
                .fetch_segment(&segment, grant, "expired", &[NumaNode::UNKNOWN])
                .await
                .is_err()
        );
    }
}
