use super::*;
use crate::memory::pool::PinnedAllocation;
use crate::transfer_engine::{
    PeerSegment, Registration, RegistrationInner, Segment, TransferError,
};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
pub(crate) struct Control {
    pub(crate) result: Mutex<Option<ErrorKind>>,
    pub(crate) registrations: AtomicUsize,
    pub(crate) reads: AtomicUsize,
    cleanups: AtomicUsize,
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

impl TransferBackend for Control {
    fn transfer_endpoint(&self) -> &str {
        "control"
    }
    fn source_epoch(&self) -> &str {
        "epoch"
    }
    fn accepts_endpoint(&self, _: &str) -> bool {
        true
    }
    fn register_pinned(
        self: Arc<Self>,
        _: &[Arc<PinnedAllocation>],
        _: Permission,
    ) -> Result<Registration, TransferError> {
        panic!("GPU restore must not allocate/register pinned payloads")
    }
    unsafe fn register_cuda(
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
    unsafe fn unregister(&self, _: &mut Registration) -> Result<(), TransferError> {
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
    fn open_peer(
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
    unsafe fn transfer(
        &self,
        op: TransferOp,
        _: &PeerSegment,
        slices: &[TransferSlice],
        _: Instant,
        _: &Cancellation,
    ) -> Result<usize, TransferError> {
        assert_eq!(op, TransferOp::Read);
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
    fn shutdown(&self, _: Instant) -> Result<(), TransferError> {
        Ok(())
    }
    fn invalidate_segment(&self, _: &str) {}
    fn record_nic_load(&self) {}
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
    membership.replace_members([("control".into(), source_owner.clone())]);
    PeerRestoreBlock {
        grant: Arc::new(Grant {
            _lock: TransferLockGuard::test_detached(),
            membership,
            source_owner,
            transfer: control,
            endpoint: "control".into(),
            epoch: "epoch".into(),
            deadline: Instant::now() + std::time::Duration::from_secs(10),
        }),
        slots: validate_slots(&info, info.len()).unwrap().into(),
    }
}

#[test]
fn descriptor_validation_rejects_missing_overflowing_and_malformed_grants() {
    use orbitkv_proto::proto::engine::TransferSlotInfo;
    let valid = TransferSlotInfo {
        k_ptr: 4096,
        k_size: 8192,
        ..Default::default()
    };
    let parsed = validate_slots(std::slice::from_ref(&valid), 1).unwrap();
    assert_eq!(parsed[0].range(0, 4096, 4096).unwrap(), 8192);
    assert!(parsed[0].range(0, 8191, 2).is_err());
    assert!(parsed[0].range(1, 0, 1).is_err());
    assert!(parsed[0].range(0, usize::MAX, 1).is_err());
    assert!(validate_slots(&[], 1).is_err());
    for invalid in [
        TransferSlotInfo {
            k_ptr: 0,
            ..valid.clone()
        },
        TransferSlotInfo {
            k_ptr: u64::MAX,
            ..valid.clone()
        },
        TransferSlotInfo {
            encoding: b"[]".to_vec(),
            ..valid.clone()
        },
        TransferSlotInfo {
            encoding: vec![0; 4097],
            ..valid
        },
    ] {
        assert!(validate_slots(&[invalid], 1).is_err());
    }
}

#[tokio::test]
async fn gpu_boundary_moves_fragmented_bytes_without_pinned_registration() {
    let control = Arc::new(Control::default());
    let source = source(control.clone(), vec![]);
    let original: Vec<u8> = (0..(17 << 20)).map(|n| (n % 251) as u8).collect();
    let owner = Arc::new(Mutex::new(vec![0u8; original.len() + 32]));
    let address = owner.lock().unwrap().as_mut_ptr() as u64;
    // SAFETY: this CPU control lends exclusive target bytes through synchronous drain.
    let bytes = unsafe {
        source.read_gpu(
            original.as_ptr() as u64,
            address + 16,
            original.len(),
            0,
            owner.clone(),
        )
    }
    .unwrap();
    assert_eq!(bytes, original.len());
    assert_eq!(control.reads.load(Ordering::SeqCst), 2);
    assert_eq!(control.registrations.load(Ordering::SeqCst), 2);
    let target = owner.lock().unwrap();
    assert_eq!(&target[16..16 + original.len()], original);
    assert_eq!(&target[..16], &[0; 16]);
    assert_eq!(&target[16 + original.len()..], &[0; 16]);
    assert_eq!(Arc::strong_count(&owner), 1);
}

#[tokio::test]
async fn unknown_drain_retains_registration_backing_and_shared_source_grant() {
    let control = Arc::new(Control::default());
    *control.result.lock().unwrap() = Some(ErrorKind::Unquiesced);
    let source = source(control.clone(), vec![]);
    let grant = Arc::downgrade(&source.grant);
    let owner = Arc::new(vec![0u8; 16]);
    let backing = Arc::downgrade(&owner);
    // SAFETY: the injected control never dereferences addresses; it reports unknown drain.
    assert!(matches!(
        unsafe { source.read_gpu(4096, 8192, 16, 0, owner) },
        Err(PeerTransferError::Unquiesced)
    ));
    drop(source);
    assert!(grant.upgrade().is_some());
    assert!(backing.upgrade().is_some());
    assert_eq!(
        control.reads.load(Ordering::SeqCst),
        1,
        "uncertain READ must not replay"
    );
}

#[tokio::test]
async fn backend_unwind_and_registration_uncertainty_retain_owners_without_cleanup_retry() {
    for (failure, reads, cleanups, retained_owners) in [
        (Failure::RegistrationPanic, 0, 0, 1),
        (Failure::RegistrationUnquiesced, 0, 0, 1),
        (Failure::TransferPanic, 1, 0, 2),
        (Failure::CleanupPanic, 1, 1, 2),
    ] {
        let control = Arc::new(Control::default());
        *control.failure.lock().unwrap() = Some(failure);
        let source = source(control.clone(), vec![]);
        let grant = Arc::downgrade(&source.grant);
        let owner = Arc::new(Mutex::new(vec![0u8; 16]));
        let target = owner.lock().unwrap().as_mut_ptr() as u64;
        let input = [9u8; 16];
        let backing = Arc::downgrade(&owner);
        // SAFETY: the control uses disjoint actual host bytes or panics before use.
        let result = unsafe { source.read_gpu(input.as_ptr() as u64, target, 16, 0, owner) };
        assert!(
            matches!(result, Err(PeerTransferError::Unquiesced)),
            "{failure:?}"
        );
        drop(source);
        assert!(grant.upgrade().is_some(), "{failure:?}");
        // A returned registration must survive too, not only the caller's Arc.
        assert_eq!(backing.strong_count(), retained_owners, "{failure:?}");
        assert_eq!(
            control.registrations.load(Ordering::SeqCst),
            1,
            "{failure:?}"
        );
        assert_eq!(control.reads.load(Ordering::SeqCst), reads, "{failure:?}");
        assert_eq!(
            control.cleanups.load(Ordering::SeqCst),
            cleanups,
            "{failure:?}"
        );
    }
}

#[tokio::test]
async fn stale_epoch_and_deadline_fail_before_gpu_registration() {
    let control = Arc::new(Control::default());
    let mut source = source(control.clone(), vec![]);
    Arc::get_mut(&mut source.grant).unwrap().epoch = "stale".into();
    // SAFETY: preflight rejects before a descriptor can be submitted.
    assert!(unsafe { source.read_gpu(4096, 8192, 16, 0, Arc::new(())) }.is_err());
    let grant = Arc::get_mut(&mut source.grant).unwrap();
    grant.epoch = "epoch".into();
    grant.deadline = Instant::now();
    // SAFETY: the expired deadline rejects before any payload use.
    assert!(matches!(
        unsafe { source.read_gpu(4096, 8192, 16, 0, Arc::new(())) },
        Err(PeerTransferError::Timeout)
    ));
    assert_eq!(control.registrations.load(Ordering::SeqCst), 0);
    assert_eq!(control.reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_unregister_is_unknown_drain_even_after_a_completed_read() {
    let control = Arc::new(Control::default());
    control.cleanup_failed.store(true, Ordering::SeqCst);
    let source = source(control.clone(), vec![]);
    let owner = Arc::new(Mutex::new(vec![0u8; 16]));
    let target = owner.lock().unwrap().as_mut_ptr() as u64;
    let input = [7u8; 16];
    let retained = Arc::downgrade(&owner);
    // SAFETY: disjoint actual CPU backing is lent to the synchronous control.
    assert!(matches!(
        unsafe { source.read_gpu(input.as_ptr() as u64, target, 16, 0, owner) },
        Err(PeerTransferError::Unquiesced)
    ));
    assert!(retained.upgrade().is_some());
    assert_eq!(control.reads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn deferred_grant_rechecks_membership_before_gpu_admission() {
    let control = Arc::new(Control::default());
    let source = source(control.clone(), vec![]);
    source.grant.membership.fence();
    // SAFETY: the membership fence rejects before any descriptor use.
    assert!(unsafe { source.read_gpu(4096, 8192, 16, 0, Arc::new(())) }.is_err());
    assert_eq!(control.registrations.load(Ordering::SeqCst), 0);
    assert_eq!(control.reads.load(Ordering::SeqCst), 0);
}
