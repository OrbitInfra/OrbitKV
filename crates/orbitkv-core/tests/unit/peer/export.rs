use std::num::NonZeroU64;
use std::ptr::NonNull;

use super::*;
use crate::block::{RawBlock, Segment};
use crate::memory::numa::NumaNode;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};

fn shared_slab() -> (Vec<(StateKey, Arc<SealedBlock>)>, Arc<PinnedAllocation>) {
    let pool = PinnedAllocator::new_global(1024 * 1024, 1, false, None);
    let allocation = pool
        .allocate(NonZeroU64::new(4096).unwrap(), NumaNode::UNKNOWN)
        .unwrap();
    let blocks = (0..2)
        .map(|index| {
            let ptr = unsafe {
                NonNull::new_unchecked(allocation.as_non_null().as_ptr().add(index * 64))
            };
            let raw = RawBlock::single_segment(Segment::new(ptr, 64, Arc::clone(&allocation)));
            (
                StateKey::new("ns".into(), vec![index as u8]),
                Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)])),
            )
        })
        .collect();
    (blocks, allocation)
}

fn metadata_blocks() -> Vec<(StateKey, Arc<SealedBlock>)> {
    (0..2)
        .map(|index| {
            (
                StateKey::new("ns".into(), vec![index]),
                Arc::new(SealedBlock::from_slots(Vec::new())),
            )
        })
        .collect()
}

fn ticket(manager: &TransferLockManager) -> TransferTicket {
    TransferTicket {
        window: manager.open(Uuid::new_v4()).unwrap(),
        slot: 0,
        generation: 1,
    }
}

#[test]
fn staging_release_fences_commit_without_releasing_budget_early() {
    let blocks = metadata_blocks();
    let manager = TransferLockManager::new(Duration::ZERO, 1);
    let ticket = ticket(&manager);
    let reservation = manager.reserve(ticket, 1, blocks.len()).unwrap();

    assert_eq!(manager.inner.lock().reserved_bytes, 1);
    assert_eq!(manager.release(ticket), Ok(0));
    assert_eq!(manager.inner.lock().reserved_bytes, 1);
    assert_eq!(manager.inner.lock().active, 1);
    assert!(matches!(
        reservation.commit(blocks.clone()),
        Err(PeerError::StaleTicket)
    ));
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.inner.lock().active, 0);

    let replacement = TransferTicket {
        generation: 2,
        ..ticket
    };
    manager.lock_blocks(replacement, blocks).unwrap();
    assert_eq!(manager.release(replacement), Ok(2));
}

#[test]
fn staging_commit_uses_actual_footprint_and_failed_handoff_rolls_back() {
    let blocks = metadata_blocks();
    let manager = TransferLockManager::new(Duration::ZERO, 2);
    let first = ticket(&manager);
    let reservation = manager.reserve(first, 2, blocks.len()).unwrap();
    let prepared = reservation.commit(blocks.clone()).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    drop(prepared);
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.inner.lock().active, 0);

    let second = TransferTicket {
        generation: 2,
        ..first
    };
    let reservation = manager.reserve(second, 1, blocks.len()).unwrap();
    let prepared = reservation.commit(blocks).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    prepared.publish();
    assert_eq!(manager.release(second), Ok(2));
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
}

#[test]
fn actual_footprint_cannot_grow_past_the_shared_budget() {
    let blocks = metadata_blocks();
    let manager = TransferLockManager::new(Duration::ZERO, 1);
    let ticket = ticket(&manager);
    let reservation = manager.reserve(ticket, 1, blocks.len()).unwrap();
    assert!(matches!(
        reservation.commit_with_footprint(blocks, 2),
        Err(PeerError::BudgetExhausted)
    ));
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.inner.lock().active, 0);
}

#[test]
fn overdue_transfer_keeps_the_entire_slab_until_completion() {
    let (blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let memory = Arc::downgrade(&allocation);
    let manager = TransferLockManager::new(Duration::ZERO, bytes);
    let session = ticket(&manager);
    manager.lock_blocks(session, blocks.clone()).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, bytes);
    assert_eq!(manager.expire(), 1);
    assert_eq!(manager.expire(), 0);
    assert_eq!(
        manager.lock_blocks(ticket(&manager), blocks.clone()),
        Err(PeerError::BudgetExhausted)
    );
    drop(blocks);
    drop(allocation);
    assert!(
        memory.upgrade().is_some(),
        "eviction and timeout must not free a DMA source"
    );
    assert_eq!(manager.release(session), Ok(2));
    assert!(memory.upgrade().is_none());
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.release(session), Ok(0));
}

#[test]
fn concurrent_readers_share_one_admission_budget() {
    let (blocks, allocation) = shared_slab();
    let bytes = allocation.size_bytes();
    let manager = Arc::new(TransferLockManager::new(Duration::from_secs(30), 2 * bytes));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let manager = Arc::clone(&manager);
            let barrier = Arc::clone(&barrier);
            let blocks = blocks.clone();
            std::thread::spawn(move || {
                let ticket = ticket(&manager);
                barrier.wait();
                manager.lock_blocks(ticket, blocks).ok().map(|()| ticket)
            })
        })
        .collect();
    let sessions: Vec<_> = handles
        .into_iter()
        .filter_map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(sessions.len(), 2);
    assert_eq!(manager.inner.lock().reserved_bytes, 2 * bytes);
    manager.release(sessions[0]).unwrap();
    let replacement = ticket(&manager);
    manager.lock_blocks(replacement, blocks).unwrap();
    manager.release(sessions[1]).unwrap();
    manager.release(replacement).unwrap();
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
}

#[test]
fn a_small_slice_cannot_bypass_the_allocation_budget() {
    let (mut blocks, allocation) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, allocation.size_bytes() - 1);
    blocks.truncate(1);
    let ticket = ticket(&manager);
    assert_eq!(
        manager.lock_blocks(ticket, blocks.clone()),
        Err(PeerError::BudgetExhausted)
    );
    assert_eq!(
        manager.lock_blocks(ticket, blocks),
        Err(PeerError::StaleTicket)
    );
    assert_eq!(manager.inner.lock().reserved_bytes, 0);
    assert_eq!(manager.inner.lock().active, 0);
}

#[test]
fn closed_generations_cannot_resurrect_or_release_a_different_read() {
    let (blocks, _) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, u64::MAX);
    let first = ticket(&manager);
    // Completion wins the race against queued authorization.
    assert_eq!(manager.release(first), Ok(0));
    assert_eq!(
        manager.lock_blocks(first, blocks.clone()),
        Err(PeerError::StaleTicket)
    );
    let second = TransferTicket {
        generation: 2,
        ..first
    };
    manager.lock_blocks(second, blocks.clone()).unwrap();
    assert_eq!(
        manager.lock_blocks(second, blocks.clone()),
        Err(PeerError::StaleTicket)
    );
    assert_eq!(manager.release(first), Ok(0));
    let future = TransferTicket {
        generation: 3,
        ..first
    };
    assert_eq!(manager.release(future), Err(PeerError::StaleTicket));
    assert_eq!(
        manager.lock_blocks(future, blocks.clone()),
        Err(PeerError::StaleTicket)
    );
    assert_eq!(manager.inner.lock().active, 1);
    assert_eq!(manager.release(second), Ok(2));
    manager.lock_blocks(future, blocks).unwrap();
    assert_eq!(manager.release(second), Ok(0));
    assert_eq!(manager.release(future), Ok(2));
}

#[test]
fn idle_window_eviction_is_bounded_and_rejects_late_authorization() {
    let (blocks, _) = shared_slab();
    let manager = TransferLockManager::new(Duration::ZERO, u64::MAX);
    let active = ticket(&manager);
    manager.lock_blocks(active, blocks.clone()).unwrap();
    let idle = ticket(&manager);
    manager.release(idle).unwrap();
    // Simulates lost setup replies and requester churn; none pins payload.
    for _ in 0..MAX_TRANSFER_WINDOWS * 2 {
        manager.open(Uuid::new_v4()).unwrap();
    }
    assert_eq!(manager.inner.lock().windows.len(), MAX_TRANSFER_WINDOWS);
    assert_eq!(manager.inner.lock().active, 1);
    assert_eq!(
        manager.lock_blocks(idle, blocks),
        Err(PeerError::UnknownWindow)
    );
    assert_eq!(manager.release(idle), Ok(0));
    assert_eq!(manager.release(active), Ok(2));
}

#[test]
fn metadata_is_bounded_even_for_zero_byte_blocks() {
    let manager = TransferLockManager::new(Duration::ZERO, 1);
    let blocks = vec![(
        StateKey::new("ns".into(), vec![0]),
        Arc::new(SealedBlock::from_slots(Vec::new())),
    )];
    let mut first = None;
    for _ in 0..MAX_TRANSFER_SESSIONS {
        let ticket = ticket(&manager);
        first.get_or_insert(ticket);
        manager.lock_blocks(ticket, blocks.clone()).unwrap();
    }
    assert_eq!(manager.expire(), MAX_TRANSFER_SESSIONS);
    assert!(
        manager.open(Uuid::new_v4()).is_none(),
        "busy windows are never evicted"
    );
    let first = first.unwrap();
    assert_eq!(
        manager.lock_blocks(TransferTicket { slot: 1, ..first }, blocks),
        Err(PeerError::BudgetExhausted)
    );
    manager.release(first).unwrap();
    assert!(manager.open(Uuid::new_v4()).is_some());
}

fn exports() -> (
    PeerExports,
    Arc<orbitkv_catalog::MembershipView>,
    Arc<crate::ResidencyInventory>,
) {
    use orbitkv_catalog::MembershipView;
    use orbitkv_state::CacheOwner;

    let owner = CacheOwner {
        endpoint: "127.0.0.1:50055".into(),
        incarnation: Uuid::new_v4(),
    };
    let membership = Arc::new(MembershipView::new(owner.clone()));
    assert!(membership.renew(Instant::now(), Duration::from_secs(3600)));
    membership.replace_members([("source".into(), owner)]);
    let inventory = Arc::new(crate::ResidencyInventory::new(16 * 1024));
    let dram = Arc::new(crate::storage::dram::DramStore::with_inventory(
        1 << 20,
        false,
        None,
        Some(inventory.clone()),
        0,
    ));
    (
        PeerExports::new(
            dram,
            None,
            Some(membership.clone()),
            Some("127.0.0.1:12345".into()),
            #[cfg(feature = "orbitte")]
            None,
            Duration::ZERO,
            1 << 20,
        ),
        membership,
        inventory,
    )
}

#[tokio::test]
async fn export_revalidates_residency_and_drains_after_fencing() {
    let (exports, membership, inventory) = exports();
    let owner = membership.owner().incarnation;
    let (blocks, allocation) = shared_slab();
    exports.dram.batch_insert(blocks.clone());
    let records = inventory.page(None).unwrap();
    let ticket = TransferTicket::new(exports.open(owner, Uuid::new_v4()).unwrap(), 0, 1).unwrap();
    // Replacing the same key invalidates previously discovered evidence.
    drop(exports.dram.remove_all());
    exports.dram.batch_insert(blocks);
    assert!(matches!(
        exports.authorize(owner, ticket, &records).await,
        Err(PeerError::StaleReplica)
    ));
    assert_eq!(exports.locks.inner.lock().reserved_bytes, 0);
    let records = inventory.page(None).unwrap();
    let authorized = exports.authorize(owner, ticket, &records).await.unwrap();
    assert_eq!(authorized.len(), records.len());
    let bytes = allocation.size_bytes();
    assert_eq!(exports.locks.inner.lock().reserved_bytes, bytes);
    let memory = Arc::downgrade(&allocation);
    drop(authorized);
    drop(allocation);
    drop(exports.dram.remove_all());
    membership.fence();
    assert_eq!(
        exports.open(owner, Uuid::new_v4()),
        Err(PeerError::StaleReplica)
    );
    assert!(matches!(
        exports.authorize(owner, ticket, &records).await,
        Err(PeerError::StaleReplica)
    ));
    assert_eq!(exports.expire(), 1);
    assert!(
        memory.upgrade().is_some(),
        "fencing/expiry must preserve the DMA source"
    );
    assert_eq!(exports.release(ticket), Ok(records.len()));
    assert_eq!(exports.release(ticket), Ok(0));
    assert!(memory.upgrade().is_none());
    assert_eq!(exports.locks.inner.lock().reserved_bytes, 0);
}

#[tokio::test]
async fn export_rejects_invalid_evidence_before_reserving_resources() {
    use orbitkv_state::InventoryRecord;

    let (mut exports, membership, _inventory) = exports();
    let owner = membership.owner().incarnation;
    assert_eq!(
        exports.open(Uuid::new_v4(), Uuid::new_v4()),
        Err(PeerError::StaleReplica)
    );
    assert!(matches!(
        exports.open(owner, Uuid::nil()),
        Err(PeerError::InvalidRequest(_))
    ));
    let window = exports.open(owner, Uuid::new_v4()).unwrap();
    for (window, slot, generation) in [
        (Uuid::nil(), 0, 1),
        (window, TRANSFER_WINDOW_SLOTS, 1),
        (window, 0, 0),
    ] {
        assert!(matches!(
            TransferTicket::new(window, slot, generation),
            Err(PeerError::InvalidRequest(_))
        ));
    }
    let ticket = TransferTicket::new(window, 0, 1).unwrap();
    let valid = InventoryRecord {
        key: StateKey::new("ns".into(), vec![1]),
        sequence: 1,
        present: true,
        metadata: None,
    };
    let mut mixed = valid.clone();
    mixed.key.namespace = "other".into();
    for records in [
        vec![],
        vec![valid.clone(), mixed],
        vec![InventoryRecord {
            sequence: 0,
            ..valid.clone()
        }],
        vec![InventoryRecord {
            present: false,
            ..valid.clone()
        }],
        vec![InventoryRecord {
            key: StateKey::new("ns".into(), vec![]),
            ..valid.clone()
        }],
    ] {
        assert!(matches!(
            exports.authorize(owner, ticket, &records).await,
            Err(PeerError::InvalidRequest(_))
        ));
    }
    assert_eq!(exports.locks.inner.lock().active, 0);
    exports.endpoint = None;
    assert_eq!(
        exports.open(owner, Uuid::new_v4()),
        Err(PeerError::Unavailable)
    );
    assert!(matches!(
        exports.authorize(owner, ticket, &[valid]).await,
        Err(PeerError::Unavailable)
    ));
}

#[cfg(feature = "orbitte")]
#[tokio::test]
#[ignore = "OrbitTE source registration hardware gate: requires a real CUDA pinned allocator"]
async fn orbitte_grant_registrations_share_backing_and_fence_old_completion_generations() {
    use crate::peer::orbitte::{OrbitTeTransport, parse_endpoint};
    use crate::peer::provider::PeerTransport;
    use orbit_te::{
        EngineConfig, Permission, TransferEngine, TransferOp, TransferRequest, Transport,
    };

    let (mut exports, membership, inventory) = exports();
    let source = Arc::new(
        OrbitTeTransport::new(
            1 << 20,
            "127.0.0.1:0",
            Transport::Tcp,
            #[cfg(feature = "orbitte-rdma")]
            None,
        )
        .unwrap(),
    );
    exports.transport = Some(Arc::new(PeerTransport::OrbitTe(Arc::clone(&source))));
    let owner = membership.owner().incarnation;
    let (blocks, allocation) = shared_slab();
    let weak = Arc::downgrade(&allocation);
    let address = allocation.as_non_null().as_ptr() as usize as u64;
    // Initialize before publishing immutable source blocks; all later access is READ.
    let memory = unsafe { allocation.orbitte_memory().unwrap() };
    memory.write(0, &[0x42; 4096]).unwrap();
    drop(memory);
    exports.dram.batch_insert(blocks);
    let records = inventory.page(None).unwrap();
    let first = TransferTicket::new(exports.open(owner, Uuid::new_v4()).unwrap(), 0, 1).unwrap();
    let second = TransferTicket { slot: 1, ..first };
    drop(exports.authorize(owner, first, &records).await.unwrap());
    drop(exports.authorize(owner, second, &records).await.unwrap());
    let client = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = parse_endpoint(Transport::Tcp, &source.endpoint).unwrap();
    let epoch = source.epoch.parse().unwrap();
    assert_eq!(
        client
            .open_segment(endpoint, epoch, Duration::from_secs(2))
            .unwrap()
            .regions()
            .len(),
        1
    );
    assert_eq!(exports.release(first), Ok(2));
    let current = TransferTicket {
        generation: 2,
        ..first
    };
    drop(exports.authorize(owner, current, &records).await.unwrap());
    assert_eq!(exports.release(first), Ok(0));
    assert_eq!(exports.release(second), Ok(2));
    drop(exports.dram.remove_all());
    drop(allocation);
    assert!(weak.upgrade().is_some());
    let peer = client
        .open_segment(endpoint, epoch, Duration::from_secs(2))
        .unwrap();
    assert_eq!(peer.regions().len(), 1);
    let local = client
        .register_memory(vec![0; 64].into_boxed_slice(), Permission::LocalOnly)
        .unwrap();
    let read = client
        .submit_and_wait(
            TransferOp::Read,
            &peer,
            &[TransferRequest {
                transport: Transport::Tcp,
                local_address: local.base_address(),
                remote_address: address,
                length: 64,
            }],
            Duration::from_secs(2),
        )
        .unwrap();
    assert!(read.is_complete() && read.locally_drained);
    let mut bytes = [0; 64];
    local.read(0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x42; 64]);
    assert_eq!(exports.release(current), Ok(2));
    assert!(
        client
            .open_segment(endpoint, epoch, Duration::from_secs(2))
            .unwrap()
            .regions()
            .is_empty()
    );
    client.unregister_memory(&local).unwrap();
    client.close_requester(Duration::from_secs(2)).unwrap();
    drop(exports);
    drop(source);
    assert!(weak.upgrade().is_none());
}
