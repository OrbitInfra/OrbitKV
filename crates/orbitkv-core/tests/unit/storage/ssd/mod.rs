use super::*;

#[test]
fn only_automatic_failures_disable_gpu_admission() {
    for automatic in [true, false] {
        let state = GpuIo {
            automatic,
            enabled: AtomicBool::new(true),
        };
        state.failed("storage buffer registration failed");
        assert_eq!(state.available(), !automatic);
        state.failed("another submitted operation also failed");
        assert_eq!(state.available(), !automatic);
    }
}

#[test]
fn selective_writes_track_republication_without_pinning_payloads() {
    let mut inner = SsdInner {
        ring: SsdRingBuffer::new_sharded(vec![4096], 512),
        pending_writes: HashSet::new(),
        reuse_history: LruCache::new(2),
    };
    let key = StateKey::new("ns".into(), vec![1]);
    let other = StateKey::new("other".into(), vec![1]);
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(
        inner.admission_skip(&other, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    let fresh = StateKey::new("ns".into(), vec![2]);
    assert_eq!(
        inner.admission_skip(&fresh, true, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(inner.reuse_history.len(), 2);
    assert_eq!(
        inner.admission_skip(&key, false, SsdWritePolicy::Reuse),
        Some("cold")
    );
    inner.pending_writes.insert(key.clone());
    assert_eq!(
        inner.admission_skip(&key, true, SsdWritePolicy::Reuse),
        Some("pending")
    );
    // A failed/drained write clears pending ownership and can be retried.
    inner.pending_writes.remove(&key);
    assert_eq!(
        inner.admission_skip(&key, true, SsdWritePolicy::Reuse),
        None
    );
    assert_eq!(
        inner.admission_skip(&other, false, SsdWritePolicy::All),
        None
    );
}

#[tokio::test]
async fn neutral_leases_read_raw_and_encoded_generations_through_uring() {
    use std::num::NonZeroU64;

    use crate::block::{RawBlock, Segment};
    use crate::memory::pool::PinnedAllocator;

    let allocator = Arc::new(PinnedAllocator::new_global(
        16 * 1024,
        1,
        false,
        NonZeroU64::new(SSD_ALIGNMENT as u64),
    ));
    for encoded in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let pool = Arc::clone(&allocator);
        let store = SsdStore::new(
            SsdCacheConfig {
                cache_paths: vec![directory.path().join("cache.bin")],
                capacity_bytes: SSD_ALIGNMENT as u64,
                backend: SsdBackend::Uring,
                ..Default::default()
            },
            Arc::new(move |bytes, node| {
                pool.allocate(NonZeroU64::new(bytes)?, node.unwrap_or(NumaNode::UNKNOWN))
            }),
            {
                let allocator = Arc::clone(&allocator);
                Arc::new(move |bytes, node| {
                    allocator.allocation_footprint(
                        NonZeroU64::new(bytes)?,
                        node.unwrap_or(NumaNode::UNKNOWN),
                    )
                })
            },
            false,
            None,
        )
        .unwrap();
        let data = [if encoded { 0x71 } else { 0x32 }; SSD_ALIGNMENT];
        let allocation = allocator
            .allocate(
                NonZeroU64::new(SSD_ALIGNMENT as u64).unwrap(),
                NumaNode::UNKNOWN,
            )
            .unwrap();
        // SAFETY: this unshared allocation owns the complete target range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                allocation.mapped_ptr().host().as_ptr(),
                data.len(),
            );
        }
        let mut raw = RawBlock::single_segment(Segment::new(
            allocation.mapped_ptr().host(),
            data.len(),
            allocation,
        ));
        if encoded {
            raw.encoding = Some(vec![crate::codec::EncodedSegment {
                version: 1,
                format: orbitkv_state::StorageFormat::Exact,
                logical_bytes: data.len(),
                stored_bytes: data.len(),
                checksum: crc32fast::hash(&data),
            }]);
        }
        let block = Arc::new(SealedBlock::from_slots(vec![(raw, NumaNode::UNKNOWN)]));
        let key = StateKey::new("lease-uring".into(), vec![u8::from(encoded)]);
        store.ingest_batch(std::iter::once((&key, &block)), false);
        store.flush().await;
        drop(block);

        let lease = store
            .discover(std::slice::from_ref(&key))
            .into_iter()
            .map_while(|candidate| candidate?.pin())
            .collect::<Vec<_>>()
            .pop()
            .unwrap();
        assert!(!lease.cufile_eligible(0));
        assert!(lease.file().is_err());
        assert_eq!(lease.cost_resource(), store.io.cost_resource);
        if encoded {
            assert!(!lease.entry.fits_gpu_decode(0));
        }
        let generation = lease.entry.begin;
        let readers = Arc::clone(&lease.entry.readers);
        let restored = store
            .read_host_batch(vec![Arc::clone(&lease)])
            .await
            .unwrap()
            .remove(0)
            .1;
        let slot = restored.get_slot(0).unwrap();
        // SAFETY: the successful read initialized the owned segment above.
        let bytes = unsafe {
            std::slice::from_raw_parts(slot.segment_ptr(0).unwrap().as_ptr(), data.len())
        };
        assert_eq!(bytes, data);
        assert_eq!(slot.encoding.is_some(), encoded);

        let replacement = StateKey::new("lease-uring".into(), vec![2]);
        let slots = || {
            vec![crate::SlotMeta::new(
                smallvec::smallvec![SSD_ALIGNMENT as u64],
                NumaNode::UNKNOWN,
            )]
        };
        assert!(
            store
                .inner
                .lock()
                .ring
                .reserve(&replacement, slots(), index::Encoding::Raw)
                .is_none(),
            "the selected generation must remain pinned after host materialization"
        );
        drop(lease);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while readers.load(Ordering::Acquire) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed host reader released its generation");
        let next = store
            .inner
            .lock()
            .ring
            .reserve(&replacement, slots(), index::Encoding::Raw)
            .unwrap();
        assert_ne!(next.begin, generation);
    }
}

pub(super) fn queued_read_store() -> (Arc<SsdStore>, tokio::sync::mpsc::Receiver<PrefetchBatch>) {
    queued_read_store_with_inventory(None, true)
}

fn queued_read_store_with_inventory(
    inventory: Option<Arc<crate::storage::inventory::ResidencyInventory>>,
    commit: bool,
) -> (Arc<SsdStore>, tokio::sync::mpsc::Receiver<PrefetchBatch>) {
    use std::os::fd::AsRawFd;

    let file = tempfile::tempfile().unwrap();
    file.set_len(SSD_ALIGNMENT as u64).unwrap();
    let io = Arc::new(
        UringIoEngine::new_multi(
            vec![file.as_raw_fd()],
            UringConfig {
                threads: 2,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let (write_tx, _) = tokio::sync::mpsc::channel(1);
    let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(1);
    let store = Arc::new(SsdStore {
        gpu_io: Arc::new(GpuIo {
            automatic: false,
            enabled: AtomicBool::new(false),
        }),
        read_path: None,
        _files: vec![file],
        cufile_files: Vec::new(),
        io,
        write_tx,
        write_policy: SsdWritePolicy::All,
        prefetch_tx,
        inner: Mutex::new(SsdInner {
            ring: SsdRingBuffer::new_sharded(vec![SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64),
            pending_writes: HashSet::new(),
            reuse_history: LruCache::new(1),
        }),
        allocate_fn: Arc::new(|_, _| None),
        allocation_footprint_fn: Arc::new(|bytes, _| Some(bytes)),
        is_numa: false,
        inventory,
    });
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let mut inner = store.inner.lock();
    inner
        .ring
        .reserve(
            &key,
            vec![crate::SlotMeta::new(
                smallvec::smallvec![SSD_ALIGNMENT as u64],
                NumaNode::UNKNOWN,
            )],
            index::Encoding::Raw,
        )
        .unwrap();
    drop(inner);
    if commit {
        store.commit_write(&key, true);
    }
    (store, prefetch_rx)
}

#[test]
fn ssd_evidence_appears_only_after_commit_and_survives_dram_eviction() {
    let inventory = Arc::new(crate::storage::inventory::ResidencyInventory::new(
        16 * 1024,
    ));
    let (store, _queued) = queued_read_store_with_inventory(Some(Arc::clone(&inventory)), false);
    let key = StateKey::new("queued-lease".into(), vec![0]);

    assert!(inventory.page(None).unwrap().is_empty());

    store.commit_write(&key, true);
    let ssd = inventory.page(None).unwrap();
    assert_eq!(ssd.len(), 1);
    assert_eq!(
        ssd[0].metadata.unwrap().medium,
        orbitkv_state::ReplicaMedium::Ssd
    );
    assert_eq!(
        ssd[0].metadata.unwrap().stored_bytes,
        Some(SSD_ALIGNMENT as u64)
    );

    let dram = crate::storage::dram::DramStore::with_inventory(
        4096,
        false,
        None,
        Some(Arc::clone(&inventory)),
        0,
    );
    dram.batch_insert(vec![(
        key.clone(),
        Arc::new(SealedBlock::for_policy_test(2048)),
    )]);
    assert_eq!(
        inventory.page(None).unwrap()[0].metadata.unwrap().medium,
        orbitkv_state::ReplicaMedium::Dram
    );
    dram.remove_all();
    assert_eq!(
        inventory.page(None).unwrap()[0].metadata.unwrap().medium,
        orbitkv_state::ReplicaMedium::Ssd
    );

    let replacement = StateKey::new("queued-lease".into(), vec![1]);
    let retired = {
        let mut inner = store.inner.lock();
        let mut slot =
            crate::SlotMeta::new(smallvec::smallvec![SSD_ALIGNMENT as u64], NumaNode::UNKNOWN);
        slot.encoding = Some(vec![crate::codec::EncodedSegment {
            version: 1,
            format: orbitkv_state::StorageFormat::Exact,
            logical_bytes: SSD_ALIGNMENT,
            stored_bytes: SSD_ALIGNMENT,
            checksum: 0,
        }]);
        inner
            .ring
            .reserve(&replacement, vec![slot], index::Encoding::Encoded)
            .unwrap();
        inner.ring.take_retired()
    };
    assert_eq!(retired, [key]);
    store.retire_inventory(retired);
    assert!(inventory.page(None).unwrap().is_empty());

    store.commit_write(&replacement, true);

    assert_eq!(
        inventory.page(None).unwrap()[0].metadata.unwrap().medium,
        orbitkv_state::ReplicaMedium::Ssd
    );
    let entry = store.inner.lock().ring.get(&replacement).unwrap().clone();
    store.invalidate_encoded_entry(&replacement, &entry);
    assert!(inventory.page(None).unwrap().is_empty());
}

#[test]
fn export_pins_exact_ssd_evidence_and_accounts_staging_allocations() {
    let inventory = Arc::new(crate::storage::inventory::ResidencyInventory::new(
        16 * 1024,
    ));
    let (store, _queued) = queued_read_store_with_inventory(Some(Arc::clone(&inventory)), true);
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let record = inventory.page(None).unwrap().pop().unwrap();
    let readers = Arc::clone(&store.inner.lock().ring.get(&key).unwrap().readers);
    let leases = store
        .pin_residencies(std::slice::from_ref(&record))
        .unwrap();
    assert_eq!(readers.load(Ordering::Acquire), 1);
    assert_eq!(store.staging_footprint(&leases), Some(SSD_ALIGNMENT as u64));
    drop(leases);
    assert_eq!(readers.load(Ordering::Acquire), 0);

    inventory.change(
        &key,
        orbitkv_state::ReplicaMedium::Dram,
        Some(orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Dram,
            representation: orbitkv_state::ReplicaRepresentation::Raw,
            stored_bytes: Some(SSD_ALIGNMENT as u64),
        }),
    );
    let leases = store
        .pin_residencies(std::slice::from_ref(&record))
        .unwrap();
    drop(leases);
    inventory.change(&key, orbitkv_state::ReplicaMedium::Ssd, None);
    assert!(store.pin_residencies(&[record]).is_none());
    assert_eq!(readers.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn cancelled_ssd_authorization_keeps_admission_with_queued_batch() {
    use orbitkv_catalog::MembershipView;
    use orbitkv_state::CacheOwner;

    let inventory = Arc::new(crate::storage::inventory::ResidencyInventory::new(
        16 * 1024,
    ));
    let (store, mut queued) = queued_read_store_with_inventory(Some(Arc::clone(&inventory)), true);
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let records = inventory.page(None).unwrap();
    let readers = Arc::clone(&store.inner.lock().ring.get(&key).unwrap().readers);
    let owner = CacheOwner {
        endpoint: "127.0.0.1:50055".into(),
        incarnation: uuid::Uuid::new_v4(),
    };
    let membership = Arc::new(MembershipView::new(owner.clone()));
    assert!(membership.renew(
        std::time::Instant::now(),
        std::time::Duration::from_secs(30)
    ));
    membership.replace_members([("source".into(), owner.clone())]);
    let dram = Arc::new(crate::storage::dram::DramStore::with_inventory(
        1 << 20,
        false,
        None,
        Some(inventory),
        0,
    ));
    let exports = crate::PeerExports::new(
        dram,
        Some(Arc::clone(&store)),
        Some(membership),
        Some("127.0.0.1:12345".into()),
        #[cfg(feature = "orbitte")]
        None,
        std::time::Duration::from_secs(30),
        SSD_ALIGNMENT as u64,
    );
    let ticket = crate::TransferTicket::new(
        exports
            .open(owner.incarnation, uuid::Uuid::new_v4())
            .unwrap(),
        0,
        1,
    )
    .unwrap();
    let mut authorization = Box::pin(exports.authorize(owner.incarnation, ticket, &records));
    assert!(futures::poll!(authorization.as_mut()).is_pending());
    let batch = queued.recv().await.unwrap();
    assert_eq!(exports.transfer_accounting(), (1, SSD_ALIGNMENT as u64));
    assert_eq!(readers.load(Ordering::Acquire), 1);

    drop(authorization);
    assert!(batch.done_tx.is_closed());
    assert_eq!(exports.transfer_accounting(), (1, SSD_ALIGNMENT as u64));
    assert_eq!(readers.load(Ordering::Acquire), 1);
    drop(batch);
    assert_eq!(exports.transfer_accounting(), (0, 0));
    assert_eq!(readers.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn candidates_do_not_pin_and_cannot_authorize_a_replaced_generation() {
    let (store, queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let mut candidates = store.discover(&[key.clone(), StateKey::new("ns".into(), vec![1])]);
    assert!(candidates.pop().unwrap().is_none());
    let candidate = candidates.pop().unwrap().unwrap();
    assert_eq!(candidate.entry.len, SSD_ALIGNMENT as u64);
    assert_eq!(candidate.entry.slots[0].total_size(), SSD_ALIGNMENT as u64);
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 0);
    assert!(!candidate.cufile_eligible(64 * 1024));
    assert_eq!(queued.len(), 0, "discovery must not enqueue payload reads");
    let lease = candidate.pin().unwrap();
    assert!(Arc::ptr_eq(&lease.entry.readers, &candidate.entry.readers));
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 1);
    drop(lease);

    // A replacement may reuse the same key and physical offset. The old
    // snapshot must neither pin that generation nor prevent its allocation.
    let mut inner = store.inner.lock();
    let replacement = StateKey::new("queued-lease".into(), vec![2]);
    for next in [&replacement, &key] {
        assert!(
            inner
                .ring
                .reserve(next, candidate.entry.slots.clone(), index::Encoding::Raw)
                .is_some()
        );
        assert!(inner.ring.commit(next, true));
    }
    assert_eq!(
        inner.ring.get(&key).unwrap().file_offset,
        candidate.entry.file_offset
    );
    drop(inner);
    assert!(candidate.pin().is_none());
    assert_eq!(candidate.entry.readers.load(Ordering::Acquire), 0);
    let fresh = store
        .discover(std::slice::from_ref(&key))
        .pop()
        .unwrap()
        .unwrap();
    assert!(fresh.pin().is_some());
    drop(store);
    assert!(fresh.pin().is_none());
}

#[tokio::test]
async fn host_routes_preserve_source_priority_permissions_and_complete_coverage() {
    use crate::QueryMode;
    use crate::planning::read::{HostReadRoute, ReadPlan, ReadTarget};

    let (store, queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let missing = StateKey::new("queued-lease".into(), vec![1]);
    let readers = Arc::clone(&store.inner.lock().ring.get(&key).unwrap().readers);

    for mode in [
        QueryMode::Demand,
        QueryMode::Prepare,
        QueryMode::WaitForFullPrefix,
    ] {
        for peer_available in [false, true] {
            for allow_ssd in [false, true] {
                let mut plan = ReadPlan::new(&[key.clone(), missing.clone()], mode, Some(&store));
                assert!(plan.deferred_ssd(&store, 0).is_none());
                assert!(
                    matches!(plan.target, ReadTarget::HostReady) == (mode == QueryMode::Prepare)
                );
                #[cfg(any(feature = "mooncake", feature = "orbitte"))]
                plan.rows[0].set_peers(vec![orbitkv_state::ReplicaLocation {
                    owner: orbitkv_state::CacheOwner {
                        endpoint: "source".into(),
                        incarnation: uuid::Uuid::from_u128(1),
                    },
                    sequence: 1,
                    metadata: orbitkv_state::ReplicaMetadata {
                        medium: orbitkv_state::ReplicaMedium::Dram,
                        representation: orbitkv_state::ReplicaRepresentation::Raw,
                        stored_bytes: Some(4096),
                    },
                }]);
                let route = plan.host_route(peer_available, allow_ssd, 0);
                let expected_peer =
                    cfg!(any(feature = "mooncake", feature = "orbitte")) && peer_available;
                if mode == QueryMode::WaitForFullPrefix {
                    assert!(route.is_none(), "neither source covers the complete demand");
                } else {
                    match route {
                        #[cfg(any(feature = "mooncake", feature = "orbitte"))]
                        Some(HostReadRoute::Peer(peer)) => {
                            assert!(expected_peer);
                            assert_eq!(peer.block_count(), 1);
                        }
                        Some(HostReadRoute::Ssd(ssd)) => {
                            assert!(allow_ssd && !expected_peer);
                            assert_eq!(readers.load(Ordering::Acquire), 0);
                            let leases = ssd.acquire(0).unwrap();
                            assert_eq!(leases.len(), 1);
                            assert!(Arc::ptr_eq(&leases[0].entry.readers, &readers));
                            assert_eq!(readers.load(Ordering::Acquire), 1);
                        }
                        None => assert!(!allow_ssd && !expected_peer),
                    }
                }
                assert_eq!(readers.load(Ordering::Acquire), 0);
                assert_eq!(
                    queued.len(),
                    0,
                    "route selection must not submit payload reads"
                );
            }
        }
    }
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[tokio::test]
async fn peer_ssd_route_is_explicit_and_follows_local_ssd_priority() {
    use crate::QueryMode;
    use crate::cost::{CostEstimateKey, CostObservationKind, Representation};
    use crate::planning::peer::PeerSource;
    use crate::planning::read::{HostReadRoute, ReadPlan};

    let (store, queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let peer_ssd = orbitkv_state::ReplicaLocation {
        owner: orbitkv_state::CacheOwner {
            endpoint: "source".into(),
            incarnation: uuid::Uuid::from_u128(1),
        },
        sequence: 1,
        metadata: orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Ssd,
            representation: orbitkv_state::ReplicaRepresentation::Raw,
            stored_bytes: Some(SSD_ALIGNMENT as u64),
        },
    };

    let mut local = ReadPlan::new(std::slice::from_ref(&key), QueryMode::Demand, Some(&store));
    local.rows[0].set_peers(vec![peer_ssd.clone()]);
    let Some(HostReadRoute::Ssd(local_route)) = local.host_route(true, true, 0) else {
        panic!("local SSD must remain ahead of peer SSD");
    };
    assert_eq!(
        local_route.cost_estimate_key(),
        Some(CostEstimateKey::new(
            CostObservationKind::LocalSsdHostReady,
            store.io.cost_resource,
            Representation::Raw,
            SSD_ALIGNMENT as u64,
            1,
        ))
    );
    let old = store.inner.lock().ring.get(&key).unwrap().clone();
    {
        let mut inner = store.inner.lock();
        for next in [StateKey::new("queued-lease".into(), vec![1]), key.clone()] {
            inner
                .ring
                .reserve(&next, old.slots.clone(), index::Encoding::Raw)
                .unwrap();
            assert!(inner.ring.commit(&next, true));
        }
    }
    assert!(local_route.acquire(0).is_none());
    let Some(HostReadRoute::Peer(fallback)) = local.host_route(true, false, 0) else {
        panic!("stale local SSD evidence must preserve the peer SSD fallback");
    };
    assert_eq!(fallback.next_segment(0).unwrap().source, PeerSource::Ssd);

    let mut remote = ReadPlan::new(std::slice::from_ref(&key), QueryMode::Demand, Some(&store));
    remote.rows[0].set_peers(vec![peer_ssd]);
    let Some(HostReadRoute::Peer(route)) = remote.host_route(true, false, 0) else {
        panic!("peer SSD should be executable when local host staging is disabled");
    };
    assert_eq!(route.next_segment(0).unwrap().source, PeerSource::Ssd);
    assert_eq!(queued.len(), 0, "planning cannot stage either source");
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[tokio::test]
async fn cross_medium_selection_is_explicit_and_preserves_equal_coverage() {
    const CHILD: &str = "ORBITKV_TEST_CROSS_MEDIUM_ROUTE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        use crate::QueryMode;
        use crate::cost::{
            CostEstimateKey, CostObservationKind, ExecutionResource, Representation, resource_id,
        };
        use crate::planning::peer::{FetchPlan, PeerSource};
        use crate::planning::read::{HostReadRoute, ReadPlan};

        let (store, _queued) = queued_read_store();
        let key = StateKey::new("queued-lease".into(), vec![0]);
        let owner = orbitkv_state::CacheOwner {
            endpoint: "peer".into(),
            incarnation: uuid::Uuid::from_u128(44),
        };
        let mut plan = ReadPlan::new(std::slice::from_ref(&key), QueryMode::Demand, Some(&store));
        plan.rows[0].set_peers(vec![orbitkv_state::ReplicaLocation {
            owner: owner.clone(),
            sequence: 1,
            metadata: orbitkv_state::ReplicaMetadata {
                medium: orbitkv_state::ReplicaMedium::Dram,
                representation: orbitkv_state::ReplicaRepresentation::Raw,
                stored_bytes: Some(SSD_ALIGNMENT as u64),
            },
        }]);
        let local_key = plan
            .ssd(SsdReadPath::Uring, 0)
            .unwrap()
            .cost_estimate_key()
            .unwrap();
        let peer_key = CostEstimateKey::new(
            CostObservationKind::PeerDramHostReady,
            ExecutionResource::Peer(resource_id(&owner)),
            Representation::Raw,
            SSD_ALIGNMENT as u64,
            1,
        );
        for _ in 0..4 {
            crate::cost::observe_for_test(local_key, 0.01, std::time::Instant::now());
            crate::cost::observe_for_test(peer_key, 0.02, std::time::Instant::now());
        }
        {
            let peer = FetchPlan::new(&mut plan.rows, 1, PeerSource::Dram).unwrap();
            assert_eq!(peer.complete_cost_estimate_key(), Some(peer_key));
        }
        assert!(matches!(
            plan.host_route(true, true, 0),
            Some(HostReadRoute::Ssd(_))
        ));
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "storage::ssd::tests::cross_medium_selection_is_explicit_and_preserves_equal_coverage",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("ORBITKV_COST_OBSERVATIONS", "1")
        .env("ORBITKV_COST_SELECTION", "1")
        .env("ORBITKV_CROSS_MEDIUM_SELECTION", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[tokio::test]
async fn consumed_restore_plan_deduplicates_sources_and_rejects_mixed_paths() {
    let (store, _queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let lease = store
        .discover(&[key])
        .pop()
        .unwrap()
        .unwrap()
        .pin()
        .unwrap();
    let cufile = crate::RestoreSource::Ssd {
        lease: Arc::clone(&lease),
        path: crate::SsdReadPath::Cufile,
        allow_uring_fallback: false,
    };
    let plan = crate::planning::restore::RestorePlan::new(2, [(9, &cufile), (9, &cufile)]).unwrap();
    assert_eq!(plan.device_id(), 2);
    assert_eq!(plan.ssd_path(), Some(crate::SsdReadPath::Cufile));
    assert_eq!(plan.ssd_source_bytes(), SSD_ALIGNMENT as u64);
    assert_eq!(plan.ssd_source_fragments(), 1);
    assert!(!plan.has_memory());

    let automatic = crate::RestoreSource::Ssd {
        lease: Arc::clone(&lease),
        path: crate::SsdReadPath::Cufile,
        allow_uring_fallback: true,
    };
    assert!(
        crate::planning::restore::RestorePlan::new(2, [(9, &cufile), (9, &automatic)])
            .unwrap_err()
            .contains("fallback policies")
    );

    let uring = crate::RestoreSource::Ssd {
        lease,
        path: crate::SsdReadPath::Uring,
        allow_uring_fallback: false,
    };
    assert!(
        crate::planning::restore::RestorePlan::new(2, [(9, &cufile), (9, &uring)])
            .unwrap_err()
            .contains("mix SSD read routes")
    );
}

#[tokio::test]
async fn cancelled_host_read_keeps_the_same_generation_owned_by_its_queue() {
    let (store, mut queued) = queued_read_store();
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let lease = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin())
        .collect::<Vec<_>>()
        .pop()
        .unwrap();
    let readers = Arc::clone(&lease.entry.readers);
    let source = Arc::downgrade(&lease);
    let mut read = Box::pin(store.read_host_batch(vec![Arc::clone(&lease)]));
    assert!(futures::poll!(read.as_mut()).is_pending());
    drop(read);
    drop(lease);

    let batch = queued.recv().await.unwrap();
    assert!(batch.done_tx.is_closed());
    assert_eq!(readers.load(Ordering::Acquire), 1);
    let request = &batch.requests[0];
    assert!(Arc::ptr_eq(&request.entry.readers, &readers));
    assert!(Arc::ptr_eq(request, &source.upgrade().unwrap()));
    drop(batch);
    assert_eq!(readers.load(Ordering::Acquire), 0);
    assert!(source.upgrade().is_none());

    queued.close();
    let lease = store
        .discover(std::slice::from_ref(&key))
        .into_iter()
        .map_while(|candidate| candidate?.pin())
        .collect::<Vec<_>>()
        .pop()
        .unwrap();
    assert!(
        store
            .read_host_batch(vec![Arc::clone(&lease)])
            .await
            .is_err()
    );
    assert_eq!(readers.load(Ordering::Acquire), 1);
    drop(lease);
    assert_eq!(readers.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn ssd_planning_preserves_default_preparation_and_explicit_route_rules() {
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

    for (path, mode, should_plan) in [
        (None, QueryMode::Demand, false),
        (Some(SsdReadPath::Uring), QueryMode::Demand, true),
        (Some(SsdReadPath::Cufile), QueryMode::Demand, false),
        (Some(SsdReadPath::Uring), QueryMode::Prepare, false),
        (Some(SsdReadPath::Uring), QueryMode::Warmup, false),
    ] {
        let (mut store, queued) = queued_read_store();
        Arc::get_mut(&mut store).unwrap().read_path = path;
        let key = StateKey::new("queued-lease".into(), vec![0]);
        let version = store.inner.lock().ring.get(&key).unwrap().readers.clone();
        let read = ReadPlan::new(std::slice::from_ref(&key), mode, Some(&store));
        let plan = read.deferred_ssd(&store, 64 * 1024);
        assert!(read.ssd(SsdReadPath::Uring, 64 * 1024).is_some());
        assert!(read.ssd(SsdReadPath::Cufile, 64 * 1024).is_none());
        assert_eq!(plan.is_some(), should_plan, "path={path:?}, mode={mode:?}");
        assert_eq!(version.load(Ordering::Acquire), 0);
        assert_eq!(queued.len(), 0, "planning cannot read payloads");
        if let Some(plan) = plan {
            assert_eq!(plan.path, SsdReadPath::Uring);
            let sources = plan.acquire(64 * 1024).unwrap();
            assert_eq!(version.load(Ordering::Acquire), 1);
            drop(sources);
            assert_eq!(version.load(Ordering::Acquire), 0);
        }
    }
}

#[tokio::test]
async fn ssd_plan_revalidates_versions_and_requires_complete_selected_prefixes() {
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

    let (mut store, queued) = queued_read_store();
    Arc::get_mut(&mut store).unwrap().read_path = Some(SsdReadPath::Uring);
    let key = StateKey::new("queued-lease".into(), vec![0]);
    let missing = StateKey::new("queued-lease".into(), vec![1]);
    let keys = [key.clone(), missing.clone()];
    assert!(
        ReadPlan::new(&keys, QueryMode::WaitForFullPrefix, Some(&store))
            .deferred_ssd(&store, 64 * 1024)
            .is_none()
    );
    let partial_read = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let partial = partial_read.deferred_ssd(&store, 64 * 1024).unwrap();
    assert_eq!(partial.acquire(64 * 1024).unwrap().len(), 1);

    let plan_read = ReadPlan::new(&keys[..1], QueryMode::Demand, Some(&store));
    let plan = plan_read.deferred_ssd(&store, 64 * 1024).unwrap();
    let mut inner = store.inner.lock();
    let old = inner.ring.get(&key).unwrap().clone();
    for next in [&missing, &key] {
        inner
            .ring
            .reserve(next, old.slots.clone(), index::Encoding::Raw)
            .unwrap();
        assert!(inner.ring.commit(next, true));
    }
    drop(inner);
    assert!(plan.acquire(64 * 1024).is_none());
    assert_eq!(old.readers.load(Ordering::Acquire), 0);

    // A later source can disappear after discovery. Strict acquisition must
    // release earlier leases; ordinary demand may still use that prefix.
    store._files[0].set_len(2 * SSD_ALIGNMENT as u64).unwrap();
    let mut inner = store.inner.lock();
    inner.ring = SsdRingBuffer::new_sharded(vec![2 * SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64);
    for (key, encoding) in [
        (&key, index::Encoding::Raw),
        (&missing, index::Encoding::Encoded),
    ] {
        inner
            .ring
            .reserve(key, old.slots.clone(), encoding)
            .unwrap();
        assert!(inner.ring.commit(key, true));
    }
    let first = inner.ring.get(&key).unwrap().clone();
    let last = inner.ring.get(&missing).unwrap().clone();
    drop(inner);
    let full_read = ReadPlan::new(&keys, QueryMode::WaitForFullPrefix, Some(&store));
    let full = full_read.deferred_ssd(&store, 64 * 1024).unwrap();
    let partial_read = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let partial = partial_read.deferred_ssd(&store, 64 * 1024).unwrap();
    store.inner.lock().ring.invalidate_encoded(&missing, &last);
    assert!(full.acquire(64 * 1024).is_none());
    assert_eq!(first.readers.load(Ordering::Acquire), 0);
    let prefix = partial.acquire(64 * 1024).unwrap();
    assert_eq!(prefix.len(), 1);
    assert_eq!(first.readers.load(Ordering::Acquire), 1);
    drop(prefix);
    assert_eq!(first.readers.load(Ordering::Acquire), 0);
    assert_eq!(queued.len(), 0);
}

#[tokio::test]
async fn batched_host_reads_hold_selected_generations_and_reject_foreign_stores() {
    use crate::QueryMode;
    use crate::planning::read::ReadPlan;

    let (store, mut queued) = queued_read_store();
    let (other, other_queue) = queued_read_store();
    let keys = [
        StateKey::new("queued-lease".into(), vec![0]),
        StateKey::new("queued-lease".into(), vec![1]),
    ];
    let (slots, versions) = {
        let mut inner = store.inner.lock();
        let slots = inner.ring.get(&keys[0]).unwrap().slots.clone();
        inner.ring =
            SsdRingBuffer::new_sharded(vec![2 * SSD_ALIGNMENT as u64], SSD_ALIGNMENT as u64);
        for key in &keys {
            inner
                .ring
                .reserve(key, slots.clone(), index::Encoding::Raw)
                .unwrap();
            assert!(inner.ring.commit(key, true));
        }
        let versions: Vec<_> = keys
            .iter()
            .map(|key| inner.ring.get(key).unwrap().readers.clone())
            .collect();
        (slots, versions)
    };
    let plan = ReadPlan::new(&keys, QueryMode::Prepare, Some(&store));
    let leases = plan.ssd(SsdReadPath::Uring, 0).unwrap().acquire(0).unwrap();
    assert_eq!(leases.len(), 2);
    assert!(other.read_host_batch(leases.clone()).await.is_err());
    assert!(other_queue.is_empty());
    let mut read = Box::pin(store.read_host_batch(leases));
    assert!(futures::poll!(read.as_mut()).is_pending());
    drop(read);
    let batch = queued.recv().await.unwrap();
    assert!(batch.done_tx.is_closed());
    assert_eq!(batch.requests.len(), 2);
    for (lease, version) in batch.requests.iter().zip(&versions) {
        assert!(Arc::ptr_eq(&lease.entry.readers, version));
        assert_eq!(version.load(Ordering::Acquire), 1);
    }
    assert!(
        store
            .inner
            .lock()
            .ring
            .reserve(
                &StateKey::new("queued-lease".into(), vec![2]),
                slots,
                index::Encoding::Raw
            )
            .is_none()
    );
    drop(batch);
    assert!(
        versions
            .iter()
            .all(|version| version.load(Ordering::Acquire) == 0)
    );
}

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
#[tokio::test]
async fn peer_rejection_updates_request_evidence_without_discarding_ssd_versions() {
    use crate::QueryMode;
    use crate::planning::{
        peer::{FetchPlan, PeerSource},
        read::ReadPlan,
    };
    use orbitkv_state::{CacheOwner, ReplicaLocation};

    let (store, queued) = queued_read_store();
    let keys = [
        StateKey::new("queued-lease".into(), vec![0]),
        StateKey::new("queued-lease".into(), vec![1]),
    ];
    let mut plan = ReadPlan::new(&keys, QueryMode::Demand, Some(&store));
    let location = ReplicaLocation {
        owner: CacheOwner {
            endpoint: "peer".into(),
            incarnation: uuid::Uuid::from_u128(1),
        },
        sequence: 7,
        metadata: orbitkv_state::ReplicaMetadata {
            medium: orbitkv_state::ReplicaMedium::Dram,
            representation: orbitkv_state::ReplicaRepresentation::Raw,
            stored_bytes: Some(4096),
        },
    };
    plan.rows[0].set_peers(vec![location]);
    assert!(FetchPlan::new(&mut plan.rows, 2, PeerSource::Dram).is_none());
    let mut route = FetchPlan::new(&mut plan.rows, 1, PeerSource::Dram).unwrap();
    let segment = route.next_segment(0).unwrap();
    route.reject(0, &segment);
    assert!(route.next_segment(0).is_none());
    assert!(
        plan.rows[0]
            .peer(orbitkv_state::ReplicaMedium::Dram)
            .next()
            .is_none()
    );
    assert_eq!(
        plan.rows.len(),
        2,
        "a short peer plan must not truncate other evidence"
    );
    let leases = plan.ssd(SsdReadPath::Uring, 0).unwrap().acquire(0).unwrap();
    assert_eq!(leases.len(), 1);
    assert!(
        queued.is_empty(),
        "selection and rejection must not read payloads"
    );
}
