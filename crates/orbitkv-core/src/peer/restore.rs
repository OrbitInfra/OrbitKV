//! Deferred source grants for GPU restores. DRAM-cache fills use `read` instead.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::Arc,
    time::Instant,
};

use super::{
    completion::TransferLockGuard,
    execute::{
        AuthorizationError, AuthorizationMode, FetchResult, SegmentFetcher, execute_fetch_plan,
    },
    read::{AuthorizedSegment, PeerReader, transfer_timeout_from_server},
};
use crate::planning::peer::{FetchPlan, FetchSegment};
use crate::transfer_engine::{
    Cancellation, ErrorKind, PeerTransferError, Permission, TransferBackend, TransferError,
    TransferOp, TransferSlice,
};
use crate::{SlotMeta, block::StateKey, memory::numa::NumaNode};

/// Immutable source descriptors plus the source's authorization hold. Clones
/// share one ticket across query leases and all TP/layer restore consumers.
#[derive(Clone)]
pub struct PeerRestoreBlock {
    grant: Arc<Grant>,
    pub(crate) slots: Arc<[PeerSlot]>,
}

#[derive(Clone)]
pub(crate) struct PeerSlot {
    pub(crate) meta: SlotMeta,
    addresses: Vec<u64>,
}

struct Grant {
    _lock: TransferLockGuard,
    transfer: Arc<dyn TransferBackend>,
    endpoint: String,
    epoch: String,
    deadline: Instant,
    membership: Arc<orbitkv_catalog::MembershipView>,
    source_owner: orbitkv_state::CacheOwner,
}

impl PeerSlot {
    pub(crate) fn range(
        &self,
        segment: usize,
        offset: usize,
        length: usize,
    ) -> Result<u64, String> {
        let size = *self
            .meta
            .segment_sizes
            .get(segment)
            .ok_or("missing peer segment")?;
        let end = offset.checked_add(length).ok_or("peer range overflow")?;
        if length == 0 || end as u64 > size {
            return Err("peer range exceeds authorized segment".into());
        }
        self.addresses[segment]
            .checked_add(offset as u64)
            .ok_or("peer address overflow".into())
    }
}

impl PeerRestoreBlock {
    pub(crate) fn bytes(&self) -> u64 {
        self.slots.iter().map(|slot| slot.meta.total_size()).sum()
    }

    pub(crate) fn source_id(&self) -> u64 {
        crate::cost::resource_id(&(&self.grant.endpoint, &self.grant.epoch))
    }

    /// The caller owns exclusive GPU destinations and actual imported backing.
    /// Every non-Unquiesced result proves drain; no retry follows submission.
    #[allow(
        clippy::mem_forget,
        reason = "unknown drain retains source authority and actual GPU owners"
    )]
    pub(crate) unsafe fn read_gpu(
        &self,
        remote: u64,
        local: u64,
        length: usize,
        device: i32,
        owner: Arc<dyn Send + Sync>,
    ) -> Result<usize, PeerTransferError> {
        // Each admission uses at most 16 MiB and one region. At most 128
        // restores per GPU are admitted; engine-wide byte/region limits apply.
        const CHUNK: usize = 16 << 20;
        let grant = &self.grant;
        let check_admission = || {
            if !grant.membership.permits(&grant.source_owner) {
                return Err(PeerTransferError::Failed(
                    "peer GPU source membership is fenced".into(),
                ));
            }
            crate::transfer_engine::remaining(grant.deadline)
                .map(|_| ())
                .map_err(Into::into)
        };
        check_admission()?;
        let peer = grant
            .transfer
            .open_peer(&grant.endpoint, &grant.epoch, grant.deadline)?;
        let mut done = 0;
        while done < length {
            check_admission()?;
            let bytes = CHUNK.min(length - done);
            let address = local
                .checked_add(done as u64)
                .ok_or_else(|| PeerTransferError::Failed("GPU address overflow".into()))?;
            let remote_address = remote
                .checked_add(done as u64)
                .ok_or_else(|| PeerTransferError::Failed("peer address overflow".into()))?;
            let ptr = usize::try_from(address)
                .ok()
                .and_then(|p| NonNull::new(p as *mut u8))
                .ok_or_else(|| PeerTransferError::Failed("invalid GPU destination".into()))?;
            let mut registration = match catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: the caller lends this actual backing exclusively through drain.
                unsafe {
                    Arc::clone(&grant.transfer).register_cuda(
                        address,
                        bytes,
                        device,
                        Arc::clone(&owner),
                        Permission::LocalOnly,
                    )
                }
            })) {
                Ok(Ok(registration)) => registration,
                Ok(Err(error)) if error.kind != ErrorKind::Unquiesced => {
                    return Err(error.into());
                }
                uncertain => {
                    std::mem::forget((uncertain, Arc::clone(grant), owner));
                    return Err(PeerTransferError::Unquiesced);
                }
            };
            // Keep the registration outside the unwind boundary: its destructor
            // must not unregister while a panicking backend may still access it.
            let result = catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: the registration and source ticket outlive the READ.
                let result = unsafe {
                    grant.transfer.transfer(
                        TransferOp::Read,
                        &peer,
                        &[TransferSlice {
                            local: ptr,
                            remote_address,
                            length: bytes,
                        }],
                        grant.deadline,
                        &Cancellation::default(),
                    )
                };
                if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) {
                    // SAFETY: only drained payload outcomes may begin unregister.
                    unsafe { grant.transfer.unregister(&mut registration) }.map_err(|error| {
                        TransferError::new(ErrorKind::Unquiesced, error.to_string())
                    })?;
                }
                result
            }));
            let transferred = match result {
                Ok(result) if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) => {
                    result.map_err(PeerTransferError::from)?
                }
                uncertain => {
                    std::mem::forget((uncertain, registration, Arc::clone(grant), owner));
                    return Err(PeerTransferError::Unquiesced);
                }
            };
            if transferred != bytes {
                return Err(PeerTransferError::Failed("short peer GPU READ".into()));
            }
            done += bytes;
        }
        Ok(done)
    }
}

struct RestoreReader<'a>(&'a PeerReader);

#[tonic::async_trait]
impl SegmentFetcher for RestoreReader<'_> {
    type Grant = AuthorizedSegment;
    type Block = PeerRestoreBlock;

    async fn authorize_segment(
        &self,
        segment: &FetchSegment,
        _mode: AuthorizationMode,
    ) -> Result<Self::Grant, AuthorizationError> {
        // This is source admission, not a completed HostReady route sample.
        self.0
            .authorize_segment(segment, AuthorizationMode::Lookahead)
            .await
    }

    async fn fetch_segment(
        &self,
        segment: &FetchSegment,
        grant: Self::Grant,
        _req_id: &str,
        destination_nodes: &[NumaNode],
    ) -> Result<Vec<(StateKey, Self::Block)>, ()> {
        let deadline = grant
            .authorized_at
            .checked_add(transfer_timeout_from_server(
                grant.response.lock_timeout_secs,
            ))
            .ok_or(())?;
        let slots = grant
            .response
            .blocks
            .iter()
            .map(|block| validate_slots(&block.slots, destination_nodes.len()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| log::warn!("Invalid peer restore grant: {error}"))?;
        slots
            .iter()
            .flatten()
            .try_fold(0u64, |total, slot| {
                slot.meta
                    .segment_sizes
                    .iter()
                    .try_fold(total, |total, size| total.checked_add(*size))
            })
            .ok_or(())?;
        let owner = Arc::new(Grant {
            _lock: grant.lock_guard,
            transfer: Arc::clone(&self.0.transfer),
            endpoint: grant.response.transfer_endpoint,
            epoch: grant.response.native_source_epoch,
            deadline,
            membership: Arc::clone(&self.0.membership),
            source_owner: segment.owner.clone(),
        });
        Ok(segment
            .records
            .iter()
            .zip(slots)
            .map(|(record, slots)| {
                (
                    record.key.clone(),
                    PeerRestoreBlock {
                        grant: Arc::clone(&owner),
                        slots: slots.into(),
                    },
                )
            })
            .collect())
    }
}

fn validate_slots(
    slots: &[orbitkv_proto::proto::engine::TransferSlotInfo],
    expected: usize,
) -> Result<Vec<PeerSlot>, String> {
    if slots.len() != expected {
        return Err("peer slot count differs from destination topology".into());
    }
    slots
        .iter()
        .map(|slot| {
            if slot.k_size == 0 || (slot.v_size == 0 && slot.v_ptr != 0) {
                return Err("invalid peer segment sizes".into());
            }
            let mut addresses = vec![slot.k_ptr];
            let mut sizes = vec![slot.k_size];
            if slot.v_size != 0 {
                addresses.push(slot.v_ptr);
                sizes.push(slot.v_size);
            }
            for (&address, &size) in addresses.iter().zip(&sizes) {
                if address == 0
                    || address.checked_add(size).is_none()
                    || usize::try_from(size).is_err()
                {
                    return Err("invalid peer range".into());
                }
            }
            sizes
                .iter()
                .try_fold(0u64, |total, size| total.checked_add(*size))
                .ok_or("peer slot size overflow")?;
            let mut meta = SlotMeta::new(sizes.into(), NumaNode::UNKNOWN);
            if !slot.encoding.is_empty() {
                if slot.encoding.len() > 4096 {
                    return Err("remote codec metadata too large".into());
                }
                let encoding: Vec<crate::codec::EncodedSegment> =
                    serde_json::from_slice(&slot.encoding).map_err(|error| error.to_string())?;
                if encoding.len() != meta.num_segments() {
                    return Err("peer codec segment count differs".into());
                }
                for (segment, size) in encoding.iter().zip(&meta.segment_sizes) {
                    segment.validate_metadata(*size as usize)?;
                }
                meta.encoding = Some(encoding);
            }
            Ok(PeerSlot { meta, addresses })
        })
        .collect()
}

impl PeerReader {
    pub(crate) async fn restore_plan(
        &self,
        plan: FetchPlan<'_>,
        req_id: &str,
        nodes: &[NumaNode],
    ) -> FetchResult<PeerRestoreBlock> {
        execute_fetch_plan(&RestoreReader(self), plan.for_gpu_restore(), req_id, nodes).await
    }
}

#[cfg(test)]
#[path = "../../tests/unit/peer/restore.rs"]
pub(crate) mod tests;
