//! Deferred source grants for GPU restores. DRAM-cache fills use `read` instead.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use super::{
    completion::TransferLockGuard,
    execute::{
        AuthorizationError, AuthorizationMode, FetchResult, SegmentFetcher, execute_fetch_plan,
    },
    read::{AuthorizedSegment, PeerReader, transfer_timeout_from_server},
    segment::{PeerIdentity, PeerSegments},
};
use crate::planning::peer::{FetchPlan, FetchSegment};
use crate::transfer_engine::{
    Cancellation, CudaRegistration, ErrorKind, PeerTransferError, TransferEngine, TransferOp,
    TransferOptions, TransferSlice,
};
use crate::{SlotMeta, block::StateKey, memory::numa::NumaNode};

/// Immutable source descriptors plus the source's authorization hold. Clones
/// share one ticket across query leases and all TP/layer restore consumers.
#[derive(Clone)]
pub struct PeerRestoreBlock {
    grant: Arc<Grant>,
    pub(crate) slots: Arc<[PeerSlot]>,
    hash: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct PeerSlot {
    pub(crate) meta: SlotMeta,
    addresses: Vec<u64>,
    pub(crate) checksums: Vec<crate::integrity::Checksum>,
}

struct Grant {
    _lock: TransferLockGuard,
    _segments: Arc<PeerSegments>,
    peer: Arc<PeerIdentity>,
    options: TransferOptions,
}

impl PeerSlot {
    pub(crate) fn checksum(
        &self,
        segment: usize,
        offset: usize,
        bytes: usize,
    ) -> Result<u32, String> {
        let base: u64 = self.meta.segment_sizes[..segment].iter().sum();
        crate::integrity::range(&self.checksums, base + offset as u64, bytes as u64)
    }
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
    pub(crate) fn invalidate_corrupt(&self) {
        let mut bad = self.grant._lock.corrupt.lock();
        // Each grant contains at most DISCOVERY_MAX_KEYS distinct hashes.
        if !bad.contains(&self.hash) {
            bad.push(self.hash.clone());
        }
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.slots.iter().map(|slot| slot.meta.total_size()).sum()
    }

    pub(crate) fn grant_id(&self) -> usize {
        Arc::as_ptr(&self.grant) as usize
    }
    pub(crate) fn same_grant(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.grant, &other.grant)
    }
    pub(crate) fn backend(&self) -> Arc<TransferEngine> {
        Arc::clone(&self.grant.peer.backend)
    }

    /// All ranges belong to persistent registrations. The engine slices and
    /// pipelines this batch; only an unknown physical outcome retains its grants.
    #[allow(
        clippy::mem_forget,
        reason = "unknown drain retains source authority and GPU registrations"
    )]
    pub(crate) unsafe fn read_gpu(
        &self,
        slices: &[TransferSlice],
        owners: &[Arc<CudaRegistration>],
    ) -> Result<usize, PeerTransferError> {
        let grant = &self.grant;
        crate::transfer_engine::remaining(grant.options.deadline)?;
        let total = crate::transfer_engine::validate_slices(slices)?;
        if slices.len() > 16_384 || total > 64 << 20 || owners.len() != slices.len() {
            return Err(PeerTransferError::Failed(
                "peer GPU batch exceeds descriptor or byte limits".into(),
            ));
        }
        if slices
            .iter()
            .zip(owners)
            .any(|(slice, owner)| !owner.covers(slice.local.as_ptr() as u64, slice.length))
        {
            return Err(PeerTransferError::Failed(
                "peer GPU range exceeds its registered tensor".into(),
            ));
        }
        let mut registrations: Vec<_> = owners.iter().collect();
        registrations.sort_unstable_by_key(|owner| Arc::as_ptr(owner));
        registrations.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _loans = registrations
            .iter()
            .map(|owner| owner.loan(&grant.peer.backend))
            .collect::<Result<Vec<_>, _>>()?;
        let peer = grant.peer.open(grant.options.deadline)?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: registrations and source authority are held through physical drain.
            unsafe {
                peer.transfer(
                    TransferOp::Read,
                    slices,
                    grant.options,
                    &Cancellation::default(),
                )
            }
        }));
        match result {
            Ok(result) if !matches!(&result, Err(error) if error.kind == ErrorKind::Unquiesced) => {
                let transferred = result.map_err(PeerTransferError::from)?;
                if transferred != total {
                    return Err(PeerTransferError::Failed("short peer GPU READ".into()));
                }
                Ok(transferred)
            }
            uncertain => {
                for owner in owners {
                    owner.retain();
                }
                std::mem::forget((uncertain, Arc::clone(grant), owners.to_vec()));
                Err(PeerTransferError::Unquiesced)
            }
        }
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
        options: TransferOptions,
    ) -> Result<Self::Grant, AuthorizationError> {
        // This is source admission, not a completed HostReady route sample.
        self.0
            .authorize_segment(segment, AuthorizationMode::Lookahead, options)
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
            _segments: grant.segments,
            peer: grant.peer,
            options: TransferOptions {
                deadline: grant.options.deadline.min(deadline),
                ..grant.options
            },
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
                        hash: record.key.hash.clone(),
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
            let checksums = crate::integrity::from_wire(&slot.checksums, meta.total_size())?;
            Ok(PeerSlot {
                meta,
                addresses,
                checksums,
            })
        })
        .collect()
}

impl PeerReader {
    pub(crate) async fn restore_plan(
        &self,
        plan: FetchPlan<'_>,
        req_id: &str,
        nodes: &[NumaNode],
        options: TransferOptions,
    ) -> FetchResult<PeerRestoreBlock> {
        execute_fetch_plan(&RestoreReader(self), plan, req_id, nodes, options).await
    }
}

#[cfg(test)]
#[path = "../../tests/unit/peer/restore.rs"]
pub(crate) mod tests;
