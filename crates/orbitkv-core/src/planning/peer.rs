use orbitkv_state::{CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord};
use smallvec::SmallVec;

use super::replica::ReplicaSet;
use crate::cost::{
    CostEstimateKey, CostObservationKind, ExecutionResource, SelectionScope, resource_id,
    select_route, selection_enabled,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeerSource {
    Dram,
    Ssd,
}

impl PeerSource {
    pub(crate) fn medium(self) -> orbitkv_state::ReplicaMedium {
        match self {
            Self::Dram => orbitkv_state::ReplicaMedium::Dram,
            Self::Ssd => orbitkv_state::ReplicaMedium::Ssd,
        }
    }

    fn cost_observation_kind(self) -> CostObservationKind {
        match self {
            Self::Dram => CostObservationKind::PeerDramHostReady,
            Self::Ssd => CostObservationKind::PeerSsdHostReady,
        }
    }
}

pub(crate) struct FetchSegment {
    pub(crate) owner: CacheOwner,
    pub(crate) source: PeerSource,
    pub(crate) records: Vec<InventoryRecord>,
    pub(crate) stored_bytes: Option<u64>,
    pub(crate) representation: orbitkv_state::ReplicaRepresentation,
}

impl FetchSegment {
    pub(crate) fn cost_estimate_key(&self) -> CostEstimateKey {
        CostEstimateKey::new(
            self.source.cost_observation_kind(),
            ExecutionResource::Peer(resource_id(&self.owner)),
            self.representation,
            self.stored_bytes.unwrap_or(0),
            self.records.len(),
        )
    }
}

pub(crate) struct FetchPlan<'a> {
    rows: &'a mut [ReplicaSet],
    source: PeerSource,
    host_costs: bool,
}

struct PeerChoice<'a> {
    owner: &'a CacheOwner,
    count: usize,
    stored_bytes: Option<u64>,
    representation: orbitkv_state::ReplicaRepresentation,
}

impl<'a> FetchPlan<'a> {
    #[cfg(test)]
    pub(crate) fn new(
        rows: &'a mut [ReplicaSet],
        required: usize,
        source: PeerSource,
    ) -> Option<Self> {
        let prefix = Self::prefix_len(rows, source);
        (prefix > 0 && prefix >= required).then(|| Self::from_prefix(rows, prefix, source))
    }

    pub(super) fn prefix_len(rows: &[ReplicaSet], source: PeerSource) -> usize {
        rows.iter()
            .take_while(|row| row.peer(source.medium()).next().is_some())
            .count()
    }

    pub(super) fn from_prefix(
        rows: &'a mut [ReplicaSet],
        prefix: usize,
        source: PeerSource,
    ) -> Self {
        Self {
            rows: &mut rows[..prefix],
            source,
            host_costs: true,
        }
    }

    /// HostReady measurements do not price a direct GPU completion target.
    pub(crate) fn for_gpu_restore(mut self) -> Self {
        self.host_costs = false;
        self
    }

    pub(crate) fn reject(&mut self, start: usize, segment: &FetchSegment) {
        for row in &mut self.rows[start..][..segment.records.len()] {
            row.reject_peer(&segment.owner);
        }
    }

    pub(crate) fn block_count(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn next_segment(&self, start: usize) -> Option<FetchSegment> {
        let choice = self.selected_choice(start)?;
        let mut records = Vec::with_capacity(choice.count);
        for row in &self.rows[start..start + choice.count] {
            let replica = row
                .peer(self.source.medium())
                .find(|replica| &replica.owner == choice.owner)?;
            records.push(InventoryRecord {
                key: row.key.clone(),
                sequence: replica.sequence,
                present: true,
                metadata: Some(replica.metadata),
            });
        }
        Some(FetchSegment {
            owner: choice.owner.clone(),
            source: self.source,
            records,
            stored_bytes: choice.stored_bytes,
            representation: choice.representation,
        })
    }

    pub(crate) fn complete_cost_estimate_key(&self) -> Option<CostEstimateKey> {
        let choice = self.selected_choice(0)?;
        (choice.count == self.rows.len()).then(|| {
            CostEstimateKey::new(
                self.source.cost_observation_kind(),
                ExecutionResource::Peer(resource_id(choice.owner)),
                choice.representation,
                choice.stored_bytes.unwrap_or(0),
                choice.count,
            )
        })
    }

    fn selected_choice(&self, start: usize) -> Option<PeerChoice<'_>> {
        let row = self.rows.get(start)?;
        let mut candidates: SmallVec<[(&CacheOwner, usize); 4]> = SmallVec::new();
        for candidate in row.peer(self.source.medium()) {
            let mut bytes = row.key.namespace.len();
            let count = self.rows[start..]
                .iter()
                .take(DISCOVERY_MAX_KEYS)
                .map_while(|row| {
                    row.peer(self.source.medium())
                        .find(|r| r.owner == candidate.owner)?;
                    bytes = bytes.saturating_add(row.key.hash.len());
                    (bytes <= DISCOVERY_MAX_BYTES).then_some(())
                })
                .count();
            if count == 0 {
                continue;
            }
            candidates.push((&candidate.owner, count));
        }
        candidates.sort_unstable_by(|(left_owner, left_count), (right_owner, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_owner.cmp(right_owner))
        });
        let count = candidates.first()?.1;
        candidates.retain(|(_, coverage)| *coverage == count);
        let selected = if self.host_costs && selection_enabled() {
            let keys: SmallVec<[CostEstimateKey; 4]> = candidates
                .iter()
                .map(|(owner, _)| {
                    let (stored_bytes, representation) =
                        self.shape_for_owner(start, count, owner)?;
                    Some(CostEstimateKey::new(
                        self.source.cost_observation_kind(),
                        ExecutionResource::Peer(resource_id(owner)),
                        representation.unwrap_or_default(),
                        stored_bytes.unwrap_or(0),
                        count,
                    ))
                })
                .collect::<Option<_>>()?;
            select_route(&keys, 0, SelectionScope::PeerOwner)
        } else {
            0
        };
        let owner = candidates.get(selected)?.0;
        let (stored_bytes, representation) = self.shape_for_owner(start, count, owner)?;
        Some(PeerChoice {
            owner,
            count,
            stored_bytes,
            representation: representation.unwrap_or_default(),
        })
    }

    fn shape_for_owner(
        &self,
        start: usize,
        count: usize,
        owner: &CacheOwner,
    ) -> Option<(Option<u64>, Option<orbitkv_state::ReplicaRepresentation>)> {
        let mut stored_bytes = Some(0u64);
        let mut representation = None;
        for row in &self.rows[start..start + count] {
            let replica = row
                .peer(self.source.medium())
                .find(|replica| &replica.owner == owner)?;
            stored_bytes = stored_bytes
                .zip(replica.metadata.stored_bytes)
                .and_then(|(total, bytes)| total.checked_add(bytes));
            let next = replica.metadata.representation;
            representation = Some(match representation {
                None => next,
                Some(orbitkv_state::ReplicaRepresentation::Unknown) => {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(_) if next == orbitkv_state::ReplicaRepresentation::Unknown => {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(previous) if previous == next => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
        }
        Some((stored_bytes, representation))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
