use orbitkv_state::{CacheOwner, DISCOVERY_MAX_BYTES, DISCOVERY_MAX_KEYS, InventoryRecord};
use smallvec::SmallVec;

use super::replica::ReplicaSet;

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
}

pub(crate) struct FetchSegment {
    pub(crate) owner: CacheOwner,
    pub(crate) source: PeerSource,
    pub(crate) records: Vec<InventoryRecord>,
}

pub(crate) struct FetchPlan<'a> {
    rows: &'a mut [ReplicaSet],
    source: PeerSource,
}

struct PeerChoice<'a> {
    owner: &'a CacheOwner,
    count: usize,
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
        }
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
        let owner = candidates.first()?.0;
        Some(PeerChoice { owner, count })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/peer.rs"]
mod tests;
