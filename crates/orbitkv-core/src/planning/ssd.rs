use std::sync::Arc;

use crate::SsdReadPath;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use crate::cost::{CostEstimateKey, CostObservationKind};
use crate::storage::ssd::{SsdReadLease, SsdStore};

use super::read::{ReadPlan, ReadTarget};
use super::replica::ReplicaSet;

/// A route over request-owned evidence, not another copy of the SSD inventory.
pub(crate) struct SsdReadPlan<'a> {
    rows: &'a [ReplicaSet],
    pub(crate) path: SsdReadPath,
    pub(crate) allow_uring_fallback: bool,
    required: usize,
}

impl ReadPlan {
    pub(crate) fn deferred_ssd(
        &self,
        store: &SsdStore,
        codec_budget: usize,
    ) -> Option<SsdReadPlan<'_>> {
        if self.target != ReadTarget::EngineRestore {
            return None;
        }
        let (path, allow_uring_fallback) =
            deferred_route(store.read_path, store.gpu_io.available())?;
        self.ssd_route(path, codec_budget, allow_uring_fallback)
    }

    pub(crate) fn ssd(&self, path: SsdReadPath, codec_budget: usize) -> Option<SsdReadPlan<'_>> {
        self.ssd_route(path, codec_budget, false)
    }

    fn ssd_route(
        &self,
        path: SsdReadPath,
        codec_budget: usize,
        allow_uring_fallback: bool,
    ) -> Option<SsdReadPlan<'_>> {
        let count = self
            .rows
            .iter()
            .take_while(|row| row.local_ssd().is_some())
            .count();
        if count == 0 || count < self.required {
            return None;
        }
        let rows = &self.rows[..count];
        if path == SsdReadPath::Cufile
            && rows.iter().any(|row| {
                !row.local_ssd()
                    .is_some_and(|source| source.cufile_eligible(codec_budget))
            })
        {
            return None;
        }
        Some(SsdReadPlan {
            rows,
            path,
            allow_uring_fallback,
            required: self.required,
        })
    }
}

fn deferred_route(
    explicit: Option<SsdReadPath>,
    cufile_available: bool,
) -> Option<(SsdReadPath, bool)> {
    match explicit {
        Some(path) => Some((path, false)),
        None if cufile_available => Some((SsdReadPath::Cufile, true)),
        None => None,
    }
}

impl SsdReadPlan<'_> {
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn block_count(&self) -> usize {
        self.rows.len()
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn cost_estimate_key(&self) -> Option<CostEstimateKey> {
        let first = self.rows.first()?.local_ssd()?;
        let resource = first.cost_resource()?;
        let mut bytes = 0u64;
        let mut representation = None;
        for row in self.rows {
            let candidate = row.local_ssd()?;
            if candidate.cost_resource()? != resource {
                return None;
            }
            let metadata = candidate.replica_metadata();
            bytes = bytes.checked_add(metadata.stored_bytes?)?;
            representation = Some(match representation {
                None => metadata.representation,
                Some(orbitkv_state::ReplicaRepresentation::Unknown) => {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(_)
                    if metadata.representation == orbitkv_state::ReplicaRepresentation::Unknown =>
                {
                    orbitkv_state::ReplicaRepresentation::Unknown
                }
                Some(previous) if previous == metadata.representation => previous,
                Some(_) => orbitkv_state::ReplicaRepresentation::Mixed,
            });
        }
        Some(CostEstimateKey::new(
            CostObservationKind::LocalSsdHostReady,
            resource,
            representation.unwrap_or_default(),
            bytes,
            self.rows.len(),
        ))
    }

    pub(crate) fn acquire(self, codec_budget: usize) -> Option<Vec<Arc<SsdReadLease>>> {
        let leases: Vec<_> = self
            .rows
            .iter()
            .map_while(|row| row.local_ssd()?.pin())
            .collect();
        if leases.len() < self.required
            || (self.path == SsdReadPath::Cufile
                && leases
                    .iter()
                    .any(|lease| !lease.cufile_eligible(codec_budget)))
        {
            return None;
        }
        Some(leases)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/ssd.rs"]
mod tests;
