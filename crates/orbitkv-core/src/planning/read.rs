use std::sync::Arc;

use crate::QueryMode;
use crate::block::StateKey;
use crate::storage::ssd::SsdStore;

use super::peer::{FetchPlan, PeerSource};
use super::replica::ReplicaSet;
use super::ssd::SsdReadPlan;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadTarget {
    HostReady,
    EngineRestore,
}

/// A metadata-only acquisition route. The read target selects host residency
/// or deferred engine restore; execution owns source leases and transfer grants.
pub(crate) enum HostReadRoute<'a> {
    Ssd(SsdReadPlan<'a>),
    Peer(FetchPlan<'a>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostReadSelection {
    Ssd,
    Peer { prefix: usize, source: PeerSource },
}

/// Unresolved sources for one admitted query batch. DRAM prefix holds already
/// belong to the caller; these records remain metadata until route acquisition.
pub(crate) struct ReadPlan {
    pub(crate) rows: Vec<ReplicaSet>,
    pub(crate) target: ReadTarget,
    pub(crate) required: usize,
    pub(crate) wait_for_full_prefix: bool,
}

impl ReadPlan {
    pub(crate) fn new(keys: &[StateKey], mode: QueryMode, ssd: Option<&Arc<SsdStore>>) -> Self {
        let mut rows: Vec<_> = keys.iter().cloned().map(ReplicaSet::new).collect();
        if let Some(ssd) = ssd {
            for (row, candidate) in rows.iter_mut().zip(ssd.discover(keys)) {
                if let Some(candidate) = candidate {
                    row.set_ssd(candidate);
                }
            }
        }
        Self {
            rows,
            wait_for_full_prefix: mode == QueryMode::WaitForFullPrefix,
            target: if matches!(mode, QueryMode::Warmup | QueryMode::Prepare) {
                ReadTarget::HostReady
            } else {
                ReadTarget::EngineRestore
            },
            required: if mode == QueryMode::WaitForFullPrefix {
                keys.len()
            } else {
                1
            },
        }
    }

    pub(crate) fn host_route(
        &mut self,
        peer_available: bool,
        allow_ssd: bool,
        codec_budget: usize,
    ) -> Option<HostReadRoute<'_>> {
        let peer_dram = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Dram);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };

        let peer_ssd = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Ssd);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };
        let local_count = allow_ssd
            .then(|| self.ssd(crate::SsdReadPath::Uring, codec_budget))
            .flatten()
            .map(|route| route.block_count());
        // Coverage is compared before medium preference. Never shorten the hit
        // boundary just to select a preferred medium.
        let boundary = [local_count, peer_dram, peer_ssd]
            .into_iter()
            .flatten()
            .max()?;
        let candidates = [
            (local_count, HostReadSelection::Ssd),
            (
                peer_dram,
                HostReadSelection::Peer {
                    prefix: boundary,
                    source: PeerSource::Dram,
                },
            ),
            (
                peer_ssd,
                HostReadSelection::Peer {
                    prefix: boundary,
                    source: PeerSource::Ssd,
                },
            ),
        ];
        let selected = candidates
            .into_iter()
            .find(|(count, _)| *count == Some(boundary))?
            .1;
        match selected {
            HostReadSelection::Ssd => self
                .ssd(crate::SsdReadPath::Uring, codec_budget)
                .map(HostReadRoute::Ssd),
            HostReadSelection::Peer { prefix, source } => Some(HostReadRoute::Peer(
                FetchPlan::from_prefix(&mut self.rows, prefix, source),
            )),
        }
    }
}
