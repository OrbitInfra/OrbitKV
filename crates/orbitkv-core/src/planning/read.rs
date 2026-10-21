use std::sync::Arc;

use crate::QueryMode;
use crate::block::StateKey;
use crate::storage::ssd::SsdStore;

#[cfg(feature = "peer")]
use super::peer::{FetchPlan, PeerSource};
use super::replica::ReplicaSet;
use super::ssd::SsdReadPlan;
#[cfg(feature = "peer")]
use crate::cost::{
    CostEstimateKey, SelectionScope, cross_medium_selection_enabled, select_route, shadow_routes,
};
#[cfg(feature = "peer")]
use smallvec::SmallVec;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadTarget {
    HostReady,
    EngineRestore,
}

/// A metadata-only route whose result is host-resident state. Source leases
/// and transfer grants are acquired by the selected execution owner.
pub(crate) enum HostReadRoute<'a> {
    Ssd(SsdReadPlan<'a>),
    #[cfg(feature = "peer")]
    Peer(FetchPlan<'a>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HostReadSelection {
    Ssd,
    #[cfg(feature = "peer")]
    Peer {
        prefix: usize,
        source: PeerSource,
    },
}

/// Unresolved sources for one admitted query batch. DRAM prefix holds already
/// belong to the caller; these records remain metadata until route acquisition.
pub(crate) struct ReadPlan {
    pub(crate) rows: Vec<ReplicaSet>,
    pub(crate) target: ReadTarget,
    pub(crate) required: usize,
    #[cfg(feature = "peer")]
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
            #[cfg(feature = "peer")]
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
        #[cfg(feature = "peer")]
        let peer_dram = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Dram);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };
        #[cfg(not(feature = "peer"))]
        let _ = peer_available;

        #[cfg(feature = "peer")]
        let peer_ssd = if peer_available {
            let prefix = FetchPlan::prefix_len(&self.rows, PeerSource::Ssd);
            (prefix > 0 && prefix >= self.required).then_some(prefix)
        } else {
            None
        };
        let local_ssd = allow_ssd && self.ssd(crate::SsdReadPath::Uring, codec_budget).is_some();

        #[cfg(feature = "peer")]
        let mut selected = peer_dram
            .map(|prefix| HostReadSelection::Peer {
                prefix,
                source: PeerSource::Dram,
            })
            .or_else(|| local_ssd.then_some(HostReadSelection::Ssd))
            .or_else(|| {
                peer_ssd.map(|prefix| HostReadSelection::Peer {
                    prefix,
                    source: PeerSource::Ssd,
                })
            });
        #[cfg(not(feature = "peer"))]
        let selected = local_ssd.then_some(HostReadSelection::Ssd);

        #[cfg(feature = "peer")]
        if crate::cost::enabled()
            && let Some(default_selection) = selected
        {
            selected = Some(self.evaluate_host_routes(
                default_selection,
                peer_dram,
                local_ssd,
                peer_ssd,
                codec_budget,
            ));
        }

        match selected? {
            HostReadSelection::Ssd => self
                .ssd(crate::SsdReadPath::Uring, codec_budget)
                .map(HostReadRoute::Ssd),
            #[cfg(feature = "peer")]
            HostReadSelection::Peer { prefix, source } => Some(HostReadRoute::Peer(
                FetchPlan::from_prefix(&mut self.rows, prefix, source),
            )),
        }
    }

    #[cfg(feature = "peer")]
    fn evaluate_host_routes(
        &mut self,
        selected: HostReadSelection,
        peer_dram: Option<usize>,
        local_ssd: bool,
        peer_ssd: Option<usize>,
        codec_budget: usize,
    ) -> HostReadSelection {
        let local_count = local_ssd
            .then(|| self.ssd(crate::SsdReadPath::Uring, codec_budget))
            .flatten()
            .map(|route| route.block_count());
        let selected_count = match selected {
            HostReadSelection::Ssd => local_count,
            HostReadSelection::Peer { prefix, .. } => Some(prefix),
        };
        let Some(selected_count) = selected_count else {
            return selected;
        };

        let expected_routes = usize::from(peer_dram == Some(selected_count))
            + usize::from(local_count == Some(selected_count))
            + usize::from(peer_ssd == Some(selected_count));

        let mut routes: SmallVec<[(HostReadSelection, CostEstimateKey); 3]> = SmallVec::new();
        if peer_dram == Some(selected_count) {
            let plan = FetchPlan::from_prefix(&mut self.rows, selected_count, PeerSource::Dram);
            if let Some(key) = plan.complete_cost_estimate_key() {
                routes.push((
                    HostReadSelection::Peer {
                        prefix: selected_count,
                        source: PeerSource::Dram,
                    },
                    key,
                ));
            }
        }
        if local_count == Some(selected_count)
            && let Some(key) = self
                .ssd(crate::SsdReadPath::Uring, codec_budget)
                .and_then(|route| route.cost_estimate_key())
        {
            routes.push((HostReadSelection::Ssd, key));
        }
        if peer_ssd == Some(selected_count) {
            let plan = FetchPlan::from_prefix(&mut self.rows, selected_count, PeerSource::Ssd);
            if let Some(key) = plan.complete_cost_estimate_key() {
                routes.push((
                    HostReadSelection::Peer {
                        prefix: selected_count,
                        source: PeerSource::Ssd,
                    },
                    key,
                ));
            }
        }
        let Some(selected_index) = routes.iter().position(|(route, _)| *route == selected) else {
            return selected;
        };
        let keys: SmallVec<[CostEstimateKey; 3]> = routes.iter().map(|(_, key)| *key).collect();
        shadow_routes(&keys, selected_index);
        if !cross_medium_selection_enabled() || routes.len() != expected_routes {
            return selected;
        }
        let selected_index = select_route(&keys, selected_index, SelectionScope::CrossMedium);
        routes
            .get(selected_index)
            .map_or(selected, |(route, _)| *route)
    }
}
