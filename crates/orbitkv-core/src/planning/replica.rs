use std::sync::Weak;

use orbitkv_state::ReplicaMedium;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use orbitkv_state::{DISCOVERY_MAX_REPLICAS_PER_MEDIUM, DiscoveryCoverage, ReplicaLocation};
use smallvec::SmallVec;

use crate::block::{SealedBlock, StateKey};
use crate::storage::ssd::SsdReadCandidate;

enum Evidence {
    Memory(Weak<SealedBlock>),
    Extent(SsdReadCandidate),
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    Peer(ReplicaLocation),
}

/// A location/version hint. Its evidence owns no payload or transfer permission.
/// Medium and acquisition authority are independent: peer evidence is not a tier.
pub(crate) struct ReplicaCandidate {
    medium: ReplicaMedium,
    evidence: Evidence,
}

impl ReplicaCandidate {
    pub(crate) fn ssd(candidate: SsdReadCandidate) -> Self {
        Self {
            medium: ReplicaMedium::Ssd,
            evidence: Evidence::Extent(candidate),
        }
    }

    fn is_available(&self) -> bool {
        match &self.evidence {
            Evidence::Memory(block) => block.strong_count() > 0,
            Evidence::Extent(_) => true,
            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            Evidence::Peer(_) => true,
        }
    }

    pub(crate) fn local_ssd(&self) -> Option<&SsdReadCandidate> {
        match &self.evidence {
            Evidence::Extent(candidate) if self.medium == ReplicaMedium::Ssd => Some(candidate),
            _ => None,
        }
    }

    fn is_peer(&self) -> bool {
        match self.evidence {
            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            Evidence::Peer(_) => true,
            _ => false,
        }
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    fn peer(&self, medium: ReplicaMedium) -> Option<&ReplicaLocation> {
        match &self.evidence {
            Evidence::Peer(location) if self.medium == medium => Some(location),
            _ => None,
        }
    }
}

/// At most one replica per local store plus the directory's bounded peer set.
pub(crate) struct ReplicaSet {
    pub(crate) key: StateKey,
    replicas: SmallVec<[ReplicaCandidate; 2]>,
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    remote_coverage: Option<DiscoveryCoverage>,
}

impl ReplicaSet {
    pub(crate) fn new(key: StateKey) -> Self {
        Self {
            key,
            replicas: SmallVec::new(),
            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            remote_coverage: None,
        }
    }

    pub(crate) fn set_memory(&mut self, block: Weak<SealedBlock>) {
        self.set_local(ReplicaCandidate {
            medium: ReplicaMedium::Dram,
            evidence: Evidence::Memory(block),
        });
    }

    pub(crate) fn set_ssd(&mut self, candidate: SsdReadCandidate) {
        self.set_local(ReplicaCandidate::ssd(candidate));
    }

    fn set_local(&mut self, candidate: ReplicaCandidate) {
        self.replicas
            .retain(|existing| existing.medium != candidate.medium || existing.is_peer());
        self.replicas.push(candidate);
    }

    /// Retain only peer media with an implemented source authorization path.
    /// Unknown and HBM evidence never becomes executable from TE availability.
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn set_peers(&mut self, replicas: Vec<ReplicaLocation>) {
        self.replicas.retain(|replica| !replica.is_peer());
        for location in replicas.into_iter().filter(|location| {
            matches!(
                location.metadata.medium,
                ReplicaMedium::Dram | ReplicaMedium::Ssd
            )
        }) {
            if self.peer(location.metadata.medium).count() < DISCOVERY_MAX_REPLICAS_PER_MEDIUM
                && !self
                    .peer(location.metadata.medium)
                    .any(|peer| peer.owner == location.owner)
            {
                self.replicas.push(ReplicaCandidate {
                    medium: location.metadata.medium,
                    evidence: Evidence::Peer(location),
                });
            }
        }
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn set_remote(
        &mut self,
        coverage: DiscoveryCoverage,
        replicas: Vec<ReplicaLocation>,
    ) {
        self.remote_coverage = Some(coverage);
        self.set_peers(replicas);
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn remote_coverage(&self) -> Option<DiscoveryCoverage> {
        self.remote_coverage
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn peer(&self, medium: ReplicaMedium) -> impl Iterator<Item = &ReplicaLocation> {
        self.replicas
            .iter()
            .filter_map(move |candidate| candidate.peer(medium))
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn reject_peer(&mut self, owner: &orbitkv_state::CacheOwner) {
        self.replicas.retain(|replica| match &replica.evidence {
            Evidence::Peer(peer) => &peer.owner != owner,
            _ => true,
        });
    }

    pub(crate) fn local_ssd(&self) -> Option<&SsdReadCandidate> {
        self.replicas.iter().find_map(ReplicaCandidate::local_ssd)
    }

    pub(crate) fn is_available(&self) -> bool {
        self.replicas.iter().any(ReplicaCandidate::is_available)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/planning/replica.rs"]
mod tests;
