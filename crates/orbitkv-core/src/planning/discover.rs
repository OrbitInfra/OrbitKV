use std::sync::Arc;

use super::replica::ReplicaSet;
use crate::block::StateKey;
use crate::storage::{dram::DramStore, ssd::SsdStore};
use orbitkv_catalog::GlobalIndex;

pub(crate) fn discover(
    dram: &DramStore,
    ssd: Option<&Arc<SsdStore>>,
    index: Option<&Arc<GlobalIndex>>,
    namespace: &str,
    hashes: &[Vec<u8>],
) -> Vec<ReplicaSet> {
    #[cfg(not(feature = "peer"))]
    let _ = index;
    let keys: Vec<_> = hashes
        .iter()
        .map(|hash| StateKey::new(namespace.to_owned(), hash.clone()))
        .collect();
    let mut candidates: Vec<_> = keys
        .iter()
        .cloned()
        .zip(dram.discover(&keys))
        .map(|(key, dram)| {
            let mut candidates = ReplicaSet::new(key);
            if let Some(dram) = dram {
                candidates.set_memory(dram);
            }
            candidates
        })
        .collect();
    if let Some(ssd) = ssd {
        for (candidate, backing) in candidates.iter_mut().zip(ssd.discover(&keys)) {
            if let Some(backing) = backing {
                candidate.set_ssd(backing);
            }
        }
    }
    #[cfg(feature = "peer")]
    if let Some(index) = index {
        for (candidate, remote) in candidates.iter_mut().zip(index.lookup(&keys)) {
            candidate.set_remote(remote.coverage, remote.replicas);
        }
    }
    candidates
}
