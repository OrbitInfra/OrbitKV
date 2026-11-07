//! Fetch backing blocks in the caller-owned query future.

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;
use std::time::Instant;
use tokio::sync::OnceCell;

use futures::{StreamExt, stream};
use log::warn;
use parking_lot::Mutex;

use crate::QueryMode;
use crate::block::{QueryResult, RestoreSource, SealedBlock, StateKey};
use crate::memory::numa::NumaNode;
use crate::metrics::core_metrics;
use crate::peer::read::PeerReader;
use crate::storage::{MaterializedBlocks, ssd::SsdStore};
use crate::transfer_engine::TransferOptions;

use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};
use crate::planning::read::{HostReadRoute, ReadPlan};
use crate::storage::dram::DramStore;

const REMOTE_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REMOTE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct MaterializedRead {
    source: Option<AttributionSource>,
    cache_inserts: MaterializedBlocks,
    ready_blocks: Vec<RestoreSource>,
    missing: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ReadKey {
    keys: Vec<StateKey>,
    hit: usize,
    wait_for_full_prefix: bool,
    allow_ssd_prefetch: bool,
    destination_nodes: Vec<NumaNode>,
    host_ready: bool,
    deadline: Instant,
}
type SharedRead = OnceCell<MaterializedRead>;

pub(crate) struct ReadCoordinator {
    dram: Arc<DramStore>,
    reads: Mutex<HashMap<ReadKey, Weak<SharedRead>>>,
    ssd_store: Option<Arc<SsdStore>>,
    remote_fetch: Option<Arc<PeerReader>>,
    codec_budget: usize,
}

impl ReadCoordinator {
    pub(crate) fn new(
        dram: Arc<DramStore>,
        ssd_store: Option<Arc<SsdStore>>,
        remote_fetch: Option<Arc<PeerReader>>,
        codec_budget: usize,
    ) -> Self {
        Self {
            dram,
            codec_budget,
            reads: Mutex::new(HashMap::new()),
            ssd_store,
            remote_fetch,
        }
    }

    pub(crate) async fn read_prefix(
        &self,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        mode: QueryMode,
        destination_nodes: &[NumaNode],
        deadline: Instant,
    ) -> QueryResult {
        let options = TransferOptions::read(mode, deadline);
        let warming = matches!(mode, QueryMode::Warmup | QueryMode::Prepare);
        let wait_for_full_prefix = mode == QueryMode::WaitForFullPrefix;
        let keys: Vec<StateKey> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let (hit, prefix_blocks) = self.dram.get_prefix_blocks(&keys, warming);
        let has_remote = self.remote_fetch.is_some();
        if hit == keys.len() || (!has_remote && self.ssd_store.is_none()) {
            if !warming && let Some(ssd) = &self.ssd_store {
                ssd.ingest_batch(keys.iter().zip(&prefix_blocks), true);
            }
            record_tier_attribution(keys.len(), hit, 0, None);
            return QueryResult {
                blocks: prefix_blocks
                    .into_iter()
                    .map(RestoreSource::Memory)
                    .collect(),
                missing: keys.len() - hit,
            };
        }

        let mut plan = ReadPlan::new(&keys[hit..], mode, self.ssd_store.as_ref());
        if let Some(ssd) = &self.ssd_store
            && let Some(route) = plan.deferred_ssd(ssd, self.codec_budget)
        {
            let path = route.path;
            let allow_uring_fallback = route.allow_uring_fallback;
            if let Some(leases) = route.acquire(self.codec_budget, options) {
                let count = hit + leases.len();
                ssd.ingest_batch(keys.iter().zip(&prefix_blocks), true);
                record_tier_attribution(
                    keys.len(),
                    hit,
                    leases.len(),
                    Some(AttributionSource::Ssd),
                );
                return QueryResult {
                    blocks: prefix_blocks
                        .into_iter()
                        .map(RestoreSource::Memory)
                        .chain(leases.into_iter().map(|lease| RestoreSource::Ssd {
                            lease,
                            path,
                            allow_uring_fallback,
                        }))
                        .collect(),
                    missing: keys.len() - count,
                };
            }
        }

        let allow_ssd_prefetch = warming
            || self
                .ssd_store
                .as_ref()
                .is_none_or(|ssd| ssd.read_path.is_none());
        let key = ReadKey {
            keys: keys.clone(),
            hit,
            wait_for_full_prefix,
            allow_ssd_prefetch,
            destination_nodes: destination_nodes.to_vec(),
            host_ready: warming,
            deadline,
        };
        let read = {
            let mut reads = self.reads.lock();
            if let Some(read) = reads.get(&key).and_then(Weak::upgrade) {
                core_metrics().query_coalesced_reads.add(1, &[]);
                read
            } else {
                reads.retain(|_, read| read.strong_count() > 0);
                let read = Arc::new(SharedRead::new());
                reads.insert(key, Arc::downgrade(&read));
                read
            }
        };
        let result = read
            .get_or_init(|| async {
                let (source, blocks) = self
                    .materialize(
                        &mut plan,
                        req_id,
                        allow_ssd_prefetch,
                        destination_nodes,
                        options,
                    )
                    .await;
                let mut result =
                    build_ready_result(prefix_blocks, keys.len(), source, &keys[hit..], blocks);
                let inserts = std::mem::take(&mut result.cache_inserts);
                if warming {
                    for (_, block) in &inserts {
                        block.mark_warmed();
                    }
                }
                if warming || result.source == Some(AttributionSource::Remote) {
                    self.dram.batch_insert_reclaimable(inserts);
                } else {
                    self.dram.batch_insert(inserts);
                }
                result
            })
            .await;
        if !warming {
            let (resident_keys, resident): (Vec<_>, Vec<_>) = keys
                .iter()
                .zip(&result.ready_blocks)
                .filter_map(|(key, source)| match source {
                    RestoreSource::Memory(block) => Some((key.clone(), Arc::clone(block))),
                    _ => None,
                })
                .unzip();
            self.dram.retain_demand(&resident_keys, &resident);
            if let Some(ssd) = &self.ssd_store {
                ssd.ingest_batch(resident_keys.iter().zip(&resident), true);
            }
        }
        record_tier_attribution(
            keys.len(),
            hit,
            result.ready_blocks.len() - hit,
            result.source,
        );
        QueryResult {
            blocks: result.ready_blocks.clone(),
            missing: result.missing,
        }
    }

    /// Position-aligned membership across resident and backing tiers: entry
    /// `i` is the sealed block for `hashes[i]`, or `None` on miss. Hashes must
    /// already carry any group encoding (see `group_hash`).
    pub(crate) async fn read_membership(
        &self,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        mode: crate::QueryMode,
        destination_nodes: &[NumaNode],
        deadline: Instant,
    ) -> Vec<Option<crate::RestoreSource>> {
        let keys: Vec<StateKey> = hashes
            .iter()
            .map(|hash| StateKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let resident = self.dram.get_blocks_aligned(&keys);
        // Auxiliary state can have holes (checkpoints or evicted windows).
        // Bound independent reads and reuse prefix fetch coalescing/cancellation
        // without waiting for absent checkpoints to be published.
        stream::iter(
            hashes
                .iter()
                .cloned()
                .zip(resident)
                .map(|(hash, block)| async move {
                    if block.is_some() {
                        return block.map(crate::RestoreSource::Memory);
                    }
                    self.read_prefix(
                        req_id,
                        namespace,
                        std::slice::from_ref(&hash),
                        mode,
                        destination_nodes,
                        deadline,
                    )
                    .await
                    .blocks
                    .pop()
                }),
        )
        .buffered(8)
        .collect()
        .await
    }

    async fn materialize(
        &self,
        plan: &mut ReadPlan,
        req_id: &str,
        allow_ssd_prefetch: bool,
        destination_nodes: &[NumaNode],
        options: TransferOptions,
    ) -> (Option<AttributionSource>, Vec<(StateKey, RestoreSource)>) {
        if let Some(remote) = &self.remote_fetch {
            remote.discover(&mut plan.rows);
        }
        let peer_available = self.remote_fetch.is_some();

        let target = plan.target;
        let mut allow_local_ssd = allow_ssd_prefetch;
        for _ in 0..3 {
            match plan.host_route(peer_available, allow_local_ssd, self.codec_budget) {
                Some(HostReadRoute::Peer(route)) => {
                    if let Some(remote) = &self.remote_fetch {
                        let result =
                            fetch_peer(remote, route, target, req_id, destination_nodes, options)
                                .await;
                        if !result.can_replan() {
                            return (Some(AttributionSource::Remote), result.blocks);
                        }
                        // No payload was submitted. Replan from the retained
                        // local/peer evidence after rejected source admission.
                        continue;
                    }
                }
                Some(HostReadRoute::Ssd(route)) => {
                    if let Some(ssd) = &self.ssd_store
                        && let Some(leases) = route.acquire(self.codec_budget, options)
                    {
                        return (
                            Some(AttributionSource::Ssd),
                            ssd.read_host_batch(leases)
                                .await
                                .unwrap_or_default()
                                .into_iter()
                                .map(|(key, block)| (key, RestoreSource::Memory(block)))
                                .collect(),
                        );
                    }
                    // This exact local generation disappeared after planning.
                    // Do not select it again; peer alternatives remain eligible.
                    allow_local_ssd = false;
                    continue;
                }
                None => break,
            }
        }

        if plan.wait_for_full_prefix
            && let Some(remote) = &self.remote_fetch
        {
            let started_at = Instant::now();
            while Instant::now() < options.deadline.min(started_at + REMOTE_WAIT_TIMEOUT) {
                tokio::time::sleep(REMOTE_WAIT_POLL_INTERVAL).await;
                remote.discover(&mut plan.rows);
                if let Some(HostReadRoute::Peer(route)) =
                    plan.host_route(true, false, self.codec_budget)
                {
                    // A submitted payload failure completes this query; only
                    // missing advertisements participate in producer waiting.
                    let result =
                        fetch_peer(remote, route, target, req_id, destination_nodes, options).await;
                    if !result.can_replan() {
                        return (Some(AttributionSource::Remote), result.blocks);
                    }
                }
            }
            warn!(
                "Timed out waiting for remote prefix: req_id={req_id} timeout_secs={}",
                REMOTE_WAIT_TIMEOUT.as_secs()
            );
        }
        (None, Vec::new())
    }
}

/// Attribute each query once, including any backing fetch.
fn record_tier_attribution(
    total: usize,
    hit: usize,
    loading: usize,
    loading_source: Option<AttributionSource>,
) {
    if total == 0 {
        return;
    }
    let attribution = TierAttribution::classify(total, hit, loading, loading_source);
    record_cache_tier_block_requests(total, attribution);
}

fn build_ready_result(
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    source: Option<AttributionSource>,
    requested_keys: &[StateKey],
    fetched: Vec<(StateKey, RestoreSource)>,
) -> MaterializedRead {
    let mut ready_blocks: Vec<_> = prefix_blocks
        .into_iter()
        .map(RestoreSource::Memory)
        .collect();
    let inserts_by_key: HashMap<_, _> = fetched.iter().map(|(key, block)| (key, block)).collect();
    ready_blocks.extend(
        requested_keys
            .iter()
            .map_while(|key| inserts_by_key.get(key).map(|block| (*block).clone())),
    );
    let missing = total.saturating_sub(ready_blocks.len());
    let cache_inserts = fetched
        .into_iter()
        .filter_map(|(key, source)| match source {
            RestoreSource::Memory(block) => Some((key, block)),
            _ => None,
        })
        .collect();
    MaterializedRead {
        source,
        cache_inserts,
        ready_blocks,
        missing,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/query/read.rs"]
mod tests;

async fn fetch_peer(
    remote: &PeerReader,
    route: crate::planning::peer::FetchPlan<'_>,
    target: crate::planning::read::ReadTarget,
    req_id: &str,
    nodes: &[NumaNode],
    options: TransferOptions,
) -> crate::peer::execute::FetchResult<RestoreSource> {
    use crate::peer::execute::FetchResult;
    if target == crate::planning::read::ReadTarget::EngineRestore {
        let result = remote.restore_plan(route, req_id, nodes, options).await;
        FetchResult {
            blocks: result
                .blocks
                .into_iter()
                .map(|(key, block)| (key, RestoreSource::Peer(block)))
                .collect(),
            status: result.status,
            attempts: result.attempts,
            completed_segments: result.completed_segments,
        }
    } else {
        let result = remote.fetch_plan(route, req_id, nodes, options).await;
        FetchResult {
            blocks: result
                .blocks
                .into_iter()
                .map(|(key, block)| (key, RestoreSource::Memory(block)))
                .collect(),
            status: result.status,
            attempts: result.attempts,
            completed_segments: result.completed_segments,
        }
    }
}
