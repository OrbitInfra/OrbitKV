pub(crate) mod dram;
pub(crate) mod inventory;
pub(crate) mod publish;
pub(crate) mod ssd;

use bytesize::ByteSize;
use log::{debug, info};
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use self::ssd::SsdStore;
use crate::EngineConfig;
use crate::block::{SealedBlock, StateKey};
use crate::memory::AllocateFn;
use crate::memory::AllocationFootprintFn;
use crate::memory::numa::NumaNode;
use crate::memory::pool::{PinnedAllocation, PinnedAllocator};
use crate::metrics::core_metrics;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
use crate::peer::{provider::PeerTransport, read::PeerReader};
use orbitkv_catalog::GlobalIndex;

use crate::query::read::ReadCoordinator;
use dram::DramStore;
use publish::PublishQueue;

pub(crate) type MaterializedBlocks = Vec<(StateKey, Arc<SealedBlock>)>;

const RECLAIM_BATCH_SIZE: usize = 512;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCacheCleanupStats {
    pub evicted_blocks: usize,
    pub evicted_bytes: u64,
    pub reclaimed_bytes: u64,
    pub still_referenced_blocks: u64,
}

pub(crate) struct Storage {
    pub(crate) allocator: Arc<PinnedAllocator>,
    pub(crate) codec: crate::StorageCodec,
    pub(crate) codec_budget: usize,
    pub(crate) dram: Arc<DramStore>,
    pub(crate) reads: ReadCoordinator,
    pub(crate) writes: PublishQueue,
    pub(crate) ssd_store: Option<Arc<SsdStore>>,
    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    peer_transport: Option<Arc<PeerTransport>>,
    blockwise_alloc: bool,
    pub(crate) global_index: Option<Arc<GlobalIndex>>,
    pub(crate) inventory: Option<Arc<inventory::ResidencyInventory>>,
    pub(crate) exports: crate::peer::export::PeerExports,
}

impl Storage {
    pub(crate) fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: EngineConfig,
        numa_nodes: &[NumaNode],
    ) -> Result<Arc<Self>, String> {
        if config.codec_budget < 4096 || config.codec_budget > u32::MAX as usize {
            return Err("codec budget must be between 4 KiB and 4 GiB - 1".into());
        }
        if config.codec == crate::StorageCodec::Ans {
            crate::codec::ans::Library::load()?;
        }
        if config.cache_protected_percent > 100 {
            return Err("cache protected percent must be between 0 and 100".into());
        }
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));
        let ssd_cache_config = config.ssd_cache_config;
        #[cfg(any(feature = "mooncake", feature = "orbitte"))]
        let mooncake_nic_names = config.mooncake_nic_names;
        let blockwise_alloc = config.blockwise_alloc || ssd_cache_config.is_some();
        let transfer_lock_timeout = config.transfer_lock_timeout;
        let transfer_budget = config.transfer_budget_bytes.unwrap_or(capacity_bytes / 2);
        if transfer_budget == 0
            || transfer_budget > capacity_bytes
            || transfer_budget > i64::MAX as usize
        {
            return Err(
                "transfer budget must be positive and no larger than the pinned pool or i64::MAX"
                    .into(),
            );
        }

        if blockwise_alloc {
            info!("Blockwise allocation enabled for batch_save");
        }

        let pool_shards = config.pool_shards;

        // Create unified allocator based on NUMA configuration
        let allocator = if !numa_nodes.is_empty() {
            info!(
                "Creating NUMA-aware pinned pools for {} nodes",
                numa_nodes.len()
            );
            Arc::new(PinnedAllocator::new_numa(
                capacity_bytes,
                numa_nodes,
                pool_shards,
                use_hugepages,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                pool_shards,
                use_hugepages,
                unit_hint,
            ))
        };

        // Sub-components
        if config.membership.is_some() != config.global_index.is_some()
            || config.membership.is_some() != config.inventory.is_some()
        {
            return Err(
                "distributed cache requires membership, global index and inventory together".into(),
            );
        }
        let inventory = config.inventory.clone();
        let dram = Arc::new(DramStore::with_inventory(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
            inventory.clone(),
            (capacity_bytes as u128 * config.cache_protected_percent as u128 / 100) as u64,
        ));

        let global_index = config.global_index.clone();

        // Mooncake must be created after the allocator so it can register the
        // pinned pool. An empty rail filter lets Mooncake choose TCP fallback.
        #[cfg(any(feature = "mooncake", feature = "orbitte"))]
        let peer_transport = if global_index.is_some() {
            let advertise = &config
                .membership
                .as_ref()
                .expect("distributed configuration")
                .owner()
                .endpoint;
            let transfer = PeerTransport::new(
                config.peer_transfer_backend,
                &mooncake_nic_names,
                allocator.clone(),
                advertise,
                #[cfg(feature = "orbitte-rdma")]
                config.orbitte_rdma.clone(),
            )
            .map(Arc::new)
            .map_err(|error| format!("Failed to initialise peer transfer engine: {error}"))?;
            Some(transfer)
        } else {
            None
        };

        #[cfg(not(any(feature = "mooncake", feature = "orbitte")))]
        if global_index.is_some() {
            log::warn!(
                "Catalog was configured, but this binary was built without the `mooncake` feature; remote transfer is disabled"
            );
        }

        let is_numa = allocator.is_numa();
        let storage_ref = Arc::new(std::sync::OnceLock::<Weak<Self>>::new());
        let allocation_owner = storage_ref.clone();
        let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
            allocation_owner
                .get()?
                .upgrade()?
                .allocate(NonZeroU64::new(size)?, numa_node)
        });
        let footprint_owner = allocator.clone();
        let allocation_footprint_fn: AllocationFootprintFn = Arc::new(move |size, numa_node| {
            footprint_owner.allocation_footprint(
                NonZeroU64::new(size)?,
                numa_node.unwrap_or(NumaNode::UNKNOWN),
            )
        });
        let ssd_store = ssd_cache_config
            .map(|cfg| {
                SsdStore::new(
                    cfg,
                    allocate_fn.clone(),
                    allocation_footprint_fn.clone(),
                    is_numa,
                    inventory.clone(),
                )
                .map_err(|error| format!("Failed to initialise SSD cache: {error}"))
            })
            .transpose()?;
        let engine = Arc::new({
            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            let remote_fetch = peer_transport.as_ref().and_then(|transfer| {
                let ms = global_index.as_ref()?;
                Some(Arc::new(PeerReader::new(
                    Arc::clone(ms),
                    Arc::clone(transfer),
                    allocate_fn.clone(),
                    config
                        .membership
                        .clone()
                        .expect("distributed configuration"),
                )))
            });

            let reads = ReadCoordinator::new(
                dram.clone(),
                ssd_store.clone(),
                #[cfg(any(feature = "mooncake", feature = "orbitte"))]
                remote_fetch,
                config.codec_budget,
            );

            #[cfg(any(feature = "mooncake", feature = "orbitte"))]
            let endpoint = peer_transport
                .as_ref()
                .map(|transport| transport.transfer_endpoint().to_owned());
            #[cfg(not(any(feature = "mooncake", feature = "orbitte")))]
            let endpoint = None;
            let exports = crate::peer::export::PeerExports::new(
                dram.clone(),
                ssd_store.clone(),
                config.membership.clone(),
                endpoint,
                #[cfg(feature = "orbitte")]
                peer_transport.clone(),
                transfer_lock_timeout,
                transfer_budget as u64,
            );

            Self {
                allocator,
                codec: config.codec,
                codec_budget: config.codec_budget,
                dram: dram.clone(),
                reads,
                writes: PublishQueue::spawn(dram.clone(), ssd_store.clone())
                    .map_err(|error| error.to_string())?,
                ssd_store,
                #[cfg(any(feature = "mooncake", feature = "orbitte"))]
                peer_transport,
                blockwise_alloc,
                global_index,
                inventory,
                exports,
            }
        });

        let _ = storage_ref.set(Arc::downgrade(&engine));

        Ok(engine)
    }

    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.ssd_store.is_some()
    }

    pub(crate) fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    /// Returns true if blockwise allocation is enabled.
    pub(crate) fn blockwise_alloc(&self) -> bool {
        self.blockwise_alloc
    }

    /// Allocate pinned memory, returning `None` if the pool is exhausted after eviction.
    pub(crate) fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        let requested_bytes = size.get();
        let node = numa_node.unwrap_or(NumaNode::UNKNOWN);

        loop {
            if let Some(alloc) = self.allocator.allocate(size, node) {
                return Some(alloc);
            }

            let (freed_blocks, _freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes, node);

            if freed_blocks == 0 && largest_free < requested_bytes {
                // Final retry: absorb concurrent frees that may have raced with reclaim probing.
                if let Some(alloc) = self.allocator.allocate(size, node) {
                    return Some(alloc);
                }
                break;
            }
        }

        let (used, total) = self.allocator.usage();
        log::error!(
            "Pinned memory pool exhausted; cannot satisfy allocation: \
             requested={} used={} total={} numa={:?}",
            ByteSize(requested_bytes),
            ByteSize(used),
            ByteSize(total),
            numa_node
        );
        core_metrics().pool_alloc_failures.add(1, &[]);
        None
    }

    /// Flush the SSD writer: waits until all enqueued writes are committed.
    pub(crate) async fn flush_ssd(&self) {
        if let Some(ssd) = &self.ssd_store {
            ssd.flush().await;
        }
    }

    pub(crate) fn filter_hashes_not_in_cache_inplace(
        &self,
        namespace: &str,
        hashes: &mut HashSet<Vec<u8>>,
    ) {
        let namespace = namespace.to_string();
        let hash_vec: Vec<Vec<u8>> = hashes.iter().cloned().collect();
        let keys: Vec<StateKey> = hash_vec
            .iter()
            .map(|hash| StateKey::new(namespace.clone(), hash.clone()))
            .collect();
        let present = self.dram.contains_keys(&keys);
        for (hash, is_present) in hash_vec.into_iter().zip(present) {
            if is_present {
                hashes.remove(&hash);
            }
        }
    }

    /// Evict all blocks from the resident in-memory read cache.
    ///
    /// This preserves backing-store copies. Blocks with
    /// outstanding references may remain allocated until those holders release
    /// the last `Arc`.
    pub(crate) fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        let used_before = self.allocator.usage().0;
        let evicted = self.dram.remove_all();
        if evicted.is_empty() {
            return MemoryCacheCleanupStats::default();
        }

        let mut evicted_bytes = 0u64;
        let mut still_referenced_blocks = 0u64;
        for (_key, block) in &evicted {
            let bytes = block.memory_footprint();
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            if Arc::strong_count(block) > 1 {
                still_referenced_blocks += 1;
            }
            core_metrics()
                .cache_resident_bytes
                .add(-(bytes as i64), &[]);
        }

        let evicted_blocks = evicted.len();
        if still_referenced_blocks > 0 {
            core_metrics()
                .cache_block_evictions_still_referenced
                .add(still_referenced_blocks, &[]);
        }
        core_metrics()
            .cache_block_evictions
            .add(evicted_blocks as u64, &[]);

        drop(evicted);
        let reclaimed_bytes = used_before.saturating_sub(self.allocator.usage().0);
        if reclaimed_bytes > 0 {
            core_metrics()
                .cache_eviction_reclaimed_bytes
                .add(reclaimed_bytes, &[]);
        }

        info!(
            "Cleaned resident memory cache: evicted_blocks={} evicted_bytes={} reclaimed_bytes={} still_referenced_blocks={}",
            evicted_blocks,
            ByteSize(evicted_bytes),
            ByteSize(reclaimed_bytes),
            still_referenced_blocks
        );

        MemoryCacheCleanupStats {
            evicted_blocks,
            evicted_bytes,
            reclaimed_bytes,
            still_referenced_blocks,
        }
    }

    fn reclaim_until_allocator_can_allocate(
        &self,
        required_bytes: u64,
        target_node: NumaNode,
    ) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (
                0,
                0,
                self.allocator.largest_free_allocation_for_node(target_node),
            );
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.allocator.largest_free_allocation_for_node(target_node);

        while largest_free < required_bytes {
            let used_before = self.allocator.usage().0;

            let evicted = self
                .dram
                .remove_lru_batch(RECLAIM_BATCH_SIZE, required_bytes);

            if evicted.is_empty() {
                break;
            }

            let mut batch_bytes = 0u64;
            for (_key, block) in &evicted {
                let b = block.memory_footprint();
                batch_bytes = batch_bytes.saturating_add(b);
                core_metrics().cache_resident_bytes.add(-(b as i64), &[]);
            }

            freed_bytes = freed_bytes.saturating_add(batch_bytes);
            freed_blocks += evicted.len();

            drop(evicted);
            let used_after = self.allocator.usage().0;
            let reclaimed = used_before.saturating_sub(used_after);
            if reclaimed > 0 {
                core_metrics()
                    .cache_eviction_reclaimed_bytes
                    .add(reclaimed, &[]);
            }

            largest_free = self.allocator.largest_free_allocation_for_node(target_node);
        }

        if freed_blocks > 0 {
            debug!(
                "Reclaimed cache blocks toward allocator request: \
                 freed_blocks={} freed_bytes={} largest_free={} required={}",
                freed_blocks,
                ByteSize(freed_bytes),
                ByteSize(largest_free),
                ByteSize(required_bytes)
            );
            core_metrics()
                .cache_block_evictions
                .add(freed_blocks as u64, &[]);
        }

        (freed_blocks, freed_bytes, largest_free)
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn peer_transport(&self) -> Option<&Arc<PeerTransport>> {
        self.peer_transport.as_ref()
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn transfer_endpoint(&self) -> Option<&str> {
        self.peer_transport
            .as_ref()
            .map(|transport| transport.transfer_endpoint())
    }

    #[cfg(any(feature = "mooncake", feature = "orbitte"))]
    pub(crate) fn transfer_source_epoch(&self) -> Option<&str> {
        self.peer_transport
            .as_ref()
            .map(|transport| transport.source_epoch())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/storage/mod.rs"]
mod tests;
