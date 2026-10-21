use orbitkv_state::group_hash;

use super::{EngineError, OrbitKVEngine};
use crate::block::{QueryResult, RestoreSource};
use crate::metrics::core_metrics;
use crate::query::lease::QueryLeaseId;
use crate::query::{QueryAdmission, QueryMode, QueryOwner, QueryReservation};

impl OrbitKVEngine {
    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Argument contract:
    /// - `instance_id` must identify a registered instance.
    /// - `req_id` must be non-empty; the Cache Manager validates it.
    /// - `block_hashes` may be empty.
    ///
    /// Returns:
    /// A terminal `QueryResult` with the ready prefix and missing suffix.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "query_prefetch.count_prefix_hit")
    )]
    pub async fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
        mode: QueryMode,
    ) -> Result<QueryResult, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, 0))
            .collect();

        let status = self
            .storage
            .reads
            .read_prefix(
                req_id,
                namespace,
                &encoded,
                mode,
                topology.group_slot_numa(0)?,
            )
            .await;

        {
            let QueryResult { blocks, missing } = &status;
            let metrics = core_metrics();
            metrics.cache_block_hits.add(blocks.len() as u64, &[]);
            if *missing > 0 {
                metrics.cache_block_misses.add(*missing as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Find candidate positions without loading or pinning their payloads.
    pub async fn discover_candidates(
        &self,
        instance_id: &str,
        group_id: u32,
        hashes: &[Vec<u8>],
    ) -> Result<Vec<u32>, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        topology.group_total_slots(group_id)?;
        let mut positions = Vec::new();
        let mut unknown = 0usize;
        let requested = hashes.len();
        'discovery: for (batch, hashes) in
            hashes.chunks(orbitkv_state::DISCOVERY_MAX_KEYS).enumerate()
        {
            let encoded: Vec<_> = hashes
                .iter()
                .map(|hash| group_hash(hash, group_id))
                .collect();
            let candidates = crate::planning::discover::discover(
                &self.storage.dram,
                self.storage.ssd_store.as_ref(),
                self.storage.global_index.as_ref(),
                &topology.cache_namespace,
                &encoded,
            );
            for (position, candidate) in candidates.into_iter().enumerate() {
                debug_assert_eq!(candidate.key.hash, encoded[position]);
                if candidate.is_available() {
                    positions.push((batch * orbitkv_state::DISCOVERY_MAX_KEYS + position) as u32);
                } else {
                    #[cfg(feature = "peer")]
                    if candidate.remote_coverage().is_some_and(|coverage| {
                        coverage != orbitkv_state::DiscoveryCoverage::CompleteAtWatermarks
                    }) {
                        unknown = if group_id == 0 {
                            requested - positions.len()
                        } else {
                            unknown + 1
                        };
                    }
                    if group_id == 0 {
                        break 'discovery;
                    }
                }
            }
        }
        let metrics = core_metrics();
        metrics
            .cache_candidate_hits
            .add(positions.len() as u64, &[]);
        metrics
            .cache_candidate_misses
            .add((requested - positions.len() - unknown) as u64, &[]);
        metrics.cache_candidate_unknown.add(unknown as u64, &[]);
        Ok(positions)
    }

    /// All-or-nothing membership fetch over one hybrid-cache storage group,
    /// eligible for the same SSD prefetch and Catalog + Mooncake remote fetch
    /// as prefix queries.
    ///
    /// Where [`Self::query_group_membership`] allows sparse hits, this treats
    /// `block_hashes` as an exact want-set: misses are
    /// pulled from remote tiers, and the future waits until the whole
    /// set is fetchable (the prefix machinery over an explicit key list *is*
    /// a set fetch once the full length is required). Use it when partial
    /// state is useless — e.g. restoring a recurrent-state checkpoint on a
    /// prefill/decode handoff, where the peer that saved the set holds every
    /// member. A result with fewer blocks than requested means the
    /// set could not be completed anywhere; callers treat that as a miss.
    pub async fn query_group_membership_with_fetch(
        &self,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
        block_hashes: &[Vec<u8>],
    ) -> Result<QueryResult, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        // Same contract as the local membership query: an unknown group is a
        // bug in the caller, not an all-miss answer.
        topology.group_total_slots(group_id)?;

        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, group_id))
            .collect();

        let status = self
            .storage
            .reads
            .read_prefix(
                req_id,
                namespace,
                &encoded,
                QueryMode::WaitForFullPrefix,
                topology.group_slot_numa(group_id)?,
            )
            .await;

        {
            let QueryResult { blocks, missing } = &status;
            let metrics = core_metrics();
            metrics.cache_block_hits.add(blocks.len() as u64, &[]);
            if *missing > 0 {
                metrics.cache_block_misses.add(*missing as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Position-aligned membership query over one hybrid-cache storage group.
    ///
    /// Unlike prefix queries, every position reports independently: entry `i`
    /// is the sealed block for `block_hashes[i]` in `group_id`, or `None` on
    /// miss. Sparse hit patterns are the point — callers (e.g. the vLLM
    /// connector's hybrid reconcile) pick the rightmost hit themselves.
    /// Every group uses the same versioned content-hash encoding.
    ///
    /// The returned sources own memory references or pinned disk extents.
    /// Transfer this ownership into a scheduler lease via [`Self::create_query_lease`].
    pub async fn query_group_membership(
        &self,
        instance_id: &str,
        req_id: &str,
        group_id: u32,
        block_hashes: &[Vec<u8>],
        mode: QueryMode,
    ) -> Result<Vec<Option<RestoreSource>>, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        // Validate the group against the sealed topology; an unknown group is
        // a contract bug, not an answer of "all miss".
        topology.group_total_slots(group_id)?;

        let namespace = &topology.cache_namespace;
        let encoded: Vec<Vec<u8>> = block_hashes
            .iter()
            .map(|hash| group_hash(hash, group_id))
            .collect();
        Ok(self
            .storage
            .reads
            .read_membership(
                req_id,
                namespace,
                &encoded,
                mode,
                topology.group_slot_numa(group_id)?,
            )
            .await)
    }

    /// Create an opaque lease that owns query-ready blocks.
    pub fn create_query_lease(
        &self,
        instance_id: &str,
        blocks: Vec<RestoreSource>,
    ) -> Result<QueryLeaseId, EngineError> {
        let instance = self.get_instance(instance_id)?;
        if blocks.is_empty() {
            return Err(EngineError::InvalidArgument(
                "query lease requires at least one block".to_string(),
            ));
        }
        Ok(self
            .query_leases
            .create(instance_id, blocks, instance.world_size(), None))
    }

    /// Check the complete declared boundary before admitting any selected group.
    /// Storage identity comes from this instance's sealed registration.
    pub fn validate_recovery_demand(
        &self,
        instance_id: &str,
        demand: &orbitkv_state::RecoveryDemand,
        group_id: u32,
        hashes: usize,
    ) -> Result<(), EngineError> {
        demand
            .validate(group_id, hashes)
            .map_err(|error| EngineError::InvalidArgument(error.to_string()))?;
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        if !demand
            .groups
            .iter()
            .map(|(group, _)| *group as usize)
            .eq(0..topology.num_groups())
        {
            return Err(EngineError::InvalidArgument(
                "recovery demand must include every registered storage group exactly once".into(),
            ));
        }
        Ok(())
    }

    /// Reserve registered group bytes before a process query retains any pages.
    pub fn reserve_query(
        &self,
        instance_id: &str,
        group_id: u32,
        blocks: usize,
        mode: QueryMode,
    ) -> Result<QueryAdmission, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let bytes = topology
            .group_block_bytes(group_id)?
            .checked_mul(blocks as u64)
            .ok_or_else(|| EngineError::InvalidArgument("query bytes overflow".into()))?;
        Ok(
            match self
                .query_budget
                .reserve(instance_id, &topology.cache_namespace, bytes, mode)
            {
                QueryAdmission::Admitted(reservation) => {
                    QueryAdmission::Admitted(reservation.bind_registration(instance.generation()))
                }
                admission => admission,
            },
        )
    }

    /// Move preparation ownership into the result lease and then GPU consumers.
    pub fn finish_query(
        &self,
        reservation: QueryReservation,
        owner: QueryOwner,
        blocks: Vec<RestoreSource>,
    ) -> Result<QueryLeaseId, EngineError> {
        let instance_id = reservation.instance();
        let instances = self
            .instances
            .read()
            .map_err(|_| EngineError::Poisoned("instances"))?;
        let instance = instances
            .get(instance_id)
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.into()))?;
        if reservation.registration_generation() != Some(instance.generation())
            || instance.sealed_topology()?.cache_namespace != reservation.namespace()
        {
            return Err(EngineError::InvalidArgument(
                "query instance registration changed".into(),
            ));
        }
        let bytes = blocks
            .iter()
            .try_fold(0u64, |sum, block| sum.checked_add(block.memory_footprint()))
            .ok_or_else(|| EngineError::InvalidArgument("query result bytes overflow".into()))?;
        reservation
            .ready(bytes)
            .map_err(EngineError::InvalidArgument)?;
        Ok(self.query_leases.create(
            instance_id,
            blocks,
            instance.world_size(),
            Some((owner, reservation.clone())),
        ))
    }

    pub fn release_query_session(&self, session: u64) {
        self.query_leases
            .release_owner(|owner| owner.session == session);
    }

    /// Move a prepared lease into foreground ownership without releasing bytes.
    pub fn claim_query(&self, lease: &QueryLeaseId) {
        self.query_leases.claim(lease);
    }

    pub fn cancel_query(&self, owner: QueryOwner) {
        self.query_leases
            .release_owner(|candidate| candidate == owner);
    }

    /// Release a query lease. Returns false when the lease is unknown or expired.
    pub fn release_query_lease(&self, lease: &QueryLeaseId) -> bool {
        self.query_leases.release(lease)
    }
}
