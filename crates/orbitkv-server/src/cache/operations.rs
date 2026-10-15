use std::sync::{Arc, Mutex};

use super::read::ReadControl;
use crate::metric::hll::MultiWindowHllTracker;
use orbitkv_core::QueryLeaseId;
use orbitkv_core::{
    EngineError, LayerSave, OrbitKVEngine, PublishDiagnostic, QueryMode, QueryOwner,
    QueryReservation,
};
use thiserror::Error;

fn trace_query(stage: &'static str, input: &QueryInput, elapsed_us: u64, hit_blocks: usize) {
    orbitkv_common::timeline::record_query_path(
        stage,
        &input.request_id,
        &input.instance_id,
        input.group_id,
        input.warmup,
        input.prepare,
        elapsed_us,
        hit_blocks,
    );
}

#[derive(Clone, Debug)]
pub(crate) struct QueryInput {
    pub instance_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub request_id: String,
    pub wait_for_full_prefix: bool,
    pub group_id: u32,
    pub warmup: bool,
    pub discover: bool,
    pub materialize: bool,
    pub prepare: bool,
    pub demand: Option<orbitkv_state::RecoveryDemand>,
    pub control: Arc<ReadControl>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum QueryOutcome {
    Busy,
    Loading,
    Candidates {
        hit_positions: Vec<u32>,
    },
    Ready {
        num_hit_blocks: u64,
        lease: Vec<u8>,
        hit_positions: Vec<u32>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PublishLayerInput {
    pub layer_name: String,
    pub block_ids: Vec<u32>,
    pub block_hashes: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub(crate) struct PublishInput {
    pub instance_id: String,
    pub registration_generation: [u8; 16],
    pub tp_rank: u32,
    pub pp_rank: u32,
    pub device_id: i32,
    pub layers: Vec<PublishLayerInput>,
    pub diagnostic: Option<PublishDiagnostic>,
}

#[derive(Clone, Debug)]
pub(crate) struct RestoreLeaseInput {
    pub lease: Vec<u8>,
    pub block_ids_by_group: Vec<Vec<Option<u32>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct RestoreInput {
    pub instance_id: String,
    pub tp_rank: u32,
    pub device_id: i32,
    pub layer_groups: Vec<Vec<String>>,
    pub loads: Vec<RestoreLeaseInput>,
}

pub(crate) fn execute_restore(
    engine: &OrbitKVEngine,
    input: RestoreInput,
) -> Result<orbitkv_core::RestoreExecution, EngineError> {
    if input.device_id < 0 {
        return Err(EngineError::InvalidArgument(format!(
            "device_id {} must be >= 0",
            input.device_id
        )));
    }
    let tp_rank = usize::try_from(input.tp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("tp_rank {} does not fit usize", input.tp_rank))
    })?;
    let loads = input
        .loads
        .into_iter()
        .map(|load| {
            let lease =
                QueryLeaseId::from_bytes(&load.lease).map_err(EngineError::InvalidArgument)?;
            let groups = load
                .block_ids_by_group
                .into_iter()
                .map(|targets| {
                    targets
                        .into_iter()
                        .map(|target| target.map(|id| id as usize))
                        .collect()
                })
                .collect();
            Ok((lease, groups))
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let layer_groups = input
        .layer_groups
        .iter()
        .map(|group| group.iter().map(String::as_str).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    engine.restore(
        &input.instance_id,
        tp_rank,
        input.device_id,
        &layer_groups,
        &loads,
    )
}

pub(crate) async fn execute_publish(
    engine: &OrbitKVEngine,
    input: PublishInput,
) -> Result<(), EngineError> {
    if input.device_id < 0 {
        return Err(EngineError::InvalidArgument(format!(
            "device_id {} must be >= 0",
            input.device_id
        )));
    }
    let tp_rank = usize::try_from(input.tp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("tp_rank {} does not fit usize", input.tp_rank))
    })?;
    let pp_rank = usize::try_from(input.pp_rank).map_err(|_| {
        EngineError::InvalidArgument(format!("pp_rank {} does not fit usize", input.pp_rank))
    })?;
    let mut saves = Vec::with_capacity(input.layers.len());
    for layer in input.layers {
        if layer.block_ids.len() != layer.block_hashes.len() {
            return Err(EngineError::InvalidArgument(format!(
                "block_ids length {} does not match block_hashes {} for layer {}",
                layer.block_ids.len(),
                layer.block_hashes.len(),
                layer.layer_name
            )));
        }
        saves.push(LayerSave {
            layer_name: layer.layer_name,
            block_ids: layer.block_ids.into_iter().map(|id| id as usize).collect(),
            block_hashes: layer.block_hashes,
        });
    }
    match input.diagnostic {
        Some(diagnostic) => {
            engine
                .batch_save_kv_blocks_from_ipc_diagnostic(
                    &input.instance_id,
                    input.registration_generation,
                    tp_rank,
                    pp_rank,
                    input.device_id,
                    saves,
                    diagnostic,
                )
                .await
        }
        None => {
            engine
                .batch_save_kv_blocks_from_ipc(
                    &input.instance_id,
                    input.registration_generation,
                    tp_rank,
                    pp_rank,
                    input.device_id,
                    saves,
                )
                .await
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum ReleaseError {
    #[error("{0}")]
    InvalidLease(String),
    #[error("query lease is unknown or expired")]
    UnknownOrExpired,
}

pub(crate) fn execute_release(engine: &OrbitKVEngine, lease: &[u8]) -> Result<(), ReleaseError> {
    let lease = QueryLeaseId::from_bytes(lease).map_err(ReleaseError::InvalidLease)?;
    if engine.release_query_lease(&lease) {
        Ok(())
    } else {
        Err(ReleaseError::UnknownOrExpired)
    }
}

pub(crate) async fn execute_query(
    engine: &OrbitKVEngine,
    hll_tracker: &Arc<Mutex<MultiWindowHllTracker>>,
    input: QueryInput,
    reservation: Option<QueryReservation>,
    owner: QueryOwner,
) -> Result<QueryOutcome, EngineError> {
    let started = std::time::Instant::now();
    if input.discover {
        let hit_positions = engine
            .discover_candidates(&input.instance_id, input.group_id, &input.block_hashes)
            .await?;
        trace_query(
            "discovery_ready",
            &input,
            started.elapsed().as_micros() as u64,
            hit_positions.len(),
        );
        if input.group_id == 0 {
            record_prefix_reuse(
                engine,
                hll_tracker,
                &input.instance_id,
                &input.block_hashes,
                hit_positions.len(),
            );
        }
        return Ok(QueryOutcome::Candidates { hit_positions });
    }
    let reservation = reservation.ok_or_else(|| {
        EngineError::InvalidArgument("payload read requires a reservation".into())
    })?;
    trace_query("read_start", &input, 0, 0);
    if input.request_id.is_empty() {
        return Err(EngineError::InvalidArgument(
            "request_id must not be empty".to_string(),
        ));
    }

    let batch_blocks = if input.wait_for_full_prefix {
        input.block_hashes.len().max(1)
    } else {
        reservation.batch_blocks(input.block_hashes.len(), input.control.batch_bytes)
    };
    let mut blocks = Vec::new();
    let mut hit_positions = Vec::new();
    let mut complete_evidence = true;
    for (batch, hashes) in input.block_hashes.chunks(batch_blocks).enumerate() {
        if !input.control.can_submit(batch) {
            complete_evidence = false;
            trace_query(
                "read_stopped",
                &input,
                started.elapsed().as_micros() as u64,
                blocks.len(),
            );
            break;
        }
        // The awaited batch owns its buffers even if the receiver, deadline or
        // selected revision disappears. Only the next submission is stopped.
        if input.group_id == 0 {
            let status = engine
                .count_prefix_hit_blocks_with_prefetch(
                    &input.instance_id,
                    &input.request_id,
                    hashes,
                    if input.prepare {
                        QueryMode::Prepare
                    } else if input.warmup {
                        QueryMode::Warmup
                    } else if input.wait_for_full_prefix {
                        QueryMode::WaitForFullPrefix
                    } else {
                        QueryMode::Demand
                    },
                )
                .await?;
            let complete = status.blocks.len() == hashes.len();
            blocks.extend(status.blocks);
            if !complete {
                break;
            }
        } else if input.wait_for_full_prefix {
            let status = engine
                .query_group_membership_with_fetch(
                    &input.instance_id,
                    &input.request_id,
                    input.group_id,
                    hashes,
                )
                .await?;
            hit_positions.extend(0..status.blocks.len() as u32);
            blocks.extend(status.blocks);
        } else {
            let hits = engine
                .query_group_membership(
                    &input.instance_id,
                    &input.request_id,
                    input.group_id,
                    hashes,
                    if input.prepare {
                        QueryMode::Prepare
                    } else if input.warmup {
                        QueryMode::Warmup
                    } else {
                        QueryMode::Demand
                    },
                )
                .await?;
            for (position, block) in hits.into_iter().enumerate() {
                if let Some(block) = block {
                    hit_positions.push((batch * batch_blocks + position) as u32);
                    blocks.push(block);
                }
            }
        }
    }
    trace_query(
        "source_ready",
        &input,
        started.elapsed().as_micros() as u64,
        blocks.len(),
    );
    if input.warmup {
        return Ok(QueryOutcome::Ready {
            num_hit_blocks: 0,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        });
    }
    let hit = blocks.len();
    // A selected boundary needs every page in this group's declared range.
    // Drop incomplete sources and their reservation before creating any lease.
    if input.demand.is_some() && (!complete_evidence || hit != input.block_hashes.len()) {
        return Ok(QueryOutcome::Ready {
            num_hit_blocks: 0,
            lease: Vec::new(),
            hit_positions: Vec::new(),
        });
    }
    if input.group_id == 0 && !input.materialize && !input.prepare && complete_evidence {
        record_prefix_reuse(
            engine,
            hll_tracker,
            &input.instance_id,
            &input.block_hashes,
            hit,
        );
    }
    let lease = if hit == 0
        || (input.group_id > 0 && input.wait_for_full_prefix && hit != input.block_hashes.len())
    {
        Vec::new()
    } else {
        engine
            .finish_query(reservation, owner, blocks)?
            .to_bytes()
            .to_vec()
    };
    Ok(QueryOutcome::Ready {
        num_hit_blocks: hit as u64,
        lease,
        hit_positions,
    })
}

pub(crate) fn record_prefix_reuse(
    engine: &OrbitKVEngine,
    hll_tracker: &Mutex<MultiWindowHllTracker>,
    instance_id: &str,
    hashes: &[Vec<u8>],
    hits: usize,
) {
    if let Ok(namespace) = engine.instance_namespace(instance_id)
        && let Ok(mut tracker) = hll_tracker.lock()
    {
        tracker.record_namespaced_misses(&namespace, hashes.len() as u64, &hashes[hits..]);
    }
}
