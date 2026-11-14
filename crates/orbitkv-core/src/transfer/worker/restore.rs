use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::block::{SealedBlock, StateKey};
use crate::planning::restore::RestorePlan;
use crate::storage::MaterializedBlocks;
use crate::{EngineError, SsdReadPath};

use super::{LayerTransferData, TransferPayload};

fn ssd_path(layers: &[LayerTransferData]) -> Result<Option<SsdReadPath>, String> {
    let mut selected = None;
    for block in layers.iter().flat_map(|layer| &layer.blocks) {
        if let TransferPayload::Ssd { path, .. } = block.block {
            if selected.is_some_and(|selected| selected != path) {
                return Err("one restore task cannot mix SSD read routes".into());
            }
            selected = Some(path);
        }
    }
    Ok(selected)
}

pub(super) fn validate_plan(
    plan: &RestorePlan,
    layers: &[LayerTransferData],
) -> Result<(), EngineError> {
    let path = ssd_path(layers).map_err(EngineError::Storage)?;
    if path != plan.ssd_path() {
        return Err(EngineError::InvalidArgument(
            "restore plan source path differs from transfer payloads".into(),
        ));
    }
    Ok(())
}

pub(super) fn set_ssd_path(layers: &mut [LayerTransferData], path: SsdReadPath) {
    for block in layers.iter_mut().flat_map(|layer| &mut layer.blocks) {
        if let TransferPayload::Ssd { path: selected, .. } = &mut block.block {
            *selected = path;
        }
    }
}

/// Runs only on the independent SSD host lane. The existing bounded reader
/// owns submitted I/O; the task keeps engine mappings through final GPU completion.
pub(super) fn materialize_host(layers: &mut [LayerTransferData]) -> Result<(), EngineError> {
    super::ssd::validate_host_sources(layers)?;
    let mut sources = Vec::new();
    let mut seen = HashSet::new();
    for block in layers.iter().flat_map(|layer| &layer.blocks) {
        if let TransferPayload::Ssd { source, path, .. } = &block.block {
            if *path != SsdReadPath::Uring {
                return Err(EngineError::Storage(
                    "cuFile source reached the SSD host lane".into(),
                ));
            }
            if seen.insert(Arc::as_ptr(&source.entry.readers) as usize) {
                sources.push(Arc::clone(source));
            }
        }
    }
    let batches = host_read_batches(
        sources
            .iter()
            .map(|source| (Arc::as_ptr(&source.store) as usize, &source.key)),
    );
    let reads = batches.into_iter().map(|indices| {
        let sources: Vec<_> = indices
            .into_iter()
            .map(|index| Arc::clone(&sources[index]))
            .collect();
        async move {
            let blocks = sources[0].store.read_host_batch(sources.clone()).await?;
            bind_host_results(
                sources
                    .iter()
                    .map(|source| (Arc::as_ptr(&source.entry.readers) as usize, &source.key)),
                blocks,
            )
        }
    });
    // Drain every batch even when another batch fails. No GPU source is replaced
    // until every requested immutable generation has a validated host result.
    let results = futures::executor::block_on(futures::future::join_all(reads));
    let blocks: HashMap<_, _> = results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    for block in layers.iter_mut().flat_map(|layer| &mut layer.blocks) {
        if let TransferPayload::Ssd {
            source,
            slot_id,
            offset,
            ..
        } = &block.block
        {
            let sealed = blocks
                .get(&(Arc::as_ptr(&source.entry.readers) as usize))
                .ok_or_else(|| EngineError::Storage("SSD host result is missing".into()))?;
            block.block = TransferPayload::Cached {
                sealed: Arc::clone(sealed),
                slot_id: *slot_id,
                offset: *offset,
            };
        }
    }
    Ok(())
}

const MAX_HOST_READ_BATCH_BLOCKS: usize = 16;

fn host_read_batches<'a>(
    sources: impl IntoIterator<Item = (usize, &'a StateKey)>,
) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut open: HashMap<usize, (usize, HashSet<&StateKey>)> = HashMap::new();
    for (index, (store, key)) in sources.into_iter().enumerate() {
        let (batch, keys) = open.entry(store).or_insert_with(|| {
            batches.push(Vec::new());
            (batches.len() - 1, HashSet::new())
        });
        // Results are keyed by semantic state, while the caller owns physical
        // generations. Keep different generations of the same key in separate
        // batches so that each result has exactly one leased source.
        if batches[*batch].len() == MAX_HOST_READ_BATCH_BLOCKS || !keys.insert(key) {
            batches.push(Vec::new());
            *batch = batches.len() - 1;
            keys.clear();
            keys.insert(key);
        }
        batches[*batch].push(index);
    }
    batches
}

fn bind_host_results<'a>(
    sources: impl IntoIterator<Item = (usize, &'a StateKey)>,
    blocks: MaterializedBlocks,
) -> Result<Vec<(usize, Arc<SealedBlock>)>, EngineError> {
    let count = blocks.len();
    let mut by_key: HashMap<_, _> = blocks.into_iter().collect();
    if by_key.len() != count {
        return Err(EngineError::Storage("duplicate SSD host result".into()));
    }
    let results = sources
        .into_iter()
        .map(|(identity, key)| {
            by_key
                .remove(key)
                .map(|block| (identity, block))
                .ok_or_else(|| EngineError::Storage("SSD host result is missing".into()))
        })
        .collect::<Result<_, _>>()?;
    if !by_key.is_empty() {
        return Err(EngineError::Storage("unexpected SSD host result".into()));
    }
    Ok(results)
}

#[cfg(test)]
#[path = "../../../tests/unit/transfer/worker/restore.rs"]
mod tests;
