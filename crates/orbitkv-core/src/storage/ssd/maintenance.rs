use super::index::WatermarkStep;
use super::{SsdStore, persistence};
use crate::metrics::core_metrics;

/// One bounded SSD FIFO maintenance pass. Bytes refer to logical cache capacity,
/// including alignment and wrap gaps, not disk space returned to the filesystem.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SsdCacheCleanupStats {
    pub scanned_entries: usize,
    pub evicted_blocks: usize,
    pub evicted_bytes: u64,
    pub reclaimed_bytes: u64,
    pub blocked_shards: usize,
}

impl SsdStore {
    pub(crate) fn maintain_cache(&self) -> std::io::Result<SsdCacheCleanupStats> {
        let Some((low, high)) = self.watermarks else {
            return Ok(SsdCacheCleanupStats::default());
        };
        let Some(mut cursor) = self.watermark_sweep.try_lock() else {
            return Ok(SsdCacheCleanupStats::default());
        };
        let count = self.inner.lock().ring.shard_count();
        let mut finished = vec![false; count];
        let mut remaining = count;
        let mut stats = SsdCacheCleanupStats::default();
        while remaining != 0 && stats.scanned_entries < super::super::RECLAIM_BATCH_SIZE {
            let shard_id = *cursor;
            *cursor = (*cursor + 1) % count;
            if finished[shard_id] {
                continue;
            }
            stats.scanned_entries += 1;
            // Lock order is ring then journal. Neither write completion nor TTL
            // invalidation holds the journal while acquiring the ring. Holding
            // the ring across this one durable invalidation fences new pins.
            let mut inner = self.inner.lock();
            let reclaimed = match inner.ring.watermark_step(shard_id, low, high) {
                WatermarkStep::Idle => {
                    finished[shard_id] = true;
                    remaining -= 1;
                    0
                }
                WatermarkStep::Blocked => {
                    finished[shard_id] = true;
                    remaining -= 1;
                    stats.blocked_shards += 1;
                    core_metrics().ssd_watermark_blocked.add(1, &[]);
                    0
                }
                WatermarkStep::Pruned(bytes) => bytes,
                WatermarkStep::Evict(key, entry) => {
                    if let Some(journal) = &self.journal {
                        let mut journal = journal.lock();
                        if let Err(error) = journal.invalidate(&key, &entry) {
                            journal.fence();
                            return Err(error);
                        }
                        persistence::fault("ssd_watermark_invalidated");
                    }
                    let bytes = inner.ring.retire_watermark(&key, &entry).ok_or_else(|| {
                        std::io::Error::other(
                            "SSD watermark candidate changed under its owner lock",
                        )
                    })?;
                    self.retire_inventory([&key]);
                    stats.evicted_blocks += 1;
                    stats.evicted_bytes += entry.len;
                    core_metrics().ssd_watermark_evictions.add(1, &[]);
                    bytes
                }
            };
            stats.reclaimed_bytes += reclaimed;
            core_metrics()
                .ssd_watermark_reclaimed_bytes
                .add(reclaimed, &[]);
        }
        Ok(stats)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/storage/ssd/maintenance.rs"]
mod tests;
