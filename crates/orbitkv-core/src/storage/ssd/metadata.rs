// ============================================================================
// Seal Offload Types
//
// Shared types for sealed block metadata, used by SSD cache.
// ============================================================================

use smallvec::SmallVec;

use crate::memory::numa::NumaNode;

/// Per-slot metadata (one slot = one layer's KV cache).
///
/// Layout-agnostic: uses per-segment sizes instead of `is_split` boolean.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotMeta {
    /// Per-segment sizes. SmallVec inlines up to 2 elements on the stack
    /// (covers K-only MLA and K+V layouts; spills to heap for 3+ segments).
    pub segment_sizes: SmallVec<[u64; 2]>,
    /// Pre-computed total size across all segments (cached for consistency with RawBlock).
    pub total_size: u64,
    /// NUMA node affinity for this slot's GPU.
    pub numa_node: NumaNode,
    pub(crate) encoding: Option<Vec<crate::codec::EncodedSegment>>,
}

impl SlotMeta {
    /// Create a new SlotMeta, caching total_size from segment_sizes.
    pub fn new(segment_sizes: SmallVec<[u64; 2]>, numa_node: NumaNode) -> Self {
        let total_size = segment_sizes.iter().sum();
        Self {
            segment_sizes,
            total_size,
            numa_node,
            encoding: None,
        }
    }

    /// Total size across all segments.
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segment_sizes.len()
    }
}
