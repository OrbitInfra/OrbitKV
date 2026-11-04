//! Per-layer KV cache layout.
//!
//! Every layout OrbitKV supports (dense layer-first, fused-buffer strided,
//! MLA single-segment, K/V split) is a parameterization of one affine formula:
//! `addr = data_ptr + block_idx * block_stride + segment_idx * kv_stride`.
//!
//! Geometry validates pointer-free byte ranges once. A local binding then
//! checks that those ranges fit its GPU allocation before exposing copies.
//! Padded sizes govern pinned-memory strides and SSD iovecs; GPU copies always
//! use actual (unpadded) sizes.

use std::ops::Range;

/// How a block's segments are addressed on the GPU.
#[derive(Debug, Clone, Copy)]
enum SegmentLayout {
    /// The whole block is one contiguous range: a single segment, or K/V
    /// adjacent with no gap (`kv_stride == segment_bytes`).
    Contiguous,
    /// K and V live in separate regions, `kv_stride_bytes` apart: two copies
    /// per block.
    Split { kv_stride_bytes: usize },
}

/// One contiguous device address range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockCopy {
    pub addr: u64,
    pub bytes: usize,
}

/// Device address ranges making up one block on the GPU.
pub(crate) enum BlockCopies {
    /// One contiguous copy.
    Contiguous(BlockCopy),
    /// Two copies: K and V segments in separate regions.
    Split { k: BlockCopy, v: BlockCopy },
}

/// Validated layout geometry, independent of a process's GPU virtual addresses.
#[derive(Debug, Clone)]
pub(crate) struct KVCacheGeometry {
    /// Exclusive end of the highest addressed byte range.
    extent_bytes: usize,
    /// Number of blocks in this layer's cache.
    num_blocks: usize,
    /// Byte step between consecutive blocks (per segment region for split
    /// layouts). Determined from the final registration, including fused buffers.
    block_stride_bytes: usize,
    /// GPU-side segment size in bytes (one of K or V).
    segment_bytes: usize,
    /// Segments per block (1 contiguous/MLA, 2 for K/V).
    segments: usize,
    /// CPU/SSD-side segment stride, `segment_bytes` rounded up to SSD
    /// alignment. Equals `segment_bytes` when SSD is disabled.
    padded_segment_bytes: usize,
    seg: SegmentLayout,
}

impl KVCacheGeometry {
    /// Validate the final block stride, segment separation and host padding.
    /// `segment_bytes` is ONE segment (K or V for split layouts).
    /// `None` selects dense block spacing; alignment 1 disables host padding.
    pub(crate) fn new(
        num_blocks: usize,
        segment_bytes: usize,
        kv_stride_bytes: usize,
        segments: usize,
        block_stride: Option<usize>,
        host_alignment: usize,
    ) -> Result<Self, String> {
        if segment_bytes == 0 || num_blocks == 0 || segments == 0 {
            return Err("segment_bytes, num_blocks, and segments must be non-zero".into());
        }
        if host_alignment == 0 {
            return Err("host alignment must be > 0".into());
        }
        let block_bytes = segment_bytes
            .checked_mul(segments)
            .ok_or_else(|| "block size overflow".to_string())?;

        let seg = if segments == 1 {
            SegmentLayout::Contiguous
        } else if kv_stride_bytes == 0 {
            return Err("kv_stride_bytes must be > 0 when segments > 1".into());
        } else if kv_stride_bytes < segment_bytes {
            return Err(format!(
                "kv_stride_bytes {kv_stride_bytes} < segment_bytes {segment_bytes}: segments would overlap"
            ));
        } else if kv_stride_bytes == segment_bytes {
            SegmentLayout::Contiguous
        } else if segments == 2 {
            SegmentLayout::Split { kv_stride_bytes }
        } else {
            return Err(format!(
                "split layout (kv_stride_bytes {kv_stride_bytes} > segment_bytes {segment_bytes}) supports exactly 2 segments, got {segments}"
            ));
        };

        let padded_segment_bytes = segment_bytes
            .checked_next_multiple_of(host_alignment)
            .ok_or_else(|| "padded segment size overflow".to_string())?;
        padded_segment_bytes
            .checked_mul(segments)
            .ok_or_else(|| "padded block size overflow".to_string())?;
        let mut geometry = Self {
            extent_bytes: 0,
            num_blocks,
            block_stride_bytes: block_stride.unwrap_or(match seg {
                SegmentLayout::Contiguous => block_bytes,
                SegmentLayout::Split { .. } => segment_bytes,
            }),
            segment_bytes,
            segments,
            padded_segment_bytes,
            seg,
        };
        geometry.extent_bytes = geometry.validate_extent()?;
        Ok(geometry)
    }

    /// Prove that all offsets are representable and block ranges never alias.
    fn validate_extent(&self) -> Result<usize, String> {
        let end = match self.seg {
            SegmentLayout::Contiguous => {
                let block_bytes = self.block_bytes();
                if self.block_stride_bytes < block_bytes {
                    return Err(format!(
                        "block_stride {} must be >= block_bytes {block_bytes}: blocks would overlap",
                        self.block_stride_bytes
                    ));
                }
                (self.num_blocks - 1)
                    .checked_mul(self.block_stride_bytes)
                    .and_then(|o| o.checked_add(block_bytes))
                    .ok_or_else(|| "memory layout overflow".to_string())?
            }
            SegmentLayout::Split { kv_stride_bytes } => {
                if self.block_stride_bytes < self.segment_bytes {
                    return Err(format!(
                        "block_stride {} must be >= segment_bytes {}: blocks would overlap",
                        self.block_stride_bytes, self.segment_bytes
                    ));
                }
                self.check_split_segments_disjoint(kv_stride_bytes)?;
                let last_segment_end = (self.num_blocks - 1)
                    .checked_mul(self.block_stride_bytes)
                    .and_then(|o| o.checked_add(self.segment_bytes))
                    .ok_or_else(|| "memory layout overflow".to_string())?;
                kv_stride_bytes
                    .checked_add(last_segment_end)
                    .ok_or_else(|| "memory layout overflow".to_string())?
            }
        };
        Ok(end)
    }

    fn check_split_segments_disjoint(&self, kv_stride_bytes: usize) -> Result<(), String> {
        let max_block_distance = self.num_blocks - 1;
        let nearest = kv_stride_bytes / self.block_stride_bytes;

        for distance in [nearest, nearest.saturating_add(1)] {
            if distance == 0 || distance > max_block_distance {
                continue;
            }
            let block_distance_bytes = distance
                .checked_mul(self.block_stride_bytes)
                .ok_or_else(|| "memory layout overflow".to_string())?;
            let gap = kv_stride_bytes.abs_diff(block_distance_bytes);
            if gap < self.segment_bytes {
                return Err(format!(
                    "kv_stride_bytes {kv_stride_bytes} overlaps blocks {distance} apart"
                ));
            }
        }

        Ok(())
    }

    /// Pointer-free ranges for one block. Construction proved the last block's
    /// extent, so valid indices cannot overflow the arithmetic below.
    pub(crate) fn block_ranges(&self, block_idx: usize) -> Result<BlockRanges, String> {
        if block_idx >= self.num_blocks {
            return Err(format!(
                "block {block_idx} out of range ({} blocks)",
                self.num_blocks
            ));
        }
        let base = block_idx * self.block_stride_bytes;
        Ok(match self.seg {
            SegmentLayout::Contiguous => BlockRanges::Contiguous(base..base + self.block_bytes()),
            SegmentLayout::Split { kv_stride_bytes } => BlockRanges::Split {
                k: base..base + self.segment_bytes,
                v: base + kv_stride_bytes..base + kv_stride_bytes + self.segment_bytes,
            },
        })
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// True when K and V need separate pinned segment pools.
    pub(crate) fn is_split(&self) -> bool {
        matches!(self.seg, SegmentLayout::Split { .. })
    }

    /// Actual (unpadded) GPU-side segment size.
    pub(crate) fn segment_bytes(&self) -> usize {
        self.segment_bytes
    }

    /// Actual (unpadded) total block size on the GPU.
    fn block_bytes(&self) -> usize {
        self.segment_bytes * self.segments
    }

    /// Host-side per-segment stride (SSD-aligned).
    pub(crate) fn padded_segment_bytes(&self) -> usize {
        self.padded_segment_bytes
    }

    /// Host-side total block size: pinned allocation footprint and
    /// `RawBlock.total_size` → `SlotMeta.total_size()` for SSD I/O.
    pub(crate) fn padded_block_bytes(&self) -> usize {
        self.padded_segment_bytes * self.segments
    }
}

/// Validated byte ranges relative to an allocation's local base address.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BlockRanges {
    Contiguous(Range<usize>),
    Split { k: Range<usize>, v: Range<usize> },
}

/// One process's GPU address binding for validated cache geometry.
#[derive(Clone)]
pub(crate) struct KVCacheLayout {
    pub(crate) storage_format: orbitkv_state::StorageFormat,
    data_ptr: u64,
    geometry: KVCacheGeometry,
    pub(crate) owner: Option<std::sync::Arc<dyn Send + Sync>>,
}

impl KVCacheLayout {
    pub(crate) fn bind(
        data_ptr: u64,
        size_bytes: usize,
        geometry: KVCacheGeometry,
    ) -> Result<Self, String> {
        if data_ptr == 0 {
            return Err("data_ptr must not be null".into());
        }
        if size_bytes == 0 {
            return Err("size_bytes must be > 0".into());
        }
        data_ptr
            .checked_add(size_bytes as u64)
            .ok_or_else(|| "data_ptr + size_bytes overflows the address space".to_string())?;
        if geometry.extent_bytes > size_bytes {
            return Err(format!(
                "registered memory too small: need {} bytes, got {size_bytes}",
                geometry.extent_bytes
            ));
        }
        Ok(Self {
            storage_format: Default::default(),
            data_ptr,
            geometry,
            owner: None,
        })
    }

    pub(crate) fn geometry(&self) -> &KVCacheGeometry {
        &self.geometry
    }

    /// Bind already validated relative ranges to this process's GPU addresses.
    pub(crate) fn block_copies(&self, block_idx: usize) -> Result<BlockCopies, String> {
        let copy = |range: Range<usize>| BlockCopy {
            addr: self.data_ptr + range.start as u64,
            bytes: range.end - range.start,
        };
        Ok(match self.geometry.block_ranges(block_idx)? {
            BlockRanges::Contiguous(range) => BlockCopies::Contiguous(copy(range)),
            BlockRanges::Split { k, v } => BlockCopies::Split {
                k: copy(k),
                v: copy(v),
            },
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transfer/layout.rs"]
mod tests;

impl std::fmt::Debug for KVCacheLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCacheLayout")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}
