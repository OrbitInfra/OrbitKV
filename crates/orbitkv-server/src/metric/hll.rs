//! HyperLogLog-based cache hit rate estimation.
//!
//! Provides a minimal HyperLogLog implementation and a sliding-window tracker
//! for estimating the theoretical maximum cache hit rate over a time window.
//!
//! Hash inputs must be at least 4 bytes and should have good uniformity
//! (e.g. SHA-256, xxHash). Longer hashes give more leading-zero headroom.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Allowed range for `bucket_bits` (register index width).
pub(crate) const MIN_BUCKET_BITS: u8 = 4;
pub(crate) const MAX_BUCKET_BITS: u8 = 18;

// ============================================================================
// HyperLogLog core
// ============================================================================

/// Minimal HyperLogLog for cardinality estimation.
///
/// Input hashes are expected to have good uniformity (e.g. SHA-256).
/// The full hash is used: top `bucket_bits` bits select the register,
/// remaining bits are scanned for leading zeros.
#[derive(Debug)]
pub(crate) struct HyperLogLog {
    registers: Vec<u8>,
    bucket_bits: u8,
    /// Mask that zeros out the top `bucket_bits` bits in a big-endian u32
    /// read from hash\[0..4\]: `(1u32 << (32 - bucket_bits)) - 1`.
    lz_mask: u32,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with the given bucket bits.
    ///
    /// `bucket_bits` determines the number of buckets (2^bucket_bits) and estimation
    /// accuracy (~1.04 / sqrt(2^bucket_bits)). 16 gives 65536 buckets
    /// and ~0.4% standard error.
    pub(crate) fn new(bucket_bits: u8) -> Self {
        assert!(
            (MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits),
            "HLL bucket_bits must be in {MIN_BUCKET_BITS}..={MAX_BUCKET_BITS}, got {bucket_bits}"
        );
        Self {
            registers: vec![0u8; 1 << bucket_bits],
            bucket_bits,
            lz_mask: (1u32 << (32 - bucket_bits)) - 1,
        }
    }

    /// Insert a block hash.
    ///
    /// Treats the hash as a big-endian bit stream:
    /// - Top `bucket_bits` bits → register index
    /// - Remaining bits → count leading zeros (ρ)
    ///
    /// Shorter hashes are implicitly zero-padded; longer hashes give
    /// more leading-zero headroom and better accuracy.
    pub(crate) fn insert(&mut self, hash: &[u8]) {
        let index = bucket_index(hash, self.bucket_bits);
        let rho = count_leading_zeros(hash, self.bucket_bits, self.lz_mask) + 1;

        let reg = &mut self.registers[index];
        if rho > *reg {
            *reg = rho;
        }
    }

    /// Merge another HLL into this one (element-wise max of registers).
    pub(crate) fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(
            self.bucket_bits, other.bucket_bits,
            "cannot merge HLLs with different bucket_bits"
        );
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *b > *a {
                *a = *b;
            }
        }
    }

    /// Reset all registers to zero.
    pub(crate) fn clear(&mut self) {
        self.registers.fill(0);
    }
}

// ============================================================================
// Free helpers
// ============================================================================

/// Read byte at `index`, returning 0 for out-of-bounds positions.
fn byte_or_zero(hash: &[u8], index: usize) -> u8 {
    hash.get(index).copied().unwrap_or(0)
}

/// Top `bucket_bits` bits of the hash as a register index (bucket_bits ≤ 18 → 3 bytes suffice).
fn bucket_index(hash: &[u8], bucket_bits: u8) -> usize {
    let val = u32::from_be_bytes([
        0,
        byte_or_zero(hash, 0),
        byte_or_zero(hash, 1),
        byte_or_zero(hash, 2),
    ]);
    (val >> (24 - bucket_bits as u32)) as usize
}

/// Leading zeros in the remaining bits after the bucket index.
fn count_leading_zeros(hash: &[u8], bucket_bits: u8, lz_mask: u32) -> u8 {
    let head = u32::from_be_bytes([
        byte_or_zero(hash, 0),
        byte_or_zero(hash, 1),
        byte_or_zero(hash, 2),
        byte_or_zero(hash, 3),
    ]);
    let masked = head & lz_mask;
    if masked != 0 {
        return masked.leading_zeros() as u8 - bucket_bits;
    }
    let mut count = 32 - bucket_bits;
    for &byte in hash.get(4..).unwrap_or(&[]) {
        if byte != 0 {
            return count + byte.leading_zeros() as u8;
        }
        count += 8;
    }
    count
}

/// Cardinality of the union of two HLLs without mutating either.
fn union_cardinality(a: &HyperLogLog, b: &HyperLogLog) -> f64 {
    let merged: Vec<u8> = a
        .registers
        .iter()
        .zip(b.registers.iter())
        .map(|(&x, &y)| x.max(y))
        .collect();
    estimate_cardinality(&merged)
}

/// Standard HLL estimation with small-range correction (linear counting).
///
/// Large-range correction is omitted: with SHA-256 inputs (242+ remaining bits),
/// the threshold (~2^242) is unreachable in practice. This follows HyperLogLog++
/// (Google, 2013) which also dropped the large-range correction.
fn estimate_cardinality(registers: &[u8]) -> f64 {
    let m = registers.len() as f64;
    let alpha = alpha_m(registers.len());

    // E = α_m × m² / Σ 2^(-M[j])
    //   - 2^(-M[j]) = 1/2^M[j], the reciprocal for harmonic mean
    //   - m / Σ 2^(-M[j]) = harmonic mean of per-bucket estimates 2^M[j]
    //   - × m to scale from per-bucket to total cardinality
    //   - × α_m to correct systematic upward bias
    let sum: f64 = registers.iter().map(|&r| 2.0f64.powi(-(r as i32))).sum();
    let raw_estimate = alpha * m * m / sum;

    if raw_estimate <= 2.5 * m {
        // Small range correction: when cardinality << m, many registers are still 0
        // and the harmonic mean formula loses accuracy. Switch to Linear Counting:
        //
        //   n ≈ m × ln(m / V),  where V = number of zero registers
        //
        // Derivation: m buckets, n distinct balls → empty fraction V/m ≈ e^(-n/m),
        // solving gives n = m × ln(m/V).
        //
        // Threshold 2.5*m is empirically determined (Flajolet et al.).
        // When V=0, Linear Counting diverges (ln(∞)), but the cardinality is no
        // longer "small" so the harmonic mean estimate is accurate enough.
        let zeros = registers.iter().filter(|&&r| r == 0).count() as f64;
        if zeros > 0.0 {
            m * (m / zeros).ln()
        } else {
            raw_estimate
        }
    } else {
        raw_estimate
    }
}

/// Bias-correction constant α_m for HLL, where `m` = number of registers = 2^bucket_bits.
fn alpha_m(m: usize) -> f64 {
    match m {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / m as f64),
    }
}

// ============================================================================
// Sliding-window HLL tracker
// ============================================================================

/// Metric snapshot returned by [`HllTracker::metric`].
#[derive(Debug, Clone)]
pub(crate) struct HllMetric {
    /// Estimated number of distinct miss block identities in the window.
    pub(crate) cardinality: f64,
    /// Total block requests (including duplicates) in the window.
    pub(crate) total_requests: u64,
    /// Estimated hit rate assuming infinite cache: `(total - cardinality) / total`.
    pub(crate) estimated_hit_rate: f64,
}

struct WindowSlot {
    hll: HyperLogLog,
    start: Instant,
    request_count: u64,
}

/// Sliding-window HyperLogLog tracker for cache hit rate estimation.
///
/// Divides time into fixed-duration slots and maintains a ring of HLLs.
/// The merged cardinality across all active slots approximates the number
/// of distinct miss blocks recorded in the window. From this we derive:
///
/// ```text
/// hit_rate = (total_requests - cardinality) / total_requests
/// ```
///
/// Thread safety: wrap in `Mutex<HllTracker>` at the call site.
pub(crate) struct HllTracker {
    slots: VecDeque<WindowSlot>,
    /// Merge of all finalized slots (everything except the active back slot).
    /// Incrementally updated on slot rotation; full recompute only after expiry.
    merged: HyperLogLog,
    /// True when expired slots invalidated `merged` (needs recompute from scratch).
    merged_dirty: bool,
    slot_duration: Duration,
    window_duration: Duration,
    bucket_bits: u8,
}

impl HllTracker {
    /// Create a new tracker.
    ///
    /// - `slot_duration`: how long each time slot lasts (e.g. 1 hour)
    /// - `window_duration`: total sliding window (e.g. 24 hours)
    /// - `bucket_bits`: HLL bucket index bits (4..=18, default 16)
    pub(crate) fn new(slot_duration: Duration, window_duration: Duration, bucket_bits: u8) -> Self {
        Self {
            slots: VecDeque::new(),
            merged: HyperLogLog::new(bucket_bits),
            merged_dirty: false,
            slot_duration,
            window_duration,
            bucket_bits,
        }
    }

    /// Record a block hash request.
    ///
    /// Lazily creates/rotates slots. Slot boundaries are aligned to multiples of
    /// `slot_duration` from the first slot, so gaps without requests don't cause
    /// time drift. For example with 1h slots: if the first slot starts at 0:00
    /// Record a batch of distinct identities while accounting for a possibly
    /// larger observation count. The identities are inserted into HLL, while
    /// `total_requests` is used as the denominator. This is used by the
    /// miss-only reference: only cache misses enter HLL, but every queried
    /// block still contributes to the total observation count.
    #[cfg_attr(
        not(test),
        expect(
            clippy::expect_used,
            reason = "exclusive access: an empty deque always gets a slot above and no slot is removed before this borrow"
        )
    )]
    pub(crate) fn record_hashes_with_total<T: AsRef<[u8]>>(
        &mut self,
        hashes: &[T],
        total_requests: u64,
    ) {
        if total_requests == 0 {
            return;
        }

        let now = Instant::now();

        let need_new_slot = match self.slots.back() {
            None => true,
            Some(s) => now.duration_since(s.start) >= self.slot_duration,
        };

        if need_new_slot {
            self.merged_dirty = true;

            // Align to slot boundary: advance from last slot's start by N × slot_duration
            let aligned_start = match self.slots.back() {
                Some(last) => {
                    let elapsed = now.duration_since(last.start);
                    let periods = elapsed.as_nanos() / self.slot_duration.as_nanos();
                    last.start + self.slot_duration * periods as u32
                }
                None => now,
            };
            self.slots.push_back(WindowSlot {
                hll: HyperLogLog::new(self.bucket_bits),
                start: aligned_start,
                request_count: 0,
            });
        }

        let slot = self
            .slots
            .back_mut()
            .expect("active slot inserted before recording");
        for hash in hashes {
            slot.hll.insert(hash.as_ref());
        }
        slot.request_count += total_requests;
    }

    /// Compute and return the current metric snapshot.
    ///
    /// Triggers slot expiry and merged recomputation if needed.
    pub(crate) fn metric(&mut self) -> HllMetric {
        self.expire_old_slots(Instant::now());
        self.ensure_merged();

        // Cardinality = union(merged finalized slots, active back slot)
        let cardinality = match self.slots.back() {
            Some(back) => union_cardinality(&self.merged, &back.hll),
            None => 0.0,
        };
        let total: u64 = self.slots.iter().map(|s| s.request_count).sum();
        let hit_rate = if total > 0 {
            let c = cardinality.min(total as f64);
            (total as f64 - c) / total as f64
        } else {
            0.0
        };

        HllMetric {
            cardinality,
            total_requests: total,
            estimated_hit_rate: hit_rate,
        }
    }

    fn expire_old_slots(&mut self, now: Instant) {
        while let Some(front) = self.slots.front() {
            if now.duration_since(front.start) >= self.window_duration {
                self.slots.pop_front();
                self.merged_dirty = true; // can't un-merge, need full recompute
            } else {
                break;
            }
        }
    }

    /// Recompute `merged` from all finalized slots (all except back).
    /// Triggered by slot rotation or expiry; only runs on `metric()` calls.
    fn ensure_merged(&mut self) {
        if !self.merged_dirty {
            return;
        }
        self.merged.clear();
        let finalized = self.slots.len().saturating_sub(1);
        for slot in self.slots.iter().take(finalized) {
            self.merged.merge(&slot.hll);
        }
        self.merged_dirty = false;
    }
}

// ============================================================================
// Multi-window tracker
// ============================================================================

/// Tracks the same hash stream across multiple sliding windows in parallel.
///
/// Each window is identified by a human-readable label (`"15m"`, `"1h"`, `"1d"`).
/// `record_namespaced_misses` feeds all windows under a single lock; metric collection
/// returns one snapshot per window with the label preserved for Prometheus.
pub(crate) struct MultiWindowHllTracker {
    windows: Vec<(String, HllTracker)>,
}

impl MultiWindowHllTracker {
    /// Build a multi-window tracker. `windows` is a list of `(label, window_duration)`
    /// pairs. Slot duration is derived per-window as `clamp(window / 24, 1min, 1h)`.
    ///
    /// Panics if `windows` is empty, contains a duplicate duration, or any
    /// window is shorter than 1 minute.
    pub(crate) fn new(windows: Vec<(String, Duration)>, bucket_bits: u8) -> Self {
        assert!(
            !windows.is_empty(),
            "MultiWindowHllTracker needs at least one window"
        );
        for (label, win) in &windows {
            assert!(
                *win >= Duration::from_secs(60),
                "window {label} ({win:?}) must be at least 1 minute"
            );
        }
        for i in 0..windows.len() {
            for j in (i + 1)..windows.len() {
                assert_ne!(
                    windows[i].1, windows[j].1,
                    "duplicate window duration: {:?}",
                    windows[i].1
                );
            }
        }

        let trackers = windows
            .into_iter()
            .map(|(label, window)| {
                let slot = derive_slot_duration(window);
                (label, HllTracker::new(slot, window, bucket_bits))
            })
            .collect();

        Self { windows: trackers }
    }

    /// Record all queried blocks in the denominator but insert only the
    /// identities that were misses into HLL. Repeated misses remain deduped by
    /// HLL, preserving the infinite-cache reuse reference without discarding
    /// historical observations.
    pub(crate) fn record_namespaced_misses(
        &mut self,
        namespace: &str,
        total_requests: u64,
        miss_hashes: &[Vec<u8>],
    ) {
        if total_requests == 0 {
            return;
        }
        debug_assert!(miss_hashes.len() as u64 <= total_requests);

        let namespaced_hashes: Vec<[u8; 8]> = miss_hashes
            .iter()
            .map(|hash| namespaced_hash(namespace, hash))
            .collect();
        for (_, tracker) in &mut self.windows {
            tracker.record_hashes_with_total(&namespaced_hashes, total_requests);
        }
    }

    /// Snapshot every window. Returned in insertion order.
    pub(crate) fn metrics(&mut self) -> Vec<(String, HllMetric)> {
        self.windows
            .iter_mut()
            .map(|(label, tracker)| (label.clone(), tracker.metric()))
            .collect()
    }
}

/// Stable identity hash for `(namespace, block_hash)`.
///
/// Cluster aggregation unions raw HLL registers across nodes, so every node
/// must map the same object to the exact same bits regardless of platform,
/// architecture, or build. FNV-1a with a splitmix64 finalizer is fully
/// specified byte-by-byte; do not replace it with a hasher that does not
/// guarantee cross-platform stability (e.g. ahash, SipHash with random keys).
fn namespaced_hash(namespace: &str, block_hash: &[u8]) -> [u8; 8] {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    // Length prefix keeps (namespace, hash) boundaries unambiguous.
    for &byte in (namespace.len() as u64)
        .to_le_bytes()
        .iter()
        .chain(namespace.as_bytes())
        .chain(block_hash)
    {
        h = (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    splitmix64(h).to_be_bytes()
}

/// splitmix64 finalizer: strengthens FNV-1a's avalanche so the top bits used
/// for HLL bucket selection are uniformly distributed.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn derive_slot_duration(window: Duration) -> Duration {
    const MIN_SLOT: Duration = Duration::from_secs(60);
    const MAX_SLOT: Duration = Duration::from_secs(3600);
    let target = window / 24;
    target.clamp(MIN_SLOT, MAX_SLOT)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "../../tests/unit/metric/hll.rs"]
mod tests;
