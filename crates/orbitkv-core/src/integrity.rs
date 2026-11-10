//! Checksums and accounting; the existing replica/transfer owners decide release.

use std::time::Instant;

use crate::block::RawBlock;

pub(crate) fn checksum<'a>(slots: impl IntoIterator<Item = &'a RawBlock>) -> u32 {
    let mut crc = 0;
    for (pointer, bytes) in slots.into_iter().flat_map(RawBlock::segment_iovecs) {
        // SAFETY: sealing and read verification run only after the writer's physical
        // receipt. Slots retain initialized, immutable allocations for this pass.
        let bytes = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), bytes) };
        crc = crc32c::crc32c_append(crc, bytes);
    }
    crc
}

pub(crate) fn checked(tier: &'static str, bytes: u64, started: Instant, valid: bool) -> bool {
    let metrics = crate::metrics::core_metrics();
    let attributes = [opentelemetry::KeyValue::new("tier", tier)];
    metrics.integrity_bytes.add(bytes, &attributes);
    metrics
        .integrity_seconds
        .record(started.elapsed().as_secs_f64(), &attributes);
    if !valid {
        failed(tier, "checksum");
    }
    valid
}

pub(crate) fn failed(tier: &'static str, reason: &'static str) {
    crate::metrics::core_metrics().integrity_failures.add(
        1,
        &[
            opentelemetry::KeyValue::new("tier", tier),
            opentelemetry::KeyValue::new("reason", reason),
        ],
    );
}
