//! Checksums and accounting; the existing replica/transfer owners decide release.

use std::time::Instant;

/// End-relative CRC references partition a stored slot/extent without overlap.
/// Boundaries follow existing layer/segment descriptors, including padding.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checksum {
    pub end: u64,
    pub crc: u32,
}

pub(crate) const MAX_CHECKSUMS: usize = 65_536;

pub(crate) fn validate(checksums: &[Checksum], bytes: u64) -> Result<(), String> {
    if checksums.is_empty() || checksums.len() > MAX_CHECKSUMS {
        return Err("missing or excessive integrity references".into());
    }
    let mut end = 0;
    for part in checksums {
        if part.end <= end || part.end > bytes {
            return Err("invalid integrity range".into());
        }
        end = part.end;
    }
    if end != bytes {
        return Err("incomplete integrity references".into());
    }
    Ok(())
}

pub(crate) fn from_wire(bytes: &[u8], length: u64) -> Result<Vec<Checksum>, String> {
    if bytes.is_empty() || bytes.len() > 1 << 20 {
        return Err("invalid peer checksum metadata size".into());
    }
    let checksums: Vec<Checksum> =
        serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    validate(&checksums, length)?;
    Ok(checksums)
}

pub(crate) fn combine(left: u32, right: u32, bytes: u64) -> u32 {
    let mut hash = crc32fast::Hasher::new_with_initial(left);
    hash.combine(&crc32fast::Hasher::new_with_initial_len(right, bytes));
    hash.finalize()
}

/// Padding is emitted by the engine, so its CRC needs no payload allocation/scan.
pub(crate) fn zeros(mut bytes: u64) -> u32 {
    let mut result = 0;
    let mut length = 1;
    let mut part = crc32fast::hash(&[0]);
    while bytes != 0 {
        if bytes & 1 != 0 {
            result = combine(result, part, length);
        }
        bytes >>= 1;
        if bytes != 0 {
            part = combine(part, part, length);
            length *= 2;
        }
    }
    result
}

pub(crate) fn append_zeros(checksums: &mut Vec<Checksum>, end: u64, boundaries: &[u64]) {
    let mut offset = checksums.last().map_or(0, |part| part.end);
    if end <= offset {
        return;
    }
    for boundary in boundaries
        .iter()
        .copied()
        .filter(|boundary| *boundary > offset && *boundary < end)
        .chain(std::iter::once(end))
        .collect::<Vec<_>>()
    {
        checksums.push(Checksum {
            end: boundary,
            crc: zeros(boundary - offset),
        });
        offset = boundary;
    }
}

pub(crate) fn range(checksums: &[Checksum], offset: u64, bytes: u64) -> Result<u32, String> {
    let start = if offset == 0 {
        0
    } else {
        checksums
            .binary_search_by_key(&offset, |part| part.end)
            .map_err(|_| "integrity range starts inside a stored component")?
            + 1
    };
    let end = offset
        .checked_add(bytes)
        .ok_or("integrity range overflow")?;
    let last = checksums
        .binary_search_by_key(&end, |part| part.end)
        .map_err(|_| "integrity range ends inside a stored component")?;
    if start > last {
        return Err("empty integrity range".into());
    }
    let mut previous = offset;
    let mut crc = 0;
    for part in &checksums[start..=last] {
        crc = combine(crc, part.crc, part.end - previous);
        previous = part.end;
    }
    Ok(crc)
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

#[cfg(test)]
#[path = "../tests/unit/integrity.rs"]
pub(crate) mod tests;
