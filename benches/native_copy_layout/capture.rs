//! Offline validation of captured batches through the unchanged DMA compiler.

use serde_json::{Value, json};

use super::{CopyDesc, ranges};

fn integer(value: &Value, field: &str) -> Result<u64, String> {
    value[field]
        .as_u64()
        .ok_or_else(|| format!("missing unsigned {field}"))
}

fn size(value: &Value, field: &str) -> Result<usize, String> {
    usize::try_from(integer(value, field)?).map_err(|_| format!("{field} exceeds native usize"))
}

pub(super) fn validate(summary: &Value) -> Result<Value, String> {
    let traces = summary["traces"].as_array().ok_or("missing traces")?;
    if traces.is_empty() || usize::BITS != 64 {
        return Err("nonempty capture and 64-bit native compiler required".into());
    }
    let mut batches = 0;
    let mut descriptors = 0;
    let mut dma_ranges = 0;
    for trace in traces {
        let max_pitch = size(&trace["header"], "max_pitch")?;
        let limit = size(&trace["header"], "limit")?;
        if max_pitch == 0 || !(1..=65_536).contains(&limit) {
            return Err("invalid captured maximum pitch or budget".into());
        }
        let mut retained = 0;
        for batch in trace["batches"].as_array().ok_or("missing batches")? {
            let geometry = batch["copies"].as_array().ok_or("missing copies")?;
            if geometry.is_empty() || geometry.len() != size(batch, "descriptors")? {
                return Err("incomplete captured descriptor list".into());
            }
            retained += geometry.len();
            if retained > limit {
                return Err("captured descriptors exceed budget".into());
            }
            let copies = geometry
                .iter()
                .map(|copy| {
                    Ok(CopyDesc {
                        device: integer(copy, "device")?,
                        host: size(copy, "host")? as *mut u8,
                        host_device: integer(copy, "host_device")?,
                        size: size(copy, "size")?,
                        device_allocation: size(copy, "device_allocation")?,
                        host_registration: size(copy, "host_registration")?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let compiled = ranges::dma_copies(&copies, max_pitch).count();
            if compiled != size(batch, "dma_ranges")? {
                return Err("captured DMA count contradicts production compiler".into());
            }
            batches += 1;
            descriptors += copies.len();
            dma_ranges += compiled;
        }
    }
    if batches == 0
        || traces.len() != size(summary, "executors")?
        || batches != size(summary, "batches")?
        || descriptors != size(summary, "descriptors")?
        || dma_ranges != size(summary, "dma_ranges")?
    {
        return Err("capture totals contradict actual compiled geometry".into());
    }
    Ok(json!({
        "schema": 1, "production_compiler_verified": true,
        "executors": traces.len(), "batches": batches,
        "descriptors": descriptors, "dma_ranges": dma_ranges,
        "scope": "CPU replay of the production DMA compiler over captured addresses/owners/pitch; no CUDA initialization or pointer dereference"
    }))
}
