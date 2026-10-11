use super::*;
use crate::transfer::local::SourceRange;

fn bindings(trace: &mut LocalCopyTrace, copies: &[CopyDesc]) {
    let tensor = LocalTensor {
        name: "layer".into(),
        address: 0x10000,
        bytes: 65536,
        allocation: 7,
    };
    for copy in copies {
        let raw = RawCopy {
            source: SourceRange {
                arena_id: 2,
                allocation_id: 3,
                allocation_offset: 256,
                allocation_size: 16384,
                offset: copy.host as u64 - 0x100000,
                size: copy.size as u64,
            },
            layer: "layer".into(),
            destination_offset: copy.device - tensor.address,
        };
        trace.record_geometry(&raw, copy, &tensor, 65536);
    }
}

fn copies() -> [CopyDesc; 2] {
    [0, 4096].map(|offset| CopyDesc {
        device: 0x10000 + offset,
        host: (0x100100 + offset) as *mut u8,
        host_device: 0x200100 + offset,
        size: 4096,
        device_allocation: 7,
        host_registration: 0x100000,
    })
}

fn events(trace: &LocalCopyTrace) -> Vec<serde_json::Value> {
    let mut values = Vec::new();
    trace.flush_with(|event| values.push(event));
    values
}

#[test]
fn captures_actual_submitted_order_and_one_shared_part_drain() {
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 100);
    let copies = copies();
    trace.begin_part(2, true);
    bindings(&mut trace, &copies);
    for copy in copies.iter().rev() {
        let index = trace.begin_batch(&[*copy], "direct").unwrap();
        trace.finish_batch(index, 17, true);
    }
    trace.finish_part(31, true);
    let values = events(&trace);
    assert_eq!(values.len(), 5);
    assert_eq!(values[1]["copies"][0]["destination_offset"], 4096);
    assert_eq!(values[2]["copies"][0]["source_offset"], 256);
    assert_eq!(values[1]["copies"][0]["source_allocation_bytes"], 16384);
    assert_eq!(values[1]["copies"][0]["registered_device_bytes"], 65536);
    assert_eq!(values[1]["dma_ranges"], 1);
    assert!(values[1].get("host_drain_ns").is_none());
    assert_eq!(values[3]["host_drain_ns"], 31);
    assert_eq!(values[4]["descriptors"], 2);
    assert_eq!(values[4]["complete"], true);
}

#[test]
fn partial_failure_overflow_missing_submission_and_missing_drain_are_invalid() {
    for mode in ["failure", "overflow", "lost_batch", "no_drain", "no_pitch"] {
        let pitch = if mode == "no_pitch" {
            Err("driver".into())
        } else {
            Ok(usize::MAX)
        };
        let mut trace = LocalCopyTrace::new(0, pitch, if mode == "overflow" { 4 } else { 100 });
        let copies = copies();
        trace.begin_part(2, true);
        bindings(&mut trace, &copies);
        if let Some(index) = trace.begin_batch(&copies[..1], "direct") {
            trace.finish_batch(index, 17, mode != "failure");
        }
        if mode != "lost_batch"
            && let Some(index) = trace.begin_batch(&copies[1..], "direct")
        {
            trace.finish_batch(index, 17, true);
        }
        if mode != "no_drain" {
            trace.finish_part(31, mode != "failure");
        }
        assert_eq!(events(&trace).last().unwrap()["complete"], false, "{mode}");
    }
}

#[test]
fn empty_managed_parts_are_preserved_without_inventing_h2d_bytes() {
    let mut trace = LocalCopyTrace::new(0, Ok(usize::MAX), 2);
    trace.begin_part(0, true);
    trace.finish_part(19, true);
    let values = events(&trace);
    assert_eq!(values.len(), 3);
    assert_eq!(values[1]["planned_descriptors"], 0);
    assert_eq!(values[2]["batches"], 0);
    assert_eq!(values[2]["complete"], true);
}
