use super::*;

fn resources(queue_depth: u32) -> CompletionResourceEvidence {
    CompletionResourceEvidence {
        decode_page_bytes: 4096,
        queue_depth,
        queue_parallelism: 1,
        transfer_inflight_bytes: 0,
        transfer_bandwidth_bytes_per_second: 0,
    }
}

#[test]
fn resource_evidence_is_fresh_bounded_and_resource_scoped() {
    let resource = ExecutionResource::CacheRestore {
        source_set_hash: 1,
        destination_device: 2,
        copy_backend: 0,
    };
    record(resource, resources(3), Duration::ZERO);
    assert_eq!(
        current(resource, Instant::now())
            .unwrap()
            .resources
            .queue_depth,
        3
    );

    let stale = ExecutionResource::CacheRestore {
        source_set_hash: 4,
        destination_device: 2,
        copy_backend: 0,
    };
    record(stale, resources(7), MAX_AGE + Duration::from_millis(1));
    assert!(current(stale, Instant::now()).is_none());

    for source_set_hash in 10..10 + CAPACITY as u64 {
        record(
            ExecutionResource::CacheRestore {
                source_set_hash,
                destination_device: 2,
                copy_backend: 0,
            },
            resources(1),
            Duration::ZERO,
        );
    }
    assert!(EVIDENCE.lock().len() <= CAPACITY);
}
