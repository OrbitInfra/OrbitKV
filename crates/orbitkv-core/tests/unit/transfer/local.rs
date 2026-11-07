use super::*;

#[test]
fn raw_plan_round_trip_and_framing() {
    let plan = RawRestorePlan {
        copies: vec![RawCopy {
            source: SourceRange {
                arena_id: 1,
                allocation_id: 4,
                allocation_offset: 4096,
                allocation_size: 8192,
                offset: 6144,
                size: 512,
            },
            layer: "decoder.0.key".into(),
            destination_offset: 2048,
        }],
    };
    let bytes = plan.encode_parts().unwrap().pop_front().unwrap();
    let decoded = RawRestorePart::decode(&bytes).unwrap();
    assert_eq!(decoded.copies, plan.copies);
    assert_eq!(decoded.completed_layers, vec!["decoder.0.key"]);
    for end in 0..bytes.len() {
        assert!(RawRestorePart::decode(&bytes[..end]).is_err());
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(RawRestorePart::decode(&extra).is_err());
    let mut invalid_count = bytes;
    invalid_count[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(RawRestorePart::decode(&invalid_count).is_err());
}

#[test]
fn raw_plan_encoding_partitions_without_losing_copies_and_bounds_total_metadata() {
    let copy = RawCopy {
        source: SourceRange {
            arena_id: 1,
            allocation_id: 1,
            allocation_offset: 0,
            allocation_size: 1,
            offset: 0,
            size: 1,
        },
        layer: "layer".into(),
        destination_offset: 0,
    };
    let plan = RawRestorePlan {
        copies: vec![copy.clone(); MAX_PLAN_BYTES / 63 + 1],
    };
    let parts = plan.encode_parts().unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts.iter().all(|bytes| bytes.len() <= MAX_PLAN_BYTES));
    let copies: Vec<_> = parts
        .iter()
        .flat_map(|bytes| RawRestorePart::decode(bytes).unwrap().copies)
        .collect();
    assert_eq!(copies, plan.copies);
    assert!(
        RawRestorePart::decode(&parts[0])
            .unwrap()
            .completed_layers
            .is_empty()
    );
    assert_eq!(
        RawRestorePart::decode(&parts[1]).unwrap().completed_layers,
        vec!["layer"]
    );
    for name in [String::new(), "x".repeat(u16::MAX as usize + 1)] {
        assert!(
            RawRestorePlan {
                copies: vec![RawCopy {
                    layer: name,
                    ..copy.clone()
                }]
            }
            .encode_parts()
            .is_err()
        );
    }
    let long_copy = RawCopy {
        layer: "x".repeat(u16::MAX as usize),
        ..copy
    };
    let huge = RawRestorePlan {
        copies: vec![long_copy; MAX_RESTORE_PLAN_BYTES / (59 + u16::MAX as usize) + 1],
    };
    assert!(huge.encode_parts().unwrap_err().contains("metadata limit"));
    assert!(RawRestorePart::decode(&vec![0; MAX_PLAN_BYTES + 1]).is_err());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn caller_context_survives_registration_and_readiness_success_and_failure() {
    let primary = CudaContext::new(0).unwrap();
    let destination_stream = primary.new_stream().unwrap();
    // Match the CUDA-IPC-capable allocations exported by framework tensors.
    // Cudarc's alloc_zeros uses cuMemAllocAsync, whose context attribute is null
    // and whose allocation cannot use the legacy CUDA IPC registration path.
    // SAFETY: the primary context is current; the allocation remains owned by
    // this test until the executor has been dropped below.
    let address = unsafe { result::malloc_sync(64) }.unwrap();
    let caller = CudaContext::new_non_primary(0, 0).unwrap();
    let caller_stream = caller.new_stream().unwrap();
    let current = || result::ctx::get_current().unwrap();
    let binding = |pointer| {
        LocalTensor::new("layer".into(), pointer, 64, pointer as usize, 1, 64, 0, 1).unwrap()
    };

    let mut executor =
        LocalRestoreExecutor::new(0, vec![binding(address)], Vec::new(), TransferMode::Direct)
            .unwrap();
    assert_eq!(current(), Some(caller.cu_ctx()));
    executor
        .wait_for_destination(destination_stream.cu_stream() as u64)
        .unwrap();
    assert_eq!(current(), Some(caller.cu_ctx()));
    assert!(
        executor
            .wait_for_destination(caller_stream.cu_stream() as u64)
            .is_err()
    );
    assert_eq!(current(), Some(caller.cu_ctx()));

    // A real allocation in a custom context must still be rejected. Accepting
    // another current context does not permit unrelated destination pointers.
    // SAFETY: this thread owns the live caller context and frees this allocation
    // after the constructor has rejected it without submitting GPU work.
    let custom_pointer = unsafe { result::malloc_sync(64) }.unwrap();
    let rejected = LocalRestoreExecutor::new(
        0,
        vec![binding(custom_pointer)],
        Vec::new(),
        TransferMode::Direct,
    );
    assert!(rejected.is_err());
    assert_eq!(current(), Some(caller.cu_ctx()));
    // SAFETY: custom_pointer is owned by the current caller context; rejected
    // executor creation submitted no work using it.
    unsafe { result::free_sync(custom_pointer) }.unwrap();
    drop(executor);
    primary.bind_to_thread().unwrap();
    // SAFETY: the executor was dropped after drain and the allocation
    // context was rebound; address is freed exactly once.
    unsafe { result::free_sync(address) }.unwrap();
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn strided_restore_keeps_each_allocation_boundary_inside_a_shared_arena() {
    use crate::memory::numa::NumaNode;
    use crate::memory::pool::PinnedAllocator;
    use std::num::NonZeroU64;

    let _context = CudaContext::new(0).unwrap();
    let pool = PinnedAllocator::new_global(4096, 1, false, None).unwrap();
    let sources: Vec<_> = (0..4)
        .map(|index| {
            let source = pool
                .allocate(NonZeroU64::new(64).unwrap(), NumaNode::UNKNOWN)
                .unwrap();
            // SAFETY: each new allocation is uniquely owned before publication.
            unsafe {
                source.as_non_null().as_ptr().write_bytes(0x31 + index, 64);
            }
            source
        })
        .collect();
    let plan = RawRestorePlan {
        copies: sources
            .iter()
            .enumerate()
            .map(|(index, source)| RawCopy {
                source: source.source_range(source.as_non_null(), 64).unwrap(),
                layer: "layer".into(),
                destination_offset: (index * 64) as u64,
            })
            .collect(),
    };
    assert!(
        plan.copies
            .windows(2)
            .all(|pair| pair[0].source.arena_id == pair[1].source.arena_id
                && pair[0].source.allocation_id != pair[1].source.allocation_id)
    );
    // SAFETY: this context owns the allocation until the executor is dropped.
    let device = unsafe { result::malloc_sync(256) }.unwrap();
    let tensor =
        LocalTensor::new("layer".into(), device, 256, device as usize, 4, 64, 0, 1).unwrap();
    let mut executor = LocalRestoreExecutor::new(
        0,
        vec![tensor],
        pool.payload_arenas().unwrap(),
        TransferMode::Direct,
    )
    .unwrap();
    let part = RawRestorePart::decode(&plan.encode_parts().unwrap()[0]).unwrap();
    executor
        .execute(&part, &mut Default::default(), true, || {}, None)
        .unwrap();
    let mut actual = [0u8; 256];
    // SAFETY: execute has drained the stream, and the output buffer is 256 bytes.
    unsafe { sys::cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), device, actual.len()).result() }
        .unwrap();
    for (index, row) in actual.chunks_exact(64).enumerate() {
        assert!(row.iter().all(|byte| *byte == 0x31 + index as u8));
    }
    for fault in 0..4 {
        let mut invalid = plan.clone();
        match fault {
            0 => invalid.copies[0].source.offset += invalid.copies[0].source.allocation_size,
            1 => invalid.copies[1].source.allocation_id = invalid.copies[0].source.allocation_id,
            2 => invalid.copies[0].source.allocation_id = 0,
            _ => invalid.copies[1].destination_offset = 0,
        }
        let invalid = RawRestorePart::decode(&invalid.encode_parts().unwrap()[0]).unwrap();
        assert!(
            executor
                .execute(&invalid, &mut Default::default(), true, || {}, None)
                .is_err()
        );
    }
    drop(executor);
    // SAFETY: every accepted copy has completed and no executor retains the destination.
    unsafe { result::free_sync(device) }.unwrap();
}

#[test]
#[ignore = "CPU CUDA shims: failed arena/pool registration must deregister before unmapping"]
fn failed_import_cleanup_preserves_registered_mapping() {
    use rustix::fs::{MemfdFlags, SealFlags, fcntl_add_seals, ftruncate, memfd_create};
    use rustix::process::{Resource, Rlimit, setrlimit};
    use std::os::unix::process::ExitStatusExt;

    assert_eq!(
        std::env::var("ORBITKV_REQUIRE_LOCAL_COPY_SHIM").as_deref(),
        Ok("1")
    );
    if std::env::var_os("ORBITKV_TEST_IMPORT_ADDRESS_FAILURE").is_none() {
        for (kind, refuses) in [
            ("arena", false),
            ("arena", true),
            ("pool", false),
            ("pool", true),
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "transfer::local::tests::failed_import_cleanup_preserves_registered_mapping",
                    "--ignored",
                    "--nocapture",
                ])
                .env("ORBITKV_TEST_IMPORT_KIND", kind)
                .env("ORBITKV_TEST_IMPORT_ADDRESS_FAILURE", "1")
                .env_remove("ORBITKV_TEST_IMPORT_UNREGISTER_FAILURE");
            if refuses {
                command.env("ORBITKV_TEST_IMPORT_UNREGISTER_FAILURE", "1");
            }
            let output = command.output().unwrap();
            if refuses {
                assert_eq!(output.status.signal(), Some(libc::SIGABRT), "{output:?}");
            } else {
                assert!(output.status.success(), "{output:?}");
            }
        }
        return;
    }
    setrlimit(
        Resource::Core,
        Rlimit {
            current: Some(0),
            maximum: Some(0),
        },
    )
    .unwrap();
    if std::env::var("ORBITKV_TEST_IMPORT_KIND").as_deref() == Ok("pool") {
        use crate::memory::numa::NumaNode;
        use crate::memory::pinned::{PagePolicy, PinnedMemError, PinnedMemory};
        assert!(matches!(
            PinnedMemory::allocate(4096, PagePolicy::Regular, NumaNode::UNKNOWN),
            Err(PinnedMemError::CudaGetDevicePointerFailed(_))
        ));
        return;
    }
    let fd = memfd_create(
        "failed-import",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .unwrap();
    ftruncate(&fd, 4096).unwrap();
    fcntl_add_seals(&fd, SealFlags::GROW | SealFlags::SHRINK | SealFlags::SEAL).unwrap();
    let result = ImportedArena::new(
        PayloadArena {
            id: 1,
            size: 4096,
            fd,
        },
        CudaContext::new(0).unwrap(),
    );
    assert!(
        result
            .err()
            .unwrap()
            .contains("payload CUDA address failed")
    );
}
