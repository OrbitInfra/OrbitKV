use super::*;
use crate::transfer_engine::PeerTransferError;

#[test]
fn cpu_tcp_backend_reads_and_writes_fragmented_bytes_and_drains() {
    use crate::transfer_engine::TransferOp as Operation;
    use std::ptr::NonNull;

    let make = || {
        Arc::new(
            OrbitTeTransport::new(
                4096,
                "127.0.0.1:0",
                Transport::Tcp,
                #[cfg(feature = "orbitte-rdma")]
                None,
            )
            .unwrap(),
        )
    };
    let source = make();
    let requester = make();
    let original = (0..1024).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let remote = source
        .engine
        .register_memory(
            original.clone().into_boxed_slice(),
            Permission::RemoteReadWrite,
        )
        .unwrap();
    let local = requester
        .engine
        .register_memory(vec![0xcc; 1024].into_boxed_slice(), Permission::LocalOnly)
        .unwrap();
    let backend: &dyn TransferBackend = requester.as_ref();
    let peer = backend
        .open_peer(
            &source.endpoint,
            &source.epoch,
            Instant::now() + Duration::from_secs(3),
        )
        .unwrap();
    // The first two descriptors merge to one physical owner. Bytes must not be
    // summed through their overlapping per-task completion counters.
    let slices =
        [(13, 29, 128), (141, 157, 128), (700, 800, 100)].map(|(l, r, length)| TransferSlice {
            local: NonNull::new((local.base_address() + l) as *mut u8).unwrap(),
            remote_address: remote.base_address() + r,
            length,
        });
    let cancel = Cancellation::default();
    // SAFETY: both registrations and backing remain owned and no access races.
    assert_eq!(
        unsafe {
            backend.transfer(
                Operation::Read,
                &peer,
                &slices,
                Instant::now() + Duration::from_secs(3),
                &cancel,
            )
        }
        .unwrap(),
        356
    );
    let mut got = vec![0; 1024];
    local.read(0, &mut got).unwrap();
    let mut expected = vec![0xcc; 1024];
    expected[13..269].copy_from_slice(&original[29..285]);
    expected[700..800].copy_from_slice(&original[800..900]);
    assert_eq!(got, expected);
    local.write(13, &[0x5a; 256]).unwrap();
    // SAFETY: READ has drained and the WRITE source is immutable until return.
    assert_eq!(
        unsafe {
            backend.transfer(
                Operation::Write,
                &peer,
                &slices[..2],
                Instant::now() + Duration::from_secs(3),
                &cancel,
            )
        }
        .unwrap(),
        256
    );
    remote.read(0, &mut got).unwrap();
    let mut expected = original;
    expected[29..285].fill(0x5a);
    assert_eq!(got, expected);

    for (cancelled, expired, kind) in [
        (true, false, ErrorKind::Cancelled),
        (false, true, ErrorKind::Timeout),
    ] {
        let cancellation = Cancellation::default();
        if cancelled {
            cancellation.cancel();
        }
        let deadline = Instant::now()
            + if expired {
                Duration::ZERO
            } else {
                Duration::from_secs(3)
            };
        // SAFETY: retained registrations; preflight must reject before any copy.
        let result =
            unsafe { backend.transfer(Operation::Read, &peer, &slices, deadline, &cancellation) };
        assert_eq!(result.unwrap_err().kind, kind);
    }
    let mut registration = Registration {
        inner: RegistrationInner::OrbitTe(RegistrationSet {
            transport: Arc::clone(&requester),
            regions: vec![local],
            cleanup_failed: false,
        }),
    };
    // SAFETY: every admitted operation returned with physical drain.
    unsafe { backend.unregister(&mut registration) }.unwrap();
    drop(registration);
    source.engine.unregister_memory(&remote).unwrap();
    drop(remote);
    backend
        .shutdown(Instant::now() + Duration::from_secs(3))
        .unwrap();
    source
        .shutdown(Instant::now() + Duration::from_secs(3))
        .unwrap();
    for engine in [&source.engine, &requester.engine] {
        let stats = engine.stats().unwrap();
        assert_eq!(
            (
                stats.retained_regions,
                stats.inflight_batches,
                stats.inflight_bytes,
                stats.allocated_batches
            ),
            (0, 0, 0, 0)
        );
    }
}

#[test]
fn peer_protocol_and_granted_epoch_are_preserved_across_source_restart() {
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let granted_epoch = source.source_epoch().to_string();
    let requester: Arc<dyn TransferBackend> = Arc::new(
        OrbitTeTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Tcp,
            #[cfg(feature = "orbitte-rdma")]
            None,
        )
        .unwrap(),
    );
    let named_endpoint = format!("{}{endpoint}", prefix(Transport::Tcp));
    assert!(matches!(
        requester.read(&endpoint.to_string(), &[], Duration::from_secs(2), &granted_epoch),
        Err(PeerTransferError::Failed(reason)) if reason.contains("protocol mismatch")
    ));
    assert_eq!(
        requester
            .read(&named_endpoint, &[], Duration::from_secs(2), &granted_epoch)
            .unwrap(),
        0
    );
    source.close_source(Duration::from_secs(2)).unwrap();
    drop(source);
    let replacement = TransferEngine::new(EngineConfig {
        bind_addr: endpoint,
        ..EngineConfig::default()
    })
    .unwrap();
    let fresh_epoch = replacement.source_epoch().to_string();
    assert_ne!(granted_epoch, fresh_epoch);
    assert!(matches!(
        requester.read(&named_endpoint, &[], Duration::from_secs(2), &granted_epoch),
        Err(PeerTransferError::Failed(reason)) if reason.contains("epoch")
    ));
    assert_eq!(
        requester
            .read(&named_endpoint, &[], Duration::from_secs(2), &fresh_epoch)
            .unwrap(),
        0
    );
    replacement.close_source(Duration::from_secs(2)).unwrap();
}

#[test]
fn transfer_rejects_descriptor_and_byte_budgets_before_payload_io() {
    let requester = OrbitTeTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Tcp,
        #[cfg(feature = "orbitte-rdma")]
        None,
    )
    .unwrap();
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = format!("{}{}", prefix(Transport::Tcp), source.endpoint());
    let peer = requester
        .open_peer(
            &endpoint,
            &source.source_epoch().to_string(),
            Instant::now() + Duration::from_secs(2),
        )
        .unwrap();
    let request = TransferSlice {
        local: std::ptr::NonNull::dangling(),
        remote_address: 1,
        length: SLICE_BYTES,
    };
    let cases = [
        (vec![request; MAX_BATCH_BYTES / SLICE_BYTES + 1], "64 MiB"),
        (
            vec![
                TransferSlice {
                    length: 1,
                    ..request
                };
                MAX_BATCH_REQUESTS + 1
            ],
            "descriptor budget",
        ),
    ];
    for (requests, expected) in cases {
        // SAFETY: every case exceeds an admission limit and must be rejected
        // before the deliberately unregistered addresses reach a transport.
        let error = unsafe {
            requester.transfer(
                crate::transfer_engine::TransferOp::Read,
                &peer,
                &requests,
                Instant::now() + Duration::from_secs(1),
                &Cancellation::default(),
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, ErrorKind::ResourceExhausted);
        assert!(error.message.contains(expected), "{error}");
    }
    source.close(Duration::from_secs(2)).unwrap();
}

#[cfg(not(feature = "orbitte-rdma"))]
#[test]
fn rdma_selection_requires_its_build_feature() {
    assert!(matches!(
        OrbitTeTransport::new(4096, "127.0.0.1:0", Transport::Rdma),
        Err(reason) if reason.contains("orbitte-rdma build feature")
    ));
}

#[cfg(feature = "orbitte-rdma")]
#[test]
fn unavailable_rdma_never_copies_through_tcp() {
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let source_memory = source
        .register_memory(vec![0x42; 64].into_boxed_slice(), Permission::RemoteRead)
        .unwrap();
    let requester = OrbitTeTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Rdma,
        Some(crate::OrbitTeRdmaConfig::new("orbitkv-test-absent")),
    )
    .unwrap();
    let host = Arc::new(orbit_te::HostMemory::new(vec![0xcc; 64].into_boxed_slice()).unwrap());
    let local =
        requester
            .engine
            .register_host_memory(Arc::clone(&host), 0, 64, Permission::LocalOnly);
    match &local {
        Ok(local) => {
            let requests = [TransferRequest::new(
                Transport::Rdma,
                local.base_address(),
                source_memory.base_address(),
                64,
            )];
            let endpoint = format!("{}{}", prefix(Transport::Rdma), source.endpoint());
            assert!(matches!(
                requester.read_registered(
                    &endpoint,
                    &source.source_epoch().to_string(),
                    &requests,
                    Instant::now() + Duration::from_secs(2)
                ),
                Err(PeerTransferError::Failed(_))
            ));
        }
        // Current OrbitTE may reject unavailable verbs at registration, before
        // submission. Neither boundary may silently install a TCP payload route.
        Err(error) => assert!(
            matches!(error, Error::Unavailable(_) | Error::Backend { .. }),
            "{error}"
        ),
    }
    let mut bytes = [0; 64];
    host.read(0, &mut bytes).unwrap();
    assert_eq!(
        bytes, [0xcc; 64],
        "TCP fallback would have copied source bytes"
    );
    let usage = requester.engine.stats().unwrap().rdma.unwrap();
    assert_eq!(
        (
            usage.completed_reads,
            usage.active_endpoints,
            usage.staging_bytes,
            usage.retained_endpoints
        ),
        (0, 0, 0, 0)
    );
    if let Ok(local) = local {
        requester.engine.unregister_memory(&local).unwrap();
    }
    drop(host);
    source.unregister_memory(&source_memory).unwrap();
    drop(source_memory);
    source.close_source(Duration::from_secs(2)).unwrap();
}

#[cfg(feature = "orbitte-rdma")]
#[test]
fn rdma_configuration_and_payload_namespaces_are_explicit() {
    assert!(OrbitTeTransport::new(4096, "127.0.0.1:0", Transport::Rdma, None).is_err());
    assert!(
        OrbitTeTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Tcp,
            Some(crate::OrbitTeRdmaConfig::new("orbitkv-test-absent"))
        )
        .is_err()
    );
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let requester: Arc<dyn TransferBackend> = Arc::new(
        OrbitTeTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Rdma,
            Some(crate::OrbitTeRdmaConfig::new("orbitkv-test-absent")),
        )
        .unwrap(),
    );
    assert!(!requester.accepts_endpoint(&format!("{}{endpoint}", prefix(Transport::Tcp))));
    assert!(requester.accepts_endpoint(&format!("{}{endpoint}", prefix(Transport::Rdma))));
    let named = format!("{}{endpoint}", prefix(Transport::Rdma));
    let old_epoch = source.source_epoch().to_string();
    assert_eq!(
        requester
            .read(&named, &[], Duration::from_secs(2), &old_epoch)
            .unwrap(),
        0
    );
    source.close_source(Duration::from_secs(2)).unwrap();
    drop(source);
    let replacement = TransferEngine::new(EngineConfig {
        bind_addr: endpoint,
        ..EngineConfig::default()
    })
    .unwrap();
    assert!(
        matches!(requester.read(&named, &[], Duration::from_secs(2), &old_epoch), Err(PeerTransferError::Failed(reason)) if reason.contains("epoch"))
    );
    replacement.close_source(Duration::from_secs(2)).unwrap();
}

#[cfg(feature = "orbitte-rdma")]
#[path = "orbitte/rdma_host.rs"]
mod rdma_host;

// Hardware consumers share the same trait submission as production peer READs.
impl OrbitTeTransport {
    fn read_registered(
        &self,
        endpoint: &str,
        epoch: &str,
        requests: &[TransferRequest],
        deadline: Instant,
    ) -> Result<usize, PeerTransferError> {
        let slices = requests
            .iter()
            .map(|r| TransferSlice {
                local: std::ptr::NonNull::new(r.local_address as *mut u8).unwrap(),
                remote_address: r.remote_address,
                length: r.length,
            })
            .collect::<Vec<_>>();
        let peer = self.open_peer(endpoint, epoch, deadline)?;
        // SAFETY: test callers own and register every actual buffer through this call.
        unsafe {
            self.transfer(
                crate::transfer_engine::TransferOp::Read,
                &peer,
                &slices,
                deadline,
                &Cancellation::default(),
            )
        }
        .map_err(Into::into)
    }
}
