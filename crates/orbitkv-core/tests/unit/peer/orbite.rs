use super::*;
use crate::transfer_engine::PeerTransferError;

#[path = "orbite/retirement.rs"]
mod retirement;

#[test]
fn cpu_auto_backend_reads_and_writes_fragmented_bytes_and_drains() {
    use crate::transfer_engine::TransferOp as Operation;
    use std::ptr::NonNull;

    let make = || {
        Arc::new(
            OrbiteTransport::new(
                4096,
                "127.0.0.1:0",
                Transport::Unspec,
                None,
                #[cfg(feature = "orbite-rdma")]
                None,
            )
            .unwrap(),
        )
    };
    let source = make();
    let requester = make();
    let size = 18 << 20;
    let original = (0..size).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    let remote = source
        .engine
        .register_memory(
            original.clone().into_boxed_slice(),
            Permission::RemoteReadWrite,
        )
        .unwrap();
    let local = requester
        .engine
        .register_memory(vec![0xcc; size].into_boxed_slice(), Permission::LocalOnly)
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
    let slices = [(13, 29, 128), (141, 157, 128), (700, 800, size - 900)].map(|(l, r, length)| {
        TransferSlice {
            local: NonNull::new((local.base_address() + l) as *mut u8).unwrap(),
            remote_address: remote.base_address() + r,
            length,
        }
    });
    let cancel = Cancellation::default();
    // SAFETY: both registrations and backing remain owned and no access races.
    assert_eq!(
        unsafe {
            backend.transfer(
                Operation::Read,
                &peer,
                &slices,
                crate::transfer_engine::TransferOptions::read(
                    crate::QueryMode::Demand,
                    Instant::now() + Duration::from_secs(3),
                ),
                &cancel,
            )
        }
        .unwrap(),
        size - 644
    );
    let mut got = vec![0; size];
    local.read(0, &mut got).unwrap();
    let mut expected = vec![0xcc; size];
    expected[13..269].copy_from_slice(&original[29..285]);
    expected[700..size - 200].copy_from_slice(&original[800..size - 100]);
    assert_eq!(got, expected);
    local.write(13, &[0x5a; 256]).unwrap();
    // SAFETY: READ has drained and the WRITE source is immutable until return.
    assert_eq!(
        unsafe {
            backend.transfer(
                Operation::Write,
                &peer,
                &slices[..2],
                crate::transfer_engine::TransferOptions::read(
                    crate::QueryMode::Demand,
                    Instant::now() + Duration::from_secs(3),
                ),
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
        let result = unsafe {
            backend.transfer(
                Operation::Read,
                &peer,
                &slices,
                crate::transfer_engine::TransferOptions::read(crate::QueryMode::Demand, deadline),
                &cancellation,
            )
        };
        assert_eq!(result.unwrap_err().kind, kind);
    }
    let mut registration = Registration {
        inner: RegistrationInner::Orbite(RegistrationSet {
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
    let requester = OrbiteTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Tcp,
        None,
        #[cfg(feature = "orbite-rdma")]
        None,
    )
    .unwrap();
    let deadline = || Instant::now() + Duration::from_secs(3);
    let named_endpoint = format!("{}{endpoint}", prefix(Transport::Tcp));
    assert!(matches!(
        requester.open_peer(&endpoint.to_string(), &granted_epoch, deadline()),
        Err(error) if error.message.contains("protocol mismatch")
    ));
    requester
        .open_peer(&named_endpoint, &granted_epoch, deadline())
        .unwrap();
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
        requester.open_peer(&named_endpoint, &granted_epoch, deadline()),
        Err(error) if error.kind == ErrorKind::EpochMismatch
    ));
    requester
        .open_peer(&named_endpoint, &fresh_epoch, deadline())
        .unwrap();
    replacement.close_source(Duration::from_secs(2)).unwrap();
}

#[test]
fn transfer_rejects_descriptor_and_byte_budgets_before_payload_io() {
    let requester = OrbiteTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Tcp,
        None,
        #[cfg(feature = "orbite-rdma")]
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
        length: 8 << 20,
    };
    let cases = [
        (vec![request; MAX_BATCH_BYTES / (8 << 20) + 1], "64 MiB"),
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
                crate::transfer_engine::TransferOptions::read(
                    crate::QueryMode::Demand,
                    Instant::now() + Duration::from_secs(1),
                ),
                &Cancellation::default(),
            )
        }
        .unwrap_err();
        assert_eq!(error.kind, ErrorKind::ResourceExhausted);
        assert!(error.message.contains(expected), "{error}");
    }
    source.close(Duration::from_secs(2)).unwrap();
}

#[cfg(not(feature = "orbite-rdma"))]
#[test]
fn rdma_selection_requires_its_build_feature() {
    assert!(matches!(
        OrbiteTransport::new(4096, "127.0.0.1:0", Transport::Rdma, None),
        Err(reason) if reason.contains("orbite-rdma build feature")
    ));
}

#[cfg(feature = "orbite-rdma")]
#[test]
fn unavailable_rdma_never_copies_through_tcp() {
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let source_memory = source
        .register_memory(vec![0x42; 64].into_boxed_slice(), Permission::RemoteRead)
        .unwrap();
    let requester = OrbiteTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Rdma,
        None,
        Some(crate::OrbiteRdmaConfig::new("orbitkv-test-absent")),
    )
    .unwrap();
    let host = Arc::new(orbite::HostMemory::new(vec![0xcc; 64].into_boxed_slice()).unwrap());
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
        // Current OrbiTE may reject unavailable verbs at registration, before
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

#[cfg(feature = "orbite-rdma")]
#[test]
fn rdma_configuration_and_payload_namespaces_are_explicit() {
    assert!(OrbiteTransport::new(4096, "127.0.0.1:0", Transport::Rdma, None, None).is_err());
    assert!(
        OrbiteTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Tcp,
            None,
            Some(crate::OrbiteRdmaConfig::new("orbitkv-test-absent"))
        )
        .is_err()
    );
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let requester: Arc<dyn TransferBackend> = Arc::new(
        OrbiteTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Rdma,
            None,
            Some(crate::OrbiteRdmaConfig::new("orbitkv-test-absent")),
        )
        .unwrap(),
    );
    assert!(requester.accepts_endpoint(&format!("{}{endpoint}", prefix(Transport::Tcp))));
    assert!(requester.accepts_endpoint(&format!("{}{endpoint}", prefix(Transport::Rdma))));
    let named = format!("{}{endpoint}", prefix(Transport::Rdma));
    let old_epoch = source.source_epoch().to_string();
    requester
        .open_peer(&named, &old_epoch, Instant::now() + Duration::from_secs(2))
        .unwrap();
    source.close_source(Duration::from_secs(2)).unwrap();
    drop(source);
    let replacement = TransferEngine::new(EngineConfig {
        bind_addr: endpoint,
        ..EngineConfig::default()
    })
    .unwrap();
    assert!(
        matches!(requester.open_peer(&named, &old_epoch, Instant::now() + Duration::from_secs(2)), Err(error) if error.kind == ErrorKind::EpochMismatch)
    );
    replacement.close_source(Duration::from_secs(2)).unwrap();
}

#[cfg(feature = "orbite-rdma")]
#[path = "orbite/rdma_host.rs"]
mod rdma_host;

// Hardware consumers share the same trait submission as production peer READs.
#[cfg(feature = "orbite-rdma")]
impl OrbiteTransport {
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
                crate::transfer_engine::TransferOptions::read(crate::QueryMode::Demand, deadline),
                &Cancellation::default(),
            )
        }
        .map_err(Into::into)
    }
}

#[cfg(feature = "orbite-gds")]
#[test]
fn compiled_gds_stays_unloaded_without_configuration() {
    for storage in [
        None,
        Some(crate::SsdCacheConfig {
            backend: crate::SsdBackend::Auto,
            ..Default::default()
        }),
    ] {
        let engine = OrbiteTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Tcp,
            storage.as_ref(),
            #[cfg(feature = "orbite-rdma")]
            None,
        )
        .unwrap();
        let stats = engine.engine.stats().unwrap();
        assert!(!stats.gds.available);
        assert!(stats.gds.mode.is_none());
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        assert!(
            !maps.lines().any(|line| line.contains("libcufile.so")),
            "compiling GDS must not load cuFile for the default/auto profile"
        );
    }
}

#[test]
fn kv_options_drive_unspec_policy_on_real_tcp() {
    use crate::transfer_engine::{TransferIntent, TransferOptions};
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let remote = source
        .register_memory(
            Vec::from(*b"intent!!").into_boxed_slice(),
            Permission::RemoteRead,
        )
        .unwrap();
    let cases = [
        (
            TransferIntent::ForegroundGet,
            orbite::IntentType::ForegroundGet,
            orbite::Priority::High,
        ),
        (
            TransferIntent::BackgroundPrefetch,
            orbite::IntentType::BackgroundPrefetch,
            orbite::Priority::Low,
        ),
        (
            TransferIntent::Checkpoint,
            orbite::IntentType::Checkpoint,
            orbite::Priority::Low,
        ),
    ];
    let mut policy = cases
        .iter()
        .map(|(_, intent, priority)| orbite::SelectionPolicy {
            name: format!("{intent:?}"),
            intent_type: Some(*intent),
            priority: Some(*priority),
            transports: vec![Transport::Tcp],
            ..Default::default()
        })
        .collect::<Vec<_>>();
    // A dropped or wrong scheduling field selects an unavailable route.
    policy.push(orbite::SelectionPolicy {
        name: "reject-unmapped".into(),
        transports: vec![Transport::Rdma],
        ..Default::default()
    });
    let requester = TransferEngine::new(EngineConfig {
        selector: Some(orbite::TransportSelectorConfig {
            policy,
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    let peer = requester
        .open_segment(
            source.endpoint(),
            source.source_epoch(),
            Duration::from_secs(3),
        )
        .unwrap();
    let local = requester
        .register_memory(vec![0; 8].into_boxed_slice(), Permission::LocalOnly)
        .unwrap();
    let plain = TransferRequest::new(
        Transport::Unspec,
        local.base_address(),
        remote.base_address(),
        8,
    );
    for (intent, native_intent, priority) in cases {
        let options = TransferOptions {
            intent,
            deadline: Instant::now() + Duration::from_secs(3),
        };
        let request = request_options(plain.clone(), options);
        assert_eq!(
            (request.intent, request.priority, request.deadline),
            (native_intent, priority, Some(options.deadline))
        );
        let outcome = requester
            .submit_and_wait(
                TransferOp::Read,
                &peer,
                std::slice::from_ref(&request),
                Duration::from_secs(3),
            )
            .unwrap();
        assert!(outcome.is_complete());
        let mut bytes = [0; 8];
        local.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes, b"intent!!");
        assert!(
            requester
                .submit_and_wait(
                    TransferOp::Read,
                    &peer,
                    &[request.with_priority(orbite::Priority::Medium)],
                    Duration::from_secs(3)
                )
                .is_err()
        );
        let stats = requester.stats().unwrap();
        assert_eq!((stats.inflight_batches, stats.inflight_bytes), (0, 0));
    }
    assert!(
        requester
            .submit_and_wait(TransferOp::Read, &peer, &[plain], Duration::from_secs(3))
            .is_err()
    );
    requester.unregister_memory(&local).unwrap();
    source.unregister_memory(&remote).unwrap();
}
