use super::*;
use crate::peer::provider::PeerTransport;

#[test]
fn peer_protocol_and_granted_epoch_are_preserved_across_source_restart() {
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let granted_epoch = source.source_epoch().to_string();
    let requester = PeerTransport::OrbitTe(Arc::new(
        OrbitTeTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Tcp,
            #[cfg(feature = "orbitte-rdma")]
            None,
        )
        .unwrap(),
    ));
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
fn registered_read_rejects_selection_and_budgets_before_metadata_io() {
    let requester = OrbitTeTransport::new(
        4096,
        "127.0.0.1:0",
        Transport::Tcp,
        #[cfg(feature = "orbitte-rdma")]
        None,
    )
    .unwrap();
    let endpoint = "orbitte+tcp://127.0.0.1:1";
    let request = TransferRequest {
        transport: Transport::Tcp,
        local_address: 1,
        remote_address: 1,
        length: SLICE_BYTES,
    };
    let cases = [
        (
            vec![TransferRequest {
                transport: Transport::Rdma,
                ..request.clone()
            }],
            "selection",
        ),
        (
            vec![request.clone(); MAX_READ_BYTES / SLICE_BYTES + 1],
            "64 MiB",
        ),
        (
            vec![
                TransferRequest {
                    length: 1,
                    ..request
                };
                MAX_READ_REQUESTS + 1
            ],
            "descriptor budget",
        ),
    ];
    for (requests, expected) in cases {
        assert!(
            matches!(
                requester.read_registered(endpoint, &requester.epoch, &requests, Instant::now() + Duration::from_secs(1)),
                Err(PeerTransferError::Failed(reason)) if reason.contains(expected)
            ),
            "preflight must reject {expected} before connecting"
        );
    }
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
        Some(crate::OrbitTeRdmaConfig::new("not/a/device")),
    )
    .unwrap();
    let local = requester
        .engine
        .register_memory(vec![0xcc; 64].into_boxed_slice(), Permission::LocalOnly)
        .unwrap();
    let requests = [TransferRequest {
        transport: Transport::Rdma,
        local_address: local.base_address(),
        remote_address: source_memory.base_address(),
        length: 64,
    }];
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
    let mut bytes = [0; 64];
    local.read(0, &mut bytes).unwrap();
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
    requester.engine.unregister_memory(&local).unwrap();
    drop(local);
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
            Some(crate::OrbitTeRdmaConfig::new("not/a/device"))
        )
        .is_err()
    );
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let requester = PeerTransport::OrbitTe(Arc::new(
        OrbitTeTransport::new(
            4096,
            "127.0.0.1:0",
            Transport::Rdma,
            Some(crate::OrbitTeRdmaConfig::new("not/a/device")),
        )
        .unwrap(),
    ));
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
