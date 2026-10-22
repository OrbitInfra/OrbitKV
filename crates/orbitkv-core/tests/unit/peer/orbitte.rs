use super::*;
use crate::peer::provider::PeerTransport;

#[test]
fn peer_protocol_and_granted_epoch_are_preserved_across_source_restart() {
    let source = TransferEngine::new(EngineConfig::default()).unwrap();
    let endpoint = source.endpoint();
    let granted_epoch = source.source_epoch().to_string();
    let requester = PeerTransport::OrbitTe(Arc::new(
        OrbitTeTransport::new(4096, "127.0.0.1:0").unwrap(),
    ));
    let named_endpoint = format!("{PREFIX}{endpoint}");
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
