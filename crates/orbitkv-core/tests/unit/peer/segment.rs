use super::*;

pub(in crate::peer) fn with_epoch(
    segments: &PeerSegments,
    peer: &PeerIdentity,
    epoch: &str,
) -> Arc<PeerIdentity> {
    segments
        .get(peer.identity.member.clone(), &peer.identity.endpoint, epoch)
        .unwrap()
}

pub(in crate::peer) fn fence(segments: &PeerSegments) {
    segments.membership.fence();
}
