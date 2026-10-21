use super::*;

pub(crate) mod etcd;
pub(crate) mod gate;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
mod p2p_mooncake;

#[test]
fn node_and_cluster_labels_cannot_escape_their_key_prefix() {
    for invalid in ["", "../node", "node/child", "node name", "节点"] {
        assert!(parse_label(invalid).is_err());
    }
    assert!(parse_label(&"n".repeat(129)).is_err());
    assert!(parse_label("gpu-01.host_2").is_ok());
}
