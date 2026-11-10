use sha2::{Digest, Sha256};

use crate::StateKey;
use uuid::Uuid;

pub const INVENTORY_BATCH_BYTES: usize = 512 * 1024;
pub const INVENTORY_BATCH_RECORDS: usize = 1024;
pub const INVENTORY_OPEN_MAX_BYTES: usize = 64 * 1024;
pub const INVENTORY_SCOPE_MAX_NAMESPACES: usize = 256;
pub const INVENTORY_STREAM_PROTOCOL: &str = "orbitkv/inventory-stream/v5";
const INVENTORY_SCOPE_DOMAIN: &[u8] = b"orbitkv/inventory-scope/v1\0";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "namespaces")]
pub enum InventoryScope {
    AllNamespaces,
    ExactNamespaces(Vec<String>),
}

impl InventoryScope {
    pub fn exact(namespaces: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut namespaces = namespaces.into_iter().collect::<Vec<_>>();
        if namespaces.iter().any(String::is_empty) {
            return Err("inventory scope contains an empty namespace".into());
        }
        namespaces.sort_unstable();
        namespaces.dedup();
        if namespaces.len() > INVENTORY_SCOPE_MAX_NAMESPACES {
            return Err(format!(
                "inventory scope exceeds {INVENTORY_SCOPE_MAX_NAMESPACES} namespaces"
            ));
        }
        let scope = Self::ExactNamespaces(namespaces);
        if scope.canonical_bytes().len() > INVENTORY_OPEN_MAX_BYTES {
            return Err(format!(
                "inventory scope descriptor exceeds {INVENTORY_OPEN_MAX_BYTES} bytes"
            ));
        }
        Ok(scope)
    }

    pub fn contains(&self, namespace: &str) -> bool {
        match self {
            Self::AllNamespaces => true,
            Self::ExactNamespaces(namespaces) => namespaces
                .binary_search_by(|candidate| candidate.as_str().cmp(namespace))
                .is_ok(),
        }
    }

    pub fn namespaces(&self) -> Option<&[String]> {
        match self {
            Self::AllNamespaces => None,
            Self::ExactNamespaces(namespaces) => Some(namespaces),
        }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(INVENTORY_SCOPE_DOMAIN.len() + 8);
        encoded.extend_from_slice(INVENTORY_SCOPE_DOMAIN);
        match self {
            Self::AllNamespaces => encoded.push(0),
            Self::ExactNamespaces(namespaces) => {
                encoded.push(1);
                encoded.extend_from_slice(&(namespaces.len() as u32).to_be_bytes());
                for namespace in namespaces {
                    encoded.extend_from_slice(&(namespace.len() as u32).to_be_bytes());
                    encoded.extend_from_slice(namespace.as_bytes());
                }
            }
        }
        encoded
    }

    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::AllNamespaces => "all_namespaces",
            Self::ExactNamespaces(namespaces) if namespaces.is_empty() => "no_namespaces",
            Self::ExactNamespaces(_) => "exact_namespaces",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryFence {
    pub protocol: String,
    pub cluster_uuid: Uuid,
    pub source_node_epoch: u64,
    pub source_incarnation: Uuid,
    pub inventory_sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryRecord {
    pub key: StateKey,
    pub sequence: u64,
    pub present: bool,
    /// Required for residency advertisements; authorization-only records may
    /// omit it because the authoritative owner checks its live inventory.
    pub metadata: Option<crate::ReplicaMetadata>,
}

impl InventoryRecord {
    pub fn estimated_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.key.namespace.len() + self.key.hash.len()
    }
}

#[cfg(test)]
#[path = "../tests/unit/inventory.rs"]
mod tests;
