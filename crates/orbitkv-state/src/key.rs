use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::{StateComponent, StateFormat};

pub type Digest = [u8; 32];

/// One stored slot's geometry, independent of GPU address and pool capacity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct StorageSlot {
    pub format: crate::StorageFormat,
    pub layer: String,
    pub group: u32,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub segment_bytes: usize,
    pub padded_block_bytes: usize,
    pub split: bool,
}

/// Bind the adapter identity to the representation the manager actually stores.
/// Called once when registration seals, never while probing individual blocks.
#[cfg_attr(
    not(test),
    expect(
        clippy::expect_used,
        reason = "derived serialization contains only strings, integers, booleans and sequences; Vec writes are infallible"
    )
)]
pub fn storage_namespace(identity: &str, page_first: bool, mut slots: Vec<StorageSlot>) -> String {
    slots.sort_unstable();
    slots.dedup();
    let mut digest = Sha256::new();
    // v2 requires peers to preserve encoded-residency metadata.
    digest.update(b"orbitkv.storage-identity.v2\0");
    // Only strings, booleans and integers are serialized; serialization cannot fail.
    digest.update(serde_json::to_vec(&(identity, page_first, slots)).expect("storage identity"));
    format!("orbitkv:v2:{:x}", digest.finalize())
}

/// Isolate cached prefix state after an explicit engine-owned reset.
pub fn generation_hash(hash: &[u8], generation: &[u8; 16]) -> Digest {
    let mut digest = Sha256::new();
    digest.update(b"orbitkv.prefix-generation.v1\0");
    digest.update(generation);
    digest.update((hash.len() as u64).to_le_bytes());
    digest.update(hash);
    digest.finalize().into()
}

pub fn is_storage_namespace(namespace: &str) -> bool {
    namespace.strip_prefix("orbitkv:v2:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContractError {
    #[error("token range end {end} precedes start {start}")]
    InvalidTokenRange { start: u64, end: u64 },
}

/// Half-open logical token interval represented by one state object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenRange {
    pub start: u64,
    pub end: u64,
}

impl TokenRange {
    pub fn new(start: u64, end: u64) -> Result<Self, ContractError> {
        if end < start {
            return Err(ContractError::InvalidTokenRange { start, end });
        }
        Ok(Self { start, end })
    }

    pub fn len(self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Logical recovery evidence. Adapters must supply real token coverage before
/// using this descriptor to prove a recovery boundary.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateDescriptor {
    pub content: Digest,
    pub span: TokenRange,
    pub component: StateComponent,
    pub format: StateFormat,
}

/// Materialized cache key used by DRAM, SSD and the peer directory.
///
/// `namespace` binds immutable model artifacts, computation and representation.
/// `hash` is the versioned encoding of the engine-native chained prefix hash and
/// cache group. A key match alone does not prove a multi-component boundary.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateKey {
    pub namespace: String,
    pub hash: Vec<u8>,
}

impl StateKey {
    pub fn new(namespace: String, hash: Vec<u8>) -> Self {
        Self { namespace, hash }
    }

    pub fn estimated_size(&self) -> u64 {
        (self.namespace.capacity() + self.hash.capacity() + std::mem::size_of::<Self>()) as u64
    }
}

/// Domain and length-framed engine-native content hash, including group zero.
/// There is no legacy/raw-hash key form.
pub fn group_hash(hash: &[u8], group_id: u32) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(16 + hash.len());
    encoded.extend_from_slice(b"OKS\x01");
    encoded.extend_from_slice(&group_id.to_le_bytes());
    encoded.extend_from_slice(&(hash.len() as u64).to_le_bytes());
    encoded.extend_from_slice(hash);
    encoded
}

#[cfg(test)]
#[path = "../tests/unit/key.rs"]
mod tests;
