use crate::StateKey;

/// Discovery is bounded, positive evidence, never a complete cluster inventory.
pub const DISCOVERY_MAX_KEYS: usize = 128;
pub const DISCOVERY_MAX_BYTES: usize = 64 * 1024;
pub const DISCOVERY_MAX_REPLICAS_PER_MEDIUM: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryCoverage {
    /// Every expected peer supplied a synchronized, recent inventory receipt for this cut.
    CompleteAtWatermarks,
    /// An admitted view can still supply candidates. An empty result is inconclusive.
    PartialHints,
    /// No usable discovery cut for this membership or namespace.
    Unavailable,
}

#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CacheOwner {
    pub endpoint: String,
    pub incarnation: uuid::Uuid,
}

#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ReplicaMedium {
    #[default]
    Unknown,
    Dram,
    Ssd,
    Hbm,
}

#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ReplicaRepresentation {
    #[default]
    Unknown,
    Raw,
    Ans,
    Fp8,
    TurboQuant,
    Mixed,
}

impl From<crate::StorageFormat> for ReplicaRepresentation {
    fn from(format: crate::StorageFormat) -> Self {
        match format {
            crate::StorageFormat::Ans
            | crate::StorageFormat::Ans16
            | crate::StorageFormat::AnsFp8 => Self::Ans,
            crate::StorageFormat::Fp8FromBf16 | crate::StorageFormat::Fp8FromFp16 => Self::Fp8,
            crate::StorageFormat::TurboQuant { .. } => Self::TurboQuant,
            _ => Self::Raw,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ReplicaMetadata {
    pub medium: ReplicaMedium,
    pub representation: ReplicaRepresentation,
    pub stored_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaLocation {
    pub owner: CacheOwner,
    /// Sequence of the insertion in this owner's current runtime inventory.
    pub sequence: u64,
    pub metadata: ReplicaMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockCandidates {
    pub key: StateKey,
    pub replicas: Vec<ReplicaLocation>,
    pub coverage: DiscoveryCoverage,
}

#[cfg(test)]
#[path = "../tests/unit/discovery.rs"]
mod tests;

pub fn validate_discovery_query(namespace: &str, hashes: &[Vec<u8>]) -> Result<(), &'static str> {
    if namespace.is_empty() || hashes.is_empty() || hashes.len() > DISCOVERY_MAX_KEYS {
        return Err("invalid discovery key count or namespace");
    }
    let mut bytes = namespace.len();
    for hash in hashes {
        if hash.is_empty() {
            return Err("empty discovery hash");
        }
        bytes = bytes
            .checked_add(hash.len())
            .ok_or("discovery size overflow")?;
    }
    if bytes > DISCOVERY_MAX_BYTES {
        return Err("discovery query byte limit exceeded");
    }
    Ok(())
}
