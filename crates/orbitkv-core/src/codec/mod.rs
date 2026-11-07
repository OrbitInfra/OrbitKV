use orbitkv_state::{AttentionRole, Scalar16, StorageFormat};
use serde::{Deserialize, Serialize};

pub(crate) mod ans;
pub(crate) mod cpu;
pub(crate) mod gpu;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum StorageCodec {
    #[default]
    None,
    Ans,
    Fp8,
    TurboQuant4,
    TurboQuant3,
}

impl std::str::FromStr for StorageCodec {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "ans" => Ok(Self::Ans),
            "fp8" => Ok(Self::Fp8),
            "turboquant-4" => Ok(Self::TurboQuant4),
            "turboquant-3" => Ok(Self::TurboQuant3),
            _ => Err("storage codec must be none, ans, fp8, turboquant-4 or turboquant-3".into()),
        }
    }
}

impl StorageCodec {
    pub(crate) fn format(self, requested: StorageFormat) -> StorageFormat {
        match self {
            Self::None => StorageFormat::Exact,
            Self::Ans => match requested {
                StorageFormat::Fp8FromBf16
                | StorageFormat::Fp8FromFp16
                | StorageFormat::Attention { .. } => StorageFormat::Ans16,
                StorageFormat::Fp8Native => StorageFormat::AnsFp8,
                _ => StorageFormat::Ans,
            },
            Self::Fp8 => match requested {
                StorageFormat::Attention {
                    scalar: Scalar16::Bf16,
                    ..
                } => StorageFormat::Fp8FromBf16,
                StorageFormat::Attention {
                    scalar: Scalar16::Fp16,
                    ..
                } => StorageFormat::Fp8FromFp16,
                StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16 => requested,
                _ => StorageFormat::Exact,
            },
            Self::TurboQuant4 | Self::TurboQuant3 => match requested {
                StorageFormat::Attention {
                    scalar,
                    role,
                    head_dim,
                    layer_index,
                    layer_count,
                } if (32..=256).contains(&head_dim)
                    && head_dim.is_power_of_two()
                    && layer_index >= 2
                    && layer_index < layer_count.saturating_sub(2) =>
                {
                    StorageFormat::TurboQuant {
                        scalar,
                        role,
                        head_dim,
                        seed: 42u32.wrapping_add(layer_index.wrapping_mul(1337)),
                        bits: if self == Self::TurboQuant4 { 4 } else { 3 },
                    }
                }
                _ => StorageFormat::Exact,
            },
        }
    }
}

/// Versioned segment representation travels unchanged through memory, disk and peers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncodedSegment {
    pub version: u32,
    pub format: StorageFormat,
    pub logical_bytes: usize,
    pub stored_bytes: usize,
    pub checksum: u32,
}

impl EncodedSegment {
    /// Validate representation bounds without accessing a host payload (e.g. GDS).
    pub(crate) fn validate_metadata(&self, available_bytes: usize) -> Result<(), String> {
        if self.version != 1
            || self.logical_bytes == 0
            || self.stored_bytes == 0
            || self.logical_bytes > isize::MAX as usize
            || self.stored_bytes > isize::MAX as usize
            || self.stored_bytes > available_bytes
            || (self.format != StorageFormat::Exact
                && (self.logical_bytes > 16 * 1024 * 1024 || self.stored_bytes > 32 * 1024 * 1024))
        {
            return Err("invalid encoded segment bounds/version".into());
        }
        match self.format {
            StorageFormat::Exact if self.logical_bytes == self.stored_bytes => {}
            StorageFormat::Ans | StorageFormat::AnsFp8 => {}
            StorageFormat::Ans16 if self.logical_bytes.is_multiple_of(2) => {}
            StorageFormat::Fp8FromBf16 | StorageFormat::Fp8FromFp16
                if self.stored_bytes.checked_mul(2) == Some(self.logical_bytes) => {}
            StorageFormat::TurboQuant {
                head_dim,
                role,
                bits,
                ..
            } if turboquant_bytes(self.logical_bytes, head_dim, bits, role)
                == Some(self.stored_bytes) => {}
            _ => return Err("invalid encoded segment geometry".into()),
        }
        Ok(())
    }

    pub(crate) fn validate(&self, bytes: &[u8]) -> Result<(), String> {
        self.validate_metadata(bytes.len())?;
        if crc32fast::hash(&bytes[..self.stored_bytes]) != self.checksum {
            return Err("encoded segment checksum/size mismatch".into());
        }
        Ok(())
    }
}

pub(crate) fn turboquant_bytes(
    logical: usize,
    dim: u32,
    bits: u8,
    role: AttentionRole,
) -> Option<usize> {
    if !(32..=256).contains(&dim)
        || !dim.is_power_of_two()
        || !matches!(bits, 3 | 4)
        || role == AttentionRole::KeyValue
    {
        return None;
    }
    let source_width = dim as usize
        * if role == AttentionRole::PackedKeyValue {
            4
        } else {
            2
        };
    if logical == 0 || !logical.is_multiple_of(source_width) {
        return None;
    }
    (logical / source_width).checked_mul(vector_bytes(dim, bits, role)?)
}

fn vector_bytes(dim: u32, bits: u8, role: AttentionRole) -> Option<usize> {
    let packed = (dim as usize * bits as usize).div_ceil(8);
    Some(match role {
        AttentionRole::Key => packed + 2,
        AttentionRole::Value => packed + 4,
        AttentionRole::PackedKeyValue => 2 * packed + 6,
        AttentionRole::KeyValue => return None,
    })
}

pub(crate) fn segment_format(format: StorageFormat, index: usize) -> StorageFormat {
    match format {
        StorageFormat::TurboQuant {
            scalar,
            role: AttentionRole::KeyValue,
            head_dim,
            seed,
            bits,
        } => StorageFormat::TurboQuant {
            scalar,
            role: if index == 0 {
                AttentionRole::Key
            } else {
                AttentionRole::Value
            },
            head_dim,
            seed,
            bits,
        },
        _ => format,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/mod.rs"]
mod tests;
