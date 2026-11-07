use std::sync::OnceLock;

use half::{bf16, f16};
use orbitkv_state::StorageFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Backend {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_encode(input: &[u8], output: &mut [u8], table: &[i32]) -> Option<usize> {
    use std::arch::x86_64::*;
    let end = output.len() / 16 * 16;
    // SAFETY: each iteration reads 32 input bytes, writes 16 output bytes,
    // and gathers indices in the complete 65536-entry table.
    unsafe {
        for i in (0..end).step_by(16) {
            let indices =
                _mm512_cvtepu16_epi32(_mm256_loadu_si256(input.as_ptr().add(i * 2).cast()));
            let values = _mm512_i32gather_epi32::<4>(indices, table.as_ptr());
            if _mm512_cmpgt_epi32_mask(values, _mm512_set1_epi32(255)) != 0 {
                return None;
            }
            _mm_storeu_si128(
                output.as_mut_ptr().add(i).cast(),
                _mm512_cvtepi32_epi8(values),
            );
        }
    }
    Some(end)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_decode(input: &[u8], output: &mut [u8], table: &[i32; 256]) -> usize {
    use std::arch::x86_64::*;
    let end = input.len() / 16 * 16;
    // SAFETY: each iteration reads 16 input bytes and writes 32 output bytes.
    unsafe {
        for i in (0..end).step_by(16) {
            let indices = _mm512_cvtepu8_epi32(_mm_loadu_si128(input.as_ptr().add(i).cast()));
            let values = _mm512_i32gather_epi32::<4>(indices, table.as_ptr());
            _mm256_storeu_si256(
                output.as_mut_ptr().add(i * 2).cast(),
                _mm512_cvtepi32_epi16(values),
            );
        }
    }
    end
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_encode(input: &[u8], output: &mut [u8], table: &[i32]) -> Option<usize> {
    use std::arch::x86_64::*;
    let end = output.len() / 8 * 8;
    // SAFETY: each iteration reads 16 input bytes, writes 8 output bytes,
    // and gathers indices in the complete 65536-entry table.
    unsafe {
        for i in (0..end).step_by(8) {
            let indices = _mm256_cvtepu16_epi32(_mm_loadu_si128(input.as_ptr().add(i * 2).cast()));
            let values = _mm256_i32gather_epi32::<4>(table.as_ptr(), indices);
            if _mm256_movemask_epi8(_mm256_cmpgt_epi32(values, _mm256_set1_epi32(255))) != 0 {
                return None;
            }
            let words = _mm_packus_epi32(
                _mm256_castsi256_si128(values),
                _mm256_extracti128_si256::<1>(values),
            );
            _mm_storel_epi64(
                output.as_mut_ptr().add(i).cast(),
                _mm_packus_epi16(words, words),
            );
        }
    }
    Some(end)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_decode(input: &[u8], output: &mut [u8], table: &[i32; 256]) -> usize {
    use std::arch::x86_64::*;
    let end = input.len() / 8 * 8;
    // SAFETY: each iteration reads 8 input bytes and writes 16 output bytes.
    unsafe {
        for i in (0..end).step_by(8) {
            let indices = _mm256_cvtepu8_epi32(_mm_loadl_epi64(input.as_ptr().add(i).cast()));
            let values = _mm256_i32gather_epi32::<4>(table.as_ptr(), indices);
            let words = _mm_packus_epi32(
                _mm256_castsi256_si128(values),
                _mm256_extracti128_si256::<1>(values),
            );
            _mm_storeu_si128(output.as_mut_ptr().add(i * 2).cast(), words);
        }
    }
    end
}

struct Tables {
    encode: Box<[i32]>,
    decode: [i32; 256],
}

// E4M3FN values, including subnormals. The sole positive NaN is 0x7f.
fn value(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 15;
    let mantissa = bits & 7;
    let magnitude = if exponent == 0 {
        mantissa as f32 / 512.0
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    };
    if bits & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn tables(format: StorageFormat) -> Option<&'static Tables> {
    static BF16: OnceLock<Tables> = OnceLock::new();
    static FP16: OnceLock<Tables> = OnceLock::new();
    let table = match format {
        StorageFormat::Fp8FromBf16 => &BF16,
        StorageFormat::Fp8FromFp16 => &FP16,
        _ => return None,
    };
    Some(table.get_or_init(|| {
        let bf = format == StorageFormat::Fp8FromBf16;
        let levels: Vec<_> = (0..127).map(value).collect();
        let encode = (0..=u16::MAX)
            .map(|bits| {
                let input = if bf {
                    bf16::from_bits(bits).to_f32()
                } else {
                    f16::from_bits(bits).to_f32()
                };
                let magnitude = input.abs();
                // Preserve nonfinite values and outliers exactly by declining the object.
                if !input.is_finite() || magnitude > 448.0 {
                    return 256;
                }
                let upper = levels.partition_point(|v| *v < magnitude);
                let lower = upper.saturating_sub(1);
                let down = magnitude - levels[lower];
                let up = levels[upper] - magnitude;
                let index = if down < up || (down == up && lower.is_multiple_of(2)) {
                    lower
                } else {
                    upper
                };
                (index as u16 | ((bits >> 8) & 128)) as i32
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let decode = std::array::from_fn(|bits| {
            (if bf {
                bf16::from_f32(value(bits as u8)).to_bits()
            } else {
                f16::from_f32(value(bits as u8)).to_bits()
            }) as i32
        });
        Tables { encode, decode }
    }))
}

pub(crate) fn encode(format: StorageFormat, input: &[u8], output: &mut [u8]) -> bool {
    // SAFETY: detection includes OS support for the backend's vector registers.
    unsafe { encode_with_backend(format, input, output, Backend::detect()) }
}

// SAFETY: the caller must verify support for the selected backend.
unsafe fn encode_with_backend(
    format: StorageFormat,
    input: &[u8],
    output: &mut [u8],
    backend: Backend,
) -> bool {
    if output.len().checked_mul(2) != Some(input.len()) {
        return false;
    }
    let Some(table) = tables(format) else {
        return false;
    };
    let start = match backend {
        Backend::Scalar => Some(0),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller verified AVX2 support; the length check above bounds
        // each vector read/write, and u16 indices fit the encoding table.
        Backend::Avx2 => unsafe { avx2_encode(input, output, &table.encode) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller verified AVX-512 support; the checked disjoint slices
        // and full u16 encoding table cover every vector access.
        Backend::Avx512 => unsafe { avx512_encode(input, output, &table.encode) },
    };
    let Some(start) = start else {
        return false;
    };
    for (value, out) in input[start * 2..].chunks_exact(2).zip(&mut output[start..]) {
        let encoded = table.encode[u16::from_le_bytes([value[0], value[1]]) as usize];
        if encoded > 255 {
            return false;
        }
        *out = encoded as u8;
    }
    true
}

pub(crate) fn decode(format: StorageFormat, input: &[u8], output: &mut [u8]) -> bool {
    // SAFETY: detection includes OS support for the backend's vector registers.
    unsafe { decode_with_backend(format, input, output, Backend::detect()) }
}

// SAFETY: the caller must verify support for the selected backend.
unsafe fn decode_with_backend(
    format: StorageFormat,
    input: &[u8],
    output: &mut [u8],
    backend: Backend,
) -> bool {
    if input.len().checked_mul(2) != Some(output.len()) {
        return false;
    }
    let Some(table) = tables(format) else {
        return false;
    };
    let start = match backend {
        Backend::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller verified AVX2 support; output is twice input's length
        // and every u8 input indexes the full 256-entry decoding table.
        Backend::Avx2 => unsafe { avx2_decode(input, output, &table.decode) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the caller verified AVX-512 support; the disjoint slices have
        // checked lengths and all u8 indices fit the decoding table.
        Backend::Avx512 => unsafe { avx512_decode(input, output, &table.decode) },
    };
    for (&value, out) in input[start..]
        .iter()
        .zip(output[start * 2..].chunks_exact_mut(2))
    {
        out.copy_from_slice(&(table.decode[value as usize] as u16).to_le_bytes());
    }
    true
}

#[cfg(test)]
#[path = "../../tests/unit/codec/cpu.rs"]
mod tests;
