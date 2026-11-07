use super::*;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, result};
use half::{bf16, f16};

#[test]
fn policy_protects_boundaries_opaque_state_and_rejects_corruption() {
    for mode in [StorageCodec::TurboQuant3, StorageCodec::TurboQuant4] {
        for index in 0..8 {
            let format = mode.format(StorageFormat::Attention {
                scalar: Scalar16::Bf16,
                role: AttentionRole::Key,
                head_dim: 128,
                layer_index: index,
                layer_count: 8,
            });
            assert_eq!(format == StorageFormat::Exact, !(2..6).contains(&index));
        }
        assert_eq!(
            mode.format(StorageFormat::Fp8FromBf16),
            StorageFormat::Exact
        );
        assert_eq!(
            mode.format(StorageFormat::Attention {
                scalar: Scalar16::Bf16,
                role: AttentionRole::Key,
                head_dim: 128,
                layer_index: u32::MAX - 1,
                layer_count: u32::MAX,
            }),
            StorageFormat::Exact
        );
    }
    let data = [1u8; 32];
    let mut meta = EncodedSegment {
        version: 1,
        format: StorageFormat::Fp8FromBf16,
        logical_bytes: 64,
        stored_bytes: 32,
        checksum: crc32fast::hash(&data),
    };
    assert!(meta.validate(&data).is_ok());
    assert!(meta.validate(&[2; 32]).is_err());
    meta.logical_bytes = 65;
    assert!(meta.validate(&data).is_err());
    meta.version = 2;
    assert!(meta.validate(&data).is_err());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_fp8_matches_simd_for_every_finite_in_range_bf16_and_fp16_value() {
    let ctx = CudaContext::new(0).expect("GPU qualification needs CUDA");
    let stream = ctx.new_stream().unwrap();
    let mut codec = gpu::GpuCodec::new(&ctx).unwrap();
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        let bits: Vec<u16> = (0..=u16::MAX)
            .filter(|&b| {
                let v = if format == StorageFormat::Fp8FromBf16 {
                    bf16::from_bits(b).to_f32()
                } else {
                    f16::from_bits(b).to_f32()
                };
                v.is_finite() && v.abs() <= 448.0
            })
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let input = stream.clone_htod(&bytes).unwrap();
        let (output, len) = encode_one(
            &mut codec,
            &stream,
            input.device_ptr(&stream).0,
            bytes.len(),
            format,
            64 * 1024 * 1024,
        )
        .unwrap()
        .unwrap();
        let encoded = stream.clone_dtoh(&output).unwrap();
        let mut reference = vec![0; len];
        assert!(cpu::encode(format, &bytes, &mut reference));
        assert_eq!(&encoded[..len], &reference);
        let meta = EncodedSegment {
            version: 1,
            format,
            logical_bytes: bytes.len(),
            stored_bytes: len,
            checksum: crc32fast::hash(&reference),
        };
        let reconstructed = stream.alloc_zeros::<u8>(bytes.len()).unwrap();
        decode_one(
            &mut codec,
            &stream,
            &output,
            reconstructed.device_ptr(&stream).0,
            &meta,
            64 * 1024 * 1024,
        )
        .unwrap();
        let mut reference = vec![0; bytes.len()];
        assert!(cpu::decode(format, &encoded[..len], &mut reference));
        assert_eq!(stream.clone_dtoh(&reconstructed).unwrap(), reference);
        let invalid = stream.clone_htod(&[f16::INFINITY.to_bits(); 32]).unwrap();
        assert!(
            encode_one(
                &mut codec,
                &stream,
                invalid.device_ptr(&stream).0,
                64,
                StorageFormat::Fp8FromFp16,
                1024 * 1024
            )
            .unwrap()
            .is_none()
        );
        assert!(
            encode_one(
                &mut codec,
                &stream,
                input.device_ptr(&stream).0,
                bytes.len(),
                format,
                4096
            )
            .unwrap()
            .is_none()
        );
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_turboquant_three_and_four_bits_preserve_key_norm_and_bound_error() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = gpu::GpuCodec::new(&ctx).unwrap();
    for dim in [32, 64, 128, 256] {
        for bits in [3, 4] {
            for role in [
                AttentionRole::Key,
                AttentionRole::Value,
                AttentionRole::PackedKeyValue,
            ] {
                for scalar in [Scalar16::Bf16, Scalar16::Fp16] {
                    let vectors = if role == AttentionRole::PackedKeyValue {
                        14
                    } else {
                        7
                    };
                    let values: Vec<f32> = (0..dim * vectors)
                        .map(|i| ((i as f32 * 0.1723).sin() + (i as f32 * 0.4281).cos()) * 0.7)
                        .collect();
                    let source: Vec<u16> = values
                        .iter()
                        .map(|&v| {
                            if scalar == Scalar16::Bf16 {
                                bf16::from_f32(v).to_bits()
                            } else {
                                f16::from_f32(v).to_bits()
                            }
                        })
                        .collect();
                    let input = stream.clone_htod(&source).unwrap();
                    let format = StorageFormat::TurboQuant {
                        scalar,
                        role,
                        head_dim: dim,
                        seed: 42,
                        bits,
                    };
                    let (output, len) = encode_one(
                        &mut codec,
                        &stream,
                        input.device_ptr(&stream).0,
                        source.len() * 2,
                        format,
                        1024 * 1024,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(len, 7 * vector_bytes(dim, bits, role));
                    let bytes = stream.clone_dtoh(&output).unwrap();
                    let meta = EncodedSegment {
                        version: 1,
                        format,
                        logical_bytes: source.len() * 2,
                        stored_bytes: len,
                        checksum: crc32fast::hash(&bytes[..len]),
                    };
                    meta.validate(&bytes).unwrap();
                    let target = stream.alloc_zeros::<u16>(source.len()).unwrap();
                    decode_one(
                        &mut codec,
                        &stream,
                        &output,
                        target.device_ptr(&stream).0,
                        &meta,
                        1024 * 1024,
                    )
                    .unwrap();
                    let actual: Vec<f32> = stream
                        .clone_dtoh(&target)
                        .unwrap()
                        .iter()
                        .map(|&b| {
                            if scalar == Scalar16::Bf16 {
                                bf16::from_bits(b).to_f32()
                            } else {
                                f16::from_bits(b).to_f32()
                            }
                        })
                        .collect();
                    for (index, (a, b)) in actual
                        .chunks(dim as usize)
                        .zip(values.chunks(dim as usize))
                        .enumerate()
                    {
                        let norm = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                        let error = a
                            .iter()
                            .zip(b)
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            .sqrt()
                            / norm;
                        assert!(
                            error < if bits == 3 { 0.31 } else { 0.18 },
                            "dim={dim} bits={bits} role={role:?} relative error={error}"
                        );
                        if role == AttentionRole::Key
                            || (role == AttentionRole::PackedKeyValue && index % 2 == 0)
                        {
                            assert!(
                                (a.iter().map(|v| v * v).sum::<f32>().sqrt() / norm - 1.0).abs()
                                    < 0.01
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires optional nvCOMP 5.3 library"]
fn gpu_ans_exact_roundtrip_and_budget() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let mut codec = gpu::GpuCodec::new(&ctx).unwrap();
    let data: Vec<u8> = (0..65536)
        .map(|i| if i % 7 == 0 { (i % 251) as u8 } else { 0 })
        .collect();
    let input = stream.clone_htod(&data).unwrap();
    for format in [
        StorageFormat::Ans,
        StorageFormat::Ans16,
        StorageFormat::AnsFp8,
    ] {
        assert!(
            encode_one(
                &mut codec,
                &stream,
                input.device_ptr(&stream).0,
                data.len(),
                format,
                4096
            )
            .unwrap()
            .is_none()
        );
        let (output, len) = encode_one(
            &mut codec,
            &stream,
            input.device_ptr(&stream).0,
            data.len(),
            format,
            64 * 1024 * 1024,
        )
        .unwrap()
        .unwrap();
        assert!(len < data.len());
        let encoded = stream.clone_dtoh(&output).unwrap();
        let meta = EncodedSegment {
            version: 1,
            format,
            logical_bytes: data.len(),
            stored_bytes: len,
            checksum: crc32fast::hash(&encoded[..len]),
        };
        meta.validate(&encoded).unwrap();
        let target = stream.alloc_zeros::<u8>(data.len()).unwrap();
        decode_one(
            &mut codec,
            &stream,
            &output,
            target.device_ptr(&stream).0,
            &meta,
            64 * 1024 * 1024,
        )
        .unwrap();
        assert_eq!(stream.clone_dtoh(&target).unwrap(), data);
    }
}

// Scalar numerical qualification still needs owned copies: the production API
// deliberately borrows an arena whose contents change on the next batch.
fn encode_one(
    codec: &mut gpu::GpuCodec,
    stream: &Arc<CudaStream>,
    source: u64,
    bytes: usize,
    format: StorageFormat,
    budget: usize,
) -> Result<Option<(CudaSlice<u8>, usize)>, String> {
    // SAFETY: the test caller retains the source CudaSlice through this
    // synchronous codec operation; bytes bounds its initialized device range.
    let batch = unsafe {
        codec.encode_batch(
            stream,
            &[gpu::EncodeInput {
                source,
                bytes,
                format,
            }],
            budget,
        )?
    };
    assert_eq!(batch.processed, 1);
    let Some(output) = &batch.outputs[0] else {
        return Ok(None);
    };
    let copy = stream
        .alloc_zeros::<u8>(output.meta.stored_bytes)
        .map_err(|e| e.to_string())?;
    // SAFETY: the borrowed batch owns output and copy has stored_bytes
    // capacity; both survive the explicit stream synchronization below.
    let copied = unsafe {
        result::memcpy_dtod_async(
            copy.device_ptr(stream).0,
            output.device,
            output.meta.stored_bytes,
            stream.cu_stream(),
        )
    };
    stream.synchronize().unwrap();
    copied.map_err(|e| e.to_string())?;
    let host = stream.clone_dtoh(&copy).map_err(|e| e.to_string())?;
    output.meta.validate(&host)?;
    assert!(output.device.is_multiple_of(4096));
    Ok(Some((copy, output.meta.stored_bytes)))
}

fn decode_one(
    codec: &mut gpu::GpuCodec,
    stream: &Arc<CudaStream>,
    source: &CudaSlice<u8>,
    target: u64,
    meta: &EncodedSegment,
    budget: usize,
) -> Result<(), gpu::DecodeError> {
    // SAFETY: source is borrowed and the test caller retains the disjoint
    // target allocation through the codec's synchronous drain.
    unsafe {
        codec.decode_batch(
            stream,
            &[gpu::DecodeInput {
                source: source.device_ptr(stream).0,
                source_bytes: source.len(),
                target,
                target_bytes: meta.logical_bytes,
                meta,
            }],
            budget,
        )
    }
}

mod batch;
