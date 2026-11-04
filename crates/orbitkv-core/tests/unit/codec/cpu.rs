use super::*;

fn supported_backends() -> Vec<Backend> {
    let backends = vec![Backend::Scalar];
    #[cfg(target_arch = "x86_64")]
    let backends = {
        let mut backends = backends;
        if std::arch::is_x86_feature_detected!("avx2") {
            backends.push(Backend::Avx2);
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            backends.push(Backend::Avx512);
        }
        backends
    };
    backends
}

fn scalar_value(format: StorageFormat, bits: u16) -> f32 {
    match format {
        StorageFormat::Fp8FromBf16 => bf16::from_bits(bits).to_f32(),
        StorageFormat::Fp8FromFp16 => f16::from_bits(bits).to_f32(),
        _ => unreachable!(),
    }
}

fn scalar_bits(format: StorageFormat, value: f32) -> u16 {
    match format {
        StorageFormat::Fp8FromBf16 => bf16::from_f32(value).to_bits(),
        StorageFormat::Fp8FromFp16 => f16::from_f32(value).to_bits(),
        _ => unreachable!(),
    }
}

#[test]
fn runtime_dispatch_prefers_the_widest_available_backend() {
    let backends = supported_backends();
    assert_eq!(Backend::detect(), *backends.last().unwrap());
    eprintln!(
        "CPU codec backends: {backends:?}; selected {:?}",
        Backend::detect()
    );
}

#[test]
fn all_backends_match_scalar_for_every_finite_in_range_16bit_value() {
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        let input: Vec<u8> = (0..=u16::MAX)
            .filter(|&bits| {
                let value = scalar_value(format, bits);
                value.is_finite() && value.abs() <= 448.0
            })
            .flat_map(u16::to_le_bytes)
            .collect();
        let mut expected = vec![0; input.len() / 2];
        // SAFETY: scalar needs no target features; other paths are detected above.
        assert!(unsafe { encode_with_backend(format, &input, &mut expected, Backend::Scalar) });
        for backend in supported_backends() {
            let mut encoded = vec![0xa5; expected.len()];
            assert!(unsafe { encode_with_backend(format, &input, &mut encoded, backend) });
            assert_eq!(encoded, expected, "{format:?} {backend:?}");
        }
        let mut dispatched = vec![0xa5; expected.len()];
        assert!(encode(format, &input, &mut dispatched));
        assert_eq!(dispatched, expected);

        // Include both FP8 NaN encodings and signed zero in bitwise decode checks.
        let input: Vec<u8> = (0..=u8::MAX).collect();
        let mut expected = vec![0; input.len() * 2];
        assert!(unsafe { decode_with_backend(format, &input, &mut expected, Backend::Scalar) });
        for backend in supported_backends() {
            let mut decoded = vec![0xa5; expected.len()];
            assert!(unsafe { decode_with_backend(format, &input, &mut decoded, backend) });
            assert_eq!(decoded, expected, "{format:?} {backend:?}");
        }
        let mut dispatched = vec![0xa5; expected.len()];
        assert!(decode(format, &input, &mut dispatched));
        assert_eq!(dispatched, expected);
    }
}

#[test]
fn all_backends_decline_every_nonfinite_and_out_of_range_16bit_value() {
    let backends = supported_backends();
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        for bits in 0..=u16::MAX {
            let value = scalar_value(format, bits);
            if value.is_finite() && value.abs() <= 448.0 {
                continue;
            }
            // A full vector exercises rejection in the SIMD kernel, not its tail.
            let mut input = [0; 32];
            for lane in input.chunks_exact_mut(2) {
                lane.copy_from_slice(&bits.to_le_bytes());
            }
            for &backend in &backends {
                // SAFETY: supported_backends performs runtime ISA detection.
                assert!(
                    !unsafe { encode_with_backend(format, &input, &mut [0; 16], backend) },
                    "{format:?} {backend:?} {bits:04x}"
                );
            }
        }
        // A lone invalid lane must decline the entire segment, including the
        // scalar tail after two AVX-512 vectors. Source bytes remain available
        // to the caller's raw fallback; partial encoded output is not consumed.
        for invalid in [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            512.0,
            -512.0,
        ] {
            for position in 0..33 {
                let mut input = [0; 66];
                input[position * 2..position * 2 + 2]
                    .copy_from_slice(&scalar_bits(format, invalid).to_le_bytes());
                for &backend in &backends {
                    assert!(
                        !unsafe { encode_with_backend(format, &input, &mut [0; 33], backend) },
                        "{format:?} {backend:?} invalid lane {position}"
                    );
                }
            }
        }
    }
}

#[test]
fn all_backends_handle_unaligned_tails_and_reject_invalid_lengths() {
    let backends = supported_backends();
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        for len in (0..66).chain([127, 128, 129]) {
            let mut input = vec![0xa5; len * 2 + 8];
            for (i, lane) in input[3..3 + len * 2].chunks_exact_mut(2).enumerate() {
                let value = if i % 11 == 0 {
                    -0.0
                } else {
                    (i as f32 % 97.0 - 48.0) / 8.0
                };
                lane.copy_from_slice(&scalar_bits(format, value).to_le_bytes());
            }
            let source = &input[3..3 + len * 2];
            let mut expected = vec![0; len];
            // SAFETY: only scalar or runtime-supported SIMD backends are selected.
            assert!(unsafe { encode_with_backend(format, source, &mut expected, Backend::Scalar) });
            let mut reconstructed = vec![0; len * 2];
            assert!(unsafe {
                decode_with_backend(format, &expected, &mut reconstructed, Backend::Scalar)
            });
            for &backend in &backends {
                let mut encoded = vec![0xa5; len + 12];
                assert!(unsafe {
                    encode_with_backend(format, source, &mut encoded[5..5 + len], backend)
                });
                assert_eq!(
                    &encoded[5..5 + len],
                    expected,
                    "{format:?} {backend:?} len={len}"
                );
                assert!(
                    encoded[..5]
                        .iter()
                        .chain(&encoded[5 + len..])
                        .all(|&b| b == 0xa5)
                );
                let mut decoded = vec![0xa5; len * 2 + 8];
                assert!(unsafe {
                    decode_with_backend(
                        format,
                        &encoded[5..5 + len],
                        &mut decoded[3..3 + len * 2],
                        backend,
                    )
                });
                assert_eq!(
                    &decoded[3..3 + len * 2],
                    reconstructed,
                    "{format:?} {backend:?} len={len}"
                );
                assert!(
                    decoded[..3]
                        .iter()
                        .chain(&decoded[3 + len * 2..])
                        .all(|&b| b == 0xa5)
                );
            }
        }
        for &backend in &backends {
            for (input_len, output_len) in [(0, 1), (1, 0), (3, 1), (2, 2)] {
                let mut output = vec![0xa5; output_len];
                assert!(!unsafe {
                    encode_with_backend(format, &vec![0; input_len], &mut output, backend)
                });
                assert!(output.iter().all(|&b| b == 0xa5));
            }
            for (input_len, output_len) in [(0, 1), (1, 0), (1, 1), (3, 5)] {
                let mut output = vec![0xa5; output_len];
                assert!(!unsafe {
                    decode_with_backend(format, &vec![0; input_len], &mut output, backend)
                });
                assert!(output.iter().all(|&b| b == 0xa5));
            }
        }
    }
}

#[test]
fn rounding_subnormals_signed_zero_and_reencoding() {
    for format in [StorageFormat::Fp8FromBf16, StorageFormat::Fp8FromFp16] {
        let to_bits = |x| {
            if format == StorageFormat::Fp8FromBf16 {
                bf16::from_f32(x).to_bits()
            } else {
                f16::from_f32(x).to_bits()
            }
        };
        for (x, expected) in [
            (0.0, 0),
            (-0.0, 128),
            (1.0625, 0x38),
            (1.1875, 0x3a),
            (448.0, 0x7e),
            (1.0 / 512.0, 1),
            (1.0 / 1024.0, 0),
        ] {
            let mut output = [0];
            assert!(encode(format, &to_bits(x).to_le_bytes(), &mut output));
            assert_eq!(output, [expected]);
        }
        for bits in 0..=255u8 {
            if bits & 127 == 127 {
                continue;
            }
            let mut reconstructed = [0; 2];
            assert!(decode(format, &[bits], &mut reconstructed));
            let mut encoded = [0];
            assert!(encode(format, &reconstructed, &mut encoded));
            assert_eq!(encoded, [bits]);
        }
        for x in [f32::INFINITY, f32::NAN, 1024.0] {
            assert!(!encode(format, &to_bits(x).to_le_bytes(), &mut [0]));
        }
    }
}
