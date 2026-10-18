//! CPU FP8 conversion only: production backends, an independent oracle, then timing.
//! No GPU runtime or Manager is initialized by this executable.

#[cfg_attr(
    test,
    allow(
        dead_code,
        reason = "Cargo checks this harness-free benchmark with cfg(test), leaving included CPU unit helpers unused"
    )
)]
mod cpu {
    // Keep forced-backend access inside the production module; no public API or
    // copied conversion implementation is needed for this benchmark.
    include!("../../../crates/orbitkv-core/src/codec/cpu.rs");

    use std::hint::black_box;
    use std::time::{Duration, Instant};

    const PATHS: [&str; 4] = ["scalar", "avx2", "avx512", "auto"];
    const SIZES: [usize; 3] = [4096, 256 * 1024, 16 * 1024 * 1024];
    const FORMATS: [(&str, StorageFormat); 2] = [
        ("bf16", StorageFormat::Fp8FromBf16),
        ("fp16", StorageFormat::Fp8FromFp16),
    ];

    // Outer None means unsupported hardware; inner None selects production dispatch.
    fn selected(name: &str) -> Option<Option<Backend>> {
        match name {
            "scalar" => Some(Some(Backend::Scalar)),
            "auto" => Some(None),
            #[cfg(target_arch = "x86_64")]
            "avx2" if std::arch::is_x86_feature_detected!("avx2") => Some(Some(Backend::Avx2)),
            #[cfg(target_arch = "x86_64")]
            "avx512" if std::arch::is_x86_feature_detected!("avx512f") => {
                Some(Some(Backend::Avx512))
            }
            _ => None,
        }
    }

    fn convert(
        backend: Option<Backend>,
        encoding: bool,
        format: StorageFormat,
        input: &[u8],
        output: &mut [u8],
    ) -> bool {
        match (backend, encoding) {
            // SAFETY: forced backends come only from selected(), which checks OS/ISA support.
            (Some(backend), true) => unsafe { encode_with_backend(format, input, output, backend) },
            (Some(backend), false) => unsafe {
                decode_with_backend(format, input, output, backend)
            },
            (None, true) => encode(format, input, output),
            (None, false) => decode(format, input, output),
        }
    }

    // Construct IEEE f32 bits independently of the production lookup-table builder.
    fn oracle_value(code: u8) -> f32 {
        let exponent = (code >> 3) & 15;
        let mantissa = code & 7;
        let sign = u32::from(code & 128) << 24;
        if exponent == 0 {
            let magnitude = f32::from(mantissa) / 512.0;
            f32::from_bits(magnitude.to_bits() | sign)
        } else if code & 127 == 127 {
            f32::from_bits(sign | 0x7fc0_0000)
        } else {
            f32::from_bits(sign | ((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
        }
    }

    struct Reference {
        valid: Vec<(u16, u8)>,
        rejected: Vec<u16>,
        decoded: [u16; 256],
    }

    fn reference(format: StorageFormat) -> Reference {
        let bf = format == StorageFormat::Fp8FromBf16;
        let levels: Vec<f64> = (0..127).map(|code| f64::from(oracle_value(code))).collect();
        let mut valid = Vec::new();
        let mut rejected = Vec::new();
        for bits in 0..=u16::MAX {
            let input = if bf {
                bf16::from_bits(bits).to_f32()
            } else {
                f16::from_bits(bits).to_f32()
            };
            if !input.is_finite() || input.abs() > 448.0 {
                rejected.push(bits);
                continue;
            }
            // Brute-force nearest neighbour, with even-code tie breaking, rather
            // than the production table's partition-point implementation.
            let magnitude = f64::from(input.abs());
            let mut best = 0;
            let mut distance = magnitude;
            for (code, level) in levels.iter().enumerate().skip(1) {
                let candidate = (magnitude - level).abs();
                if candidate < distance || (candidate == distance && code % 2 == 0) {
                    best = code;
                    distance = candidate;
                }
            }
            valid.push((bits, best as u8 | ((bits >> 8) as u8 & 128)));
        }
        let decoded = std::array::from_fn(|code| {
            let value = oracle_value(code as u8);
            if bf {
                bf16::from_f32(value).to_bits()
            } else {
                f16::from_f32(value).to_bits()
            }
        });
        Reference {
            valid,
            rejected,
            decoded,
        }
    }

    fn equal_bytes(actual: &[u8], expected: &[u8], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: byte length");
        if actual != expected {
            let index = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{context}: byte {index}: {} != {}",
                actual[index], expected[index]
            );
        }
    }

    fn correctness(format: StorageFormat, reference: &Reference) {
        let input: Vec<_> = reference
            .valid
            .iter()
            .flat_map(|(bits, _)| bits.to_le_bytes())
            .collect();
        let expected: Vec<_> = reference.valid.iter().map(|(_, code)| *code).collect();
        let codes: Vec<_> = (0..=u8::MAX).collect();
        let decoded: Vec<_> = reference
            .decoded
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect();
        let tails = dataset("gate", format, 130, reference);
        for path in PATHS {
            let Some(backend) = selected(path) else {
                continue;
            };
            let mut output = vec![0xa5; expected.len()];
            assert!(
                convert(backend, true, format, &input, &mut output),
                "{path}: valid encode declined"
            );
            equal_bytes(
                &output,
                &expected,
                &format!("{path}: exhaustive encode oracle"),
            );
            let mut output = vec![0xa5; decoded.len()];
            assert!(convert(backend, false, format, &codes, &mut output));
            equal_bytes(
                &output,
                &decoded,
                &format!("{path}: exhaustive decode oracle"),
            );

            // Every rejected BF16/FP16 pattern, spread across all vector lanes and a tail.
            let mut invalid = [0u8; 34];
            for (index, bits) in reference.rejected.iter().enumerate() {
                let lane = index % 17 * 2;
                invalid[lane..lane + 2].copy_from_slice(&bits.to_le_bytes());
                assert!(
                    !convert(backend, true, format, &invalid, &mut [0u8; 17]),
                    "{path}: accepted {bits:04x}"
                );
                invalid[lane..lane + 2].fill(0);
            }
            // Unaligned slices, empty input, SIMD boundaries and scalar tails.
            for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
                let mut unaligned = vec![0u8; n * 2 + 1];
                unaligned[1..].copy_from_slice(&tails.input[..n * 2]);
                let mut output = vec![0xa5; n + 2];
                assert!(convert(
                    backend,
                    true,
                    format,
                    &unaligned[1..],
                    &mut output[1..n + 1]
                ));
                equal_bytes(
                    &output[1..n + 1],
                    &tails.encoded[..n],
                    &format!("{path}: encode tail"),
                );
                assert_eq!((output[0], output[n + 1]), (0xa5, 0xa5));
                let mut restored = vec![0xa5; n * 2 + 2];
                assert!(convert(
                    backend,
                    false,
                    format,
                    &output[1..n + 1],
                    &mut restored[1..n * 2 + 1]
                ));
                equal_bytes(
                    &restored[1..n * 2 + 1],
                    &tails.decoded[..n * 2],
                    &format!("{path}: decode tail"),
                );
                assert_eq!((restored[0], restored[n * 2 + 1]), (0xa5, 0xa5));
            }
            assert!(!convert(backend, true, format, &[0; 3], &mut [0; 1]));
            assert!(!convert(backend, false, format, &[0; 1], &mut [0; 3]));
        }
    }

    struct Dataset {
        dtype: &'static str,
        format: StorageFormat,
        input: Vec<u8>,
        encoded: Vec<u8>,
        decoded: Vec<u8>,
    }

    fn dataset(
        dtype: &'static str,
        format: StorageFormat,
        bytes: usize,
        reference: &Reference,
    ) -> Dataset {
        let mut input = Vec::with_capacity(bytes);
        let mut encoded = Vec::with_capacity(bytes / 2);
        let mut decoded = Vec::with_capacity(bytes);
        let mut state = 0x2026_0920_1234_5678u64;
        for _ in 0..bytes / 2 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let (bits, code) = reference.valid[state as usize % reference.valid.len()];
            input.extend_from_slice(&bits.to_le_bytes());
            encoded.push(code);
            decoded.extend_from_slice(&reference.decoded[code as usize].to_le_bytes());
        }
        Dataset {
            dtype,
            format,
            input,
            encoded,
            decoded,
        }
    }

    fn measure(mut operation: impl FnMut(), duration: Duration, batch: usize) -> (u64, Duration) {
        for _ in 0..3 {
            operation();
        }
        let start = Instant::now();
        let mut iterations = 0;
        loop {
            for _ in 0..batch {
                operation();
            }
            iterations += batch as u64;
            let elapsed = start.elapsed();
            if elapsed >= duration {
                return (iterations, elapsed);
            }
        }
    }

    pub(super) fn benchmark() -> Result<(), String> {
        let mut seconds = 0.25f64;
        let mut samples = 3usize;
        let mut check = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seconds" => {
                    seconds = args
                        .next()
                        .ok_or("--seconds requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --seconds")?;
                }
                "--samples" => {
                    samples = args
                        .next()
                        .ok_or("--samples requires a value")?
                        .parse()
                        .map_err(|_| "Invalid --samples")?;
                }
                "--check" => check = true,
                "--bench" => {} // Cargo passes this even with harness = false.
                "--help" | "-h" => {
                    println!(
                        "cpu_codec [--seconds 0.25] [--samples 3] [--check]\nCSV on stdout, metadata/gate status on stderr. --check runs only the oracle gates."
                    );
                    return Ok(());
                }
                _ => return Err(format!("Unknown argument: {arg}")),
            }
        }
        if !seconds.is_finite() || seconds <= 0.0 || seconds > 60.0 || !(1..=100).contains(&samples)
        {
            return Err("Require 0 < --seconds <= 60 and 1 <= --samples <= 100".into());
        }
        let duration = Duration::from_secs_f64(seconds);
        eprintln!(
            "cpu_codec: arch={} auto={:?} seconds_per_sample={seconds} samples={samples} warmup_calls=3",
            std::env::consts::ARCH,
            Backend::detect()
        );
        eprintln!(
            "dataset: seed=0x2026092012345678; finite in-range 16-bit patterns sampled uniformly; buffers/tables reused; allocation and oracle excluded from timing"
        );
        for path in PATHS {
            if selected(path).is_none() {
                eprintln!("{path}: unsupported; no timing");
            }
        }
        let mut datasets = Vec::new();
        for (dtype, format) in FORMATS {
            let reference = reference(format);
            correctness(format, &reference);
            for bytes in SIZES {
                let data = dataset(dtype, format, bytes, &reference);
                for path in PATHS {
                    let Some(backend) = selected(path) else {
                        continue;
                    };
                    let mut output = vec![0xa5; bytes / 2];
                    assert!(convert(backend, true, format, &data.input, &mut output));
                    equal_bytes(
                        &output,
                        &data.encoded,
                        &format!("{dtype}/{bytes}/{path}: encode fixture"),
                    );
                    let mut output = vec![0xa5; bytes];
                    assert!(convert(backend, false, format, &data.encoded, &mut output));
                    equal_bytes(
                        &output,
                        &data.decoded,
                        &format!("{dtype}/{bytes}/{path}: decode fixture"),
                    );
                }
                datasets.push(data);
            }
            eprintln!(
                "oracle PASS {dtype}: 65536 input patterns, 256 decode codes, rejection/tails/unaligned slices, all three timed sizes"
            );
        }
        eprintln!("All oracle gates PASS before any timing.");
        if check {
            return Ok(());
        }
        println!(
            "dtype,logical_bytes,path,selected_backend,operation,sample,status,iterations,elapsed_seconds,input_bytes_per_iteration,output_bytes_per_iteration,logical_gib_per_second,ns_per_iteration"
        );
        for data in datasets {
            let bytes = data.input.len();
            for (encoding, operation) in [(true, "encode"), (false, "decode")] {
                let input = if encoding { &data.input } else { &data.encoded };
                let expected = if encoding {
                    &data.encoded
                } else {
                    &data.decoded
                };
                let mut output = vec![0; expected.len()];
                for sample in 1..=samples {
                    // Rotate backend order between samples to expose less order bias.
                    for offset in 0..PATHS.len() {
                        let path = PATHS[(offset + sample - 1) % PATHS.len()];
                        let Some(backend) = selected(path) else {
                            println!(
                                "{},{bytes},{path},,{operation},{sample},unsupported,,,,,,",
                                data.dtype
                            );
                            continue;
                        };
                        let resolved =
                            format!("{:?}", backend.unwrap_or_else(Backend::detect)).to_lowercase();
                        let (iterations, elapsed) = measure(
                            || {
                                assert!(black_box(convert(
                                    backend,
                                    encoding,
                                    data.format,
                                    black_box(input),
                                    black_box(&mut output)
                                )));
                            },
                            duration,
                            (1024 * 1024 / bytes).clamp(1, 64),
                        );
                        equal_bytes(&output, expected, &format!("Timed output changed: {path}"));
                        let elapsed = elapsed.as_secs_f64();
                        let gib_s =
                            bytes as f64 * iterations as f64 / elapsed / (1024.0f64).powi(3);
                        let ns = elapsed * 1e9 / iterations as f64;
                        println!(
                            "{},{bytes},{path},{resolved},{operation},{sample},ok,{iterations},{elapsed:.9},{},{},{gib_s:.6},{ns:.3}",
                            data.dtype,
                            input.len(),
                            output.len()
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

fn main() {
    if let Err(error) = cpu::benchmark() {
        eprintln!("cpu_codec: {error}");
        std::process::exit(2);
    }
}
