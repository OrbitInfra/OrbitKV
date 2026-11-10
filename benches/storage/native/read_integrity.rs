//! Paired CPU stored-byte scan/copy cost, not serving or GPU evidence.
use std::hint::black_box;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let passes = std::env::var("ORBITKV_BENCH_CHECKS")
        .unwrap_or_else(|_| "1".into())
        .parse::<usize>()?;
    if !(1..=3).contains(&passes) {
        return Err("ORBITKV_BENCH_CHECKS must be 1, 2 or 3".into());
    }
    println!(
        "representation,bytes,passes,sample,order,verification,iterations,seconds,ns_per_read"
    );
    for encoded in [false, true] {
        for bytes in [4096, 1 << 20, 16 << 20] {
            let input: Vec<u8> = (0..bytes).map(|i| (i * 17 + 3) as u8).collect();
            let expected = crc32c::crc32c(&input);
            let encoded_crc = crc32fast::hash(&input);
            let mut output = vec![0; bytes];
            let mut corrupt = input.clone();
            corrupt[bytes - 1] ^= 1;
            if crc32c::crc32c(&corrupt) == expected {
                return Err("corruption control failed".into());
            }
            drop(corrupt);
            for sample in 0..5 {
                for order in 0..2 {
                    let verify = (sample + order) % 2 != 0;
                    let started = Instant::now();
                    let mut iterations = 0;
                    while started.elapsed() < Duration::from_millis(200) {
                        let source = black_box(input.as_slice());
                        if verify {
                            for _ in 0..passes {
                                if crc32c::crc32c(black_box(source)) != expected {
                                    return Err("checksum mismatch".into());
                                }
                            }
                        }
                        // Existing encoded verification stays enabled in both arms.
                        if encoded && crc32fast::hash(source) != encoded_crc {
                            return Err("codec checksum mismatch".into());
                        }
                        output.copy_from_slice(source);
                        black_box(&output);
                        iterations += 1;
                    }
                    let seconds = started.elapsed().as_secs_f64();
                    if output != input {
                        return Err("copy control failed".into());
                    }
                    println!(
                        "{},{bytes},{passes},{sample},{order},{verify},{iterations},{seconds:.9},{:.3}",
                        if encoded { "encoded" } else { "raw" },
                        seconds * 1e9 / iterations as f64
                    );
                }
            }
        }
    }
    Ok(())
}
