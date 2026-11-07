#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use std::time::{Duration, Instant};

use orbitkv_channel::{CallOptions, Command, CommandCode, TransportClient};

fn percentile(sorted: &[u64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index] as f64 / 1_000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service_name = std::env::args()
        .nth(1)
        .ok_or("usage: orbitkv-channel-bench <service-name> [iterations]")?;
    let iterations = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(100_000usize);
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }

    let client = TransportClient::connect(&service_name)?;
    let options = CallOptions {
        timeout: Duration::from_secs(2),
        ..CallOptions::default()
    };
    for sequence in 0..10_000 {
        let mut command = Command::ping(sequence, 1);
        command.arg0 = sequence;
        client.call(command, options)?;
    }

    let mut samples = Vec::with_capacity(iterations);
    let total_start = Instant::now();
    for sequence in 0..iterations {
        let mut command = Command::ping(sequence as u64, 1);
        command.arg0 = sequence as u64;
        let started = Instant::now();
        let response = client.call(command, options)?;
        samples.push(started.elapsed().as_nanos() as u64);
        if response.value0 != command.arg0 + 1 {
            return Err(format!("invalid response for request {sequence}").into());
        }
    }
    let total = total_start.elapsed();
    samples.sort_unstable();

    println!("transport=iceoryx2-0.10-request-response");
    println!("processes=2");
    println!("payload_bytes=64");
    println!("iterations={iterations}");
    println!(
        "avg_us={:.3}",
        total.as_secs_f64() * 1e6 / iterations as f64
    );
    println!("p50_us={:.3}", percentile(&samples, 0.50));
    println!("p95_us={:.3}", percentile(&samples, 0.95));
    println!("p99_us={:.3}", percentile(&samples, 0.99));
    println!("max_us={:.3}", samples[iterations - 1] as f64 / 1_000.0);
    println!(
        "round_trips_per_sec={:.0}",
        iterations as f64 / total.as_secs_f64()
    );

    client.call(
        Command {
            code: CommandCode::Shutdown,
            request_id: iterations as u64 + 1,
            ..Command::ping(iterations as u64 + 1, 1)
        },
        options,
    )?;
    Ok(())
}
