#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

// Binary wrapper for OrbitKV Cache Manager
// This delegates to the orbitkv-server crate's run() function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    orbitkv_server::run()
}
