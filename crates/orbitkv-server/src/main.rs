#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    orbitkv_server::run()
}
