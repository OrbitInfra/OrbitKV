#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

use std::io::Write;
use std::time::Duration;

use orbitkv_channel::{CommandCode, Response, StatusCode, TransportServer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service_name = std::env::args()
        .nth(1)
        .ok_or("usage: orbitkv-channel-echo <service-name> [session-epoch]")?;
    let session_epoch = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1);
    let server = TransportServer::bind(&service_name)?;
    println!("READY");
    std::io::stdout().flush()?;

    let mut running = true;
    while running {
        if !server.try_serve_for_epoch(session_epoch, |command| {
            running = command.code != CommandCode::Shutdown;
            let mut response = Response::ok(command);
            if command.code == CommandCode::Ping {
                response.value0 = command.arg0.wrapping_add(1);
            } else if command.code != CommandCode::Shutdown {
                response.status = StatusCode::Invalid;
            }
            response
        })? {
            server.wait_for_request(Duration::from_millis(10))?;
        }
    }
    Ok(())
}
