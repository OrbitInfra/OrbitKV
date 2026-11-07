#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

//! Shared process logging and peer-connection defaults.

pub mod grpc;
pub mod logging;
pub mod timeline;
