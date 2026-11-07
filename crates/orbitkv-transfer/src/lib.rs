#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod engine;
mod error;
pub mod files;
mod notification;
mod types;

pub use engine::{MemoryRegistration, TransferEngine};
pub use error::{MooncakeError as TransferError, Result};
pub use types::{
    AUTO_MEMORY_LOCATION, NicLoadStat, Notification, P2P_METADATA, TransferOp, TransferPriority,
    TransferSlice,
};
