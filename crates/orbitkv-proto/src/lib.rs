#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod inventory;
mod registration;

pub mod proto {
    #[allow(
        clippy::allow_attributes_without_reason,
        reason = "prost/tonic generated modules emit allow attributes"
    )]
    pub mod engine {
        tonic::include_proto!("orbitkv");
    }
}
