//! Peer discovery, source authorization and transfer completion ownership.

mod completion;
pub(crate) mod execute;
pub(crate) mod export;
pub(crate) mod read;

#[cfg(feature = "orbite")]
pub(crate) mod orbite;

pub(crate) mod restore;
pub(crate) mod segment;
