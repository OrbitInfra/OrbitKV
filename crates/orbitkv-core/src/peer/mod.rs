//! Peer discovery, source authorization and transfer completion ownership.

#[cfg(any(feature = "mooncake", feature = "orbite"))]
mod completion;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) mod execute;
pub(crate) mod export;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) mod read;
#[cfg(feature = "mooncake")]
pub(crate) mod transport;

#[cfg(feature = "orbite")]
pub(crate) mod orbite;

#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) mod restore;
#[cfg(any(feature = "mooncake", feature = "orbite"))]
pub(crate) mod segment;
