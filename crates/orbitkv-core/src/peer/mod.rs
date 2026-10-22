//! Peer discovery, source authorization and transfer completion ownership.

#[cfg(any(feature = "mooncake", feature = "orbitte"))]
mod completion;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
mod execute;
pub(crate) mod export;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) mod read;
#[cfg(feature = "mooncake")]
pub(crate) mod transport;

#[cfg(feature = "orbitte")]
mod orbitte;
#[cfg(any(feature = "mooncake", feature = "orbitte"))]
pub(crate) mod provider;
