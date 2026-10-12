//! Bounded lifecycle metadata frames carried by the authenticated bootstrap UDS.

use std::io::{self, IoSlice, IoSliceMut, Read};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;

use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};

pub const LIFECYCLE_HEADER_BYTES: usize = 20;
pub const MAX_QUERY_TARGET_PAYLOAD: usize = 64 * 1024;
pub const MAX_LIFECYCLE_PAYLOAD: usize = 64 * 1024 * 1024;
const MAGIC: u32 = 0x4f52_424c;
const VERSION: u16 = 7;

/// Payload arenas are attached only to successful GPU registration replies.
pub const MAX_LIFECYCLE_FDS: usize = 64;
const FD_MARKER: u8 = 0xa7;

#[derive(Debug, Default)]
pub struct LifecycleReply {
    pub payload: Vec<u8>,
    pub fds: Vec<OwnedFd>,
}

/// The marker is a separate one-byte stream frame, so retrying WouldBlock never
/// repeats already-transferred descriptors. Header and payload follow normally.
pub fn send_lifecycle_fds(socket: &impl AsFd, fds: &[BorrowedFd<'_>]) -> io::Result<()> {
    if fds.len() > MAX_LIFECYCLE_FDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "too many payload arenas",
        ));
    }
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_LIFECYCLE_FDS))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !fds.is_empty() && !control.push(SendAncillaryMessage::ScmRights(fds)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "arena descriptor buffer exhausted",
        ));
    }
    loop {
        match sendmsg(
            socket,
            &[IoSlice::new(&[FD_MARKER])],
            &mut control,
            SendFlags::NOSIGNAL,
        ) {
            Ok(1) => return Ok(()),
            Ok(_) => return Err(io::ErrorKind::WriteZero.into()),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

pub(crate) fn receive_lifecycle_reply(
    mut socket: &UnixStream,
    payload_limit: usize,
) -> io::Result<(LifecycleHeader, LifecycleReply)> {
    let mut marker = [0];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_LIFECYCLE_FDS))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let message = loop {
        match recvmsg(
            socket,
            &mut [IoSliceMut::new(&mut marker)],
            &mut control,
            RecvFlags::CMSG_CLOEXEC,
        ) {
            Ok(message) => break message,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    };
    if message.bytes == 0 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    if marker != [FD_MARKER] || message.flags.contains(ReturnFlags::CTRUNC) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid lifecycle descriptor frame",
        ));
    }
    let mut fds = Vec::new();
    for message in control.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = message {
            fds.extend(rights);
        }
    }
    if fds.len() > MAX_LIFECYCLE_FDS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "too many payload arenas",
        ));
    }
    let mut header = [0; LIFECYCLE_HEADER_BYTES];
    socket.read_exact(&mut header)?;
    let header = LifecycleHeader::decode(header)?;
    if header.payload_len > payload_limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lifecycle response exceeds command limit",
        ));
    }
    let mut payload = vec![0; header.payload_len];
    socket.read_exact(&mut payload)?;
    Ok((header, LifecycleReply { payload, fds }))
}

#[derive(Clone, Copy, Debug)]
#[repr(u16)]
pub enum LifecycleCommand {
    Health = 1,
    Register = 2,
    Unregister = 3,
    Session = 4,
    ExportQueryTarget = 5,
    ConfigureShardQueries = 6,
}

impl TryFrom<u16> for LifecycleCommand {
    type Error = io::Error;

    fn try_from(value: u16) -> io::Result<Self> {
        match value {
            1 => Ok(Self::Health),
            2 => Ok(Self::Register),
            3 => Ok(Self::Unregister),
            4 => Ok(Self::Session),
            5 => Ok(Self::ExportQueryTarget),
            6 => Ok(Self::ConfigureShardQueries),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown lifecycle command",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LifecycleHeader {
    pub code: u16,
    pub epoch: u64,
    pub payload_len: usize,
}

impl LifecycleHeader {
    pub fn encode(self) -> io::Result<[u8; LIFECYCLE_HEADER_BYTES]> {
        if self.payload_len > MAX_LIFECYCLE_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "lifecycle metadata exceeds limit",
            ));
        }
        let mut bytes = [0; LIFECYCLE_HEADER_BYTES];
        bytes[..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.code.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[16..20].copy_from_slice(&(self.payload_len as u32).to_le_bytes());
        Ok(bytes)
    }

    pub fn decode(bytes: [u8; LIFECYCLE_HEADER_BYTES]) -> io::Result<Self> {
        if bytes[..4] != MAGIC.to_le_bytes() || bytes[4..6] != VERSION.to_le_bytes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "incompatible lifecycle protocol",
            ));
        }
        let header = Self {
            code: u16::from_le_bytes(bytes[6..8].try_into().expect("fixed header")),
            epoch: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed header")),
            payload_len: u32::from_le_bytes(bytes[16..20].try_into().expect("fixed header"))
                as usize,
        };
        if header.payload_len > MAX_LIFECYCLE_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lifecycle metadata exceeds limit",
            ));
        }
        Ok(header)
    }
}

#[cfg(test)]
#[path = "../tests/unit/lifecycle.rs"]
mod tests;
