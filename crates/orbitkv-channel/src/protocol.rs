use thiserror::Error;

pub const ABI_VERSION: u16 = 13;
pub const WIRE_MESSAGE_BYTES: usize = 64;
/// Response `value1` bit set after a descriptor-backed request has been
/// accepted and its generation consumed, including business-error responses.
pub const RESPONSE_FLAG_REQUEST_CONSUMED: u64 = 1;
const MAGIC: u32 = 0x4f52_4254; // ORBT
const WORD_COUNT: usize = 8;

/// iceoryx2-compatible fixed-size control message.
///
/// Variable-length hashes and page lists live in a separately registered
/// descriptor arena. This message carries only offsets and identity guards.
pub type WireMessage = [u64; WORD_COUNT];
const _: [(); WIRE_MESSAGE_BYTES] = [(); std::mem::size_of::<WireMessage>()];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum CommandCode {
    Ping = 1,
    QueryBundle = 2,
    Restore = 3,
    Publish = 4,
    Release = 5,
    Shutdown = 6,
    CancelQuery = 7,
    QueryShards = 9,
    CancelShardQuery = 10,
    ReleaseShardQuery = 11,
}

impl TryFrom<u16> for CommandCode {
    type Error = ProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ping),
            2 => Ok(Self::QueryBundle),
            3 => Ok(Self::Restore),
            4 => Ok(Self::Publish),
            5 => Ok(Self::Release),
            6 => Ok(Self::Shutdown),
            7 => Ok(Self::CancelQuery),
            9 => Ok(Self::QueryShards),
            10 => Ok(Self::CancelShardQuery),
            11 => Ok(Self::ReleaseShardQuery),
            _ => Err(ProtocolError::UnknownCode(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum StatusCode {
    Ok = 0,
    Loading = 1,
    Miss = 2,
    Invalid = 3,
    StaleSession = 4,
    StaleGeneration = 5,
    Internal = 6,
}

impl TryFrom<u16> for StatusCode {
    type Error = ProtocolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Ok),
            1 => Ok(Self::Loading),
            2 => Ok(Self::Miss),
            3 => Ok(Self::Invalid),
            4 => Ok(Self::StaleSession),
            5 => Ok(Self::StaleGeneration),
            6 => Ok(Self::Internal),
            _ => Err(ProtocolError::UnknownCode(value)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DescriptorRef {
    pub offset: u64,
    pub len: u32,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    pub code: CommandCode,
    pub request_id: u64,
    pub session_epoch: u64,
    pub descriptor: DescriptorRef,
    pub arg0: u64,
    pub arg1: u64,
}

impl Command {
    pub fn ping(request_id: u64, session_epoch: u64) -> Self {
        Self {
            code: CommandCode::Ping,
            request_id,
            session_epoch,
            descriptor: DescriptorRef::default(),
            arg0: 0,
            arg1: 0,
        }
    }

    pub fn encode(self) -> WireMessage {
        [
            encode_header(self.code as u16),
            self.request_id,
            self.session_epoch,
            self.descriptor.offset,
            u64::from(self.descriptor.len),
            self.descriptor.generation,
            self.arg0,
            self.arg1,
        ]
    }

    pub fn decode(message: WireMessage) -> Result<Self, ProtocolError> {
        let code = CommandCode::try_from(decode_header(message[0])?)?;
        let len =
            u32::try_from(message[4]).map_err(|_| ProtocolError::LengthOverflow(message[4]))?;
        Ok(Self {
            code,
            request_id: message[1],
            session_epoch: message[2],
            descriptor: DescriptorRef {
                offset: message[3],
                len,
                generation: message[5],
            },
            arg0: message[6],
            arg1: message[7],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response {
    pub status: StatusCode,
    pub request_id: u64,
    pub session_epoch: u64,
    pub descriptor: DescriptorRef,
    pub value0: u64,
    pub value1: u64,
}

impl Response {
    pub fn ok(command: Command) -> Self {
        Self {
            status: StatusCode::Ok,
            request_id: command.request_id,
            session_epoch: command.session_epoch,
            descriptor: command.descriptor,
            value0: command.arg0,
            value1: command.arg1,
        }
    }

    pub fn encode(self) -> WireMessage {
        [
            encode_header(self.status as u16),
            self.request_id,
            self.session_epoch,
            self.descriptor.offset,
            u64::from(self.descriptor.len),
            self.descriptor.generation,
            self.value0,
            self.value1,
        ]
    }

    pub fn decode(message: WireMessage) -> Result<Self, ProtocolError> {
        let status = StatusCode::try_from(decode_header(message[0])?)?;
        let len =
            u32::try_from(message[4]).map_err(|_| ProtocolError::LengthOverflow(message[4]))?;
        Ok(Self {
            status,
            request_id: message[1],
            session_epoch: message[2],
            descriptor: DescriptorRef {
                offset: message[3],
                len,
                generation: message[5],
            },
            value0: message[6],
            value1: message[7],
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("invalid wire magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported process channel ABI version: {0}")]
    UnsupportedVersion(u16),
    #[error("unknown wire code: {0}")]
    UnknownCode(u16),
    #[error("descriptor length does not fit u32: {0}")]
    LengthOverflow(u64),
}

fn encode_header(code: u16) -> u64 {
    (u64::from(MAGIC) << 32) | (u64::from(ABI_VERSION) << 16) | u64::from(code)
}

fn decode_header(header: u64) -> Result<u16, ProtocolError> {
    let magic = (header >> 32) as u32;
    if magic != MAGIC {
        return Err(ProtocolError::InvalidMagic(magic));
    }
    let version = ((header >> 16) & 0xffff) as u16;
    if version != ABI_VERSION {
        return Err(ProtocolError::UnsupportedVersion(version));
    }
    Ok((header & 0xffff) as u16)
}

#[cfg(test)]
#[path = "../tests/unit/protocol.rs"]
mod tests;
