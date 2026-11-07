use std::fs;
use std::io::{ErrorKind, IoSlice, IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use rustix::event::{EventfdFlags, eventfd};
use rustix::net::sockopt::socket_peercred;
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};
use rustix::process::geteuid;
use thiserror::Error;

use crate::{ArenaError, CompletionError, DescriptorArena, DescriptorRef, RestoreCompletions};

const BOOTSTRAP_MAGIC: u32 = 0x4f52_4242; // ORBB
// Restore grants add an engine-to-Manager reclamation doorbell.
const BOOTSTRAP_VERSION: u16 = 7;
const BOOTSTRAP_BYTES: usize = 256;
const BOOTSTRAP_FD_COUNT: usize = 5;
// Match the iceoryx2 client limit. Detached restore publishers also hold this
// budget until completion, even after the bootstrap session disconnects.
const MAX_COMPLETION_SESSIONS: usize = 64;
const SERVICE_NAME_OFFSET: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerCredentials {
    pub pid: u32,
    pub uid: u32,
    pub gid: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapInfo {
    pub service_name: String,
    pub session_epoch: u64,
    pub arena_bytes: usize,
    pub slot_capacity: usize,
    pub slot_count: usize,
    pub slot_index: usize,
    pub initial_generation: u64,
    pub client_token: u64,
}

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("bootstrap socket operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arena(#[from] ArenaError),
    #[error(transparent)]
    Completion(#[from] CompletionError),
    #[error("restore completion session budget exhausted")]
    CompletionBudget,
    #[error("peer uid {actual} is not permitted; expected {expected}")]
    PeerUid { expected: u32, actual: u32 },
    #[error("invalid bootstrap magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("unsupported bootstrap version: {0}")]
    UnsupportedVersion(u16),
    #[error("bootstrap payload length mismatch: expected {expected}, got {actual}")]
    PayloadLength { expected: usize, actual: usize },
    #[error("bootstrap expected {expected} file descriptors, got {actual}")]
    FileDescriptorCount { expected: usize, actual: usize },
    #[error("bootstrap ancillary buffer was too small")]
    AncillaryTruncated,
    #[error("bootstrap field {field} exceeds its local representation")]
    FieldOverflow { field: &'static str },
    #[error("iceoryx2 service name is too long for bootstrap: {0} bytes")]
    ServiceNameTooLong(usize),
    #[error("bootstrap iceoryx2 service name is not valid UTF-8")]
    InvalidServiceNameUtf8,
    #[error("all {0} descriptor slots are in use")]
    NoFreeSlots(usize),
    #[error("descriptor generation exhausted for slot {0}")]
    GenerationExhausted(usize),
    #[error("bootstrap client-token space is exhausted")]
    ClientTokenExhausted,
    #[error("descriptor belongs to slot {actual}, but client owns slot {expected}")]
    SlotMismatch { expected: usize, actual: usize },
    #[error("client token does not match the bootstrap session")]
    ClientTokenMismatch,
    #[error("unexpected request generation: expected {expected}, got {actual}")]
    UnexpectedGeneration { expected: u64, actual: u64 },
}

pub struct BootstrapServer {
    listener: UnixListener,
    socket_path: PathBuf,
    socket_identity: SocketIdentity,
    arena: DescriptorArena,
    service_name: String,
    session_epoch: u64,
    slots: Arc<Mutex<Vec<bool>>>,
    next_client_token: AtomicU64,
    completion_maps: Mutex<Vec<Weak<RestoreCompletions>>>,
}

impl BootstrapServer {
    pub fn bind(
        socket_path: impl AsRef<Path>,
        service_name: impl Into<String>,
        session_epoch: u64,
        arena_bytes: usize,
        slot_capacity: usize,
    ) -> Result<Self, BootstrapError> {
        let socket_path = socket_path.as_ref().to_path_buf();
        let arena = DescriptorArena::create(session_epoch, arena_bytes, slot_capacity)?;
        let listener = bind_socket(&socket_path)?;
        if let Err(error) = fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600)) {
            drop(listener);
            let _ = fs::remove_file(&socket_path);
            return Err(error.into());
        }
        let socket_identity = SocketIdentity::from_metadata(&fs::symlink_metadata(&socket_path)?);
        let slots = Arc::new(Mutex::new(vec![false; arena.slot_count()]));
        Ok(Self {
            listener,
            socket_path,
            socket_identity,
            arena,
            service_name: service_name.into(),
            session_epoch,
            slots,
            next_client_token: AtomicU64::new(session_epoch.rotate_left(17) | 1),
            completion_maps: Mutex::new(Vec::new()),
        })
    }

    pub fn accept(&self) -> Result<BootstrapSession, BootstrapError> {
        let (stream, _) = self.listener.accept()?;
        let credentials = peer_credentials(&stream)?;
        let expected_uid = geteuid().as_raw();
        if credentials.uid != expected_uid {
            return Err(BootstrapError::PeerUid {
                expected: expected_uid,
                actual: credentials.uid,
            });
        }
        let (slot_index, initial_generation) = self.allocate_slot()?;
        let client_token = match self.next_client_token.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |token| token.checked_add(2),
        ) {
            Ok(token) => token,
            Err(_) => {
                self.release_slot(slot_index);
                return Err(BootstrapError::ClientTokenExhausted);
            }
        };
        let notification = match eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK) {
            Ok(notification) => notification,
            Err(error) => {
                self.release_slot(slot_index);
                return Err(std::io::Error::from(error).into());
            }
        };
        let reply_notification = match eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK) {
            Ok(notification) => Arc::new(notification),
            Err(error) => {
                self.release_slot(slot_index);
                return Err(std::io::Error::from(error).into());
            }
        };
        let completions = match self.create_completions(client_token, notification) {
            Ok(completions) => completions,
            Err(error) => {
                self.release_slot(slot_index);
                return Err(error);
            }
        };
        let info = BootstrapInfo {
            service_name: self.service_name.clone(),
            session_epoch: self.session_epoch,
            arena_bytes: self.arena.len(),
            slot_capacity: self.arena.slot_capacity(),
            slot_count: self.arena.slot_count(),
            slot_index,
            initial_generation,
            client_token,
        };
        if let Err(error) = send_bootstrap(
            &stream,
            &info,
            self.arena.file(),
            completions.notification_fd(),
            completions.file(),
            reply_notification.as_ref(),
            completions.manager_notification_fd(),
        ) {
            self.release_slot(slot_index);
            return Err(error);
        }
        if let Err(error) = stream.set_nonblocking(true) {
            self.release_slot(slot_index);
            return Err(error.into());
        }
        Ok(BootstrapSession {
            stream,
            credentials,
            completions,
            reply_notification,
            slot_index,
            client_token,
            next_request_generation: initial_generation,
            slots: Arc::clone(&self.slots),
        })
    }

    fn create_completions(
        &self,
        token: u64,
        notification: std::os::fd::OwnedFd,
    ) -> Result<Arc<RestoreCompletions>, BootstrapError> {
        let mut maps = self
            .completion_maps
            .lock()
            .map_err(|_| ArenaError::Poisoned)?;
        maps.retain(|map| map.strong_count() != 0);
        if maps.len() >= MAX_COMPLETION_SESSIONS {
            return Err(BootstrapError::CompletionBudget);
        }
        let completions = Arc::new(RestoreCompletions::create(
            self.session_epoch,
            token,
            notification,
        )?);
        maps.push(Arc::downgrade(&completions));
        Ok(completions)
    }

    pub fn arena(&self) -> &DescriptorArena {
        &self.arena
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), BootstrapError> {
        self.listener.set_nonblocking(nonblocking)?;
        Ok(())
    }

    pub fn try_accept(&self) -> Result<Option<BootstrapSession>, BootstrapError> {
        match self.accept() {
            Ok(session) => Ok(Some(session)),
            Err(BootstrapError::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub fn descriptor_slot(&self, offset: u64) -> Result<usize, ArenaError> {
        self.arena.descriptor_slot(offset)
    }

    fn allocate_slot(&self) -> Result<(usize, u64), BootstrapError> {
        let mut slots = self.slots.lock().map_err(|_| ArenaError::Poisoned)?;
        let slot = slots
            .iter()
            .position(|active| !active)
            .ok_or(BootstrapError::NoFreeSlots(slots.len()))?;
        let current = self.arena.slot_generation(slot)?;
        let initial_generation = if current == 0 {
            1
        } else if current.is_multiple_of(2) {
            current
                .checked_add(1)
                .ok_or(BootstrapError::GenerationExhausted(slot))?
        } else {
            current
                .checked_add(2)
                .ok_or(BootstrapError::GenerationExhausted(slot))?
        };
        self.arena.reset_slot(slot, initial_generation)?;
        slots[slot] = true;
        Ok((slot, initial_generation))
    }

    fn release_slot(&self, slot: usize) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(active) = slots.get_mut(slot)
        {
            *active = false;
        }
    }
}

impl Drop for BootstrapServer {
    fn drop(&mut self) {
        if let Ok(metadata) = fs::symlink_metadata(&self.socket_path)
            && metadata.file_type().is_socket()
            && SocketIdentity::from_metadata(&metadata) == self.socket_identity
        {
            let _ = fs::remove_file(&self.socket_path);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SocketIdentity {
    device: u64,
    inode: u64,
}

impl SocketIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
        }
    }
}

pub struct BootstrapSession {
    completions: Arc<RestoreCompletions>,
    reply_notification: Arc<std::os::fd::OwnedFd>,
    stream: UnixStream,
    credentials: PeerCredentials,
    slot_index: usize,
    client_token: u64,
    next_request_generation: u64,
    slots: Arc<Mutex<Vec<bool>>>,
}

impl BootstrapSession {
    pub fn completions(&self) -> &Arc<RestoreCompletions> {
        &self.completions
    }

    pub fn reply_notification_fd(&self) -> &Arc<std::os::fd::OwnedFd> {
        &self.reply_notification
    }

    pub fn credentials(&self) -> PeerCredentials {
        self.credentials
    }

    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    pub fn notification_fd(&self) -> &std::os::fd::OwnedFd {
        self.completions.notification_fd()
    }

    pub fn notify(&self) -> Result<(), BootstrapError> {
        self.completions.notify().map_err(Into::into)
    }

    pub fn slot_index(&self) -> usize {
        self.slot_index
    }

    pub fn client_token(&self) -> u64 {
        self.client_token
    }

    pub fn is_alive(&self) -> Result<bool, BootstrapError> {
        let mut byte = [0u8; 1];
        match rustix::net::recv(
            &self.stream,
            &mut byte,
            RecvFlags::PEEK | RecvFlags::DONTWAIT,
        ) {
            Ok((_, 0)) => Ok(false),
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(true),
            Err(error) => Err(std::io::Error::from(error).into()),
        }
    }

    pub fn validate_request(
        &mut self,
        descriptor: DescriptorRef,
        client_token: u64,
        descriptor_slot: usize,
    ) -> Result<(), BootstrapError> {
        if client_token != self.client_token {
            return Err(BootstrapError::ClientTokenMismatch);
        }
        if descriptor_slot != self.slot_index {
            return Err(BootstrapError::SlotMismatch {
                expected: self.slot_index,
                actual: descriptor_slot,
            });
        }
        if descriptor.generation != self.next_request_generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: self.next_request_generation,
                actual: descriptor.generation,
            });
        }
        Ok(())
    }

    pub fn complete_request(&mut self) -> Result<(), BootstrapError> {
        self.next_request_generation = self
            .next_request_generation
            .checked_add(2)
            .ok_or(BootstrapError::GenerationExhausted(self.slot_index))?;
        Ok(())
    }
}

impl Drop for BootstrapSession {
    fn drop(&mut self) {
        if let Ok(mut slots) = self.slots.lock()
            && let Some(active) = slots.get_mut(self.slot_index)
        {
            *active = false;
        }
    }
}

pub struct BootstrapClient {
    completions: Arc<RestoreCompletions>,
    reply_notification: std::os::fd::OwnedFd,
    stream: UnixStream,
    arena: DescriptorArena,
    info: BootstrapInfo,
    next_generation: Mutex<u64>,
}

impl BootstrapClient {
    pub fn completions(&self) -> &Arc<RestoreCompletions> {
        &self.completions
    }

    pub fn connect(socket_path: impl AsRef<Path>) -> Result<Self, BootstrapError> {
        let stream = UnixStream::connect(socket_path)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        let (info, mut fds) = receive_bootstrap(&stream)?;
        let manager_notification = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 0,
        })?;
        let reply_notification = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 0,
        })?;
        let completion_fd = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 0,
        })?;
        let notification = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 0,
        })?;
        let completions = Arc::new(RestoreCompletions::open(
            completion_fd,
            notification,
            manager_notification,
            info.session_epoch,
            info.client_token,
        )?);
        let arena_fd = fds.pop().ok_or(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: 1,
        })?;
        let arena = DescriptorArena::open(arena_fd, info.session_epoch, info.arena_bytes)?;
        Ok(Self {
            stream,
            arena,
            next_generation: Mutex::new(info.initial_generation),
            completions,
            reply_notification,
            info,
        })
    }

    pub fn info_ref(&self) -> &BootstrapInfo {
        &self.info
    }

    pub fn arena(&self) -> &DescriptorArena {
        &self.arena
    }

    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    pub fn notification_fd(&self) -> &std::os::fd::OwnedFd {
        self.completions.notification_fd()
    }

    pub(crate) fn reply_notification_fd(&self) -> &std::os::fd::OwnedFd {
        &self.reply_notification
    }

    pub fn wait_for_notification(&self, timeout: Duration) -> Result<bool, BootstrapError> {
        let timeout = rustix::event::Timespec {
            tv_sec: i64::try_from(timeout.as_secs()).map_err(|_| {
                BootstrapError::FieldOverflow {
                    field: "notification_timeout",
                }
            })?,
            tv_nsec: timeout.subsec_nanos().into(),
        };
        let mut fds = [rustix::event::PollFd::new(
            self.completions.notification_fd(),
            rustix::event::PollFlags::IN,
        )];
        if rustix::event::poll(&mut fds, Some(&timeout)).map_err(std::io::Error::from)? == 0 {
            return Ok(false);
        }
        let mut value = [0u8; 8];
        let read = match rustix::io::read(self.completions.notification_fd(), &mut value) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(error) => return Err(std::io::Error::from(error).into()),
        };
        Ok(read == value.len())
    }

    pub fn write_request(&self, payload: &[u8]) -> Result<crate::DescriptorRef, BootstrapError> {
        let generation = self
            .next_generation
            .lock()
            .map_err(|_| ArenaError::Poisoned)?;
        let descriptor = self
            .arena
            .write_slot(self.info.slot_index, *generation, payload)?;
        Ok(descriptor)
    }

    pub fn read_response(
        &self,
        request: DescriptorRef,
        response: DescriptorRef,
    ) -> Result<Vec<u8>, BootstrapError> {
        let expected_generation = request
            .generation
            .checked_add(1)
            .ok_or(BootstrapError::GenerationExhausted(self.info.slot_index))?;
        if response.offset != request.offset {
            return Err(BootstrapError::SlotMismatch {
                expected: self.info.slot_index,
                actual: self.arena.descriptor_slot(response.offset)?,
            });
        }
        if response.generation != expected_generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: expected_generation,
                actual: response.generation,
            });
        }
        let payload = self.arena.read(response)?;
        self.complete_request(request)?;
        Ok(payload)
    }

    pub fn complete_request(&self, request: DescriptorRef) -> Result<(), BootstrapError> {
        let mut generation = self
            .next_generation
            .lock()
            .map_err(|_| ArenaError::Poisoned)?;
        if *generation != request.generation {
            return Err(BootstrapError::UnexpectedGeneration {
                expected: *generation,
                actual: request.generation,
            });
        }
        *generation = generation
            .checked_add(2)
            .ok_or(BootstrapError::GenerationExhausted(self.info.slot_index))?;
        Ok(())
    }
}

fn peer_credentials(stream: &UnixStream) -> Result<PeerCredentials, BootstrapError> {
    let credentials = socket_peercred(stream).map_err(std::io::Error::from)?;
    Ok(PeerCredentials {
        pid: u32::try_from(credentials.pid.as_raw_nonzero().get())
            .map_err(|_| BootstrapError::FieldOverflow { field: "peer_pid" })?,
        uid: credentials.uid.as_raw(),
        gid: credentials.gid.as_raw(),
    })
}

fn bind_socket(path: &Path) -> Result<UnixListener, std::io::Error> {
    match UnixListener::bind(path) {
        Ok(listener) => Ok(listener),
        Err(error) if error.kind() == ErrorKind::AddrInUse => match UnixStream::connect(path) {
            Ok(_) => Err(error),
            Err(connect_error)
                if matches!(
                    connect_error.kind(),
                    ErrorKind::ConnectionRefused | ErrorKind::NotFound
                ) =>
            {
                let metadata = fs::symlink_metadata(path)?;
                if !metadata.file_type().is_socket() {
                    return Err(error);
                }
                fs::remove_file(path)?;
                UnixListener::bind(path)
            }
            Err(_) => Err(error),
        },
        Err(error) => Err(error),
    }
}

fn send_bootstrap(
    stream: &UnixStream,
    info: &BootstrapInfo,
    arena_fd: &impl std::os::fd::AsFd,
    notification_fd: &impl std::os::fd::AsFd,
    completion_fd: &impl std::os::fd::AsFd,
    reply_notification_fd: &impl std::os::fd::AsFd,
    manager_notification_fd: &impl std::os::fd::AsFd,
) -> Result<(), BootstrapError> {
    let payload = encode_info(info)?;
    let borrowed = [
        arena_fd.as_fd(),
        notification_fd.as_fd(),
        completion_fd.as_fd(),
        reply_notification_fd.as_fd(),
        manager_notification_fd.as_fd(),
    ];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(BOOTSTRAP_FD_COUNT))];
    let mut control = SendAncillaryBuffer::new(&mut space);
    if !control.push(SendAncillaryMessage::ScmRights(&borrowed)) {
        return Err(BootstrapError::AncillaryTruncated);
    }
    let sent = sendmsg(
        stream,
        &[IoSlice::new(&payload)],
        &mut control,
        SendFlags::empty(),
    )
    .map_err(std::io::Error::from)?;
    if sent == 0 {
        return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into());
    }
    if sent < payload.len() {
        let mut stream = stream;
        stream.write_all(&payload[sent..])?;
    }
    Ok(())
}

fn receive_bootstrap(
    stream: &UnixStream,
) -> Result<(BootstrapInfo, Vec<std::os::fd::OwnedFd>), BootstrapError> {
    let mut payload = [0u8; BOOTSTRAP_BYTES];
    let mut iov = [IoSliceMut::new(&mut payload)];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(BOOTSTRAP_FD_COUNT))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    let message = recvmsg(stream, &mut iov, &mut control, RecvFlags::CMSG_CLOEXEC)
        .map_err(std::io::Error::from)?;
    if message.bytes == 0 {
        return Err(BootstrapError::PayloadLength {
            expected: BOOTSTRAP_BYTES,
            actual: 0,
        });
    }
    if message.flags.contains(rustix::net::ReturnFlags::CTRUNC) {
        return Err(BootstrapError::AncillaryTruncated);
    }
    let mut fds = Vec::new();
    for message in control.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = message {
            fds.extend(rights);
        }
    }
    if fds.len() != BOOTSTRAP_FD_COUNT {
        return Err(BootstrapError::FileDescriptorCount {
            expected: BOOTSTRAP_FD_COUNT,
            actual: fds.len(),
        });
    }
    if message.bytes < BOOTSTRAP_BYTES {
        let mut stream = stream;
        stream.read_exact(&mut payload[message.bytes..])?;
    }
    Ok((decode_info(payload)?, fds))
}

fn encode_info(info: &BootstrapInfo) -> Result<[u8; BOOTSTRAP_BYTES], BootstrapError> {
    let service_name = info.service_name.as_bytes();
    if service_name.len() > BOOTSTRAP_BYTES - SERVICE_NAME_OFFSET {
        return Err(BootstrapError::ServiceNameTooLong(service_name.len()));
    }
    let mut bytes = [0u8; BOOTSTRAP_BYTES];
    bytes[0..4].copy_from_slice(&BOOTSTRAP_MAGIC.to_le_bytes());
    bytes[4..6].copy_from_slice(&BOOTSTRAP_VERSION.to_le_bytes());
    bytes[8..16].copy_from_slice(&info.session_epoch.to_le_bytes());
    bytes[16..24].copy_from_slice(
        &u64::try_from(info.arena_bytes)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "arena_bytes",
            })?
            .to_le_bytes(),
    );
    bytes[24..28].copy_from_slice(
        &u32::try_from(info.slot_capacity)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_capacity",
            })?
            .to_le_bytes(),
    );
    bytes[28..32].copy_from_slice(
        &u32::try_from(info.slot_count)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_count",
            })?
            .to_le_bytes(),
    );
    bytes[32..36].copy_from_slice(
        &u32::try_from(info.slot_index)
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "slot_index",
            })?
            .to_le_bytes(),
    );
    bytes[36..38].copy_from_slice(
        &u16::try_from(service_name.len())
            .map_err(|_| BootstrapError::FieldOverflow {
                field: "service_name_len",
            })?
            .to_le_bytes(),
    );
    bytes[40..48].copy_from_slice(&info.initial_generation.to_le_bytes());
    bytes[48..56].copy_from_slice(&info.client_token.to_le_bytes());
    bytes[SERVICE_NAME_OFFSET..SERVICE_NAME_OFFSET + service_name.len()]
        .copy_from_slice(service_name);
    Ok(bytes)
}

fn decode_info(bytes: [u8; BOOTSTRAP_BYTES]) -> Result<BootstrapInfo, BootstrapError> {
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != BOOTSTRAP_MAGIC {
        return Err(BootstrapError::InvalidMagic(magic));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != BOOTSTRAP_VERSION {
        return Err(BootstrapError::UnsupportedVersion(version));
    }
    let service_name_len = u16::from_le_bytes([bytes[36], bytes[37]]) as usize;
    if service_name_len > BOOTSTRAP_BYTES - SERVICE_NAME_OFFSET {
        return Err(BootstrapError::ServiceNameTooLong(service_name_len));
    }
    let service_name = String::from_utf8(
        bytes[SERVICE_NAME_OFFSET..SERVICE_NAME_OFFSET + service_name_len].to_vec(),
    )
    .map_err(|_| BootstrapError::InvalidServiceNameUtf8)?;
    Ok(BootstrapInfo {
        service_name,
        session_epoch: u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]),
        arena_bytes: usize::try_from(u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]))
        .map_err(|_| BootstrapError::FieldOverflow {
            field: "arena_bytes",
        })?,
        slot_capacity: u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize,
        slot_count: u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) as usize,
        slot_index: u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize,
        initial_generation: u64::from_le_bytes([
            bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
        ]),
        client_token: u64::from_le_bytes([
            bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
        ]),
    })
}

#[cfg(test)]
#[path = "../tests/unit/bootstrap.rs"]
mod tests;
