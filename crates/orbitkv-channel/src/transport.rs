use std::io::Read;
use std::os::fd::OwnedFd;
use std::os::unix::fs::MetadataExt;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use iceoryx2::active_request::ActiveRequest;
use iceoryx2::config::Config;
use iceoryx2::port::SendError;
use iceoryx2::port::client::RequestSendError;
use iceoryx2::port::listener::{Listener, ListenerWaitError};
use iceoryx2::port::notifier::Notifier;
use iceoryx2::prelude::*;
use iceoryx2::service::{messaging_pattern::MessagingPattern, service_hash::ServiceHash};
use rustix::event::{PollFd, PollFlags, Timespec, poll};
use thiserror::Error;

use crate::{Command, ProtocolError, Response, WireMessage};

type ThreadSafeIpcService = ipc_threadsafe::Service;
type IpcClient =
    iceoryx2::port::client::Client<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcServer =
    iceoryx2::port::server::Server<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcActiveRequest = ActiveRequest<ThreadSafeIpcService, WireMessage, (), WireMessage, ()>;
type IpcService = iceoryx2::service::port_factory::request_response::PortFactory<
    ThreadSafeIpcService,
    WireMessage,
    (),
    WireMessage,
    (),
>;

struct ProcessConfig {
    config: Config,
    directories: Vec<(std::path::PathBuf, u64, u64)>,
}

fn process_config() -> Result<&'static Config, TransportError> {
    static CONFIG: LazyLock<Result<ProcessConfig, String>> = LazyLock::new(|| {
        let Some(path) = std::env::var_os("ORBITKV_CHANNEL_CONFIG") else {
            return Ok(ProcessConfig {
                config: Config::global_config().clone(),
                directories: Vec::new(),
            });
        };
        let path = std::path::Path::new(&path);
        if !path.is_absolute() {
            return Err("ORBITKV_CHANNEL_CONFIG must name an absolute file".into());
        }
        let fd = rustix::fs::open(
            path,
            rustix::fs::OFlags::RDONLY
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::NONBLOCK,
            rustix::fs::Mode::empty(),
        )
        .map_err(|error| error.to_string())?;
        let metadata = rustix::fs::fstat(&fd).map_err(|error| error.to_string())?;
        if rustix::fs::FileType::from_raw_mode(metadata.st_mode)
            != rustix::fs::FileType::RegularFile
            || metadata.st_size < 0
            || metadata.st_size > 65536
        {
            return Err("channel configuration must be a regular file of at most 64 KiB".into());
        }
        let mut input = String::new();
        let file = std::fs::File::from(fd);
        (&file)
            .take(65537)
            .read_to_string(&mut input)
            .map_err(|error| error.to_string())?;
        if input.len() > 65536 {
            return Err("channel configuration exceeded its byte bound".into());
        }
        let after = rustix::fs::fstat(&file).map_err(|error| error.to_string())?;
        if after.st_dev != metadata.st_dev
            || after.st_ino != metadata.st_ino
            || after.st_size != metadata.st_size
            || after.st_mtime != metadata.st_mtime
            || after.st_mtime_nsec != metadata.st_mtime_nsec
            || after.st_ctime != metadata.st_ctime
            || after.st_ctime_nsec != metadata.st_ctime_nsec
        {
            return Err("channel configuration changed while being read".into());
        }
        let config: Config = toml::from_str(&input).map_err(|error| error.to_string())?;
        let root = std::path::Path::new(config.global.root_path().as_str());
        if !root.is_absolute()
            || !root.is_dir()
            || root.canonicalize().map_err(|error| error.to_string())? != root
            || config.global.prefix == Config::default().global.prefix
        {
            return Err(
                "explicit channel configuration needs an absolute root and private prefix".into(),
            );
        }
        let root_metadata = root.metadata().map_err(|error| error.to_string())?;
        let mut directories = vec![(root.to_path_buf(), root_metadata.dev(), root_metadata.ino())];
        for directory in [
            config.global.node.directory.as_str(),
            config.global.service.directory.as_str(),
        ] {
            let directory = std::path::Path::new(directory);
            if directory
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                return Err(
                    "channel node and service directories must remain inside the private root"
                        .into(),
                );
            }
            let mut current = root.to_path_buf();
            for component in directory.components() {
                current.push(component);
                let metadata = std::fs::symlink_metadata(&current).map_err(|error| {
                    format!("private channel directories must exist before Node creation: {error}")
                })?;
                if !metadata.is_dir()
                    || metadata.file_type().is_symlink()
                    || current.canonicalize().map_err(|error| error.to_string())? != current
                {
                    return Err("private channel directories cannot contain symlinks".into());
                }
                directories.push((current.clone(), metadata.dev(), metadata.ino()));
            }
        }
        Ok(ProcessConfig {
            config,
            directories,
        })
    });
    CONFIG
        .as_ref()
        .map_err(|error| TransportError::Node(error.clone()))
        .and_then(|owner| {
            for (path, device, inode) in &owner.directories {
                let metadata = path
                    .symlink_metadata()
                    .map_err(|error| TransportError::Node(error.to_string()))?;
                if !metadata.is_dir()
                    || metadata.file_type().is_symlink()
                    || (metadata.dev(), metadata.ino()) != (*device, *inode)
                    || path
                        .canonicalize()
                        .map_err(|error| TransportError::Node(error.to_string()))?
                        != *path
                {
                    return Err(TransportError::Node(
                        "private channel directory changed before Node creation".into(),
                    ));
                }
            }
            Ok(&owner.config)
        })
}

#[derive(Clone, Copy, Debug)]
pub struct CallOptions {
    pub timeout: Duration,
    pub spin_iterations: u32,
}

impl Default for CallOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            spin_iterations: 64,
        }
    }
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("invalid service name: {0}")]
    InvalidServiceName(String),
    #[error("failed to create iceoryx2 node: {0}")]
    Node(String),
    #[error("failed to open iceoryx2 service: {0}")]
    Service(String),
    #[error("failed to create iceoryx2 port: {0}")]
    Port(String),
    #[error("failed to start process channel thread: {0}")]
    Thread(String),
    #[error("failed to send process channel message: {0}")]
    Send(String),
    #[error("failed to send process channel request: {0}")]
    RequestSend(#[source] RequestSendError),
    #[error("failed to receive process channel message: {0}")]
    Receive(String),
    #[error("process channel request {request_id} timed out")]
    Timeout { request_id: u64 },
    #[error("Cache Manager process exited before request {request_id} completed")]
    PeerExited { request_id: u64 },
    #[error("response request id mismatch: expected {expected}, got {actual}")]
    RequestMismatch { expected: u64, actual: u64 },
    #[error("response session epoch mismatch: expected {expected}, got {actual}")]
    SessionMismatch { expected: u64, actual: u64 },
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

impl TransportError {
    pub(crate) fn request_not_submitted(&self) -> bool {
        matches!(
            self,
            Self::RequestSend(
                RequestSendError::ExceedsMaxActiveRequests
                    | RequestSendError::SendError(
                        SendError::LoanError(_) | SendError::ConnectionError(_)
                    )
            )
        )
    }
}

fn service_open_error(
    node: &iceoryx2::node::Node<ThreadSafeIpcService>,
    name: &ServiceName,
    pattern: MessagingPattern,
    error: impl std::fmt::Display,
) -> TransportError {
    let config = &node.config().global;
    let hash = ServiceHash::new::<
        <ThreadSafeIpcService as iceoryx2::service::Service>::ServiceNameHasher,
    >(name, pattern);
    let path = std::path::PathBuf::from(config.service_dir().as_str()).join(format!(
        "{}{}{}",
        config.prefix, hash, config.service.static_config_storage_suffix
    ));
    let metadata = std::fs::metadata(&path).map(|metadata| {
        use std::os::unix::fs::MetadataExt;
        (metadata.dev(), metadata.ino(), metadata.len())
    });
    let mount_namespace = std::fs::read_link("/proc/self/ns/mnt");
    TransportError::Service(format!(
        "{error}; name={name:?} pattern={pattern:?} pid={} static_config={path:?} metadata={metadata:?} mount_namespace={mount_namespace:?}",
        std::process::id()
    ))
}

pub struct TransportClient {
    client: IpcClient,
    request_notifier: Notifier<ThreadSafeIpcService>,
    _service: IpcService,
    _node: iceoryx2::node::Node<ThreadSafeIpcService>,
}

impl TransportClient {
    pub fn connect(service_name: &str) -> Result<Self, TransportError> {
        let node = NodeBuilder::new()
            .config(process_config()?)
            .create::<ThreadSafeIpcService>()
            .map_err(|error| TransportError::Node(error.to_string()))?;
        let name = service_name.try_into().map_err(
            |error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            },
        )?;
        let service = node
            .service_builder(&name)
            .request_response::<WireMessage, WireMessage>()
            .open()
            .map_err(|error| {
                service_open_error(&node, &name, MessagingPattern::RequestResponse, error)
            })?;
        if service.static_config().max_servers() != 1 {
            return Err(TransportError::Service(
                "process channel requests require exactly one Manager recipient".into(),
            ));
        }
        let client = service
            .client_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        let request_name: ServiceName = format!("{service_name}/requests")
            .as_str()
            .try_into()
            .map_err(|error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            })?;
        let request_events =
            node.service_builder(&request_name)
                .event()
                .open()
                .map_err(|error| {
                    service_open_error(&node, &request_name, MessagingPattern::Event, error)
                })?;
        let request_notifier = request_events
            .notifier_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        Ok(Self {
            client,
            request_notifier,
            _service: service,
            _node: node,
        })
    }

    pub fn call(&self, command: Command, options: CallOptions) -> Result<Response, TransportError> {
        self.call_inner(command, options, None)
    }

    /// A publish cannot time out while the peer still owns its source pages.
    /// The pidfd distinguishes a dead Cache Manager from an IPC stall, so an
    /// ambiguous stall keeps the caller blocked and its pages pinned.
    pub fn call_until_peer_exit(
        &self,
        command: Command,
        options: CallOptions,
        peer: &OwnedFd,
        reply_notification: &OwnedFd,
    ) -> Result<Response, TransportError> {
        self.call_inner(command, options, Some((peer, reply_notification)))
    }

    fn call_inner(
        &self,
        command: Command,
        options: CallOptions,
        publish: Option<(&OwnedFd, &OwnedFd)>,
    ) -> Result<Response, TransportError> {
        let pending = self
            .client
            .send_copy(command.encode())
            .map_err(TransportError::RequestSend)?;
        // The request is already submitted. A missed doorbell cannot return a
        // pre-submission error or let a Publish caller release its GPU pages.
        // Manager maintenance also rechecks the queue, independently of wakes.
        match self.request_notifier.notify() {
            Ok(0) => log::warn!(
                "request {} is queued but its Manager notification has no listener",
                command.request_id
            ),
            Ok(_) => {}
            Err(error) => log::warn!(
                "request {} is queued but its Manager notification failed: {error}",
                command.request_id
            ),
        }
        let started = Instant::now();
        let deadline = publish.is_none().then(|| started + options.timeout);
        let mut next_warning = started + options.timeout;
        let mut peer_gone = false;
        let mut spins = 0;
        loop {
            if publish.is_some() && Instant::now() >= next_warning {
                log::warn!(
                    "publish request {} is still pending after {:?}; retaining source pages without a matched completion",
                    command.request_id,
                    started.elapsed()
                );
                next_warning = Instant::now() + Duration::from_secs(60);
            }
            if let Some(message) = pending
                .receive()
                .map_err(|error| TransportError::Receive(error.to_string()))?
            {
                let response = Response::decode(*message)?;
                if response.request_id != command.request_id {
                    return Err(TransportError::RequestMismatch {
                        expected: command.request_id,
                        actual: response.request_id,
                    });
                }
                if response.status != crate::StatusCode::StaleSession
                    && response.session_epoch != command.session_epoch
                {
                    return Err(TransportError::SessionMismatch {
                        expected: command.session_epoch,
                        actual: response.session_epoch,
                    });
                }
                return Ok(response);
            }
            // A response queued before Manager exit is still authoritative.
            // Always inspect it before acting on the previous poll's pidfd.
            if peer_gone {
                return Err(TransportError::PeerExited {
                    request_id: command.request_id,
                });
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(TransportError::Timeout {
                    request_id: command.request_id,
                });
            }
            if spins < options.spin_iterations {
                spins += 1;
                std::hint::spin_loop();
            } else if let Some((peer, reply_notification)) = publish {
                let mut fds = [
                    PollFd::new(reply_notification, PollFlags::IN),
                    PollFd::new(peer, PollFlags::IN),
                ];
                // Notifications are hints. A lost wake only delays the next
                // response check; it cannot end Publish or release its pages.
                let timeout = Timespec {
                    tv_sec: 0,
                    tv_nsec: 10_000_000,
                };
                match poll(&mut fds, Some(&timeout)) {
                    Ok(_) => {}
                    Err(rustix::io::Errno::INTR) => continue,
                    Err(error) => return Err(TransportError::Receive(error.to_string())),
                }
                peer_gone = fds[1].revents().contains(PollFlags::IN);
                if fds
                    .iter()
                    .any(|fd| fd.revents().intersects(PollFlags::ERR | PollFlags::NVAL))
                {
                    return Err(TransportError::Receive(
                        "invalid Publish notification or Manager pidfd".to_string(),
                    ));
                }
                if fds[0].revents().contains(PollFlags::IN) {
                    let mut value = [0u8; 8];
                    match rustix::io::read(reply_notification, &mut value) {
                        Ok(8) => {}
                        Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => {}
                        Ok(_) => {
                            return Err(TransportError::Receive(
                                "invalid Publish notification length".to_string(),
                            ));
                        }
                        Err(error) => return Err(TransportError::Receive(error.to_string())),
                    }
                }
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// No logical error or process-liveness observation ends submitted device work.
    pub(crate) fn retain_unproven_publish(request_id: u64) -> ! {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr().lock(),
            "Publish {request_id} has no physical completion proof; retaining caller and source owners"
        );
        loop {
            std::thread::park_timeout(Duration::from_secs(60));
        }
    }
}

pub(crate) fn peer_exited(peer: &OwnedFd) -> Result<bool, TransportError> {
    let mut fds = [PollFd::new(peer, PollFlags::IN)];
    let timeout = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    poll(&mut fds, Some(&timeout)).map_err(|error| TransportError::Receive(error.to_string()))?;
    if fds[0]
        .revents()
        .intersects(PollFlags::ERR | PollFlags::NVAL)
    {
        return Err(TransportError::Receive(
            "invalid Manager process descriptor".to_string(),
        ));
    }
    Ok(fds[0].revents().contains(PollFlags::IN))
}

pub struct TransportServer {
    server: IpcServer,
    request_listener: Listener<ThreadSafeIpcService>,
    _service: IpcService,
    _node: iceoryx2::node::Node<ThreadSafeIpcService>,
}

/// Owns one iceoryx2 request until its response is sent. The Cache Manager can move
/// this handle to a worker so a long GPU copy does not stop the dispatcher.
pub struct DeferredResponse(IpcActiveRequest);

impl DeferredResponse {
    pub fn send(self, response: Response) -> Result<(), TransportError> {
        self.0
            .send_copy(response.encode())
            .map_err(|error| TransportError::Send(error.to_string()))
    }

    /// Publish the response before waking its waiter. Notification failure is
    /// not a send failure: the client also rechecks the response periodically.
    pub fn send_and_notify(
        self,
        response: Response,
        notification: &OwnedFd,
    ) -> Result<(), TransportError> {
        self.send(response)?;
        match rustix::io::write(notification, &1u64.to_ne_bytes()) {
            Ok(_) | Err(rustix::io::Errno::AGAIN) => {}
            Err(error) => log::warn!(
                "response {} is queued but its Publish notification failed: {error}",
                response.request_id
            ),
        }
        Ok(())
    }
}

impl TransportServer {
    pub fn bind(service_name: &str) -> Result<Self, TransportError> {
        let node = NodeBuilder::new()
            .config(process_config()?)
            .create::<ThreadSafeIpcService>()
            .map_err(|error| TransportError::Node(error.to_string()))?;
        let name = service_name.try_into().map_err(
            |error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            },
        )?;
        let service = node
            .service_builder(&name)
            .request_response::<WireMessage, WireMessage>()
            .max_servers(1)
            // Each CacheClient can own separate control and Publish ports.
            // Keep room for concurrent producers plus foreground requests.
            .max_clients(64)
            .max_nodes(65)
            .create()
            .map_err(|error| TransportError::Service(error.to_string()))?;
        let server = service
            .server_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        let request_name: ServiceName = format!("{service_name}/requests")
            .as_str()
            .try_into()
            .map_err(|error: iceoryx2::service::service_name::ServiceNameError| {
                TransportError::InvalidServiceName(error.to_string())
            })?;
        let request_events = node
            .service_builder(&request_name)
            .event()
            .max_nodes(65)
            .max_notifiers(64)
            .max_listeners(1)
            .event_id_max_value(0)
            .disable_notifier_created_event()
            .disable_notifier_dropped_event()
            .disable_notifier_dead_event()
            .create()
            .map_err(|error| TransportError::Service(error.to_string()))?;
        let request_listener = request_events
            .listener_builder()
            .create()
            .map_err(|error| TransportError::Port(error.to_string()))?;
        Ok(Self {
            server,
            request_listener,
            _service: service,
            _node: node,
        })
    }

    /// Briefly spin for a hot producer, then sleep until a request doorbell or
    /// the next maintenance deadline.
    /// A wake is only a hint; callers always retry the authoritative request queue.
    pub fn wait_for_request(&self, timeout: Duration) -> Result<(), TransportError> {
        let started = Instant::now();
        // Clear old/coalesced notifications before the queue check. A producer
        // enqueues before notifying: arrivals before this check are visible in
        // the queue, and later notifications persist through timed_wait.
        match self.request_listener.try_wait(|_| {}) {
            Ok(_) => {}
            Err(ListenerWaitError::InterruptSignal) => return Ok(()),
            Err(error) => return Err(TransportError::Receive(error.to_string())),
        }
        for _ in 0..64 {
            if self
                .server
                .has_requests()
                .map_err(|error| TransportError::Receive(error.to_string()))?
            {
                return Ok(());
            }
            std::hint::spin_loop();
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        // iceoryx2's Unix datagram wait truncates to SO_RCVTIMEO microseconds.
        // A zero timeval disables that timeout and would block maintenance.
        if remaining < Duration::from_micros(1) {
            return Ok(());
        }
        match self.request_listener.timed_wait(|_| {}, remaining) {
            Ok(_) | Err(ListenerWaitError::InterruptSignal) => Ok(()),
            Err(error) => Err(TransportError::Receive(error.to_string())),
        }
    }

    pub fn try_serve(
        &self,
        handler: impl FnOnce(Command) -> Response,
    ) -> Result<bool, TransportError> {
        let Some(request) = self
            .server
            .receive()
            .map_err(|error| TransportError::Receive(error.to_string()))?
        else {
            return Ok(false);
        };
        let command = Command::decode(*request)?;
        request
            .send_copy(handler(command).encode())
            .map_err(|error| TransportError::Send(error.to_string()))?;
        Ok(true)
    }

    /// Reject commands created for an older Cache Manager session before dispatch.
    pub fn try_serve_for_epoch(
        &self,
        session_epoch: u64,
        handler: impl FnOnce(Command) -> Response,
    ) -> Result<bool, TransportError> {
        self.try_serve(|command| {
            if command.session_epoch == session_epoch {
                handler(command)
            } else {
                Response {
                    status: crate::StatusCode::StaleSession,
                    request_id: command.request_id,
                    session_epoch,
                    descriptor: command.descriptor,
                    value0: 0,
                    value1: 0,
                }
            }
        })
    }

    /// The handler may retain the reply handle and send after this call returns.
    /// A stale epoch is always answered before it reaches the handler.
    pub fn try_serve_deferred_for_epoch(
        &self,
        session_epoch: u64,
        handler: impl FnOnce(Command, DeferredResponse) -> Result<(), TransportError>,
    ) -> Result<bool, TransportError> {
        let Some(request) = self
            .server
            .receive()
            .map_err(|error| TransportError::Receive(error.to_string()))?
        else {
            return Ok(false);
        };
        let command = Command::decode(*request)?;
        let reply = DeferredResponse(request);
        if command.session_epoch != session_epoch {
            reply.send(Response {
                status: crate::StatusCode::StaleSession,
                request_id: command.request_id,
                session_epoch,
                descriptor: command.descriptor,
                value0: 0,
                value1: 0,
            })?;
        } else {
            handler(command, reply)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "../tests/unit/transport.rs"]
mod tests;
