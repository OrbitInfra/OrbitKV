use std::io::Write;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rustix::net::sockopt::socket_peercred;
use rustix::process::{PidfdFlags, pidfd_open};
use thiserror::Error;

use crate::lifecycle::{
    LifecycleCommand, LifecycleHeader, LifecycleReply, MAX_LIFECYCLE_PAYLOAD,
    MAX_QUERY_TARGET_PAYLOAD, receive_lifecycle_reply,
};
use crate::{
    BootstrapClient, BootstrapError, CacheProtocolError, CallOptions, Command, CommandCode,
    CompletionObservationRequest, PublishRequest, QueryBundleResponse,
    RESPONSE_FLAG_REQUEST_CONSUMED, ReleaseRequest, RestoreRequest, RestoreResponse, RestoreState,
    StatusCode, TransportClient, TransportError,
};

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error(transparent)]
    Bootstrap(#[from] BootstrapError),
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Recovery(#[from] orbitkv_state::RecoveryError),
    #[error(transparent)]
    Codec(#[from] CacheProtocolError),
    #[error(transparent)]
    Completion(#[from] crate::CompletionError),
    #[error("cache request returned {0:?}")]
    Status(StatusCode),
    #[error("cache session is ambiguous after a failed call; reconnect required")]
    SessionRequiresReconnect,
    #[error("restore operation {operation_id} timed out")]
    RestoreTimeout { operation_id: u64 },
    #[error("cannot observe Cache Manager process death; refusing to start a GPU transfer")]
    GpuPeerUnobservable,
    #[error("Cache Manager rejected lifecycle operation ({code}): {message}")]
    Lifecycle { code: u16, message: String },
    #[error("Publish activity observation is temporarily unavailable")]
    PublishObservationUnavailable,
    #[error("Publish caller admission is full")]
    PublishAdmissionFull,
}

const MAX_PUBLISH_CALLERS: usize = 1024;

#[derive(Clone, Debug, Default)]
pub struct PublishProgress {
    pub active: usize,
    pub retained: usize,
    pub active_interval: Duration,
    pub overdue: bool,
    pub manager_alive: Option<bool>,
}

#[derive(Default)]
struct PublishActivity {
    active: usize,
    retained: usize,
    started: Option<Instant>,
}

pub struct ChannelClient {
    bootstrap: BootstrapClient,
    client: TransportClient,
    options: CallOptions,
    call_lock: Mutex<()>,
    lifecycle_lock: Mutex<()>,
    poisoned: AtomicBool,
    peer_process: Option<OwnedFd>,
    publish_activity: Mutex<PublishActivity>,
}

impl ChannelClient {
    pub fn connect(
        bootstrap_socket: impl AsRef<Path>,
        options: CallOptions,
    ) -> Result<Self, ChannelError> {
        let bootstrap = BootstrapClient::connect(bootstrap_socket)?;
        let peer_process = socket_peercred(bootstrap.stream())
            .ok()
            .and_then(|credentials| pidfd_open(credentials.pid, PidfdFlags::empty()).ok());
        bootstrap
            .stream()
            .set_read_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        bootstrap
            .stream()
            .set_write_timeout(Some(options.timeout))
            .map_err(BootstrapError::Io)?;
        let client = TransportClient::connect(&bootstrap.info_ref().service_name)?;
        Ok(Self {
            bootstrap,
            client,
            options,
            call_lock: Mutex::new(()),
            lifecycle_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            peer_process,
            publish_activity: Mutex::new(PublishActivity::default()),
        })
    }

    pub fn service_name(&self) -> &str {
        &self.bootstrap.info_ref().service_name
    }

    /// Observe the authenticated Manager process, independently of channel
    /// health or a stalled request. This is not a device-operation fence.
    pub fn peer_alive(&self) -> Result<bool, ChannelError> {
        let peer = self
            .peer_process
            .as_ref()
            .ok_or(ChannelError::GpuPeerUnobservable)?;
        crate::transport::peer_exited(peer)
            .map(|exited| !exited)
            .map_err(Into::into)
    }

    /// Local activity plus pidfd liveness; never a device completion receipt.
    pub fn publish_progress(&self) -> Result<PublishProgress, ChannelError> {
        let activity = self
            .publish_activity
            .try_lock()
            .map_err(|_| ChannelError::PublishObservationUnavailable)?;
        let elapsed = activity
            .started
            .map_or(Duration::ZERO, |started| started.elapsed());
        Ok(PublishProgress {
            active: activity.active,
            retained: activity.retained,
            active_interval: elapsed,
            overdue: activity.active != 0 && elapsed >= self.options.timeout,
            manager_alive: self
                .peer_process
                .as_ref()
                .map(|peer| crate::transport::peer_exited(peer).map(|exited| !exited))
                .transpose()?,
        })
    }

    /// Registration and liveness metadata use UDS, independently of the hot descriptor slot.
    pub fn lifecycle(
        &self,
        command: LifecycleCommand,
        payload: &[u8],
    ) -> Result<LifecycleReply, ChannelError> {
        let _guard = self
            .lifecycle_lock
            .lock()
            .map_err(|_| ChannelError::SessionRequiresReconnect)?;
        if self.poisoned.load(Ordering::Acquire) {
            return Err(ChannelError::SessionRequiresReconnect);
        }
        let exchange = || -> std::io::Result<(u16, LifecycleReply)> {
            let header = LifecycleHeader {
                code: command as u16,
                epoch: self.session_epoch(),
                payload_len: payload.len(),
            }
            .encode()?;
            let mut stream = self.bootstrap.stream();
            let timeout = match command {
                LifecycleCommand::Register | LifecycleCommand::Unregister => {
                    self.options.timeout.max(Duration::from_secs(120))
                }
                _ => self.options.timeout,
            };
            stream.set_read_timeout(Some(timeout))?;
            stream.write_all(&header)?;
            stream.write_all(payload)?;
            let payload_limit = if matches!(command, LifecycleCommand::ExportQueryTarget) {
                MAX_QUERY_TARGET_PAYLOAD
            } else {
                MAX_LIFECYCLE_PAYLOAD
            };
            let (header, reply) = receive_lifecycle_reply(stream, payload_limit)?;
            if header.epoch != self.session_epoch() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stale lifecycle session",
                ));
            }
            let unexpected = match command {
                LifecycleCommand::Register => false,
                LifecycleCommand::ExportQueryTarget => !reply.fds.is_empty(),
                _ => !reply.payload.is_empty() || !reply.fds.is_empty(),
            };
            if unexpected && header.code == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unexpected lifecycle attachments",
                ));
            }
            Ok((header.code, reply))
        };
        match exchange() {
            Ok((0, reply)) => Ok(reply),
            Ok((code, reply)) => Err(ChannelError::Lifecycle {
                code,
                message: String::from_utf8_lossy(&reply.payload).into_owned(),
            }),
            Err(error) => {
                self.close();
                // A partial frame must never be reused as a new request.
                Err(BootstrapError::Io(error).into())
            }
        }
    }

    /// End the process-channel session explicitly, without waiting for client destruction.
    pub fn close(&self) {
        self.poisoned.store(true, Ordering::Release);
        let _ = self.bootstrap.stream().shutdown(std::net::Shutdown::Both);
    }

    pub fn session_epoch(&self) -> u64 {
        self.bootstrap.info_ref().session_epoch
    }

    pub(crate) fn session_token(&self) -> u64 {
        self.bootstrap.info_ref().client_token
    }

    /// Borrowed notification descriptor. It remains valid only while this
    /// client is alive and must not be closed by the caller.
    pub fn notification_fd(&self) -> RawFd {
        self.bootstrap.notification_fd().as_raw_fd()
    }

    pub fn wait_for_notification(&self, timeout: Duration) -> Result<bool, ChannelError> {
        self.bootstrap
            .wait_for_notification(timeout)
            .map_err(Into::into)
    }

    pub fn query_bundle(
        &self,
        request_id: u64,
        request: &crate::QueryCommand,
    ) -> Result<QueryBundleResponse, ChannelError> {
        let payload = request.encode()?;
        let payload =
            self.call_descriptor(CommandCode::QueryBundle, request_id, &payload, 0, None)?;
        match QueryBundleResponse::decode(&payload) {
            Ok(response) => Ok(response),
            Err(error) => {
                self.close();
                Err(error.into())
            }
        }
    }

    pub fn query_shards(
        &self,
        request_id: u64,
        request: &crate::QueryCommand,
    ) -> Result<crate::ShardQueryResponse, ChannelError> {
        let payload = request.encode()?;
        let payload =
            self.call_descriptor(CommandCode::QueryShards, request_id, &payload, 0, None)?;
        crate::ShardQueryResponse::decode(&payload).map_err(|error| {
            self.close();
            error.into()
        })
    }

    pub fn cancel_shard_query(
        &self,
        request_id: u64,
        request: &crate::CancelQueryRequest,
    ) -> Result<(), ChannelError> {
        self.call_descriptor(
            CommandCode::CancelShardQuery,
            request_id,
            &request.encode()?,
            0,
            None,
        )?;
        Ok(())
    }

    pub fn release_shard_query(
        &self,
        request_id: u64,
        control_id: Vec<u8>,
    ) -> Result<(), ChannelError> {
        if control_id.len() != 16 || control_id.iter().all(|byte| *byte == 0) {
            return Err(CacheProtocolError::InvalidShardPayload.into());
        }
        self.call_descriptor(
            CommandCode::ReleaseShardQuery,
            request_id,
            &control_id,
            0,
            None,
        )?;
        Ok(())
    }

    pub fn cancel_query(
        &self,
        request_id: u64,
        request: &crate::CancelQueryRequest,
    ) -> Result<(), ChannelError> {
        let payload = request.encode()?;
        self.call_descriptor(CommandCode::CancelQuery, request_id, &payload, 0, None)?;
        Ok(())
    }

    pub fn observe_completion(
        &self,
        request_id: u64,
        observation: &CompletionObservationRequest,
    ) -> Result<(), ChannelError> {
        let payload = observation.encode()?;
        self.call_descriptor(
            CommandCode::ObserveCompletion,
            request_id,
            &payload,
            0,
            None,
        )?;
        Ok(())
    }

    pub fn release(&self, request_id: u64, lease: Vec<u8>) -> Result<(), ChannelError> {
        let payload = ReleaseRequest { lease }.encode()?;
        let _ = self.call_descriptor(CommandCode::Release, request_id, &payload, 0, None)?;
        Ok(())
    }

    pub fn publish(&self, request_id: u64, request: &PublishRequest) -> Result<(), ChannelError> {
        let peer = self
            .peer_process
            .as_ref()
            .ok_or(ChannelError::GpuPeerUnobservable)?;
        let payloads = publish_payloads(request, self.bootstrap.info_ref().slot_capacity)?;
        {
            let mut activity = self
                .publish_activity
                .lock()
                .map_err(|_| ChannelError::PublishObservationUnavailable)?;
            if activity.active >= MAX_PUBLISH_CALLERS {
                return Err(ChannelError::PublishAdmissionFull);
            }
            activity.active += 1;
            activity.started.get_or_insert_with(Instant::now);
        }
        let result = (|| {
            for payload in payloads {
                // One logical publish can use several descriptor generations.
                // Every chunk retains the source pages until its D2H completes.
                let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.call_descriptor(CommandCode::Publish, request_id, &payload, 0, Some(peer))
                }));
                match call {
                    Ok(result) => result?,
                    Err(_) => {
                        self.mark_publish_retained();
                        TransportClient::retain_unproven_publish(request_id)
                    }
                };
            }
            Ok(())
        })();
        let mut activity = self
            .publish_activity
            .lock()
            .map_err(|_| ChannelError::PublishObservationUnavailable)?;
        activity.active -= 1;
        if activity.active == 0 {
            activity.started = None;
        }
        result
    }

    fn mark_publish_retained(&self) {
        let mut activity = self
            .publish_activity
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        activity.retained += 1;
    }

    pub fn restore_submit(
        &self,
        request_id: u64,
        request: &RestoreRequest,
    ) -> Result<u64, ChannelError> {
        if self.peer_process.is_none() {
            return Err(ChannelError::GpuPeerUnobservable);
        }
        let payload = request.encode()?;
        let completions = self.bootstrap.completions();
        let operation_id = completions.reserve()?;
        if let Err(error) = self.call_descriptor(
            CommandCode::Restore,
            request_id,
            &payload,
            operation_id,
            None,
        ) {
            if completions.cancel(operation_id)? {
                // Winning cancellation proves the Manager cannot start this
                // operation, even if its queued descriptor arrives later.
                return Err(error);
            }
            // Claim won: the Manager owns execution and the shared result.
            // An ambiguous ACK cannot discard the only handle to that work.
            log::warn!("Restore {operation_id} claimed despite submission error: {error}");
        }
        Ok(operation_id)
    }

    pub fn restore_poll(&self, operation_id: u64) -> Result<RestoreResponse, ChannelError> {
        let response = self.bootstrap.completions().poll(operation_id)?;
        // Descriptor admission may be closed while previously claimed DMA
        // still runs. A socket HUP is not process death or transfer completion.
        if response.state == RestoreState::Pending && !self.peer_alive()? {
            self.close();
            // A terminal publication can race the first read and process exit.
            let final_response = self.bootstrap.completions().poll(operation_id)?;
            if final_response.state != RestoreState::Pending {
                return Ok(final_response);
            }
            match self.bootstrap.completions().state(operation_id)? {
                crate::GrantState::Managed => {
                    return Ok(RestoreResponse {
                        message:
                            "Cache Manager exited without a Restore completion; resources retained"
                                .into(),
                        ..final_response
                    });
                }
                crate::GrantState::Active | crate::GrantState::ActiveMore => {}
                _ => return Err(ChannelError::SessionRequiresReconnect),
            }
        }
        Ok(response)
    }

    pub fn restore_completions(&self) -> Arc<crate::RestoreCompletions> {
        Arc::clone(self.bootstrap.completions())
    }

    fn call_descriptor(
        &self,
        code: CommandCode,
        request_id: u64,
        payload: &[u8],
        operation_id: u64,
        peer: Option<&OwnedFd>,
    ) -> Result<Vec<u8>, ChannelError> {
        let mut retain = false;
        let result = (|| {
            let _call = self
                .call_lock
                .lock()
                .map_err(|_| BootstrapError::Arena(crate::ArenaError::Poisoned))?;
            if self.poisoned.load(Ordering::Acquire) {
                return Err(ChannelError::SessionRequiresReconnect);
            }
            let descriptor = self.bootstrap.write_request(payload)?;
            let info = self.bootstrap.info_ref();
            let command = Command {
                code,
                request_id,
                session_epoch: info.session_epoch,
                descriptor,
                arg0: info.client_token,
                arg1: operation_id,
            };
            let call = match peer {
                Some(peer) => self.client.call_until_peer_exit(
                    command,
                    self.options,
                    peer,
                    self.bootstrap.reply_notification_fd(),
                ),
                None => self.client.call(command, self.options),
            };
            let response = match call {
                Ok(response) => response,
                Err(error) => {
                    self.close();
                    retain = peer.is_some() && !error.request_not_submitted();
                    return Err(error.into());
                }
            };
            // Invalid replies cannot end source ownership after device submission.
            let mut ambiguous = || {
                self.close();
                retain = peer.is_some();
            };
            if response.status != StatusCode::Ok {
                if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED != 0 {
                    self.bootstrap
                        .complete_request(descriptor)
                        .inspect_err(|_| ambiguous())?;
                } else {
                    self.close();
                    retain = peer.is_some() && response.status != StatusCode::StaleSession;
                }
                return Err(ChannelError::Status(response.status));
            }
            if response.value1 & RESPONSE_FLAG_REQUEST_CONSUMED == 0 {
                ambiguous();
                return Err(ChannelError::SessionRequiresReconnect);
            }
            if code == CommandCode::Restore {
                self.bootstrap
                    .complete_request(descriptor)
                    .inspect_err(|_| ambiguous())?;
                return Ok(Vec::new());
            }
            let payload = self
                .bootstrap
                .read_response(descriptor, response.descriptor)
                .inspect_err(|_| ambiguous())?;
            Ok(payload)
        })();
        if retain {
            self.mark_publish_retained();
            TransportClient::retain_unproven_publish(request_id);
        }
        result
    }
}

fn publish_payloads(
    request: &PublishRequest,
    capacity: usize,
) -> Result<Vec<Vec<u8>>, ChannelError> {
    let payload_len = request.encoded_len(None)?;
    if payload_len <= capacity {
        return Ok(vec![request.encode()?]);
    }
    let too_large = |len| {
        ChannelError::Bootstrap(BootstrapError::Arena(crate::ArenaError::PayloadTooLarge {
            len,
            capacity,
        }))
    };
    let block_count = request
        .layers
        .iter()
        .map(|layer| layer.block_ids.len())
        .max()
        .unwrap_or(0);
    if block_count == 0 {
        return Err(too_large(payload_len));
    }

    let mut payloads = Vec::new();
    let mut start = 0;
    while start < block_count {
        let mut end = start + 1;
        let mut chunk_len = request.encoded_len(Some(start..end))?;
        if chunk_len > capacity {
            return Err(too_large(chunk_len));
        }
        // The first block accounts for every layer active at this start.
        // Later blocks add only IDs/hashes; ragged layers can end, not join.
        while end < block_count {
            let block_len = request.block_payload_len(end)?;
            if block_len > capacity - chunk_len {
                break;
            }
            chunk_len += block_len;
            end += 1;
        }
        payloads.push(request.encode_range(Some(start..end))?);
        start = end;
    }
    Ok(payloads)
}

#[cfg(test)]
#[path = "../tests/unit/client.rs"]
mod tests;
