//! Lifecycle metadata over the same authenticated UDS that bootstraps local IPC.
use std::collections::HashMap;
use std::os::fd::AsFd;
use std::time::Duration;

use orbitkv_channel::lifecycle::{
    LIFECYCLE_HEADER_BYTES, LifecycleCommand, LifecycleHeader, MAX_LIFECYCLE_PAYLOAD,
    send_lifecycle_fds,
};
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::watch;

use crate::cache::lifecycle::{ControlError, LifecycleService};
use crate::cache::query_control::QueryControlService;
use crate::cache::shards::ShardQueries;
use crate::proto::engine::{RegisterContextRequest, SessionRequest, UnregisterRequest};

pub(crate) async fn serve(
    stream: std::os::unix::net::UnixStream,
    epoch: u64,
    lifecycle: LifecycleService,
    mut shutdown: watch::Receiver<bool>,
    query_control: Option<QueryControlService>,
    shards: Option<ShardQueries>,
    token: u64,
) {
    let connection = stream.try_clone().ok();
    let mut owners = HashMap::new();
    let result = async {
        let mut stream = UnixStream::from_std(stream)?;
        loop {
            if *shutdown.borrow_and_update() {
                return Ok::<(), std::io::Error>(());
            }
            let mut bytes = [0; LIFECYCLE_HEADER_BYTES];
            tokio::select! {
                _ = wait_for_shutdown(&mut shutdown) => return Ok::<(), std::io::Error>(()),
                result = stream.read_exact(&mut bytes) => { result?; }
            }
            let header = LifecycleHeader::decode(bytes)?;
            if header.epoch != epoch {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stale lifecycle epoch",
                ));
            }
            let command = LifecycleCommand::try_from(header.code)?;
            let mut payload = vec![0; header.payload_len];
            tokio::select! {
                _ = wait_for_shutdown(&mut shutdown) => return Ok(()),
                result = tokio::time::timeout(
                    Duration::from_secs(10),
                    stream.read_exact(&mut payload),
                ) => { result??; }
            }
            #[cfg(test)]
            super::test_pause::pause("lifecycle_after_request").await;
            let result = dispatch(
                command,
                &payload,
                &lifecycle,
                &mut owners,
                query_control.as_ref(),
                shards.as_ref(),
                token,
            )
            .await;
            let (code, body, arenas) = match result {
                Ok((body, arenas)) => (0, body, arenas),
                Err(error) => (error.code, error.message.into_bytes(), Vec::new()),
            };
            if body.len() > MAX_LIFECYCLE_PAYLOAD {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "lifecycle response exceeds limit",
                ));
            }
            let response = LifecycleHeader {
                code,
                epoch,
                payload_len: body.len(),
            }
            .encode()?;
            tokio::time::timeout(Duration::from_secs(120), async {
                let fds = arenas
                    .iter()
                    .map(|arena| arena.fd.as_fd())
                    .collect::<Vec<_>>();
                stream
                    .async_io(tokio::io::Interest::WRITABLE, || {
                        send_lifecycle_fds(&stream, &fds)
                    })
                    .await?;
                stream.write_all(&response).await?;
                stream.write_all(&body).await
            })
            .await??;
        }
    }
    .await;
    if let Err(error) = result {
        log::debug!("Process-channel lifecycle connection closed: {error}");
    }
    if let Some(connection) = connection {
        let _ = connection.shutdown(std::net::Shutdown::Both);
    }
    if let Some(shards) = shards {
        shards.close_session(token);
    }
    for (instance, token) in owners {
        lifecycle.close_session(&instance, token).await;
    }
}

async fn wait_for_shutdown(shutdown: &mut watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow_and_update() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

async fn dispatch(
    command: LifecycleCommand,
    payload: &[u8],
    lifecycle: &LifecycleService,
    owners: &mut HashMap<String, u64>,
    query_control: Option<&QueryControlService>,
    shards: Option<&ShardQueries>,
    token: u64,
) -> Result<(Vec<u8>, Vec<orbitkv_core::PayloadArena>), ControlError> {
    match command {
        LifecycleCommand::ConfigureShardQueries => {
            let request = crate::proto::engine::ConfigureShardQueriesRequest::decode(payload)
                .map_err(|error| ControlError::invalid_argument(error.to_string()))?;
            let control = query_control.ok_or_else(|| {
                ControlError::invalid_argument("registered query control is not enabled")
            })?;
            shards
                .ok_or_else(|| ControlError::invalid_argument("shard queries are not enabled"))?
                .configure(token, request, control)
                .map_err(|error| ControlError::invalid_argument(error.to_string()))?;
        }
        LifecycleCommand::ExportQueryTarget => {
            let request = SessionRequest::decode(payload)
                .map_err(|error| ControlError::invalid_argument(error.to_string()))?;
            let control = query_control.ok_or_else(|| {
                ControlError::invalid_argument("registered query control is not enabled")
            })?;
            let target = control
                .export(request)
                .map_err(|error| ControlError::invalid_argument(error.message()))?;
            return Ok((target.encode_to_vec(), Vec::new()));
        }
        LifecycleCommand::Health => {
            if !payload.is_empty() {
                return Err(ControlError::invalid_argument(
                    "health payload must be empty",
                ));
            }
        }
        LifecycleCommand::Register => {
            let request = RegisterContextRequest::decode(payload)
                .map_err(|e| ControlError::invalid_argument(e.to_string()))?;
            let arenas = lifecycle.payload_arenas()?;
            if arenas.len() > 64 {
                return Err(ControlError::invalid_argument(
                    "payload arena count exceeds session limit",
                ));
            }
            let manager_device_id = lifecycle
                .register(crate::wire::registration(request))
                .await?;
            let mut body = Vec::with_capacity(4 + arenas.len() * 16);
            body.extend_from_slice(&manager_device_id.to_le_bytes());
            for arena in &arenas {
                body.extend_from_slice(&arena.id.to_le_bytes());
                body.extend_from_slice(&arena.size.to_le_bytes());
            }
            return Ok((body, arenas));
        }
        LifecycleCommand::Unregister => {
            let request = UnregisterRequest::decode(payload)
                .map_err(|e| ControlError::invalid_argument(e.to_string()))?;
            if request.instance_id.is_empty() {
                return Err(ControlError::invalid_argument(
                    "instance_id must not be empty",
                ));
            }
            lifecycle.cleanup(&request.instance_id).await?;
        }
        LifecycleCommand::Session => {
            let request = SessionRequest::decode(payload)
                .map_err(|e| ControlError::invalid_argument(e.to_string()))?;
            if request.instance_id.is_empty() || request.tp_size == 0 || request.world_size == 0 {
                return Err(ControlError::invalid_argument(
                    "session requires instance_id and nonzero topology",
                ));
            }
            if !owners.is_empty() && !owners.contains_key(&request.instance_id) {
                return Err(ControlError::invalid_argument(
                    "one inference instance per lifecycle connection",
                ));
            }
            let instance_id = request.instance_id.clone();
            let token = lifecycle
                .open_session(crate::wire::session(request))
                .await?;
            owners.insert(instance_id, token);
        }
    }
    Ok((Vec::new(), Vec::new()))
}

#[cfg(test)]
#[path = "../../tests/unit/endpoint/session.rs"]
mod tests;
