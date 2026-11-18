use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, FromRequest, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, routing::post};
use orbitkv_state::{InventoryScope, is_storage_namespace};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::cluster::InventoryRuntime;
use crate::cluster::router::{Cursor, RouterError};

const REQUEST_BYTES: usize = 320 * 1024;
const RESPONSE_BYTES: usize = 3 * 1024 * 1024;
const CONFIG_BYTES: usize = 32 * 1024;
const SLOTS_PER_ROUTE: usize = 8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Credential {
    namespace: String,
    token_sha256: String,
}

pub(crate) struct Credentials(HashMap<String, InventoryScope>);

impl Credentials {
    pub(crate) fn load(
        path: &Path,
        addr: SocketAddr,
        allowed: &InventoryScope,
    ) -> Result<Self, String> {
        if !addr.ip().is_loopback() {
            return Err("--router-credentials requires a loopback --http-addr".into());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|file| file.take(CONFIG_BYTES as u64 + 1).read_to_end(&mut bytes))
            .map_err(|error| format!("cannot read router credentials: {error}"))?;
        if bytes.len() > CONFIG_BYTES {
            return Err("router credentials exceed 32 KiB".into());
        }
        let entries: Vec<Credential> =
            serde_json::from_slice(&bytes).map_err(|_| "invalid router credentials JSON")?;
        if entries.is_empty() || entries.len() > 64 {
            return Err("router credentials require 1..=64 entries".into());
        }
        let mut scopes = HashMap::new();
        for entry in entries {
            if !is_storage_namespace(&entry.namespace)
                || !allowed.contains(&entry.namespace)
                || !is_lower_hex_64(entry.token_sha256.as_bytes())
            {
                return Err("router credential requires an admitted exact namespace and SHA-256 fingerprint".into());
            }
            let scope = InventoryScope::exact([entry.namespace])?;
            if scopes.insert(entry.token_sha256, scope).is_some() {
                return Err("duplicate router token fingerprint".into());
            }
        }
        Ok(Self(scopes))
    }

    fn authenticate(&self, headers: &HeaderMap) -> Option<&InventoryScope> {
        let mut values = headers.get_all(header::AUTHORIZATION).iter();
        let token = values.next()?.as_bytes().strip_prefix(b"Bearer ")?;
        if values.next().is_some() || !is_lower_hex_64(token) {
            return None;
        }
        self.0.get(&format!("{:x}", Sha256::digest(token)))
    }
}

fn is_lower_hex_64(bytes: &[u8]) -> bool {
    bytes.len() == 64
        && bytes
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

struct RouterState {
    credentials: Credentials,
    inventory: InventoryRuntime,
    lookups: Arc<Semaphore>,
    polls: Arc<Semaphore>,
}

pub(crate) fn routes(credentials: Credentials, inventory: InventoryRuntime) -> Router {
    Router::new()
        .route("/v1/router/lookup", post(lookup))
        .route("/v1/router/events", post(events))
        .layer(DefaultBodyLimit::max(REQUEST_BYTES))
        .with_state(Arc::new(RouterState {
            credentials,
            inventory,
            lookups: Arc::new(Semaphore::new(SLOTS_PER_ROUTE)),
            polls: Arc::new(Semaphore::new(SLOTS_PER_ROUTE)),
        }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LookupRequest {
    hashes: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EventsRequest {
    cursor: Cursor,
    #[serde(default)]
    wait_ms: u64,
}

type HttpError = (StatusCode, &'static str);

impl From<RouterError> for HttpError {
    fn from(error: RouterError) -> Self {
        match error {
            RouterError::Invalid(reason) => (StatusCode::BAD_REQUEST, reason),
            RouterError::Resync => (StatusCode::CONFLICT, "resync_required"),
            RouterError::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "membership_unavailable"),
            RouterError::Inventory => (StatusCode::SERVICE_UNAVAILABLE, "inventory_unavailable"),
        }
    }
}

impl RouterState {
    async fn read<T: DeserializeOwned>(
        &self,
        request: Request,
        slots: &Arc<Semaphore>,
    ) -> Result<(&InventoryScope, T, OwnedSemaphorePermit), HttpError> {
        let scope = self
            .credentials
            .authenticate(request.headers())
            .ok_or((StatusCode::UNAUTHORIZED, "unauthorized"))?;
        let slot = slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "busy"))?;
        let Json(query) = tokio::time::timeout(
            Duration::from_secs(1),
            Json::<T>::from_request(request, &()),
        )
        .await
        .map_err(|_| (StatusCode::REQUEST_TIMEOUT, "request_body_timeout"))?
        .map_err(|rejection| (rejection.status(), "invalid_request"))?;
        Ok((scope, query, slot))
    }
}

async fn lookup(
    State(state): State<Arc<RouterState>>,
    request: Request,
) -> Result<Response, HttpError> {
    let (scope, query, slot) = state.read::<LookupRequest>(request, &state.lookups).await?;
    let result = state.inventory.router_lookup(scope, query.hashes)?;
    reply(&result, slot)
}

async fn events(
    State(state): State<Arc<RouterState>>,
    request: Request,
) -> Result<Response, HttpError> {
    let (scope, query, slot) = state.read::<EventsRequest>(request, &state.polls).await?;
    let result = state
        .inventory
        .router_events(scope, query.cursor, query.wait_ms)
        .await?;
    reply(&result, slot)
}

// Bytes keeps the slot until the transport drops its final reference, even for a slow reader.
struct ReplyBody {
    bytes: Vec<u8>,
    _slot: OwnedSemaphorePermit,
}

impl AsRef<[u8]> for ReplyBody {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Write for ReplyBody {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > RESPONSE_BYTES - self.bytes.len() {
            return Err(io::Error::other("router response exceeds its byte limit"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn reply<T: Serialize>(value: &T, slot: OwnedSemaphorePermit) -> Result<Response, HttpError> {
    let mut body = ReplyBody {
        bytes: Vec::new(),
        _slot: slot,
    };
    serde_json::to_writer(&mut body, value).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "response_encoding_failed",
        )
    })?;
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        Body::from(Bytes::from_owner(body)),
    )
        .into_response())
}

#[cfg(test)]
#[path = "../../tests/unit/http_server/router.rs"]
mod tests;
