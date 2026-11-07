use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing::get, routing::post};
use log::{info, warn};
use orbitkv_core::OrbitKVEngine;
use prometheus::{Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::registry::RegistryHandle;

#[derive(Clone)]
struct AppState {
    engine: Arc<OrbitKVEngine>,
    lifecycle: crate::cache::lifecycle::LifecycleService,
    prometheus_registry: Option<Registry>,
    inventory: Option<crate::cluster::InventoryRuntime>,
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let Some(ref registry) = state.prometheus_registry else {
        return (StatusCode::NOT_FOUND, "metrics not enabled".to_string());
    };
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    (
        StatusCode::OK,
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("# Error encoding metrics: {e}")),
    )
}

#[derive(Serialize)]
struct InstancesResponse {
    instances: Vec<String>,
}

async fn list_instances_handler(
    State(state): State<AppState>,
) -> Result<Json<InstancesResponse>, (StatusCode, String)> {
    let instances = state
        .engine
        .list_instance_ids()
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(Json(InstancesResponse { instances }))
}

#[derive(Deserialize)]
struct CleanupQuery {
    id: Option<String>,
}

#[derive(Serialize)]
struct CleanupResponse {
    removed_instances: Vec<String>,
    removed_tensors: usize,
}

#[derive(Serialize)]
struct MemoryCacheCleanupResponse {
    evicted_blocks: usize,
    evicted_bytes: u64,
    reclaimed_bytes: u64,
    still_referenced_blocks: u64,
}

/// POST /instances/cleanup[?id=<instance_id>]
///
/// Without `id`: remove all instances and release all CUDA IPC tensors.
/// With `id`:    remove only the specified instance.
///
/// Releasing CUDA IPC tensors takes the GIL and runs a blocking
/// `torch.cuda.empty_cache()`. That work runs on the dedicated registry thread
/// behind [`RegistryHandle`]; the handler only `.await`s the reply, so a
/// slow/wedged cleanup never occupies an async worker (the outage where a few
/// `cleanup` calls hung every endpoint, `/health` and `/metrics` included).
async fn cleanup_handler(
    State(state): State<AppState>,
    Query(query): Query<CleanupQuery>,
) -> impl IntoResponse {
    let ids = match query
        .id
        .map_or_else(|| state.engine.list_instance_ids(), |id| Ok(vec![id]))
    {
        Ok(ids) => ids,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                error.to_string().into_response(),
            );
        }
    };
    let mut removed_tensors = 0;
    let mut removed_instances = Vec::new();
    for id in ids {
        match state.lifecycle.cleanup(&id).await {
            Ok(removed) => {
                removed_tensors += removed;
                removed_instances.push(id);
            }
            Err(error) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    error.to_string().into_response(),
                );
            }
        }
    }
    (
        StatusCode::OK,
        Json(CleanupResponse {
            removed_instances,
            removed_tensors,
        })
        .into_response(),
    )
}

/// POST /cache/memory/cleanup
///
/// Drops resident in-memory cache blocks while preserving backing-store data.
async fn cleanup_memory_cache_handler(
    State(state): State<AppState>,
) -> Result<Json<MemoryCacheCleanupResponse>, (StatusCode, String)> {
    let stats = state
        .engine
        .cleanup_memory_cache()
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(Json(MemoryCacheCleanupResponse {
        evicted_blocks: stats.evicted_blocks,
        evicted_bytes: stats.evicted_bytes,
        reclaimed_bytes: stats.reclaimed_bytes,
        still_referenced_blocks: stats.still_referenced_blocks,
    }))
}

#[derive(Serialize)]
struct MetadataResponse {
    #[serde(flatten)]
    engine: orbitkv_core::MetadataStatus,
    stream: crate::cluster::InventoryRuntimeStatus,
}

async fn metadata_handler(State(state): State<AppState>) -> Json<Option<MetadataResponse>> {
    Json(
        match (state.engine.metadata_status(), state.inventory.as_ref()) {
            (Some(engine), Some(inventory)) => Some(MetadataResponse {
                engine,
                stream: inventory.status(),
            }),
            _ => None,
        },
    )
}

#[derive(Deserialize)]
struct OwnerMetadataQuery {
    after: Option<uuid::Uuid>,
    #[serde(default = "default_owner_status_limit")]
    limit: usize,
}

fn default_owner_status_limit() -> usize {
    64
}

async fn owner_metadata_handler(
    State(state): State<AppState>,
    Query(query): Query<OwnerMetadataQuery>,
) -> impl IntoResponse {
    if query.limit == 0 || query.limit > 128 {
        return (
            StatusCode::BAD_REQUEST,
            "owner metadata limit must be in 1..=128".to_string(),
        )
            .into_response();
    }
    match state
        .engine
        .metadata_owner_statuses(query.after, query.limit)
    {
        Some(owners) => Json(owners).into_response(),
        None => (
            StatusCode::CONFLICT,
            "distributed inventory is not configured",
        )
            .into_response(),
    }
}

async fn sync_cache_handler(State(state): State<AppState>) -> impl IntoResponse {
    match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        state.engine.flush_saves(),
    )
    .await
    {
        Ok(()) => match state.inventory.as_ref() {
            Some(inventory) => match inventory.capture_fence() {
                Ok(fence) => (
                    StatusCode::OK,
                    Json(serde_json::json!({"inventory_fence": fence})).into_response(),
                ),
                Err(error) => (StatusCode::SERVICE_UNAVAILABLE, error.into_response()),
            },
            None => (
                StatusCode::OK,
                Json(serde_json::json!({"local_flush_complete": true})).into_response(),
            ),
        },
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "cache synchronization timed out".into_response(),
        ),
    }
}

#[derive(Deserialize)]
struct AwaitInventoryRequest {
    inventory_fence: orbitkv_state::InventoryFence,
    scope_digest: String,
    timeout_ms: u64,
}

async fn await_inventory_handler(
    State(state): State<AppState>,
    Json(request): Json<AwaitInventoryRequest>,
) -> impl IntoResponse {
    let Some(inventory) = state.inventory.as_ref() else {
        return (
            StatusCode::CONFLICT,
            "distributed inventory is not configured".to_string(),
        );
    };
    let scope = match decode_hex(&request.scope_digest) {
        Ok(scope) => scope,
        Err(error) => return (StatusCode::BAD_REQUEST, error),
    };
    match inventory
        .await_fence(
            &request.inventory_fence,
            &scope,
            std::time::Duration::from_millis(request.timeout_ms),
        )
        .await
    {
        Ok(()) => (StatusCode::OK, "inventory fence installed".to_string()),
        Err(error) if error.contains("timed out") => (StatusCode::GATEWAY_TIMEOUT, error),
        Err(error) if error.contains("limit reached") => (StatusCode::TOO_MANY_REQUESTS, error),
        Err(error) => (StatusCode::PRECONDITION_FAILED, error),
    }
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() != 64 || !value.is_ascii() {
        return Err("scope_digest must contain 64 hexadecimal characters".into());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).map_err(|_| "invalid scope_digest")?;
            u8::from_str_radix(text, 16).map_err(|_| "invalid scope_digest".to_string())
        })
        .collect()
}

/// Start HTTP server for health check, optional Prometheus metrics, and instance management.
pub async fn start_http_server(
    addr: std::net::SocketAddr,
    engine: Arc<OrbitKVEngine>,
    registry: RegistryHandle,
    enable_prometheus: bool,
    prometheus_registry: Option<Registry>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let lifecycle = crate::cache::lifecycle::LifecycleService::new(Arc::clone(&engine), registry);
    start_http_server_with_lifecycle(
        addr,
        engine,
        lifecycle,
        enable_prometheus,
        prometheus_registry,
        shutdown,
        None,
    )
    .await
}

pub(crate) async fn start_http_server_with_lifecycle(
    addr: std::net::SocketAddr,
    engine: Arc<OrbitKVEngine>,
    lifecycle: crate::cache::lifecycle::LifecycleService,
    enable_prometheus: bool,
    prometheus_registry: Option<Registry>,
    shutdown: Arc<Notify>,
    inventory: Option<crate::cluster::InventoryRuntime>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        engine,
        lifecycle,
        prometheus_registry: if enable_prometheus {
            prometheus_registry
        } else {
            None
        },
        inventory,
    };

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .route("/instances", get(list_instances_handler))
        .route("/instances/cleanup", post(cleanup_handler))
        .route("/cache/sync", post(sync_cache_handler))
        .route("/cache/metadata/await", post(await_inventory_handler))
        .route("/cache/metadata/owners", get(owner_metadata_handler))
        .route("/cache/metadata", get(metadata_handler))
        .route("/cache/memory/cleanup", post(cleanup_memory_cache_handler));

    if enable_prometheus {
        app = app.route("/metrics", get(metrics_handler));
        info!(
            "Starting HTTP server on {} (/health, /metrics, /instances, /instances/cleanup, /cache/sync, /cache/metadata, /cache/memory/cleanup)",
            addr
        );
    } else {
        info!(
            "Starting HTTP server on {} (/health, /instances, /instances/cleanup, /cache/sync, /cache/metadata, /cache/memory/cleanup)",
            addr
        );
    }

    let app = app.with_state(state);

    let handle = tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await
        {
            warn!("HTTP server stopped with error: {err}");
        }
    });

    Ok(handle)
}
