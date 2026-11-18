//! CPU process-crash control, compiled only with test-hooks. It drives the real
//! Manager's storage/OrbiTE path; CUDA allocation and model serving are excluded.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "explicit test-only process controls assert their prerequisites and fixture contracts"
)]

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use orbitkv_proto::proto::engine::{
    OpenTransferWindowRequest, QueryBlocksForTransferRequest, TransferTicket,
    engine_client::EngineClient,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{SsdReadLease, SsdStore};
use crate::block::{RawBlock, RestoreSource, SealedBlock, Segment, StateKey};
use crate::memory::numa::NumaNode;
use crate::transfer_engine::TransferOptions;

pub fn enabled() -> bool {
    if std::env::var_os("ORBITKV_TEST_SSD_PROCESS").is_none() {
        return false;
    }
    for required in ["ORBITKV_REQUIRE_ENGINE_SSD", "ORBITE_REQUIRE_IO_URING"] {
        assert_eq!(
            std::env::var(required).as_deref(),
            Ok("1"),
            "{required} is required"
        );
    }
    true
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Command {
    sequence: u64,
    operation: String,
    #[serde(default)]
    id: u8,
    #[serde(default)]
    namespace: u8,
    #[serde(default)]
    encoded: bool,
    #[serde(default)]
    wide: bool,
    #[serde(default)]
    ttl_millis: Option<u64>,
    grant: Option<GrantProbe>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GrantProbe {
    source: orbitkv_state::CacheOwner,
    requester: uuid::Uuid,
    residency: u64,
    window: Option<String>,
}

impl GrantProbe {
    async fn request(&self, key: &StateKey) -> Result<Value, tonic::Status> {
        let mut client = EngineClient::connect(format!("http://{}", self.source.endpoint))
            .await
            .unwrap();
        let owner = self.source.incarnation.to_string();
        let window = match &self.window {
            Some(window) => window.clone(),
            None => {
                client
                    .open_transfer_window(OpenTransferWindowRequest {
                        owner_incarnation: owner.clone(),
                        requester_incarnation: self.requester.to_string(),
                    })
                    .await?
                    .into_inner()
                    .window_id
            }
        };
        let reply = client
            .query_blocks_for_transfer(QueryBlocksForTransferRequest {
                namespace: key.namespace.clone(),
                block_hashes: vec![key.hash.clone()],
                owner_incarnation: owner,
                residency_sequences: vec![self.residency],
                ticket: Some(TransferTicket {
                    window_id: window.clone(),
                    slot: 0,
                    generation: 1,
                }),
                background_prefetch: false,
                transfer_timeout_millis: 10000,
                cache_protocol: orbitkv_state::INVENTORY_STREAM_PROTOCOL.into(),
            })
            .await?
            .into_inner();
        assert_eq!(reply.blocks.len(), 1);
        assert_eq!(reply.blocks[0].block_hash, key.hash);
        Ok(json!({"window": window, "epoch": reply.native_source_epoch}))
    }
}

fn key(command: &Command) -> StateKey {
    StateKey::new(
        format!("orbitkv:v2:{:064x}", command.namespace),
        orbitkv_state::group_hash(&[command.id], 0),
    )
}

fn fixture(
    storage: &crate::storage::Storage,
    id: u8,
    encoded: bool,
    wide: bool,
) -> Arc<SealedBlock> {
    let sizes = if wide { vec![8; 64] } else { vec![137, 631] };
    let slots = sizes
        .into_iter()
        .enumerate()
        .map(|(index, bytes)| {
            let allocation = storage
                .allocate(NonZeroU64::new(bytes).unwrap(), None)
                .unwrap();
            // SAFETY: the fresh allocation has no other observer; the complete
            // fixture range is initialized before handing it to the native writer.
            unsafe {
                std::ptr::write_bytes(
                    allocation.as_non_null().as_ptr(),
                    id.wrapping_add(index as u8),
                    bytes as usize,
                );
            };
            let mut raw = RawBlock::single_segment(
                Segment::new(allocation.as_non_null(), bytes as usize, allocation).unwrap(),
            );
            if encoded {
                raw.encoding = Some(vec![crate::EncodedSegment {
                    version: 1,
                    format: orbitkv_state::StorageFormat::Fp8FromBf16,
                    logical_bytes: bytes as usize * 2,
                    stored_bytes: bytes as usize,
                    checksum: crc32fast::hash(&vec![id.wrapping_add(index as u8); bytes as usize]),
                }]);
            }
            (raw, NumaNode::UNKNOWN)
        })
        .collect();
    Arc::new(SealedBlock::from_slots(slots))
}

fn pin(store: &Arc<SsdStore>, key: &StateKey) -> Option<Arc<SsdReadLease>> {
    store
        .discover(std::slice::from_ref(key))
        .remove(0)?
        .pin(TransferOptions::read(
            crate::QueryMode::Demand,
            Instant::now() + Duration::from_secs(90),
        ))
}

fn block_contents(block: &SealedBlock) -> Value {
    let slots: Vec<_> = block
        .slots()
        .iter()
        .map(|slot| {
            let mut bytes = Vec::new();
            for (ptr, len) in slot.segment_iovecs() {
                // SAFETY: the completed host read owns the immutable backing of every segment.
                bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) });
            }
            json!({"bytes": bytes, "encoded": slot.encoding.is_some()})
        })
        .collect();
    json!({"present": true, "slots": slots})
}

async fn read(store: Arc<SsdStore>, lease: Arc<SsdReadLease>) -> Value {
    match store.read_host_batch(vec![lease]).await {
        Ok(blocks) if !blocks.is_empty() => block_contents(&blocks[0].1),
        Ok(_) => json!({"present": false}),
        Err(error) => json!({"present": false, "error": error.to_string()}),
    }
}

impl SsdStore {
    fn process_report(&self) -> Value {
        let inner = self.inner.lock();
        json!({
            "recovery": self.recovery_report(),
            "pending_writes": inner.pending_writes.len(),
            "ring": inner.ring.process_report(),
        })
    }
}

pub fn spawn(engine: Arc<crate::OrbitKVEngine>) {
    if !enabled() {
        return;
    }
    let directory = PathBuf::from(std::env::var_os("ORBITKV_TEST_SSD_PROCESS").unwrap());
    assert!(directory.is_absolute() && directory.is_dir());
    tokio::spawn(async move {
        let store = engine.storage.ssd_store.as_ref().unwrap().clone();
        let mut last = 0;
        let mut held = None;
        let mut read_task = None;
        std::fs::write(directory.join("ready"), b"CPU-only real Manager storage").unwrap();
        loop {
            let path = directory.join("command.json");
            if !path.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
                continue;
            }
            let bytes = std::fs::read(path).unwrap();
            assert!(bytes.len() <= 64 * 1024);
            let command: Command = serde_json::from_slice(&bytes).unwrap();
            if command.sequence <= last {
                tokio::time::sleep(Duration::from_millis(5)).await;
                continue;
            }
            assert!(command.sequence <= 10_000);
            last = command.sequence;
            let key = key(&command);
            let mut result = match command.operation.as_str() {
                "write" => {
                    let block = fixture(&engine.storage, command.id, command.encoded, command.wide);
                    block
                        .set_expiration(
                            crate::storage::expiry::Expiration::new(
                                command.ttl_millis.map(Duration::from_millis),
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    store.ingest_batch(std::iter::once((&key, &block)), false);
                    store.flush().await;
                    drop(block);
                    json!({"present": store.discover(std::slice::from_ref(&key))[0].is_some(), "hash": key.hash})
                }
                "read" => match pin(&store, &key) {
                    Some(lease) => read(store.clone(), lease).await,
                    None => json!({"present": false}),
                },
                "peer_read" => {
                    let result = engine
                        .storage
                        .reads
                        .read_prefix(
                            "ssd-owner-handoff",
                            &key.namespace,
                            std::slice::from_ref(&key.hash),
                            crate::QueryMode::Prepare,
                            &[NumaNode::UNKNOWN; 2],
                            Instant::now() + Duration::from_secs(20),
                        )
                        .await;
                    match result.blocks.as_slice() {
                        [RestoreSource::Memory(block)] => block_contents(block),
                        [] => json!({"present": false}),
                        _ => panic!("expected one materialized host block"),
                    }
                }
                "grant" => match command.grant.as_ref().unwrap().request(&key).await {
                    Ok(grant) => grant,
                    Err(status) => json!({"grpc_code": format!("{:?}", status.code())}),
                },
                "hold" => {
                    assert!(held.is_none());
                    held = pin(&store, &key);
                    json!({"present": held.is_some()})
                }
                "release" => {
                    held = None;
                    json!({})
                }
                "read_start" => {
                    assert!(read_task.is_none());
                    let lease = pin(&store, &key).unwrap();
                    read_task = Some(tokio::spawn(read(store.clone(), lease)));
                    json!({})
                }
                "read_finish" => read_task.take().unwrap().await.unwrap(),
                "report" | "stop" => json!({}),
                _ => panic!("unknown test command"),
            };
            result["sequence"] = json!(last);
            result["state"] = store.process_report();
            let temporary = directory.join("reply.tmp");
            std::fs::write(&temporary, serde_json::to_vec(&result).unwrap()).unwrap();
            std::fs::rename(temporary, directory.join("reply.json")).unwrap();
            if command.operation == "stop" {
                break;
            }
        }
    });
}
