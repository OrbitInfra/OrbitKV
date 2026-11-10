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

use serde::Deserialize;
use serde_json::{Value, json};

use super::{SsdReadLease, SsdStore};
use crate::block::{RawBlock, SealedBlock, Segment, StateKey};
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
                    format: orbitkv_state::StorageFormat::Exact,
                    logical_bytes: bytes as usize,
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

async fn read(store: Arc<SsdStore>, lease: Arc<SsdReadLease>) -> Value {
    match store.read_host_batch(vec![lease]).await {
        Ok(mut blocks) => {
            let block = blocks.remove(0).1;
            let slots: Vec<_> = block
                .slots()
                .iter()
                .map(|slot| {
                    let (ptr, bytes) = slot.segment_iovecs().next().unwrap();
                    // SAFETY: the read's native receipt initialized and drained this
                    // entire range. The retained block owns its immutable backing.
                    let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), bytes) };
                    json!({"bytes": bytes, "encoded": slot.encoding.is_some()})
                })
                .collect();
            json!({"present": true, "slots": slots})
        }
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
                    store.ingest_batch(std::iter::once((&key, &block)), false);
                    store.flush().await;
                    drop(block);
                    json!({"present": store.discover(std::slice::from_ref(&key))[0].is_some()})
                }
                "read" => match pin(&store, &key) {
                    Some(lease) => read(store.clone(), lease).await,
                    None => json!({"present": false}),
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
