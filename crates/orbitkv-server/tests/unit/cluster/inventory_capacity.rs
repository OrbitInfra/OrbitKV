use super::*;
use crate::cluster::Cluster;
use orbitkv_catalog::GlobalIndex;
use orbitkv_core::ResidencyInventory;
use serde_json::json;

const SOURCES: usize = 16;
const KEYS_PER_SOURCE: usize = 2048;
const ACTIVE_KEYS: usize = 1536;
const WINDOW_SHIFT: usize = 128;
const ROUNDS: usize = 60;
const INDEX_BYTES: usize = 16 * 1024 * 1024;

struct Node {
    cluster: Cluster,
    view: Arc<MembershipView>,
    index: Arc<GlobalIndex>,
    inventory: Arc<ResidencyInventory>,
    runtime: InventoryRuntime,
    server: tokio::task::JoinHandle<()>,
    keys: Vec<StateKey>,
    sequence: Vec<u64>,
    present: Vec<bool>,
    head: u64,
    window: usize,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "requires ETCD_BIN and ORBITKV_METADATA_ARTIFACT_DIR; 60-second real-stream capacity qualification"]
async fn sustained_all_to_all_stream_churn_is_exact_bounded_and_etcd_constant() {
    let artifact = std::path::PathBuf::from(
        std::env::var_os("ORBITKV_METADATA_ARTIFACT_DIR")
            .expect("capacity evidence requires ORBITKV_METADATA_ARTIFACT_DIR"),
    );
    std::fs::create_dir_all(&artifact).unwrap();
    let etcd = Etcd::start(1).await;
    let mut nodes = Vec::new();
    for source in 0..SOURCES {
        let port = free_port();
        let membership = view(port);
        let index = Arc::new(GlobalIndex::new(
            membership.clone(),
            INDEX_BYTES,
            Arc::new(orbitkv_state::InventoryScope::AllNamespaces),
        ));
        let inventory = Arc::new(
            ResidencyInventory::with_publish_coalescing(16 * 1024 * 1024, Duration::from_millis(2))
                .unwrap(),
        );
        let cluster = Cluster::join(
            &etcd.endpoints,
            "stream-capacity",
            &format!("source-{source}"),
            120,
            membership.clone(),
            inventory.clone(),
            index.clone(),
        )
        .await
        .unwrap();
        let runtime = cluster.inventory();
        let server = serve(runtime.clone(), port).await;
        let keys = (0..KEYS_PER_SOURCE)
            .map(|key| {
                StateKey::new(
                    "capacity".into(),
                    [source.to_le_bytes(), key.to_le_bytes()].concat(),
                )
            })
            .collect();
        nodes.push(Node {
            cluster,
            view: membership,
            index,
            inventory,
            runtime,
            server,
            keys,
            sequence: vec![0; KEYS_PER_SOURCE],
            present: vec![false; KEYS_PER_SOURCE],
            head: 0,
            window: 0,
        });
    }
    wait_for(|| {
        nodes.iter().all(|node| {
            nodes
                .iter()
                .all(|peer| node.view.permits(peer.view.owner()))
        })
    })
    .await;

    for (source, node) in nodes.iter_mut().enumerate() {
        for key in 0..ACTIVE_KEYS {
            node.head += 1;
            node.sequence[key] = node.head;
            node.present[key] = true;
            let medium = medium(source, key);
            node.inventory
                .test_change(&node.keys[key], medium, Some(metadata(medium)))
                .unwrap();
        }
    }
    let fences = nodes
        .iter()
        .map(|node| node.runtime.capture_fence().unwrap())
        .collect::<Vec<_>>();
    for fence in fences.iter().skip(1) {
        nodes[0]
            .runtime
            .await_fence(
                fence,
                &all_namespaces_scope_digest(),
                Duration::from_secs(30),
            )
            .await
            .unwrap();
    }
    assert_exact(&nodes[0].index, &nodes);
    let mut client = etcd_client::Client::connect(&etcd.endpoints, None)
        .await
        .unwrap();
    let before = client.status().await.unwrap();
    let test_before = process_sample([std::process::id()]);
    let etcd_before = process_sample(etcd.pids());

    let started = std::time::Instant::now();
    let mut visibility_ms = Vec::with_capacity(ROUNDS);
    let mut mutation_ms = Vec::with_capacity(ROUNDS);
    for round in 1..=ROUNDS {
        tokio::time::sleep_until((started + Duration::from_secs(round as u64)).into()).await;
        let mutation_started = std::time::Instant::now();
        for (source, node) in nodes.iter_mut().enumerate() {
            let previous = node.window;
            for offset in 0..WINDOW_SHIFT {
                let key = (previous + offset) % KEYS_PER_SOURCE;
                node.head += 1;
                node.sequence[key] = node.head;
                node.present[key] = false;
                node.inventory
                    .test_change(&node.keys[key], medium(source, key), None)
                    .unwrap();
            }
            for offset in 0..WINDOW_SHIFT {
                let key = (previous + ACTIVE_KEYS + offset) % KEYS_PER_SOURCE;
                node.head += 1;
                node.sequence[key] = node.head;
                node.present[key] = true;
                let medium = medium(source, key);
                node.inventory
                    .test_change(&node.keys[key], medium, Some(metadata(medium)))
                    .unwrap();
            }
            node.window = (previous + WINDOW_SHIFT) % KEYS_PER_SOURCE;
        }
        mutation_ms.push(mutation_started.elapsed().as_secs_f64() * 1000.0);
        let fences = nodes
            .iter()
            .map(|node| node.runtime.capture_fence().unwrap())
            .collect::<Vec<_>>();
        let visibility_started = std::time::Instant::now();
        for fence in fences.iter().skip(1) {
            nodes[0]
                .runtime
                .await_fence(
                    fence,
                    &all_namespaces_scope_digest(),
                    Duration::from_secs(30),
                )
                .await
                .unwrap();
        }
        visibility_ms.push(visibility_started.elapsed().as_secs_f64() * 1000.0);
        assert_exact(&nodes[0].index, &nodes);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let after = client.status().await.unwrap();
    let test_after = process_sample([std::process::id()]);
    let etcd_after = process_sample(etcd.pids());
    let changes = SOURCES * ROUNDS * WINDOW_SHIFT * 2;
    let status = nodes[0].index.status();
    let stream_status = nodes[0].runtime.status();
    let keys = client
        .get(
            "/orbitkv/v2/stream-capacity/",
            Some(
                etcd_client::GetOptions::new()
                    .with_prefix()
                    .with_keys_only(),
            ),
        )
        .await
        .unwrap();
    assert!(keys.kvs().iter().all(|kv| {
        let key = kv.key_str().unwrap();
        !key.contains("/blocks/") && !key.contains("/publishers/")
    }));
    assert!(status.accounted_bytes <= INDEX_BYTES);
    assert!(stream_status.outbound_queue_bytes_peak <= (32 * 1024 * 1024) as u64);
    assert!(stream_status.receiver_sessions_peak <= 128);
    assert!(stream_status.source_sessions_peak <= 128);
    let result = json!({
        "accepted": percentile(&visibility_ms, 0.99) <= 50.0,
        "sources": SOURCES,
        "active_records_per_source": ACTIVE_KEYS,
        "changes": changes,
        "changes_per_second": changes as f64 / elapsed,
        "elapsed_seconds": elapsed,
        "mutation_ms": summary(&mutation_ms),
        "visibility_ms": summary(&visibility_ms),
        "index": status,
        "stream": stream_status,
        "etcd_db_size_before": before.db_size(),
        "etcd_db_size_after": after.db_size(),
        "etcd_db_growth_bytes": after.db_size() - before.db_size(),
        "test_process_before": test_before,
        "test_process_after": test_after,
        "etcd_process_before": etcd_before,
        "etcd_process_after": etcd_after,
        "etcd_keys": keys.kvs().len(),
        "thresholds": {
            "visibility_p99_ms": 50,
            "index_bytes": INDEX_BYTES,
            "stream_sessions": 128,
            "aggregate_queue_bytes": 32 * 1024 * 1024,
        },
    });
    std::fs::write(
        artifact.join("stream-capacity.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert!(result["accepted"].as_bool().unwrap(), "{result}");
    for node in nodes {
        node.server.abort();
        node.cluster.shutdown().await;
    }
}

fn medium(source: usize, key: usize) -> ReplicaMedium {
    if (source + key).is_multiple_of(2) {
        ReplicaMedium::Dram
    } else {
        ReplicaMedium::Ssd
    }
}

fn assert_exact(index: &GlobalIndex, nodes: &[Node]) {
    for (source, node) in nodes.iter().enumerate().skip(1) {
        for chunk in node.keys.chunks(128) {
            let rows = index.lookup(chunk);
            for row in &rows {
                let key = node.keys.iter().position(|key| key == &row.key).unwrap();
                if node.present[key] {
                    assert_eq!(row.replicas.len(), 1);
                    assert_eq!(row.replicas[0].owner, *node.view.owner());
                    assert_eq!(row.replicas[0].sequence, node.sequence[key]);
                    assert_eq!(row.replicas[0].metadata.medium, medium(source, key));
                } else {
                    assert!(row.replicas.is_empty());
                }
            }
        }
    }
}

fn summary(values: &[f64]) -> serde_json::Value {
    json!({
        "samples": values.len(),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values.iter().copied().fold(0.0, f64::max),
    })
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * percentile).round() as usize]
}

fn process_sample(pids: impl IntoIterator<Item = u32>) -> serde_json::Value {
    let mut cpu_ticks = 0u64;
    let mut rss_kib = 0u64;
    let mut hwm_kib = 0u64;
    let mut processes = 0u64;
    for pid in pids {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, stat)) = stat.rsplit_once(") ") else {
            continue;
        };
        let columns = stat.split_whitespace().collect::<Vec<_>>();
        cpu_ticks += columns[11].parse::<u64>().unwrap() + columns[12].parse::<u64>().unwrap();
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                rss_kib += value
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
            }
            if let Some(value) = line.strip_prefix("VmHWM:") {
                hwm_kib += value
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
            }
        }
        processes += 1;
    }
    json!({"cpu_ticks": cpu_ticks, "rss_kib": rss_kib, "hwm_kib": hwm_kib, "processes": processes})
}
