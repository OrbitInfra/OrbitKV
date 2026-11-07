use super::*;

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cluster::Cluster;
use orbitkv_catalog::GlobalIndex;
use orbitkv_core::ResidencyInventory;
use serde_json::json;

fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    u64::try_from(now.tv_sec).unwrap() * 1_000_000_000 + u64::try_from(now.tv_nsec).unwrap()
}

fn write_json(path: &Path, value: &serde_json::Value) {
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    std::fs::rename(temporary, path).unwrap();
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name} for metadata pressure source"))
}

fn number(name: &str) -> usize {
    required(name)
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
}

async fn wait_for_file(path: &Path, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "external metadata-only pressure source for S2.10 isolation qualification"]
async fn external_metadata_only_pressure_source() {
    let endpoints = required("ORBITKV_PRESSURE_ETCD_ENDPOINTS")
        .split(',')
        .map(str::to_string)
        .collect::<Vec<_>>();
    let cluster_name = required("ORBITKV_PRESSURE_CLUSTER");
    let node_id = required("ORBITKV_PRESSURE_NODE");
    let namespace = required("ORBITKV_PRESSURE_NAMESPACE");
    let artifact = PathBuf::from(required("ORBITKV_PRESSURE_ARTIFACT_DIR"));
    let go = PathBuf::from(required("ORBITKV_PRESSURE_GO_FILE"));
    let stop = PathBuf::from(required("ORBITKV_PRESSURE_STOP_FILE"));
    let mode = required("ORBITKV_PRESSURE_MODE");
    assert!(matches!(mode.as_str(), "quiet" | "pressure"));
    let stall_round = std::env::var("ORBITKV_PRESSURE_STALL_ROUND")
        .ok()
        .map(|value| value.parse::<usize>().unwrap());
    let stall_ms = std::env::var("ORBITKV_PRESSURE_STALL_MS")
        .ok()
        .map(|value| value.parse::<u64>().unwrap())
        .unwrap_or(0);
    assert_eq!(stall_round.is_some(), stall_ms > 0);
    let rounds = number("ORBITKV_PRESSURE_ROUNDS");
    let cadence_ms = number("ORBITKV_PRESSURE_CADENCE_MS");
    assert!(stall_round.is_none_or(|round| round < rounds));
    let keys = number("ORBITKV_PRESSURE_KEYS");
    let active = number("ORBITKV_PRESSURE_ACTIVE_KEYS");
    let shift = number("ORBITKV_PRESSURE_WINDOW_SHIFT");
    assert!(rounds > 0 && cadence_ms > 0 && shift > 0);
    assert!(active > shift && keys >= active + shift);
    std::fs::create_dir_all(&artifact).unwrap();

    let port = free_port();
    let membership = view(port);
    let index = Arc::new(GlobalIndex::new(
        membership.clone(),
        64 * 1024 * 1024,
        Arc::new(InventoryScope::exact([namespace.clone()]).unwrap()),
    ));
    let inventory = Arc::new(
        ResidencyInventory::with_publish_coalescing(16 * 1024 * 1024, Duration::from_millis(2))
            .unwrap(),
    );
    let cluster = Cluster::join(
        &endpoints,
        &cluster_name,
        &node_id,
        120,
        membership.clone(),
        inventory.clone(),
        index.clone(),
    )
    .await
    .unwrap();
    let runtime = cluster.inventory();
    let server = serve(runtime.clone(), port).await;
    let keys = (0..keys)
        .map(|key| StateKey::new(namespace.clone(), key.to_le_bytes().to_vec()))
        .collect::<Vec<_>>();
    for key in keys.iter().take(active) {
        inventory
            .test_change(
                key,
                ReplicaMedium::Dram,
                Some(metadata(ReplicaMedium::Dram)),
            )
            .unwrap();
    }
    let initial_sequence = inventory.sequence();
    let ready = json!({
        "measurement_contract": "s2.10-performance-v2",
        "mode": mode,
        "node": node_id,
        "owner": membership.owner(),
        "scope_digest": hex(&index.scope().digest()),
        "initial_sequence": initial_sequence,
        "inventory": inventory.status(),
        "stream": runtime.status(),
    });
    write_json(&artifact.join("pressure-source-ready.json"), &ready);
    wait_for_file(&go, Duration::from_secs(120)).await;

    let origin_lower = monotonic_ns();
    let started = std::time::Instant::now();
    let origin_upper = monotonic_ns();
    write_json(
        &artifact.join("pressure-started.json"),
        &json!({
            "started_mono_ns_lower": origin_lower,
        "started_mono_ns_upper": origin_upper,
        "started_unix_ns": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        }),
    );
    let mut samples = std::fs::File::create(artifact.join("pressure-samples.jsonl")).unwrap();
    let mut window = 0;
    let minimum_gap = Duration::from_micros(cadence_ms as u64 * 500);
    let mut not_before = started;
    for round in 0..rounds {
        let scheduled = started + Duration::from_millis((round * cadence_ms) as u64);
        tokio::time::sleep_until(scheduled.max(not_before).into()).await;
        if stall_round == Some(round) {
            tokio::time::sleep(Duration::from_millis(stall_ms)).await;
        }
        assert!(
            !stop.exists(),
            "pressure source stopped before completing its schedule"
        );
        let scheduled_seconds = (round * cadence_ms) as f64 / 1000.0;
        let actual_seconds = started.elapsed().as_secs_f64();
        let mut first_publication_mono_ns = 0;
        let sequence_before = inventory.sequence();
        if mode == "pressure" {
            for offset in 0..shift {
                inventory
                    .test_change(
                        &keys[(window + offset) % keys.len()],
                        ReplicaMedium::Dram,
                        None,
                    )
                    .unwrap();
                if offset == 0 {
                    first_publication_mono_ns = inventory.status().last_change_mono_ns;
                }
            }
            for offset in 0..shift {
                inventory
                    .test_change(
                        &keys[(window + active + offset) % keys.len()],
                        ReplicaMedium::Dram,
                        Some(metadata(ReplicaMedium::Dram)),
                    )
                    .unwrap();
            }
            window = (window + shift) % keys.len();
        }
        writeln!(
            samples,
            "{}",
            json!({
                "round": round, "scheduled_seconds": scheduled_seconds,
                "injected_stall_ms": if stall_round == Some(round) { stall_ms } else { 0 },
                "actual_seconds": actual_seconds, "inventory": inventory.status(),
                "sequence_before": sequence_before,
                "scheduled_mono_ns_lower": origin_lower + (round * cadence_ms) as u64 * 1_000_000,
                "scheduled_mono_ns_upper": origin_upper + (round * cadence_ms) as u64 * 1_000_000,
                "first_publication_mono_ns": first_publication_mono_ns,
                "last_publication_mono_ns": if mode == "pressure" { inventory.status().last_change_mono_ns } else { 0 },
            })
        )
        .unwrap();
        not_before = std::time::Instant::now() + minimum_gap;
    }
    tokio::time::sleep_until(
        (started + Duration::from_millis((rounds * cadence_ms) as u64)).into(),
    )
    .await;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let mut final_records = Vec::new();
    let mut after = None;
    loop {
        let page = inventory
            .scoped_page(after.as_ref(), &index.scope())
            .unwrap();
        final_records.extend(page.records.iter().map(|record| {
            json!({
                "key": { "namespace": record.key.namespace, "hash": record.key.hash },
                "sequence": record.sequence,
                "present": record.present,
                "metadata": record.metadata,
            })
        }));
        if page.complete {
            break;
        }
        after = page.next;
    }
    let result = json!({
        "measurement_contract": "s2.10-performance-v2",
        "mode": mode,
        "rounds": rounds,
        "minimum_gap_micros": minimum_gap.as_micros(),
        "stall_round": stall_round,
        "stall_ms": stall_ms,
        "cadence_ms": cadence_ms,
        "elapsed_seconds": elapsed_seconds,
        "final_sequence": inventory.sequence(),
        "final_records": final_records,
        "owner": membership.owner(),
        "changes": inventory.sequence() - initial_sequence,
        "inventory": inventory.status(),
        "stream": runtime.status(),
        "index": index.status(),
    });
    write_json(&artifact.join("pressure-source-result.json"), &result);
    wait_for_file(&stop, Duration::from_secs(120)).await;
    server.abort();
    cluster.shutdown().await;
}
