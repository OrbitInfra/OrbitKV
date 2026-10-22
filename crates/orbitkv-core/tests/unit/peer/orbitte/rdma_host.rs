use std::env;
use std::io::Write;
use std::thread;

use super::*;

const GUARD: usize = 64;
const TIMEOUT: Duration = Duration::from_secs(15);

fn pattern(seed: u64, index: usize) -> u8 {
    let value = (index as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(seed)
        .rotate_left((index % 63) as u32);
    (value ^ (value >> 17) ^ (value >> 41)) as u8
}

fn regions(payload: usize) -> Vec<(usize, usize, usize)> {
    let quarter = payload / 4;
    (0..3)
        .map(|index| {
            (
                GUARD + index * quarter + 5,
                GUARD + index * quarter + 11,
                quarter - 23,
            )
        })
        .collect()
}

fn finish(transport: OrbitTeTransport, memory: RegisteredMemory) {
    transport.engine.close_requester(TIMEOUT).unwrap();
    transport.engine.close_source(TIMEOUT).unwrap();
    transport.engine.unregister_memory(&memory).unwrap();
    drop(memory);
    let stats = transport.engine.stats().unwrap();
    let rdma = stats.rdma.unwrap();
    assert_eq!(
        (
            stats.retained_regions,
            stats.retained_bytes,
            stats.active_source_operations,
            stats.inflight_batches,
            stats.inflight_bytes
        ),
        (0, 0, 0, 0, 0)
    );
    assert_eq!(
        (
            rdma.active_endpoints,
            rdma.staging_bytes,
            rdma.retained_endpoints,
            rdma.retained_bytes
        ),
        (0, 0, 0, 0)
    );
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    assert!(maps.contains("libibverbs"));
    assert!(!maps.contains("libtent") && !maps.contains("libmooncake"));
    assert!(
        !maps.contains("libcudart") && !maps.contains("libcuda.so"),
        "host-only test must not load a CUDA runtime or driver"
    );
    if let Some(path) = env::var_os("ORBITKV_ORBITTE_MAPS") {
        std::fs::write(path, maps).unwrap();
    }
    println!(
        "ORBITKV_RDMA_HOST_STATS {}",
        serde_json::json!({
            "completed_reads": rdma.completed_reads, "completed_writes": rdma.completed_writes,
            "active_endpoints": rdma.active_endpoints, "staging_bytes": rdma.staging_bytes,
            "retained_endpoints": rdma.retained_endpoints, "retained_bytes": rdma.retained_bytes,
            "registered_regions": stats.retained_regions, "active_source_operations": stats.active_source_operations,
            "scope": "host-only registered read execution; not pinned allocator, Manager, GPU or physical-fault qualification"
        })
    );
    drop(transport);
    assert_eq!(*REGISTERED_POOLS.lock().unwrap(), (0, 0));
}

#[test]
#[ignore = "two-host host-RDMA component gate; use OrbitTE tests/hardware/run.py with runner=orbitkv-peer-host; never initializes CUDA"]
fn registered_host_rdma_peer() {
    let role = env::var("ORBITKV_ORBITTE_ROLE").unwrap();
    let payload: usize = env::var("ORBITKV_ORBITTE_PAYLOAD")
        .unwrap()
        .parse()
        .unwrap();
    let seed: u64 = env::var("ORBITKV_ORBITTE_SEED").unwrap().parse().unwrap();
    assert!((128..=8 << 20).contains(&payload));
    let device = env::var("ORBITKV_ORBITTE_RDMA_DEVICE").unwrap();
    let port = env::var("ORBITKV_ORBITTE_RDMA_PORT")
        .unwrap()
        .parse()
        .unwrap();
    let gid_index = env::var("ORBITKV_ORBITTE_RDMA_GID_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let transport = OrbitTeTransport::new(
        (32 << 20) as u64,
        &env::var("ORBITKV_ORBITTE_BIND").unwrap(),
        Transport::Rdma,
        Some(crate::OrbitTeRdmaConfig {
            port,
            gid_index,
            ..crate::OrbitTeRdmaConfig::new(device)
        }),
    )
    .unwrap();
    let mut source_bytes = vec![0xa5; payload + 2 * GUARD];
    for (index, byte) in source_bytes[GUARD..GUARD + payload].iter_mut().enumerate() {
        *byte = pattern(seed, index);
    }
    if role == "source" {
        let memory = transport
            .engine
            .register_memory(
                source_bytes.clone().into_boxed_slice(),
                Permission::RemoteRead,
            )
            .unwrap();
        println!(
            "ORBITTE_READY {} {} {} {}",
            transport.endpoint,
            transport.epoch,
            memory.base_address(),
            memory.len()
        );
        std::io::stdout().flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(45);
        loop {
            let messages = transport.engine.receive_notifications().unwrap();
            if !messages.is_empty() {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].name, "host_read_done");
                assert_eq!(messages[0].message, seed.to_string());
                break;
            }
            assert!(Instant::now() < deadline, "requester completion deadline");
            thread::sleep(Duration::from_millis(2));
        }
        let mut observed = vec![0; memory.len()];
        memory.read(0, &mut observed).unwrap();
        assert_eq!(observed, source_bytes, "READ changed source or guards");
        finish(transport, memory);
        return;
    }
    assert_eq!(role, "requester");
    let remote: Vec<String> =
        serde_json::from_str(&env::var("ORBITKV_ORBITTE_REMOTE").unwrap()).unwrap();
    assert_eq!(remote.len(), 4);
    let endpoint = &remote[0];
    let epoch = &remote[1];
    let address: u64 = remote[2].parse().unwrap();
    assert_eq!(remote[3].parse::<usize>().unwrap(), source_bytes.len());
    let memory = transport
        .engine
        .register_memory(
            vec![0xcc; source_bytes.len()].into_boxed_slice(),
            Permission::LocalOnly,
        )
        .unwrap();
    let entries = regions(payload);
    let requests: Vec<_> = entries
        .iter()
        .map(|(local, remote, length)| TransferRequest {
            transport: Transport::Rdma,
            local_address: memory.base_address() + *local as u64,
            remote_address: address + *remote as u64,
            length: *length,
        })
        .collect();
    let wrong_payload = endpoint.replacen(prefix(Transport::Rdma), prefix(Transport::Tcp), 1);
    assert!(
        matches!(transport.read_registered(&wrong_payload, epoch, &requests, Instant::now() + TIMEOUT), Err(PeerTransferError::Failed(reason)) if reason.contains("protocol mismatch"))
    );
    let mut old_epoch = epoch.clone();
    old_epoch.replace_range(..1, if &epoch[..1] == "0" { "1" } else { "0" });
    assert!(
        matches!(transport.read_registered(endpoint, &old_epoch, &requests, Instant::now() + TIMEOUT), Err(PeerTransferError::Failed(reason)) if reason.contains("epoch"))
    );
    assert_eq!(
        transport
            .engine
            .stats()
            .unwrap()
            .rdma
            .unwrap()
            .completed_reads,
        0
    );
    for group in [&requests[..1], &requests[1..]] {
        let expected: usize = group.iter().map(|request| request.length).sum();
        assert_eq!(
            transport
                .read_registered(endpoint, epoch, group, Instant::now() + TIMEOUT)
                .unwrap(),
            expected
        );
    }
    let mut expected = vec![0xcc; memory.len()];
    for (local, remote, length) in entries {
        expected[local..local + length].copy_from_slice(&source_bytes[remote..remote + length]);
    }
    let mut observed = vec![0; memory.len()];
    memory.read(0, &mut observed).unwrap();
    assert_eq!(observed, expected, "full payload/guards/unrequested bytes");
    assert_eq!(
        transport
            .engine
            .stats()
            .unwrap()
            .rdma
            .unwrap()
            .completed_reads,
        requests.len() as u64
    );
    let peer = transport
        .engine
        .open_segment(
            parse_endpoint(Transport::Rdma, endpoint).unwrap(),
            epoch.parse().unwrap(),
            TIMEOUT,
        )
        .unwrap();
    transport
        .engine
        .send_notification(
            &peer,
            &orbit_te::Notification {
                name: "host_read_done".into(),
                message: seed.to_string(),
            },
            TIMEOUT,
        )
        .unwrap();
    finish(transport, memory);
}
