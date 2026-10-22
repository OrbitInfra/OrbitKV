use super::*;

#[test]
fn cache_policy_controls_validate_protection_and_admission() {
    assert!(
        Cli::try_parse_from(["orbitkv-cache-manager", "--cache-protected-percent", "101"]).is_err()
    );
    assert!(
        Cli::try_parse_from(["orbitkv-cache-manager", "--ssd-write-policy", "unknown"]).is_err()
    );
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--cache-protected-percent",
        "80",
        "--ssd-write-policy",
        "reuse",
        "--ssd-cache-path",
        "/tmp/policy-test",
    ])
    .unwrap();
    assert_eq!(cli.cache_protected_percent, 80);
    assert_eq!(cli.ssd_write_policy, orbitkv_core::SsdWritePolicy::Reuse);
    assert_eq!(cli.ssd_backend, orbitkv_core::SsdBackend::Auto);
    assert_eq!(cli.ssd_read_path, None);
}

#[test]
fn ssd_read_route_is_independent_of_backend_initialization() {
    for (value, expected) in [
        ("uring", orbitkv_core::SsdReadPath::Uring),
        ("cufile", orbitkv_core::SsdReadPath::Cufile),
    ] {
        let cli = Cli::try_parse_from([
            "orbitkv-cache-manager",
            "--ssd-cache-path",
            "/tmp/route-test",
            "--ssd-backend",
            "cufile",
            "--ssd-read-path",
            value,
        ])
        .unwrap();
        assert_eq!(cli.ssd_backend, orbitkv_core::SsdBackend::Cufile);
        assert_eq!(cli.ssd_read_path, Some(expected));
    }
    assert!(
        Cli::try_parse_from([
            "orbitkv-cache-manager",
            "--ssd-cache-path",
            "/tmp/route-test",
            "--ssd-read-path",
            "auto",
        ])
        .is_err()
    );
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--ssd-read-path", "uring",]).is_err());
}

#[test]
fn cli_membership_requires_stable_node_identity() {
    let flags = [
        "orbitkv-cache-manager",
        "--etcd-endpoints",
        "http://127.0.0.1:2379",
    ];
    assert!(Cli::try_parse_from(flags).is_err());
    let cli = Cli::try_parse_from(flags.into_iter().chain(["--node-id", "node-a"])).unwrap();
    assert_eq!(cli.node_id.as_deref(), Some("node-a"));
    assert_eq!(cli.peer_advertise_addr, None);
    assert_eq!(cli.membership_ttl_secs, 30);
    assert_eq!(cli.inventory_stream_coalesce_ms, 0);
    assert!(cli.metadata_namespace.is_empty());
    assert!(!cli.metadata_empty_scope);
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--membership-ttl-secs", "0"]).is_err());
    assert!(
        Cli::try_parse_from([
            "orbitkv-cache-manager",
            "--inventory-stream-coalesce-ms",
            "6"
        ])
        .is_err()
    );
    let advertised = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--etcd-endpoints",
        "http://127.0.0.1:2379",
        "--node-id",
        "node-a",
        "--peer-advertise-addr",
        "10.0.0.2:50055",
    ])
    .unwrap();
    assert_eq!(
        advertised.peer_advertise_addr.unwrap().to_string(),
        "10.0.0.2:50055"
    );
}

#[test]
fn cli_metadata_scope_distinguishes_all_exact_and_empty() {
    let namespace = orbitkv_state::storage_namespace("model", false, Vec::new());
    let exact = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--etcd-endpoints",
        "http://127.0.0.1:2379",
        "--node-id",
        "node-a",
        "--metadata-namespace",
        &namespace,
        "--metadata-namespace",
        &namespace,
    ])
    .unwrap();
    assert_eq!(exact.metadata_namespace, [namespace.clone(), namespace]);
    assert!(!exact.metadata_empty_scope);
    let empty = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--etcd-endpoints",
        "http://127.0.0.1:2379",
        "--node-id",
        "node-a",
        "--metadata-empty-scope",
    ])
    .unwrap();
    assert!(empty.metadata_empty_scope);
    assert!(
        Cli::try_parse_from([
            "orbitkv-cache-manager",
            "--etcd-endpoints",
            "http://127.0.0.1:2379",
            "--node-id",
            "node-a",
            "--metadata-namespace",
            "model-display-name",
        ])
        .is_err()
    );
}

#[test]
fn parse_hll_windows_canonicalizes_labels() {
    let windows = parse_hll_windows("15m,60m,24h").unwrap();

    assert_eq!(windows, expected_hll_windows());
}

#[test]
fn cli_default_metric_hll_windows_parses_without_panic() {
    let cli = Cli::try_parse_from(["orbitkv-cache-manager"]).unwrap();

    assert_eq!(
        parse_hll_windows(&cli.metric_hll_windows).unwrap(),
        expected_hll_windows()
    );
    assert_eq!(cli.metric_hll_bucket_bits, 16);
}

#[test]
fn cli_accepts_channel_options() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--channel-service",
        "orbitkv/test/server",
        "--channel-session-epoch",
        "42",
        "--bootstrap-socket",
        "/tmp/orbitkv-test.sock",
        "--descriptor-arena-size",
        "4mb",
        "--descriptor-slot-size",
        "32kb",
    ])
    .unwrap();
    assert_eq!(cli.channel_service.as_deref(), Some("orbitkv/test/server"));
    assert_eq!(cli.channel_session_epoch, Some(42));
    assert_eq!(
        cli.bootstrap_socket.as_deref(),
        Some(std::path::Path::new("/tmp/orbitkv-test.sock"))
    );
    assert_eq!(cli.descriptor_arena_size, 4 * 1024 * 1024);
    assert_eq!(cli.descriptor_slot_size, 32 * 1024);
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--enable-grpc"]).is_err());
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--disable-channel"]).is_err());
}

#[test]
fn cli_nics_accepts_comma_separated_values() {
    let cli =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0,mlx5_1,mlx5_2"]).unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_nics_trims_comma_separated_values() {
    let cli =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0, mlx5_1, mlx5_2"]).unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_nics_rejects_empty_comma_separated_values() {
    let err =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0,,mlx5_1"]).unwrap_err();

    assert!(err.to_string().contains("empty NIC name"), "{err}");
}

#[test]
fn cli_nics_accepts_repeated_values_after_flag() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--nics",
        "mlx5_0",
        "mlx5_1",
        "mlx5_2",
    ])
    .unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_explicit_metric_hll_windows_parses_without_panic() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--metric-hll-windows",
        "15m,1h,24h",
    ])
    .unwrap();

    assert_eq!(
        parse_hll_windows(&cli.metric_hll_windows).unwrap(),
        expected_hll_windows()
    );
}

#[test]
fn cli_rejects_invalid_metric_hll_windows() {
    let err = Cli::try_parse_from(["orbitkv-cache-manager", "--metric-hll-windows", "15m,,1h"])
        .unwrap_err();

    assert!(err.to_string().contains("empty window"), "{err}");
}

fn expected_hll_windows() -> Vec<(String, Duration)> {
    vec![
        ("15m".to_string(), Duration::from_secs(15 * 60)),
        ("1h".to_string(), Duration::from_secs(60 * 60)),
        ("1d".to_string(), Duration::from_secs(24 * 60 * 60)),
    ]
}

#[test]
fn parse_hll_windows_rejects_empty_tokens() {
    let err = parse_hll_windows("15m,,1h").unwrap_err();

    assert!(err.contains("empty window"), "{err}");
}

#[test]
fn parse_hll_windows_rejects_duplicate_durations() {
    let err = parse_hll_windows("1h,60m").unwrap_err();

    assert!(err.contains("duplicate HLL window duration"), "{err}");
}

#[test]
fn query_control_requires_loopback_before_runtime_initialization() {
    for (bind, advertised, enabled, accepted) in [
        ("127.0.0.1:50055", None, true, true),
        ("[::1]:50055", None, true, true),
        ("127.0.0.2:50055", Some("127.0.0.3:50056"), true, true),
        ("[::1]:50055", Some("[::1]:50056"), true, true),
        ("0.0.0.0:50055", Some("127.0.0.1:50055"), true, false),
        ("[::]:50055", Some("[::1]:50055"), true, false),
        ("192.0.2.1:50055", Some("127.0.0.1:50055"), true, false),
        ("127.0.0.1:50055", Some("192.0.2.1:50055"), true, false),
        ("[::1]:50055", Some("[2001:db8::1]:50055"), true, false),
        ("127.0.0.1:0", Some("127.0.0.1:50055"), true, false),
        ("127.0.0.1:50055", Some("127.0.0.1:0"), true, false),
        ("0.0.0.0:50055", Some("192.0.2.1:50055"), false, true),
    ] {
        let mut args = vec!["orbitkv-cache-manager", "--addr", bind];
        if let Some(advertised) = advertised {
            args.extend(["--peer-advertise-addr", advertised]);
        }
        if enabled {
            args.push("--enable-query-control");
        }
        let cli = Cli::try_parse_from(args).unwrap();
        let address = cli.query_control_address();
        assert_eq!(
            address.is_ok(),
            accepted,
            "{bind}, {advertised:?}, {enabled}"
        );
        if accepted {
            assert_eq!(
                address.unwrap(),
                enabled.then(|| advertised.unwrap_or(bind).parse().unwrap())
            );
        }
    }
}

#[cfg(feature = "orbitte-rdma")]
#[test]
fn orbitte_rdma_cli_requires_explicit_device_and_valid_port() {
    let base = [
        "orbitkv-cache-manager",
        "--peer-transfer-backend",
        "orbitte-rdma",
    ];
    assert!(Cli::try_parse_from(base).is_err());
    assert!(
        Cli::try_parse_from(base.into_iter().chain([
            "--orbitte-rdma-device",
            "mlx5_0",
            "--orbitte-rdma-port",
            "0"
        ]))
        .is_err()
    );
    assert!(
        Cli::try_parse_from(["orbitkv-cache-manager", "--orbitte-rdma-gid-index", "3"]).is_err()
    );
    let cli = Cli::try_parse_from(base.into_iter().chain([
        "--orbitte-rdma-device",
        "mlx5_0",
        "--orbitte-rdma-port",
        "1",
        "--orbitte-rdma-gid-index",
        "3",
    ]))
    .unwrap();
    assert_eq!(
        cli.peer_transfer_backend,
        orbitkv_core::PeerTransferBackend::OrbitTeRdma
    );
    assert_eq!(cli.orbitte_rdma_device.as_deref(), Some("mlx5_0"));
    assert_eq!((cli.orbitte_rdma_port, cli.orbitte_rdma_gid_index), (1, 3));
    assert_eq!(
        Cli::try_parse_from(["orbitkv-cache-manager"])
            .unwrap()
            .peer_transfer_backend,
        orbitkv_core::PeerTransferBackend::Mooncake
    );
}
