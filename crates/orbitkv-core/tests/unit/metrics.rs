use super::*;

#[test]
fn initial_ownership_metrics_export_before_cache_activity() {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "metrics::tests::initial_ownership_metrics_child",
            "--ignored",
            "--nocapture",
        ])
        .env("ORBITKV_INITIAL_METRICS_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
#[ignore = "Invoked by the initial metrics test with a fresh global provider"]
fn initial_ownership_metrics_child() {
    if std::env::var("ORBITKV_INITIAL_METRICS_CHILD").as_deref() != Ok("1") {
        return;
    }
    let registry = prometheus::Registry::new();
    let reader = opentelemetry_prometheus::exporter()
        .with_registry(registry.clone())
        .build()
        .unwrap();
    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(reader)
        .build();
    global::set_meter_provider(provider.clone());
    let metrics = core_metrics();
    let counters = [
        (
            "orbitkv_query_reserved_bytes",
            &metrics.query_reserved_bytes,
        ),
        ("orbitkv_inflight_bytes", &metrics.inflight_bytes),
        (
            "orbitkv_transfer_lock_active",
            &metrics.transfer_lock_active,
        ),
        (
            "orbitkv_transfer_reserved_bytes",
            &metrics.transfer_reserved_bytes,
        ),
        (
            "orbitkv_ssd_prefetch_inflight",
            &metrics.ssd_prefetch_inflight,
        ),
        (
            "orbitkv_ssd_read_pinned_bytes",
            &metrics.ssd_read_pinned_bytes,
        ),
        (
            "orbitkv_ssd_write_queue_pending",
            &metrics.ssd_write_queue_pending,
        ),
        ("orbitkv_ssd_write_inflight", &metrics.ssd_write_inflight),
        #[cfg(feature = "mooncake")]
        (
            "orbitkv_transfer_completion_outstanding",
            &metrics.transfer_completion_outstanding,
        ),
    ];
    let assert_gauges = |expected| {
        let families = registry.gather();
        for (name, _) in &counters {
            let family = families
                .iter()
                .find(|family| family.name() == *name)
                .unwrap_or_else(|| panic!("Missing ownership metric: {name}"));
            assert_eq!(family.get_metric().len(), 1, "{name}");
            assert_eq!(
                family.get_metric()[0].get_gauge().value(),
                expected,
                "{name}"
            );
        }
    };
    assert_gauges(0.0);
    for (_, counter) in &counters {
        counter.add(7, &[]);
    }
    assert!(std::ptr::eq(metrics, core_metrics()));
    assert_gauges(7.0);
    for (_, counter) in &counters {
        counter.add(-7, &[]);
    }
    assert_gauges(0.0);
    let load_bytes = |expected| {
        let families = registry.gather();
        let family = families
            .iter()
            .find(|family| family.name() == "orbitkv_load_bytes_total")
            .unwrap();
        assert_eq!(family.get_metric().len(), 1);
        assert_eq!(family.get_metric()[0].get_counter().value(), expected);
    };
    load_bytes(0.0);
    metrics.load_bytes.add(4096, &[]);
    load_bytes(4096.0);
    provider.shutdown().unwrap();
}

#[test]
fn cache_residence_duration_boundaries_cover_one_second_to_one_day() {
    assert_eq!(
        cache_residence_duration_seconds_boundaries(),
        vec![
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1_800.0, 3_600.0, 7_200.0, 21_600.0,
            43_200.0, 86_400.0,
        ]
    );
}

#[test]
fn cache_residence_reason_attributes_are_fixed_and_low_cardinality() {
    assert_eq!(
        &*CACHE_RESIDENCE_REASON_PRESSURE,
        &[KeyValue::new("reason", "pressure")]
    );
    assert_eq!(
        &*CACHE_RESIDENCE_REASON_CLEANUP,
        &[KeyValue::new("reason", "cleanup")]
    );
}

#[cfg(feature = "mooncake")]
#[test]
fn remote_fetch_plan_segment_boundaries_cover_failures_and_fragmented_plans() {
    assert_eq!(
        remote_fetch_plan_segment_boundaries(),
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0,
        ]
    );
}
