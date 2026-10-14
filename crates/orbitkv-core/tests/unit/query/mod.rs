use super::*;

fn reserve(budget: &Arc<QueryBudget>, instance: &str, bytes: u64) -> QueryReservation {
    match budget.reserve(instance, "state", bytes, QueryMode::Demand) {
        QueryAdmission::Admitted(reservation) => reservation,
        _ => panic!("expected admission"),
    }
}

#[test]
fn bytes_remain_charged_until_all_consumers_finish() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let first = reserve(&budget, "a", 70);
    assert!(matches!(
        budget.reserve("a", "state", 11, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    let second = reserve(&budget, "b", 30);
    assert!(matches!(
        budget.reserve("c", "state", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "state", 81, QueryMode::Demand),
        QueryAdmission::TooLarge
    ));
    assert_eq!(budget.total_peak.load(Ordering::Relaxed), 100);
    assert_eq!(budget.instance_peak.load(Ordering::Relaxed), 70);
    assert!(first.ready(71).is_err());
    first.ready(40).unwrap();
    let gpu = first.clone();
    gpu.restoring();
    drop(first);
    assert_eq!(budget.usage.lock().total, 70);
    drop(gpu);
    assert_eq!(budget.usage.lock().total, 30);
    drop(second);
    assert_eq!(budget.usage.lock().total, 0);
    assert!(budget.usage.lock().instances.is_empty());
    assert_eq!(budget.total_peak.load(Ordering::Relaxed), 100);
    assert_eq!(budget.instance_peak.load(Ordering::Relaxed), 70);
    let next = reserve(&budget, "a", 80);
    assert_eq!(budget.total_peak.load(Ordering::Relaxed), 100);
    assert_eq!(budget.instance_peak.load(Ordering::Relaxed), 80);
    drop(next);
    assert!(budget.usage.lock().instances.is_empty());
}

#[test]
fn warmups_leave_foreground_headroom_and_hold_bytes_until_io_drains() {
    let budget = QueryBudget::new(100, 80).unwrap();
    let QueryAdmission::Admitted(first) = budget.reserve("a", "ns", 20, QueryMode::Warmup) else {
        panic!("warmup should fit");
    };
    assert!(
        first.ready(20).is_err(),
        "warmup cannot become a restore lease"
    );
    assert!(matches!(
        budget.reserve("a", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "ns", 21, QueryMode::Warmup),
        QueryAdmission::TooLarge
    ));
    let QueryAdmission::Admitted(second) = budget.reserve("b", "ns", 5, QueryMode::Warmup) else {
        panic!("global warmup budget should fit");
    };
    assert!(matches!(
        budget.reserve("c", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    let foreground = reserve(&budget, "a", 60);
    let other_foreground = reserve(&budget, "b", 15);
    let submitted_io = first.clone();
    drop(first);
    assert_eq!(budget.usage.lock().warming, 25);
    drop(submitted_io);
    assert_eq!(budget.usage.lock().warming, 5);
    drop((second, foreground, other_foreground));
    let usage = budget.usage.lock();
    assert_eq!(usage.total, 0);
    assert_eq!(usage.warming, 0);
    assert!(usage.instances.is_empty());
    assert!(usage.warming_instances.is_empty());
}

#[test]
fn foreground_ownership_suppresses_new_warmups_until_the_last_gpu_owner_releases() {
    let budget = QueryBudget::new(100, 100).unwrap();
    let foreground = reserve(&budget, "a", 10);
    for phase in [Phase::Preparing, Phase::Ready, Phase::Restoring] {
        match phase {
            Phase::Ready => foreground.ready(10).unwrap(),
            Phase::Restoring => foreground.restoring(),
            _ => {}
        }
        assert!(matches!(
            budget.reserve("b", "ns", 1, QueryMode::Warmup),
            QueryAdmission::Busy
        ));
    }
    let gpu = foreground.clone();
    drop(foreground);
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Busy
    ));
    drop(gpu);
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Warmup),
        QueryAdmission::Admitted(_)
    ));
}

#[test]
fn prepared_pages_keep_headroom_until_claim_and_total_bytes_until_gpu_completion() {
    let budget = QueryBudget::new(400, 400).unwrap();
    let QueryAdmission::Admitted(prepared) = budget.reserve("a", "ns", 100, QueryMode::Prepare)
    else {
        panic!("preparation should fit");
    };
    assert_eq!(prepared.batch_blocks(10, 24), 2);
    assert_eq!(prepared.batch_blocks(10, 1), 1);
    prepared.ready(80).unwrap();
    assert_eq!(budget.usage.lock().warming, 80);
    assert!(matches!(
        budget.reserve("b", "ns", 21, QueryMode::Prepare),
        QueryAdmission::Busy
    ));
    let io = prepared.clone();
    prepared.claim();
    prepared.claim();
    assert_eq!(budget.usage.lock().warming, 0);
    assert_eq!(budget.usage.lock().total, 80);
    prepared.restoring();
    drop(prepared);
    assert_eq!(budget.usage.lock().total, 80);
    drop(io);
    assert_eq!(budget.usage.lock().total, 0);
    assert!(budget.usage.lock().warming_instances.is_empty());
}

#[test]
fn owned_lookahead_can_overlap_foreground_only_within_both_budgets() {
    let budget = QueryBudget::new(100, 100).unwrap();
    let foreground = reserve(&budget, "a", 70);
    let QueryAdmission::Admitted(prepared) = budget.reserve("a", "ns", 25, QueryMode::Prepare)
    else {
        panic!("owned lookahead should use available speculative headroom");
    };
    assert!(matches!(
        budget.reserve("b", "ns", 1, QueryMode::Prepare),
        QueryAdmission::Busy
    ));
    assert!(matches!(
        budget.reserve("b", "ns", 6, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    let remaining_foreground = reserve(&budget, "b", 5);
    prepared.ready(25).unwrap();
    assert_eq!(budget.usage.lock().total, 100);
    drop((foreground, prepared, remaining_foreground));
    assert_eq!(budget.usage.lock().total, 0);
}

#[test]
fn exported_peaks_cover_scrape_gaps_without_blocking_admission_or_retaining_owners() {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "query::tests::query_peak_metrics_child",
            "--ignored",
            "--nocapture",
        ])
        .env("ORBITKV_QUERY_PEAK_METRICS_CHILD", "1")
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
#[ignore = "Invoked in a fresh process by the budget metrics export test"]
fn query_peak_metrics_child() {
    if std::env::var("ORBITKV_QUERY_PEAK_METRICS_CHILD").as_deref() != Ok("1") {
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
    let budget = QueryBudget::new(100, 80).unwrap();
    let weak = Arc::downgrade(&budget.total_peak);

    let locked = budget.usage.lock();
    let (tx, rx) = std::sync::mpsc::channel();
    let collector_registry = registry.clone();
    let collector = std::thread::spawn(move || tx.send(collector_registry.gather()).unwrap());
    let observed = rx.recv_timeout(std::time::Duration::from_secs(5));
    drop(locked);
    collector.join().unwrap();
    assert!(
        observed.is_ok(),
        "scraping must not wait for admission's mutex"
    );
    assert_exported_peaks(&registry, 0, 0);

    let first = reserve(&budget, "a", 70);
    let second = reserve(&budget, "a", 10);
    let other = reserve(&budget, "b", 20);
    assert!(matches!(
        budget.reserve("b", "state", 1, QueryMode::Demand),
        QueryAdmission::Busy
    ));
    first.ready(40).unwrap();
    let consumer = first.clone();
    consumer.restoring();
    drop((first, second, other));
    drop(consumer);
    assert!(budget.usage.lock().instances.is_empty());
    assert_exported_peaks(&registry, 100, 80);
    drop(budget);
    assert!(
        weak.upgrade().is_none(),
        "metric callbacks must not keep a budget alive"
    );
    let replacement = QueryBudget::new(100, 80).unwrap();
    assert_exported_peaks(&registry, 0, 0);
    drop(replacement);
    provider.shutdown().unwrap();
}

fn assert_exported_peaks(registry: &prometheus::Registry, total: u64, instance: u64) {
    let values: HashMap<_, _> = registry
        .gather()
        .iter()
        .filter_map(|family| {
            let name = family.name();
            if !matches!(
                name,
                "orbitkv_query_reserved_peak_bytes" | "orbitkv_query_instance_reserved_peak_bytes"
            ) {
                return None;
            }
            assert_eq!(family.get_metric().len(), 1);
            let metric = &family.get_metric()[0];
            assert!(
                metric
                    .get_label()
                    .iter()
                    .all(|label| !label.name().contains("instance"))
            );
            Some((name.to_string(), metric.get_gauge().value() as u64))
        })
        .collect();
    assert_eq!(
        values.get("orbitkv_query_reserved_peak_bytes"),
        Some(&total)
    );
    assert_eq!(
        values.get("orbitkv_query_instance_reserved_peak_bytes"),
        Some(&instance)
    );
}

#[test]
fn fresh_query_owner_exports_initial_phase_series() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "query::tests::initial_phase_series_child",
            "--ignored",
            "--nocapture",
        ])
        .env("ORBITKV_INITIAL_QUERY_PHASE_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "Invoked in a fresh process by the query phase exporter gate"]
fn initial_phase_series_child() {
    if std::env::var("ORBITKV_INITIAL_QUERY_PHASE_CHILD").as_deref() != Ok("1") {
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
    opentelemetry::global::set_meter_provider(provider.clone());
    let budget = QueryBudget::new(64, 64).unwrap();
    let assert_phases = |active: Option<&str>| {
        let families = registry.gather();
        let family = families
            .iter()
            .find(|family| family.name() == "orbitkv_query_reserved_bytes_by_phase")
            .expect("fresh query owner must export each phase before any query");
        let values = family
            .get_metric()
            .iter()
            .map(|metric| {
                let phase = metric
                    .get_label()
                    .iter()
                    .find(|label| label.name() == "phase")
                    .expect("query phase label");
                (phase.value().to_string(), metric.get_gauge().value())
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(values.len(), 6);
        for phase in [
            "warming",
            "preloading",
            "prepared",
            "preparing",
            "ready",
            "restoring",
        ] {
            assert_eq!(
                values.get(phase),
                Some(&if active == Some(phase) { 7.0 } else { 0.0 }),
                "{phase}"
            );
        }
    };
    assert_phases(None);
    let QueryAdmission::Admitted(reservation) =
        budget.reserve("initial-metrics", "initial-metrics", 7, QueryMode::Demand)
    else {
        panic!("query must fit initialized budget");
    };
    assert_phases(Some("preparing"));
    reservation.ready(7).unwrap();
    assert_phases(Some("ready"));
    drop(reservation);
    assert_phases(None);
    // A second owner initializes labels without resetting another owner's live
    // values or introducing extra label dimensions.
    let QueryAdmission::Admitted(reservation) =
        budget.reserve("initial-metrics", "initial-metrics", 7, QueryMode::Demand)
    else {
        panic!("second query must fit");
    };
    let _second_budget = QueryBudget::new(64, 64).unwrap();
    assert_phases(Some("preparing"));
    drop(reservation);
    assert_phases(None);
    provider.shutdown().unwrap();
}
