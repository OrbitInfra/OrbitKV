"""Timeline evidence must belong to measured requests and one local clock."""

import json

from benches.common.timeline import collect


def test_timeline_does_not_pair_distinct_process_clocks_or_count_seed_requests(tmp_path):
    events = [
        {"request_id": "cmpl-abc-0-a1b2c3", "stage": "queued", "pid": 1, "monotonic_ns": 0},
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "restore_submit",
            "pid": 2,
            "monotonic_ns": 10,
        },
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "gpu_ready",
            "pid": 2,
            "monotonic_ns": 1000010,
        },
        {
            "request_id": "cmpl-abc-0-a1b2c3",
            "stage": "first_use",
            "pid": 1,
            "monotonic_ns": 3000000,
        },
        {"request_id": "seed", "stage": "queued", "pid": 1, "monotonic_ns": 0},
    ]
    (tmp_path / "engine.log").write_text(
        "\n".join("prefix cache_timeline " + json.dumps(e) for e in events)
    )
    event = {
        "request_id": "cmpl-abc-0-a1b2c3",
        "stage": "source_ready",
        "pid": 3,
        "warmup": True,
        "elapsed_us": 1500,
    }
    (tmp_path / "manager.log").write_text("cache_timeline " + json.dumps(event))
    summary = collect(tmp_path, [{"request_id": "cmpl-abc"}, {"request_id": "untraced"}])
    assert summary["measured_requests"] == 2
    assert summary["traced_requests"] == 1
    assert summary["stage_counts"]["queued"] == 1
    assert summary["intervals"]["queued_to_first_use_ms"]["p50"] == 3
    assert summary["intervals"]["restore_ms"]["p50"] == 1
    assert summary["intervals"]["warmup_prepare_ms"]["p50"] == 1.5


def test_batched_completion_is_linked_without_counting_each_request_as_a_transfer(tmp_path):
    (tmp_path / "engine.log").write_text(
        "\n".join(
            "cache_timeline "
            + json.dumps({"request_id": rid, "stage": "restore_link", "restore_key": "manager:1:2"})
            for rid in ("a", "b")
        )
    )
    (tmp_path / "manager.log").write_text(
        "\n".join(
            "cache_timeline "
            + json.dumps(
                {"stage": "restore_notification", "restore_key": key, "elapsed_us": elapsed}
            )
            for key, elapsed in (("manager:1:2", 3500), ("manager:2:2", 999999))
        )
    )
    summary = collect(tmp_path, [{"request_id": "a"}, {"request_id": "b"}])
    assert summary["traced_requests"] == 2
    assert summary["intervals"]["completion_signal_ms"] == {
        "count": 1,
        "p50": 3.5,
        "p95": 3.5,
        "p99": 3.5,
    }
    assert summary["completion_coverage"]["linked_restore_batches"] == 1
    assert summary["completion_coverage"]["batches_with_notification"] == 1
    assert summary["completion_coverage"]["batches_without_notification"] == 0


def test_shared_memory_completion_reports_client_latency_and_missing_observations(tmp_path):
    events = []
    for rid, key in (("a", "manager:1:2"), ("b", "manager:1:2"), ("c", "manager:1:3")):
        events.extend(
            [
                {"request_id": rid, "stage": "restore_submit", "pid": 2, "monotonic_ns": 100},
                {"request_id": rid, "stage": "restore_link", "restore_key": key},
            ]
        )
        if rid != "c":
            events.append(
                {
                    "request_id": rid,
                    "stage": "gpu_ready",
                    "pid": 2,
                    "monotonic_ns": 3_000_100,
                }
            )
    (tmp_path / "engine.log").write_text(
        "\n".join("cache_timeline " + json.dumps(event) for event in events)
    )
    (tmp_path / "manager.log").write_text(
        "\n".join(
            "cache_timeline "
            + json.dumps({"stage": stage, "restore_key": "manager:1:2", "elapsed_us": elapsed})
            for stage, elapsed in (("restore_complete", 1500), ("restore_notification", 250))
        )
    )
    samples = [{"request_id": rid} for rid in ("a", "b", "c")]
    summary = collect(tmp_path, samples)
    assert summary["completion_coverage"] == {
        "linked_restore_batches": 2,
        "batches_with_worker_terminal": 1,
        "batches_with_local_terminal": 0,
        "batches_with_notification": 1,
        "batches_without_worker_terminal": 1,
        "batches_without_notification": 1,
        "client_restore_submissions": 3,
        "client_restore_intervals": 2,
    }
    assert summary["intervals"]["restore_ms"] == {
        "count": 2,
        "p50": 3,
        "p95": 3,
        "p99": 3,
    }
    assert summary["intervals"]["manager_restore_ms"]["count"] == 1
    assert summary["intervals"]["completion_signal_ms"]["p50"] == 0.25
    assert json.loads((tmp_path / "timeline-summary.json").read_text()) == summary


def test_equal_pids_in_distinct_logs_do_not_create_a_restore_interval(tmp_path):
    for filename, stage, now in (
        ("manager.log", "restore_submit", 100),
        ("engine.log", "gpu_ready", 1_000_100),
    ):
        (tmp_path / filename).write_text(
            "cache_timeline "
            + json.dumps({"request_id": "a", "stage": stage, "pid": 1, "monotonic_ns": now})
        )
    summary = collect(tmp_path, [{"request_id": "a"}])
    assert summary["intervals"]["restore_ms"] == {
        "count": 0,
        "p50": None,
        "p95": None,
        "p99": None,
    }
    assert summary["completion_coverage"]["client_restore_submissions"] == 0
    assert summary["completion_coverage"]["client_restore_intervals"] == 0


def test_native_timing_uses_reported_durations_not_manager_clock(tmp_path):
    links = [
        {"stage": "restore_link", "request_id": rid, "restore_key": "manager:1:2:3"}
        for rid in ("a", "b")
    ]
    links.append(
        {
            "stage": "local_restore_observed",
            "restore_key": "manager:1:2:3",
            "elapsed_ns": 250_000,
            "success": True,
        }
    )
    (tmp_path / "engine.log").write_text(
        "\n".join("cache_timeline " + json.dumps(event) for event in links)
    )
    event = {
        "stage": "local_restore_complete",
        "restore_key": "manager:1:2:3",
        "success": True,
        "readiness_ns": 100_000,
        "dispatched_ns": 300_000,
        "dequeued_ns": 600_000,
        "claimed_ns": 1_000_000,
        "submitted_ns": 1_500_000,
        "drained_ns": 2_000_000,
        "at_unix_ns": 999_000_000_000,
    }
    (tmp_path / "manager.log").write_text("cache_timeline " + json.dumps(event))
    result = collect(tmp_path, [{"request_id": "a"}, {"request_id": "b"}])
    intervals = result["intervals"]
    assert intervals["native_restore_ms"]["count"] == 1
    assert intervals["native_restore_ms"]["p50"] == 2
    assert intervals["native_consumer_wait_ms"]["p50"] == 0.25
    assert intervals["native_readiness_ms"]["p50"] == 0.1
    assert intervals["native_drain_wait_ms"]["p50"] == 0.5
    assert intervals["manager_restore_ms"]["count"] == 0
    assert result["completion_coverage"]["batches_with_local_terminal"] == 1
    assert result["completion_coverage"]["batches_without_notification"] == 0
    event["success"] = False
    (tmp_path / "manager.log").write_text("cache_timeline " + json.dumps(event))
    result = collect(tmp_path, [{"request_id": "a"}])
    assert result["intervals"]["native_restore_ms"]["count"] == 0
