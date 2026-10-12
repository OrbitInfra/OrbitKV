"""Reject unmatched cohorts, absent physical work, and invented cost evidence."""

import copy
import csv
import json
import signal
from argparse import Namespace
from types import SimpleNamespace

import pytest

from benches import cost_observations as costs
from benches.metrics import REQUIRED_OWNERSHIP_GAUGES, delta, engine_itl_summary, metrics

SLO = {"ttft_ms": 2000, "average_decode_ms_per_token": 100, "itl_ms": 100}


def run_files(directory, mode="on", tier="ssd", job=None):
    directory.mkdir(parents=True)
    manifest = {
        "cost_observations": "1" if mode == "on" else "0",
        "observation_mode": "sampled",
        "manager_present": True,
        "manager_backend": "orbitkv",
        "manager_binary_sha256": "one-prebuilt-manager",
        "model_revision": "immutable-model",
        "packages": {"vllm": "0.29.0"},
        "gpu": "same-device",
        "kv_bytes_per_token": 147456,
        "arguments": {
            "engine": "vllm",
            "backend": "orbitkv",
            "workload": "sustained",
            "observation_mode": "sampled",
            "concurrencies": [4],
            "max_requests": 4,
            "duration_seconds": 3600,
            "working_set": 2,
            "ssd_gib": 16 if tier == "ssd" else 0,
        },
    }
    if job:
        manifest["arguments"] = copy.deepcopy(job["arguments"])
        manifest["manager_command"] = [job["manager"]]
    manifest["observation_mode"] = manifest["arguments"]["observation_mode"]
    if mode in ("direct", "kernel"):
        manifest["arguments"]["orbitkv_transfer_backend"] = mode
        (directory / "manager.log").write_text(
            f"GPU worker initialized: device=0 backend={mode} device_id=0\n"
        )
    count = manifest["arguments"]["max_requests"]
    counters = {
        "orbitkv_load_bytes_total": 8192,
        "orbitkv_save_bytes_total": 8192,
        "orbitkv_ssd_prefetch_bytes_total": 8192 if tier == "ssd" else 0,
        "orbitkv_ssd_write_bytes_total": 16384 if tier == "ssd" else 0,
    }
    if mode == "on":
        counters.update(
            {
                "orbitkv_cost_operations_total{path=ssd_read,outcome=success}": 2,
                "orbitkv_cost_operations_total{path=ssd_read,outcome=cancelled}": 1,
                "orbitkv_cost_prediction_absolute_error_seconds_count": 2,
                "orbitkv_cost_prediction_absolute_error_seconds_sum": 0.004,
                "orbitkv_cost_shadow_decisions_total": 2,
            }
        )
    window = {
        "concurrency": 4,
        "submitted_requests": count,
        "peak_client_inflight": 4,
        "stop_reason": "request_limit",
        "wall_seconds": 1,
        "drain_seconds": 0.1,
        "working_set_tokens": 8192,
        "manager_delta": counters,
        "manager_before": dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
        "manager_after": dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
        "sampled_peak_bytes": {"orbitkv_query_reserved_bytes": 0, "orbitkv_inflight_bytes": 0},
        "observation_mode": manifest["arguments"]["observation_mode"],
        "manager_present": True,
        "manager_backend": "orbitkv",
        "memory_samples": 1,
        "metrics_delta": {
            "vllm:inter_token_latency_seconds_count": 100,
            "vllm:inter_token_latency_seconds_sum": 1,
            "vllm:inter_token_latency_seconds_bucket{le=0.01}": 80,
            "vllm:inter_token_latency_seconds_bucket{le=0.1}": 100,
            "vllm:inter_token_latency_seconds_bucket{le=+Inf}": 100,
        },
    }
    samples = [
        {
            "concurrency": 4,
            "index": index,
            "kind": "reuse",
            "prefix_index": 0,
            "length": 4096,
            "submitted_seconds": 0,
            "started_seconds": 0,
            "finished_seconds": 0.2,
            "ttft_ms": 10,
            "e2e_ms": 110,
            "usage": {"completion_tokens": 16},
            "cached_tokens": 4096,
            "matches_reference_output": True,
            "prompt_sha256": str(index),
            "text": "same",
        }
        for index in range(count)
    ]
    for name, value in (
        ("manifest.json", manifest),
        (
            "storage.json",
            {"direct_io": True, "mount": "same-overlay-mount", "capacity_bytes": 16 * 1024**3},
        ),
        ("process.json", {"exit_code": 0}),
    ):
        (directory / name).write_text(json.dumps(value))
    (directory / "windows.jsonl").write_text(json.dumps(window) + "\n")
    (directory / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    return window


def arguments(tmp_path):
    return Namespace(
        comparison="observations",
        tiers=["dram", "ssd"],
        engines=["vllm", "sglang"],
        storage_codecs=["none", "ans"],
        encoded_tiers=["ssd"],
        pairs=3,
        output=tmp_path,
        model=tmp_path / "model",
        manager=tmp_path / "manager",
        ssd_dir=tmp_path / "ssd",
        requests=64,
        seed=42,
        observation_mode="sampled",
    )


def test_plan_reverses_order_and_keeps_engine_tier_codec_controls_matched(tmp_path):
    jobs = costs.plan(arguments(tmp_path))
    assert len(jobs) == 2 * 3 * 3 * 2
    assert [job["mode"] for job in jobs[:6]] == ["off", "on", "on", "off", "off", "on"]
    for job in jobs:
        command = job["command"]
        assert "cargo" not in command
        assert "--trace-transfers" not in command
        assert command[command.index("--ssd-gib") + 1] == ("16" if job["tier"] == "ssd" else "0")
        assert command[command.index("--storage-codec") + 1] == job["codec"]
        assert command[command.index("--max-requests") + 1] == "64"


def test_transfer_backend_plan_reverses_both_engines_without_changing_other_arguments(tmp_path):
    args = arguments(tmp_path)
    args.comparison, args.tiers, args.storage_codecs = "transfer-backends", ["dram"], ["none"]
    jobs = costs.plan(args)
    assert len(jobs) == 12
    assert [job["mode"] for job in jobs[:6]] == [
        "direct",
        "kernel",
        "kernel",
        "direct",
        "direct",
        "kernel",
    ]
    for job in jobs:
        assert job["arguments"]["orbitkv_transfer_backend"] == job["mode"]
        assert job["arguments"]["ssd_gib"] == 0
        command = job["command"]
        assert command[command.index("--orbitkv-transfer-backend") + 1] == job["mode"]


@pytest.mark.parametrize("workers", ["", "direct", "kernel\ndirect"])
def test_backend_comparison_requires_real_worker_selection(tmp_path, workers):
    directory = tmp_path / "kernel"
    run_files(directory, "kernel", "dram")
    (directory / "manager.log").write_text(
        "\n".join(
            f"GPU worker initialized: device=0 backend={mode} device_id=0"
            for mode in workers.splitlines()
        )
    )
    with pytest.raises(ValueError, match="Manager workers"):
        costs.read_run(directory, "kernel", "dram", SLO)


def test_backend_pairs_only_allow_the_selected_backend_to_change(tmp_path):
    for mode in ("direct", "kernel"):
        run_files(tmp_path / mode, mode, "dram")
    direct = costs.read_run(tmp_path / "direct", "direct", "dram", SLO)
    kernel = costs.read_run(tmp_path / "kernel", "kernel", "dram", SLO)
    pair = costs.compare_pair(direct, kernel, "transfer-backends")
    assert pair["overhead_percent"] == dict.fromkeys(costs.BUDGETS, 0)
    assert pair["direct"]["cost"] == pair["kernel"]["cost"] == {}
    for key in ("seed", "ssd_read_path", "host_gib"):
        changed = copy.deepcopy(kernel)
        changed["manifest"]["arguments"][key] = "different"
        with pytest.raises(ValueError, match="Unmatched run argument"):
            costs.compare_pair(direct, changed, "transfer-backends")
    for key, value in (
        ("cost_observations", "1"),
        ("storage_codec", "ans"),
        ("orbitkv_transfer_backend", None),
    ):
        changed = copy.deepcopy(kernel)
        target = (
            changed["manifest"] if key == "cost_observations" else changed["manifest"]["arguments"]
        )
        target[key] = value
        with pytest.raises(ValueError, match="Backend comparison requires"):
            costs.compare_pair(direct, changed, "transfer-backends")
    with pytest.raises(ValueError, match="Unmatched run argument"):
        costs.compare_pair(direct, kernel)


def test_backend_report_uses_direct_kernel_labels_and_rejects_observation_samples(tmp_path):
    args = arguments(tmp_path)
    args.comparison, args.tiers, args.storage_codecs = "transfer-backends", ["dram"], ["none"]
    args.engines = ["vllm"]
    jobs = costs.plan(args)
    for job in jobs:
        run_files(tmp_path / "runs" / job["name"], job["mode"], "dram", job)
    result = costs.summarize(args, jobs, SLO)
    assert result["status"] == "passed"
    assert result["control"] == "direct" and result["candidate"] == "kernel"
    assert "both D2H saves and H2D restores" in result["scope"]
    assert "off" not in result["cells"][0]["pairs"][0]
    costs.write_summary(tmp_path / "final", result)
    with (tmp_path / "final/summary.csv").open() as output:
        (row,) = list(csv.DictReader(output))
    assert (row["comparison"], row["control"], row["candidate"]) == (
        "transfer-backends",
        "direct",
        "kernel",
    )
    file = tmp_path / "runs" / jobs[1]["name"] / "windows.jsonl"
    window = json.loads(file.read_text())
    window["manager_delta"]["orbitkv_cost_operations_total"] = 1
    file.write_text(json.dumps(window))
    failed = costs.summarize(args, jobs, SLO)
    assert failed["status"] == "failed"
    assert any("Disabled observations" in message for message in failed["cells"][0]["failures"])


def test_backend_comparison_requires_d2h_and_h2d_bytes(tmp_path):
    directory = tmp_path / "direct"
    window = run_files(directory, "direct", "dram")
    window["manager_delta"].pop("orbitkv_save_bytes_total")
    (directory / "windows.jsonl").write_text(json.dumps(window))
    run = costs.read_run(directory, "direct", "dram", SLO)
    assert any("GPU save bytes" in message for message in run["evidence"]["failures"])


def test_cost_labels_keep_terminal_outcomes_and_stage_boundaries(monkeypatch):
    monkeypatch.setattr(
        "benches.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text="\n".join(
                [
                    'orbitkv_cost_operations_total{path="ssd_read",outcome="success"} 2',
                    'orbitkv_cost_operations_total{path="ssd_read",outcome="cancelled"} 3',
                    'orbitkv_cost_stage_seconds_sum{path="ssd_read",stage="service",outcome="success"} 0.2',
                    'orbitkv_cost_stage_seconds_sum{path="ssd_read",stage="total",outcome="success"} 0.4',
                    'orbitkv_cost_shadow_candidates_total{path="ssd_read",evidence="unknown"} 1',
                ]
            ),
            raise_for_status=lambda: None,
        ),
    )
    observed = metrics("http://manager")
    assert observed["orbitkv_cost_operations_total"] == 5
    assert observed["orbitkv_cost_operations_total{path=ssd_read,outcome=cancelled}"] == 3
    assert (
        observed["orbitkv_cost_stage_seconds_sum{path=ssd_read,stage=service,outcome=success}"]
        == 0.2
    )
    assert observed["orbitkv_cost_shadow_candidates_total{path=ssd_read,evidence=unknown}"] == 1


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_engine_itl_buckets_keep_zero_deltas_and_window_quantile_bounds(monkeypatch, engine):
    prefix = f"{engine}:inter_token_latency_seconds"
    monkeypatch.setattr(
        "benches.metrics.requests.get",
        lambda *a, **kw: SimpleNamespace(
            text="\n".join(
                [
                    f'{prefix}_bucket{{engine="0",le="0.01"}} 0',
                    f'{prefix}_bucket{{engine="0",le="0.1"}} 96',
                    f'{prefix}_bucket{{engine="0",le="1.0"}} 100',
                    f'{prefix}_bucket{{engine="0",le="+Inf"}} 100',
                    f'{prefix}_count{{engine="0"}} 100',
                    f'{prefix}_sum{{engine="0"}} 5',
                    'unrelated_latency_bucket{le="0.1"} 100',
                ]
            ),
            raise_for_status=lambda: None,
        ),
    )
    counters = delta({}, metrics("http://engine"))
    assert counters[f"{prefix}_bucket{{le=0.01}}"] == 0
    assert "unrelated_latency_bucket" not in counters
    summary = engine_itl_summary(engine, counters, 100)
    assert summary["status"] == "observed"
    assert summary["p50_ms"] == pytest.approx((0.01 + 0.09 * 50 / 96) * 1000)
    assert summary["p95_bounds_ms"] == [10, 100]
    assert summary["p99_bounds_ms"] == [100, 1000]
    assert summary["within_slo_fraction"] == 0.96
    assert summary["mean_ms"] == 50
    assert "No request-level goodput attribution" in summary["scope"]
    json.dumps(summary, allow_nan=False)


def test_missing_or_overflow_itl_is_not_an_invented_zero_latency():
    missing = engine_itl_summary("vllm", {}, 100)
    assert missing["status"] == "missing"
    assert missing["p99_ms"] is None
    counters = {
        "vllm:inter_token_latency_seconds_count": 100,
        "vllm:inter_token_latency_seconds_bucket{le=0.1}": 90,
        "vllm:inter_token_latency_seconds_bucket{le=+Inf}": 100,
    }
    overflow = engine_itl_summary("vllm", counters, 100)
    assert overflow["p99_ms"] is None
    assert overflow["p99_bounds_ms"] == [100, None]
    assert overflow["mean_ms"] is None
    counters["vllm:inter_token_latency_seconds_bucket{le=0.1}"] = 101
    assert engine_itl_summary("vllm", counters, 100)["status"] == "invalid_buckets"


@pytest.mark.parametrize(
    "absent",
    [
        "orbitkv_load_bytes_total",
        "orbitkv_ssd_prefetch_bytes_total",
        "orbitkv_ssd_write_bytes_total",
        "orbitkv_cost_prediction_absolute_error_seconds_count",
        "orbitkv_cost_shadow_decisions_total",
    ],
)
def test_missing_tier_or_prediction_evidence_cannot_pass(tmp_path, absent):
    window = run_files(tmp_path / "run")
    window["manager_delta"].pop(absent)
    (tmp_path / "run/windows.jsonl").write_text(json.dumps(window))
    run = costs.read_run(tmp_path / "run", "on", "ssd", SLO)
    assert run["evidence"]["failures"]


@pytest.mark.parametrize(
    "metric",
    [
        "orbitkv_load_failures_total",
        "orbitkv_storage_codec_decode_failures_total",
        "orbitkv_ssd_write_failures_total",
    ],
)
def test_successful_byte_counters_do_not_hide_execution_errors(tmp_path, metric):
    window = run_files(tmp_path / "run")
    window["manager_delta"][metric] = 1
    (tmp_path / "run/windows.jsonl").write_text(json.dumps(window))
    run = costs.read_run(tmp_path / "run", "on", "ssd", SLO)
    assert any(metric in failure for failure in run["evidence"]["failures"])


def test_encoded_composite_requires_real_work_but_not_nonexistent_shadow_alternatives(tmp_path):
    window = run_files(tmp_path / "run")
    window["manager_delta"].pop("orbitkv_cost_shadow_decisions_total")
    window["manager_delta"].update(
        {
            "orbitkv_storage_codec_batches_total_encode": 2,
            "orbitkv_storage_codec_batches_total_decode": 2,
        }
    )
    manifest_path = tmp_path / "run/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arguments"]["storage_codec"] = "ans"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "run/windows.jsonl").write_text(json.dumps(window))
    run = costs.read_run(tmp_path / "run", "on", "ssd", SLO)
    assert run["evidence"]["failures"] == []
    assert run["evidence"]["shadow_scope"].startswith("Not applicable")
    window["manager_delta"].pop("orbitkv_storage_codec_batches_total_decode")
    (tmp_path / "run/windows.jsonl").write_text(json.dumps(window))
    assert costs.read_run(tmp_path / "run", "on", "ssd", SLO)["evidence"]["failures"]


def test_report_compares_prompt_hashes_and_retains_output_differences(tmp_path):
    for mode in ("off", "on"):
        run_files(tmp_path / mode, mode)
    off = costs.read_run(tmp_path / "off", "off", "ssd", SLO)
    on = costs.read_run(tmp_path / "on", "on", "ssd", SLO)
    assert on["evidence"]["prediction_absolute_error_seconds_mean"] == 0.002
    assert on["evidence"]["goodput_requests_per_second"] == 4
    on["samples"][0]["text"] = "different"
    pair = costs.compare_pair(off, on)
    assert pair["output_mismatches"] == 1
    assert pair["overhead_percent"] == dict.fromkeys(costs.BUDGETS, 0)
    for key in (
        "host_gib",
        "seed",
        "storage_codec",
        "trace_transfers",
        "orbitkv_transfer_backend",
        "ssd_dir",
        "ssd_read_path",
    ):
        changed = copy.deepcopy(on)
        changed["manifest"]["arguments"][key] = "different"
        with pytest.raises(ValueError, match="Unmatched run argument"):
            costs.compare_pair(off, changed)
    on["samples"][0]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="prompt"):
        costs.compare_pair(off, on)


def test_unmatched_mode_and_artifacts_are_rejected(tmp_path):
    run_files(tmp_path / "run", "off")
    with pytest.raises(ValueError, match="instrumentation mode"):
        costs.read_run(tmp_path / "run", "on", "ssd", SLO)
    off = costs.read_run(tmp_path / "run", "off", "ssd", SLO)
    on = copy.deepcopy(off)
    on["manifest"]["manager_binary_sha256"] = "rebuilt-manager"
    with pytest.raises(ValueError, match="artifact"):
        costs.compare_pair(off, on)


@pytest.mark.parametrize("status", [None, {}, {"exit_code": 1}, {"exit_code": -9}])
def test_complete_samples_cannot_hide_failed_or_unknown_process_exit(tmp_path, status):
    run_files(tmp_path / "run")
    process = tmp_path / "run/process.json"
    if status is None:
        process.unlink()
    else:
        process.write_text(json.dumps(status))
    with pytest.raises((OSError, ValueError)):
        costs.read_run(tmp_path / "run", "on", "ssd", SLO)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("engine", "sglang"),
        ("storage_codec", "ans"),
        ("max_requests", 4),
        ("ssd_gib", 8),
        ("ssd_read_path", "uring"),
    ],
)
def test_two_matching_runs_cannot_replace_predeclared_configuration(tmp_path, key, value):
    job = costs.plan(arguments(tmp_path))[0]
    directory = tmp_path / "runs" / job["name"]
    run_files(directory, "off", "dram", job)
    run = costs.read_run(directory, "off", "dram", SLO)
    costs.validate_planned_run(run, job)
    run["manifest"]["arguments"][key] = value
    with pytest.raises(ValueError, match="predeclared"):
        costs.validate_planned_run(run, job)


def test_predeclared_cohort_requires_every_unique_prompt_and_manager(tmp_path):
    job = costs.plan(arguments(tmp_path))[0]
    directory = tmp_path / "runs" / job["name"]
    run_files(directory, "off", "dram", job)
    run = costs.read_run(directory, "off", "dram", SLO)
    for change in ("missing", "duplicate", "hash", "manager"):
        altered = copy.deepcopy(run)
        if change == "missing":
            altered["samples"].pop()
        elif change == "duplicate":
            altered["samples"][0]["index"] = altered["samples"][1]["index"]
        elif change == "hash":
            altered["samples"][0].pop("prompt_sha256")
        else:
            altered["manifest"]["manager_command"] = ["/different/manager"]
        with pytest.raises(ValueError, match="[Pp]redeclared"):
            costs.validate_planned_run(altered, job)


def test_different_ssd_mounts_are_not_a_paired_comparison(tmp_path):
    for mode in ("off", "on"):
        run_files(tmp_path / mode, mode)
    off = costs.read_run(tmp_path / "off", "off", "ssd", SLO)
    on = costs.read_run(tmp_path / "on", "on", "ssd", SLO)
    on["storage"]["mount"] = "different-mount"
    with pytest.raises(ValueError, match="SSD storage"):
        costs.compare_pair(off, on)


def test_preflight_failure_never_starts_a_runtime(tmp_path, monkeypatch):
    output = tmp_path / "measurement"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cost-observations",
            "--model",
            str(tmp_path / "model"),
            "--manager",
            str(tmp_path / "manager"),
            "--ssd-dir",
            str(tmp_path / "ssd"),
            "--output",
            str(output),
            "--storage-codecs",
            "none",
        ],
    )
    monkeypatch.setattr(
        costs, "preflight", lambda args: ["GPU inaccessible in this execution context"]
    )

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Runtime launched after failed preflight")

    monkeypatch.setattr(costs.subprocess, "run", unexpected_runtime)
    monkeypatch.setattr(costs.subprocess, "Popen", unexpected_runtime)
    with pytest.raises(SystemExit) as error:
        costs.main()
    assert error.value.code == 1
    assert json.loads((output / "final/summary.json").read_text())["status"] == "blocked"
    assert not (output / "runs").exists()


def test_interrupt_allows_single_node_to_clean_up_its_owned_services(tmp_path, monkeypatch):
    output = tmp_path / "measurement"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cost-observations",
            "--model",
            str(tmp_path / "model"),
            "--manager",
            str(tmp_path / "manager"),
            "--ssd-dir",
            str(tmp_path / "ssd"),
            "--output",
            str(output),
            "--storage-codecs",
            "none",
        ],
    )
    monkeypatch.setattr(costs, "preflight", lambda args: [])
    events = []

    class Process:
        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return -signal.SIGINT

        def send_signal(self, sig):
            events.append(("signal", sig))

        def kill(self):
            pytest.fail("Single-node was killed before its service cleanup completed")

    monkeypatch.setattr(costs.subprocess, "Popen", Process)
    with pytest.raises(KeyboardInterrupt):
        costs.main()
    assert events == [("wait", None), ("signal", signal.SIGINT), ("wait", 90)]


def test_absolute_slo_exceedance_is_reported_separately_from_overhead(tmp_path):
    run_files(tmp_path / "run")
    run = costs.read_run(
        tmp_path / "run",
        "on",
        "ssd",
        {
            "ttft_ms": 5,
            "average_decode_ms_per_token": 5,
            "itl_ms": 100,
        },
    )
    assert len(run["evidence"]["failures"]) == 2
    assert run["evidence"]["slo_passing_requests"] == 0


def test_incomplete_pairs_are_failures_not_zero_overhead(tmp_path):
    args = arguments(tmp_path)
    jobs = costs.plan(args)
    result = costs.summarize(args, jobs, SLO)
    assert result["status"] == "failed"
    assert all(cell["paired_median_overhead_percent"] == {} for cell in result["cells"])
    costs.write_summary(tmp_path / "final", result)
    assert (tmp_path / "final/summary.csv").is_file()
    assert "Official engine histogram" in result["itl"]


def test_three_pairs_use_median_of_paired_ratios_and_enforce_budget(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.engines, args.storage_codecs = ["vllm"], ["none"]
    jobs = costs.plan(args)[:6]
    for job in jobs:
        run_files(tmp_path / "runs" / job["name"], job["mode"], "dram", job)
    original = costs.read_run

    def adjusted(directory, mode, tier, slo):
        run = original(directory, mode, tier, slo)
        repetition = int(directory.name.split("-pair-")[1].split("-")[0])
        run["evidence"]["ttft_p95_ms"] = (100, 200, 300)[repetition - 1]
        if mode == "on":
            run["evidence"]["ttft_p95_ms"] *= (1.01, 1.07, 2)[repetition - 1]
        return run

    monkeypatch.setattr(costs, "read_run", adjusted)
    result = costs.summarize(args, jobs, SLO)
    (cell,) = result["cells"]
    assert cell["paired_median_overhead_percent"]["ttft_p95_ms"] == pytest.approx(7)
    assert cell["status"] == "failed"
    assert any("exceeds 5.0% budget" in message for message in cell["failures"])
