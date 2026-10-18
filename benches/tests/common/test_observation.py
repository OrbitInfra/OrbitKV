"""Keep request timing free of polling without weakening boundary checks."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest

from benches.common.metrics import (
    REQUIRED_OWNERSHIP_GAUGES,
    codec_summary,
    measure,
    sampled_peak,
    validate_observation,
    workload_phases,
)
from benches.common.report import collect_run


@pytest.mark.parametrize("manager", [None, "manager"])
def test_boundary_mode_has_no_request_time_observer(monkeypatch, manager):
    foreground = False
    calls = []

    def metrics(url):
        assert not foreground, "Metrics were polled inside the request interval"
        calls.append(url)
        return dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0) if url else {}

    def thread(*args, **kwargs):
        pytest.fail("Boundary mode must not construct an observer thread")

    monkeypatch.setattr("benches.common.metrics.metrics", metrics)
    monkeypatch.setattr("benches.common.metrics.threading.Thread", thread)
    monkeypatch.setattr("benches.common.metrics.time.sleep", lambda seconds: None)
    backend = "orbitkv" if manager is not None else "native"
    with measure("engine", manager, 0, backend=backend, observation_mode="boundary") as result:
        foreground = True
        foreground = False
    validate_observation("boundary", [result], manager_backend="orbitkv" if manager else None)
    assert calls == ["engine", manager, manager, manager, manager, "engine"]
    assert result["memory_samples"] == 0
    assert result["sampled_peak_bytes"] is None
    assert sampled_peak([result], "orbitkv_pool_used_bytes") is None


def test_sampled_mode_owns_and_joins_the_real_observer(monkeypatch):
    main = threading.current_thread()
    observed = threading.Event()

    def metrics(url):
        if threading.current_thread() is not main:
            observed.set()
        return {**dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0), "orbitkv_pool_used_bytes": 42}

    monkeypatch.setattr("benches.common.metrics.metrics", metrics)
    monkeypatch.setattr("benches.common.metrics.time.sleep", lambda seconds: None)
    with measure("engine", "manager", 0, backend="orbitkv", observation_mode="sampled") as result:
        assert observed.wait(2), "Real metrics observer did not execute"
    validate_observation("sampled", [result], manager_backend="orbitkv")
    assert result["memory_samples"] >= 1
    assert sampled_peak([result], "orbitkv_pool_used_bytes") == 42


@pytest.mark.parametrize("failure", ["http", "drain"])
def test_boundary_mode_still_rejects_snapshot_and_drain_failures(monkeypatch, failure):
    calls = 0

    def metrics(url):
        nonlocal calls
        calls += 1
        if calls > 2 and failure == "http":
            raise RuntimeError("HTTP snapshot failed")
        return {
            **dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
            "orbitkv_query_reserved_bytes": 1 if calls > 2 else 0,
        }

    clock = iter([0, 40])
    monkeypatch.setattr("benches.common.metrics.metrics", metrics)
    monkeypatch.setattr("benches.common.metrics.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("benches.common.metrics.time.sleep", lambda seconds: None)
    with (
        pytest.raises(RuntimeError if failure == "http" else TimeoutError),
        measure("engine", "manager", 0, backend="orbitkv", observation_mode="boundary"),
    ):
        pass


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_mode",
        "mixed_mode",
        "missing_peaks",
        "invented_peak",
        "samples",
        "manager",
        "missing_backend",
        "foreign_backend",
    ],
)
def test_incomplete_or_mixed_observation_evidence_is_rejected(corruption):
    row = {
        "observation_mode": "boundary",
        "manager_present": True,
        "manager_backend": "orbitkv",
        "manager_before": dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
        "manager_after": dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
        "memory_samples": 0,
        "sampled_peak_bytes": None,
    }
    broken = copy.deepcopy(row)
    if corruption == "missing_mode":
        del broken["observation_mode"]
    elif corruption == "mixed_mode":
        broken["observation_mode"] = "sampled"
    elif corruption == "missing_peaks":
        del broken["sampled_peak_bytes"]
    elif corruption == "invented_peak":
        broken["sampled_peak_bytes"] = {"orbitkv_pool_used_bytes": 0}
    elif corruption == "samples":
        broken["memory_samples"] = 1
    elif corruption == "manager":
        broken["manager_present"] = False
    elif corruption == "missing_backend":
        del broken["manager_backend"]
    else:
        broken["manager_backend"] = "lmcache"
    with pytest.raises(ValueError):
        validate_observation("boundary", [row, broken], manager_backend="orbitkv")


@pytest.mark.parametrize("mode", ["sampled", "boundary"])
@pytest.mark.parametrize(
    "corruption", ["empty", "missing_query", "missing_inflight", "nan", "negative", "optional_nan"]
)
def test_live_measurement_rejects_invalid_actual_http_ownership(monkeypatch, mode, corruption):
    gauges = dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0)
    if corruption == "empty":
        gauges.clear()
    elif corruption == "missing_query":
        del gauges["orbitkv_query_reserved_bytes"]
    elif corruption == "missing_inflight":
        del gauges["orbitkv_inflight_bytes"]
    elif corruption == "optional_nan":
        gauges["orbitkv_ssd_cufile_inflight_batches"] = "NaN"
    else:
        gauges["orbitkv_query_reserved_bytes"] = "NaN" if corruption == "nan" else -1
    body = "\n".join(f"{name} {value}" for name, value in gauges.items())
    monkeypatch.setattr(
        "benches.common.metrics.requests.get",
        lambda url, **kw: SimpleNamespace(
            text=body if url == "manager/metrics" else "", raise_for_status=lambda: None
        ),
    )
    with (
        pytest.raises(ValueError, match="OrbitKV ownership"),
        measure("engine", "manager", 0, backend="orbitkv", observation_mode=mode),
    ):
        pytest.fail("Invalid ownership evidence reached the foreground interval")


@pytest.mark.parametrize("mode", ["sampled", "boundary"])
def test_invalid_http_ownership_after_admission_is_rejected(monkeypatch, mode):
    main = threading.current_thread()
    observed = threading.Event()
    manager_calls = 0
    body = "\n".join(f"{name} 0" for name in REQUIRED_OWNERSHIP_GAUGES)

    def get(url, **kwargs):
        nonlocal manager_calls
        text = ""
        if url == "manager/metrics":
            manager_calls += 1
            text = body if manager_calls == 1 else ""
            if threading.current_thread() is not main:
                observed.set()
        return SimpleNamespace(text=text, raise_for_status=lambda: None)

    monkeypatch.setattr("benches.common.metrics.requests.get", get)
    monkeypatch.setattr("benches.common.metrics.time.sleep", lambda seconds: None)
    with (
        pytest.raises(ValueError, match="Missing required OrbitKV ownership"),
        measure("engine", "manager", 0, backend="orbitkv", observation_mode=mode),
    ):
        if mode == "sampled":
            assert observed.wait(2)


@pytest.mark.parametrize("mode", ["sampled", "boundary"])
def test_foreign_manager_snapshots_do_not_claim_orbitkv_ownership(monkeypatch, mode):
    observed = threading.Event()
    main = threading.current_thread()

    def get(*args, **kwargs):
        if threading.current_thread() is not main:
            observed.set()
        return SimpleNamespace(text="lmcache_cache_size 4096\n", raise_for_status=lambda: None)

    monkeypatch.setattr("benches.common.metrics.requests.get", get)
    monkeypatch.setattr("benches.common.metrics.time.sleep", lambda seconds: None)
    with measure("engine", "manager", 0, backend="lmcache", observation_mode=mode) as result:
        if mode == "sampled":
            assert observed.wait(2)
    validate_observation(mode, [result], manager_backend="lmcache")
    assert result["manager_before"] == result["manager_after"] == {"lmcache_cache_size": 4096}
    assert result["sampled_peak_bytes"] == ({} if mode == "sampled" else None)
    with pytest.raises(ValueError, match="observation metadata"):
        validate_observation(mode, [result], manager_backend="orbitkv")


def report_fixture(directory, workload, mode):
    from benches.serving.concurrent import PATTERNS, phases
    from benches.tests.serving.fixtures import evidence

    args, samples, measurements = evidence()
    args.update(backend="orbitkv", workload=workload, observation_mode=mode)
    measurement_file = "windows.jsonl"
    if workload == "serial":
        args.update(lengths=[64], repeats=1)
        samples = [
            {
                "length": 64,
                "repeat": 0,
                "phase": phase,
                "ttft_ms": 10,
                "e2e_ms": 20,
                "matches_cold_output": True,
                "usage": {"completion_tokens": 2},
                "metrics_delta": {},
                "manager_delta": {},
            }
            for phase in workload_phases()
        ]
        measurements = samples
        measurement_file = "samples.jsonl"
    elif workload == "concurrent":
        args.update(concurrencies=[1], repeats=1)
        measurements = [
            {
                "concurrency": 1,
                "pattern": pattern,
                "repeat": 0,
                "phase": phase,
                "wall_seconds": 0.02,
                "manager_delta": {},
            }
            for pattern in PATTERNS
            for phase in phases(False)
        ]
        samples = [
            {
                **row,
                "index": 0,
                "ttft_ms": 10,
                "e2e_ms": 20,
                "cached_tokens": 0,
                "matches_cold_output": True,
                "usage": {"completion_tokens": 2},
            }
            for row in measurements
        ]
        measurement_file = "batches.jsonl"
    for row in measurements:
        row.update(
            observation_mode=mode,
            manager_present=True,
            manager_backend="orbitkv",
            manager_before=dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
            manager_after=dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0),
            memory_samples=1 if mode == "sampled" else 0,
            sampled_peak_bytes={"orbitkv_query_reserved_bytes": 0, "orbitkv_inflight_bytes": 0}
            if mode == "sampled"
            else None,
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": args,
                "observation_mode": mode,
                "manager_present": True,
                "manager_backend": "orbitkv",
            }
        )
    )
    (directory / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    (directory / measurement_file).write_text("\n".join(map(json.dumps, measurements)))
    return measurement_file, measurements


@pytest.mark.parametrize("workload", ["serial", "concurrent", "sustained"])
@pytest.mark.parametrize("mode", ["sampled", "boundary"])
@pytest.mark.parametrize(
    "corruption", ["missing_before", "missing_after", "empty", "nonzero", "nan", "optional_nonzero"]
)
def test_offline_report_rejects_ownership_corruption(tmp_path, workload, mode, corruption):
    measurement_file, measurements = report_fixture(tmp_path, workload, mode)
    assert collect_run(tmp_path)["summary"], "Complete control fixture must be accepted"
    row = measurements[0]
    if corruption.startswith("missing_"):
        del row[f"manager_{corruption.removeprefix('missing_')}"]
    elif corruption == "empty":
        row["manager_before"] = {}
    elif corruption == "nonzero":
        row["manager_after"]["orbitkv_inflight_bytes"] = 4096
    elif corruption == "optional_nonzero":
        row["manager_after"]["orbitkv_transfer_completion_outstanding"] = 1
    else:
        row["manager_after"]["orbitkv_query_reserved_bytes"] = float("nan")
    (tmp_path / measurement_file).write_text("\n".join(map(json.dumps, measurements)))
    with pytest.raises(ValueError, match="ownership|boundary snapshot"):
        collect_run(tmp_path)


def test_report_requires_manifest_and_sample_profiling_to_match(tmp_path):
    sample_file, samples = report_fixture(tmp_path, "serial", "boundary")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arguments"].update(engine="sglang", sglang_profile=True)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="profiling"):
        collect_run(tmp_path)
    for sample in samples:
        sample["profiling"] = {
            "mode": "sglang-public-cpu-cuda" if sample["phase"] == "after_pressure" else "off"
        }
    (tmp_path / sample_file).write_text("\n".join(map(json.dumps, samples)))
    summaries = collect_run(tmp_path)["summary"]
    assert (
        next(row for row in summaries if row["phase"] == "after_pressure")["profiling_mode"]
        == "sglang-public-cpu-cuda"
    )
    manifest["arguments"]["sglang_profile"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="profiling"):
        collect_run(tmp_path)


def test_boundary_codec_snapshots_cannot_be_reported_as_sampled_peaks():
    row = {
        "manager_before": {"orbitkv_storage_codec_workspace_bytes": 17},
        "manager_after": {"orbitkv_storage_codec_workspace_bytes": 42},
        "sampled_peak_bytes": None,
    }
    summary = codec_summary([row])
    assert summary["sampled_peak_codec_workspace_bytes"] is None
    assert summary["boundary_max_codec_workspace_bytes"] == 42


def test_boundary_snapshot_budget_violation_remains_a_failure():
    from benches.serving.sustained import validate
    from benches.tests.serving.fixtures import evidence

    args, samples, windows = evidence()
    args["query_budget_gib"] = 1
    windows[0]["manager_after"]["orbitkv_query_reserved_peak_bytes"] = 1024**3 + 1
    with pytest.raises(ValueError, match="total byte budget"):
        validate(args, samples, windows)
