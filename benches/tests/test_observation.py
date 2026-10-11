"""Keep request timing free of polling without weakening boundary checks."""

import copy
import threading

import pytest

from benches.metrics import codec_summary, measure, sampled_peak, validate_observation


@pytest.mark.parametrize("manager", [None, "manager"])
def test_boundary_mode_has_no_request_time_observer(monkeypatch, manager):
    foreground = False
    calls = []

    def metrics(url):
        assert not foreground, "Metrics were polled inside the request interval"
        calls.append(url)
        return {"orbitkv_query_reserved_bytes": 0} if url else {}

    def thread(*args, **kwargs):
        pytest.fail("Boundary mode must not construct an observer thread")

    monkeypatch.setattr("benches.metrics.metrics", metrics)
    monkeypatch.setattr("benches.metrics.threading.Thread", thread)
    monkeypatch.setattr("benches.metrics.time.sleep", lambda seconds: None)
    with measure("engine", manager, 0, observation_mode="boundary") as result:
        foreground = True
        foreground = False
    validate_observation("boundary", [result], manager_present=manager is not None)
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
        return {"orbitkv_pool_used_bytes": 42, "orbitkv_query_reserved_bytes": 0}

    monkeypatch.setattr("benches.metrics.metrics", metrics)
    monkeypatch.setattr("benches.metrics.time.sleep", lambda seconds: None)
    with measure("engine", "manager", 0, observation_mode="sampled") as result:
        assert observed.wait(2), "Real metrics observer did not execute"
    validate_observation("sampled", [result], manager_present=True)
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
        return {"orbitkv_query_reserved_bytes": 1 if calls > 2 else 0}

    clock = iter([0, 40])
    monkeypatch.setattr("benches.metrics.metrics", metrics)
    monkeypatch.setattr("benches.metrics.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("benches.metrics.time.sleep", lambda seconds: None)
    with (
        pytest.raises(RuntimeError if failure == "http" else TimeoutError),
        measure("engine", "manager", 0, observation_mode="boundary"),
    ):
        pass


@pytest.mark.parametrize(
    "corruption",
    ["missing_mode", "mixed_mode", "missing_peaks", "invented_peak", "samples", "manager"],
)
def test_incomplete_or_mixed_observation_evidence_is_rejected(corruption):
    row = {
        "observation_mode": "boundary",
        "manager_present": True,
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
    else:
        broken["manager_present"] = False
    with pytest.raises(ValueError):
        validate_observation("boundary", [row, broken], manager_present=True)


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
    from benches.sustained import validate
    from benches.tests.test_sustained import evidence

    args, samples, windows = evidence()
    args["query_budget_gib"] = 1
    windows[0]["manager_after"]["orbitkv_query_reserved_peak_bytes"] = 1024**3 + 1
    with pytest.raises(ValueError, match="total byte budget"):
        validate(args, samples, windows)
