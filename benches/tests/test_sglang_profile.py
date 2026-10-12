"""Profiler cleanup and missing-CUDA evidence must fail diagnostic admission."""

import gzip
import json
from types import SimpleNamespace

import pytest
import requests

from benches.metrics import summarize
from benches.sglang_profile import sglang_profile


@pytest.fixture
def profiler_http(monkeypatch, tmp_path):
    calls = []
    output = tmp_path / "trace"
    state = {"start_error": None, "stop_error": None, "trace": {"traceEvents": [{"cat": "kernel"}]}}

    def post(url, **kwargs):
        calls.append(url)
        error = state["start_error" if url.endswith("start_profile") else "stop_error"]
        if url.endswith("stop_profile") and state["trace"] is not None:
            with gzip.open(output / "restore-TP-0.trace.json.gz", "wt") as target:
                json.dump(state["trace"], target)

        def raise_for_status():
            if error is not None:
                raise error

        return SimpleNamespace(
            status_code=500 if error else 200,
            text="Profiler response",
            raise_for_status=raise_for_status,
        )

    monkeypatch.setattr("benches.sglang_profile.requests.post", post)
    return output, calls, state


def test_success_requires_gpu_trace_and_retains_receipts(profiler_http):
    output, calls, _ = profiler_http
    with sglang_profile("http://engine", output) as control:
        pass
    assert calls == ["http://engine/start_profile", "http://engine/stop_profile"]
    assert control["traces"][0]["GPU_kernels"] == 1
    assert len(control["traces"][0]["sha256"]) == 64
    assert json.loads((output / "CONTROL.json").read_text()) == control


@pytest.mark.parametrize("trace", [None, {"traceEvents": [{"cat": "cpu_op"}]}])
def test_missing_or_cpu_only_trace_invalidates_cell(profiler_http, trace):
    output, calls, state = profiler_http
    state["trace"] = trace
    with pytest.raises(ValueError, match="trace|GPU"), sglang_profile("http://engine", output):
        pass
    assert len(calls) == 2
    assert "stop_or_trace_error" in json.loads((output / "CONTROL.json").read_text())


def test_partial_start_failure_still_stops_profiler(profiler_http):
    output, calls, state = profiler_http
    state["start_error"] = requests.HTTPError("partial start")
    with (
        pytest.raises(requests.HTTPError, match="partial start"),
        sglang_profile("http://engine", output),
    ):
        pytest.fail("Failed profiler start cannot admit workload")
    assert calls[-1].endswith("stop_profile")
    assert "workload_error" in json.loads((output / "CONTROL.json").read_text())


def test_stop_failure_preserves_foreground_failure(profiler_http):
    output, _, state = profiler_http
    state["stop_error"] = requests.HTTPError("stop failure")
    with (
        pytest.raises(RuntimeError, match="foreground failure"),
        sglang_profile("http://engine", output),
    ):
        raise RuntimeError("foreground failure")
    control = json.loads((output / "CONTROL.json").read_text())
    assert "foreground failure" in control["workload_error"]
    assert "stop failure" in control["stop_or_trace_error"]


def test_summary_preserves_diagnostic_identity_and_rejects_mixed_modes():
    sample = {
        "length": 8192,
        "phase": "after_pressure",
        "ttft_ms": 1,
        "e2e_ms": 2,
        "matches_cold_output": True,
        "cache_source": "external",
        "manager_delta": {},
        "profiling": {"mode": "sglang-public-cpu-cuda"},
    }
    assert summarize([sample], [8192])[0]["profiling_mode"] == "sglang-public-cpu-cuda"
    with pytest.raises(ValueError, match="profiling"):
        summarize([sample, {**sample, "profiling": {"mode": "off"}}], [8192])
