"""Shared deployment evidence must preserve each instance's work and ownership."""

import copy
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from benches.serving import shared_manager_pressure as pressure


def args():
    return SimpleNamespace(
        model="model",
        seed="case",
        duration_seconds=1,
        progress_gap_seconds=1,
        max_requests=9,
        concurrency=1,
        working_set=2,
        cold_every=4,
        prompt_tokens=64,
        output_tokens=2,
        query_mib=384,
        instance_query_mib=192,
        host_mib=512,
    )


def test_two_instance_quotas_and_all_admitted_requests_drain(monkeypatch):
    active = dict.fromkeys(pressure.ENGINES, 0)
    peak = active.copy()
    completed = active.copy()
    lock = threading.Lock()

    def generate(url, engine, model, tokens, output):
        with lock:
            active[engine] += 1
            peak[engine] = max(peak[engine], active[engine])
        time.sleep(0.002 if engine == "vllm" else 0.01)
        with lock:
            active[engine] -= 1
            completed[engine] += 1
        return {
            "text": "native",
            "ttft_ms": 1,
            "e2e_ms": 2,
            "usage": {"prompt_tokens": len(tokens), "completion_tokens": output},
        }

    monkeypatch.setattr(pressure, "generate", generate)
    prefixes = {
        engine: [
            {
                "tokens": [1] * 64,
                "text": "native",
                "usage": {"prompt_tokens": 64, "completion_tokens": 2},
            }
        ]
        * 2
        for engine in pressure.ENGINES
    }
    emitted = []
    rows, window = pressure.run_window(
        args(), dict.fromkeys(pressure.ENGINES, "url"), [1, 2], prefixes, emitted.append
    )
    pressure.validate_window(args(), rows, window)
    assert peak == dict.fromkeys(pressure.ENGINES, 1)
    assert active == dict.fromkeys(pressure.ENGINES, 0)
    assert completed == dict.fromkeys(pressure.ENGINES, 9) == window["submitted"]
    assert len(rows) == len(emitted) == 18
    assert window["stop_reason"] == "request_limit"
    assert sum(row["kind"] == "cold" for row in rows) == 4
    assert {row["input_seed"] for row in rows if row["kind"] == "cold"} == {
        "case:vllm:cold:0",
        "case:vllm:cold:1",
        "case:sglang:cold:0",
        "case:sglang:cold:1",
    }


def evidence():
    row = {
        "index": 0,
        "submitted_seconds": 0,
        "started_seconds": 0.001,
        "finished_seconds": 0.1,
        "ttft_ms": 10,
        "e2e_ms": 50,
        "usage": {"prompt_tokens": 64, "completion_tokens": 2},
        "matches_reference_output": True,
    }
    rows = [{**row, "engine": engine} for engine in pressure.ENGINES]
    window = {
        "submitted": dict.fromkeys(pressure.ENGINES, 1),
        "peak_client_inflight": dict.fromkeys(pressure.ENGINES, 1),
        "wall_seconds": 1.1,
        "stop_reason": "duration",
        "failures": [],
    }
    return rows, window


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "missing",
        "oracle",
        "pending_oracle",
        "token_count",
        "timing",
        "nan",
        "gap",
        "admission",
        "early",
        "failure",
        "extra",
    ],
)
def test_incomplete_or_unsafe_evidence_rejected(corruption):
    rows, window = copy.deepcopy(evidence())
    if corruption == "duplicate":
        rows.append(rows[0].copy())
    elif corruption == "missing":
        rows.pop()
    elif corruption in ("oracle", "pending_oracle"):
        rows[0]["matches_reference_output"] = False if corruption == "oracle" else None
    elif corruption == "token_count":
        rows[0]["usage"]["completion_tokens"] = 1
    elif corruption == "timing":
        rows[0]["submitted_seconds"] = 0.2
    elif corruption == "nan":
        rows[0]["e2e_ms"] = float("nan")
    elif corruption == "gap":
        rows[0]["finished_seconds"] = 1.05
    elif corruption == "admission":
        window["peak_client_inflight"]["vllm"] = 2
    elif corruption == "early":
        window["wall_seconds"] = 0.5
    elif corruption == "failure":
        window["failures"].append({"error": "timeout"})
    elif corruption == "extra":
        rows.append({**rows[0], "engine": "unknown"})
    with pytest.raises(ValueError):
        pressure.validate_window(args(), rows, window, require_oracle=True)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "budget",
        "leak",
        "missing_drain",
        "staging_leak",
        "staging_missing",
        "staging_nonfinite",
        "no_ssd",
        "missing_peak",
        "instance_peak",
        "owner_peak",
        "missing_owner",
        "reset_owner",
        "unsampled_peak",
        "inconsistent_current",
    ],
)
def test_real_route_evidence_and_resource_bounds(corruption):
    after = dict.fromkeys(pressure.DRAIN, 0)
    after.update(
        {
            "orbitkv_save_bytes_total": 4096,
            "orbitkv_load_bytes_total": 4096,
            "orbitkv_ssd_write_bytes_total": 4096,
            "orbitkv_ssd_prefetch_bytes_total": 4096,
        }
    )
    before = dict.fromkeys(pressure.RESERVATION_PEAKS, 0)
    after.update(dict.fromkeys(pressure.RESERVATION_PEAKS, 4096))
    peaks = {"orbitkv_query_reserved_bytes": 4096, "orbitkv_pool_used_bytes": 4096}
    if corruption == "instance_peak":
        after[pressure.RESERVATION_PEAKS[1]] = 193 * 1024**2
    elif corruption == "owner_peak":
        after[pressure.RESERVATION_PEAKS[0]] = 385 * 1024**2
    elif corruption == "missing_owner":
        after.pop(pressure.RESERVATION_PEAKS[1])
    elif corruption == "reset_owner":
        before[pressure.RESERVATION_PEAKS[0]] = 8192
    elif corruption == "unsampled_peak":
        peaks[pressure.RESERVATION_PEAKS[0]] = 8192
    elif corruption == "inconsistent_current":
        peaks["orbitkv_query_reserved_bytes"] = 8192
    elif corruption == "budget":
        peaks["orbitkv_query_reserved_bytes"] = 385 * 1024**2
    elif corruption == "leak":
        after["orbitkv_ssd_read_pinned_bytes"] = 4096
    elif corruption == "missing_drain":
        after.pop("orbitkv_ssd_read_pinned_bytes")
    elif corruption == "staging_leak":
        after["orbitkv_ssd_gpu_staging_bytes"] = 4096
    elif corruption == "staging_missing":
        peaks["orbitkv_ssd_gpu_staging_bytes"] = 4096
    elif corruption == "staging_nonfinite":
        after["orbitkv_ssd_gpu_staging_bytes"] = float("nan")
    elif corruption == "no_ssd":
        after["orbitkv_ssd_prefetch_bytes_total"] = 0
    elif corruption == "missing_peak":
        peaks.pop("orbitkv_pool_used_bytes")
    if corruption:
        with pytest.raises(ValueError):
            pressure.validate_resources(args(), before, after, peaks)
    else:
        assert (
            pressure.validate_resources(args(), before, after, peaks)[
                "orbitkv_ssd_prefetch_bytes_total"
            ]
            == 4096
        )


def test_failure_stops_admission_but_preserves_other_instance_completion(monkeypatch):
    entered = threading.Barrier(2)

    def generate(url, engine, model, tokens, output):
        entered.wait(timeout=2)
        if engine == "vllm":
            raise RuntimeError("HTTP failure")
        time.sleep(0.02)
        return {
            "text": "native",
            "ttft_ms": 1,
            "e2e_ms": 20,
            "usage": {"prompt_tokens": len(tokens), "completion_tokens": output},
        }

    monkeypatch.setattr(pressure, "generate", generate)
    prefixes = {
        engine: [
            {
                "tokens": [1] * 64,
                "text": "native",
                "usage": {"prompt_tokens": 64, "completion_tokens": 2},
            }
        ]
        * 2
        for engine in pressure.ENGINES
    }
    emitted = []
    rows, window = pressure.run_window(
        args(), dict.fromkeys(pressure.ENGINES, "url"), [1], prefixes, emitted.append
    )
    assert window["submitted"] == dict.fromkeys(pressure.ENGINES, 1)
    assert window["failures"][0]["error"] == "HTTP failure"
    assert len(rows) == len(emitted) == 1
    assert rows[0]["engine"] == "sglang"
    with pytest.raises(ValueError, match="Pressure request failed"):
        pressure.validate_window(args(), rows, window)


@pytest.mark.parametrize("field", ["text", "prompt_tokens", "completion_tokens"])
def test_native_oracle_requires_matching_text_and_both_token_counts(field):
    result = {"text": "native", "usage": {"prompt_tokens": 64, "completion_tokens": 2}}
    assert pressure.matches_native(result, copy.deepcopy(result))
    reference = copy.deepcopy(result)
    if field == "text":
        reference["text"] = "different"
    else:
        reference["usage"][field] += 1
    assert not pressure.matches_native(result, reference)


@pytest.mark.parametrize("foreground_failure", [False, True])
def test_real_sampler_failure_is_strict_and_preserves_foreground_error(
    tmp_path, foreground_failure
):
    class UnavailableMetrics(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, format, *values):
            pass

    server = HTTPServer(("127.0.0.1", 0), UnavailableMetrics)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    options = args()
    options.output = tmp_path
    options.sample_interval = 0.01
    expected = ValueError if foreground_failure else RuntimeError
    try:
        with (
            pytest.raises(expected, match="foreground failed" if foreground_failure else "sampler"),
            pressure.resource_sampler(
                options, f"http://127.0.0.1:{server.server_port}", dict(os.environ)
            ),
        ):
            deadline = time.monotonic() + 10
            while not (tmp_path / "resources.summary.json").exists():
                assert time.monotonic() < deadline, "Sampler did not finish its failed scrape"
                time.sleep(0.01)
            if foreground_failure:
                raise ValueError("foreground failed")
        cleanup = json.loads((tmp_path / "sampler-cleanup.json").read_text())
        assert cleanup["exit_code"] == 1 and not cleanup["forced_kill"]
        assert json.loads((tmp_path / "resources.summary.json").read_text())["errors"]
        if foreground_failure:
            assert cleanup["primary_failure"] == {
                "type": "ValueError",
                "message": "foreground failed",
            }
        else:
            assert cleanup["primary_failure"] is None
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()
