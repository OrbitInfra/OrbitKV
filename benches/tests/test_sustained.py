"""Keep sustained measurements bounded and reject incomplete performance evidence."""

import copy
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benches import sustained
from benches.report import collect_run


@pytest.mark.parametrize("request_limit", [5, 100])
def test_window_bounds_inflight_and_drains_admitted_requests(monkeypatch, request_limit):
    active = 0
    peak = 0
    completed = 0
    lock = threading.Lock()

    def generate(*args):
        nonlocal active, peak, completed
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.012)
        with lock:
            active -= 1
            completed += 1
        return {
            "text": "reference",
            "ttft_ms": 1,
            "e2e_ms": 12,
            "usage": {"prompt_tokens": len(args[3]), "completion_tokens": 2},
        }

    @contextmanager
    def measure(*args, backend, observation_mode):
        result = {"observation_mode": observation_mode}
        started = time.perf_counter()
        yield result
        result.update(
            wall_seconds=time.perf_counter() - started,
            drain_seconds=0,
            manager_after={},
            manager_delta={},
            sampled_peak_bytes={},
        )

    monkeypatch.setattr(sustained, "generate", generate)
    monkeypatch.setattr(sustained, "measure", measure)
    args = SimpleNamespace(
        engine="vllm",
        backend="native",
        model="model",
        duration_seconds=1 if request_limit == 5 else 0.045,
        max_requests=request_limit,
        settle_seconds=0,
        observation_mode="sampled",
        seed=42,
        reuse_ratio=0.75,
        lengths=[64],
        output_tokens=2,
        working_set=1,
        concurrencies=[2],
    )
    emitted = []
    samples, window = sustained.run_window(
        args,
        "engine",
        None,
        [1, 2, 3],
        [{"tokens": [1] * 64, "text": "reference"}],
        2,
        emitted.append,
    )
    sustained.validate(vars(args), samples, [window])
    assert 0 < peak <= 2
    assert active == 0
    assert completed == len(samples) == len(emitted) == window["submitted_requests"]
    assert window["wall_seconds"] >= max(s["finished_seconds"] for s in samples)
    assert window["stop_reason"] == ("request_limit" if request_limit == 5 else "duration")
    assert window["submitted_requests"] <= request_limit


def evidence():
    args = {
        "engine": "vllm",
        "backend": "native",
        "observation_mode": "boundary",
        "workload": "sustained",
        "concurrencies": [2],
        "max_requests": 100,
        "duration_seconds": 1,
        "working_set": 1,
    }
    rows = [
        {
            "concurrency": 2,
            "index": i,
            "kind": "reuse",
            "prefix_index": 0,
            "submitted_seconds": 0.99,
            "started_seconds": 0.99,
            "finished_seconds": 1.1,
            "ttft_ms": 10,
            "e2e_ms": 110,
            "usage": {"completion_tokens": 2},
            "cached_tokens": 64,
            "matches_reference_output": i == 0,
        }
        for i in range(2)
    ]
    windows = [
        {
            "concurrency": 2,
            "submitted_requests": 2,
            "peak_client_inflight": 2,
            "stop_reason": "duration",
            "wall_seconds": 1.1,
            "drain_seconds": 0.2,
            "working_set_tokens": 64,
            "manager_after": {},
            "manager_before": {},
            "sampled_peak_bytes": None,
            "observation_mode": "boundary",
            "manager_present": False,
            "manager_backend": None,
            "memory_samples": 0,
            "manager_delta": {"orbitkv_ssd_prefetch_bytes_total": 4096},
        }
    ]
    return args, rows, windows


def test_report_counts_window_bytes_once_and_preserves_output_differences(tmp_path):
    args, samples, windows = evidence()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "arguments": args,
                "observation_mode": "boundary",
                "manager_present": False,
                "manager_backend": None,
            }
        )
    )
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, samples)))
    (tmp_path / "windows.jsonl").write_text("\n".join(map(json.dumps, windows)))
    (row,) = collect_run(tmp_path)["summary"]
    assert row["orbitkv_ssd_prefetch_bytes_total"] == 4096
    assert row["output_mismatches"] == 1
    assert row["reuse_ttft_p95_ms"] == 10
    assert row["cold_ttft_p95_ms"] is None
    assert row["requests_per_second"] == pytest.approx(2 / 1.1)
    assert row["cache_sources"] == {"cached_tier_unknown": 2, "miss": 0}
    (tmp_path / "failure.json").write_text('{"message":"request failed"}')
    with pytest.raises(ValueError, match="Failed run"):
        collect_run(tmp_path)


def test_codec_batch_counters_are_window_totals_not_per_request():
    _, samples, windows = evidence()
    windows[0]["manager_delta"].update(
        {
            "orbitkv_storage_codec_bytes_total_logical": 8192,
            "orbitkv_storage_codec_bytes_total_stored": 4096,
            "orbitkv_storage_codec_batches_total_encode": 3,
            "orbitkv_storage_codec_batch_segments_sum_encode": 32,
            "orbitkv_storage_codec_batch_segments_count_encode": 3,
        }
    )
    (row,) = sustained.summarize(samples, windows)
    assert row["n"] == 2
    assert row["orbitkv_storage_codec_batches_total_encode"] == 3
    assert row["orbitkv_storage_codec_batch_segments_sum_encode"] == 32
    assert row["encoded_publication_stored_fraction"] == 0.5
    assert row["output_mismatches"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "early",
        "leak",
        "budget",
        "speculative_budget",
        "protected_budget",
        "timing",
        "nan",
        "prefix",
        "concurrency",
    ],
)
def test_incomplete_or_invalid_windows_cannot_be_reported(corruption):
    args, samples, windows = copy.deepcopy(evidence())
    if corruption == "missing":
        samples.pop()
    elif corruption == "duplicate":
        samples.append(samples[0])
    elif corruption == "early":
        windows[0]["wall_seconds"] = 0.5
    elif corruption == "leak":
        windows[0]["manager_after"]["orbitkv_query_reserved_bytes"] = 4096
    elif corruption == "protected_budget":
        windows[0]["sampled_peak_bytes"] = {}
        args.update(host_gib=4, cache_protected_percent=80)
        limit = 4 * 1024**3 * 80 // 100
        windows[0]["sampled_peak_bytes"]["orbitkv_cache_protected_bytes"] = limit
        sustained.validate(args, samples, windows)
        windows[0]["sampled_peak_bytes"]["orbitkv_cache_protected_bytes"] += 1
    elif corruption in ("budget", "speculative_budget"):
        windows[0]["sampled_peak_bytes"] = {}
        args["query_budget_gib"] = 3
        name, limit = (
            ("orbitkv_query_reserved_bytes", 3 * 1024**3)
            if corruption == "budget"
            else ("orbitkv_query_speculative_reserved_bytes", 3 * 1024**3 // 4)
        )
        windows[0]["sampled_peak_bytes"][name] = limit
        sustained.validate(args, samples, windows)
        windows[0]["sampled_peak_bytes"][name] += 1
    elif corruption == "timing":
        samples[0]["submitted_seconds"] = 1.01
    elif corruption == "nan":
        samples[0]["e2e_ms"] = float("nan")
    elif corruption == "prefix":
        samples[0]["prefix_index"] = 1
    elif corruption == "concurrency":
        windows[0]["peak_client_inflight"] = 3
    with pytest.raises(ValueError):
        sustained.validate(args, samples, windows)
