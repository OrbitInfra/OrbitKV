"""Incomplete native driver captures cannot authorize attribution or promotion."""

import copy
import json

import pytest

from benches.cuda_submission import analyze


@pytest.fixture
def capture(tmp_path):
    trace = [
        {
            "stage": "cuda_driver_open",
            "schema": 1,
            "pid": 100,
            "capacity": 262144,
            "clock": "CLOCK_MONOTONIC",
        }
    ]
    fixture = []
    begin = 1000
    for direction in ("h2d", "d2h"):
        row = {
            "schema": 1,
            "native_pid": 100,
            "cuda_device_uuid": "GPU-fixture",
            "case": "d64-r64",
            "direction": direction,
            "byte_source_guard_oracle": "passed before each warmup and after final submission",
            "samples": 2,
            "bytes": 4194304,
            "descriptors": 64,
            "dma_ranges": 64,
        }
        for backend in ("kernel", "dma"):
            row[backend] = {}
            for mode in ("event_off", "event_on"):
                windows = []
                for _ in range(2):
                    windows.append(
                        {
                            "begin_ns": begin,
                            "submitted_ns": begin + 1000,
                            "drained_ns": begin + 2000,
                            "tid": 101,
                        }
                    )
                    names = (
                        ["cuMemcpyHtoDAsync_v2", "cuLaunchKernel"]
                        if backend == "kernel"
                        else ["cuMemcpyHtoDAsync_v2"] * 64
                    )
                    for index, name in enumerate(names):
                        trace.append(
                            {
                                "stage": "cuda_driver_api",
                                "id": len(trace) - 1,
                                "name": name,
                                "begin_ns": begin + 10 + index * 10,
                                "end_ns": begin + 15 + index * 10,
                                "tid": 101,
                                "context": 1,
                                "correlation": len(trace),
                                "result": 0,
                            }
                        )
                    # An overlapping driver interval must not be counted twice.
                    trace.append(
                        {
                            "stage": "cuda_driver_api",
                            "id": len(trace) - 1,
                            "name": "cuCtxGetCurrent",
                            "begin_ns": begin + 11,
                            "end_ns": begin + 14,
                            "tid": 101,
                            "context": 1,
                            "correlation": len(trace),
                            "result": 0,
                        }
                    )
                    begin += 10000
                row[backend][mode] = {
                    "submission_windows": windows,
                    "call_to_return_ms": [2, 2],
                    "cpu_submission_ms": [1, 1],
                    "host_wait_ms": [1, 1],
                }
        fixture.append(row)
    trace.append(
        {"stage": "cuda_driver_complete", "pid": 100, "calls": len(trace) - 1, "complete": True}
    )
    trace_path, fixture_path = tmp_path / "trace.jsonl", tmp_path / "fixture.jsonl"

    def write():
        trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
        fixture_path.write_text("".join(json.dumps(row) + "\n" for row in fixture))
        return trace_path, fixture_path

    return trace, fixture, write


def test_native_driver_union_is_not_packing_or_gpu_time(capture):
    _, _, write = capture
    result = analyze(*write())
    assert len(result["groups"]) == 8
    sample = result["groups"][0]["samples"][0]
    assert sample["driver_api_union_wall_ns"] == 10
    assert sample["unattributed_host_wall_ns"] == 990
    assert "not proven packing" in result["scope"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_footer", "complete"),
        ("overflow", "capacity"),
        ("missing_calls", "count"),
        ("wrong_pid", "PID"),
        ("wrong_tid", "No driver"),
        ("wrong_count", "geometry"),
        ("copy_error", "failure"),
        ("cross_boundary", "boundary"),
        ("missing_windows", "windows"),
        ("no_oracle", "guard"),
        ("missing_direction", "directional"),
        ("overlap", "Overlapping"),
    ],
)
def test_invalid_capture_fails_closed(capture, mutation, match):
    trace, fixture, write = capture
    if mutation == "missing_footer":
        trace.pop()
    elif mutation == "overflow":
        trace[0]["capacity"] = 1
    elif mutation == "missing_calls":
        trace[-1]["calls"] += 1
    elif mutation == "wrong_pid":
        fixture[0]["native_pid"] = 999
    elif mutation == "wrong_tid":
        fixture[0]["kernel"]["event_off"]["submission_windows"][0]["tid"] = 999
    elif mutation == "wrong_count":
        fixture[0]["dma_ranges"] = 63
    elif mutation == "copy_error":
        trace[1]["result"] = 1
    elif mutation == "cross_boundary":
        trace[1]["begin_ns"] -= 100
    elif mutation == "missing_windows":
        fixture[0]["kernel"]["event_off"].pop("submission_windows")
    elif mutation == "no_oracle":
        fixture[0].pop("byte_source_guard_oracle")
    elif mutation == "missing_direction":
        fixture.pop()
    elif mutation == "overlap":
        clocks = fixture[0]["kernel"]["event_off"]["submission_windows"]
        clocks[1] = copy.deepcopy(clocks[0])
    with pytest.raises(ValueError, match=match):
        analyze(*write())
