"""Incomplete native driver captures cannot authorize attribution or promotion."""

import copy
import json

import pytest

from benches.tests.transfer.fixtures import log, records
from benches.transfer.cuda_submission import analyze, analyze_engine
from benches.transfer.local_restore_shapes import _collect


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
                        else [
                            "cuMemcpyHtoDAsync_v2" if direction == "h2d" else "cuMemcpyDtoHAsync_v2"
                        ]
                        * 64
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


@pytest.mark.parametrize("backend", ["direct", "kernel"])
@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "multipart",
        "late_clock",
        "pid",
        "clock",
        "thread",
        "window",
        "drain",
        "missing_copy",
        "wrong_copy",
        "generic_copy",
        "error",
        "changed_log",
    ],
)
def test_consumed_engine_driver_windows_require_complete_matching_inputs(
    tmp_path, monkeypatch, backend, mutation
):
    events = records()
    events[0]["clock"] = "CLOCK_MONOTONIC"
    events[1].update(backend=backend, tid=101, begin_ns=1000, submitted_ns=2000)
    events[2]["drained_ns"] = 3000
    names = ["cuMemcpyHtoDAsync_v2"]
    if backend == "kernel":
        names.append("cuLaunchKernel")
    driver = [
        {
            "stage": "cuda_driver_open",
            "schema": 1,
            "pid": 17,
            "capacity": 262144,
            "clock": "CLOCK_MONOTONIC",
        }
    ]
    for index, name in enumerate(names):
        driver.append(
            {
                "stage": "cuda_driver_api",
                "id": index,
                "name": name,
                "tid": 101,
                "begin_ns": 1010 + index * 20,
                "end_ns": 1020 + index * 20,
                "context": 1,
                "correlation": index,
                "result": 0,
            }
        )
    if mutation in {"multipart", "late_clock"}:
        events[2]["final_part"] = False
        second = copy.deepcopy(events)
        for event in second:
            event["sequence"] = 1
        second[0]["records_before"] = 3
        second[1].update(part_id=1, begin_ns=5000, submitted_ns=6000)
        second[2].update(part_id=1, drained_ns=7000, final_part=True)
        second[-1].update(parts=2, batches=2, descriptors=2, records_reserved=6)
        if mutation == "late_clock":
            second[0]["clock"] = "CLOCK_REALTIME"
        events.extend(second)
        for api in copy.deepcopy(driver[1:]):
            api.update(
                id=len(driver) - 1, begin_ns=api["begin_ns"] + 4000, end_ns=api["end_ns"] + 4000
            )
            driver.append(api)
    elif mutation == "pid":
        driver[0]["pid"] = 999
    elif mutation == "clock":
        events[0]["clock"] = "CLOCK_REALTIME"
    elif mutation == "thread":
        events[1]["tid"] = 999
    elif mutation == "window":
        events[1]["begin_ns"] = 1015
    elif mutation == "drain":
        events[2]["drained_ns"] = 1500
    elif mutation == "missing_copy":
        driver[1]["name"] = "cuCtxGetCurrent"
    elif mutation == "wrong_copy":
        driver[1]["name"] = "cuMemcpyDtoHAsync_v2"
    elif mutation == "generic_copy":
        driver[1]["name"] = "cuMemcpyAsync"
    elif mutation == "error":
        driver[1]["result"] = 1
    driver.append(
        {
            "stage": "cuda_driver_complete",
            "pid": driver[0]["pid"],
            "calls": len(driver) - 1,
            "complete": True,
        }
    )
    trace = tmp_path / "driver.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in driver))
    engine_log = log(tmp_path, events)

    def compiler_control(path, fixture):
        assert path == engine_log and fixture == tmp_path / "native-fixture"
        result = _collect(path) | {"native_geometry_validation": {"scope": "unit control"}}
        if mutation == "changed_log":
            with path.open("a") as output:
                output.write("input mutated after compilation\n")
        return result

    monkeypatch.setattr("benches.transfer.local_restore_shapes.collect", compiler_control)
    if mutation not in {None, "multipart"}:
        with pytest.raises(ValueError):
            analyze_engine(trace, engine_log, tmp_path / "native-fixture")
    else:
        result = analyze_engine(trace, engine_log, tmp_path / "native-fixture")
        batch = result["batches"][0]
        assert batch["driver_api_union_wall_ns"] == 10 * len(names)
        assert batch["unattributed_host_wall_ns"] == 1000 - 10 * len(names)
        assert batch["drained_ns"] == 3000
        assert result["native_geometry_validation"]["scope"] == "unit control"
        assert len(result["batches"]) == (2 if mutation == "multipart" else 1)
