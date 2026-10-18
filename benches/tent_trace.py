"""Validate pinned native TENT traces without adding overlapping call times."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .artifacts import external_path
from .shared_cache_profile import validate_query_window

STAGES = frozenset({"metadata_rpc", "endpoint_construct", "endpoint_connect", "bootstrap_rpc"})


def covered_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total


def stage_summary(events: list[dict], window: tuple[int, int] | None = None) -> dict:
    result = {}
    for stage in sorted(STAGES):
        intervals = [
            (row["begin_mono_ns"], row["end_mono_ns"]) for row in events if row["stage"] == stage
        ]
        if window is not None:
            intervals = [(max(begin, window[0]), min(end, window[1])) for begin, end in intervals]
            intervals = [(begin, end) for begin, end in intervals if begin < end]
        durations = [(end - begin) / 1e6 for begin, end in intervals]
        result[stage] = {
            "calls": len(durations),
            "duration_median_ms": statistics.median(durations) if durations else None,
            "duration_max_ms": max(durations) if durations else None,
            "overlap_union_ms": covered_ns(intervals) / 1e6,
        }
    return result


def summarize(trace: dict, role: str, pid: int, cold_sample: dict | None = None) -> dict:
    if role not in {"source", "consumer"}:
        raise ValueError("Expected source or consumer role")
    if (
        trace.get("schema") != "tent.native-trace.v1"
        or trace.get("clock") != "CLOCK_MONOTONIC"
        or type(trace.get("pid")) is not int
        or trace["pid"] != pid
        or type(pid) is not int
        or pid <= 0
        or type(trace.get("limit")) is not int
        or trace.get("limit") != 4096
        or trace.get("overflow") is not False
    ):
        raise ValueError("Invalid native trace identity, clock, limit or overflow")
    events = trace.get("events")
    seen = trace.get("seen")
    if (
        not isinstance(events, list)
        or type(seen) is not int
        or not 0 <= seen <= 4096
        or seen != len(events)
    ):
        raise ValueError("Native trace count does not match retained events")
    for row in events:
        if (
            not isinstance(row, dict)
            or row.get("stage") not in STAGES
            or type(row.get("status")) is not int
            or row["status"] != 0
        ):
            raise ValueError("Native trace has an unknown stage or unsuccessful call")
        if any(
            type(row.get(field)) is not int for field in ("begin_mono_ns", "end_mono_ns", "tid")
        ):
            raise ValueError("Native trace timestamps and thread IDs must be integers")
        if not 0 <= row["begin_mono_ns"] <= row["end_mono_ns"] or row["tid"] <= 0:
            raise ValueError("Invalid native trace interval or thread ID")
    required = STAGES if role == "consumer" else {"endpoint_construct"}
    if not required <= {row["stage"] for row in events}:
        raise ValueError("Native trace lacks a required stage")
    result = {
        "schema": "tent.native-trace.summary.v1",
        "pid": pid,
        "role": role,
        "clock": trace["clock"],
        "stages": stage_summary(events),
        "scope": "Per-host diagnostic; stages overlap and cannot be added or compared across host clocks",
    }
    if cold_sample is not None:
        if role != "consumer":
            raise ValueError("A source trace cannot use the consumer's clock window")
        fields = ("begin_mono_ns", "submit_return_mono_ns", "terminal_mono_ns", "freed_mono_ns")
        times = [cold_sample.get(field) for field in fields]
        if any(type(value) is not int or value < 0 for value in times) or times != sorted(times):
            raise ValueError("Invalid consumer cold READ clock boundaries")
        begin, _, _, end = times
        stages = stage_summary(events, (begin, end))
        if any(stages[stage]["calls"] == 0 for stage in required):
            raise ValueError("Required stages do not overlap this consumer's cold READ")
        result["cold_read"] = {"observed_ms": (end - begin) / 1e6, "stages": stages}
    return result


def summarize_manager(trace: dict, cell_result: dict) -> dict:
    """Link one consumer Manager trace to its local fixture's query windows."""
    if cell_result.get("status") != "PASS_DESCRIPTIVE_PROFILE" or any(
        key in cell_result for key in ("error", "cleanup_error")
    ):
        raise ValueError("Manager cell must have completed with successful cleanup")
    cell = cell_result["cell"]
    host = cell["target_host"]
    boot = cell_result["before"][host]["boot_id"]
    if host == cell["source_host"] or cell_result["after"][host]["boot_id"] != boot:
        raise ValueError("Consumer must retain its own host and boot throughout the cell")
    owners = {}
    for name, expected_exit in (("manager", 0), ("fixture", 143)):
        ready = [
            row
            for row in cell_result["readiness"]
            if row.get("host") == host and row.get("name") == name
        ]
        stopped = [
            row
            for row in cell_result["exits"]
            if row.get("host") == host and row.get("name") == name
        ]
        if len(ready) != 1 or len(stopped) != 1:
            raise ValueError("Expected one same-host readiness and actual wait per owner")
        ready, stopped = ready[0], stopped[0]
        if (
            type(ready.get("pid")) is not int
            or ready["pid"] <= 0
            or type(ready.get("start_ticks")) is not int
            or ready["start_ticks"] <= 0
            or stopped.get("start_ticks") != ready["start_ticks"]
            or any(
                row.get("cell") != cell["id"] or row.get("boot_id") != boot
                for row in (ready, stopped)
            )
            or stopped.get("pid") != ready["pid"]
            or ready.get("exit_code") is not None
            or ready.get("actual_wait_reap") is not False
            or stopped.get("actual_wait_reap") is not True
            or stopped.get("forced_cleanup") is not False
            or type(stopped.get("exit_code")) is not int
            or stopped["exit_code"] != expected_exit
        ):
            raise ValueError("Manager trace requires matching live owners and normal actual waits")
        owners[name] = ready
    if owners["manager"]["pid"] == owners["fixture"]["pid"]:
        raise ValueError("Manager and query actor must be distinct processes")
    if (
        trace.get("boot_id") != boot
        or type(trace.get("start_ticks")) is not int
        or trace["start_ticks"] != owners["manager"]["start_ticks"]
    ):
        raise ValueError("Native trace does not belong to this Manager's boot and process start")
    result = summarize(trace, "consumer", owners["manager"]["pid"])
    evidence = cell_result["evidence"]
    if evidence.get("status") != "PASS_DESCRIPTIVE_PROFILE" or not evidence.get("samples"):
        raise ValueError("Manager profile lacks validated remote samples")
    windows = []
    queries = set()
    previous_ready = 0
    actor = None
    for iteration, sample in enumerate(evidence["samples"]):
        window = sample["query_window"]
        validate_query_window(window, sample["timing"]["query_ms"])
        query = sample.get("query_id")
        identity = (window["instance"], window["namespace"])
        if (
            window["pid"] != owners["fixture"]["pid"]
            or window["start_ticks"] != owners["fixture"]["start_ticks"]
            or window["boot_id"] != boot
            or window["begin_mono_ns"] < previous_ready
            or type(sample.get("iteration")) is not int
            or sample["iteration"] != iteration
            or sample.get("phase") not in {"cold", "warmup", "measured"}
            or (sample["phase"] == "cold") != (iteration == 0)
            or not isinstance(query, str)
            or not query
            or query in queries
            or (actor is not None and identity != actor)
        ):
            raise ValueError("Query windows differ from the local actor, cell order or identity")
        actor = identity
        queries.add(query)
        previous_ready = window["ready_mono_ns"]
        windows.append(
            {
                "iteration": iteration,
                "phase": sample["phase"],
                "query_id": query,
                "actor": window,
                "observed_ms": sample["timing"]["query_ms"],
                "stages": stage_summary(
                    trace["events"], (window["begin_mono_ns"], window["ready_mono_ns"])
                ),
            }
        )
    result.update(
        schema="tent.manager-trace.summary.v1",
        host=host,
        boot_id=boot,
        cell=cell["id"],
        client_query_windows=windows,
        scope=(
            "Same-host client query-prefetch windows include IPC and polling sleeps; "
            "they are not native READ terminal/free boundaries. Stage temporal unions "
            "may overlap, do not establish causality and cannot be added or compared "
            "across host clocks. Zero warm setup calls are valid."
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--endpoint-stdout", type=Path)
    inputs.add_argument("--manager-cell", type=Path)
    parser.add_argument("--endpoint-exit-code", type=int)
    parser.add_argument("--role", choices=("source", "consumer"), required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.manager_cell is not None:
        if args.role != "consumer" or args.endpoint_exit_code is not None:
            raise ValueError("Manager mode uses consumer role and actual waits from its cell")
        result = summarize_manager(
            json.loads(args.trace.read_text()), json.loads(args.manager_cell.read_text())
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    if args.endpoint_exit_code != 0:
        raise ValueError("Endpoint process must exit zero")
    records = []
    for line in args.endpoint_stdout.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("tent_stage_probe"):
            records.append(record)
    ready = [row for row in records if row.get("event") == "ready"]
    stopped = [row for row in records if row.get("event") == "stopped"]
    if (
        len(ready) != 1
        or len(stopped) != 1
        or any(stopped[0][key] != 0 for key in ("registered_regions", "active_batches"))
    ):
        raise ValueError("Expected one endpoint identity and a drained stop record")
    sample = None
    if args.role == "consumer":
        reads = [row for row in records if row.get("event") == "read_complete"]
        if len(reads) != 1 or not reads[0]["samples"]:
            raise ValueError("Expected one completed READ command")
        sample = reads[0]["samples"][0]
    result = summarize(json.loads(args.trace.read_text()), args.role, ready[0]["pid"], sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
