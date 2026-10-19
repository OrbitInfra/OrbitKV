"""Validate pinned native TENT traces without adding overlapping call times."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benches.common.artifacts import external_path
from benches.common.metrics import geometric_mean, geometric_mean_interval
from benches.distributed.shared_cache_profile import validate_query_window

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


def manager_owners(cell_result: dict, host: str) -> dict:
    if cell_result.get("status") != "PASS_DESCRIPTIVE_PROFILE" or any(
        key in cell_result for key in ("error", "cleanup_error")
    ):
        raise ValueError("Manager cell must have completed with successful cleanup")
    cell = cell_result["cell"]
    boot = cell_result["before"][host]["boot_id"]
    if cell_result["after"][host]["boot_id"] != boot:
        raise ValueError("Manager must retain its own host and boot throughout the cell")
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
    return owners


def manager_windows(cell_result: dict, events: list[dict]) -> list[dict]:
    cell = cell_result["cell"]
    host = cell["target_host"]
    if host == cell["source_host"]:
        raise ValueError("Consumer and source must be distinct hosts")
    owners = manager_owners(cell_result, host)
    boot = cell_result["before"][host]["boot_id"]
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
                "stages": stage_summary(events, (window["begin_mono_ns"], window["ready_mono_ns"])),
            }
        )
    return windows


def summarize_manager(trace: dict, cell_result: dict) -> dict:
    """Link one consumer Manager trace to its local fixture's query windows."""
    cell = cell_result["cell"]
    host = cell["target_host"]
    owners = manager_owners(cell_result, host)
    boot = cell_result["before"][host]["boot_id"]
    if (
        trace.get("boot_id") != boot
        or type(trace.get("start_ticks")) is not int
        or trace["start_ticks"] != owners["manager"]["start_ticks"]
    ):
        raise ValueError("Native trace does not belong to this Manager's boot and process start")
    result = summarize(trace, "consumer", owners["manager"]["pid"])
    result.update(
        schema="tent.manager-trace.summary.v1",
        host=host,
        boot_id=boot,
        cell=cell["id"],
        client_query_windows=manager_windows(cell_result, trace["events"]),
        scope=(
            "Same-host client query-prefetch windows include IPC and polling sleeps; "
            "they are not native READ terminal/free boundaries. Stage temporal unions "
            "may overlap, do not establish causality and cannot be added or compared "
            "across host clocks. Zero warm setup calls are valid."
        ),
    )
    return result


def validate_observer_pilot(contract: dict) -> None:
    """Freeze the bounded eight-cell overhead experiment, not deployment support."""
    if (
        contract.get("observer_pilot") != "manager-native-observer.v1"
        or type(contract.get("warmup")) is not int
        or contract["warmup"] != 5
        or type(contract.get("samples")) is not int
        or contract["samples"] != 50
        or set(contract["hosts"]) != {"h0", "h1"}
        or len(contract["cells"]) != 8
        or len({cell["id"] for cell in contract["cells"]}) != 8
        or not contract.get("required_idle_metrics")
    ):
        raise ValueError("Expected the frozen eight-cell Manager observer pilot")
    for index, cell in enumerate(contract["cells"]):
        source, target = ("h0", "h1") if index < 4 else ("h1", "h0")
        pair = 1 + index % 4 // 2
        observer = ("off", "on", "on", "off")[index % 4]
        if (
            cell.get("kind") != "manager"
            or cell.get("source_host") != source
            or cell.get("target_host") != target
            or cell.get("pair_id") != f"{source}-{target}-p{pair}"
            or cell.get("observer") != observer
            or cell.get("ordinal") != index
            or cell.get("payload") != 8 << 20
            or cell.get("segment_bytes") != 65536
            or not isinstance(cell["id"], str)
            or not cell["id"]
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in cell["id"])
        ):
            raise ValueError("Pilot matrix differs from the frozen order, direction or geometry")
        for host in (source, target):
            launches = contract["launches"][host][cell["id"]]
            for name in ("manager", "fixture"):
                planned = launches[name]
                enabled = name == "manager" and observer == "on"
                if (
                    planned["kind"] != name
                    or planned["runtime"].get("observer_enabled") is not enabled
                    or bool(planned["environment"].get("TENT_NATIVE_TRACE_PATH")) is not enabled
                    or any(
                        key in planned["environment"]
                        for key in ("MC_FORCE_TCP", "LD_PRELOAD", "TENT_NATIVE_TRACE_PID")
                    )
                ):
                    raise ValueError("Pilot launch differs from its declared observer mode")


def validate_observer_cell(contract: dict, result: dict) -> dict:
    cell = result["cell"]
    enabled = cell["observer"] == "on"
    if result.get("evidence", {}).get("payload_bytes") != cell["payload"]:
        raise ValueError("Pilot payload evidence differs from its cell")
    samples = result["evidence"]["samples"]
    if [row.get("phase") for row in samples] != ["cold"] + ["warmup"] * 5 + ["measured"] * 50:
        raise ValueError("Pilot must retain one cold, five warmup and fifty measured samples")
    for sample in samples:
        values = [sample.get("manager_fetch_ms"), sample.get("timing", {}).get("query_restore_ms")]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError("Pilot primary timings must be positive and finite")
    hashes = samples[0].get("sha256")
    if (
        not isinstance(hashes, list)
        or len(hashes) != 4
        or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
        or any(
            sample.get("sha256") != hashes
            or sample.get("remote_bytes") != cell["payload"]
            or sample.get("h2d_bytes") != cell["payload"]
            for sample in samples
        )
    ):
        raise ValueError("Pilot lacks complete repeated remote/GPU byte evidence")
    for key in ("pre_stop_idle", "cleanup_idle"):
        quiet = result[key]
        if len(quiet) != 2 or any(
            row.get(name) != 0 for row in quiet for name in contract["required_idle_metrics"]
        ):
            raise ValueError("Pilot lacks its complete idle proof")
    summaries = {}
    if set(result["native_traces"]) != {cell["source_host"], cell["target_host"]}:
        raise ValueError("Pilot needs both stopped Manager trace receipts")
    for host, receipt in result["native_traces"].items():
        owners = manager_owners(result, host)
        stopped = next(
            row for row in result["exits"] if row["host"] == host and row["name"] == "manager"
        )
        if (
            receipt.get("host") != host
            or receipt.get("cell") != cell["id"]
            or receipt.get("observer_enabled") is not enabled
            or receipt.get("stop") != stopped
            or stopped.get("was_running") is not True
            or result["runtime_admission"][host].get("cell") != cell["id"]
            or result["runtime_admission"][host].get("host") != host
        ):
            raise ValueError("Pilot trace receipt differs from the actual stopped owner")
        for name, owner in owners.items():
            admitted = result["runtime_admission"][host]["owners"][name]
            if any(admitted.get(key) != owner[key] for key in ("pid", "start_ticks")):
                raise ValueError("Pilot runtime admission belongs to a different process")
        content = receipt.get("content")
        if not enabled:
            if content is not None or receipt.get("sha256") is not None:
                raise ValueError("Off cell must have no native trace")
            continue
        if (
            not isinstance(content, str)
            or len(content.encode()) > 1024 * 1024
            or hashlib.sha256(content.encode()).hexdigest() != receipt.get("sha256")
        ):
            raise ValueError("Pilot trace content differs from its bounded digest")
        trace = json.loads(content)
        owner = owners["manager"]
        if any(trace.get(key) != owner[key] for key in ("pid", "boot_id", "start_ticks")):
            raise ValueError("Pilot trace is not this host's Manager process")
        summaries[host] = (
            summarize_manager(trace, result)
            if host == cell["target_host"]
            else summarize(trace, "source", owner["pid"])
        )
    if not enabled:
        manager_windows(result, [])
    return summaries


def observer_overhead(contract: dict, results: list[dict]) -> dict:
    """Check completed independent pairs before admitting another cell."""
    validate_observer_pilot(contract)
    if len(results) > 8 or any(
        result["cell"] != cell for result, cell in zip(results, contract["cells"], strict=False)
    ):
        raise ValueError("Pilot results must be the original ordered matrix prefix")
    for result in results:
        validate_observer_cell(contract, result)
    pairs = []
    for index in range(0, len(results) - 1, 2):
        matched = {result["cell"]["observer"]: result for result in results[index : index + 2]}
        metrics = {}
        for metric in ("manager_fetch_ms", "query_restore_ms"):
            values = {}
            for mode, result in matched.items():
                measured = [
                    row for row in result["evidence"]["samples"] if row["phase"] == "measured"
                ]
                values[mode] = [
                    row[metric] if metric == "manager_fetch_ms" else row["timing"][metric]
                    for row in measured
                ]
            ratio = statistics.median(values["on"]) / statistics.median(values["off"])
            tails = {
                mode: sorted(rows)[math.ceil(len(rows) * 0.99) - 1] for mode, rows in values.items()
            }
            difference = tails["on"] - tails["off"]
            metrics[metric] = {
                "median_ratio": ratio,
                "nearest_rank_p99_ms": tails,
                "tail_difference_ms": difference,
                "passed": ratio <= 1.05 and difference <= 0.25,
            }
        pairs.append(
            {
                "pair_id": matched["on"]["cell"]["pair_id"],
                "source_host": matched["on"]["cell"]["source_host"],
                "metrics": metrics,
            }
        )
    directions = {}
    for source in ("h0", "h1"):
        rows = [row for row in pairs if row["source_host"] == source]
        if len(rows) != 2:
            continue
        directions[source] = {}
        for metric in ("manager_fetch_ms", "query_restore_ms"):
            ratios = [row["metrics"][metric]["median_ratio"] for row in rows]
            mean, ci = geometric_mean(ratios), geometric_mean_interval(ratios, seed=42, draws=10000)
            directions[source][metric] = {
                "independent_pairs": 2,
                "geometric_mean_ratio": mean,
                "pair_bootstrap_95_ci": ci,
                "passed": mean <= 1.05 and ci[1] <= 1.05,
            }
    passed = all(metric["passed"] for row in pairs for metric in row["metrics"].values()) and all(
        metric["passed"] for row in directions.values() for metric in row.values()
    )
    return {
        "status": "OVERHEAD_FAIL"
        if not passed
        else "MEASUREMENT_GUARDS_PASS"
        if len(results) == 8
        else "INCOMPLETE",
        "completed_cells": len(results),
        "pairs": pairs,
        "directions": directions,
        "tail_scope": "Nearest-rank p99 equals max with fifty samples; no stable tail qualification",
        "uncertainty_scope": "Only two independent pairs per direction; the bootstrap upper reduces to the worst pair, not population confidence",
        "postflight_required": True,
        "scope": "Instrumentation overhead only; cold samples descriptive, no serving or transport qualification",
    }


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
