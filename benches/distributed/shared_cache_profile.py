"""Measure one cold and repeated remote DRAM restores through real Managers.

Consumer DRAM eviction, barriers, GPU clearing, HTTP and the payload oracle are
outside the query/restore clocks. Each sample must prove an actual remote read.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import uuid

import requests

from benches.common.artifacts import external_path
from benches.common.metrics import REMOTE_STAGES
from benches.common.workload import evict_host_cache
from benches.distributed.shared_cache import IDLE_METRICS, drain, synchronize


def validate_query_window(window: dict, query_ms: float) -> None:
    if not isinstance(window, dict) or window.get("clock") != "CLOCK_MONOTONIC":
        raise ValueError("Expected a client query-prefetch CLOCK_MONOTONIC window")
    if any(
        type(window.get(field)) is not int or window[field] <= 0 for field in ("pid", "start_ticks")
    ):
        raise ValueError("Invalid query actor PID or process start")
    for field in ("instance", "namespace", "boot_id"):
        if not isinstance(window.get(field), str) or not window[field]:
            raise ValueError("Missing query actor identity")
    if str(uuid.UUID(window["boot_id"])) != window["boot_id"]:
        raise ValueError("Invalid query actor boot ID")
    begin, ready = (window.get(field) for field in ("begin_mono_ns", "ready_mono_ns"))
    if any(type(value) is not int for value in (begin, ready)) or not 0 <= begin < ready:
        raise ValueError("Invalid client query-prefetch clock boundaries")
    if (
        type(query_ms) not in (int, float)
        or not math.isfinite(query_ms)
        or not math.isclose(query_ms, (ready - begin) / 1e6, rel_tol=1e-12, abs_tol=1e-9)
    ):
        raise ValueError("Query duration differs from the recorded client window")


def sample_evidence(before: dict, after: dict, restored: dict, payload_bytes: int) -> dict:
    """Reject local hits, incomplete source release and ambiguous stage counts."""
    counters = {}
    for label, name in (
        ("remote_bytes", "orbitkv_remote_fetch_bytes_total"),
        ("h2d_bytes", "orbitkv_load_bytes_total"),
    ):
        if name not in after:
            raise AssertionError(f"Missing counter: {name}")
        counters[label] = int(after[name] - before.get(name, 0))
        if counters[label] != payload_bytes:
            raise AssertionError(f"Expected {payload_bytes} {label}, got {counters[label]}")
    if restored["checked_payload_bytes"] != payload_bytes or len(restored["sha256"]) != 4:
        raise AssertionError("GPU fixture did not verify the entire declared payload")
    stages = {}
    for stage in REMOTE_STAGES:
        count = f"orbitkv_remote_stage_duration_seconds_count_{stage}"
        total = f"orbitkv_remote_stage_duration_seconds_sum_{stage}"
        if count not in after or total not in after or after[count] - before.get(count, 0) != 1:
            raise AssertionError(f"Expected exactly one completed {stage} stage")
        duration = (after[total] - before.get(total, 0)) * 1000
        if not math.isfinite(duration) or duration <= 0:
            raise AssertionError(f"Invalid {stage} duration: {duration}")
        stages[stage] = duration
    count = "orbitkv_remote_fetch_duration_seconds_count"
    total = "orbitkv_remote_fetch_duration_seconds_sum"
    if after.get(count, 0) - before.get(count, 0) != 1 or total not in after:
        raise AssertionError("Expected one remote fetch timing per sample")
    fetch_ms = (after[total] - before.get(total, 0)) * 1000
    if not math.isfinite(fetch_ms) or fetch_ms <= 0:
        raise AssertionError("Invalid remote fetch duration")
    timing = restored["timing"]
    if any(not math.isfinite(value) or value < 0 for value in timing.values()):
        raise AssertionError("Invalid GPU worker timing")
    validate_query_window(restored.get("query_window"), timing["query_ms"])
    if not isinstance(restored.get("query_id"), str) or not restored["query_id"]:
        raise ValueError("Missing query identity")
    return {
        **counters,
        "remote_stages_ms": stages,
        "manager_fetch_ms": fetch_ms,
        "tent_read_payload_gb_s": payload_bytes / (stages["read"] * 1e6),
        "timing": timing,
        "query_window": restored["query_window"],
    }


def summarize(samples: list[dict]) -> dict:
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    values = {
        "manager_fetch_ms": [row["manager_fetch_ms"] for row in measured],
        "tent_read_payload_gb_s": [row["tent_read_payload_gb_s"] for row in measured],
    }
    for stage in REMOTE_STAGES:
        values[stage + "_ms"] = [row["remote_stages_ms"][stage] for row in measured]
    for field in measured[0]["timing"]:
        values[field] = [row["timing"][field] for row in measured]
    return {
        "measured_samples": len(measured),
        "statistics": {
            name: {
                "p50": statistics.median(numbers),
                "p95_nearest_rank": sorted(numbers)[math.ceil(len(numbers) * 0.95) - 1],
                "min": min(numbers),
                "max": max(numbers),
            }
            for name, numbers in values.items()
        },
        "tail_scope": "Descriptive single-run p95/max; no stable p99 or performance qualification",
    }


def profile(args: argparse.Namespace) -> dict:
    if args.source_url == args.target_url or args.source_manager == args.target_manager:
        raise ValueError("Independent source and consumer are required")
    if not 1 <= args.warmup <= 100 or not 1 <= args.samples <= 10000 or args.payload_bytes <= 0:
        raise ValueError("Invalid bounded workload")

    def command(url, operation):
        response = requests.post(url, json={"operation": operation}, timeout=60)
        if not response.ok:
            raise RuntimeError(response.text)
        return response.json()

    saved = command(args.source_url, "save")
    if saved["saved_bytes"] != args.payload_bytes:
        raise AssertionError("Source payload differs from frozen profile")
    fence = synchronize(args.source_manager, args.target_manager)
    result = {"source_fence": fence, "payload_bytes": args.payload_bytes, "samples": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_hashes = None
    for iteration in range(1 + args.warmup + args.samples):
        phase = "cold" if iteration == 0 else "warmup" if iteration <= args.warmup else "measured"
        preparation_started = time.monotonic_ns()
        eviction = None
        if iteration:
            eviction = evict_host_cache(args.target_manager)
            if eviction["cleanup"]["evicted_bytes"] != args.payload_bytes:
                raise AssertionError("Consumer DRAM eviction did not match the previous replica")
            synchronize(args.target_manager, args.source_manager)
        _, before = drain(args.source_manager, args.target_manager)
        prepared = time.monotonic_ns()
        restored = command(args.target_url, "load")
        returned = time.monotonic_ns()
        source_after, after = drain(args.source_manager, args.target_manager)
        evidence = sample_evidence(before, after, restored, args.payload_bytes)
        if expected_hashes is None:
            expected_hashes = restored["sha256"]
        elif restored["sha256"] != expected_hashes:
            raise AssertionError("Repeated restore changed payload or sentinel gaps")
        gauges = []
        for snapshot in [source_after, after]:
            observed = {name: snapshot[name] for name in IDLE_METRICS if name in snapshot}
            if any(value != 0 for value in observed.values()):
                raise AssertionError(f"Resources did not drain: {observed}")
            gauges.append(
                {
                    "observed_zero": observed,
                    "unexported": [name for name in IDLE_METRICS if name not in snapshot],
                }
            )
        for name in ["orbitkv_transfer_lock_active", "orbitkv_transfer_reserved_bytes"]:
            if name not in source_after:
                raise AssertionError(f"Source ownership gauge missing: {name}")
        for name in ["orbitkv_query_reserved_bytes", "orbitkv_transfer_completion_outstanding"]:
            if name not in after:
                raise AssertionError(f"Consumer ownership gauge missing: {name}")
        result["samples"].append(
            {
                "iteration": iteration,
                "phase": phase,
                "query_id": restored["query_id"],
                "sha256": restored["sha256"],
                "preparation_ms": (prepared - preparation_started) / 1e6,
                "http_and_oracle_wall_ms": (returned - prepared) / 1e6,
                "eviction": eviction,
                "before": before,
                "after": after,
                "ownership": gauges,
                **evidence,
            }
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result.update(status="PASS_DESCRIPTIVE_PROFILE", summary=summarize(result["samples"]))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-url", "target-url", "source-manager", "target-manager"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--payload-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=external_path, required=True)
    result = profile(parser.parse_args())
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
