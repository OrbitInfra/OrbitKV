"""Analyze independent-pair S2.10 observer process-isolation runs."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import statistics
from pathlib import Path

from benches.cluster.live_store_measurements import _summary
from benches.cluster.summarize_live_store_diagnostics import summarize as summarize_stages
from benches.common.artifacts import external_path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _bootstrap_geomean(ratios: list[float], seed: int, draws: int = 10_000) -> dict:
    estimate = math.exp(statistics.fmean(math.log(value) for value in ratios))
    rng = random.Random(seed)
    samples = sorted(
        math.exp(statistics.fmean(math.log(rng.choice(ratios)) for _ in ratios))
        for _ in range(draws)
    )
    return {
        "pairs": len(ratios),
        "geometric_mean": estimate,
        "bootstrap_95_ci": [
            samples[round((draws - 1) * 0.025)],
            samples[round((draws - 1) * 0.975)],
        ],
        "unit": "independent matched pair; requests within a run are not resampled",
    }


def _overlaps(starts: list[int], ends: list[int], start: int, end: int) -> bool:
    position = bisect.bisect_right(starts, end) - 1
    return position >= 0 and ends[position] >= start


def _tail_rows(
    samples: list[dict],
    observer: list[dict],
    operation: str,
    selected: list[dict],
) -> list[dict]:
    starts = [row["poll_start_mono_ns"] for row in observer]
    ends = [row["poll_end_mono_ns"] for row in observer]
    result = []
    for sample in selected:
        channel = sample["save_channel"] if operation == "save" else sample["query_channels"][-1]
        outer_start = sample[f"{operation}_start_mono_ns"]
        outer_end = sample[f"{operation}_end_mono_ns"]
        returned = channel["returned_mono_ns"]
        outer_ms = sample["manager_save_submit_ms" if operation == "save" else "local_query_ms"]
        post_return_ms = sample[f"{operation}_post_return_ms"]
        result.append(
            {
                "round": sample["round"],
                "outer_ms": outer_ms,
                "post_return_ms": post_return_ms,
                "post_return_share": post_return_ms / outer_ms if outer_ms else None,
                "observer_overlaps_outer": _overlaps(starts, ends, outer_start, outer_end),
                "observer_overlaps_post_return": _overlaps(starts, ends, returned, outer_end),
            }
        )
    return result


def _stream_delta(result: dict, name: str) -> int:
    return result["metadata_after"]["stream"][name] - result["metadata_before"]["stream"][name]


def _analyze_run(root: Path, run: dict, settings: dict) -> tuple[dict, list[str]]:
    path = root / run["name"]
    result = json.loads((path / "isolation-result.json").read_text())
    samples = [row for row in _rows(path / "samples.jsonl") if row["measured"]]
    observer = _rows(path / "observer-samples.jsonl")
    stages = summarize_stages(path)
    stage_by_request = {
        (row["operation"], row["request_id"]): row
        for row in stages["stage_samples"]
        if row["measured"]
    }
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{run['name']}: {message}")

    require(result["status"] == "passed", f"status={result['status']}")
    require(result["observer_mode"] == run["observer_mode"], "observer mode mismatch")
    require(result["condition"] == "pressure" and result["medium"] == "ssd", "scope mismatch")
    require(len(samples) == settings["samples"], f"measured samples={len(samples)}")
    require(len(observer) == settings["observer_poll_count"], f"observer polls={len(observer)}")
    require(
        result["observer_exit"]["samples"] == settings["observer_poll_count"],
        "observer exit sample count mismatch",
    )
    require(not (path / "observer-error.json").exists(), "observer error record exists")
    require(
        {name: row["sha256"] for name, row in result["artifacts"].items()}
        == settings["runtime_hashes"],
        "runtime hashes mismatch",
    )
    require(
        all(row["ssd_read_bytes"] == settings["payload_bytes"] for row in samples),
        "SSD read oracle mismatch",
    )
    exposure = result["exposure"]
    require(
        exposure["observer_poll_count"] == settings["observer_poll_count"],
        "exposure observer count mismatch",
    )
    phase = exposure["observer_foreground_phase"]
    require(phase is not None, "missing actual observer/foreground phase")
    if phase is not None:
        scheduled_phase_ms = (
            settings["foreground_phase_offset_ms"] - settings["observer_phase_offset_ms"]
        )
        require(
            abs(phase["scheduled_ms"] - scheduled_phase_ms) < 1e-9,
            "scheduled phase mismatch",
        )
        require(
            abs(phase["deviation_ms"]) <= settings["validity"]["max_phase_deviation_ms"],
            f"actual phase deviation={phase['deviation_ms']}",
        )
    require(
        exposure["observer_scheduling_deviation_ms"]["max"]
        <= settings["validity"]["max_schedule_deviation_ms"],
        "observer schedule deviation exceeded",
    )
    require(
        exposure["foreground_scheduling_deviation_ms"]["max"]
        <= settings["validity"]["max_schedule_deviation_ms"],
        "foreground schedule deviation exceeded",
    )
    require(
        settings["validity"]["source_rate_ratio"][0]
        <= exposure["source_rate_ratio"]
        <= settings["validity"]["source_rate_ratio"][1],
        "pressure source rate invalid",
    )

    metrics = {
        name: []
        for name in (
            "manager_save_submit_ms",
            "local_query_ms",
            "save_post_return_ms",
            "query_post_return_ms",
            "save_native_channel_ms",
            "query_native_channel_ms",
            "save_manager_execute_ms",
            "query_manager_execute_ms",
            "ssd_queue_ms",
            "ssd_execute_ms",
        )
    }
    for sample in samples:
        save_channel = sample["save_channel"]
        save_stage = stage_by_request[("save", save_channel["request_id"])]
        query_channels = sample["query_channels"]
        query_stages = [
            stage_by_request[("query", channel["request_id"])] for channel in query_channels
        ]
        metrics["manager_save_submit_ms"].append(sample["manager_save_submit_ms"])
        metrics["local_query_ms"].append(sample["local_query_ms"])
        metrics["save_post_return_ms"].append(sample["save_post_return_ms"])
        metrics["query_post_return_ms"].append(sample["query_post_return_ms"])
        metrics["save_native_channel_ms"].append(
            (save_channel["returned_mono_ns"] - save_channel["submitted_mono_ns"]) / 1_000_000
        )
        metrics["query_native_channel_ms"].append(
            (query_channels[-1]["returned_mono_ns"] - query_channels[0]["submitted_mono_ns"])
            / 1_000_000
        )
        metrics["save_manager_execute_ms"].append(save_stage["manager_execute_ms"])
        metrics["query_manager_execute_ms"].append(
            sum(stage["manager_execute_ms"] for stage in query_stages)
        )
        metrics["ssd_queue_ms"].append(save_stage["ssd_queue_ms"])
        metrics["ssd_execute_ms"].append(save_stage["ssd_execute_ms"])
    summaries = {name: _summary(values) for name, values in metrics.items()}
    tails = {}
    for operation, primary in (
        ("save", "manager_save_submit_ms"),
        ("query", "local_query_ms"),
    ):
        ordered = sorted(samples, key=lambda row: row[primary], reverse=True)
        threshold = summaries[primary]["p99"]
        p99_tail = [row for row in samples if row[primary] >= threshold]
        tails[operation] = {
            "top_10": _tail_rows(samples, observer, operation, ordered[:10]),
            "p99_tail": _tail_rows(samples, observer, operation, p99_tail),
        }
    return (
        {
            "pair": run["pair"],
            "observer_mode": run["observer_mode"],
            "seed": run["seed"],
            "metrics": summaries,
            "tails": tails,
            "observer": {
                "polls": len(observer),
                "response_bytes": result["observer_exit"]["response_bytes"],
                "sampler_cpu_ns": result["observer_exit"]["sampler_cpu_ns"],
                "poll_duration_ms": exposure["observer_poll_ms"],
                "actual_gap_ms": exposure["observer_poll_gap_ms"],
                "schedule_deviation_ms": exposure["observer_scheduling_deviation_ms"],
                "phase": phase,
            },
            "pressure": {
                "rounds": exposure["source_rounds"],
                "changes": result["pressure_result"]["changes"],
                "source_rate_ratio": exposure["source_rate_ratio"],
                "frames_received": _stream_delta(result, "frames_received"),
                "encoded_bytes_received": _stream_delta(result, "encoded_bytes_received"),
            },
            "oracles": {
                "payload_sha256": result["payload_oracle_sha256"],
                "keys_sha256": result["key_oracle_sha256"],
                "manager_exit": result["manager_exit"],
                "pressure_exit": result["pressure_exit"],
                "final_drain_metrics": result["final_drain_metrics"],
            },
        },
        errors,
    )


def analyze(inputs: Path, root: Path) -> dict:
    settings = json.loads((inputs / "settings.json").read_text())
    reports = {}
    errors = []
    for run in settings["runs"]:
        reports[run["name"]], run_errors = _analyze_run(root, run, settings)
        errors.extend(run_errors)

    pairs = {}
    primary_names = (
        "manager_save_submit_ms",
        "local_query_ms",
        "save_post_return_ms",
        "query_post_return_ms",
        "save_native_channel_ms",
        "query_native_channel_ms",
        "save_manager_execute_ms",
        "query_manager_execute_ms",
        "ssd_queue_ms",
        "ssd_execute_ms",
    )
    for pair in sorted({run["pair"] for run in settings["runs"]}):
        members = [run for run in settings["runs"] if run["pair"] == pair]
        if len(members) != 2 or {run["observer_mode"] for run in members} != {
            "in-process",
            "helper-process",
        }:
            errors.append(f"{pair}: pair does not contain exactly one A and one B")
            continue
        a_run = next(run for run in members if run["observer_mode"] == "in-process")
        b_run = next(run for run in members if run["observer_mode"] == "helper-process")
        a = reports[a_run["name"]]
        b = reports[b_run["name"]]
        if a_run["seed"] != b_run["seed"]:
            errors.append(f"{pair}: seed mismatch")
        for oracle in ("payload_sha256", "keys_sha256"):
            if a["oracles"][oracle] != b["oracles"][oracle]:
                errors.append(f"{pair}: {oracle} mismatch")
        for field in ("rounds", "changes"):
            if a["pressure"][field] != b["pressure"][field]:
                errors.append(f"{pair}: pressure {field} mismatch")
        for field in ("response_bytes",):
            ratio = _ratio(b["observer"][field], a["observer"][field])
            lower, upper = settings["validity"]["paired_observer_work_ratio"]
            if ratio is None or not lower <= ratio <= upper:
                errors.append(f"{pair}: observer {field} ratio={ratio}")
        for field in ("frames_received", "encoded_bytes_received"):
            ratio = _ratio(b["pressure"][field], a["pressure"][field])
            lower, upper = settings["validity"]["paired_pressure_work_ratio"]
            if ratio is None or not lower <= ratio <= upper:
                errors.append(f"{pair}: pressure {field} ratio={ratio}")
        phase_difference_ms = abs(
            b["observer"]["phase"]["actual_ms"] - a["observer"]["phase"]["actual_ms"]
        )
        if phase_difference_ms > settings["validity"]["max_paired_phase_difference_ms"]:
            errors.append(f"{pair}: actual phase difference={phase_difference_ms}ms")
        metrics = {}
        for name in primary_names:
            a_value = a["metrics"][name]["p99"]
            b_value = b["metrics"][name]["p99"]
            metrics[name] = {
                "a_p99_ms": a_value,
                "b_p99_ms": b_value,
                "absolute_difference_ms": b_value - a_value,
                "ratio": _ratio(b_value, a_value),
            }
        pairs[pair] = {
            "order": [run["observer_mode"] for run in members],
            "a": a_run["name"],
            "b": b_run["name"],
            "metrics": metrics,
        }

    query_ratios = [pair["metrics"]["query_post_return_ms"]["ratio"] for pair in pairs.values()]
    inference = (
        _bootstrap_geomean(query_ratios, settings["bootstrap_seed"]) if query_ratios else None
    )
    criteria = settings["causal_decision"]
    positive_pairs = sum(ratio < 1 for ratio in query_ratios)
    median_absolute_reduction_ms = (
        statistics.median(
            -pair["metrics"]["query_post_return_ms"]["absolute_difference_ms"]
            for pair in pairs.values()
        )
        if pairs
        else None
    )
    native_comparable = all(
        pair["metrics"][metric]["absolute_difference_ms"] <= criteria["max_native_p99_increase_ms"]
        or pair["metrics"][metric]["ratio"] <= criteria["max_native_p99_ratio"]
        for pair in pairs.values()
        for metric in (
            "save_native_channel_ms",
            "query_native_channel_ms",
            "save_manager_execute_ms",
            "query_manager_execute_ms",
        )
    )
    supports_interference = bool(
        not errors
        and len(pairs) == 4
        and positive_pairs >= criteria["minimum_positive_pairs"]
        and inference["geometric_mean"] <= criteria["maximum_post_return_geomean_ratio"]
        and inference["bootstrap_95_ci"][1] < 1
        and median_absolute_reduction_ms >= criteria["minimum_median_absolute_reduction_ms"]
        and native_comparable
    )
    return {
        "contract": settings["contract"],
        "status": "invalid" if errors else "completed_pending_review",
        "scope": "benchmark observer diagnosis only; not isolation qualification",
        "runs": reports,
        "pairs": pairs,
        "independent_pair_inference": inference,
        "validity_errors": errors,
        "decision": {
            "supports_observer_interference": supports_interference,
            "positive_pairs": positive_pairs,
            "median_absolute_reduction_ms": median_absolute_reduction_ms,
            "native_stages_comparable": native_comparable,
            "criteria": criteria,
            "authorization_if_positive": "benchmark harness repair only",
            "does_not_establish": [
                "production Manager/channel/metadata/SSD defect",
                "DRAM or SSD isolation qualification",
                "support beyond four owners",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=external_path, required=True)
    parser.add_argument("--root", type=external_path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    result = analyze(args.inputs, args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "status": result["status"],
                "decision": result["decision"],
                "validity_errors": result["validity_errors"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
