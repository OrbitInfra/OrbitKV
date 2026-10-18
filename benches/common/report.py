"""Rebuild comparable per-run CSV/JSON summaries without starting a GPU runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from benches.common.artifacts import external_path
from benches.common.metrics import (
    MANAGER_BACKENDS,
    OBSERVATION_MODES,
    cache_source,
    summarize,
    validate_observation,
    workload_phases,
)


def compare_outputs(run: dict, reference: dict) -> dict:
    """Compare matched inputs and runtimes against an unencoded reference backend."""
    args, control = run["manifest"]["arguments"], reference["manifest"]["arguments"]
    if (
        args.get("observation_mode") not in OBSERVATION_MODES
        or control.get("observation_mode") not in OBSERVATION_MODES
    ):
        raise ValueError("Output comparison requires explicit observation modes")
    if control.get("storage_codec", "none") != "none":
        raise ValueError("Output reference must use --storage-codec none")
    for key in (
        "engine",
        "model",
        "workload",
        "lengths",
        "repeats",
        "concurrencies",
        "duration_seconds",
        "max_requests",
        "working_set",
        "reuse_ratio",
        "output_tokens",
        "gpu_tokens",
        "prefill_tokens",
        "host_gib",
        "ssd_gib",
        "query_budget_gib",
        "cache_protected_percent",
        "ssd_write_policy",
        "queue_warmup",
        "prepare_requests",
        "read_batch_mib",
        "read_timeout_ms",
        "read_max_batches",
        "seed",
        "deterministic_inference",
        "enforce_eager",
        "observation_mode",
    ):
        if args.get(key) != control.get(key):
            raise ValueError(
                f"Output reference has different {key}: {args.get(key)!r} != {control.get(key)!r}"
            )
    for key in ("model_revision", "kv_bytes_per_token"):
        if run["manifest"].get(key) != reference["manifest"].get(key):
            raise ValueError(f"Output reference has different {key}")
    for key in ("gpu", "python", "cpu_affinity"):
        if run["manifest"][key] != reference["manifest"][key]:
            raise ValueError(f"Output reference has different {key}")
    for package in (args["engine"], "torch", "transformers"):
        if run["manifest"]["packages"][package] != reference["manifest"]["packages"][package]:
            raise ValueError(f"Output reference has different {package} version")

    def identity(sample):
        return tuple(
            sample.get(key)
            for key in (
                "concurrency",
                "pattern",
                "length",
                "repeat",
                "phase",
                "index",
            )
        )

    samples = [
        json.loads(line)
        for line in (Path(run["directory"]) / "samples.jsonl").read_text().splitlines()
    ]
    baseline = {
        identity(sample): sample
        for line in (Path(reference["directory"]) / "samples.jsonl").read_text().splitlines()
        for sample in [json.loads(line)]
    }
    compared, mismatches, unverified = set(), set(), set()
    for sample in samples:
        key = identity(sample)
        other = baseline.get(key)
        if other is None:
            continue
        if not sample.get("prompt_sha256") or not other.get("prompt_sha256"):
            unverified.add(key)
            continue
        if sample["prompt_sha256"] != other["prompt_sha256"]:
            raise ValueError(f"Output reference prompt differs for {key}")
        compared.add(key)
        if sample["text"] != other["text"]:
            mismatches.add(key)
    for row in run["summary"]:
        group = {
            identity(s)
            for s in samples
            if all(
                s.get(key) == row[key]
                for key in ("concurrency", "pattern", "length", "phase")
                if key in row
            )
        }
        row.update(
            reference_compared_requests=len(group & compared),
            reference_output_mismatches=len(group & mismatches),
            reference_uncompared_requests=len(group - compared),
        )
    return {
        "directory": reference["directory"],
        "compared_requests": len(compared),
        "output_mismatches": len(mismatches),
        "unverified_input_requests": len(unverified),
        "run_uncompared_requests": len(samples) - len(compared),
        "reference_uncompared_requests": len(baseline) - len(compared),
        "scope": "Exact text on matching prompt hashes; synthetic workload diagnostic, not task quality or batch-invariant correctness. Duration-limited runs may compare only a shared request prefix.",
    }


def collect_run(directory: Path) -> dict:
    if (directory / "failure.json").exists():
        raise ValueError(f"Failed run has no complete latency result: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    args = manifest["arguments"]
    profiling = args.get("sglang_profile", False)
    if type(profiling) is not bool or (
        profiling and (args["engine"] != "sglang" or args.get("workload", "serial") != "serial")
    ):
        raise ValueError(f"Invalid SGLang profiling configuration: {directory}")
    mode = args.get("observation_mode")
    if mode not in OBSERVATION_MODES or manifest.get("observation_mode") != mode:
        raise ValueError(f"Missing or inconsistent observation mode: {directory}")
    manager_present = manifest.get("manager_present")
    if type(manager_present) is not bool:
        raise ValueError(f"Missing or invalid Manager observation metadata: {directory}")
    if manager_present != (args.get("backend") in MANAGER_BACKENDS):
        raise ValueError(f"Backend and Manager presence do not match: {directory}")
    manager_backend = args["backend"] if manager_present else None
    if "manager_backend" not in manifest or manifest["manager_backend"] != manager_backend:
        raise ValueError(f"Missing or inconsistent Manager backend: {directory}")
    timeline = directory / "timeline-summary.json"
    timeline_summary = json.loads(timeline.read_text()) if timeline.exists() else None
    usage = directory / "manager-usage.json"
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    if args.get("workload") in ("concurrent", "sustained"):
        from benches.serving import concurrent, sustained

        workload = sustained if args["workload"] == "sustained" else concurrent
        measurement_file = "windows.jsonl" if args["workload"] == "sustained" else "batches.jsonl"
        batches = [
            json.loads(line) for line in (directory / measurement_file).read_text().splitlines()
        ]
        validate_observation(mode, batches, manager_backend=manager_backend)
        workload.validate(args, samples, batches)
        summary = workload.summarize(samples, batches)
    else:
        expected = {
            (length, repeat, phase)
            for length in args["lengths"]
            for repeat in range(args["repeats"])
            for phase in workload_phases(bool(args.get("ssd_gib", 0)))
        }
        actual = {(s["length"], s["repeat"], s["phase"]) for s in samples}
        if actual != expected or len(samples) != len(expected):
            raise ValueError(f"Incomplete or duplicated workload: {directory}")
        for sample in samples:
            expected_profiling = (
                "sglang-public-cpu-cuda"
                if profiling and sample["phase"] == "after_pressure"
                else "off"
            )
            if sample.get("profiling", {}).get("mode", "off") != expected_profiling:
                raise ValueError(f"Sample profiling mode differs from manifest: {directory}")
            for metric in ("ttft_ms", "e2e_ms"):
                if not math.isfinite(sample[metric]) or sample[metric] < 0:
                    raise ValueError(f"Invalid {metric} in {directory}")
            sample["cache_source"] = cache_source(args["engine"], sample)
        validate_observation(mode, samples, manager_backend=manager_backend)
        summary = summarize(samples, args["lengths"])
    return {
        "directory": str(directory.resolve()),
        "manifest": manifest,
        "timeline": timeline_summary,
        "storage": json.loads((directory / "storage.json").read_text())
        if args.get("ssd_gib", 0)
        else None,
        "manager_usage": json.loads(usage.read_text()) if usage.exists() else None,
        "native_io": json.loads((directory / "native-io.json").read_text())
        if args.get("gds_stats")
        else None,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=external_path, required=True, help="Empty report directory"
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        help="Matched unencoded reference, including native or another cache backend",
    )
    args = parser.parse_args()
    runs = [collect_run(directory) for directory in args.runs]
    if args.reference_run:
        reference = collect_run(args.reference_run)
        for run in runs:
            run["output_reference"] = compare_outputs(run, reference)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in runs:
        config = run["manifest"]["arguments"]
        for summary in run["summary"]:
            rows.append(
                {
                    "run": run["directory"],
                    "engine": config["engine"],
                    "backend": config["backend"],
                    "observation_mode": config["observation_mode"],
                    "orbitkv_transfer_backend": config.get("orbitkv_transfer_backend"),
                    "gpu_tokens": config["gpu_tokens"],
                    "host_gib": config["host_gib"],
                    "ssd_gib": config.get("ssd_gib", 0),
                    "ssd_backend": config.get("ssd_backend", "uring"),
                    "ssd_read_path": config.get("ssd_read_path"),
                    "storage_codec": config.get("storage_codec", "none"),
                    "storage_codec_budget_bytes_per_worker": config.get("storage_codec_budget"),
                    "engine_kv_bytes": run["manifest"].get("capacity", {}).get("engine_kv_bytes"),
                    "working_set_logical_bytes": run["manifest"]
                    .get("capacity", {})
                    .get("working_set_logical_bytes"),
                    "output_tokens": config.get("output_tokens"),
                    "duration_seconds": config.get("duration_seconds"),
                    "max_requests": config.get("max_requests"),
                    "working_set": config.get("working_set"),
                    "reuse_ratio": config.get("reuse_ratio"),
                    "seed": config.get("seed"),
                    "deterministic_inference": config.get("deterministic_inference"),
                    "query_budget_gib": config.get("query_budget_gib"),
                    "prefill_tokens": config.get("prefill_tokens", 8192),
                    "cache_protected_percent": config.get("cache_protected_percent", 0),
                    "ssd_write_policy": config.get("ssd_write_policy", "all"),
                    "queue_warmup": config.get("queue_warmup"),
                    "prepare_requests": config.get("prepare_requests"),
                    "read_batch_mib": config.get("read_batch_mib"),
                    "read_timeout_ms": config.get("read_timeout_ms"),
                    "read_max_batches": config.get("read_max_batches"),
                    **summary,
                    "cache_sources": json.dumps(summary["cache_sources"], sort_keys=True),
                }
            )
    (args.output / "summary.json").write_text(json.dumps(runs, indent=2, allow_nan=False) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
