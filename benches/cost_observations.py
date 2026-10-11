"""Paired observation overhead or fixed transfer-backend serving comparisons.

Run from the repository root. This harness never builds native artifacts. Raw
runs and logs remain under runs/; final/ contains only the reviewed-comparison
shape, including missing evidence and failed gates. No dynamic-policy claim is
made by either same-binary comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

from .artifacts import external_path
from .metrics import OBSERVATION_MODES, engine_itl_summary, percentile
from .report import collect_run
from .runtime import ROOT

BUDGETS = {
    "requests_per_second": 3.0,
    "ttft_p50_ms": 3.0,
    "ttft_p95_ms": 5.0,
    "ttft_p99_ms": 5.0,
}
COMPARISONS = {
    "observations": ("off", "on"),
    "transfer-backends": ("direct", "kernel"),
}
MATCHED_ARGUMENTS = (
    "engine",
    "backend",
    "model",
    "workload",
    "lengths",
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
    "ssd_backend",
    "ssd_read_path",
    "ssd_dir",
    "orbitkv_transfer_backend",
    "storage_codec",
    "storage_codec_budget",
    "query_budget_gib",
    "queue_warmup",
    "prepare_requests",
    "trace_transfers",
    "seed",
    "read_batch_mib",
    "read_timeout_ms",
    "read_max_batches",
    "cache_protected_percent",
    "ssd_write_policy",
    "settle_seconds",
    "observation_mode",
)


def plan(args) -> list[dict]:
    jobs = []
    modes = COMPARISONS[args.comparison]
    for engine in args.engines:
        for tier in args.tiers:
            for codec in args.storage_codecs:
                if codec != "none" and tier not in args.encoded_tiers:
                    continue
                for repetition in range(1, args.pairs + 1):
                    for mode in modes if repetition % 2 else reversed(modes):
                        name = f"{engine}-{tier}-{codec}-pair-{repetition}-{mode}"
                        arguments = {
                            "engine": engine,
                            "backend": "orbitkv",
                            "model": str(args.model),
                            "workload": "sustained",
                            "lengths": [1024, 4096],
                            "concurrencies": [4],
                            "gpu_tokens": 8192,
                            "prefill_tokens": 4096,
                            "output_tokens": 16,
                            "host_gib": 16 if tier == "dram" else 1,
                            "ssd_gib": 0 if tier == "dram" else 16,
                            "query_budget_gib": 0.75,
                            "ssd_backend": "uring",
                            "ssd_read_path": None,
                            "working_set": 12,
                            "reuse_ratio": 0.75,
                            "duration_seconds": 3600,
                            "max_requests": args.requests,
                            "queue_warmup": "off",
                            "prepare_requests": "off",
                            "storage_codec": codec,
                            "storage_codec_budget": 64 * 1024**2,
                            "read_batch_mib": 0,
                            "read_timeout_ms": 0,
                            "read_max_batches": 0,
                            "cache_protected_percent": 0,
                            "ssd_write_policy": "all",
                            "trace_transfers": False,
                            "orbitkv_transfer_backend": mode
                            if args.comparison == "transfer-backends"
                            else None,
                            "settle_seconds": 1.2,
                            "observation_mode": args.observation_mode,
                            "seed": args.seed,
                            "ssd_dir": str(args.ssd_dir) if tier == "ssd" else None,
                            "output": str(args.output / "runs" / name),
                        }
                        command = [
                            str(ROOT / ".venv" / f"{engine}-release" / "bin/python"),
                            "-m",
                            "benches.single_node",
                        ]
                        for key, value in arguments.items():
                            if value is not None and value is not False:
                                command.extend(
                                    (
                                        f"--{key.replace('_', '-')}",
                                        *(
                                            str(v)
                                            for v in (value if isinstance(value, list) else [value])
                                        ),
                                    )
                                )
                        jobs.append(
                            {
                                "engine": engine,
                                "tier": tier,
                                "codec": codec,
                                "pair": repetition,
                                "mode": mode,
                                "name": name,
                                "command": command,
                                "arguments": arguments,
                                "manager": str(args.manager),
                            }
                        )
    return jobs


def preflight(args) -> list[str]:
    failures = []
    for path in (
        args.manager,
        args.model / "config.json",
        args.ssd_dir,
        *(ROOT / ".venv" / f"{engine}-release" / "bin/python" for engine in args.engines),
    ):
        if not path.exists():
            failures.append(f"Missing prerequisite: {path}")
    if "ans" in args.storage_codecs:
        library = os.environ.get("ORBITKV_NVCOMP_LIBRARY")
        if not library or not Path(library).is_file():
            failures.append("ANS control requires ORBITKV_NVCOMP_LIBRARY pointing to libnvcomp")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode:
            failures.append(f"GPU unavailable: {(result.stdout + result.stderr).strip()}")
    except (OSError, subprocess.TimeoutExpired) as error:
        failures.append(f"GPU prerequisite failed: {error}")
    # Cargo may restage Mooncake even without an active Manager. Refuse competing
    # builds and serving runtimes; never kill a process owned by another task.
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = path.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(path.parent.name) == os.getpid() or not words:
            continue
        executable = Path(os.fsdecode(words[0])).name
        serving = any(
            word in (b"vllm.entrypoints.openai.api_server", b"sglang.launch_server")
            for word in words
        )
        if (
            executable in ("cargo", "rustc", "orbitkv-cache-manager", "orbitkv-cache-manager-py")
            or serving
        ):
            failures.append(f"Competing build/runtime process: pid={path.parent.name} {executable}")
    return failures


def read_run(directory: Path, mode: str, tier: str, slo: dict) -> dict:
    if mode not in ("off", "on", "direct", "kernel"):
        raise ValueError(f"Unknown comparison mode: {mode}")
    process = json.loads((directory / "process.json").read_text())
    if process.get("exit_code") != 0:
        raise ValueError(f"Serving process did not exit successfully: {process}")
    run = collect_run(directory)
    manifest, rows = run["manifest"], run["summary"]
    expected = "1" if mode == "on" else "0"
    if manifest.get("cost_observations") != expected:
        raise ValueError("Run manifest does not prove the requested instrumentation mode")
    if mode in COMPARISONS["transfer-backends"]:
        if manifest["arguments"].get("orbitkv_transfer_backend") != mode:
            raise ValueError("Run manifest does not prove the requested transfer backend")
        backends = re.findall(
            r"GPU worker initialized: device=\d+ backend=(direct|kernel)\b",
            (directory / "manager.log").read_text(),
        )
        if not backends or set(backends) != {mode}:
            raise ValueError("Manager workers did not use only the requested transfer backend")
    if len(rows) != 1 or rows[0].get("stop_reason") != "request_limit":
        raise ValueError("Paired overhead requires one complete fixed-request sustained window")
    (window,) = [
        json.loads(line) for line in (directory / "windows.jsonl").read_text().splitlines()
    ]
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    counters = window["manager_delta"]
    failures = []
    for name in (
        "orbitkv_load_failures_total",
        "orbitkv_storage_codec_decode_failures_total",
        "orbitkv_ssd_write_failures_total",
    ):
        if counters.get(name, 0) != 0:
            failures.append(f"Unexpected execution failures: {name}={counters[name]}")
    if counters.get("orbitkv_load_bytes_total", 0) <= 0:
        failures.append("No measured GPU restore bytes")
    if (
        mode in COMPARISONS["transfer-backends"]
        and counters.get("orbitkv_save_bytes_total", 0) <= 0
    ):
        failures.append("No measured GPU save bytes for D2H/H2D backend comparison")
    if tier == "ssd":
        for name in ("orbitkv_ssd_prefetch_bytes_total", "orbitkv_ssd_write_bytes_total"):
            if counters.get(name, 0) <= 0:
                failures.append(f"No measured SSD evidence: {name}")
        if run["storage"].get("direct_io") is not True:
            failures.append("SSD run has no O_DIRECT evidence")
    cost = {key: value for key, value in counters.items() if key.startswith("orbitkv_cost_")}
    prediction_count = cost.get("orbitkv_cost_prediction_absolute_error_seconds_count", 0)
    shadow_applicable = manifest["arguments"].get("storage_codec", "none") == "none"
    if mode == "on" and prediction_count <= 0:
        failures.append("No executed-work prediction-error samples")
    if (
        mode == "on"
        and shadow_applicable
        and cost.get("orbitkv_cost_shadow_decisions_total", 0) <= 0
    ):
        failures.append("No real-candidate shadow decisions")
    if mode != "on" and any(value != 0 for value in cost.values()):
        failures.append("Disabled observations still produced cost samples")
    if manifest["arguments"].get("storage_codec", "none") != "none":
        for operation in ("encode", "decode"):
            name = f"orbitkv_storage_codec_batches_total_{operation}"
            if counters.get(name, 0) <= 0:
                failures.append(f"No measured encoded work: {name}")
    averages = [
        (row["e2e_ms"] - row["ttft_ms"]) / max(1, row["usage"]["completion_tokens"] - 1)
        for row in samples
    ]
    good = sum(
        sample["ttft_ms"] <= slo["ttft_ms"] and average <= slo["average_decode_ms_per_token"]
        for sample, average in zip(samples, averages, strict=True)
    )
    if rows[0]["ttft_p95_ms"] > slo["ttft_ms"]:
        failures.append("TTFT p95 exceeds the predeclared SLO")
    if percentile(averages, 0.95) > slo["average_decode_ms_per_token"]:
        failures.append("Response-average decode p95 exceeds the predeclared SLO")
    itl = engine_itl_summary(
        manifest["arguments"]["engine"], window.get("metrics_delta", {}), slo["itl_ms"]
    )
    if itl["status"] != "observed" or itl["within_slo_fraction"] is None:
        failures.append("No complete engine ITL histogram with the predeclared SLO boundary")
    elif itl["within_slo_fraction"] < 0.95:
        failures.append("Engine ITL p95 exceeds the predeclared SLO")
    evidence = {
        **rows[0],
        "goodput_requests_per_second": good / window["wall_seconds"],
        "slo_passing_requests": good,
        "average_decode_ms_per_token_p95": percentile(averages, 0.95),
        "average_decode_ms_per_token_p99": percentile(averages, 0.99),
        "prediction_absolute_error_seconds_mean": (
            cost.get("orbitkv_cost_prediction_absolute_error_seconds_sum", 0) / prediction_count
            if prediction_count
            else None
        ),
        "cost": cost,
        "engine_itl": itl,
        "shadow_scope": (
            "Disabled during fixed transfer-backend comparison"
            if mode in COMPARISONS["transfer-backends"]
            else (
                "Raw direct/kernel candidate observations"
                if shadow_applicable
                else "Not applicable: encoded composite paths have no matched shadow alternatives in P4.1"
            )
        ),
        "manager_usage": {
            key: run["manager_usage"][key]
            for key in ("workload_seconds", "delta", "scope", "io_note")
            if key in run["manager_usage"]
        }
        if run["manager_usage"]
        else None,
        "failures": failures,
    }
    return {
        "manifest": manifest,
        "samples": samples,
        "evidence": evidence,
        "storage": run["storage"],
    }


def compare_pair(off: dict, on: dict, comparison: str = "observations") -> dict:
    control_mode, candidate_mode = COMPARISONS[comparison]
    if comparison == "transfer-backends":
        for run, mode in ((off, control_mode), (on, candidate_mode)):
            manifest = run["manifest"]
            if (
                manifest.get("cost_observations") != "0"
                or manifest["arguments"].get("orbitkv_transfer_backend") != mode
                or manifest["arguments"].get("storage_codec", "none") != "none"
            ):
                raise ValueError(
                    "Backend comparison requires explicit direct/kernel, raw storage and observations off"
                )
    for key in MATCHED_ARGUMENTS:
        if comparison == "transfer-backends" and key == "orbitkv_transfer_backend":
            continue
        if off["manifest"]["arguments"].get(key) != on["manifest"]["arguments"].get(key):
            raise ValueError(f"Unmatched run argument: {key}")
    for key in ("manager_binary_sha256", "model_revision", "packages", "gpu", "kv_bytes_per_token"):
        if not off["manifest"].get(key) or off["manifest"].get(key) != on["manifest"].get(key):
            raise ValueError(f"Missing or unmatched run artifact: {key}")
    for key in ("storage_environment", "library_path", "python", "git_commit"):
        if off["manifest"].get(key) != on["manifest"].get(key):
            raise ValueError(f"Unmatched runtime/source environment: {key}")
    if off["manifest"]["arguments"].get("ssd_gib", 0):
        for key in ("mount", "capacity_bytes"):
            if off["storage"].get(key) is None or off["storage"].get(key) != on["storage"].get(key):
                raise ValueError(f"Unmatched SSD storage: {key}")
    control = {sample["index"]: sample for sample in off["samples"]}
    candidate = {sample["index"]: sample for sample in on["samples"]}
    if control.keys() != candidate.keys():
        raise ValueError("Unmatched fixed request cohort")
    for index, sample in candidate.items():
        if not sample.get("prompt_sha256") or sample["prompt_sha256"] != control[index].get(
            "prompt_sha256"
        ):
            raise ValueError(f"Unverified or unmatched prompt: {index}")
    overhead = {}
    for key in BUDGETS:
        before, after = off["evidence"][key], on["evidence"][key]
        if not all(math.isfinite(value) and value > 0 for value in (before, after)):
            raise ValueError(f"Invalid paired measurement: {key}")
        overhead[key] = 100 * (
            1 - after / before if key == "requests_per_second" else after / before - 1
        )
    return {
        "overhead_percent": overhead,
        "compared_requests": len(control),
        "output_mismatches": sum(candidate[i]["text"] != control[i]["text"] for i in control),
        control_mode: off["evidence"],
        candidate_mode: on["evidence"],
    }


def validate_planned_run(run: dict, job: dict) -> None:
    for key, expected in job["arguments"].items():
        if run["manifest"]["arguments"].get(key) != expected:
            raise ValueError(f"Run differs from predeclared {key}: expected {expected!r}")
    command = run["manifest"].get("manager_command", [])
    if not command or command[0] != job["manager"]:
        raise ValueError("Run did not use the predeclared Manager artifact")
    count = job["arguments"]["max_requests"]
    if len(run["samples"]) != count or {row["index"] for row in run["samples"]} != set(
        range(count)
    ):
        raise ValueError("Run did not complete the predeclared unique request cohort")
    if any(not row.get("prompt_sha256") for row in run["samples"]):
        raise ValueError("Predeclared request cohort has unverified prompt hashes")


def summarize(args, jobs: list[dict], slo: dict) -> dict:
    cells = []
    modes = COMPARISONS[args.comparison]
    planned = {job["name"]: job for job in jobs}
    for engine, tier, codec in dict.fromkeys((j["engine"], j["tier"], j["codec"]) for j in jobs):
        pairs, failures = [], []
        for repetition in range(1, args.pairs + 1):
            try:
                runs = {
                    mode: read_run(
                        args.output / "runs" / f"{engine}-{tier}-{codec}-pair-{repetition}-{mode}",
                        mode,
                        tier,
                        slo,
                    )
                    for mode in modes
                }
                for mode, run in runs.items():
                    validate_planned_run(
                        run, planned[f"{engine}-{tier}-{codec}-pair-{repetition}-{mode}"]
                    )
                pair = compare_pair(runs[modes[0]], runs[modes[1]], args.comparison)
                pair["pair"] = repetition
                pairs.append(pair)
                failures.extend(
                    f"pair {repetition} {mode}: {failure}"
                    for mode in modes
                    for failure in pair[mode]["failures"]
                )
            except (OSError, ValueError, KeyError) as error:
                failures.append(f"pair {repetition}: {error}")
        overhead = (
            {
                key: statistics.median(pair["overhead_percent"][key] for pair in pairs)
                for key in BUDGETS
            }
            if len(pairs) == args.pairs
            else {}
        )
        failures.extend(
            f"{key} overhead {value:.3f}% exceeds {BUDGETS[key]}% budget"
            for key, value in overhead.items()
            if value > BUDGETS[key]
        )
        cells.append(
            {
                "engine": engine,
                "tier": tier,
                "codec": codec,
                "status": "failed" if failures else "passed",
                "paired_median_overhead_percent": overhead,
                "failures": failures,
                "pairs": pairs,
            }
        )
    return {
        "comparison": args.comparison,
        "control": modes[0],
        "candidate": modes[1],
        "status": "passed"
        if cells and all(cell["status"] == "passed" for cell in cells)
        else "failed",
        "budgets_percent": BUDGETS,
        "slo": slo,
        "cells": cells,
        "scope": (
            "Finite matched cohorts; per-pair ratios then median, not pooled percentiles. Fixed kernel versus direct registration affects both D2H saves and H2D restores. Observations are disabled on both sides; the existing 25ms metrics sampler is unchanged. Fresh Managers do not accumulate both candidates in one estimator. This is not a restore-only microbenchmark, dynamic-policy qualification or native GDS/RDMA evidence."
            if args.comparison == "transfer-backends"
            else "Finite matched cohorts; per-pair ratios then median, not pooled percentiles. End-to-end overhead with the existing 25ms Manager metrics sampling, including extra metric serialization and parsing; not isolated observer hot-path cost. Same-host io_uring only; no native GDS/RDMA or dynamic-policy qualification."
        ),
        "artifacts": "Checks the same Manager SHA-256, package versions and model revision. Python adapters, native shared libraries and benchmark source must remain frozen while the matrix runs; documentation edits are allowed.",
        "itl": "Official engine histogram bucket deltas provide approximate p50/p95/p99 and a p95 SLO at an exact bucket boundary. vLLM engine-core timing and SGLang tokenizer-receipt timing have different boundaries. Goodput uses only request TTFT and response-average decode, never assigns aggregate ITL to requests.",
        "output": "Exact text differences are retained diagnostics; concurrent greedy output is not assumed batch invariant. Run engine correctness gates separately.",
        "contention": "Manager process CPU/I/O include preparation. GPU contention and engine CPU are not measured by this harness.",
    }


def write_summary(directory: Path, result: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (directory / "summary.csv").open("w") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "comparison",
                "control",
                "candidate",
                "engine",
                "tier",
                "codec",
                "status",
                *BUDGETS,
            ],
        )
        writer.writeheader()
        for cell in result.get("cells", []):
            writer.writerow(
                {
                    **{key: result[key] for key in ("comparison", "control", "candidate")},
                    **{key: cell[key] for key in ("engine", "tier", "codec", "status")},
                    **cell["paired_median_overhead_percent"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", choices=COMPARISONS, default="observations")
    parser.add_argument("--tiers", nargs="+", choices=("dram", "ssd"), default=["dram", "ssd"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manager", type=Path, required=True, help="Prebuilt Manager; never rebuilt here"
    )
    parser.add_argument("--ssd-dir", type=Path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    parser.add_argument(
        "--engines", nargs="+", choices=("vllm", "sglang"), default=["vllm", "sglang"]
    )
    parser.add_argument(
        "--storage-codecs", nargs="+", choices=("none", "ans"), default=["none", "ans"]
    )
    parser.add_argument(
        "--encoded-tiers",
        nargs="+",
        choices=("dram", "ssd"),
        default=["ssd"],
        help="Representative encoded controls (default SSD); raw runs cover both tiers",
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default="sampled")
    parser.add_argument("--ttft-slo-ms", type=float, default=2000)
    parser.add_argument("--average-decode-slo-ms", type=float, default=100)
    parser.add_argument(
        "--itl-slo-ms",
        type=float,
        default=100,
        help="Engine ITL p95 SLO; must match an official histogram bucket",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.comparison == "transfer-backends" and args.storage_codecs != ["none"]:
        parser.error("Transfer-backend comparison requires --storage-codecs none")
    if args.pairs < 3 or not 32 <= args.requests <= 100000:
        parser.error("Use at least three pairs and 32–100000 fixed requests")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.ttft_slo_ms, args.average_decode_slo_ms, args.itl_slo_ms)
    ):
        parser.error("SLO thresholds must be finite and positive")
    if any(
        len(set(values)) != len(values)
        for values in (args.engines, args.tiers, args.storage_codecs, args.encoded_tiers)
    ):
        parser.error("Engine and codec lists must not contain duplicates")
    for name in ("model", "manager", "ssd_dir", "output"):
        setattr(args, name, getattr(args, name).resolve())
    jobs = plan(args)
    slo = {
        "ttft_ms": args.ttft_slo_ms,
        "average_decode_ms_per_token": args.average_decode_slo_ms,
        "itl_ms": args.itl_slo_ms,
    }
    if args.report_only:
        saved = json.loads((args.output / "plan.json").read_text())
        if saved != {"jobs": jobs, "slo": slo, "budgets_percent": BUDGETS}:
            parser.error("Report arguments must match the predeclared plan")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "plan.json").write_text(
            json.dumps({"jobs": jobs, "slo": slo, "budgets_percent": BUDGETS}, indent=2) + "\n"
        )
        failures = preflight(args)
        if failures or args.preflight_only:
            result = {
                "comparison": args.comparison,
                "control": COMPARISONS[args.comparison][0],
                "candidate": COMPARISONS[args.comparison][1],
                "status": "blocked" if failures else "ready",
                "failures": failures,
                "cells": [],
                "slo": slo,
                "budgets_percent": BUDGETS,
            }
            write_summary(args.output / "final", result)
            print(json.dumps(result, indent=2))
            raise SystemExit(1 if failures else 0)
        started = time.monotonic()
        for index, job in enumerate(jobs, 1):
            env = dict(os.environ)
            cache = args.output / "runs" / "compiler-cache" / job["engine"]
            env.update(
                {
                    "ORBITKV_CACHE_MANAGER_BINARY": str(args.manager),
                    "ORBITKV_COST_OBSERVATIONS": "1" if job["mode"] == "on" else "0",
                    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                    "TRITON_CACHE_DIR": str(cache / "triton"),
                    "VLLM_CACHE_ROOT": str(cache / "vllm"),
                }
            )
            log = args.output / "runs" / f"{job['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"Running {index}/{len(jobs)} {job['name']}", flush=True)
            job_started = time.monotonic()
            with (
                log.open("w") as output,
                subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                ) as process,
            ):
                try:
                    process.wait()
                except KeyboardInterrupt:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            directory = args.output / "runs" / job["name"]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "process.json").write_text(
                json.dumps(
                    {
                        "exit_code": process.returncode,
                        "duration_seconds": time.monotonic() - job_started,
                    },
                    indent=2,
                )
                + "\n"
            )
            try:
                evidence = read_run(
                    args.output / "runs" / job["name"], job["mode"], job["tier"], slo
                )
                validate_planned_run(evidence, job)
                failures = evidence["evidence"]["failures"]
            except (OSError, ValueError, KeyError) as error:
                failures = [str(error)]
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(jobs),
                        "name": job["name"],
                        "exit_code": process.returncode,
                        "run_seconds": round(time.monotonic() - job_started, 2),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "evidence_failures": failures,
                    }
                ),
                flush=True,
            )
    result = summarize(args, jobs, slo)
    write_summary(args.output / "final", result)
    print(
        json.dumps(
            {"status": result["status"], "summary": str(args.output / "final/summary.json")},
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
