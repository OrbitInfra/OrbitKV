"""Installed two-engine SSD pressure with bounded progress and native output controls."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import selectors
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from benches.common.artifacts import external_path
from benches.common.metrics import delta, metrics, percentile
from benches.common.runtime import ROOT, free_port
from benches.common.workload import generate

ENGINES = ("vllm", "sglang")
RESERVATION_PEAKS = (
    "orbitkv_query_reserved_peak_bytes",
    "orbitkv_query_instance_reserved_peak_bytes",
)
DRAIN = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_inflight_bytes",
    "orbitkv_ssd_write_inflight",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_read_pinned_bytes",
)


def write_json(path: Path, value) -> None:
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prompt(vocabulary: list[int], length: int, seed: str) -> list[int]:
    rng = random.Random(seed)
    return [rng.choice(vocabulary) for _ in range(length)]


def identity(seed: str, engine: str, index: int, working_set: int, cold_every: int) -> dict:
    cold = index % cold_every == cold_every - 1
    return {
        "engine": engine,
        "index": index,
        "kind": "cold" if cold else "reuse",
        "prefix_index": None if cold else (index - index // cold_every) % working_set,
        "input_seed": f"{seed}:{engine}:cold:{index // cold_every}" if cold else None,
    }


def matches_native(result: dict, reference: dict) -> bool:
    return result["text"] == reference["text"] and all(
        result["usage"].get(name) == reference["usage"].get(name)
        for name in ("prompt_tokens", "completion_tokens")
    )


def run_window(args, urls, vocabulary, prefixes, emit):
    started = time.monotonic()
    deadline = started + args.duration_seconds
    pending = {}
    submitted = dict.fromkeys(ENGINES, 0)
    peak = dict.fromkeys(ENGINES, 0)
    completed_at = dict.fromkeys(ENGINES, started)
    rows, failures = [], []
    stop_admission = False

    def request(engine, item, tokens):
        begun = time.monotonic() - started
        result = generate(urls[engine], engine, str(args.model), tokens, args.output_tokens)
        reference = prefixes[engine][item["prefix_index"]] if item["kind"] == "reuse" else None
        return {
            **result,
            **item,
            "tokens": tokens,
            "started_seconds": begun,
            "finished_seconds": time.monotonic() - started,
            "matches_reference_output": matches_native(result, reference)
            if reference is not None
            else None,
        }

    with ThreadPoolExecutor(max_workers=2 * args.concurrency) as executor:
        while True:
            now = time.monotonic()
            for engine in ENGINES:
                if (
                    any(item["engine"] == engine for item in pending.values())
                    and now - completed_at[engine] > args.progress_gap_seconds
                    and not stop_admission
                ):
                    failures.append({"engine": engine, "error": "progress gap exceeded"})
                    stop_admission = True
                active = sum(item["engine"] == engine for item in pending.values())
                while (
                    not stop_admission
                    and active < args.concurrency
                    and submitted[engine] < args.max_requests
                    and time.monotonic() < deadline
                ):
                    item = identity(
                        args.seed, engine, submitted[engine], args.working_set, args.cold_every
                    )
                    item["submitted_seconds"] = time.monotonic() - started
                    tokens = (
                        prompt(vocabulary, args.prompt_tokens, item["input_seed"])
                        if item["kind"] == "cold"
                        else prefixes[engine][item["prefix_index"]]["tokens"]
                    )
                    pending[executor.submit(request, engine, item, tokens)] = item
                    submitted[engine] += 1
                    active += 1
                    peak[engine] = max(peak[engine], active)
            if not pending:
                break
            done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    row = future.result()
                    rows.append(row)
                    emit(row)
                    completed_at[row["engine"]] = started + row["finished_seconds"]
                    if row["matches_reference_output"] is False:
                        failures.append({**item, "error": "native output mismatch"})
                        stop_admission = True
                except Exception as error:
                    failures.append({**item, "error": str(error)})
                    stop_admission = True
    window = {
        "wall_seconds": time.monotonic() - started,
        "submitted": submitted,
        "peak_client_inflight": peak,
        "stop_reason": "failure"
        if failures
        else "duration"
        if time.monotonic() >= deadline
        else "request_limit",
        "failures": failures,
    }
    return rows, window


def validate_window(args, rows, window, require_oracle=False):
    if window["failures"]:
        raise ValueError(f"Pressure request failed: {window['failures']}")
    if not math.isfinite(window["wall_seconds"]) or window["wall_seconds"] <= 0:
        raise ValueError("Invalid pressure window duration")
    for engine in ENGINES:
        samples = [row for row in rows if row["engine"] == engine]
        count = window["submitted"][engine]
        if (
            not 0 < count <= args.max_requests
            or len(samples) != count
            or {row["index"] for row in samples} != set(range(count))
        ):
            raise ValueError(f"Missing or duplicate {engine} requests")
        if not 0 < window["peak_client_inflight"][engine] <= args.concurrency:
            raise ValueError(f"Unbounded {engine} admission")
        for row in samples:
            for field in (
                "submitted_seconds",
                "started_seconds",
                "finished_seconds",
                "ttft_ms",
                "e2e_ms",
            ):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError(f"Invalid pressure timing: {field}")
            if not (
                row["submitted_seconds"]
                <= row["started_seconds"]
                <= row["finished_seconds"]
                <= window["wall_seconds"]
                and row["submitted_seconds"] < args.duration_seconds
                and row["ttft_ms"] <= row["e2e_ms"]
            ):
                raise ValueError("Inconsistent pressure timing")
            if (
                row["usage"].get("prompt_tokens") != args.prompt_tokens
                or row["usage"].get("completion_tokens") != args.output_tokens
            ):
                raise ValueError("Pressure token count mismatch")
            if row["matches_reference_output"] is False or (
                require_oracle and row["matches_reference_output"] is not True
            ):
                raise ValueError("Missing or mismatched native output oracle")
        completions = sorted(row["finished_seconds"] for row in samples)
        gaps = [
            completions[0],
            *(b - a for a, b in zip(completions, completions[1:], strict=False)),
        ]
        if max(gaps) > args.progress_gap_seconds:
            raise ValueError(f"{engine} exceeded its progress gap")
    if len(rows) != sum(window["submitted"].values()):
        raise ValueError("Unexpected pressure engine")
    if window["stop_reason"] == "duration":
        if window["wall_seconds"] < args.duration_seconds:
            raise ValueError("Early pressure window")
    elif window["stop_reason"] != "request_limit" or not all(
        count == args.max_requests for count in window["submitted"].values()
    ):
        raise ValueError("Invalid pressure stop reason")


def validate_resources(args, before, after, peaks):
    for name in DRAIN:
        if after.get(name) != 0:
            raise ValueError(f"Undrained or absent resource: {name}")
    staging = "orbitkv_ssd_gpu_staging_bytes"
    if any(staging in values for values in (before, after, peaks)) and after.get(staging) != 0:
        raise ValueError(f"Undrained or absent cuFile resource: {staging}")
    for name, limit in (
        ("orbitkv_query_reserved_bytes", args.query_mib * 1024**2),
        ("orbitkv_pool_used_bytes", args.host_mib * 1024**2),
    ):
        if name not in peaks or peaks[name] > limit:
            raise ValueError(f"Resource peak absent or over budget: {name}")
    for name, limit in zip(
        RESERVATION_PEAKS,
        (args.query_mib * 1024**2, args.instance_query_mib * 1024**2),
        strict=True,
    ):
        previous, peak = before.get(name), after.get(name)
        if (
            previous is None
            or peak is None
            or not math.isfinite(previous)
            or not math.isfinite(peak)
            or not 0 <= previous <= peak <= limit
            or peak < peaks.get(name, 0)
            or (
                name == RESERVATION_PEAKS[0] and peak < peaks.get("orbitkv_query_reserved_bytes", 0)
            )
        ):
            raise ValueError(f"Owner reservation peak missing, reset or over budget: {name}")
    changes = delta(before, after)
    for name in (
        "orbitkv_save_bytes_total",
        "orbitkv_load_bytes_total",
        "orbitkv_ssd_write_bytes_total",
        "orbitkv_ssd_prefetch_bytes_total",
    ):
        if changes.get(name, 0) <= 0:
            raise ValueError(f"Mixed pressure did not consume required path: {name}")
    return changes


def sample_resources(args):
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    forbidden = [name for name in ("torch", "orbitkv", "vllm", "sglang") if name in sys.modules]
    print(json.dumps({"ready": not forbidden, "forbidden_modules": forbidden}), flush=True)
    peaks, errors, count = {}, [], 0
    next_at = time.monotonic()
    try:
        with args.sample_output.open("x") as output:
            while not stopped.is_set():
                if count >= args.sample_limit:
                    raise RuntimeError("Resource sampler limit reached")
                begun = time.monotonic_ns()
                values = metrics(args.sample_url)
                count += 1
                for name, value in values.items():
                    peaks[name] = max(value, peaks.get(name, 0))
                output.write(
                    json.dumps(
                        {
                            "index": count - 1,
                            "started_ns": begun,
                            "ended_ns": time.monotonic_ns(),
                            "metrics": values,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                output.flush()
                next_at += args.sample_interval
                stopped.wait(max(0, next_at - time.monotonic()))
    except Exception as error:
        errors.append(str(error))
    write_json(
        args.sample_output.with_suffix(".summary.json"),
        {
            "samples": count,
            "peaks": peaks,
            "errors": errors,
            "forbidden_modules": forbidden,
        },
    )
    return int(bool(errors or forbidden or not count))


@contextlib.contextmanager
def resource_sampler(args, url, env):
    path = args.output / "resources.jsonl"
    command = [
        sys.executable,
        "-I",
        "-c",
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from benches.serving.shared_manager_pressure import main; raise SystemExit(main())",
        str(ROOT),
        "--sample-url",
        url,
        "--sample-output",
        str(path),
        "--sample-interval",
        str(args.sample_interval),
        "--sample-limit",
        str(math.ceil((args.duration_seconds + 120) / args.sample_interval) + 16),
    ]
    with (args.output / "sampler.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=log, text=True)
        forced = False
        primary_error = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=15):
                    raise TimeoutError("Sampler ready timeout")
                ready = json.loads(process.stdout.readline())
            write_json(args.output / "sampler-ready.json", {"command": command, **ready})
            if not ready["ready"] or ready["forbidden_modules"]:
                raise RuntimeError(f"Sampler loaded inference/native modules: {ready}")
            yield
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if process.poll() is None and not path.with_suffix(".summary.json").exists():
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True
                process.kill()
                process.wait(timeout=15)
            process.stdout.close()
            write_json(
                args.output / "sampler-cleanup.json",
                {
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "forced_kill": forced,
                    "primary_failure": {
                        "type": type(primary_error).__name__,
                        "message": str(primary_error),
                    }
                    if primary_error is not None
                    else None,
                },
            )
            if (forced or process.returncode != 0) and primary_error is None:
                raise RuntimeError("Resource sampler failed or required forced cleanup")


def run(args):
    # Only the parent imports test support. Child engines/Manager get a clean
    # installed environment; their import path never contains source orbitkv.
    sys.path.insert(0, str(ROOT / "python"))
    from transformers import AutoTokenizer

    from tests.support.installed_artifacts import isolated_environment
    from tests.support.installed_serving import (
        engine_command,
        probe_installation,
        service,
        wait_for_drain,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    args.ssd_path.mkdir(parents=True, exist_ok=False)
    interpreters = {"vllm": args.vllm_python, "sglang": args.sglang_python}
    directories, environments, installed, commands, options, urls = {}, {}, {}, {}, {}, {}
    for engine in ENGINES:
        directory = args.output / engine
        directory.mkdir()
        directories[engine] = directory
        env = isolated_environment(dict(os.environ), directory)
        env.update(
            ORBITKV_CACHE_SCOPE=args.output.name, ORBITKV_TRANSFER_BACKEND=args.transfer_backend
        )
        environments[engine] = env
        installed[engine] = probe_installation(
            interpreters[engine], engine, env, directory, "installed-before", native=True
        )
        port = free_port()
        urls[engine] = f"http://127.0.0.1:{port}"
        command, options[engine] = engine_command(
            engine, interpreters[engine], str(args.model), port, args.transfer_backend
        )
        if engine == "vllm":
            command += ["--block-size", "64", "--num-gpu-blocks-override", "64"]
        commands[engine] = command
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "A researcher measures reusable context and validates accurate cache recovery. ",
        add_special_tokens=False,
    )
    prefixes = {
        engine: [
            {"tokens": prompt(vocabulary, args.prompt_tokens, f"{args.seed}:{engine}:{i}")}
            for i in range(args.working_set)
        ]
        for engine in ENGINES
    }
    manager_port, http_port = free_port(), free_port()
    manager_url = f"http://127.0.0.1:{http_port}"
    manager = [
        str(args.vllm_python.parent / "orbitkv-cache-manager"),
        "--addr",
        f"127.0.0.1:{manager_port}",
        "--http-addr",
        f"127.0.0.1:{http_port}",
        "--pool-size",
        f"{args.host_mib}mb",
        "--query-budget",
        f"{args.query_mib}mb",
        "--query-instance-budget",
        f"{args.instance_query_mib}mb",
        "--ssd-cache-path",
        str(args.ssd_path / "cache.bin"),
        "--ssd-cache-capacity",
        f"{args.ssd_gib}gb",
        "--ssd-backend",
        "uring",
        "--ssd-read-path",
        "uring",
        "--log-level",
        "debug" if args.profile else "info",
    ]
    write_json(
        args.output / "manifest.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "engine_commands": commands,
            "cache_options": options,
            "manager_command": manager,
            "installed_before": installed,
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                text=True,
            ),
            "capacity_scope": "Owner lifetime global/single-instance query reservation maxima; sampled physical pool usage",
            "timing_scope": "Client streaming TTFT to nonempty text and request completion",
        },
    )
    rows, window = [], {}
    try:
        with (args.output / "native-controls.jsonl").open("x") as controls:

            def control(engine, tokens, kind, index):
                result = generate(urls[engine], engine, str(args.model), tokens, args.output_tokens)
                controls.write(
                    json.dumps(
                        {
                            "engine": engine,
                            "kind": kind,
                            "index": index,
                            "tokens": tokens,
                            "result": result,
                        }
                    )
                    + "\n"
                )
                controls.flush()
                return result

            with contextlib.ExitStack() as stack:
                for engine in ENGINES:
                    stack.enter_context(
                        service(
                            commands[engine],
                            urls[engine],
                            environments[engine],
                            directories[engine],
                            f"native-before-{engine}",
                        )
                    )
                for engine in ENGINES:
                    for i, prefix in enumerate(prefixes[engine]):
                        prefix.update(control(engine, prefix["tokens"], "reuse", i))

            for engine in ENGINES:
                environments[engine].update(
                    ORBITKV_PORT=str(manager_port),
                    ORBITKV_SGLANG_ENDPOINT=f"unix:///tmp/orbitkv-{manager_port}.sock",
                )
            with service(manager, manager_url, environments["vllm"], args.output, "manager"):
                with contextlib.ExitStack() as stack:
                    for engine in ENGINES:
                        stack.enter_context(
                            service(
                                commands[engine] + options[engine],
                                urls[engine],
                                environments[engine],
                                directories[engine],
                                f"cached-{engine}",
                            )
                        )
                    for engine in ENGINES:
                        for prefix in prefixes[engine]:
                            result = generate(
                                urls[engine],
                                engine,
                                str(args.model),
                                prefix["tokens"],
                                args.output_tokens,
                            )
                            if not matches_native(result, prefix):
                                raise RuntimeError(f"{engine} warm-up differed from native control")
                    before = wait_for_drain(http_port)
                    write_json(args.output / "metrics-before.json", before)
                    with resource_sampler(args, manager_url, environments["vllm"]):
                        with (args.output / "samples.jsonl").open("x") as sample_file:

                            def emit(row):
                                sample_file.write(json.dumps(row, allow_nan=False) + "\n")
                                sample_file.flush()

                            rows, window = run_window(args, urls, vocabulary, prefixes, emit)
                        write_json(args.output / "window.json", window)
                        validate_window(args, rows, window)
                        after = wait_for_drain(http_port)
                        write_json(args.output / "metrics-after.json", after)
                    observation = json.loads((args.output / "resources.summary.json").read_text())
                    changes = validate_resources(args, before, after, observation["peaks"])
                write_json(
                    args.output / "metrics-after-engine-exit.json", wait_for_drain(http_port)
                )

            with (args.output / "oracles.jsonl").open("x") as oracles:
                with contextlib.ExitStack() as stack:
                    for engine in ENGINES:
                        stack.enter_context(
                            service(
                                commands[engine],
                                urls[engine],
                                environments[engine],
                                directories[engine],
                                f"native-after-{engine}",
                            )
                        )
                    for row in rows:
                        engine = row["engine"]
                        reference = (
                            control(engine, row["tokens"], "cold", row["index"])
                            if row["kind"] == "cold"
                            else prefixes[engine][row["prefix_index"]]
                        )
                        row["matches_reference_output"] = matches_native(row, reference)
                        oracles.write(
                            json.dumps(
                                {
                                    "engine": engine,
                                    "index": row["index"],
                                    "matches_native": row["matches_reference_output"],
                                    "actual_text_sha256": hashlib.sha256(
                                        row["text"].encode()
                                    ).hexdigest(),
                                    "native_text_sha256": hashlib.sha256(
                                        reference["text"].encode()
                                    ).hexdigest(),
                                }
                            )
                            + "\n"
                        )
                        oracles.flush()
                validate_window(args, rows, window, require_oracle=True)
            for engine in ENGINES:
                after_install = probe_installation(
                    interpreters[engine],
                    engine,
                    environments[engine],
                    directories[engine],
                    "installed-after",
                    native=True,
                )
                if installed[engine]["distributions"] != after_install["distributions"]:
                    raise RuntimeError(f"{engine} installed files changed")
            summary = {}
            for engine in ENGINES:
                samples = [row for row in rows if row["engine"] == engine]
                completions = sorted(row["finished_seconds"] for row in samples)
                summary[engine] = {
                    "requests": len(samples),
                    "cold_requests": sum(r["kind"] == "cold" for r in samples),
                    "requests_per_second": len(samples) / window["wall_seconds"],
                    "max_completion_gap_seconds": max(
                        [
                            completions[0],
                            *(b - a for a, b in zip(completions, completions[1:], strict=False)),
                        ]
                    ),
                    "native_output_mismatches": 0,
                    "tail_scope": "descriptive"
                    if len(samples) < 1000
                    else "local run, independent pairs required",
                    **{
                        f"{field}_p{p}": percentile([r[field] for r in samples], p / 100)
                        for field in ("ttft_ms", "e2e_ms")
                        for p in (50, 95, 99)
                    },
                }
            write_json(
                args.output / "result.json",
                {
                    "state": "LOCAL_GATE_PASSED",
                    "independent_acceptance": False,
                    "engines": summary,
                    "manager_delta": changes,
                    "sampled_peaks": observation["peaks"],
                    "owner_reservation_peaks": {name: after[name] for name in RESERVATION_PEAKS},
                    "owner_peak_scope": "Since Manager budget creation, including warm-up; maximum across all instances, no instance labels",
                    "resource_samples": observation["samples"],
                },
            )
            print(json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(
            args.output / "failure.json", {"error": str(error), "type": type(error).__name__}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-url")
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--vllm-python", type=lambda p: Path(p).absolute())
    parser.add_argument("--sglang-python", type=lambda p: Path(p).absolute())
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=external_path)
    parser.add_argument("--ssd-path", type=external_path)
    parser.add_argument("--transfer-backend", choices=("direct", "kernel"), default="direct")
    parser.add_argument("--duration-seconds", type=float, default=300)
    parser.add_argument("--progress-gap-seconds", type=float, default=30)
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--working-set", type=int, default=16)
    parser.add_argument("--cold-every", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=768)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--host-mib", type=int, default=512)
    parser.add_argument("--query-mib", type=int, default=384)
    parser.add_argument("--instance-query-mib", type=int, default=192)
    parser.add_argument("--ssd-gib", type=int, default=8)
    parser.add_argument("--seed", default="pair-01")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (
        any(
            not math.isfinite(value) or value <= 0
            for value in (
                args.duration_seconds,
                args.progress_gap_seconds,
                args.sample_interval,
            )
        )
        or any(
            value <= 0
            for value in (
                args.max_requests,
                args.concurrency,
                args.working_set,
                args.prompt_tokens,
                args.output_tokens,
                args.host_mib,
                args.query_mib,
                args.instance_query_mib,
                args.ssd_gib,
                args.sample_limit,
            )
        )
        or args.cold_every < 2
    ):
        parser.error("durations and budgets must be positive; cold-every must be at least two")
    if args.sample_url:
        if not args.sample_output:
            parser.error("sample-output is required")
        return sample_resources(args)
    if not all((args.vllm_python, args.sglang_python, args.model, args.output, args.ssd_path)):
        parser.error("installed interpreters, model, output and fresh SSD path are required")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
