"""Measure cold prefill, HBM hits, and external-cache reuse after HBM pressure.

Run from the repository root with the selected engine's Python environment:
python -m benches.single_node --engine sglang --backend orbitkv --model /path/to/model --output /tmp/orbitkv-single-node
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from . import concurrent, sustained
from .artifacts import external_path
from .launch import STORAGE_CODECS, configure, storage_codec_budget
from .metrics import codec_summary, delta, metrics, summarize
from .runtime import manifest, process_usage, server, storage_manifest
from .workload import run_workload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument(
        "--backend",
        choices=["native", "cpu", "orbitkv", "lmcache", "flexkv"],
        required=True,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable engine Graph execution explicitly for matched eager controls",
    )
    parser.add_argument(
        "--installed-artifact",
        action="store_true",
        help="Use installed packages and the wheel's Manager; reject a source OrbitKV import",
    )
    parser.add_argument(
        "--output",
        type=external_path,
        required=True,
        help="Empty result directory outside the source checkout",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--workload", choices=["serial", "concurrent", "sustained"], default="serial"
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60,
        help="Sustained admission window per concurrency; admitted requests drain afterward",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10000,
        help="Maximum sustained requests per window, bounding retained samples",
    )
    parser.add_argument(
        "--working-set",
        type=int,
        default=12,
        help="Number of independently prepared prefixes in a sustained window",
    )
    parser.add_argument(
        "--reuse-ratio",
        type=float,
        default=0.75,
        help="Probability of requesting a prepared prefix; other requests are cold",
    )
    parser.add_argument(
        "--query-budget-gib", type=float, help="OrbitKV global query ownership budget"
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--gpu-tokens", type=int, default=16384)
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=8192,
        help="Engine prefill token batch limit; independent of total GPU KV capacity",
    )
    parser.add_argument("--host-gib", type=int, default=16)
    parser.add_argument("--storage-codec", choices=STORAGE_CODECS, default="none")
    parser.add_argument(
        "--storage-codec-budget",
        type=storage_codec_budget,
        default=64 * 1024**2,
        help="GPU codec scratch per worker, in bytes or kb/mb/gb (default: 64mb)",
    )
    parser.add_argument(
        "--ssd-dir",
        type=Path,
        help="Existing SSD mount directory; use a private temporary subdirectory for the cache",
    )
    parser.add_argument(
        "--ssd-gib",
        type=int,
        default=0,
        help="OrbitKV SSD capacity; sustained traffic naturally evicts DRAM, other workloads add a forced-eviction phase",
    )
    parser.add_argument(
        "--orbitkv-transfer-backend",
        choices=["direct", "kernel"],
        help="Fix the OrbitKV H2D/D2H backend; default: vLLM model choice or SGLang direct",
    )
    parser.add_argument("--cache-protected-percent", type=int, default=0)
    parser.add_argument("--ssd-write-policy", choices=["all", "reuse"], default="all")
    parser.add_argument("--ssd-backend", choices=["auto", "uring", "cufile"], default="uring")
    parser.add_argument(
        "--ssd-read-path",
        choices=["uring", "cufile"],
        help="Override SSD restores independently of backend initialization and writes",
    )
    parser.add_argument(
        "--gds-stats",
        type=Path,
        help="NVIDIA gds_stats executable; require native read/write evidence before stopping the Manager",
    )
    parser.add_argument("--queue-warmup", choices=["on", "off"], default="off")
    parser.add_argument("--prepare-requests", choices=["on", "off"], default="off")
    parser.add_argument("--read-batch-mib", type=int, default=0)
    parser.add_argument("--read-timeout-ms", type=int, default=0)
    parser.add_argument("--read-max-batches", type=int, default=0)
    parser.add_argument("--trace-transfers", action="store_true")
    parser.add_argument(
        "--deterministic-inference",
        action="store_true",
        help="Enable the engine's deterministic kernels for output-parity qualification",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--settle-seconds", type=float, default=1.2)
    args = parser.parse_args()
    if args.backend != "orbitkv" and (
        args.storage_codec != "none" or args.storage_codec_budget != 64 * 1024**2 or args.ssd_dir
    ):
        parser.error("storage codec and SSD directory controls require --backend orbitkv")
    if (args.ssd_backend != "uring" or args.ssd_read_path or args.gds_stats) and (
        args.backend != "orbitkv" or not args.ssd_gib
    ):
        parser.error("GPU storage controls require --backend orbitkv and --ssd-gib")
    if args.gds_stats and args.ssd_backend not in ("auto", "cufile"):
        parser.error("--gds-stats requires --ssd-backend auto or cufile")
    if args.ssd_read_path == "cufile" and args.ssd_backend == "uring":
        parser.error("--ssd-read-path cufile requires --ssd-backend auto or cufile")
    if args.gds_stats and args.ssd_read_path == "uring":
        parser.error("--gds-stats requires native reads; --ssd-read-path uring is a host control")
    if args.ssd_read_path is not None and (
        args.queue_warmup == "on" or args.prepare_requests == "on"
    ):
        parser.error(
            "explicit --ssd-read-path measurement requires --queue-warmup off and "
            "--prepare-requests off; aggregate counters cannot separate demand and preparation reads"
        )
    if not 0 <= args.cache_protected_percent <= 100:
        parser.error("--cache-protected-percent must be between 0 and 100")
    if args.backend != "orbitkv" and (
        args.cache_protected_percent or args.ssd_write_policy != "all"
    ):
        parser.error("cache policies require --backend orbitkv")
    if args.ssd_write_policy != "all" and not args.ssd_gib:
        parser.error("--ssd-write-policy reuse requires --ssd-gib")
    if min(args.read_batch_mib, args.read_timeout_ms, args.read_max_batches) < 0:
        parser.error("read controls must be nonnegative")
    if args.prepare_requests == "on" and args.queue_warmup == "on":
        parser.error("compare prepared ownership and unowned warming in separate runs")
    if args.backend != "orbitkv" and (
        args.prepare_requests == "on"
        or args.read_batch_mib
        or args.read_timeout_ms
        or args.read_max_batches
    ):
        parser.error("request preparation and read controls require --backend orbitkv")
    if args.read_max_batches and not args.read_batch_mib:
        parser.error("--read-max-batches requires --read-batch-mib")
    if args.backend != "orbitkv" and (args.queue_warmup == "on" or args.trace_transfers):
        parser.error("--queue-warmup on and --trace-transfers require --backend orbitkv")
    args.model = args.model.resolve()
    if args.orbitkv_transfer_backend and args.backend != "orbitkv":
        parser.error("--orbitkv-transfer-backend requires --backend orbitkv")
    if args.ssd_gib < 0 or (args.ssd_gib and args.backend != "orbitkv"):
        parser.error("--ssd-gib must be nonnegative and requires --backend orbitkv")
    if args.query_budget_gib is not None and (
        args.backend != "orbitkv" or not 0 < args.query_budget_gib <= args.host_gib
    ):
        parser.error("--query-budget-gib requires OrbitKV and 0 < budget <= host capacity")
    if (
        not args.concurrencies
        or len(set(args.concurrencies)) != len(args.concurrencies)
        or any(c not in (1, 4, 8) for c in args.concurrencies)
    ):
        parser.error("--concurrencies must be distinct values from 1, 4, 8")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty so measurements cannot mix across runs")
    if (
        args.repeats < 1
        or len(set(args.lengths)) != len(args.lengths)
        or min(args.lengths) < 64
        or max(args.lengths) >= args.gpu_tokens * 3 // 4
    ):
        parser.error(
            "use positive repeats and distinct lengths between 64 and 3/4 of GPU token capacity"
        )
    if args.gpu_tokens % 64 or args.host_gib <= 0 or not 1 <= args.output_tokens <= 64:
        parser.error(
            "use page-aligned GPU capacity, positive host capacity, and 1–64 output tokens"
        )
    if args.prefill_tokens < 64 or args.prefill_tokens % 64:
        parser.error("--prefill-tokens must be a positive multiple of 64")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("--settle-seconds must be finite and nonnegative")
    if args.workload == "sustained":
        if not math.isfinite(args.duration_seconds) or args.duration_seconds < 1:
            parser.error("--duration-seconds must be finite and at least 1")
        if not 1 <= args.max_requests <= 100000 or not 1 <= args.working_set <= 1024:
            parser.error("use 1–100000 max requests and 1–1024 working-set prefixes")
        if not math.isfinite(args.reuse_ratio) or not 0 <= args.reuse_ratio <= 1:
            parser.error("--reuse-ratio must be finite and between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        parser.error("this fixed-capacity benchmark currently targets dense Qwen3")
    bytes_per_token = (
        2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * 2
    )

    ssd_directory = None
    args.ssd_path = args.output / "cache.bin"
    try:
        if args.ssd_gib and args.ssd_dir:
            args.ssd_dir = args.ssd_dir.resolve(strict=True)
            ssd_directory = Path(tempfile.mkdtemp(prefix="orbitkv-bench-", dir=args.ssd_dir))
            args.ssd_path = ssd_directory / "cache.bin"
        launch = configure(args, bytes_per_token)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest(args, launch, bytes_per_token), indent=2, allow_nan=False) + "\n"
        )
        with contextlib.ExitStack() as stack:
            if launch.manager_command:
                manager = stack.enter_context(
                    server(
                        launch.manager_command,
                        launch.env,
                        launch.manager_url,
                        args.output / "manager.log",
                        launch.manager_health_path,
                    )
                )
                if args.ssd_gib:
                    (args.output / "storage.json").write_text(
                        json.dumps(storage_manifest(manager.pid, args.ssd_path), indent=2) + "\n"
                    )
            stack.enter_context(
                server(
                    launch.command,
                    launch.env,
                    launch.base_url,
                    args.output / "engine.log",
                    engine=args.engine,
                )
            )
            if launch.manager_command:
                usage_before = process_usage(manager.pid)
                workload_manager_before = metrics(launch.manager_url)
                workload_started = time.monotonic()
            if args.workload in ("concurrent", "sustained"):
                workload = sustained if args.workload == "sustained" else concurrent
                samples, batches = workload.run_workload(args, launch.base_url, launch.manager_url)
                workload.validate(vars(args), samples, batches)
                summary = workload.summarize(samples, batches)
            else:
                samples = run_workload(args, launch.base_url, launch.manager_url)
                summary = summarize(samples, args.lengths)
            if launch.manager_command:
                usage_after = process_usage(manager.pid)
                workload_manager_after = metrics(launch.manager_url)
                workload_measurement = {
                    "manager_before": workload_manager_before,
                    "manager_after": workload_manager_after,
                    "manager_delta": delta(workload_manager_before, workload_manager_after),
                }
                (args.output / "manager-usage.json").write_text(
                    json.dumps(
                        {
                            "workload_seconds": time.monotonic() - workload_started,
                            "delta": {
                                key: value - usage_before[key] for key, value in usage_after.items()
                            },
                            "scope": "Manager during the whole workload, including warmup/pressure; excludes engine CPU",
                            "io_note": "Linux process I/O accounting is not a GPU DMA byte counter; use cuFile metrics too",
                            **workload_measurement,
                            "codec": codec_summary([workload_measurement]),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            if args.trace_transfers and args.backend == "orbitkv":
                from .timeline import collect

                collect(args.output, samples)
            if args.gds_stats:
                from .gds import native_io_stats

                result = subprocess.run(
                    [str(args.gds_stats), "-p", str(manager.pid), "-l", "3"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                (args.output / "gds-stats.txt").write_text(result.stdout + result.stderr)
                result.check_returncode()
                native = native_io_stats(result.stdout)
                native["backend_fallbacks"] = metrics(launch.manager_url).get(
                    "orbitkv_ssd_backend_fallbacks_total", 0
                )
                (args.output / "native-io.json").write_text(json.dumps(native, indent=2) + "\n")
                if native["backend_fallbacks"]:
                    raise RuntimeError("Native qualification rejected automatic backend fallback")
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n"
        )
        raise
    finally:
        if args.ssd_gib:
            args.ssd_path.unlink(missing_ok=True)
        if ssd_directory:
            shutil.rmtree(ssd_directory)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
