"""Explicit engine/backend launch configurations for matched cache budgets."""

from __future__ import annotations

import importlib.machinery
import json
import os
import re
import sys
import sysconfig
from argparse import ArgumentTypeError, Namespace
from dataclasses import dataclass
from pathlib import Path

from benches.common.runtime import ROOT, free_port

STORAGE_CODECS = ("none", "ans", "fp8", "turboquant-4", "turboquant-3")


def storage_codec_budget(value: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?", value.strip().lower())
    if match:
        amount, unit = match.groups()
        size = float(amount) * {None: 1, "b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}[unit]
        if 4096 <= size < 1024**3 * 4:
            return int(size)
    raise ArgumentTypeError("codec budget must be 4 KiB to 4 GiB - 1, in bytes or kb/mb/gb")


@dataclass(frozen=True)
class Launch:
    command: list[str]
    env: dict[str, str]
    base_url: str
    manager_command: list[str] | None
    manager_url: str | None
    manager_health_path: str
    backend_configuration: dict


def configure(args: Namespace, bytes_per_token: int) -> Launch:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    for variable, directory in (
        ("VLLM_CACHE_ROOT", "vllm-cache"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor-cache"),
        ("TRITON_CACHE_DIR", "triton-cache"),
        ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
        ("FLASHINFER_WORKSPACE_BASE", "flashinfer-cache"),
    ):
        env[variable] = str(args.output / directory)
    env.update(PYTHONHASHSEED="0", VLLM_LOG_STATS_INTERVAL="1")
    if args.engine == "vllm":
        env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    env.pop("VLLM_BATCH_INVARIANT", None)
    if args.engine == "vllm" and args.deterministic_inference:
        env["VLLM_BATCH_INVARIANT"] = "1"
    env.pop("ORBITKV_TRANSFER_BACKEND", None)
    site_paths = [
        path for path in sys.path if Path(path).name in {"site-packages", "dist-packages"}
    ]
    package_dir = ROOT / "python/orbitkv"
    if args.installed_artifact:
        if args.backend == "orbitkv":
            spec = importlib.machinery.PathFinder.find_spec("orbitkv", site_paths)
            if spec is None or spec.origin is None:
                raise RuntimeError("Installed benchmark requires the complete OrbitKV wheel")
            package_dir = Path(spec.origin).resolve().parent
            if package_dir.parent not in {Path(path).resolve() for path in site_paths}:
                raise RuntimeError(
                    f"Installed benchmark resolved OrbitKV outside site-packages: {package_dir}"
                )
            if not (package_dir / "orbitkv-cache-manager-py").is_file():
                raise RuntimeError("Installed OrbitKV wheel does not contain its Cache Manager")
            site_paths = [str(package_dir.parent), *site_paths]
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*site_paths, str(args.output)]))
    else:
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(args.output), *site_paths])
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    manager_url = None
    manager_health_path = "/health"
    manager_command = None
    backend_configuration = {}
    cache_port = None
    cache_config = None
    if args.backend == "orbitkv":
        manager_port = free_port()
        manager_http = free_port()
        manager_url = f"http://127.0.0.1:{manager_http}"
        env.update(
            ORBITKV_PORT=str(manager_port),
            ORBITKV_SGLANG_ENDPOINT=f"unix:///tmp/orbitkv-{manager_port}.sock",
            PYO3_PYTHON=sys.executable,
            PYTHONHOME=sys.base_prefix,
            ORBITKV_QUEUE_WARMUP="1" if args.queue_warmup == "on" else "0",
            ORBITKV_PREPARE_REQUESTS="1" if args.prepare_requests == "on" else "0",
            ORBITKV_TRACE_TRANSFERS="1" if args.trace_transfers else "0",
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [sysconfig.get_config_var("LIBDIR"), env.get("LD_LIBRARY_PATH", "")]
        )
        manager_command = [
            env.get(
                "ORBITKV_CACHE_MANAGER_BINARY",
                str(package_dir / "orbitkv-cache-manager-py"),
            ),
            "--addr",
            f"127.0.0.1:{manager_port}",
            "--http-addr",
            f"127.0.0.1:{manager_http}",
            "--pool-size",
            f"{args.host_gib}gb",
            "--enable-prometheus",
            "--storage-codec",
            args.storage_codec,
            "--storage-codec-budget",
            str(args.storage_codec_budget),
        ]
        backend_configuration.update(
            artifact_mode="installed" if args.installed_artifact else "source",
            orbitkv_package_dir=str(package_dir),
            orbitkv_transfer_backend=args.orbitkv_transfer_backend,
            storage_codec=args.storage_codec,
            storage_codec_budget_bytes_per_worker=args.storage_codec_budget,
        )
        if args.engine == "sglang" and args.orbitkv_transfer_backend is not None:
            env["ORBITKV_TRANSFER_BACKEND"] = args.orbitkv_transfer_backend
        if args.ssd_gib:
            manager_command += [
                "--ssd-cache-path",
                str(args.ssd_path),
                "--ssd-cache-capacity",
                f"{args.ssd_gib}gb",
            ]
            backend_configuration.update(
                ssd_gib=args.ssd_gib,
                io="O_DIRECT",
                ssd_backend=args.ssd_backend,
                ssd_read_path=args.ssd_read_path,
            )
            manager_command += [
                "--ssd-write-policy",
                args.ssd_write_policy,
                "--ssd-backend",
                args.ssd_backend,
            ]
            if args.ssd_read_path is not None:
                manager_command += ["--ssd-read-path", args.ssd_read_path]
        manager_command += ["--cache-protected-percent", str(args.cache_protected_percent)]
        backend_configuration["cache_protected_percent"] = args.cache_protected_percent
        backend_configuration["ssd_write_policy"] = args.ssd_write_policy
        if args.query_budget_gib is not None:
            manager_command += ["--query-budget", str(int(args.query_budget_gib * 1024**3))]
            backend_configuration["query_budget_bytes"] = int(args.query_budget_gib * 1024**3)
        backend_configuration["queue_warmup"] = args.queue_warmup
        backend_configuration["trace_transfers"] = args.trace_transfers
        backend_configuration["prepare_requests"] = args.prepare_requests
        for flag, value in (
            ("--query-read-batch", args.read_batch_mib * 1024**2),
            ("--query-read-timeout-ms", args.read_timeout_ms),
            ("--query-read-max-batches", args.read_max_batches),
        ):
            if value:
                manager_command += [flag, str(value)]
        backend_configuration["read_controls"] = {
            "batch_mib": args.read_batch_mib,
            "timeout_ms": args.read_timeout_ms,
            "max_batches": args.read_max_batches,
        }
        if not args.installed_artifact:
            plugin = args.output / "orbitkv_benchmark-0.0.dist-info"
            plugin.mkdir()
            (plugin / "METADATA").write_text("Name: orbitkv-benchmark\nVersion: 0.0\n")
            (plugin / "entry_points.txt").write_text(
                "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n"
            )
    elif args.backend == "lmcache":
        env["LMCACHE_TRACK_USAGE"] = "false"
        cache_port, cache_http = free_port(), free_port()
        manager_url = f"http://127.0.0.1:{cache_http}"
        manager_health_path = "/healthcheck"
        manager_command = [
            sys.executable,
            "-m",
            "lmcache.cli.main",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(cache_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(cache_http),
            "--l1-size-gb",
            str(args.host_gib),
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            "64",
        ]
        if args.engine == "sglang":
            backend_configuration = {
                "chunk_size": 64,
                "mp_host": "127.0.0.1",
                "mp_port": cache_port,
            }
            cache_config = args.output / "lmcache.json"
            cache_config.write_text(json.dumps(backend_configuration, indent=2))
    elif args.backend == "flexkv":
        backend_configuration = {
            "FLEXKV_CPU_CACHE_GB": str(args.host_gib),
            "FLEXKV_SSD_CACHE_GB": "0",
            "FLEXKV_ENABLE_GDS": "0",
            "FLEXKV_ENABLE_MPS": "0",
            "FLEXKV_ENABLE_METRICS": "0",
            "FLEXKV_SERVER_RECV_PORT": f"ipc://{args.output}/flexkv.sock",
        }
        env.pop("FLEXKV_CONFIG_PATH", None)
        env.update(backend_configuration)
    command = (
        vllm_command(args, port, bytes_per_token, cache_port)
        if args.engine == "vllm"
        else sglang_command(args, port, cache_config)
    )
    backend_configuration["execution_mode"] = "eager" if args.enforce_eager else "engine-default"
    return Launch(
        command,
        env,
        base_url,
        manager_command,
        manager_url,
        manager_health_path,
        backend_configuration,
    )


def vllm_command(
    args: Namespace, port: int, bytes_per_token: int, cache_port: int | None
) -> list[str]:
    pressure_tokens = (
        max(args.lengths) if args.workload == "sustained" else args.gpu_tokens * 3 // 4
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--block-size",
        "64",
        "--enable-prefix-caching",
        "--kv-cache-memory-bytes",
        str(bytes_per_token * args.gpu_tokens),
        "--max-model-len",
        str(pressure_tokens + 64),
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        str(args.prefill_tokens),
        "--generation-config",
        "vllm",
        "--seed",
        "42",
        "--shutdown-timeout",
        "30",
        "--enable-prompt-tokens-details",
    ]
    if args.deterministic_inference:
        command += ["--attention-backend", "FLASH_ATTN"]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.backend == "cpu":
        connector = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": args.host_gib * 1024**3,
                "block_size": 64,
            },
        }
    elif args.backend == "orbitkv":
        connector = {
            "kv_connector": "OrbitKVConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "orbitkv.vllm",
        }
        if args.orbitkv_transfer_backend:
            connector["kv_connector_extra_config"] = {
                "orbitkv.transfer_backend": args.orbitkv_transfer_backend
            }
    elif args.backend == "lmcache":
        connector = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "127.0.0.1",
                "lmcache.mp.port": cache_port,
            },
        }
    elif args.backend == "flexkv":
        connector = {"kv_connector": "FlexKVConnectorV1", "kv_role": "kv_both"}
    if args.backend != "native":
        command += ["--kv-transfer-config", json.dumps(connector)]
    return command


def sglang_command(args: Namespace, port: int, cache_config: Path | None) -> list[str]:
    pressure_tokens = (
        max(args.lengths) if args.workload == "sustained" else args.gpu_tokens * 3 // 4
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--nccl-port",
        str(free_port()),
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--page-size",
        "64",
        "--max-total-tokens",
        str(args.gpu_tokens),
        "--context-length",
        str(pressure_tokens + 64),
        "--max-running-requests",
        "8",
        "--cuda-graph-max-bs-decode",
        "8",
        "--cuda-graph-max-bs-prefill",
        "8",
        "--chunked-prefill-size",
        str(args.prefill_tokens),
        "--random-seed",
        "42",
        "--enable-cache-report",
        "--enable-metrics",
    ]
    if args.deterministic_inference:
        command.append("--enable-deterministic-inference")
    if args.enforce_eager:
        command += [
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
    if args.backend == "cpu":
        command += [
            "--enable-hierarchical-cache",
            "--hicache-size",
            str(args.host_gib),
            "--hicache-write-policy",
            "write_through",
        ]
    elif args.backend == "orbitkv":
        command += [
            "--radix-cache-backend",
            "orbitkv",
            "--enable-unified-cache-external-linker",
        ]
    elif args.backend == "lmcache":
        command += ["--enable-lmcache", "--lmcache-config-file", str(cache_config)]
    elif args.backend == "flexkv":
        command += ["--enable-flexkv"]
    return command
