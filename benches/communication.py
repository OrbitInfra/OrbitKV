"""Measure real local Query and GPU Restore with independently built artifacts.

Select the Python package with PYTHONPATH and its matching Manager with --manager.
This harness never builds artifacts or switches implementations at runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import sysconfig
import time
from pathlib import Path

import requests

from .artifacts import external_path
from .metrics import metrics
from .runtime import free_port, process_usage, server, storage_manifest


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manager", type=Path, default=os.environ.get("ORBITKV_CACHE_MANAGER_BINARY")
    )
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--tier",
        choices=("dram", "ssd"),
        default="dram",
        help="ssd measures forced io_uring restores only, with physical-read and GPU-byte controls",
    )
    parser.add_argument("--output", type=external_path, required=True)
    parser.add_argument(
        "--ssd-cache-path",
        type=external_path,
        help="fresh SSD data path on the tested mount, independent of retained output",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, nargs="+", default=[4096, 262144, 4194304])
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--layout",
        choices=("contiguous", "split"),
        default="contiguous",
        help="GPU layout; split places equally sized K/V halves in separate block arrays",
    )
    parser.add_argument(
        "--restore-batch-size",
        type=int,
        default=1,
        help="independent query leases submitted together for the same total restore payload",
    )
    parser.add_argument("--idle-ms", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--idle-seconds", type=float, default=2.0)
    parser.add_argument("--pool-mib", type=int, default=256)
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA ordinal visible to this process"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.manager is None:
        parser.error("provide --manager or ORBITKV_CACHE_MANAGER_BINARY")
    if args.iterations < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("iterations/repeats must be positive and warmup nonnegative")
    if args.block_bytes < 1 or args.pool_mib < 1 or args.layers < 1 or args.device < 0:
        parser.error("block-bytes/pool-mib/layers must be positive and device nonnegative")
    if args.restore_batch_size < 1:
        parser.error("restore-batch-size must be positive")
    if args.layout == "split" and args.block_bytes % 2:
        parser.error("split layout requires an even block-bytes value for equal K/V halves")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds <= 0:
        parser.error("idle-seconds must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in args.idle_ms):
        parser.error("idle-ms values must be finite and nonnegative")
    if any(size < 1 or size % args.block_bytes for size in args.payload_bytes):
        parser.error("payload-bytes values must be positive multiples of block-bytes")
    if max(args.payload_bytes) // args.block_bytes > 1024:
        parser.error("at most 1024 blocks per operation; increase block-bytes for larger payloads")
    if any(
        size // args.block_bytes < args.restore_batch_size
        or (size // args.block_bytes) % args.restore_batch_size
        for size in args.payload_bytes
    ):
        parser.error(
            "each payload's block count must be at least and divisible by restore-batch-size"
        )
    if max(args.payload_bytes) * args.layers * 2 > args.pool_mib * 1024**2:
        parser.error("pool must hold at least twice the largest payload across all layers")
    if len(set(args.idle_ms)) != len(args.idle_ms) or len(set(args.payload_bytes)) != len(
        args.payload_bytes
    ):
        parser.error("payload-bytes and idle-ms must not contain duplicates")
    if args.tier == "ssd" and args.block_bytes // (2 if args.layout == "split" else 1) % 512:
        parser.error("SSD segment bytes must be multiples of 512 for O_DIRECT")
    if args.ssd_cache_path is not None:
        if args.tier != "ssd":
            parser.error("ssd-cache-path requires --tier ssd")
        if args.ssd_cache_path.exists():
            parser.error("ssd-cache-path must be fresh; existing data will not be overwritten")
    args.manager = args.manager.resolve()
    return args


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("measurements must be finite, nonnegative, and nonempty")
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def summarize(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("no measured samples")
    timings = [key for key in samples[0] if key.endswith("_ns")]
    return {
        **{
            key.removesuffix("_ns") + "_us": distribution([row[key] / 1000 for row in samples])
            for key in timings
        },
        "query_calls": distribution([row["query_calls"] for row in samples]),
    }


def ensure_quiet() -> None:
    """A competing native build can replace mapped libraries as well as skew CPU."""
    competitors = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = path.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        executable = Path(os.fsdecode(words[0])).name
        if executable in {
            "cargo",
            "rustc",
            "orbitkv-cache-manager",
            "orbitkv-cache-manager-py",
        } or any(
            word in {b"vllm.entrypoints.openai.api_server", b"sglang.launch_server"}
            for word in words
        ):
            competitors.append(f"{path.parent.name}:{executable}")
    if competitors:
        raise RuntimeError(
            "Stop competing builds/runtimes before measuring: " + ", ".join(competitors)
        )


def manager_environment(package_root: Path) -> dict[str, str]:
    env = dict(os.environ, PYO3_PYTHON=sys.executable, PYTHONHOME=sys.base_prefix)
    # Follow the package actually imported by the caller, including external
    # revision snapshots. Never insert this checkout's ROOT/python.
    python_paths = [
        str(package_root),
        *env.get("PYTHONPATH", "").split(os.pathsep),
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in python_paths if path))
    library_paths = [sysconfig.get_config_var("LIBDIR"), env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(path for path in library_paths if path)
    env.setdefault("RUST_LOG", "warn")
    return env


def query_ready(client, native, instance, hashes, request, expected_blocks, timeout) -> tuple:
    deadline = time.monotonic() + timeout
    calls = 0
    while True:
        result = client.query_prefetch(instance, hashes, request)
        calls += 1
        if isinstance(result, native.QueryReady):
            if result.num_hit_blocks != expected_blocks or bool(result.lease) != bool(
                expected_blocks
            ):
                if result.lease:
                    client.release(result.lease)
                raise AssertionError(
                    f"query {request}: expected {expected_blocks} blocks, got {result.num_hit_blocks}"
                )
            return result, calls
        if not isinstance(result, native.QueryLoading) or not result.admitted:
            raise AssertionError(f"query {request} was not admitted: {result!r}")
        if time.monotonic() >= deadline:
            client.cancel_query(instance, request)
            raise TimeoutError(f"query {request} did not become ready")
        # No artificial query-poll sleep: measure the real submit-to-ready path
        # and retain its API call count. This is not an isolated one-RPC timer.


def sample(
    client,
    registration,
    native,
    instance,
    device,
    layers,
    hashes,
    targets,
    restore_batches,
    operation,
    request,
    idle,
    timeout,
):
    import torch

    query_calls = 0
    if operation == "restore":
        loads = []
        try:
            for index, (batch_hashes, batch_targets) in enumerate(restore_batches):
                ready, calls = query_ready(
                    client,
                    native,
                    instance,
                    batch_hashes,
                    f"{request}-lease-{index}",
                    len(batch_targets),
                    timeout,
                )
                query_calls += calls
                loads.append((ready.lease, [batch_targets]))
        except BaseException:
            # No restore was submitted. Retire leases acquired before a failed
            # query; session teardown handles any failed cleanup RPC.
            for lease, _ in loads:
                with contextlib.suppress(Exception):
                    client.release(lease)
            raise
    elif operation == "restore_empty":
        loads = []
    elif operation == "publish":
        # Unique keys force real D2H. Repeated publication of resident keys is
        # filtered before GPU copy and cannot measure the Publish data path.
        fresh = [
            hashlib.sha256(f"{request}-{index}".encode()).digest() for index in range(len(targets))
        ]
        sources = list(range(len(targets)))
        saves = [(layer, sources, fresh) for layer in layers]
    # For restore this follows lease acquisition, so the Manager can actually
    # become idle before the measured submission instead of being woken by Query.
    if idle:
        time.sleep(idle)
    thread_start = time.thread_time_ns()
    process_start = time.process_time_ns()
    started = time.perf_counter_ns()
    if operation in {"restore", "restore_empty"}:
        handle = client.start_restore(
            instance,
            0,
            device,
            [layers],
            loads,
            ready_stream=torch.cuda.current_stream().cuda_stream,
        )
        submitted = time.perf_counter_ns()
        thread_submitted = time.thread_time_ns()
        process_submitted = time.process_time_ns()
        status = client.wait_restore(handle, timeout=timeout)
        ended = time.perf_counter_ns()
        process_ended = time.process_time_ns()
        thread_ended = time.thread_time_ns()
        if not status.done or not status.success:
            raise AssertionError(f"restore was not successful: {status.message}")
        phases = {
            "submit_wall_ns": submitted - started,
            "ready_wait_wall_ns": ended - submitted,
            "submit_thread_cpu_ns": thread_submitted - thread_start,
            "ready_wait_thread_cpu_ns": thread_ended - thread_submitted,
            "submit_process_cpu_ns": process_submitted - process_start,
            "ready_wait_process_cpu_ns": process_ended - process_submitted,
        }
    elif operation == "publish":
        ok, message = client.save(registration, saves)
        ended = time.perf_counter_ns()
        process_ended = time.process_time_ns()
        thread_ended = time.thread_time_ns()
        if not ok:
            raise AssertionError(f"Publish failed: {message}")
        phases = {}
    else:
        ready, query_calls = query_ready(
            client,
            native,
            instance,
            hashes,
            request,
            0 if operation == "query_miss" else len(targets),
            timeout,
        )
        ended = time.perf_counter_ns()
        process_ended = time.process_time_ns()
        thread_ended = time.thread_time_ns()
        if ready.lease:
            client.release(ready.lease)
        phases = {}
    # A failed/ambiguous restore is never converted into a release(lease) or a
    # destination reuse. The caller keeps tensors alive through Manager teardown.
    return {
        "wall_ns": ended - started,
        "thread_cpu_ns": thread_ended - thread_start,
        "process_cpu_ns": process_ended - process_start,
        "query_calls": query_calls,
        **phases,
    }


def verify_bytes(torch, tensor, expected, start, count) -> None:
    actual = tensor[:, :, start : start + count].cpu()
    if not torch.equal(actual, expected[:, :, :count]):
        raise AssertionError("GPU restore bytes differ from the independently retained source")


def cleanup_published(manager_url: str, timeout: float, *, wait_for_ssd: bool = False) -> None:
    # Complete sealing before eviction. Both calls are outside sample latency;
    # whole-cohort CPU explicitly includes this bounded-cache maintenance.
    response = requests.post(f"{manager_url}/cache/sync", timeout=timeout)
    response.raise_for_status()
    if wait_for_ssd:
        # /cache/sync seals the DRAM publication, not asynchronous SSD writes.
        # Wait outside the restore timer while write owners still retain pages.
        deadline = time.monotonic() + timeout
        while True:
            observed = metrics(manager_url)
            if observed.get("orbitkv_ssd_write_bytes_total", 0) > 0 and all(
                observed.get(name, 0) == 0
                for name in (
                    "orbitkv_ssd_write_inflight",
                    "orbitkv_ssd_write_queue_pending",
                    "orbitkv_query_reserved_bytes",
                )
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"SSD publication ownership did not drain: {observed}")
            time.sleep(0.001)
    response = requests.post(f"{manager_url}/cache/memory/cleanup", timeout=timeout)
    response.raise_for_status()
    if response.json()["still_referenced_blocks"]:
        raise AssertionError("Publish cleanup still has referenced blocks")


def populate(client, registration, native, instance, device, layers, hashes, timeout) -> None:
    ok, message = client.save(
        registration, [(layer, list(range(len(hashes))), hashes) for layer in layers]
    )
    if not ok:
        raise RuntimeError(f"Publish failed: {message}")
    deadline = time.monotonic() + timeout
    demand = native.BlockHashes(hashes)
    while True:
        ready = client.query_prefetch(instance, demand, "populate")
        if isinstance(ready, native.QueryReady):
            if ready.lease:
                client.release(ready.lease)
            if ready.num_hit_blocks == len(hashes):
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("published payload never became fully resident")
        time.sleep(0.001)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    if not args.manager.is_file() or not os.access(args.manager, os.X_OK):
        raise FileNotFoundError(f"Manager is not an executable file: {args.manager}")
    ensure_quiet()
    import torch

    native = importlib.import_module("orbitkv.orbitkv")
    gpu = importlib.import_module("orbitkv.client.gpu")
    package = importlib.import_module("orbitkv")
    package_root = Path(package.__file__).resolve().parent.parent
    artifacts = [args.manager, Path(native.__file__).resolve(), Path(gpu.__file__).resolve()]
    artifact_hashes = {str(path): digest(path) for path in artifacts}
    if not torch.cuda.is_available():
        raise RuntimeError("one CUDA GPU and a CUDA-enabled torch are required")
    torch.cuda.set_device(args.device)
    device = gpu.resolve_device_id()
    env = manager_environment(package_root)
    port, http = free_port(), free_port()
    while http == port:
        http = free_port()
    budget = max(16 * 1024**2, max(args.payload_bytes) * args.layers * 2)
    command = [
        str(args.manager),
        "--addr",
        f"127.0.0.1:{port}",
        "--http-addr",
        f"127.0.0.1:{http}",
        "--pool-size",
        str(args.pool_mib * 1024**2),
        "--query-budget",
        str(budget),
        "--query-instance-budget",
        str(budget),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    cache_path = args.ssd_cache_path or args.output / "cache.bin"
    if args.tier == "ssd":
        command += [
            "--ssd-backend",
            "uring",
            "--ssd-read-path",
            "uring",
            "--ssd-cache-path",
            str(cache_path),
            "--ssd-cache-capacity",
            str(max(64 * 1024**2, max(args.payload_bytes) * args.layers * 2)),
        ]
    manifest = {
        "label": args.label,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "artifacts_sha256": artifact_hashes,
        "harness_sha256": digest(Path(__file__)),
        "python": sys.version,
        "python_executable": sys.executable,
        "package_root": str(package_root),
        "orbitkv_version": native.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "gpu": str(torch.cuda.get_device_properties(args.device)),
        "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "manager_command": command,
        "manager_environment": {
            key: env.get(key)
            for key in (
                "PYO3_PYTHON",
                "PYTHONHOME",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "RUST_LOG",
                "ORBITKV_COST_OBSERVATIONS",
                "ORBITKV_TRACE_TRANSFERS",
                "CUDA_MPS_PIPE_DIRECTORY",
                "CUDA_MPS_LOG_DIRECTORY",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            )
        },
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        "manager_cpu_resolution_seconds": 1 / os.sysconf("SC_CLK_TCK"),
        "percentiles": "linear interpolation at (n - 1) * q; computed independently per cohort",
        "scope": {
            "query": "submit through QueryReady, including native polling; release is outside latency",
            "restore": "one start_restore through native wait_restore ready; equal, nonoverlapping query leases partition the unchanged total payload; hash views and target chunks are precomputed, all lease acquisition is outside latency; SSD mode forces io_uring with zero DRAM before each sample, fills destinations with a sentinel and checks GPU bytes after every sample; the client timer includes Python/native call overhead, allocation, validation and the selected H2D backend",
            "publish": "save acknowledgement including metadata chunking and actual D2H of fresh keys; sealing synchronization and cache eviction are outside sample latency",
            "cpu": "sample CPU is caller thread/process; Manager cohort CPU includes preparation queries, releases, Publish sealing/cleanup HTTP, Python loop gaps and prescribed idle intervals, excludes warmup and byte validation; /proc tick-quantized totals are not precise per-RPC CPU measurements",
            "payload": "raw uint8 layers with contiguous or split K/V GPU storage, 4 KiB logical blocks by default; split K/V each holds half of block-bytes, preserving total payload; --payload-bytes is per layer, output payload_bytes is total across layers; Query reports logical payload and 32-byte hash count, not transported KV bytes",
            "ready": "client observation of terminal GPU transfer evidence, not an isolated CUDA duration or completion-delivery interval",
            "instrumentation": "clock-read overhead retained; no percentile subtraction or cross-process clock subtraction",
        },
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "status.json", {"state": "running"})
    # Keep these tensor owners in this frame until the server context has exited,
    # including timeout/error paths where the Manager may still own GPU DMA.
    pages = None
    expected = None
    client = None
    cohorts = []
    try:
        with server(
            command, env, f"http://127.0.0.1:{http}", args.output / "manager.log"
        ) as manager:
            if args.tier == "ssd":
                manifest["storage"] = storage_manifest(manager.pid, cache_path)
            manifest["manager_pid"] = manager.pid
            manifest["manager_cpu_affinity"] = sorted(os.sched_getaffinity(manager.pid))
            write_json(args.output / "manifest.json", manifest)
            client = native.CacheManagerClient(f"/tmp/orbitkv-{port}.sock")
            instance = f"communication-{os.getpid()}"
            count = max(args.payload_bytes) // args.block_bytes
            segments = 2 if args.layout == "split" else 1
            segment_bytes = args.block_bytes // segments
            layers = [f"layer-{index}" for index in range(args.layers)]
            expected = torch.empty((args.layers, segments, count, segment_bytes), dtype=torch.uint8)
            for layer in range(args.layers):
                for segment in range(segments):
                    expected[layer, segment].copy_(
                        (
                            torch.arange(count * segment_bytes, dtype=torch.int64)
                            + layer * 17
                            + segment * 71
                        )
                        .remainder_(251)
                        .to(torch.uint8)
                        .reshape(count, segment_bytes)
                    )
            pages = torch.full(
                (args.layers, segments, count * 2, segment_bytes),
                253,
                dtype=torch.uint8,
                device=f"cuda:{args.device}",
            )
            pages[:, :, :count].copy_(expected)
            torch.cuda.synchronize()
            client.start_session_watcher(instance, instance, 1, 1)
            registration = client.register_context_batch(
                instance,
                instance,
                0,
                0,
                1,
                1,
                device,
                layers,
                [gpu.serialize_gpu_buffer(pages[index]) for index in range(args.layers)],
                [count * 2] * args.layers,
                [segment_bytes] * args.layers,
                [count * 2 * segment_bytes if segments == 2 else 0] * args.layers,
                [segments] * args.layers,
                "direct",
                False,
                tensors=[pages[index] for index in range(args.layers)],
            )
            hashes = [
                hashlib.sha256(f"communication-hit-{index}".encode()).digest()
                for index in range(count)
            ]
            misses = [
                hashlib.sha256(f"communication-miss-{index}".encode()).digest()
                for index in range(count)
            ]
            # Publish seals asynchronously: establish full residency before any
            # correctness or timed query demands an exact hit count.
            populate(client, registration, native, instance, device, layers, hashes, args.timeout)
            if args.tier == "ssd":
                cleanup_published(
                    f"http://127.0.0.1:{http}", args.timeout, wait_for_ssd=args.tier == "ssd"
                )
                if metrics(f"http://127.0.0.1:{http}").get("orbitkv_cache_resident_bytes", 0):
                    raise AssertionError("SSD validation requires empty Manager DRAM")
            batches = {}
            for size in args.payload_bytes:
                blocks = size // args.block_bytes
                targets = list(range(count, count + blocks))
                hit_hashes = native.BlockHashes(hashes[:blocks])
                chunk = blocks // args.restore_batch_size
                restore_batches = [
                    (hit_hashes[start : start + chunk], targets[start : start + chunk])
                    for start in range(0, blocks, chunk)
                ]
                batches[size] = (
                    hit_hashes,
                    native.BlockHashes(misses[:blocks]),
                    targets,
                    restore_batches,
                )
                pages[:, :, count : count + blocks].fill_(253)
                torch.cuda.synchronize()
                sample(
                    client,
                    registration,
                    native,
                    instance,
                    device,
                    layers,
                    hit_hashes,
                    targets,
                    restore_batches,
                    "restore",
                    f"verify-{size}",
                    0,
                    args.timeout,
                )
                verify_bytes(torch, pages, expected, count, blocks)
            batches[0] = (native.BlockHashes([]), native.BlockHashes([]), [], [])
            manifest["gpu_byte_validation"] = {
                "passed": True,
                "bytes_per_layer": args.payload_bytes,
                "layers": args.layers,
                "layout": args.layout,
                "restore_batch_size": args.restore_batch_size,
            }
            idle_before = process_usage(manager.pid)
            idle_started = time.perf_counter_ns()
            time.sleep(args.idle_seconds)
            idle_wall = (time.perf_counter_ns() - idle_started) / 1e9
            idle_after = process_usage(manager.pid)
            idle_cpu = idle_after["cpu_seconds"] - idle_before["cpu_seconds"]
            manifest["idle_window"] = {
                "wall_seconds": idle_wall,
                "manager_cpu_seconds": idle_cpu,
                "manager_cpu_cores": idle_cpu / idle_wall,
                "scope": "all Manager threads after GPU validation; no harness API calls during sleep; existing session watcher remains connected",
            }
            write_json(args.output / "manifest.json", manifest)
            cases = [
                (operation, size, idle)
                for operation in (
                    ("restore",)
                    if args.tier == "ssd"
                    else ("query_miss", "query_hit", "restore", "publish")
                )
                for size in args.payload_bytes
                for idle in args.idle_ms
            ]
            if args.tier == "dram":
                cases += [("restore_empty", 0, idle) for idle in args.idle_ms]
            with (args.output / "samples.jsonl").open("w") as raw:
                for repeat in range(args.repeats):
                    for case_index, (operation, size, idle_ms) in enumerate(
                        cases if repeat % 2 == 0 else reversed(cases)
                    ):
                        hit_hashes, miss_hashes, targets, restore_batches = batches[size]
                        demand = miss_hashes if operation == "query_miss" else hit_hashes
                        prefix = f"r{repeat}-c{case_index}"
                        if operation in {"query_hit", "restore"}:
                            populate(
                                client,
                                registration,
                                native,
                                instance,
                                device,
                                layers,
                                hashes,
                                args.timeout,
                            )
                        if operation == "publish":
                            cleanup_published(
                                f"http://127.0.0.1:{http}",
                                args.timeout,
                                wait_for_ssd=args.tier == "ssd",
                            )
                        for index in range(args.warmup):
                            if args.tier == "ssd":
                                cleanup_published(
                                    f"http://127.0.0.1:{http}",
                                    args.timeout,
                                    wait_for_ssd=args.tier == "ssd",
                                )
                                if metrics(f"http://127.0.0.1:{http}").get(
                                    "orbitkv_cache_resident_bytes", 0
                                ):
                                    raise AssertionError("SSD warmup was masked by DRAM")
                                pages[:, :, count : count + len(targets)].fill_(253)
                                torch.cuda.synchronize()
                            sample(
                                client,
                                registration,
                                native,
                                instance,
                                device,
                                layers,
                                demand,
                                targets,
                                restore_batches,
                                operation,
                                f"{prefix}-warm-{index}",
                                idle_ms / 1000,
                                args.timeout,
                            )
                            if operation == "publish":
                                cleanup_published(
                                    f"http://127.0.0.1:{http}",
                                    args.timeout,
                                    wait_for_ssd=args.tier == "ssd",
                                )
                        if operation == "restore":
                            pages[:, :, count : count + len(targets)].fill_(253)
                            torch.cuda.synchronize()
                        counter = {
                            "restore": "orbitkv_load_bytes_total",
                            "publish": "orbitkv_save_bytes_total",
                        }.get(operation)
                        bytes_before = (
                            metrics(f"http://127.0.0.1:{http}").get(counter, 0) if counter else 0
                        )
                        ssd_before = (
                            metrics(f"http://127.0.0.1:{http}") if args.tier == "ssd" else {}
                        )
                        if args.tier == "ssd":
                            write_json(args.output / f"{prefix}-ssd-before.json", ssd_before)
                        before = process_usage(manager.pid)
                        cohort_cpu = time.process_time_ns()
                        cohort_started = time.perf_counter_ns()
                        rows = []
                        for index in range(args.iterations):
                            if args.tier == "ssd":
                                cleanup_published(
                                    f"http://127.0.0.1:{http}",
                                    args.timeout,
                                    wait_for_ssd=args.tier == "ssd",
                                )
                                if metrics(f"http://127.0.0.1:{http}").get(
                                    "orbitkv_cache_resident_bytes", 0
                                ):
                                    raise AssertionError("SSD sample was masked by DRAM")
                                pages[:, :, count : count + len(targets)].fill_(253)
                                torch.cuda.synchronize()
                            rows.append(
                                sample(
                                    client,
                                    registration,
                                    native,
                                    instance,
                                    device,
                                    layers,
                                    demand,
                                    targets,
                                    restore_batches,
                                    operation,
                                    f"{prefix}-sample-{index}",
                                    idle_ms / 1000,
                                    args.timeout,
                                )
                            )
                            if args.tier == "ssd":
                                verify_bytes(torch, pages, expected, count, len(targets))
                            if operation == "publish":
                                cleanup_published(
                                    f"http://127.0.0.1:{http}",
                                    args.timeout,
                                    wait_for_ssd=args.tier == "ssd",
                                )
                        elapsed = time.perf_counter_ns() - cohort_started
                        cohort_cpu = time.process_time_ns() - cohort_cpu
                        after = process_usage(manager.pid)
                        copied_bytes = 0
                        if counter:
                            copied_bytes = (
                                metrics(f"http://127.0.0.1:{http}").get(counter, 0) - bytes_before
                            )
                            expected_bytes = args.iterations * size * args.layers
                            if copied_bytes != expected_bytes:
                                raise AssertionError(
                                    f"{operation} transferred {copied_bytes} bytes, expected {expected_bytes}"
                                )
                        if operation == "restore":
                            verify_bytes(torch, pages, expected, count, len(targets))
                        ssd_observed = {}
                        if args.tier == "ssd":
                            observed = metrics(f"http://127.0.0.1:{http}")
                            write_json(args.output / f"{prefix}-ssd-after.json", observed)
                            for name in (
                                "orbitkv_ssd_prefetch_bytes_total",
                                "orbitkv_ssd_prefetch_duration_seconds_count",
                                "orbitkv_ssd_prefetch_duration_seconds_sum",
                                "orbitkv_ssd_prefetch_failures_total",
                                "orbitkv_ssd_uring_read_operations_total",
                            ):
                                ssd_observed[name] = observed.get(name, 0) - ssd_before.get(name, 0)
                            if ssd_observed["orbitkv_ssd_prefetch_bytes_total"] != expected_bytes:
                                raise AssertionError(
                                    f"SSD bytes differ from GPU bytes: {ssd_observed}"
                                )
                            if ssd_observed["orbitkv_ssd_prefetch_duration_seconds_count"] <= 0:
                                raise AssertionError("SSD batch completion evidence is missing")
                            if ssd_observed["orbitkv_ssd_uring_read_operations_total"] <= 0:
                                raise AssertionError("Queued io_uring READV evidence is missing")
                            if ssd_observed["orbitkv_ssd_prefetch_failures_total"]:
                                raise AssertionError(f"SSD reads failed: {ssd_observed}")
                        key = {
                            "tier": args.tier,
                            "repeat": repeat,
                            "operation": operation,
                            "payload_bytes": size * args.layers,
                            "bytes_per_layer": size,
                            "layers": args.layers,
                            "layout": args.layout,
                            "blocks": len(targets),
                            "hash_bytes": len(targets) * 32,
                            "restore_batch_size": len(restore_batches)
                            if operation == "restore"
                            else 0,
                            "idle_ms": idle_ms,
                        }
                        for index, row in enumerate(rows):
                            raw.write(
                                json.dumps({**key, "sample": index, **row}, allow_nan=False) + "\n"
                            )
                        raw.flush()
                        cohorts.append(
                            {
                                **key,
                                "samples": summarize(rows),
                                "cohort_wall_seconds": elapsed / 1e9,
                                "cohort_client_cpu_seconds": cohort_cpu / 1e9,
                                "cohort_manager_cpu_seconds": after["cpu_seconds"]
                                - before["cpu_seconds"],
                                "gpu_bytes_verified": operation == "restore",
                                "completed_copy_bytes": copied_bytes if counter else None,
                                "ssd_observed": ssd_observed,
                            }
                        )
                        print(
                            f"{args.label} repeat={repeat} {operation} layout={args.layout} bytes={size} leases={key['restore_batch_size']} idle_ms={idle_ms:g} p50_us={cohorts[-1]['samples']['wall_us']['p50']:.2f} p99_us={cohorts[-1]['samples']['wall_us']['p99']:.2f}",
                            flush=True,
                        )
            if args.tier == "ssd":
                observed = metrics(f"http://127.0.0.1:{http}")
                retained = {
                    name: observed.get(name, 0)
                    for name in (
                        "orbitkv_ssd_read_pinned_bytes",
                        "orbitkv_ssd_prefetch_inflight",
                        "orbitkv_ssd_write_inflight",
                        "orbitkv_ssd_write_queue_pending",
                    )
                }
                write_json(args.output / "drain.json", retained)
                if any(retained.values()):
                    raise AssertionError(f"SSD ownership has not drained: {retained}")
            ok, message = client.unregister_context(instance)
            if not ok:
                raise RuntimeError(f"context drain failed: {message}")
            client.close()
            client = None
        if args.tier == "ssd":
            write_json(
                args.output / "manager-exit.json",
                {"pid": manager.pid, "exit_code": manager.returncode},
            )
            if manager.returncode != 0:
                raise RuntimeError(f"SSD Manager did not exit normally: {manager.returncode}")
        if {str(path): digest(path) for path in artifacts} != artifact_hashes:
            raise RuntimeError("native artifacts or GPU registration helper changed during the run")
        write_json(args.output / "results.json", {"label": args.label, "cohorts": cohorts})
        write_json(args.output / "status.json", {"state": "complete", "cohorts": len(cohorts)})
    except BaseException as error:
        write_json(
            args.output / "status.json",
            {"state": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    finally:
        # The owned server context has already stopped/drained the Manager.
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
