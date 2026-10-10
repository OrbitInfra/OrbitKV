"""Own benchmark service processes and capture the execution environment."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import requests

from .shutdown import validate_engine_shutdown

ROOT = Path(__file__).resolve().parents[1]
_PR_SET_CHILD_SUBREAPER = 36


def free_port() -> int:
    """Keep delayed listeners outside outgoing TCP and TENT auto-port ranges."""
    low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    for _ in range(128):
        port = 1024 + secrets.randbelow(65536 - 1024)
        # TENT chooses its RPC port independently before HTTP startup.
        if low <= port <= high or 15000 <= port < 17000:
            continue
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                continue
            return sock.getsockname()[1]
    raise RuntimeError("No free benchmark listener port found")


@contextlib.contextmanager
def server(
    command: list[str],
    env: dict[str, str],
    url: str,
    log: Path,
    health_path: str = "/health",
    *,
    engine: str | None = None,
):
    if engine is not None and engine not in {"vllm", "sglang"}:
        raise ValueError(f"Unsupported shutdown protocol: {engine}")
    if int(Path("/proc/self/stat").read_text().split(maxsplit=1)[0]) != os.getpid():
        raise RuntimeError("Service cleanup requires /proc in the benchmark's PID namespace")
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot own orphaned service workers")
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    primary_error = None
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited: {log}\n{log.read_text()[-6000:]}")
            try:
                if requests.get(url + health_path, timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"Server startup timed out: {log}")
        yield process
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup = log.with_suffix(".cleanup.json")
        try:
            start = None
            if engine is not None:
                with contextlib.suppress(OSError):
                    start = log.stat().st_size
            record = stop_owned_process(process, cleanup)
            if engine is not None:
                try:
                    if start is None:
                        raise ValueError("Missing native engine shutdown log boundary")
                    end = log.stat().st_size
                    with log.open("rb") as output:
                        output.seek(start)
                        shutdown_log = output.read(65536)
                    record.update(
                        shutdown_log_start_bytes=start,
                        shutdown_log_end_bytes=end,
                        shutdown_log_truncated=end - start > len(shutdown_log),
                    )
                    if record["shutdown_log_truncated"]:
                        raise ValueError("Truncated native engine shutdown log")
                    validate_engine_shutdown(shutdown_log.decode(errors="replace"), engine)
                except (OSError, ValueError) as error:
                    record["shutdown_error"] = str(error)
                    raise
                finally:
                    cleanup.write_text(json.dumps(record, indent=2) + "\n")
            if (
                not record["was_running"]
                or record["exit_code"] != 0
                or record["forced_kill"]
                or record["remaining_group"]
                or record["process_group_errors"]
            ):
                raise RuntimeError(f"Service shutdown failed: {cleanup}")
        except BaseException as error:
            if primary_error is None:
                raise
            with contextlib.suppress(OSError, ValueError):
                print(
                    f"Service cleanup also failed: {error!r}; evidence: {cleanup}",
                    file=sys.stderr,
                )


def _process_group_members(pgid: int) -> list[tuple[int, str]]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid:
            members.append((int(entry.name), fields[0]))
    return sorted(members)


def stop_owned_process(process: subprocess.Popen, cleanup: Path | None = None) -> dict:
    """Reap the service before callers can release exported GPU allocations."""
    interrupted = False

    def defer_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.getsignal(signal.SIGINT)
    defer_sigint = (
        threading.current_thread() is threading.main_thread() and previous != signal.SIG_IGN
    )
    if defer_sigint:
        signal.signal(signal.SIGINT, defer_interrupt)
    graceful = True
    warned = False
    exit_code = None
    remaining = []
    adopted_children = []
    process_group_errors = 0
    last_process_group_error = None
    pre_stop_exit_code = process.poll()
    try:
        while True:
            try:
                # Sending SIGKILL is a request, not evidence that a process stuck
                # in a driver has exited. Keep this scope alive until wait reaps it.
                try:
                    if not graceful:
                        os.killpg(process.pid, signal.SIGKILL)
                    elif pre_stop_exit_code is None:
                        os.kill(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    process_group_errors += 1
                    last_process_group_error = repr(error)[:512]
                exit_code = process.wait(timeout=30 if graceful else 10)
                break
            except subprocess.TimeoutExpired:
                if not graceful and not warned:
                    with contextlib.suppress(OSError, ValueError):
                        print(
                            f"Waiting for killed service {process.pid} to exit; GPU owners remain held",
                            file=sys.stderr,
                            flush=True,
                        )
                    warned = True
                graceful = False
            except KeyboardInterrupt:
                # Also covers a wait interrupted outside the main thread or an
                # exception raised by a custom wait implementation.
                interrupted = True
                graceful = False
        deadline = time.monotonic() + 30
        while True:
            try:
                try:
                    owned_children_pending = False
                    try:
                        while True:
                            child, status = os.waitpid(-process.pid, os.WNOHANG)
                            if child == 0:
                                owned_children_pending = True
                                break
                            adopted_children.append(
                                {"pid": child, "exit_code": os.waitstatus_to_exitcode(status)}
                            )
                    except ChildProcessError:
                        pass
                    remaining = _process_group_members(process.pid)
                except (OSError, ValueError, IndexError) as error:
                    process_group_errors += 1
                    last_process_group_error = repr(error)[:512]
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        remaining = []
                        break
                    except OSError:
                        pass
                    graceful = False
                    with contextlib.suppress(OSError):
                        os.killpg(process.pid, signal.SIGKILL)
                    time.sleep(0.1)
                    continue
                if not any(state != "Z" for _, state in remaining):
                    if remaining and owned_children_pending:
                        time.sleep(0.1)
                        continue
                    break
                if not graceful or time.monotonic() >= deadline:
                    graceful = False
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        process_group_errors += 1
                        last_process_group_error = repr(error)[:512]
                time.sleep(0.1)
            except KeyboardInterrupt:
                interrupted = True
                graceful = False
    finally:
        if defer_sigint:
            signal.signal(signal.SIGINT, previous)
        record = {
            "pid": process.pid,
            "exit_code": exit_code,
            "forced_kill": not graceful,
            "reaped": exit_code is not None,
            "interrupted": interrupted,
            "was_running": pre_stop_exit_code is None,
            "pre_stop_exit_code": pre_stop_exit_code,
            "remaining_group": [pid for pid, _ in remaining],
            "adopted_children": adopted_children,
            "process_group_errors": process_group_errors,
            "last_process_group_error": last_process_group_error,
        }
        if cleanup is not None:
            cleanup.write_text(json.dumps(record, indent=2) + "\n")
    if interrupted:
        raise KeyboardInterrupt
    return record


def storage_manifest(pid: int, cache_path: Path) -> dict:
    flags = []
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if fd.resolve() != cache_path.resolve():
                continue
            info = Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text()
        except FileNotFoundError:
            continue
        flags.extend(
            int(line.split()[1], 8) for line in info.splitlines() if line.startswith("flags:")
        )
    if not flags or not all(value & os.O_DIRECT for value in flags):
        raise RuntimeError("SSD measurement requires an open O_DIRECT cache file")
    return {
        "path": str(cache_path),
        "capacity_bytes": cache_path.stat().st_size,
        "fd_flags": flags,
        "direct_io": True,
        "payload_retained_after_run": False,
        "mount": subprocess.check_output(
            ["findmnt", "--json", "--target", str(cache_path), "--output", "TARGET,SOURCE,FSTYPE"],
            text=True,
        ),
        "devices": subprocess.check_output(
            ["lsblk", "--json", "--output", "NAME,TYPE,SIZE,ROTA,MODEL"], text=True
        ),
        "notes": "Device inventory does not prove which physical disk backs an overlay mount.",
    }


def process_usage(pid: int) -> dict:
    """Process CPU and I/O counters, excluding child inference processes."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    io = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
    return {
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "read_bytes": int(io["read_bytes"]),
        "write_bytes": int(io["write_bytes"]),
    }


def manifest(args: Namespace, launch, bytes_per_token: int) -> dict:
    packages = [args.engine, "torch", "transformers", "numpy", "prometheus_client"]
    if args.backend in ("lmcache", "flexkv"):
        packages.append(args.backend)
    working_set_tokens = (
        sum(args.lengths[index % len(args.lengths)] for index in range(args.working_set))
        if args.workload == "sustained"
        else None
    )
    manager_sha256 = None
    if args.backend == "orbitkv":
        digest = hashlib.sha256()
        with Path(launch.manager_command[0]).open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024**2), b""):
                digest.update(chunk)
        manager_sha256 = digest.hexdigest()
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "engine_command": launch.command,
        "manager_command": launch.manager_command,
        "manager_binary_sha256": manager_sha256,
        "cost_observations": "1" if launch.env.get("ORBITKV_COST_OBSERVATIONS") == "1" else "0",
        "backend_configuration": launch.backend_configuration,
        "library_path": launch.env.get("LD_LIBRARY_PATH", ""),
        "python_path": launch.env["PYTHONPATH"],
        "storage_environment": {
            name: launch.env[name]
            for name in (
                "ORBITKV_NVCOMP_LIBRARY",
                "LD_PRELOAD",
                "CUFILE_ENV_PATH_JSON",
                "CUFILE_ALLOW_COMPAT_MODE",
                "CUFILE_FORCE_COMPAT_MODE",
            )
            if name in launch.env
        },
        "compiler_cache": {
            name: launch.env[name]
            for name in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
            if name in launch.env
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python": sys.version,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "kv_bytes_per_token": bytes_per_token,
        "capacity": {
            "engine_kv_bytes": args.gpu_tokens * bytes_per_token,
            "host_pool_bytes": 0 if args.backend == "native" else args.host_gib * 1024**3,
            "ssd_bytes": args.ssd_gib * 1024**3,
            "codec_scratch_budget_bytes_per_worker": args.storage_codec_budget
            if args.backend == "orbitkv"
            else None,
            "working_set_tokens": working_set_tokens,
            "working_set_logical_bytes": working_set_tokens * bytes_per_token
            if working_set_tokens is not None
            else None,
            "notes": "Configured budgets, not measured effective capacity. GPU codec scratch and cuFile staging are additional to engine KV. Encoded-publication counters exclude raw-only slots and do not establish whole-cache savings.",
        },
        "quality_scope": "Synthetic token prompts; exact generated-text comparisons are diagnostic, not task accuracy or general model quality. The deterministic_inference argument records whether engine deterministic kernels were requested; otherwise greedy decoding is not guaranteed batch invariant.",
        "ttft_scope": "Client time to first nonempty streamed text, including HTTP/scheduling; not engine time to first token ID.",
        "model_revision": (args.model / ".revision").read_text().strip()
        if (args.model / ".revision").exists()
        else None,
        "concurrency": args.concurrencies if args.workload != "serial" else 1,
        "notes": (
            "Sustained closed-loop traffic with at most C client requests in flight. Throughput includes admitted requests' drain time; cache counters cover the whole window and post-request drain. Prepared-prefix output comparisons are diagnostic, not batch-invariant correctness proofs. Memory peaks are sampled lower bounds."
            if args.workload == "sustained"
            else "Closed-loop bursts; tier counters belong to entire batches, not individual requests. Memory peaks are sampled every 25 ms."
            if args.workload == "concurrent"
            else "Serial latency experiment; pressure traffic and startup are excluded from request timings."
        ),
    }
    return manifest
