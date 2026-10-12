"""Pytest fixtures for OrbitKV connector integration tests.

Provides fixtures for automatically starting/stopping the Cache Manager
and test helpers for connector testing against a running server.
"""

import contextlib
import errno
import json
import logging
import os
import secrets
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import requests

from .gpu_identity import identity_group_members, inherited_gpu_lock_fds
from .metrics import fetch_orbitkv_metrics
from .paths import PYTHON_ROOT, REPO_ROOT

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_POOL_SIZE = "100mb"
SERVER_STARTUP_TIMEOUT = 60  # seconds (increased for slow GPU init)
SERVER_READY_CHECK_INTERVAL = 1.0  # seconds


def evict_dram_after_ssd_writes(http_port: int) -> None:
    """Force subsequent recovery through SSD after the engine has stopped."""
    deadline = time.monotonic() + 30
    while True:
        observed = fetch_orbitkv_metrics(http_port)
        if observed.get("orbitkv_ssd_write_bytes_total", 0) > 0 and not any(
            observed.get(name, 0)
            for name in (
                "orbitkv_ssd_write_inflight",
                "orbitkv_ssd_write_queue_pending",
                "orbitkv_inflight_bytes",
            )
        ):
            break
        assert time.monotonic() < deadline, observed
        time.sleep(0.1)
    cleaned = requests.post(f"http://127.0.0.1:{http_port}/cache/memory/cleanup", timeout=30)
    cleaned.raise_for_status()
    assert cleaned.json()["evicted_blocks"] > 0
    assert cleaned.json()["still_referenced_blocks"] == 0


# =============================================================================
# Utility Functions
# =============================================================================


def _torch():
    return pytest.importorskip("torch", reason="torch is required for GPU integration tests")


def find_available_port() -> int:
    """Find a listener port outside outgoing TCP and TENT auto-port ranges."""
    low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
    for _ in range(128):
        port = 1024 + secrets.randbelow(65536 - 1024)
        # Pinned TENT CoroRpcAgent starts before the engine HTTP listener and
        # independently chooses an RPC port in [15000, 17000).
        if low <= port <= high or 15000 <= port < 17000:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                s.listen(1)
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                continue
            return s.getsockname()[1]
    raise RuntimeError("No free test listener port found")


def find_cache_manager_binary() -> str | None:
    """
    Locate the Cache Manager binary.

    Search order:
    1. Installed orbitkv-cache-manager-py in package directory
    2. cargo target/release/orbitkv-cache-manager
    3. cargo target/debug/orbitkv-cache-manager
    """
    if configured := os.environ.get("ORBITKV_CACHE_MANAGER_BINARY"):
        binary = Path(configured).resolve()
        if not binary.is_file():
            raise FileNotFoundError(binary)
        return str(binary)

    # 1. Check installed package binary
    try:
        from orbitkv._cache_manager import get_cache_manager_binary

        binary = get_cache_manager_binary()
        if Path(binary).exists():
            return binary
    except ImportError:
        pass

    # 2. Check cargo build outputs
    project_root = REPO_ROOT
    for build_type in ["release", "debug"]:
        cargo_binary = project_root / "target" / build_type / "orbitkv-cache-manager"
        if cargo_binary.exists():
            return str(cargo_binary)

    return None


def manager_pythonpath(
    configured_paths: list[str],
    site_packages: list[str],
    runtime_python_paths: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if runtime_python_paths is None:
        roots = [str(PYTHON_ROOT), *configured_paths]
    else:
        roots = list(runtime_python_paths)
    return tuple(dict.fromkeys([*roots, *site_packages]))


def wait_for_server_ready(
    bootstrap_socket: str,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    process: subprocess.Popen | None = None,
) -> bool:
    """Wait for the local Cache Manager to accept lifecycle connections."""
    import importlib

    orbitkv_module = importlib.import_module("orbitkv.orbitkv")

    start_time = time.time()
    last_error = None
    while time.time() - start_time < timeout:
        if process is not None and process.poll() is not None:
            return False
        try:
            client = orbitkv_module.CacheManagerClient(bootstrap_socket)
            ok, _ = client.health()
            client.close()
            if ok:
                return True
        except Exception as e:
            last_error = str(e)
        time.sleep(SERVER_READY_CHECK_INTERVAL)

    if last_error:
        print(f"Last health check error: {last_error}", file=sys.stderr)
    return False


# =============================================================================
# Test Helpers: ClientContext
# =============================================================================


def initialize_kv_cache(
    device: "torch.device",
    num_blocks: int = 64,
    num_layers: int = 1,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 128,
    dtype: "torch.dtype | None" = None,
) -> list["torch.Tensor"]:
    """
    Initialize KV cache tensors on GPU for testing.

    Creates tensors in KV-first layout: (2, num_blocks, block_size, num_heads, head_size)
    where the first dimension is [K, V].
    """
    torch = _torch()
    if dtype is None:
        dtype = torch.bfloat16

    torch.random.manual_seed(42)

    gpu_tensors = [
        torch.rand(
            (2, num_blocks, block_size, num_heads, head_size),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    return gpu_tensors


class ClientContext:
    """
    Client context that represents a vLLM instance.

    This class abstracts a vLLM instance by managing:
    - GPU KV cache tensors (like WorkerAdapter)
    - Query operations (like SchedulerAdapter)
    - Context registration/unregistration
    """

    def __init__(
        self,
        client,
        instance_id: str,
        namespace: str,
        device_id: int = 0,
        num_blocks: int = 64,
        num_layers: int = 1,
        block_size: int = 16,
        num_heads: int = 8,
        head_size: int = 128,
        dtype: "torch.dtype | None" = None,
    ):
        torch = _torch()
        if dtype is None:
            dtype = torch.bfloat16
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"device_id {device_id} >= available GPUs ({torch.cuda.device_count()})"
            )

        self.client = client
        self.instance_id = instance_id
        self.namespace = namespace
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        # Initialize KV cache tensors
        self.gpu_kv_caches = initialize_kv_cache(
            self.device, num_blocks, num_layers, block_size, num_heads, head_size, dtype
        )

        # Map layer index to layer name (for compatibility with vLLM)
        self._layer_names = [f"layer_{i}" for i in range(num_layers)]
        self._registered = False

    def register_kv_caches(self) -> None:
        """Register KV cache tensors with the engine server (like WorkerAdapter.register_kv_caches)."""
        from orbitkv.client.gpu import serialize_gpu_buffer

        if self._registered:
            return

        kv_caches = {name: self.gpu_kv_caches[i] for i, name in enumerate(self._layer_names)}

        layer_names: list[str] = []
        wrapper_bytes_list: list[bytes] = []
        num_blocks_list: list[int] = []
        bytes_per_block_list: list[int] = []
        kv_stride_bytes_list: list[int] = []
        segments_list: list[int] = []

        for layer_name, kv_cache in kv_caches.items():
            wrapper_bytes = serialize_gpu_buffer(kv_cache)

            shape = tuple(kv_cache.shape)
            stride = tuple(kv_cache.stride())
            element_size = kv_cache.element_size()

            if len(shape) >= 2 and shape[0] == 2:
                num_blocks = shape[1]
                bytes_per_block = stride[1] * element_size
                kv_stride_bytes = stride[0] * element_size
                segments = 2
            else:
                num_blocks = shape[0]
                bytes_per_block = stride[0] * element_size
                kv_stride_bytes = 0
                segments = 1

            layer_names.append(layer_name)
            wrapper_bytes_list.append(wrapper_bytes)
            num_blocks_list.append(num_blocks)
            bytes_per_block_list.append(bytes_per_block)
            kv_stride_bytes_list.append(kv_stride_bytes)
            segments_list.append(segments)

        # One batch per device, like the production worker: the engine seals
        # the instance topology once all world_size devices have registered.
        ok, message = self.client.register_context_batch(
            self.instance_id,
            self.namespace,
            0,  # tp_rank
            0,  # pp_rank
            1,  # tp_size
            1,  # world_size
            self.device_id,
            layer_names,
            wrapper_bytes_list,
            num_blocks_list,
            bytes_per_block_list,
            kv_stride_bytes_list,
            segments_list,
            "direct",
            False,
            tensors=list(kv_caches.values()),
            layer_attention=[
                (self.head_size, "kv", index, self.num_layers)
                if str(t.dtype) in {"torch.bfloat16", "torch.float16"}
                else (0, "", 0, 0)
                for index, t in enumerate(kv_caches.values())
            ],
            layer_formats=[
                {"torch.bfloat16": "bf16", "torch.float16": "fp16"}.get(str(t.dtype), "exact")
                for t in kv_caches.values()
            ],
        )

        if not ok:
            raise RuntimeError(f"Register context failed for {layer_names}: {message}")

        self._registered = True

    def unregister_context(self) -> None:
        """Unregister context from server (like WorkerAdapter.unregister_context)."""
        if not self._registered:
            return

        try:
            ok, message = self.client.unregister_context(self.instance_id)
            if not ok:
                logger.warning(f"Unregister context failed: {message}")
        except Exception as e:
            logger.warning(f"Unregister context exception: {e}")

        self._registered = False

    def query(self, block_hashes: list[bytes]) -> dict:
        """Query available attention-prefix blocks.

        Args:
            block_hashes: List of block hashes to query

        Returns:
            Query result dict
        """
        from orbitkv import BlockHashes

        return self.client.query_prefetch(
            self.instance_id, BlockHashes(block_hashes), req_id="test"
        )

    def get_kv_cache(self, layer: int = 0) -> "torch.Tensor":
        """Get KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer]

    def get_tensor_slice(self, layer: int, start_block: int, num_blocks: int) -> "torch.Tensor":
        """Get a slice of the KV cache tensor for a specific layer."""
        return self.gpu_kv_caches[layer][:, start_block : start_block + num_blocks]


class CacheManagerProcess:
    """Manages a Cache Manager subprocess for testing."""

    def __init__(
        self,
        port: int,
        pool_size: str = DEFAULT_POOL_SIZE,
        devices: str = "0",
        *,
        http_port: int | None = None,
        channel_service: str | None = None,
        channel_session_epoch: int | None = None,
        bootstrap_socket: str | None = None,
        ssd_cache_path: Path | None = None,
        ssd_cache_capacity: str = "256mb",
        ssd_backend: str = "uring",
        ssd_read_path: str | None = None,
        query_budget: str | None = None,
        query_instance_budget: str | None = None,
        log_path: Path | None = None,
        extra_args: tuple[str, ...] = (),
        runtime_python_paths: tuple[str, ...] | None = None,
        inherited_fds: tuple[int, ...] = (),
    ):
        self.port = port
        self.pool_size = pool_size
        self.devices = devices
        self.http_port = http_port
        self.channel_service = channel_service
        self.channel_session_epoch = channel_session_epoch
        self.bootstrap_socket = bootstrap_socket or f"/tmp/orbitkv-{port}.sock"
        self.ssd_cache_path = ssd_cache_path
        self.ssd_cache_capacity = ssd_cache_capacity
        self.ssd_backend = ssd_backend
        self.ssd_read_path = ssd_read_path
        self.query_budget = query_budget
        self.query_instance_budget = query_instance_budget
        self._configured_log_path = log_path
        self.extra_args = extra_args
        self.runtime_python_paths = runtime_python_paths
        self.inherited_fds = inherited_fds or inherited_gpu_lock_fds()
        self.pythonpath: tuple[str, ...] | None = None
        self.process: subprocess.Popen | None = None
        self.command: tuple[str, ...] | None = None
        self._binary_path = find_cache_manager_binary()
        self._log_path: Path | None = None
        self._log_file = None

    def start(self) -> bool:
        """Start the server process. Returns True if successful."""
        if not self._binary_path:
            return False

        env = os.environ.copy()
        env["PYO3_PYTHON"] = sys.executable
        env["PYTHONHOME"] = sys.base_prefix

        # Add libpython to LD_LIBRARY_PATH if available
        if libdir := sysconfig.get_config_var("LIBDIR"):
            env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"

        # Set PYTHONPATH to include python package and venv site-packages
        configured_paths = [path for path in env.get("PYTHONPATH", "").split(":") if path]
        site_packages = [
            path
            for path in dict.fromkeys(
                [
                    sysconfig.get_path("purelib"),
                    *(path for path in sys.path if Path(path).name == "site-packages"),
                ]
            )
            if path
        ]
        self.pythonpath = manager_pythonpath(
            configured_paths,
            site_packages,
            self.runtime_python_paths,
        )
        env["PYTHONPATH"] = os.pathsep.join(self.pythonpath)

        cmd = [
            self._binary_path,
            "--addr",
            f"127.0.0.1:{self.port}",
            "--pool-size",
            self.pool_size,
            "--devices",
            self.devices,
        ]
        if self.http_port is not None:
            cmd.extend(["--http-addr", f"127.0.0.1:{self.http_port}"])
        if self.query_budget is not None:
            cmd.extend(["--query-budget", self.query_budget])
        if self.query_instance_budget is not None:
            cmd.extend(["--query-instance-budget", self.query_instance_budget])
        if self.ssd_cache_path is not None:
            cmd.extend(
                [
                    "--ssd-cache-path",
                    str(self.ssd_cache_path),
                    "--ssd-cache-capacity",
                    self.ssd_cache_capacity,
                    "--ssd-backend",
                    self.ssd_backend,
                    "--enable-prometheus",
                ]
            )
            if self.ssd_read_path is not None:
                cmd.extend(["--ssd-read-path", self.ssd_read_path])
        if self.channel_service is not None:
            cmd.extend(["--channel-service", self.channel_service])
        if self.channel_session_epoch is not None:
            cmd.extend(
                [
                    "--channel-session-epoch",
                    str(self.channel_session_epoch),
                ]
            )
        cmd.extend(["--bootstrap-socket", self.bootstrap_socket])
        cmd.extend(self.extra_args)
        self.command = tuple(cmd)

        # Route logs to a tempfile so the pipe buffer cannot fill up and
        # block the server mid-startup, and so tests can read the log
        # contents via read_logs() (used by integration tests that assert
        # on server-side log signals).
        if self._configured_log_path is not None:
            self._log_path = self._configured_log_path
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("wb")
        else:
            fd, path = tempfile.mkstemp(prefix=f"orbitkv-cache-manager-{self.port}-", suffix=".log")
            self._log_path = Path(path)
            self._log_file = os.fdopen(fd, "wb")

        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                cwd="/tmp",
                preexec_fn=os.setsid,
                pass_fds=self.inherited_fds,
            )
        except (FileNotFoundError, PermissionError):
            self._close_log()
            return False

        return wait_for_server_ready(self.bootstrap_socket, process=self.process)

    def stop(self) -> None:
        """Stop the server process."""
        if self.process is None:
            self._close_log()
            return
        process = self.process
        forced_kill = False
        try:
            if self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            forced_kill = True
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=2)
        except (ProcessLookupError, OSError):
            self.process.wait(timeout=2)
        finally:
            members, errors = identity_group_members(process.pid)
            if self._configured_log_path is not None:
                Path(str(self._configured_log_path) + ".cleanup.json").write_text(
                    json.dumps(
                        {
                            "pid": process.pid,
                            "exit_code": process.returncode,
                            "forced_kill": forced_kill,
                            "remaining_processes": members,
                            "inspection_errors": errors,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            self.process = None
            self._close_log()
            if self.bootstrap_socket is not None:
                with contextlib.suppress(OSError):
                    Path(self.bootstrap_socket).unlink()

    def terminate_gracefully(self, timeout: float = 10) -> tuple[int, float]:
        """Require SIGTERM shutdown without falling back to SIGKILL."""
        if self.process is None:
            raise RuntimeError("Cache Manager is not running")
        process = self.process
        started = time.monotonic()
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        returncode = process.wait(timeout=timeout)
        elapsed = time.monotonic() - started
        self.process = None
        self._close_log()
        with contextlib.suppress(OSError):
            Path(self.bootstrap_socket).unlink()
        return returncode, elapsed

    def is_running(self) -> bool:
        """Check if server process is still running."""
        return self.process is not None and self.process.poll() is None

    def read_logs(self) -> str:
        """Return current server log contents (stdout + stderr merged)."""
        if not self._log_path or not self._log_path.exists():
            return ""
        return self._log_path.read_text(errors="replace")

    def _close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None
        if self._configured_log_path is None and self._log_path and self._log_path.exists():
            with contextlib.suppress(OSError):
                self._log_path.unlink()
