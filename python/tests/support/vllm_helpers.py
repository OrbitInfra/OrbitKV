"""Shared utilities for vLLM E2E tests.

Extracted from test_vllm_e2e_correctness.py to enable reuse across test modules.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import requests

from tests.support.cache_manager import find_available_port
from tests.support.gpu_identity import identity_group_members, inherited_gpu_lock_fds
from tests.support.paths import PYTHON_ROOT, REPO_ROOT

DEFAULT_VLLM_SEED = 42


def _linear_attention_kind(model: str) -> str | None:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    text_config = config.get("text_config", config)
    if (text_config.get("linear_attn_config") or {}).get("kda_layers"):
        return "kda"
    if "linear_attention" in (text_config.get("layer_types") or ()):
        return "linear_attention"
    return None


def adapt_prompt_for_hybrid_cache(model: str, prompt: str) -> str:
    kind = _linear_attention_kind(model)
    if kind is None:
        return prompt
    family = hashlib.sha256(prompt[:80].encode()).hexdigest()[:8]
    repetitions = 160 if kind == "kda" else 80
    prefix = (
        f"Hybrid cache boundary {family} contains deterministic background context. " * repetitions
    )
    return prefix + prompt


def e2e_max_tokens(model: str) -> int:
    return 8 if _linear_attention_kind(model) else 50


def _detect_orbitkv_cargo_features() -> list[str]:
    try:
        import torch
    except ImportError:
        return []

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return []

    # Keep parity with the default feature set (cuda-12 + mooncake) so the e2e
    # server build shares the cargo cache with maturin dev builds instead of
    # invalidating it on every alternation.
    major = cuda_version.split(".", maxsplit=1)[0]
    return ["cuda-13", "mooncake"] if major == "13" else []


class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    def __init__(
        self,
        model: str,
        port: int,
        use_orbitkv: bool = False,
        orbitkv_port: int | None = None,
        log_file: Path | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        extra_args: list[str] | None = None,
        startup_timeout: int = 600,
        use_noop_connector: bool = False,
        kv_transfer_config: dict[str, object] | None = None,
        server_label: str | None = None,
        env_overrides: dict[str, str] | None = None,
        transfer_backend: str | None = None,
        prefix_caching: bool | None = None,
    ):
        self.model = model
        self.port = port
        self.use_orbitkv = use_orbitkv
        self.orbitkv_port = orbitkv_port
        self.transfer_backend = transfer_backend
        self.prefix_caching = prefix_caching
        self.log_file = log_file
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.use_noop_connector = use_noop_connector
        self.kv_transfer_config = kv_transfer_config
        self.server_label = server_label
        self.env_overrides = env_overrides or {}
        self.health_endpoints = ["/health", "/v1/models"]
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    def __enter__(self):
        """Start the vLLM server."""
        if self.tensor_parallel_size < 1 or self.pipeline_parallel_size < 1:
            raise ValueError("tensor/pipeline parallel sizes must be >= 1")
        world_size = self.tensor_parallel_size * self.pipeline_parallel_size
        if world_size > 4:
            raise ValueError(
                f"tensor_parallel_size * pipeline_parallel_size must be <= 4 (got {world_size})"
            )

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        python_source = str(PYTHON_ROOT)
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{current_pythonpath}:{python_source}" if current_pythonpath else python_source
        )
        linear_attention = _linear_attention_kind(self.model)
        if linear_attention:
            env.pop("VLLM_BATCH_INVARIANT", None)
        else:
            env["VLLM_BATCH_INVARIANT"] = "1"

        if self.orbitkv_port is not None:
            env["ORBITKV_PORT"] = str(self.orbitkv_port)
        env.update(self.env_overrides)

        # Resolve vllm binary from the current Python environment's bin dir
        import sys

        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")

        prefix_caching = (
            linear_attention is not None if self.prefix_caching is None else self.prefix_caching
        )
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--attention-backend",
            # FLASH_ATTN keeps outputs batch-invariant for the baseline/warm
            # comparison, but it cannot serve MLA models on every GPU
            # generation (e.g. sm_120 has no FlashMLA); allow overriding with
            # an MLA-capable backend like TRITON_MLA there.
            os.environ.get("VLLM_TEST_ATTN_BACKEND", "FLASH_ATTN"),
            "--generation-config",
            "vllm",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.pipeline_parallel_size),
        ]

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if linear_attention:
            cmd.extend(
                [
                    "--max-num-seqs",
                    "1",
                    "--max-num-batched-tokens",
                    "2048" if linear_attention == "kda" else "1024",
                ]
            )

        if self.kv_transfer_config is not None:
            cmd.extend(["--kv-transfer-config", json.dumps(self.kv_transfer_config)])
        elif self.use_orbitkv or self.use_noop_connector:
            connector_name = "OrbitKVConnector" if self.use_orbitkv else "NoopKVConnector"
            connector_module = (
                "orbitkv.vllm" if self.use_orbitkv else "tests.support.noop_vllm_connector"
            )
            kv_config: dict[str, object] = {
                "kv_connector": connector_name,
                "kv_role": "kv_both",
                "kv_connector_module_path": connector_module,
            }
            # The server no longer selects a transfer backend; the connector
            # does. Force it here so --orbitkv-transfer-backend still exercises
            # the chosen path end-to-end, regardless of the model's MLA default.
            extra_config: dict[str, object] = {}
            if self.use_orbitkv and self.transfer_backend is not None:
                extra_config["orbitkv.transfer_backend"] = self.transfer_backend
            if extra_config:
                kv_config["kv_connector_extra_config"] = extra_config
            cmd.extend(["--kv-transfer-config", json.dumps(kv_config)])

        cmd.extend(self.extra_args)

        server_label = self.server_label or (
            "OrbitKV" if self.use_orbitkv or self.kv_transfer_config is not None else "Baseline"
        )
        print(f"\n[{server_label}] Starting vLLM server on port {self.port}")

        if self.log_file:
            print(f"[{server_label}] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
                pass_fds=inherited_gpu_lock_fds(),
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                preexec_fn=os.setsid,
                pass_fds=inherited_gpu_lock_fds(),
            )

        self._wait_for_ready(timeout=self.startup_timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the API owner and require its engine workers to drain."""
        from tests.support.installed_serving import process_group_members

        if self.process:
            process = self.process
            with suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                process.poll()
                if not process_group_members(process.pid):
                    break
                time.sleep(0.1)
            remaining = process_group_members(process.pid)
            if remaining:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            cleanup = {
                "pid": process.pid,
                "exit_code": process.returncode,
                "forced_kill": bool(remaining),
                "remaining_processes": process_group_members(process.pid),
            }
            if self.log_file:
                Path(str(self.log_file) + ".cleanup.json").write_text(
                    json.dumps(cleanup, indent=2) + "\n"
                )
            self.process = None
            if self.log_handle:
                self.log_handle.close()
            assert not remaining and not cleanup["remaining_processes"], cleanup
            assert process.returncode in {0, -signal.SIGTERM}, cleanup
        elif self.log_handle:
            self.log_handle.close()

    def _wait_for_ready(self, timeout: int = 180):
        """Wait for the server to be ready to accept requests."""
        start_time = time.time()
        print("Waiting for server to be ready...")
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                hint = f", see {self.log_file}" if self.log_file else ""
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}{hint}")
            for endpoint in self.health_endpoints:
                url = f"http://localhost:{self.port}{endpoint}"
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        print(f"Server is ready! (checked {endpoint})\n")
                        time.sleep(2)  # Extra buffer
                        return
                except requests.exceptions.RequestException:
                    continue
            time.sleep(2)

        raise TimeoutError(f"Server did not become ready within {timeout} seconds")


class CacheManager:
    """Context manager for Cache Manager lifecycle.

    Auto-starts Cache Manager with prometheus metrics enabled.
    Picks random available ports for local socket naming and HTTP.
    """

    def __init__(
        self,
        log_file: Path | None = None,
        pool_size: str = "30gb",
        log_level: str | None = None,
        cargo_features: list[str] | None = None,
        use_hugepages: bool = False,
        devices: str | None = None,
        server_binary: str | None = None,
        extra_args: tuple[str, ...] = (),
    ):
        self.cache_port = find_available_port()
        self.http_port = find_available_port()
        self.pool_size = pool_size
        self.log_level = log_level
        self.use_hugepages = use_hugepages
        self.devices = devices
        self.server_binary = server_binary or os.environ.get("ORBITKV_CACHE_MANAGER_BINARY")
        self.extra_args = extra_args
        self.cargo_features = (
            cargo_features if cargo_features is not None else _detect_orbitkv_cargo_features()
        )
        self.log_file = log_file
        self.process: subprocess.Popen | None = None
        self.log_handle = None

    @property
    def metrics_port(self) -> int:
        return self.http_port

    @property
    def bootstrap_socket(self) -> str:
        return f"/tmp/orbitkv-{self.cache_port}.sock"

    def __enter__(self):
        project_root = REPO_ROOT

        if self.server_binary is None:
            cmd = ["cargo", "run", "-r"]
            if self.cargo_features:
                cmd.append("--no-default-features")
                cmd.extend(["--features", ",".join(self.cargo_features)])
            cmd.extend(["--bin", "orbitkv-cache-manager", "--"])
        else:
            cmd = [self.server_binary]
        cmd.extend(
            [
                "--addr",
                f"127.0.0.1:{self.cache_port}",
                "--http-addr",
                f"0.0.0.0:{self.http_port}",
                "--pool-size",
                self.pool_size,
                "--enable-prometheus",
            ]
        )
        if self.use_hugepages:
            cmd.append("--use-hugepages")
        if self.devices is not None:
            cmd.extend(["--devices", self.devices])
        if self.log_level is not None:
            cmd.extend(["--log-level", self.log_level])
        cmd.extend(self.extra_args)

        # Cache Manager embeds Python via PyO3 (for CUDA device detection via torch)
        import sys
        import sysconfig

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        env["PYO3_PYTHON"] = sys.executable
        env["PYTHONHOME"] = sys.base_prefix
        if libdir := sysconfig.get_config_var("LIBDIR"):
            existing_library_path = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = (
                f"{existing_library_path}:{libdir}" if existing_library_path else libdir
            )
        python_dir = str(PYTHON_ROOT)
        site_packages = sysconfig.get_path("purelib")
        configured_paths = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
        env["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys(
                [*configured_paths, python_dir, *([site_packages] if site_packages else [])]
            )
        )

        if self.server_binary is None:
            launch_label = f"cargo run -r features={','.join(self.cargo_features) or 'default'}"
        else:
            launch_label = self.server_binary
        print(
            f"\n[Cache Manager] {launch_label} on UDS={self.bootstrap_socket}, HTTP={self.http_port}"
        )

        if self.log_file:
            print(f"[Cache Manager] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                pass_fds=inherited_gpu_lock_fds(),
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
                pass_fds=inherited_gpu_lock_fds(),
            )

        self._wait_for_ready()
        return self

    def _wait_for_ready(self, timeout: int = 300):
        """Wait for Cache Manager HTTP health endpoint."""
        start = time.time()
        print("Waiting for Cache Manager to be ready...")
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"Cache Manager exited with code {self.process.returncode}")
            try:
                resp = requests.get(f"http://localhost:{self.http_port}/health", timeout=1)
                if resp.status_code == 200:
                    print("Cache Manager is ready!\n")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        raise TimeoutError(f"Cache Manager did not become ready within {timeout}s")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            print("\n[Cache Manager] Stopping...")
            process = self.process
            forced_kill = False
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                if process.poll() is None:
                    forced_kill = True
                    with suppress(ProcessLookupError, OSError):
                        os.killpg(process.pid, signal.SIGKILL)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print("Cache Manager did not exit after SIGKILL")
            members, errors = identity_group_members(process.pid)
            cleanup = {
                "pid": process.pid,
                "exit_code": process.returncode,
                "forced_kill": forced_kill,
                "remaining_processes": members,
                "inspection_errors": errors,
            }
            if self.log_file:
                Path(str(self.log_file) + ".cleanup.json").write_text(
                    json.dumps(cleanup, indent=2) + "\n"
                )
            print("Cache Manager stopped.\n")
        if self.log_handle:
            self.log_handle.close()


def call_openai_api(
    port: int,
    model: str,
    prompt: str | list[int],
    max_tokens: int = 50,
    temperature: float = 0.0,
    seed: int | None = None,
) -> dict:
    """Call vLLM's OpenAI-compatible API.

    Returns dict with 'text' and 'logprobs' (if available).
    """
    url = f"http://localhost:{port}/v1/completions"

    if seed is None:
        seed = DEFAULT_VLLM_SEED

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": 5,  # Request logprobs for detailed comparison
    }

    response = requests.post(url, json=payload, timeout=120)
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} from {url}: {response.text[:4096]}", response=response
        )
    data = response.json()

    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "logprobs": choice.get("logprobs"),
        "finish_reason": choice["finish_reason"],
        "usage": data.get("usage"),
    }
