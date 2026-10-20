"""SGLang restores GPU KV through OrbitKV's CUDA IPC data path."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import requests

from tests.support.cache_manager import evict_dram_after_ssd_writes, find_available_port
from tests.support.gpu_identity import identity_group_members, inherited_gpu_lock_fds
from tests.support.metrics import fetch_orbitkv_codec_bytes, fetch_orbitkv_metrics
from tests.support.paths import PYTHON_ROOT

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def _stop_server(process: subprocess.Popen, directory: Path, timeout: float = 30) -> None:
    primary_error = sys.exc_info()[1]
    strict = os.environ.get("ORBITKV_TEST_STRICT_SHUTDOWN") == "1"
    start_ticks = getattr(process, "_orbitkv_start_ticks", None)
    if start_ticks is None:
        try:
            start_ticks = int(
                Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, ValueError, IndexError) as identity_error:
            if strict:
                error = RuntimeError("Strict shutdown cannot establish the native leader identity")
                if primary_error is not None:
                    raise primary_error from error
                raise error from identity_error
    forced_kill = False
    errors = []
    inspection_errors = []
    before_forced_kill = []
    members, group_known = [], False
    unexpected_error = None

    def wait_for_workers(wait_seconds):
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                members, inspected = identity_group_members(process.pid)
            except (OSError, ValueError, IndexError) as error:
                members, inspected = [], [f"{type(error).__name__}: {error}"]
            inspection_errors.extend(inspected)
            if inspected:
                return members, False
            if not members or time.monotonic() >= deadline:
                return members, True
            time.sleep(0.01)

    def force_cleanup():
        nonlocal forced_kill, before_forced_kill, members, group_known
        if strict:
            errors.append("Strict shutdown retains the process owner without forced cleanup")
            return
        forced_kill = True
        before_forced_kill = members
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"Group SIGKILL failed: {type(error).__name__}: {error}")
        try:
            process.wait(timeout=min(timeout, 5))
        except subprocess.TimeoutExpired:
            errors.append("SGLang leader did not exit after SIGKILL")
        except OSError as error:
            errors.append(f"Forced leader wait failed: {type(error).__name__}: {error}")
        members, group_known = wait_for_workers(min(timeout, 5))

    try:
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f"SIGTERM failed: {type(error).__name__}: {error}")
        leader_wait_failed = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            leader_wait_failed = True
            errors.append("SGLang normal leader exit timed out")
        except OSError as error:
            leader_wait_failed = True
            errors.append(f"Leader wait failed: {type(error).__name__}: {error}")
        members, group_known = wait_for_workers(0 if leader_wait_failed else timeout)
        if leader_wait_failed or members or not group_known:
            force_cleanup()
    except BaseException as error:
        unexpected_error = error
        group_known = False
        errors.append(f"Unexpected cleanup error: {type(error).__name__}: {error}")
        try:
            force_cleanup()
        except BaseException as fallback_error:
            group_known = False
            errors.append(
                f"Forced cleanup error: {type(fallback_error).__name__}: {fallback_error}"
            )
            if isinstance(fallback_error, (KeyboardInterrupt, SystemExit)) and not isinstance(
                unexpected_error, (KeyboardInterrupt, SystemExit)
            ):
                unexpected_error = fallback_error

    if forced_kill:
        errors.append("SGLang required forced process-group cleanup")
    if process.returncode != 0:
        errors.append(f"SGLang leader exit code {process.returncode}")
    if members:
        errors.append("SGLang workers remain after cleanup")
    if inspection_errors:
        errors.append("SGLang process-group inspection failed")
    if not group_known:
        errors.append("SGLang process-group ownership is unknown")
    cleanup = {
        "pid": process.pid,
        "start_ticks": start_ticks,
        "exit_code": process.returncode,
        "sigterm_target": process.pid,
        "forced_kill": forced_kill,
        "before_forced_kill": before_forced_kill,
        "remaining_processes": members,
        "process_group_known": group_known,
        "reaped": process.returncode is not None,
        "interrupted": isinstance(unexpected_error, (KeyboardInterrupt, SystemExit)),
        "inspection_errors": inspection_errors,
        "errors": errors,
        "primary_failure": (
            {"type": type(primary_error).__name__, "message": str(primary_error)}
            if primary_error is not None
            else None
        ),
    }
    try:
        (directory / f"sglang-{process.pid}.cleanup.json").write_text(
            json.dumps(cleanup, indent=2) + "\n"
        )
        if isinstance(unexpected_error, (KeyboardInterrupt, SystemExit)):
            raise unexpected_error
        if errors:
            raise AssertionError(f"SGLang cleanup failed: {cleanup}") from unexpected_error
    except BaseException as cleanup_error:
        if primary_error is not None:
            raise primary_error from cleanup_error
        if (
            isinstance(unexpected_error, (KeyboardInterrupt, SystemExit))
            and cleanup_error is not unexpected_error
        ):
            raise unexpected_error from cleanup_error
        raise


@pytest.mark.parametrize(
    "channel_server",
    [
        pytest.param(
            {"tier": tier, "pool_size": "512mb", "ssd_cache_capacity": "8gb", "installed": True},
            id=tier,
        )
        for tier in ("dram", "ssd")
    ],
    indirect=True,
)
def test_sglang_direct_gpu_cache_recovery(channel_server, request, tmp_path):
    pytest.importorskip("sglang")
    model = request.config.getoption("--model")
    if not Path(model).exists():
        pytest.skip("pass --model with a local model path")

    if not any(
        entry.value == "orbitkv.sglang.plugin:register"
        for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
    ):
        plugin_dir = tmp_path / "orbitkv_source_plugin-0.0.dist-info"
        plugin_dir.mkdir()
        (plugin_dir / "METADATA").write_text("Name: orbitkv-source-plugin\nVersion: 0.0\n")
        (plugin_dir / "entry_points.txt").write_text(
            "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n"
        )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--trust-remote-code",
        "--kv-cache-dtype",
        request.config.getoption("--kv-cache-dtype"),
        "--load-format",
        request.config.getoption("--sglang-load-format"),
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--context-length",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-prefill-tokens",
        "2048",
        "--random-seed",
        "42",
        "--enable-deterministic-inference",
        "--page-size",
        "64",
        "--radix-cache-backend",
        "orbitkv",
        "--enable-unified-cache-external-linker",
        "--enable-cache-report",
    ]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["TORCH_EXTENSIONS_DIR"] = str(tmp_path / "torch-extensions")
    env["PYTHONPATH"] = os.pathsep.join(
        [env.get("PYTHONPATH", ""), str(PYTHON_ROOT), str(tmp_path)]
    )
    env["ORBITKV_SGLANG_ENDPOINT"] = f"unix://{channel_server.bootstrap_socket}"
    env["ORBITKV_TRANSFER_BACKEND"] = request.config.getoption("--orbitkv-transfer-backend")
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path / "flashinfer")
    log_path = tmp_path / "sglang-direct.log"

    def start_server(fingerprint: str | None = None) -> tuple[subprocess.Popen, str]:
        port = find_available_port()
        base_url = f"http://127.0.0.1:{port}"
        launch_cmd = list(cmd)
        launch_cmd[launch_cmd.index("--port") + 1] = str(port)
        # Torch's listener starts after worker initialization; avoid an
        # intervening outgoing connection claiming its selected port.
        launch_cmd += ["--nccl-port", str(find_available_port())]
        launch_env = dict(env)
        if fingerprint is not None:
            launch_env["ORBITKV_MODEL_FINGERPRINT"] = fingerprint
        with log_path.open("a") as log_file:
            process = subprocess.Popen(
                launch_cmd,
                env=launch_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=inherited_gpu_lock_fds(),
            )
            process._orbitkv_start_ticks = int(
                Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(f"SGLang exited during startup:\n{log_path.read_text()[-8000:]}")
                try:
                    if requests.get(f"{base_url}/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail(f"SGLang startup timed out:\n{log_path.read_text()[-8000:]}")
            return process, base_url
        except BaseException:
            _stop_server(process, tmp_path)
            raise

    from transformers import AutoTokenizer

    from tests.support.registration_campaign import record_process_group_runtime

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    fragment = tokenizer.encode("A CUDA IPC cache correctness test for a long context. ")
    # Keep one uncached token after a sealed recurrent checkpoint. Matching a
    # 512-token prompt excludes its last token and cannot use checkpoint 512.
    tokens = (fragment * (513 // len(fragment) + 1))[:513]
    payload = {
        "input_ids": tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
        "return_logprob": True,
    }
    process, base_url = start_server()
    try:
        first = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        first.raise_for_status()
        time.sleep(3)
        before_native = fetch_orbitkv_metrics(channel_server.http_port).get(
            "orbitkv_load_bytes_total", 0
        )
        native = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        native.raise_for_status()
        assert native.json()["meta_info"]["cached_tokens"] >= 512
        assert (
            fetch_orbitkv_metrics(channel_server.http_port).get("orbitkv_load_bytes_total", 0)
            == before_native
        )
        flushed = requests.post(f"{base_url}/flush_cache?timeout=30", timeout=40)
        flushed.raise_for_status()
        second = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        second.raise_for_status()
        assert second.json()["meta_info"]["cached_tokens"] >= 64, (
            second.json(),
            fetch_orbitkv_metrics(channel_server.http_port),
            channel_server.read_logs()[-8000:],
        )
        channel_server.record_runtime()
        record_process_group_runtime(
            process, process._orbitkv_start_ticks, log_path, time.monotonic() + 180
        )
    finally:
        _stop_server(process, tmp_path)

    # Keep the Cache Manager alive while SGLang's HBM prefix tree disappears.
    if channel_server.ssd_cache_path is not None:
        evict_dram_after_ssd_writes(channel_server.http_port)
    before_restart = fetch_orbitkv_metrics(channel_server.http_port)
    process, base_url = start_server()
    try:
        # One request can restore the shared prefix while the others still have
        # pending queries. Their new HBM hits must retire that obsolete interest.
        arrivals = Barrier(4)

        def restore(_):
            arrivals.wait(timeout=10)
            response = requests.post(f"{base_url}/generate", json=payload, timeout=90)
            response.raise_for_status()
            return response

        with ThreadPoolExecutor(max_workers=4) as executor:
            restored = list(executor.map(restore, range(4)))
        restored_tokens = [response.json()["meta_info"]["cached_tokens"] for response in restored]
        assert all(tokens >= 512 for tokens in restored_tokens), (
            restored_tokens,
            {
                name: value
                for name, value in fetch_orbitkv_metrics(channel_server.http_port).items()
                if name.startswith(("orbitkv_ssd_", "orbitkv_load_", "orbitkv_query_"))
            },
            channel_server.read_logs()[-8000:],
        )
        after_restart_load = fetch_orbitkv_metrics(channel_server.http_port).get(
            "orbitkv_load_bytes_total", 0
        )
        assert after_restart_load > before_restart.get("orbitkv_load_bytes_total", 0), (
            "Cache Manager did not restore GPU KV"
        )
        if channel_server.ssd_cache_path is not None:
            compression = request.config.getoption("--storage-codec") != "none"
            if channel_server.ssd_backend == "cufile":
                assert (
                    fetch_orbitkv_metrics(channel_server.http_port).get(
                        "orbitkv_ssd_cufile_write_bytes_total", 0
                    )
                    > 0
                )
            read_path = channel_server.ssd_read_path or channel_server.ssd_backend
            read_metrics = {
                "uring": ("orbitkv_ssd_prefetch_bytes_total",),
                "cufile": ("orbitkv_ssd_cufile_read_bytes_total",),
                "auto": ("orbitkv_ssd_prefetch_bytes_total", "orbitkv_ssd_cufile_read_bytes_total"),
            }[read_path]
            recovered = fetch_orbitkv_metrics(channel_server.http_port)
            if read_path == "cufile":
                assert recovered.get("orbitkv_ssd_prefetch_bytes_total", 0) == before_restart.get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                ), "cuFile recovery bounced through host SSD prefetch"
            elif read_path == "uring":
                assert recovered.get(
                    "orbitkv_ssd_cufile_read_bytes_total", 0
                ) == before_restart.get("orbitkv_ssd_cufile_read_bytes_total", 0), (
                    "io_uring recovery unexpectedly used cuFile reads"
                )
            if compression:
                assert recovered.get("orbitkv_storage_codec_duration_seconds_count", 0) > 0
                assert recovered.get("orbitkv_storage_codec_decode_failures_total", 0) == 0
                encoded = fetch_orbitkv_codec_bytes(channel_server.http_port)
                print(f"Encoded slot bytes: {encoded}")
                if request.config.getoption("--kv-cache-dtype") == "auto":
                    assert encoded["logical"] > 0, "no KV segments were encoded"
                    limit = 0.875 if request.config.getoption("--storage-codec") == "ans" else 0.55
                    assert encoded["stored"] <= encoded["logical"] * limit
            assert (
                sum(recovered.get(key, 0) - before_restart.get(key, 0) for key in read_metrics) > 0
            )
        record_process_group_runtime(
            process, process._orbitkv_start_ticks, log_path, time.monotonic() + 180
        )
    finally:
        _stop_server(process, tmp_path)

    # A changed model identity must miss even with identical page hashes and the
    # same model, sampling seed, and Cache Manager process.
    process, base_url = start_server(fingerprint="9" * 64)
    try:
        cold = requests.post(f"{base_url}/generate", json=payload, timeout=90)
        cold.raise_for_status()
        assert cold.json()["meta_info"]["cached_tokens"] == 0
        # Cold and prefix-reuse execution can differ numerically, even in native
        # SGLang. Compare the same computation shape, not two different baselines.
        for reference, responses in ((cold, (first,)), (native, (second, *restored))):
            expected_probs = [
                item[0] for item in reference.json()["meta_info"]["output_token_logprobs"]
            ]
            assert len(expected_probs) == 8 and all(map(math.isfinite, expected_probs))
            for response in responses:
                assert response.json()["text"] == reference.json()["text"]
                assert response.json()["output_ids"] == reference.json()["output_ids"]
                probabilities = [
                    item[0] for item in response.json()["meta_info"]["output_token_logprobs"]
                ]
                assert probabilities == pytest.approx(expected_probs, abs=0.05)
        registrations = [
            line
            for line in log_path.read_text().splitlines()
            if "OrbitKV GPU linker registered" in line
        ]
        assert len(registrations) >= 3
        assert all(
            f"transfer backend {env['ORBITKV_TRANSFER_BACKEND']}" in line for line in registrations
        )
        record_process_group_runtime(
            process, process._orbitkv_start_ticks, log_path, time.monotonic() + 180
        )
    finally:
        _stop_server(process, tmp_path)
