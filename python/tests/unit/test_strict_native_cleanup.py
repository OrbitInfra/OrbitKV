"""Healthy installed gates must retain ownership when native cleanup fails."""

import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.cache_manager import ClientContext
from tests.support.vllm_helpers import CacheManager, VLLMServer


@pytest.mark.parametrize("result", [False, RuntimeError("native failure")])
def test_strict_unregister_preserves_failed_registration(result):
    def unregister(_instance):
        if isinstance(result, Exception):
            raise result
        return result, "rejected"

    context = object.__new__(ClientContext)
    context._registered = True
    context.instance_id = "instance"
    context.client = SimpleNamespace(unregister_context=unregister)
    with pytest.raises(RuntimeError):
        context.unregister_context(strict=True)
    assert context._registered


@pytest.mark.parametrize("owner_type", [CacheManager, VLLMServer])
def test_strict_vllm_helpers_never_kill_live_owners(owner_type, tmp_path, monkeypatch):
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout.readline().strip() == "ready"
    owner = object.__new__(owner_type)
    owner.process = process
    owner._process_start_ticks = int(
        Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    owner.log_file = tmp_path / "owner.log"
    owner.log_handle = owner.log_file.open("w")
    wait = process.wait
    monkeypatch.setenv("ORBITKV_TEST_STRICT_SHUTDOWN", "1")
    monkeypatch.setattr(process, "wait", lambda timeout: wait(timeout=0.05))
    if owner_type is VLLMServer:
        values = iter([0, 100])
        monkeypatch.setattr("tests.support.vllm_helpers.time.monotonic", lambda: next(values))
    primary = ValueError("original failure")
    try:
        with pytest.raises(ValueError, match="original failure"):
            owner.__exit__(ValueError, primary, None)
        assert owner.process is process and process.poll() is None
        assert not owner.log_handle.closed
        receipt = json.loads(Path(str(owner.log_file) + ".cleanup.json").read_text())
        assert not receipt["forced_kill"]
        assert receipt["exit_code"] is None and not receipt["reaped"]
        assert receipt["start_ticks"] == owner._process_start_ticks
    finally:
        monkeypatch.undo()
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        owner.log_handle.close()


@pytest.mark.parametrize("failure", ["already-exited", "readiness-timeout", "receipt-write"])
def test_vllm_startup_failure_reaps_child_and_preserves_primary_error(
    tmp_path, monkeypatch, failure
):
    real_popen = subprocess.Popen
    children = []

    def launch(_argv, **kwargs):
        code = (
            "raise SystemExit(7)" if failure == "already-exited" else "import time; time.sleep(30)"
        )
        process = real_popen([sys.executable, "-B", "-c", code], **kwargs)
        children.append(process)
        return process

    primary = RuntimeError("controlled startup failure")

    def readiness_failed(owner, timeout):
        if failure == "already-exited":
            deadline = time.monotonic() + 5
            while True:
                state = (
                    Path(f"/proc/{owner.process.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
                if state == "Z":
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert owner.process.returncode is None
        raise primary

    monkeypatch.setenv("ORBITKV_TEST_STRICT_SHUTDOWN", "1")
    monkeypatch.delenv("ORBITKV_TEST_GPU_LOCK_FDS", raising=False)
    monkeypatch.setattr("tests.support.vllm_helpers.subprocess.Popen", launch)
    monkeypatch.setattr(VLLMServer, "_wait_for_ready", readiness_failed)
    owner = VLLMServer(model="/unused-cpu-model", port=1, log_file=tmp_path / "startup.log")
    write_text = Path.write_text
    if failure == "receipt-write":

        def fail_receipt(path, *args, **kwargs):
            if str(path).endswith(".cleanup.json"):
                raise OSError(errno.ENOSPC, "controlled receipt write failure")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_receipt)
    try:
        with pytest.raises(RuntimeError, match="controlled startup failure") as caught:
            owner.__enter__()
        assert caught.value is primary
        assert len(children) == 1
        process = children[0]
        assert process.returncode == (7 if failure == "already-exited" else -signal.SIGTERM)
        assert not Path(f"/proc/{process.pid}").exists()
        if failure == "receipt-write":
            assert (
                isinstance(primary.__cause__, OSError) and primary.__cause__.errno == errno.ENOSPC
            )
            assert owner.process is process and not owner.log_handle.closed
        else:
            receipt = json.loads(Path(str(owner.log_file) + ".cleanup.json").read_text())
            assert (
                receipt["pid"] == process.pid
                and receipt["start_ticks"] == owner._process_start_ticks
            )
            assert receipt["exit_code"] == process.returncode and receipt["reaped"] is True
            assert receipt["forced_kill"] is False and receipt["remaining_processes"] == []
            assert receipt["inspection_errors"] == []
            assert owner.process is None and owner.log_handle.closed
    finally:
        for process in children:
            if process.returncode is None:
                process.terminate()
                process.wait(timeout=5)
        if owner.log_handle is not None:
            owner.log_handle.close()
