"""Healthy installed gates must retain ownership when native cleanup fails."""

import json
import os
import signal
import subprocess
import sys
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
