"""Real subprocess controls for task-owned Manager non-killing cleanup."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.cache_manager import CacheManagerProcess


def child(ignore=False):
    code = "import signal,time;"
    if ignore:
        code += "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
    code += "print('ready',flush=True);time.sleep(30)"
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    assert process.stdout.readline().strip() == "ready"
    return process


def manager(process, tmp_path):
    owner = CacheManagerProcess(port=1, bootstrap_socket=str(tmp_path / "manager.sock"))
    owner.process = process
    owner._process_start_ticks = (
        Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    return owner


def finish(process, owner):
    # This is synthetic-control teardown, not graceful Manager/native drain.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    owner.process = None


def test_normal_cleanup_signals_and_reaps(tmp_path):
    process = child()
    owner = manager(process, tmp_path)
    try:
        record = owner.stop_without_kill(timeout=2)
        assert record["status"] == "exited"
        assert record["sigterm_sent"] and record["reaped"]
        assert not record["forced_cleanup"]
        assert record["returncode"] == -signal.SIGTERM
        assert owner.process is None
    finally:
        finish(process, owner)


def test_timeout_retains_owner_and_never_resignals(tmp_path, monkeypatch):
    process = child(ignore=True)
    owner = manager(process, tmp_path)
    try:
        record = owner.stop_without_kill(timeout=0.05)
        assert record["status"] == "quarantined" and record["reason"] == "termination_timeout"
        assert owner.process is process and process.poll() is None
        assert not record["forced_cleanup"] and not record["reaped"]
        monkeypatch.setattr(
            os, "killpg", lambda *_: pytest.fail("must not signal after quarantine")
        )
        repeat = owner.stop_without_kill(timeout=0.05)
        assert repeat["reason"] == "prior_graceful_shutdown_timeout"
        assert not repeat["sigterm_sent"]
    finally:
        monkeypatch.undo()
        finish(process, owner)


def test_identity_mismatch_refuses_signal(tmp_path, monkeypatch):
    process = child()
    owner = manager(process, tmp_path)
    owner._process_start_ticks = "0"
    try:
        monkeypatch.setattr(os, "killpg", lambda *_: pytest.fail("must not signal wrong identity"))
        record = owner.stop_without_kill(timeout=0.05)
        assert record["status"] == "quarantined" and record["reason"] == "process_identity_changed"
        assert owner.process is process and process.poll() is None
        assert not record["sigterm_sent"] and not record["reaped"]
    finally:
        monkeypatch.undo()
        finish(process, owner)
