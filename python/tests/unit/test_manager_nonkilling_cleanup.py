"""Real subprocess controls for task-owned Manager non-killing cleanup."""

import json
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


def test_strict_stop_retains_live_process_and_log_after_timeout(tmp_path, monkeypatch):
    process = child(ignore=True)
    owner = manager(process, tmp_path)
    owner._configured_log_path = tmp_path / "manager.log"
    owner._log_file = owner._configured_log_path.open("w")
    wait = process.wait
    monkeypatch.setenv("ORBITKV_TEST_STRICT_SHUTDOWN", "1")
    monkeypatch.setattr(process, "wait", lambda timeout: wait(timeout=0.05))
    try:
        with pytest.raises(RuntimeError, match="owner retained"):
            owner.stop()
        assert owner.process is process and process.poll() is None
        assert not owner._log_file.closed
        receipt = json.loads((tmp_path / "manager.log.cleanup.json").read_text())
        assert receipt["status"] == "quarantined"
        assert receipt["exit_code"] is None and not receipt["reaped"]
        assert not receipt["forced_kill"] and receipt["start_ticks"] > 0
    finally:
        monkeypatch.undo()
        finish(process, owner)
        owner._log_file.close()


@pytest.mark.parametrize("inspection", [False, True], ids=["remaining-worker", "unknown-group"])
def test_waited_leader_does_not_release_unresolved_group(tmp_path, monkeypatch, inspection):
    process = child()
    owner = manager(process, tmp_path)
    owner._log_file = (tmp_path / "manager.log").open("w")
    monkeypatch.setattr(
        "tests.support.cache_manager.identity_group_members",
        lambda _pid: ([], ["inspection denied"]) if inspection else ([{"pid": 42}], []),
    )
    try:
        receipt = owner.stop_without_kill(timeout=2)
        assert receipt["status"] == "quarantined"
        assert receipt["reason"] == "process_group_not_drained" and receipt["reaped"]
        assert owner.process is process and not owner._log_file.closed
    finally:
        monkeypatch.undo()
        finish(process, owner)
        owner._log_file.close()
