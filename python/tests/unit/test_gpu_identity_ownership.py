import ctypes
import errno
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import gpu_identity
from tests.support.gpu_identity import (
    drain_identity_processes,
    identity_gpu_locks,
    quarantine_identity_gpus,
    require_clean_identity_log,
)


def test_gpu_lock_survives_controller_exit_until_inheriting_owner_exits(tmp_path):
    with identity_gpu_locks(tmp_path, ["7"]) as descriptors:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            pass_fds=descriptors,
        )
        assert child.stdout.readline() == "ready\n"
    try:
        with (
            (tmp_path / "gpu-7.lock").open("a") as competing,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.communicate("release\n", timeout=5)
    assert child.returncode == 0
    with identity_gpu_locks(tmp_path, ["7"]):
        pass


def test_unknown_client_drain_preserves_owner_and_blocks_next_admission(tmp_path):
    release = tmp_path / "release"
    manager_stops = []
    with identity_gpu_locks(tmp_path, ["7"]) as descriptors:
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            pass_fds=descriptors,
        )
        assert child.stdout.readline() == "ready\n"
        try:
            manager = SimpleNamespace(
                process=None, terminate_gracefully=lambda **_: manager_stops.append("SIGTERM")
            )
            cleanup = drain_identity_processes(child, manager, release, timeout=0.02)
            assert cleanup["remaining_processes"][0]["pid"] == child.pid
            assert child.poll() is None
            assert not manager_stops
            quarantine_identity_gpus(tmp_path, ["7"], tmp_path, cleanup)
        finally:
            child.communicate("release\n", timeout=5)
    with pytest.raises(RuntimeError, match="quarantined"), identity_gpu_locks(tmp_path, ["7"]):
        pytest.fail("A quarantined GPU was admitted")


@pytest.mark.parametrize(
    "failure",
    [
        "ERROR physical drain failed",
        "thread panicked at drain",
        "Exception ignored in: cleanup\nTraceback (most recent call last):\nRuntimeError: drain",
    ],
)
def test_zero_exit_does_not_hide_manager_lifecycle_failure(failure):
    with pytest.raises(AssertionError):
        require_clean_identity_log(f"INFO normal work\n{failure}\nINFO shutdown complete")


@pytest.mark.parametrize(
    ("failure", "inspection_failed"),
    [
        pytest.param(FileNotFoundError(errno.ENOENT, "gone"), False, id="disappeared-path"),
        pytest.param(ProcessLookupError(errno.ESRCH, "gone"), False, id="disappeared-process"),
        pytest.param(PermissionError(errno.EACCES, "denied"), True, id="denied-inspection"),
        pytest.param(OSError(errno.EIO, "unreadable"), True, id="failed-inspection"),
    ],
)
def test_disappearing_process_scan_keeps_live_owners_and_inspection_errors(
    monkeypatch, failure, inspection_failed
):
    fields = ["S", "1", "77", *(["0"] * 16), "42"]

    class ProcessStat:
        def __init__(self, pid, error=None):
            self.parent = SimpleNamespace(name=str(pid))
            self.error = error

        def read_text(self):
            if self.error is not None:
                raise self.error
            return f"{self.parent.name} (live owner) " + " ".join(fields)

        def __str__(self):
            return f"/proc/{self.parent.name}/stat"

    root = SimpleNamespace(glob=lambda _: [ProcessStat(123), ProcessStat(456, failure)])
    monkeypatch.setattr(gpu_identity, "Path", lambda _: root)
    members, errors = gpu_identity.identity_group_members(77)
    assert members == [{"pid": 123, "state": "S", "start_ticks": 42}]
    assert bool(errors) is inspection_failed
    if inspection_failed:
        assert "/proc/456/stat" in errors[0]


def test_inspection_failure_preserves_quarantine_record(monkeypatch, tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready',flush=True);sys.stdin.readline()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert child.stdout.readline() == "ready\n"

    class Denied:
        def __truediv__(self, _):
            return self

        def read_text(self):
            raise PermissionError("CPU control: process inspection denied")

    monkeypatch.setattr(
        gpu_identity,
        "Path",
        lambda value: Denied() if str(value) == f"/proc/{child.pid}" else Path(value),
    )
    try:
        cleanup = drain_identity_processes(
            child, SimpleNamespace(process=None), tmp_path / "release", timeout=0.02
        )
        assert cleanup["remaining_processes"]
        assert any("inspection" in error for error in cleanup["errors"])
        quarantine_identity_gpus(tmp_path, ["7"], tmp_path, cleanup)
        assert (tmp_path / "gpu-7.quarantine.json").exists()
    finally:
        child.communicate("release\n", timeout=5)
    assert child.returncode == 0


@pytest.mark.parametrize("role", ["client", "manager"])
def test_clean_leader_exit_does_not_hide_live_descendant(tmp_path, role):
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    child_pid = None
    release = tmp_path / "child-release"
    try:
        with identity_gpu_locks(tmp_path, ["7"]) as descriptors:
            child_code = "import sys,time;from pathlib import Path\nwhile not Path(sys.argv[1]).exists():time.sleep(0.01)"
            leader_code = "import subprocess,sys,json;from pathlib import Path\np=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],pass_fds=tuple(json.loads(sys.argv[4])))\nPath(sys.argv[3]).write_text(str(p.pid))"
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    leader_code,
                    child_code,
                    str(release),
                    str(tmp_path / "pid"),
                    json.dumps(descriptors),
                ],
                pass_fds=descriptors,
                start_new_session=True,
            )
            assert leader.wait(timeout=5) == 0
            child_pid = int((tmp_path / "pid").read_text())
            manager = SimpleNamespace(process=leader if role == "manager" else None)

            def terminate_gracefully(**_):
                manager.process = None
                return leader.wait(timeout=5), 0.0

            manager.terminate_gracefully = terminate_gracefully
            cleanup = drain_identity_processes(
                leader if role == "client" else None,
                manager,
                tmp_path / "client-release",
                timeout=0.02,
            )
            assert any(
                process.get("pid") == child_pid for process in cleanup["remaining_processes"]
            )
            assert cleanup["errors"]
            quarantine_identity_gpus(tmp_path, ["7"], tmp_path, cleanup)
    finally:
        release.touch()
        if child_pid is not None:
            deadline = time.monotonic() + 5
            while True:
                waited, status = os.waitpid(child_pid, os.WNOHANG)
                if waited == child_pid:
                    assert os.waitstatus_to_exitcode(status) == 0
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0
    with pytest.raises(RuntimeError, match="quarantined"), identity_gpu_locks(tmp_path, ["7"]):
        pytest.fail("A live descendant was treated as drained")
