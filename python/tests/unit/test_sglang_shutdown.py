"""Real process controls for the SGLang E2E shutdown owner."""

import contextlib
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e import test_sglang_direct_e2e as sglang_direct

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process ownership")

CHILD = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGUSR1, lambda *_: os._exit(0))
Path(sys.argv[1]).write_text('ready')
while True:
    time.sleep(1)
"""

LEADER = """
import json, os, signal, subprocess, sys, time
from pathlib import Path
mode, child_source, ready, child_ready, fd = sys.argv[1:]
child = subprocess.Popen([sys.executable, '-B', '-c', child_source, child_ready],
                         pass_fds=(int(fd),))
def shutdown(*_):
    if mode in ('clean', 'leader_nonzero'):
        child.send_signal(signal.SIGUSR1)
        child.wait(timeout=5)
    sys.exit(3 if mode == 'leader_nonzero' else 0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == 'leader_timeout' else shutdown)
deadline = time.monotonic() + 5
while not Path(child_ready).exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('worker readiness timed out')
    time.sleep(0.01)
Path(ready).write_text(json.dumps({'child': child.pid}))
while True:
    time.sleep(1)
"""


@pytest.fixture
def owned_server(request, tmp_path):
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    process = None
    child_pid = None
    lock_path = tmp_path / "gpu-lease.lock"
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready = tmp_path / "leader-ready.json"
            mode = "orphan" if request.param == "inspection" else request.param
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    LEADER,
                    mode,
                    CHILD,
                    str(ready),
                    str(tmp_path / "worker-ready"),
                    str(lock.fileno()),
                ],
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
            deadline = time.monotonic() + 5
            while not ready.exists():
                assert process.poll() is None, "leader exited before readiness"
                assert time.monotonic() < deadline, "leader readiness timed out"
                time.sleep(0.01)
            child_pid = json.loads(ready.read_text())["child"]
        yield process, child_pid, lock_path, request.param
    finally:
        try:
            children = {child_pid} if child_pid is not None else set()
            if process is not None:
                members, _ = sglang_direct.identity_group_members(process.pid)
                children.update(member["pid"] for member in members if member["pid"] != process.pid)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            for child in children:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        waited, _ = os.waitpid(child, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if waited == child:
                        break
                    assert time.monotonic() < deadline, "adopted worker was not reaped"
                    time.sleep(0.01)
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.parametrize(
    "owned_server",
    ["clean", "orphan", "inspection", "leader_timeout", "leader_nonzero"],
    indirect=True,
)
def test_sglang_shutdown_rejects_incomplete_ownership(owned_server, tmp_path, monkeypatch):
    process, child_pid, lock_path, mode = owned_server
    if mode == "inspection":
        inspect = sglang_direct.identity_group_members
        injected = False

        def inspect_once(group):
            nonlocal injected
            members, errors = inspect(group)
            if not injected:
                injected = True
                errors.append("injected proc inspection failure")
            return members, errors

        monkeypatch.setattr(sglang_direct, "identity_group_members", inspect_once)
    if mode == "clean":
        sglang_direct._stop_server(process, tmp_path, timeout=0.5)
    else:
        with pytest.raises(AssertionError, match="SGLang cleanup failed"):
            sglang_direct._stop_server(process, tmp_path, timeout=0.5)
    cleanup = json.loads((tmp_path / f"sglang-{process.pid}.cleanup.json").read_text())
    assert cleanup["sigterm_target"] == process.pid
    assert cleanup["forced_kill"] == (mode in {"orphan", "inspection", "leader_timeout"})
    assert bool(cleanup["errors"]) == (mode != "clean")
    assert cleanup["primary_failure"] is None
    assert cleanup["exit_code"] == {"leader_timeout": -9, "leader_nonzero": 3}.get(mode, 0)
    if mode in {"orphan", "inspection", "leader_timeout"}:
        assert any(member["pid"] == child_pid for member in cleanup["before_forced_kill"])
    else:
        assert cleanup["before_forced_kill"] == cleanup["remaining_processes"] == []
    assert bool(cleanup["inspection_errors"]) == (mode == "inspection")
    with lock_path.open("a") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("owned_server", ["orphan"], indirect=True)
def test_sglang_shutdown_preserves_the_foreground_failure(owned_server, tmp_path):
    process, _, lock_path, _ = owned_server
    primary = ValueError("foreground output oracle failed")
    with pytest.raises(ValueError, match="foreground output oracle failed") as failed:
        try:
            raise primary
        finally:
            sglang_direct._stop_server(process, tmp_path, timeout=0.5)
    assert failed.value is primary and isinstance(primary.__cause__, AssertionError)
    cleanup = json.loads((tmp_path / f"sglang-{process.pid}.cleanup.json").read_text())
    assert cleanup["forced_kill"] and cleanup["errors"]
    assert cleanup["primary_failure"] == {"type": "ValueError", "message": str(primary)}
    with lock_path.open("a") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    ("owned_server", "operation"),
    [
        ("clean", "signal"),
        ("clean", "wait"),
        ("orphan", "group_kill"),
        ("orphan", "inspection"),
    ],
    indirect=["owned_server"],
)
def test_sglang_shutdown_records_early_operation_errors(
    owned_server, operation, tmp_path, monkeypatch
):
    process, child_pid, lock_path, _ = owned_server
    primary = ValueError("foreground output oracle failed")

    def deny_operation(*args, **kwargs):
        raise PermissionError(f"denied {operation}")

    with monkeypatch.context() as patch:
        if operation == "group_kill":
            patch.setattr(sglang_direct.os, "killpg", deny_operation)
        elif operation == "inspection":
            patch.setattr(sglang_direct, "identity_group_members", deny_operation)
        else:
            patch.setattr(
                process, "send_signal" if operation == "signal" else "wait", deny_operation
            )
        with pytest.raises(ValueError, match="foreground output oracle failed") as failed:
            try:
                raise primary
            finally:
                sglang_direct._stop_server(process, tmp_path, timeout=0.2)
    assert failed.value is primary and isinstance(primary.__cause__, AssertionError)
    cleanup = json.loads((tmp_path / f"sglang-{process.pid}.cleanup.json").read_text())
    reported_errors = cleanup["errors"] + cleanup["inspection_errors"]
    assert any(f"denied {operation}" in error for error in reported_errors)
    assert cleanup["forced_kill"]
    assert cleanup["process_group_known"] == (operation != "inspection")
    assert cleanup["primary_failure"] == {"type": "ValueError", "message": str(primary)}
    if operation == "group_kill":
        assert any(member["pid"] == child_pid for member in cleanup["remaining_processes"])
        with lock_path.open("a") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        assert cleanup["reaped"] == (operation != "wait")
        if operation == "inspection":
            assert "SGLang process-group ownership is unknown" in cleanup["errors"]
        with lock_path.open("a") as probe:
            deadline = time.monotonic() + 5
            while True:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "signalled worker still holds its lease"
                    time.sleep(0.01)


@pytest.mark.parametrize("owned_server", ["orphan"], indirect=True)
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("operation", ["signal", "wait", "inspection", "primary_failure"])
def test_sglang_shutdown_interrupt_stops_workers_and_releases_lease(
    owned_server, interrupt_type, operation, tmp_path, monkeypatch
):
    process, child_pid, lock_path, _ = owned_server
    interrupt = interrupt_type("shutdown interrupted")
    primary = ValueError("foreground output oracle failed")
    target, attribute = (
        (sglang_direct, "identity_group_members")
        if operation in {"inspection", "primary_failure"}
        else (process, "send_signal" if operation == "signal" else "wait")
    )
    original = getattr(target, attribute)
    interrupted = False

    def interrupt_once(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise interrupt
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, attribute, interrupt_once)
        if operation == "primary_failure":
            with pytest.raises(ValueError) as failed:
                try:
                    raise primary
                finally:
                    sglang_direct._stop_server(process, tmp_path, timeout=0.5)
            assert failed.value is primary and primary.__cause__ is interrupt
        else:
            with pytest.raises(interrupt_type) as failed:
                sglang_direct._stop_server(process, tmp_path, timeout=0.5)
            assert failed.value is interrupt

    assert process.returncode is not None
    cleanup = json.loads((tmp_path / f"sglang-{process.pid}.cleanup.json").read_text())
    assert cleanup["interrupted"] and cleanup["forced_kill"] and cleanup["reaped"]
    assert cleanup["process_group_known"]
    assert all(member["state"] == "Z" for member in cleanup["remaining_processes"])
    assert any(interrupt_type.__name__ in error for error in cleanup["errors"])
    assert cleanup["primary_failure"] == (
        {"type": "ValueError", "message": str(primary)} if operation == "primary_failure" else None
    )
    with lock_path.open("a") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    waited, status = os.waitpid(child_pid, 0)
    assert waited == child_pid and os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL


@pytest.mark.parametrize("owned_server", ["clean", "orphan", "leader_timeout"], indirect=True)
def test_strict_shutdown_keeps_unknown_owner_and_lease(owned_server, tmp_path, monkeypatch):
    process, child_pid, lock_path, mode = owned_server
    monkeypatch.setenv("ORBITKV_TEST_STRICT_SHUTDOWN", "1")
    if mode == "clean":
        sglang_direct._stop_server(process, tmp_path, timeout=2)
    else:
        with pytest.raises(AssertionError, match="SGLang cleanup failed"):
            sglang_direct._stop_server(process, tmp_path, timeout=0.05)
        assert Path(f"/proc/{child_pid}").exists()
        if mode == "leader_timeout":
            assert process.poll() is None
        with lock_path.open("a") as lock, pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = json.loads((tmp_path / f"sglang-{process.pid}.cleanup.json").read_text())
    assert not receipt["forced_kill"] and receipt["start_ticks"] > 0
    assert bool(receipt["errors"]) == (mode != "clean")
