"""Service teardown must retain caller-owned GPU exports until actual exit."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benches import runtime


def test_server_rejects_a_different_proc_pid_namespace_before_spawning(monkeypatch, tmp_path):
    actual_pid = os.getpid()
    monkeypatch.setattr(runtime.os, "getpid", lambda: actual_pid + 1)

    def launch(*args, **kwargs):
        pytest.fail("service launched before checking the PID namespace")

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    with (
        pytest.raises(RuntimeError, match="PID namespace"),
        runtime.server(["manager"], {}, "http://manager", tmp_path / "manager.log"),
    ):
        pass
    assert not (tmp_path / "manager.log").exists()


@pytest.mark.parametrize(
    ("steps", "body_failure", "expected_error", "broken_stderr"),
    [
        (["exit"], False, None, False),
        (["timeout", "timeout", "exit"], True, RuntimeError, False),
        (["timeout", "timeout", "exit"], True, RuntimeError, True),
        (["signal", "interrupt", "signal", "exit"], False, KeyboardInterrupt, False),
    ],
)
def test_server_reaps_before_caller_releases_exports_even_after_kill_timeouts_and_interrupts(
    monkeypatch, tmp_path, steps, body_failure, expected_error, broken_stderr
):
    state = {"owner_held": True, "reaped": False}
    pending = iter(steps)
    sent = []
    handler = {"current": signal.default_int_handler}

    class Process:
        pid = 123456

        def poll(self):
            return None

        def wait(self, timeout):
            assert state["owner_held"], "caller released its export before service exit"
            step = next(pending)
            if step == "signal":
                handler["current"](signal.SIGINT, None)
                raise subprocess.TimeoutExpired("manager", timeout)
            if step == "timeout":
                raise subprocess.TimeoutExpired("manager", timeout)
            if step == "interrupt":
                raise KeyboardInterrupt
            state["reaped"] = True
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(runtime.signal, "getsignal", lambda _: handler["current"])
    monkeypatch.setattr(runtime.signal, "signal", lambda _, value: handler.update(current=value))
    if broken_stderr:

        def fail_write(_text):
            raise BrokenPipeError("log reader closed")

        monkeypatch.setattr(runtime.sys, "stderr", SimpleNamespace(write=fail_write))
    expectation = pytest.raises(expected_error) if expected_error else contextlib.nullcontext()
    with expectation:
        try:
            with runtime.server(["manager"], {}, "http://manager", tmp_path / "manager.log"):
                if body_failure:
                    raise RuntimeError("GPU completion was not established")
        finally:
            assert state["reaped"], "teardown unwound with a live service"
            state["owner_held"] = False

    assert list(pending) == []
    assert sent[0] == (Process.pid, signal.SIGTERM)
    assert all(item == (Process.pid, signal.SIGKILL) for item in sent[1:])
    assert handler["current"] is signal.default_int_handler
    cleanup = json.loads((tmp_path / "manager.cleanup.json").read_text())
    assert cleanup["pid"] == Process.pid
    assert cleanup["exit_code"] == 0 and cleanup["reaped"]
    assert cleanup["forced_kill"] == (len(sent) > 1)


@pytest.mark.parametrize("failure", [None, "native_error", "log_missing", "body_and_native_error"])
def test_real_engine_parent_owns_child_shutdown_and_native_errors_invalidate_measurement(
    monkeypatch, tmp_path, failure
):
    ready = tmp_path / "ready"
    log = tmp_path / "engine.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        """
import signal, subprocess, sys
from pathlib import Path

child = subprocess.Popen(
    [sys.executable, '-u', '-c', 'print("ready", flush=True); input()'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)
assert child.stdout.readline().strip() == 'ready'

def stop(signum, frame):
    child.stdin.write('stop\\n')
    child.stdin.flush()
    status = child.wait(timeout=5)
    if status != 0:
        raise RuntimeError(f'child received group SIGTERM: {status}')
    if sys.argv[2] in {'native_error', 'body_and_native_error'}:
        print('EngineDeadError: intentional failure control', flush=True)
    print('API server: engine client stopped', flush=True)
    print('Application shutdown complete.', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text(str(child.pid))
while True:
    signal.pause()
""",
        str(ready),
        str(failure),
    ]
    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    error = RuntimeError if failure == "body_and_native_error" else ValueError if failure else None
    expectation = pytest.raises(error) if error else contextlib.nullcontext()
    with (
        expectation as caught,
        runtime.server(command, dict(os.environ), "http://cpu-control", log, engine="vllm"),
    ):
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), log.read_text()
        if failure == "log_missing":
            log.unlink()
        if failure == "body_and_native_error":
            raise RuntimeError("original workload failure")
    if failure == "body_and_native_error":
        assert str(caught.value) == "original workload failure"
    cleanup = json.loads(log.with_suffix(".cleanup.json").read_text())
    assert cleanup["exit_code"] == 0 and cleanup["reaped"] and not cleanup["forced_kill"]
    assert cleanup["remaining_group"] == []
    assert ("shutdown_error" in cleanup) == bool(failure)
    assert not Path(f"/proc/{ready.read_text()}").exists(), "child was not reaped by its engine"


@pytest.mark.parametrize(
    "fault", ["forced_group_cleanup", "unreaped_child", "early_exit", "nonzero_exit"]
)
def test_leader_exit_alone_does_not_establish_successful_service_cleanup(
    monkeypatch, tmp_path, fault
):
    sent = []
    polls = iter([None, 0 if fault == "early_exit" else None])
    groups = iter(
        [[(123457, "S")], []]
        if fault == "forced_group_cleanup"
        else [[(123457, "Z")]]
        if fault == "unreaped_child"
        else [[]]
    )
    clock = iter(range(100, 1000, 100))
    owner_held = True

    class Process:
        pid = 123456

        def poll(self):
            return next(polls)

        def wait(self, timeout):
            return 7 if fault == "nonzero_exit" else 0

    def members(pgid):
        assert owner_held, "live descendants outlasted the caller's exports"
        return next(groups)

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(runtime.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(runtime, "_process_group_members", members)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    with (
        pytest.raises(RuntimeError, match="Service shutdown failed"),
        runtime.server(["manager"], {}, "http://manager", tmp_path / "manager.log"),
    ):
        pass
    owner_held = False
    cleanup = json.loads((tmp_path / "manager.cleanup.json").read_text())
    assert cleanup["reaped"]
    assert cleanup["forced_kill"] == (fault == "forced_group_cleanup")
    assert cleanup["remaining_group"] == ([123457] if fault == "unreaped_child" else [])
    assert (Process.pid, signal.SIGKILL) in sent if fault == "forced_group_cleanup" else True
    assert not sent if fault == "early_exit" else sent[0] == (Process.pid, signal.SIGTERM)


@pytest.mark.parametrize("body_failure", [False, True])
def test_proc_inspection_failure_retains_ownership_until_real_orphan_is_killed_and_reaped(
    monkeypatch, tmp_path, body_failure
):
    ready = tmp_path / "ready"
    log = tmp_path / "engine.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        """
import signal, subprocess, sys
from pathlib import Path

child = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])

def stop(signum, frame):
    print('API server: engine client stopped', flush=True)
    print('Application shutdown complete.', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text(str(child.pid))
while True:
    signal.pause()
""",
        str(ready),
    ]
    owner_held = True
    child_pid = None
    inspect = runtime._process_group_members

    def denied(pgid):
        assert owner_held, "inspection failed after the caller released its exports"
        if child_pid is not None and Path(f"/proc/{child_pid}").exists():
            raise PermissionError(13, "controlled worker stat failure", f"/proc/{child_pid}/stat")
        return inspect(pgid)

    monkeypatch.setattr(runtime.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    monkeypatch.setattr(runtime, "_process_group_members", denied)
    expected = "original workload failure" if body_failure else "Service shutdown failed"
    try:
        with (
            pytest.raises(RuntimeError, match=expected),
            runtime.server(command, dict(os.environ), "http://cpu-control", log, engine="vllm"),
        ):
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), log.read_text()
            child_pid = int(ready.read_text())
            if body_failure:
                raise RuntimeError("original workload failure")
    finally:
        owner_held = False
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child_pid, 0)
    cleanup = json.loads(log.with_suffix(".cleanup.json").read_text())
    assert cleanup["exit_code"] == 0 and cleanup["reaped"]
    assert cleanup["forced_kill"] and cleanup["process_group_errors"] >= 1
    assert cleanup["remaining_group"] == []
    assert cleanup["adopted_children"] == [{"pid": child_pid, "exit_code": -signal.SIGKILL}]
    assert not Path(f"/proc/{child_pid}").exists()
