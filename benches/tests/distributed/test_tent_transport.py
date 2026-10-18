"""Legacy diagnostic C ABI drain and native-library comparison admission."""

import ctypes as c
import io
import json
from types import SimpleNamespace

import pytest

from benches.distributed import tent_compare, tent_stage


@pytest.mark.parametrize(
    ("polls", "submitted_failure", "timeout", "free_failures", "expected_failure"),
    [
        ([(0, 1), (0, 4)], None, 1000, 1, None),
        ([(0, 1), (0, 4)], "partial submit", 1000, 0, "partial submit"),
        ([(-1, 1), (0, 1), (0, 4)], None, 1000, 1, "poll: -1"),
        ([KeyboardInterrupt(), (0, 1), (0, 4)], None, 1000, 0, "KeyboardInterrupt"),
        ([(0, 99), (0, 4)], None, 1000, 0, "unknown status"),
        ([(0, 1), (0, 4)], None, 0, 0, "deadline"),
        *[([(0, status)], None, 1000, 0, "terminal") for status in (2, 3, 5, 6)],
    ],
)
def test_errors_keep_owner_until_terminal_and_accepted_free(
    monkeypatch, polls, submitted_failure, timeout, free_failures, expected_failure
):
    pending = iter(polls)
    state = {"terminal": False, "cancel_calls": 0, "free_calls": 0, "owner_held": True}

    def poll(engine, batch, task, pointer):
        assert state["owner_held"]
        assert (engine, batch, task) == (1, 2, 0)
        step = next(pending)
        if isinstance(step, BaseException):
            raise step
        rc, status = step
        result = c.cast(pointer, c.POINTER(tent_stage.Status)).contents
        result.status = status
        result.transferred_bytes = 4096
        state["terminal"] = rc == 0 and status in (2, 3, 4, 5, 6)
        return rc

    def cancel(*_):
        assert state["owner_held"]
        state["cancel_calls"] += 1
        return 0

    def free(*_):
        assert state["terminal"] and state["owner_held"]
        state["free_calls"] += 1
        if state["free_calls"] <= free_failures:
            return -1
        state["owner_held"] = False
        return 0

    monkeypatch.setattr(tent_stage.time, "sleep", lambda _: None)
    api = SimpleNamespace(tent_task_status=poll, tent_cancel_task=cancel, tent_free_batch=free)
    if expected_failure:
        with pytest.raises(RuntimeError, match=expected_failure):
            tent_stage.drain_batch(api, 1, 2, submitted_failure, timeout)
    else:
        result = tent_stage.drain_batch(api, 1, 2, submitted_failure, timeout)
        assert result["transferred_bytes"] == 4096
        assert result["poll_calls"] == len(polls)
    assert not state["owner_held"]
    assert state["free_calls"] == free_failures + 1
    assert state["cancel_calls"] == int(len(polls) > 1 and expected_failure is not None)


def test_pinned_tent_c_abi_layout_on_64_bit_hosts():
    assert c.sizeof(c.c_void_p) == 8
    assert c.sizeof(tent_stage.Request) == 48
    assert tent_stage.Request.hint.offset == 44
    assert c.sizeof(tent_stage.MemoryOptions) == 344
    assert tent_stage.MemoryOptions.internal.offset == 336
    assert c.sizeof(tent_stage.Status) == 16
    assert tent_stage.Status.transferred_bytes.offset == 8
    assert c.sizeof(tent_stage.Notification) == 4360
    assert tent_stage.Notification.msg.offset == 264
    assert c.sizeof(tent_stage.Notifications) == 16
    assert tent_stage.Notifications.records.offset == 8


@pytest.mark.parametrize("fault", ["corrupt", "duplicate", "native_error", "missing"])
def test_notification_validation_releases_every_native_record_allocation(monkeypatch, fault):
    record = (tent_stage.Notification * 1)(tent_stage.Notification(0, b"name", b"payload"))
    state = {"calls": 0, "frees": 0}

    def receive(engine, pointer):
        assert engine == 1
        info = c.cast(pointer, c.POINTER(tent_stage.Notifications)).contents
        assert info.num_records == 0 and not info.records
        state["calls"] += 1
        info.num_records = int(fault != "missing")
        info.records = record if info.num_records else None
        if fault == "corrupt":
            record[0].msg = b"wrong"
        return -1 if fault == "native_error" else 0

    def free(pointer):
        state["frees"] += 1

    monkeypatch.setattr(tent_stage.time, "sleep", lambda _: None)
    api = SimpleNamespace(tent_recv_notifs=receive, tent_free_notifs=free)
    with pytest.raises(RuntimeError):
        tent_stage.receive_notifications(api, 1, {b"name": b"payload"}, timeout=0.01)
    assert state["calls"] == state["frees"]


def test_exporter_control_errors_wait_for_explicit_source_stop(monkeypatch):
    monkeypatch.setattr(tent_stage.sys, "stdin", io.StringIO('invalid\n{"operation":"stop"}\n'))
    waiting = []
    monkeypatch.setattr(tent_stage.time, "sleep", lambda _: waiting.append(True))
    tent_stage.wait_source_stop()
    assert waiting == [True]


@pytest.mark.parametrize("failure", ["runtime_changed", "probe_missing"])
def test_prelaunch_failure_never_enters_native_cell_or_qualifies(tmp_path, monkeypatch, failure):
    contract = {
        "hosts": [{"alias": "test-host"}],
        "libraries": {"baseline": {}},
        "design": {},
        "guards": {},
        "remote_probe": "/frozen/probe.py",
        "input_hashes": {"probe": "expected"},
    }
    monkeypatch.setattr(
        tent_compare,
        "snapshot",
        lambda *_: {
            "gpu": "",
            "libraries": {} if failure == "runtime_changed" else {"baseline": {}},
        },
    )

    def missing(*_):
        raise RuntimeError("probe missing")

    monkeypatch.setattr(tent_compare, "ssh", missing)
    with pytest.raises(RuntimeError):
        tent_compare.run(contract, tmp_path)
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert result["state"] == "INVALID_PRELAUNCH"
    assert not result["cell_process_started"] and not result["independent_acceptance"]
    assert result["cells"] == 0 and result["held_owners"] == []
    assert "summary" not in result and "production_qualified" not in result
