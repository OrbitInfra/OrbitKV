"""CPU controls for launch admission and strict lifecycle evidence."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import registration_campaign as campaign


def test_cpu_ready_owner_holds_then_normally_releases_real_resource_locks(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gpu = fake_bin / "nvidia-smi"
    gpu.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('GPU-fixture, 0' if '--query-gpu=uuid,memory.used' in sys.argv else '')\n"
    )
    gpu.chmod(0o755)
    lease = tmp_path / "gpu.lock"
    lease.touch()
    protected_lease = tmp_path / "protected.lock"
    protected_lease.touch()
    protected = tmp_path / "protected"
    protected.write_text("CPU fixture; no CUDA work")
    contract = {
        "schema": "orbitkv.explicit-registration-campaign.v1",
        "phase": "precheck",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "input_sha256": {},
        "protected_gpu0": {
            "quarantine": str(protected),
            "quarantine_sha256": campaign.digest(protected),
            "lease": str(protected_lease),
            "lease_inode": protected_lease.stat().st_ino,
            "guardian_pid": os.getpid(),
            "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
        },
        "locks": [
            {"path": str(lease), "device": lease.stat().st_dev, "inode": lease.stat().st_ino}
        ],
        "quarantine": str(tmp_path / "absent-quarantine"),
        "gpu_uuid": "GPU-fixture",
        "cpus": sorted(os.sched_getaffinity(0))[:1],
        "review_wait_seconds": 1,
        "campaign_seconds": 1,
        "cleanup_seconds": 1,
        "cells": [{"kind": "native", "passed": 2}],
    }
    path = tmp_path / "CONTRACT.json"
    path.write_text(json.dumps(contract))
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    process = subprocess.Popen(
        [sys.executable, "-B", campaign.__file__, str(path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not (tmp_path / "READY.json").exists():
        assert process.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with lease.open("a") as contender, pytest.raises(BlockingIOError):
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert json.loads((tmp_path / "NOT-LAUNCHED.json").read_text())["GPU_launches"] == 0
    with lease.open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not Path(f"/proc/{process.pid}").exists()


@pytest.mark.parametrize("fault", ["missing", "forced", "not-reaped", "bool-exit", "traceback"])
def test_successful_junit_does_not_override_missing_native_ownership_evidence(tmp_path, fault):
    (tmp_path / "junit.xml").write_text("<testsuite><testcase name='passed'/></testsuite>")
    row = {
        "pid": 17,
        "start_ticks": 18,
        "forced_kill": False,
        "reaped": True,
        "exit_code": 0,
        "remaining_processes": [],
        "inspection_errors": [],
    }
    if fault == "forced":
        row["forced_kill"] = True
    elif fault == "not-reaped":
        row["reaped"] = False
    elif fault == "bool-exit":
        row["exit_code"] = False
    if fault != "missing":
        (tmp_path / "manager.log.cleanup.json").write_text(json.dumps(row))
    (tmp_path / "manager.log").write_text("CPU native receipt fixture")
    (tmp_path / "pytest.log").write_text("CPU pytest fixture")
    if fault == "traceback":
        (tmp_path / "worker.log").write_text("Traceback (most recent call last): device failure")
    cell = {
        "passed": 1,
        "cleanup_receipts": 1,
        "normal_exit_codes": [0],
        "kind": "vllm-worker",
        "adopted_exit_codes": [0],
        "log_sets": [{"pattern": "pytest.log", "count": 1}, {"pattern": "manager.log", "count": 1}],
    }
    with pytest.raises(RuntimeError):
        campaign.validate_cell(
            cell,
            tmp_path,
            {"pid": 1, "start_ticks": 2},
            [{"pid": 1, "start_ticks": 2}, {"pid": 17, "start_ticks": 18}],
        )


def valid_evidence(directory):
    (directory / "junit.xml").write_text("<testsuite><testcase name='passed'/></testsuite>")
    (directory / "pytest.log").write_text("CPU local validator fixture")
    (directory / "manager.log").write_text("CPU local service fixture")
    row = {
        "pid": 17,
        "start_ticks": 18,
        "forced_kill": False,
        "reaped": True,
        "exit_code": 0,
        "remaining_processes": [],
        "inspection_errors": [],
    }
    (directory / "manager.log.cleanup.json").write_text(json.dumps(row))
    owner = {"pid": 1, "start_ticks": 2}
    observed = [owner, {"pid": 17, "start_ticks": 18}]
    cell = {
        "passed": 1,
        "cleanup_receipts": 1,
        "normal_exit_codes": [0],
        "kind": "vllm-worker",
        "adopted_exit_codes": [0],
        "log_sets": [{"pattern": "pytest.log", "count": 1}, {"pattern": "manager.log", "count": 1}],
    }
    return cell, owner, observed


@pytest.mark.parametrize("adopted", [False, True])
def test_complete_observed_receipts_and_logs_can_pass(tmp_path, adopted):
    cell, owner, observed = valid_evidence(tmp_path)
    if adopted:
        child = {"pid": 19, "start_ticks": 20}
        observed.append(child)
        (tmp_path / "ADOPTED-19.json").write_text(
            json.dumps(dict(child, exit_code=0, actual_wait=True))
        )
    campaign.validate_cell(cell, tmp_path, owner, observed, time.monotonic() + 1)


@pytest.mark.parametrize("exit_code", [1, -9, False])
def test_adopted_nonzero_or_noninteger_exit_cannot_pass(tmp_path, exit_code):
    cell, owner, observed = valid_evidence(tmp_path)
    child = {"pid": 19, "start_ticks": 20}
    observed.append(child)
    (tmp_path / "ADOPTED-19.json").write_text(
        json.dumps(dict(child, exit_code=exit_code, actual_wait=True))
    )
    with pytest.raises(RuntimeError, match="Adopted native exit"):
        campaign.validate_cell(cell, tmp_path, owner, observed)


@pytest.mark.parametrize(
    "fault", ["missing-log", "duplicate-owner", "unobserved-owner", "unaccounted-child"]
)
def test_logs_and_unique_observed_owners_are_required(tmp_path, fault):
    cell, owner, observed = valid_evidence(tmp_path)
    if fault == "missing-log":
        (tmp_path / "manager.log").unlink()
    elif fault == "duplicate-owner":
        cell["cleanup_receipts"] = 2
        (tmp_path / "other.log.cleanup.json").write_bytes(
            (tmp_path / "manager.log.cleanup.json").read_bytes()
        )
    elif fault == "unobserved-owner":
        observed.pop()
    else:
        observed.append({"pid": 19, "start_ticks": 20})
    with pytest.raises(RuntimeError):
        campaign.validate_cell(cell, tmp_path, owner, observed)


def test_same_content_lock_replacement_rejected_while_old_inode_is_held(tmp_path):
    lease = tmp_path / "selected.lock"
    lease.write_text("same contents")
    guard = tmp_path / "guardian.lock"
    guard.touch()
    marker = tmp_path / "guardian.json"
    marker.write_text("fixture")
    contract = {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "input_sha256": {},
        "locks": [
            {"path": str(lease), "device": lease.stat().st_dev, "inode": lease.stat().st_ino}
        ],
        "protected_gpu0": {
            "quarantine": str(marker),
            "quarantine_sha256": campaign.digest(marker),
            "lease": str(guard),
            "lease_inode": guard.stat().st_ino,
            "guardian_pid": os.getpid(),
            "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
        },
    }
    with lease.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.check_inputs(contract, [held.fileno()])
        lease.rename(tmp_path / "original-held.lock")
        lease.write_text("same contents")
        with lease.open("r+") as replacement:
            fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="resource path identity"):
            campaign.check_inputs(contract, [held.fileno()])


@pytest.mark.parametrize("field", ["controller_pid", "controller_start_ticks", "ready_sha256"])
def test_grant_cannot_be_replayed_into_another_ready_incarnation(field):
    owner = {"pid": 1, "start_ticks": 2}
    ready = {"controller": owner, "contract_sha256": "a" * 64}
    grant = {
        "approved": True,
        "phase": "precheck",
        "contract_sha256": "a" * 64,
        "ready_sha256": "b" * 64,
        "controller_pid": 1,
        "controller_start_ticks": 2,
        "review_sha256": "c" * 64,
    }
    campaign.validate_grant(grant, ready, "b" * 64, "precheck", owner)
    grant[field] = "wrong" if field == "ready_sha256" else 3
    with pytest.raises(RuntimeError, match="incarnation"):
        campaign.validate_grant(grant, ready, "b" * 64, "precheck", owner)


def test_loader_and_bytecode_influences_are_removed_not_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_AUDIT", "/never-loaded-fixture")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path))
    cell = {
        "environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-fixture",
            "ORBITKV_TEST_STRICT_SHUTDOWN": "1",
            "MC_FORCE_TCP": "1",
            "LD_LIBRARY_PATH": "/frozen-fixture",
        }
    }
    env = campaign.cell_environment(cell, "GPU-fixture")
    assert "LD_AUDIT" not in env and "PYTHONPYCACHEPREFIX" not in env
    cell["environment"]["LD_AUDIT"] = "/never-loaded-fixture"
    with pytest.raises(RuntimeError):
        campaign.cell_environment(cell, "GPU-fixture")


def test_inventory_uses_one_remaining_deadline(monkeypatch):
    limits = []

    def run(argv, **kwargs):
        limits.append(kwargs["timeout"])
        if "--query-gpu=uuid,memory.used" in argv:
            time.sleep(0.04)
            return SimpleNamespace(stdout="GPU-fixture, 0")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.inventory("GPU-fixture", time.monotonic() + 0.1)
    assert 0 < limits[1] < limits[0] <= 0.1


def test_expired_or_oversized_oracle_cannot_get_success(tmp_path, monkeypatch):
    cell, owner, observed = valid_evidence(tmp_path)
    with pytest.raises(TimeoutError):
        campaign.validate_cell(cell, tmp_path, owner, observed, time.monotonic() - 1)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (campaign.MAX_JSON_BYTES + 1))
    with pytest.raises(RuntimeError, match="oversized"):
        campaign.read_json(oversized, time.monotonic() + 1)
    parse = campaign.json.loads

    def slow(*args, **kwargs):
        time.sleep(0.03)
        return parse(*args, **kwargs)

    monkeypatch.setattr(campaign.json, "loads", slow)
    with pytest.raises(TimeoutError):
        campaign.read_json(tmp_path / "manager.log.cleanup.json", time.monotonic() + 0.01)


@pytest.mark.parametrize("fault", ["final-integrity", "final-result"])
def test_final_failure_enters_retention_while_real_lock_is_still_held(tmp_path, monkeypatch, fault):
    lease = tmp_path / "selected.lock"
    lease.touch()
    guard = tmp_path / "guardian.lock"
    guard.touch()
    marker = tmp_path / "guardian.json"
    marker.write_text("CPU fixture")
    owner = campaign.identity(os.getpid())
    contract = {
        "schema": "orbitkv.explicit-registration-campaign.v1",
        "phase": "precheck",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "input_sha256": {},
        "protected_gpu0": {
            "quarantine": str(marker),
            "quarantine_sha256": campaign.digest(marker),
            "lease": str(guard),
            "lease_inode": guard.stat().st_ino,
            "guardian_pid": owner["pid"],
            "guardian_start_ticks": owner["start_ticks"],
        },
        "locks": [
            {"path": str(lease), "device": lease.stat().st_dev, "inode": lease.stat().st_ino}
        ],
        "quarantine": str(tmp_path / "absent"),
        "gpu_uuid": "GPU-fixture",
        "cpus": sorted(os.sched_getaffinity(0))[:1],
        "review_wait_seconds": 10,
        "campaign_seconds": 10,
        "cleanup_seconds": 10,
        "cells": [
            {
                "kind": "native",
                "passed": 2,
                "name": "cell",
                "argv": [],
                "cwd": str(tmp_path),
                "seconds": 5,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "GPU-fixture",
                    "ORBITKV_TEST_STRICT_SHUTDOWN": "1",
                    "MC_FORCE_TCP": "1",
                },
            }
        ],
    }
    path = tmp_path / "CONTRACT.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(sys, "argv", [campaign.__file__, str(path)])
    monkeypatch.setattr(campaign.ctypes, "CDLL", lambda *_: SimpleNamespace(prctl=lambda *_: 0))
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    monkeypatch.setattr(campaign.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(campaign, "inventory", lambda *_: {"memory_mib": 0, "compute": []})
    monkeypatch.setattr(campaign, "descendants", lambda *_: [])
    # Other tests validate terminal evidence; this control isolates ownership
    # after those checks, using no Manager, CUDA or native subprocess.
    monkeypatch.setattr(campaign, "validate_cell", lambda *_: None)

    class CpuProcess:
        pid = os.getpid()

        def poll(self):
            return 0

        def wait(self, **_kwargs):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: CpuProcess())
    write = campaign.write

    def publish(target, value, deadline=None):
        if target.name == "RESULT.json" and fault == "final-result":
            raise OSError("controlled final RESULT persistence failure")
        write(target, value, deadline)
        if target.name == "READY.json":
            write(
                tmp_path / "LAUNCH-APPROVED.json",
                {
                    "approved": True,
                    "phase": "precheck",
                    "contract_sha256": value["contract_sha256"],
                    "ready_sha256": campaign.digest(target),
                    "controller_pid": owner["pid"],
                    "controller_start_ticks": owner["start_ticks"],
                    "review_sha256": "c" * 64,
                },
            )

    monkeypatch.setattr(campaign, "write", publish)
    check = campaign.check_inputs
    calls = 0

    def verify(*args):
        nonlocal calls
        calls += 1
        if calls == 6 and fault == "final-integrity":
            raise RuntimeError("controlled final input-integrity failure")
        check(*args)

    monkeypatch.setattr(campaign, "check_inputs", verify)

    class RetentionEntered(BaseException):
        pass

    def retain(_root, reason, locks):
        assert "controlled final" in reason
        assert os.fstat(locks[0]).st_ino == lease.stat().st_ino
        with lease.open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # End only this CPU test spy. Production quarantine does not return.
        raise RetentionEntered

    monkeypatch.setattr(campaign, "quarantine", retain)
    with pytest.raises(RetentionEntered):
        campaign.main()
    assert not (tmp_path / "RESULT.json").exists()
