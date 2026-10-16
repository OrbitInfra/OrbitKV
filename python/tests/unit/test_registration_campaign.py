"""CPU controls for launch admission and strict lifecycle evidence."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
            "lease": str(lease),
            "lease_inode": lease.stat().st_ino,
            "guardian_pid": os.getpid(),
            "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
        },
        "locks": [{"path": str(lease), "inode": lease.stat().st_ino}],
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
        (tmp_path / "manager.cleanup.json").write_text(json.dumps(row))
    if fault == "traceback":
        (tmp_path / "worker.log").write_text("Traceback (most recent call last): device failure")
    cell = {"passed": 1, "cleanup_receipts": 1, "normal_exit_codes": [0], "kind": "vllm-worker"}
    with pytest.raises(RuntimeError):
        campaign.validate_cell(cell, tmp_path, {"pid": 1, "start_ticks": 2})
