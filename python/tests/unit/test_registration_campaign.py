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


def cpu_ready_inputs(tmp_path):
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
        "protected_resources": [
            {
                "quarantine": str(protected),
                "quarantine_sha256": campaign.digest(protected),
                "lease": str(protected_lease),
                "lease_inode": protected_lease.stat().st_ino,
                "guardian_pid": os.getpid(),
                "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
            }
        ],
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
    return path, env, lease


@pytest.mark.parametrize("supervised", [False, True])
def test_cpu_ready_owner_holds_then_normally_releases_real_resource_locks(tmp_path, supervised):
    path, env, lease = cpu_ready_inputs(tmp_path)
    argv = [sys.executable, "-B", campaign.__file__]
    if supervised:
        argv.append("--supervise")
    argv.append(str(path))
    process = subprocess.Popen(
        argv,
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
    if supervised:
        record = json.loads((tmp_path / "CONTROLLER-WAIT.json").read_text())
        assert record["supervisor"]["pid"] == process.pid
        assert record["controller"]["parent"] == process.pid
        assert record["exit_code"] == 0 and record["actual_wait"] is True
        assert record["forced_cleanup"] is False and record["wait_error"] is None
        assert not Path(f"/proc/{record['controller']['pid']}").exists()


@pytest.mark.parametrize(
    "fault", [None, "owner", "file", "lock-fd", "unlocked", "file-reopen", "lock-reuse"]
)
def test_protected_snapshot_binds_real_lock_owner_and_immutable_bytes(tmp_path, monkeypatch, fault):
    protected = tmp_path / "protected"
    protected.write_bytes(b"original evidence")
    lease = tmp_path / "protected.lock"
    with lease.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = campaign.identity(os.getpid())
        metadata = protected.stat()
        snapshot = {
            "owner": owner,
            "files": [
                {
                    "path": str(protected),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "bytes": metadata.st_size,
                    "sha256": campaign.digest(protected),
                }
            ],
            "locks": [
                {
                    "path": str(lease),
                    "fd": lock.fileno(),
                    "device": lease.stat().st_dev,
                    "inode": lease.stat().st_ino,
                }
            ],
        }
        contract = {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "input_sha256": {},
            "protected_resources": [],
            "locks": [],
            "protected_snapshots": [snapshot],
        }
        if fault == "owner":
            owner["start_ticks"] += 1
        elif fault == "file":
            protected.write_bytes(b"modified evidence")
        elif fault == "lock-fd":
            snapshot["locks"][0]["inode"] += 1
        elif fault == "unlocked":
            fcntl.flock(lock, fcntl.LOCK_UN)
        elif fault in ("file-reopen", "lock-reuse"):
            opened = os.open
            trigger = (
                str(protected)
                if fault == "file-reopen"
                else f"/proc/{os.getpid()}/fdinfo/{lock.fileno()}"
            )
            mutated = False

            def race(path, flags, *args, **kwargs):
                nonlocal mutated
                if str(path) == trigger and not mutated:
                    mutated = True
                    if fault == "file-reopen":
                        replacement = tmp_path / "replacement"
                        replacement.write_bytes(protected.read_bytes())
                        os.replace(replacement, protected)
                    else:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        lease.unlink()
                        fd = opened(lease, os.O_RDWR | os.O_CREAT, 0o600)
                        os.dup2(fd, lock.fileno())
                        os.close(fd)
                        fcntl.flock(lock, fcntl.LOCK_EX)
                return opened(path, flags, *args, **kwargs)

            monkeypatch.setattr(os, "open", race)
        if fault is None:
            campaign.check_inputs(contract, deadline=time.monotonic() + 1)
        else:
            with pytest.raises(RuntimeError):
                campaign.check_inputs(contract, deadline=time.monotonic() + 1)


def test_supervisor_records_real_nonzero_controller_exit_without_gpu_launch(tmp_path):
    path, env, lease = cpu_ready_inputs(tmp_path)
    contract = json.loads(path.read_text())
    contract["boot_id"] = "invalid-CPU-fixture"
    path.write_text(json.dumps(contract))
    process = subprocess.Popen(
        [sys.executable, "-B", campaign.__file__, "--supervise", str(path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.communicate(timeout=5)
    assert process.returncode != 0
    record = json.loads((tmp_path / "CONTROLLER-WAIT.json").read_text())
    assert record["controller"]["parent"] == process.pid
    assert record["exit_code"] == 1 and record["actual_wait"] is True
    assert record["wait_error"] is None and record["forced_cleanup"] is False
    assert not (tmp_path / "READY.json").exists()
    assert not Path(f"/proc/{record['controller']['pid']}").exists()
    with lease.open("a") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


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


@pytest.mark.parametrize(
    "fault", [None, "missing", "wrong-pid", "wrong-status", "duplicate", "nonzero"]
)
def test_actual_parent_wait_trace_binds_native_child_exit(tmp_path, fault):
    cell, owner, observed = valid_evidence(tmp_path)
    child = {"pid": 19, "start_ticks": 20, "parent": 17}
    observed[1]["parent"] = 1
    observed.append(child)
    cell["process_trace_binary"] = "/frozen/strace"
    code = 7 if fault == "nonzero" else 0
    waited = 21 if fault == "wrong-pid" else 19
    line = f"1.0 wait4(-1, [{{WIFEXITED(s) && WEXITSTATUS(s) == {code}}}], 0, NULL) = {waited}\n"
    (tmp_path / "process.trace.1").write_text(
        "0.0 clone(child_stack=NULL, flags=SIGCHLD) = 17\n"
        "3.0 wait4(17, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 17\n"
        "4.0 +++ exited with 0 +++\n"
    )
    (tmp_path / "process.trace.17").write_text(
        "0.1 clone(child_stack=NULL, flags=SIGCHLD) = 19\n"
        + line * (2 if fault == "duplicate" else 1)
        + "2.0 +++ exited with 0 +++\n"
    )
    if fault != "missing":
        status = 8 if fault == "wrong-status" else code
        (tmp_path / "process.trace.19").write_text(f"1.1 +++ exited with {status} +++\n")
    if fault is None:
        campaign.validate_cell(cell, tmp_path, owner, observed)
    else:
        with pytest.raises((RuntimeError, OSError)):
            campaign.validate_cell(cell, tmp_path, owner, observed)


@pytest.mark.parametrize(
    "fault",
    [None, "nonzero", "unlisted-task", "duplicate-birth", "conflicting-exit", "quoted-exit"],
)
def test_trace_census_checks_children_missed_by_proc_sampling(tmp_path, fault):
    cell, owner, observed = valid_evidence(tmp_path)
    observed[1]["parent"] = owner["pid"]
    cell["process_trace_binary"] = "/frozen/strace"
    code = 7 if fault == "nonzero" else 0
    birth = "1.0 clone(child_stack=NULL, flags=SIGCHLD) = 19\n"
    (tmp_path / "process.trace.1").write_text(
        "0.0 clone(child_stack=NULL, flags=SIGCHLD) = 17\n"
        "4.0 wait4(17, [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 17\n"
        "5.0 +++ exited with 0 +++\n"
    )
    (tmp_path / "process.trace.17").write_text(
        birth * (2 if fault == "duplicate-birth" else 1)
        + f"2.0 wait4(19, [{{WIFEXITED(s) && WEXITSTATUS(s) == {code}}}], 0, NULL) = 19\n"
        + "3.0 +++ exited with 0 +++\n"
    )
    terminal = f"2.1 +++ exited with {code} +++\n"
    if fault == "conflicting-exit":
        terminal += "2.2 +++ exited with 7 +++\n"
    elif fault == "quoted-exit":
        terminal = '2.1 execve("/bin/echo", ["+++ exited with 0 +++"], []) = 0\n'
    (tmp_path / "process.trace.19").write_text(terminal)
    if fault == "unlisted-task":
        (tmp_path / "process.trace.21").write_text("2.1 +++ exited with 0 +++\n")
    if fault is None:
        campaign.validate_cell(cell, tmp_path, owner, observed)
    else:
        with pytest.raises(RuntimeError):
            campaign.validate_cell(cell, tmp_path, owner, observed)


def test_trace_wait_in_parent_thread_accounts_child_and_resumed_syscall(tmp_path):
    (tmp_path / "process.trace.1").write_text(
        "0.0 clone(child_stack=NULL, flags=SIGCHLD) = 19\n"
        "0.1 clone3({flags=CLONE_VM|CLONE_THREAD|CLONE_PARENT_SETTID}, 88) = 21\n"
        "3.0 +++ exited with 0 +++\n"
    )
    (tmp_path / "process.trace.19").write_text("1.1 +++ exited with 0 +++\n")
    (tmp_path / "process.trace.21").write_text(
        "1.0 wait4(19, <unfinished ...>\n"
        "1.2 <... wait4 resumed>[{WIFEXITED(s) && WEXITSTATUS(s) == 0}], 0, NULL) = 19\n"
        "2.0 +++ exited with 0 +++\n"
    )
    rows = campaign.process_trace_inventory(tmp_path, 1)
    assert rows[19]["waited_by"] == 21 and rows[19]["parent"] == 1
    assert rows[21]["group"] == 1 and rows[21]["thread"] is True


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
        "protected_resources": [
            {
                "quarantine": str(marker),
                "quarantine_sha256": campaign.digest(marker),
                "lease": str(guard),
                "lease_inode": guard.stat().st_ino,
                "guardian_pid": os.getpid(),
                "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
            }
        ],
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


@pytest.mark.parametrize(
    "fault", [None, "payload", "extra-file", "symlink-target", "manifest", "duplicate", "traversal"]
)
def test_runtime_manifest_checks_bytes_targets_and_complete_census(tmp_path, fault):
    root = tmp_path / "site-packages"
    root.mkdir()
    payload = root / "native.so"
    payload.write_bytes(b"frozen CPU payload")
    alias = root / "native-link.so"
    alias.symlink_to(payload)
    rows = [
        {
            "path": path.name,
            "realpath": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": campaign.digest(path),
        }
        for path in (payload, alias)
    ]
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "traversal":
        rows[0] = dict(rows[0], path="../site-packages/native.so")
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"root": str(root), "files": rows}))
    declaration = {"path": str(manifest), "sha256": campaign.digest(manifest)}
    contract = runtime_input_contract(tmp_path, [declaration])
    if fault == "payload":
        payload.write_bytes(b"changed CPU bytes")
    elif fault == "extra-file":
        (root / "sitecustomize.py").write_text("unexpected import influence")
    elif fault == "symlink-target":
        replacement = tmp_path / "same-bytes.so"
        replacement.write_bytes(payload.read_bytes())
        alias.unlink()
        alias.symlink_to(replacement)
    elif fault == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    if fault is None:
        campaign.check_inputs(contract, deadline=time.monotonic() + 1)
    else:
        with pytest.raises(RuntimeError):
            campaign.check_inputs(contract, deadline=time.monotonic() + 1)


def runtime_input_contract(directory, manifests):
    guard = directory / "protected.lock"
    guard.touch()
    marker = directory / "protected.json"
    marker.write_text("CPU fixture")
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "input_sha256": {},
        "runtime_manifests": manifests,
        "locks": [],
        "protected_resources": [
            {
                "quarantine": str(marker),
                "quarantine_sha256": campaign.digest(marker),
                "lease": str(guard),
                "lease_inode": guard.stat().st_ino,
                "guardian_pid": os.getpid(),
                "guardian_start_ticks": campaign.identity(os.getpid())["start_ticks"],
            }
        ],
    }


def test_cell_checks_consumed_roots_and_final_checks_all_roots(tmp_path):
    manifests = []
    for name in ("vllm", "sglang"):
        root = tmp_path / name
        root.mkdir()
        library = root / "native.so"
        library.write_bytes(b"frozen CPU payload")
        manifest = tmp_path / f"{name}.json"
        manifest.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "files": [
                        {
                            "path": library.name,
                            "realpath": str(library),
                            "bytes": library.stat().st_size,
                            "sha256": campaign.digest(library),
                        }
                    ],
                }
            )
        )
        manifests.append({"path": str(manifest), "sha256": campaign.digest(manifest)})
    contract = runtime_input_contract(tmp_path, manifests)
    selected = [manifests[0]["path"]]
    campaign.check_inputs(contract, runtime_manifest_paths=selected)
    for invalid in ([], ["undeclared"], selected * 2):
        with pytest.raises(RuntimeError, match="runtime roots"):
            campaign.check_inputs(contract, runtime_manifest_paths=invalid)
    (tmp_path / "sglang/native.so").write_bytes(b"changed unused cell input")
    campaign.check_inputs(contract, runtime_manifest_paths=selected)
    with pytest.raises(RuntimeError, match="payload changed"):
        campaign.check_inputs(contract)
    (tmp_path / "vllm/native.so").write_bytes(b"changed used cell input")
    with pytest.raises(RuntimeError, match="payload changed"):
        campaign.check_inputs(contract, runtime_manifest_paths=selected)


def test_expired_startup_verification_cannot_reach_resource_or_gpu_admission(monkeypatch):
    now = [200.0]
    monkeypatch.setattr(campaign.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sys, "argv", [campaign.__file__, "/unused-CPU-contract"])

    def read(_path, _limit, deadline):
        assert deadline == 380.0
        now[0] = 381.0
        return b"{}"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Expired startup reached resource/GPU admission")

    monkeypatch.setattr(campaign, "read_bytes", read)
    monkeypatch.setattr(campaign, "check_inputs", unexpected)
    monkeypatch.setattr(campaign, "inventory", unexpected)
    monkeypatch.setattr(campaign.subprocess, "Popen", unexpected)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        campaign.main()


def test_last_startup_stat_crossing_deadline_cannot_acquire_resources(tmp_path, monkeypatch):
    lease = tmp_path / "selected.lock"
    lease.touch()
    contract = runtime_input_contract(tmp_path, [])
    contract.update(
        schema="orbitkv.explicit-registration-campaign.v1",
        phase="precheck",
        review_wait_seconds=10,
        campaign_seconds=10,
        cleanup_seconds=10,
        cpus=sorted(os.sched_getaffinity(0))[:1],
        cells=[{"kind": "native", "passed": 2}],
        locks=[{"path": str(lease), "device": lease.stat().st_dev, "inode": lease.stat().st_ino}],
    )
    path = tmp_path / "CONTRACT.json"
    path.write_text(json.dumps(contract))
    now = [0.0]
    monkeypatch.setattr(campaign.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sys, "argv", [campaign.__file__, str(path)])
    monkeypatch.setattr(campaign.ctypes, "CDLL", lambda *_: SimpleNamespace(prctl=lambda *_: 0))
    monkeypatch.setattr(campaign.signal, "signal", lambda *_: None)
    lstat = Path.lstat

    def late_stat(selected, *args, **kwargs):
        metadata = lstat(selected, *args, **kwargs)
        if selected == lease:
            now[0] = 181.0
        return metadata

    monkeypatch.setattr(Path, "lstat", late_stat)
    open_file = campaign.os.open

    def forbid_late_admission(selected, *args, **kwargs):
        if Path(selected) == lease:
            raise AssertionError("Expired startup attempted resource open")
        return open_file(selected, *args, **kwargs)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Expired startup reached affinity/GPU admission")

    monkeypatch.setattr(campaign.os, "open", forbid_late_admission)
    monkeypatch.setattr(campaign.os, "sched_setaffinity", unexpected)
    monkeypatch.setattr(campaign, "inventory", unexpected)
    monkeypatch.setattr(campaign.subprocess, "Popen", unexpected)
    with pytest.raises(TimeoutError, match="deadline exhausted"):
        campaign.main()


def test_runtime_manifest_deadline_and_nonregular_inputs_cannot_pass(tmp_path):
    fifo = tmp_path / "not-a-library"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file"):
        campaign.digest(fifo, time.monotonic() + 1)
    with pytest.raises(TimeoutError):
        campaign.check_runtime_manifest({}, time.monotonic() - 1)


@pytest.mark.parametrize("appeared", ["regular", "broken-symlink", "fifo"])
def test_absent_import_search_inputs_cannot_appear_after_freeze(tmp_path, appeared):
    path = tmp_path / "python311.zip"
    contract = runtime_input_contract(tmp_path, [])
    contract["absent_inputs"] = [str(path)]
    campaign.check_inputs(contract)
    if appeared == "regular":
        path.write_bytes(b"unexpected import input")
    elif appeared == "broken-symlink":
        path.symlink_to(tmp_path / "missing")
    else:
        os.mkfifo(path)
    with pytest.raises(RuntimeError, match="absent input appeared"):
        campaign.check_inputs(contract)


def test_fresh_host_protected_resource_list_is_explicit_and_does_not_release_other_owners(tmp_path):
    contract = runtime_input_contract(tmp_path, [])
    contract["protected_resources"] = []
    campaign.check_inputs(contract)
    contract["protected_resources"] = None
    with pytest.raises(RuntimeError, match="protected resource declarations"):
        campaign.check_inputs(contract)
    del contract["protected_resources"]
    with pytest.raises(KeyError):
        campaign.check_inputs(contract)


@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate", "wrong-pid", "wrong-file", "extra-root", "user-site"]
)
def test_manager_importer_origin_is_identity_bound_and_inside_frozen_roots(fault):
    expected = {
        "torch_file": "/frozen/site/torch/__init__.py",
        "gpu_module_file": "/frozen/site/orbitkv/client/gpu.py",
        "prefix": "/usr",
        "base_prefix": "/usr",
        "allowed_sys_path": ["/frozen/site", "/usr/lib/python3.11"],
    }
    runtime = {
        "pid": 17,
        "torch_file": expected["torch_file"],
        "prefix": "/usr",
        "base_prefix": "/usr",
        "sys_path": expected["allowed_sys_path"],
        "no_user_site": 1,
    }
    helper = {"pid": 17, "gpu_module_file": expected["gpu_module_file"]}
    if fault == "wrong-pid":
        runtime["pid"] = 18
    elif fault == "wrong-file":
        helper["gpu_module_file"] = "/unfrozen/orbitkv/client/gpu.py"
    elif fault == "extra-root":
        runtime["sys_path"] = [*runtime["sys_path"], "/unfrozen/site"]
    elif fault == "user-site":
        runtime["no_user_site"] = 0
    text = (
        "INFO CUDA registration Python runtime: " + json.dumps(runtime) + "\n"
        "INFO CUDA registration Python helper: " + json.dumps(helper) + "\n"
    )
    if fault == "missing":
        text = text.splitlines()[0]
    elif fault == "duplicate":
        text += text
    if fault is None:
        campaign.validate_python_origin(text, expected, 17, time.monotonic() + 1)
    else:
        with pytest.raises(RuntimeError):
            campaign.validate_python_origin(text, expected, 17, time.monotonic() + 1)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "wrong-pid",
        "wrong-command",
        "wrong-port",
        "no-sigterm",
        "no-marker",
        "no-runtime",
        "runtime-pid",
        "runtime-pytest",
        "runtime-library",
        "runtime-hash",
        "runtime-inode",
        "runtime-omission",
        "runtime-omit-both",
        "runtime-required-absent",
        "runtime-code-over-bound",
        "runtime-code-empty",
        "runtime-code-reversed",
        "overlay-no-proof",
        "overlay-wrong-code",
        "overlay-opened-inode",
        "overlay-missing-inode",
        "no-credit",
        "credit-owner",
        "credit-replaced",
        "credit-missing-value",
    ],
)
def test_native_cell_requires_command_record_bound_to_observed_manager(tmp_path, fault):
    cell, owner, observed = valid_evidence(tmp_path)
    cell.update(
        kind="native",
        environment={
            "ORBITKV_CACHE_MANAGER_BINARY": "/frozen/manager",
            "CUDA_VISIBLE_DEVICES": "GPU-fixture",
        },
        manager_python={
            "torch_file": "/frozen/site/torch/__init__.py",
            "gpu_module_file": "/frozen/site/orbitkv/client/gpu.py",
            "prefix": "/usr",
            "base_prefix": "/usr",
            "allowed_sys_path": ["/frozen/site"],
        },
        runtime_mapped_files={
            "/frozen/manager": {"realpath": "/frozen/manager", "bytes": 1, "sha256": "a" * 64},
            "/frozen/lib.so": {"realpath": "/frozen/lib.so", "bytes": 1, "sha256": "b" * 64},
        },
    )
    manager_runtime = {
        "owner": {"pid": 17, "start_ticks": 18},
        "maps": "1000-2000 r-xp 00000000 00:01 7 /frozen/manager\n2000-3000 r-xp 00000000 00:01 8 /frozen/lib.so\n",
        "physical_termination_claimed": False,
        "files": [
            dict(path=path, device=1, inode=7 + index, **values)
            for index, (path, values) in enumerate(cell["runtime_mapped_files"].items())
        ],
    }
    if fault == "runtime-pid":
        manager_runtime["owner"]["pid"] = 19
    elif fault == "runtime-pytest":
        manager_runtime["owner"] = owner
    elif fault == "runtime-library":
        manager_runtime["files"][1]["path"] = "/unfrozen/lib.so"
    elif fault == "runtime-hash":
        manager_runtime["files"][1]["sha256"] = "c" * 64
    elif fault == "runtime-inode":
        manager_runtime["files"][1]["inode"] = 10
    elif fault == "runtime-omission":
        manager_runtime["files"].pop()
    elif fault == "runtime-omit-both":
        manager_runtime["files"].pop()
        manager_runtime["maps"] = manager_runtime["maps"].splitlines()[0] + "\n"
    elif fault in ("runtime-code-over-bound", "runtime-code-empty", "runtime-code-reversed"):
        start = 8192
        end = (
            start + campaign.MAX_MAPPED_EXECUTABLE_BYTES + 4096
            if fault == "runtime-code-over-bound"
            else start
            if fault == "runtime-code-empty"
            else start - 4096
        )
        manager_runtime["maps"] = (
            manager_runtime["maps"].splitlines()[0]
            + f"\n{start:x}-{end:x} r-xp 00000000 00:01 8 /frozen/lib.so\n"
        )
    elif fault == "runtime-required-absent":
        cell["runtime_mapped_files"]["/frozen/required.so"] = {
            "realpath": "/frozen/required.so",
            "bytes": 1,
            "sha256": "d" * 64,
        }
    elif fault in (
        "overlay-no-proof",
        "overlay-wrong-code",
        "overlay-opened-inode",
        "overlay-missing-inode",
    ):
        manager_runtime["files"][1].update(device=2, mapped_device=1, mapped_inode=8)
        if fault == "overlay-wrong-code":
            manager_runtime["files"][1]["executable_regions"] = [
                {
                    "start": 8192,
                    "end": 12288,
                    "file_offset": 0,
                    "file_sha256": "a" * 64,
                    "mapped_sha256": "b" * 64,
                }
            ]
        elif fault == "overlay-opened-inode":
            manager_runtime["files"][1]["inode"] = 987
        elif fault == "overlay-missing-inode":
            del manager_runtime["files"][1]["inode"]
    if fault != "no-runtime":
        (tmp_path / "manager.manager-runtime.json").write_text(json.dumps(manager_runtime))
    credit = {
        "owner": owner,
        "evidence": {
            "schema": "orbitkv.cuda-credit-observation.v1",
            "registration_generation": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "export_owner_generation": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "physical_termination_claimed": False,
            "reclamation_authorized": False,
            "credits": [
                {
                    "counter_name": "/counter",
                    "counter_offset": 1,
                    "disposition": "importer_owned",
                    "exclusive_value_at_export": 1,
                    "observation": {
                        "same_object": True,
                        "observed_value": 1,
                        "device": 1,
                        "inode": 2,
                        "path_device": 1,
                        "path_inode": 2,
                    },
                }
            ],
        },
    }
    if fault == "credit-owner":
        credit["owner"] = {"pid": 3}
    elif fault == "credit-replaced":
        credit["evidence"]["credits"][0]["observation"]["same_object"] = False
    elif fault == "credit-missing-value":
        credit["evidence"]["credits"][0]["observation"]["observed_value"] = None
    if fault != "no-credit":
        (tmp_path / "registration.credit-owner.json").write_text(json.dumps(credit))
    runtime = {
        "pid": 17,
        "torch_file": cell["manager_python"]["torch_file"],
        "sys_path": ["/frozen/site"],
        "prefix": "/usr",
        "base_prefix": "/usr",
        "no_user_site": 1,
    }
    helper = {"pid": 17, "gpu_module_file": cell["manager_python"]["gpu_module_file"]}
    (tmp_path / "manager.log").write_text(
        "CUDA registration Python runtime: " + json.dumps(runtime) + "\n"
        "CUDA registration Python helper: "
        + json.dumps(helper)
        + "\n"
        + ("" if fault == "no-marker" else "SIGTERM received, shutting down\n")
    )
    cleanup = tmp_path / "manager.log.cleanup.json"
    row = json.loads(cleanup.read_text())
    row["sigterm_sent"] = fault != "no-sigterm"
    cleanup.write_text(json.dumps(row))
    for index in range(4):
        (tmp_path / f"{index}.native-release.json").write_text(json.dumps(dict(owner, closed=True)))
    start = {
        "pid": 17,
        "start_ticks": 18,
        "log_path": str(tmp_path / "manager.log"),
        "port": 12345,
        "http_port": 12346,
        "bootstrap_socket": "/tmp/CPU-fixture.sock",
        "channel_service": "CPU-fixture",
        "channel_session_epoch": 7,
        "argv": [
            "/frozen/manager",
            "--addr",
            "127.0.0.1:12345",
            "--http-addr",
            "127.0.0.1:12346",
            "--bootstrap-socket",
            "/tmp/CPU-fixture.sock",
            "--channel-service",
            "CPU-fixture",
            "--channel-session-epoch",
            "7",
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-fixture",
            "MC_FORCE_TCP": "1",
            "PYTHONNOUSERSITE": "1",
        },
    }
    if fault == "wrong-pid":
        start["pid"] = 19
    elif fault == "wrong-command":
        start["argv"][0] = "/unfrozen/manager"
    elif fault == "wrong-port":
        start["port"] = 12347
    if fault != "missing":
        (tmp_path / "manager.manager-start.json").write_text(json.dumps(start))
    if fault is None:
        campaign.validate_cell(cell, tmp_path, owner, observed)
    else:
        with pytest.raises(RuntimeError):
            campaign.validate_cell(cell, tmp_path, owner, observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("registration_generation", True),
        ("export_owner_generation", [1]),
        ("registration_generation", "00000000-0000-0000-0000-000000000000"),
        ("exclusive_value_at_export", True),
        ("exclusive_value_at_export", 1.0),
        ("path_device", True),
        ("path_inode", 2.0),
        ("counter_name", "/../not-counter"),
        ("counter_name", "/counter\0other"),
        ("duplicate_physical_slot", True),
    ],
)
def test_native_credit_oracle_refuses_malformed_types_and_aliased_slots(
    tmp_path, monkeypatch, field, value
):
    import copy

    captured = []
    validate = campaign.validate_cell
    monkeypatch.setattr(
        campaign, "validate_cell", lambda *args, **kwargs: captured.append((args, kwargs))
    )
    test_native_cell_requires_command_record_bound_to_observed_manager(tmp_path, None)
    monkeypatch.setattr(campaign, "validate_cell", validate)
    path = tmp_path / "registration.credit-owner.json"
    row = json.loads(path.read_text())
    evidence = row["evidence"]
    credit = evidence["credits"][0]
    if field in ("registration_generation", "export_owner_generation"):
        evidence[field] = value
    elif field in ("path_device", "path_inode"):
        credit["observation"][field] = value
    elif field == "duplicate_physical_slot":
        evidence["credits"].append(copy.deepcopy(credit))
        evidence["credits"][1]["counter_name"] = "/other-name"
    else:
        credit[field] = value
    path.write_text(json.dumps(row))
    args, kwargs = captured[0]
    with pytest.raises((RuntimeError, ValueError)):
        validate(*args, **kwargs)


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
        "protected_resources": [
            {
                "quarantine": str(marker),
                "quarantine_sha256": campaign.digest(marker),
                "lease": str(guard),
                "lease_inode": guard.stat().st_ino,
                "guardian_pid": owner["pid"],
                "guardian_start_ticks": owner["start_ticks"],
            }
        ],
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

    def verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6 and fault == "final-integrity":
            raise RuntimeError("controlled final input-integrity failure")
        check(*args, **kwargs)

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
