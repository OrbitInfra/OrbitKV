"""CPU checks shared by installed tests and distributed benchmark owners."""

import errno
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.support import runtime_checks as checks


@pytest.mark.parametrize("fault", [None, "owner-start", "write"])
def test_service_runtime_records_live_renamed_worker_and_retains_cleanup_owner(
    tmp_path, monkeypatch, fault
):
    child_source = """
import sys,time
from pathlib import Path
root=Path(sys.argv[1])
(root/'child-ready').touch()
deadline=time.monotonic()+45
while not(root/'child-stop').exists() and time.monotonic()<deadline:
    time.sleep(.01)
"""
    parent_source = """
import json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1])
child=subprocess.Popen(['VLLM::EngineCore','-I','-S','-B','-c',sys.argv[2],str(root)],executable=sys.executable)
try:
    deadline=time.monotonic()+45
    while not(root/'child-ready').exists() and time.monotonic()<deadline:
        if child.poll() is not None:raise RuntimeError('worker exited before ready')
        time.sleep(.01)
    (root/'ready.json').write_text(json.dumps({'worker':child.pid}))
    while not(root/'stop').exists() and time.monotonic()<deadline:
        time.sleep(.01)
finally:
    (root/'child-stop').touch()
    code=child.wait(timeout=10)
raise SystemExit(code)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", parent_source, str(tmp_path), child_source],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready.json").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        owner = checks.identity(process.pid)
        worker = json.loads((tmp_path / "ready.json").read_text())["worker"]
        start = owner["start_ticks"] + (fault == "owner-start")
        if fault == "write":

            def fail_write(*_args, **_kwargs):
                raise OSError(errno.ENOSPC, "runtime evidence full")

            monkeypatch.setattr(checks, "write", fail_write)
        if fault is None:
            checks.record_process_group_runtime(
                process, start, tmp_path / "service.log", time.monotonic() + 20
            )
            path = next(tmp_path.glob("service.log.*.process-runtime.json"))
            recorded = json.loads(path.read_text())
            assert recorded["owner"] == owner
            assert {row["owner"]["pid"] for row in recorded["processes"]} == {process.pid, worker}
            observed_worker = next(
                row for row in recorded["processes"] if row["owner"]["pid"] == worker
            )
            assert observed_worker["argv"][0] == "VLLM::EngineCore"
            assert all(row["files"] and row["maps"] for row in recorded["processes"])
        else:
            error = RuntimeError if fault == "owner-start" else OSError
            with pytest.raises(error, match="owner identity changed|runtime evidence full"):
                checks.record_process_group_runtime(
                    process, start, tmp_path / "service.log", time.monotonic() + 20
                )
            assert not list(tmp_path.glob("*.process-runtime.json"))
        assert process.poll() is None
    finally:
        (tmp_path / "stop").touch()
        code = process.wait(timeout=15)
        errors = process.stderr.read().decode()
        process.stderr.close()
        assert code == 0, errors


@pytest.mark.parametrize(
    "fault", [None, "mutable-file", "root-replaced", "nodes-replaced", "symlink", "mode"]
)
def test_private_runtime_domain_preserves_directories_but_allows_owned_files(tmp_path, fault):
    root = tmp_path / "domain"
    root.mkdir(mode=0o700)
    nodes, services = root / "nodes", root / "services"
    nodes.mkdir(mode=0o700)
    services.mkdir(mode=0o700)
    directories = []
    for path in (root, nodes, services):
        metadata = path.stat()
        directories.append(
            {
                "path": str(path),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": metadata.st_mode,
                "uid": metadata.st_uid,
            }
        )
    contract = {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "input_sha256": {},
        "protected_resources": [],
        "locks": [],
        "mutable_runtime_domain": {"root": str(root), "directories": directories},
    }
    checks.check_inputs(contract)
    if fault == "mutable-file":
        (nodes / "owned-node").write_bytes(b"runtime changes this file")
        (services / "owned-service").write_bytes(b"runtime owns this service")
    elif fault == "root-replaced":
        root.rename(tmp_path / "retired-domain")
        root.mkdir(mode=0o700)
        nodes.mkdir(mode=0o700)
        services.mkdir(mode=0o700)
    elif fault in ("nodes-replaced", "symlink"):
        nodes.rename(root / "retired-nodes")
        if fault == "symlink":
            nodes.symlink_to(root / "retired-nodes", target_is_directory=True)
        else:
            nodes.mkdir(mode=0o700)
    elif fault == "mode":
        nodes.chmod(0o750)
    if fault in (None, "mutable-file"):
        checks.check_inputs(contract)
    else:
        with pytest.raises(RuntimeError, match="directory identity changed"):
            checks.check_inputs(contract)


@pytest.mark.parametrize(
    "fault", [None, "owner", "file", "lock-fd", "unlocked", "file-reopen", "lock-reuse"]
)
def test_protected_snapshot_binds_real_lock_owner_and_immutable_bytes(tmp_path, monkeypatch, fault):
    protected = tmp_path / "protected"
    protected.write_bytes(b"original evidence")
    lease = tmp_path / "protected.lock"
    with lease.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = checks.identity(os.getpid())
        metadata = protected.stat()
        snapshot = {
            "owner": owner,
            "files": [
                {
                    "path": str(protected),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "bytes": metadata.st_size,
                    "sha256": checks.digest(protected),
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
            checks.check_inputs(contract, deadline=time.monotonic() + 1)
        else:
            with pytest.raises(RuntimeError):
                checks.check_inputs(contract, deadline=time.monotonic() + 1)


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
                "quarantine_sha256": checks.digest(marker),
                "lease": str(guard),
                "lease_inode": guard.stat().st_ino,
                "guardian_pid": os.getpid(),
                "guardian_start_ticks": checks.identity(os.getpid())["start_ticks"],
            }
        ],
    }
    with lease.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks.check_inputs(contract, [held.fileno()])
        lease.rename(tmp_path / "original-held.lock")
        lease.write_text("same contents")
        with lease.open("r+") as replacement:
            fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="resource path identity"):
            checks.check_inputs(contract, [held.fileno()])


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
            "sha256": checks.digest(path),
        }
        for path in (payload, alias)
    ]
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "traversal":
        rows[0] = dict(rows[0], path="../site-packages/native.so")
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"root": str(root), "files": rows}))
    declaration = {"path": str(manifest), "sha256": checks.digest(manifest)}
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
        checks.check_inputs(contract, deadline=time.monotonic() + 1)
    else:
        with pytest.raises(RuntimeError):
            checks.check_inputs(contract, deadline=time.monotonic() + 1)


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
                "quarantine_sha256": checks.digest(marker),
                "lease": str(guard),
                "lease_inode": guard.stat().st_ino,
                "guardian_pid": os.getpid(),
                "guardian_start_ticks": checks.identity(os.getpid())["start_ticks"],
            }
        ],
    }


def executable_input(path):
    metadata = path.stat()
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "sha256": checks.digest(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "bytes": metadata.st_size,
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
                            "sha256": checks.digest(library),
                        }
                    ],
                }
            )
        )
        manifests.append({"path": str(manifest), "sha256": checks.digest(manifest)})
    contract = runtime_input_contract(tmp_path, manifests)
    selected = [manifests[0]["path"]]
    checks.check_inputs(contract, runtime_manifest_paths=selected)
    for invalid in ([], ["undeclared"], selected * 2):
        with pytest.raises(RuntimeError, match="runtime roots"):
            checks.check_inputs(contract, runtime_manifest_paths=invalid)
    (tmp_path / "sglang/native.so").write_bytes(b"changed unused cell input")
    checks.check_inputs(contract, runtime_manifest_paths=selected)
    with pytest.raises(RuntimeError, match="payload changed"):
        checks.check_inputs(contract)
    (tmp_path / "vllm/native.so").write_bytes(b"changed used cell input")
    with pytest.raises(RuntimeError, match="payload changed"):
        checks.check_inputs(contract, runtime_manifest_paths=selected)


def test_failed_parallel_root_check_joins_other_readers(tmp_path, monkeypatch):
    manifests = []
    for name in ("bad", "held"):
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
                            "sha256": checks.digest(library),
                        }
                    ],
                }
            )
        )
        manifests.append({"path": str(manifest), "sha256": checks.digest(manifest)})
    contract = runtime_input_contract(tmp_path, manifests)
    (tmp_path / "bad/native.so").write_bytes(b"changed CPU bytes")
    entered, failed, release, closed, finished = (threading.Event() for _ in range(5))
    check = checks.check_runtime_manifest
    deadline = time.monotonic() + 5
    errors = []

    def verify(declaration, received_deadline):
        assert received_deadline == deadline
        if declaration == manifests[0]:
            assert entered.wait(2)
            try:
                check(declaration, received_deadline)
            finally:
                failed.set()
        else:
            try:
                with (tmp_path / "held/native.so").open("rb"):
                    entered.set()
                    assert release.wait(2)
                    check(declaration, received_deadline)
            finally:
                closed.set()

    def verify_all():
        try:
            checks.check_inputs(contract, deadline=deadline)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(checks, "check_runtime_manifest", verify)
    caller = threading.Thread(target=verify_all)
    caller.start()
    try:
        assert entered.wait(2)
        assert failed.wait(2)
        assert not finished.wait(0.05), "Integrity failure returned with a live file reader"
    finally:
        release.set()
        caller.join(3)
    assert not caller.is_alive() and finished.is_set() and closed.is_set()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "payload changed" in str(errors[0])


def test_runtime_manifest_deadline_and_nonregular_inputs_cannot_pass(tmp_path):
    fifo = tmp_path / "not-a-library"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="regular file"):
        checks.digest(fifo, time.monotonic() + 1)
    with pytest.raises(TimeoutError):
        checks.check_runtime_manifest({}, time.monotonic() - 1)


@pytest.mark.parametrize("appeared", ["regular", "broken-symlink", "fifo"])
def test_absent_import_search_inputs_cannot_appear_after_freeze(tmp_path, appeared):
    path = tmp_path / "python311.zip"
    contract = runtime_input_contract(tmp_path, [])
    contract["absent_inputs"] = [str(path)]
    checks.check_inputs(contract)
    if appeared == "regular":
        path.write_bytes(b"unexpected import input")
    elif appeared == "broken-symlink":
        path.symlink_to(tmp_path / "missing")
    else:
        os.mkfifo(path)
    with pytest.raises(RuntimeError, match="absent input appeared"):
        checks.check_inputs(contract)


def test_fresh_host_protected_resource_list_is_explicit_and_does_not_release_other_owners(tmp_path):
    contract = runtime_input_contract(tmp_path, [])
    contract["protected_resources"] = []
    checks.check_inputs(contract)
    contract["protected_resources"] = None
    with pytest.raises(RuntimeError, match="protected resource declarations"):
        checks.check_inputs(contract)
    del contract["protected_resources"]
    with pytest.raises(KeyError):
        checks.check_inputs(contract)


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
        checks.validate_python_origin(text, expected, 17, time.monotonic() + 1)
    else:
        with pytest.raises(RuntimeError):
            checks.validate_python_origin(text, expected, 17, time.monotonic() + 1)


@pytest.mark.parametrize("fault", [None, "changed", "missing", "extra"])
def test_repeated_registration_helpers_keep_one_exact_runtime_origin(fault):
    expected = {
        "torch_file": "/site/torch/__init__.py",
        "gpu_module_file": "/site/orbitkv/client/gpu.py",
        "prefix": "/usr",
        "base_prefix": "/usr",
        "allowed_sys_path": ["/site"],
        "helper_records": 6,
    }
    runtime = {
        "pid": 17,
        "torch_file": expected["torch_file"],
        "prefix": "/usr",
        "base_prefix": "/usr",
        "sys_path": ["/site"],
        "no_user_site": 1,
    }
    helper = {"pid": 17, "gpu_module_file": expected["gpu_module_file"]}
    rows = [json.dumps(helper)] * 6
    if fault == "changed":
        rows[-1] = json.dumps(dict(helper, gpu_module_file="/wrong/gpu.py"))
    elif fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append(rows[0])
    text = (
        "CUDA registration Python runtime: "
        + json.dumps(runtime)
        + "\n"
        + "\n".join("CUDA registration Python helper: " + row for row in rows)
    )
    if fault is None:
        checks.validate_python_origin(text, expected, 17)
    else:
        with pytest.raises(RuntimeError):
            checks.validate_python_origin(text, expected, 17)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "lost-execute",
        "never-executable",
        "replacement",
        "alias",
        "payload",
        "relative",
        "duplicate",
    ],
)
def test_executable_inputs_require_unchanged_identity_and_execute_access(tmp_path, fault):
    tool = tmp_path / "compiler"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o400 if fault == "never-executable" else 0o500)
    metadata = tool.stat()
    executable = {
        "path": str(tool),
        "realpath": str(tool.resolve()),
        "sha256": checks.digest(tool),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "bytes": metadata.st_size,
    }
    contract = runtime_input_contract(tmp_path, [])
    contract["executable_inputs"] = [executable]
    if fault == "lost-execute":
        tool.chmod(0o400)
    elif fault == "replacement":
        tool.rename(tmp_path / "retired-compiler")
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o500)
    elif fault == "alias":
        target = tmp_path / "other-compiler"
        tool.rename(target)
        tool.symlink_to(target)
    elif fault == "payload":
        tool.chmod(0o700)
        tool.write_text("#!/bin/sh\nexit 1\n")
        tool.chmod(0o500)
    elif fault == "relative":
        executable["path"] = tool.name
    elif fault == "duplicate":
        contract["executable_inputs"].append(dict(executable))
    if fault is None:
        checks.check_inputs(contract)
    else:
        with pytest.raises(RuntimeError, match="executable"):
            checks.check_inputs(contract)
