"""Execute one frozen installed registration gate with durable resource ownership."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

MAX_JSON_BYTES = 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_FILES = 4096
MAX_OBSERVED_PROCESSES = 1024
MAX_RUNTIME_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_FILES = 200000
MAX_MAPPED_EXECUTABLE_BYTES = 4 * 1024 * 1024 * 1024


def remaining(deadline: float | None) -> float:
    if deadline is None:
        return 10.0
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Shared evidence/cleanup deadline exhausted")
    return value


def read_bytes(path: Path, limit: int, deadline: float | None = None) -> bytes:
    remaining(deadline)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise RuntimeError(f"Invalid or oversized evidence: {path}")
        chunks = []
        size = 0
        while True:
            remaining(deadline)
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise RuntimeError(f"Evidence grew beyond its bound: {path}")
        remaining(deadline)
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_json(path: Path, deadline: float | None = None) -> dict:
    value = json.loads(read_bytes(path, MAX_JSON_BYTES, deadline))
    remaining(deadline)
    return value


def evidence_paths(directory: Path, pattern: str, deadline: float | None = None) -> list[Path]:
    paths = []
    for index, path in enumerate(directory.rglob(pattern)):
        remaining(deadline)
        if index >= MAX_EVIDENCE_FILES or path.is_symlink() or not path.is_file():
            raise RuntimeError("Invalid or excessive evidence paths")
        if not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError("Evidence leaves the owned output directory")
        paths.append(path)
    remaining(deadline)
    return sorted(paths)


def digest(path: Path, deadline: float | None = None) -> str:
    value = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError(f"Frozen input is not a regular file: {path}")
        while True:
            remaining(deadline)
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            value.update(chunk)
    finally:
        os.close(fd)
    remaining(deadline)
    return value.hexdigest()


def check_runtime_manifest(declaration: dict, deadline: float | None = None) -> None:
    remaining(deadline)
    raw = read_bytes(Path(declaration["path"]), MAX_RUNTIME_MANIFEST_BYTES, deadline)
    if hashlib.sha256(raw).hexdigest() != declaration["sha256"]:
        raise RuntimeError("Frozen runtime manifest changed")
    manifest = json.loads(raw)
    remaining(deadline)
    root = Path(manifest["root"])
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise RuntimeError("Runtime manifest root is not a canonical directory")
    files = manifest["files"]
    if not isinstance(files, list) or not files or len(files) > MAX_RUNTIME_FILES:
        raise RuntimeError("Runtime manifest file count is invalid")
    declared = set()
    for item in files:
        remaining(deadline)
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != item["path"]
            or item["path"] in declared
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
        ):
            raise RuntimeError("Invalid or duplicate runtime file declaration")
        declared.add(item["path"])
        path = root / relative
        resolved = path.resolve(strict=True)
        if str(resolved) != item["realpath"]:
            raise RuntimeError(f"Runtime file target changed: {path}")
        if (
            not resolved.is_file()
            or resolved.stat().st_size != item["bytes"]
            or digest(resolved, deadline) != item["sha256"]
            or path.resolve(strict=True) != resolved
        ):
            raise RuntimeError(f"Frozen runtime payload changed: {path}")
    actual = set()
    for index, path in enumerate(root.rglob("*")):
        remaining(deadline)
        if index >= MAX_RUNTIME_FILES * 4:
            raise RuntimeError("Runtime directory census bound exceeded")
        if path.is_symlink() and path.is_dir():
            raise RuntimeError("Runtime directory symlinks require a separate frozen root")
        if path.is_file() or path.is_symlink():
            actual.add(path.relative_to(root).as_posix())
            if len(actual) > MAX_RUNTIME_FILES:
                raise RuntimeError("Runtime file census bound exceeded")
        elif not path.is_dir():
            raise RuntimeError("Runtime census contains a non-regular input")
    if actual != declared:
        raise RuntimeError("Runtime file census differs from the frozen manifest")
    remaining(deadline)


def identity(pid: int, deadline: float | None = None) -> dict:
    remaining(deadline)
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def write(path: Path, value: dict, deadline: float | None = None) -> None:
    remaining(deadline)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        remaining(deadline)
        os.fsync(stream.fileno())
        remaining(deadline)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        remaining(deadline)
        os.fsync(fd)
        remaining(deadline)
    finally:
        os.close(fd)


def descendants(deadline: float | None = None) -> list[dict]:
    table = {}
    for index, path in enumerate(Path("/proc").glob("[0-9]*/stat")):
        remaining(deadline)
        if index >= 65536:
            raise RuntimeError("Process inventory bound exceeded")
        try:
            item = identity(int(path.parent.name), deadline)
        except (FileNotFoundError, ProcessLookupError):
            continue
        table[item["pid"]] = item
    if os.getpid() not in table:
        raise RuntimeError("Controller is outside its /proc namespace")
    owned = {os.getpid()}
    while True:
        larger = owned | {pid for pid, item in table.items() if item["parent"] in owned}
        if larger == owned:
            return [table[pid] for pid in sorted(owned - {os.getpid()})]
        owned = larger


def inventory(gpu_uuid: str, deadline: float | None = None) -> dict:
    gpu = (
        subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=min(10, remaining(deadline)),
            check=True,
        )
        .stdout.strip()
        .split(",")
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=min(10, remaining(deadline)),
        check=True,
    ).stdout
    if len(gpu) != 2 or gpu[0].strip() != gpu_uuid:
        raise RuntimeError("GPU inventory does not match the frozen UUID")
    remaining(deadline)
    return {
        "uuid": gpu_uuid,
        "memory_mib": int(gpu[1]),
        "compute": [row for row in apps.splitlines() if gpu_uuid in row],
    }


def check_runtime_domain(domain: dict, deadline: float | None = None) -> None:
    root = Path(domain["root"])
    directories = domain["directories"]
    expected_paths = {str(root), str(root / "nodes"), str(root / "services")}
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or not isinstance(directories, list)
        or len(directories) != 3
        or {row["path"] for row in directories} != expected_paths
    ):
        raise RuntimeError("Invalid private runtime directory declarations")
    for row in directories:
        remaining(deadline)
        path = Path(row["path"])
        metadata = path.lstat()
        values = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "uid": metadata.st_uid,
        }
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or path.resolve(strict=True) != path
            or any(
                type(row[field]) is not int or row[field] != value
                for field, value in values.items()
            )
        ):
            raise RuntimeError("Private runtime directory identity changed")
    remaining(deadline)


def check_inputs(
    contract: dict,
    locks: list[int] | None = None,
    deadline: float | None = None,
    runtime_manifest_paths: list[str] | None = None,
) -> None:
    remaining(deadline)
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != contract["boot_id"]:
        raise RuntimeError("Host boot identity changed")
    for path, expected in contract["input_sha256"].items():
        if digest(Path(path), deadline) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    executables = contract.get("executable_inputs", [])
    if not isinstance(executables, list) or len(executables) > 64:
        raise RuntimeError("Invalid executable input declarations")
    seen_executables = set()
    for executable in executables:
        remaining(deadline)
        if (
            not isinstance(executable, dict)
            or not isinstance(executable.get("path"), str)
            or not Path(executable["path"]).is_absolute()
            or executable["path"] in seen_executables
            or not isinstance(executable.get("realpath"), str)
            or not Path(executable["realpath"]).is_absolute()
            or not isinstance(executable.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", executable["sha256"])
            or any(
                type(executable.get(field)) is not int or executable[field] < 0
                for field in ("device", "inode", "mode", "uid", "gid", "bytes")
            )
        ):
            raise RuntimeError("Invalid executable input identity")
        seen_executables.add(executable["path"])
        path = Path(executable["path"])
        metadata = path.stat()
        observed = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "bytes": metadata.st_size,
        }
        if (
            not stat.S_ISREG(metadata.st_mode)
            or str(path.resolve(strict=True)) != executable["realpath"]
            or any(executable[field] != value for field, value in observed.items())
            or not os.access(path, os.X_OK)
            or digest(path, deadline) != executable["sha256"]
        ):
            raise RuntimeError(f"Frozen executable changed or cannot execute: {path}")
    if contract.get("qualification") == "pending-save-bytes-v2":
        for cell in contract["cells"]:
            tracer = cell.get("process_trace_binary")
            if (
                not isinstance(tracer, str)
                or not Path(tracer).is_absolute()
                or tracer not in seen_executables
                or tracer not in contract["input_sha256"]
            ):
                raise RuntimeError("Installed Save gate requires a frozen process tracer")
            declared = next(item for item in executables if item["path"] == tracer)
            if contract["input_sha256"][tracer] != declared["sha256"]:
                raise RuntimeError("Process tracer digests disagree")
    absent = contract.get("absent_inputs", [])
    if not isinstance(absent, list) or len(absent) > MAX_EVIDENCE_FILES:
        raise RuntimeError("Invalid absent input declarations")
    for name in absent:
        remaining(deadline)
        path = Path(name)
        if not path.is_absolute():
            raise RuntimeError("Absent input must have an absolute path")
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError(f"Frozen absent input appeared: {path}")
    manifests = contract.get("runtime_manifests", [])
    if runtime_manifest_paths is not None:
        available = {manifest["path"]: manifest for manifest in manifests}
        if (
            not isinstance(runtime_manifest_paths, list)
            or not runtime_manifest_paths
            or any(not isinstance(path, str) for path in runtime_manifest_paths)
            or len(set(runtime_manifest_paths)) != len(runtime_manifest_paths)
            or len(available) != len(manifests)
            or not set(runtime_manifest_paths) <= set(available)
        ):
            raise RuntimeError("Cell runtime roots differ from the frozen declarations")
        manifests = [available[path] for path in runtime_manifest_paths]
    if len(manifests) < 2:
        for manifest in manifests:
            check_runtime_manifest(manifest, deadline)
    else:
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            readers = [
                executor.submit(check_runtime_manifest, manifest, deadline)
                for manifest in manifests
            ]
            for reader in readers:
                reader.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if "mutable_runtime_domain" in contract:
        check_runtime_domain(contract["mutable_runtime_domain"], deadline)
    protected_resources = contract["protected_resources"]
    if not isinstance(protected_resources, list) or len(protected_resources) > 64:
        raise RuntimeError("Invalid protected resource declarations")
    for protected in protected_resources:
        if (
            digest(Path(protected["quarantine"]), deadline) != protected["quarantine_sha256"]
            or Path(protected["lease"]).stat().st_ino != protected["lease_inode"]
            or identity(protected["guardian_pid"], deadline)["start_ticks"]
            != protected["guardian_start_ticks"]
        ):
            raise RuntimeError("Protected resource identity changed")
    snapshots = contract.get("protected_snapshots", [])
    if not isinstance(snapshots, list) or len(snapshots) > 64:
        raise RuntimeError("Invalid protected owner snapshots")
    for snapshot in snapshots:
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("files"), list)
            or not isinstance(snapshot.get("locks"), list)
            or len(snapshot["files"]) > 4096
            or len(snapshot["locks"]) > 64
            or not isinstance(snapshot.get("processes", []), list)
            or len(snapshot.get("processes", [])) > 64
        ):
            raise RuntimeError("Protected snapshot evidence exceeds its bound")
        for process in [snapshot.get("owner"), *snapshot.get("processes", [])]:
            if (
                not isinstance(process, dict)
                or set(process) != {"pid", "parent", "group", "session", "start_ticks"}
                or any(type(value) is not int or value < 0 for value in process.values())
                or process["pid"] <= 0
                or process["start_ticks"] <= 0
            ):
                raise RuntimeError("Protected process identity schema is invalid")
        if identity(snapshot["owner"]["pid"], deadline) != snapshot["owner"]:
            raise RuntimeError("Protected owner incarnation changed")
        for declaration in snapshot["files"]:
            remaining(deadline)
            if (
                not isinstance(declaration, dict)
                or not isinstance(declaration.get("path"), str)
                or not Path(declaration["path"]).is_absolute()
                or any(
                    type(declaration.get(field)) is not int or declaration[field] < 0
                    for field in ("device", "inode", "bytes")
                )
                or not isinstance(declaration.get("sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", declaration["sha256"])
            ):
                raise RuntimeError("Protected file identity schema is invalid")
            path = Path(declaration["path"])
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                metadata = os.fstat(fd)
                expected = (declaration["device"], declaration["inode"], declaration["bytes"])
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino, metadata.st_size) != expected
                ):
                    raise RuntimeError("Protected immutable file changed")
                value = hashlib.sha256()
                size = 0
                while True:
                    remaining(deadline)
                    chunk = os.read(fd, min(1024 * 1024, declaration["bytes"] + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > declaration["bytes"]:
                        raise RuntimeError("Protected file grew during hashing")
                    value.update(chunk)
                final = os.fstat(fd)
                current = path.lstat()
                if (
                    value.hexdigest() != declaration["sha256"]
                    or (current.st_dev, current.st_ino, current.st_size) != expected
                    or not stat.S_ISREG(current.st_mode)
                    or (final.st_mtime_ns, final.st_ctime_ns, final.st_size)
                    != (metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_size)
                ):
                    raise RuntimeError("Protected immutable file changed while inspected")
            finally:
                os.close(fd)
        for lock in snapshot["locks"]:
            if (
                not isinstance(lock, dict)
                or not isinstance(lock.get("path"), str)
                or not Path(lock["path"]).is_absolute()
                or any(
                    type(lock.get(field)) is not int or lock[field] < 0
                    for field in ("device", "inode", "fd")
                )
            ):
                raise RuntimeError("Protected lock identity schema is invalid")
            path = Path(f"/proc/{snapshot['owner']['pid']}/fd/{lock['fd']}")
            metadata = path.stat()
            fields = read_bytes(
                Path(f"/proc/{snapshot['owner']['pid']}/fdinfo/{lock['fd']}"),
                MAX_JSON_BYTES,
                deadline,
            ).decode()
            rows = [row.split() for row in fields.splitlines() if row.startswith("lock:")]
            expected_key = (
                f"{os.major(lock['device']):02x}:{os.minor(lock['device']):02x}:{lock['inode']}"
            )
            final = path.stat()
            pathname = Path(lock["path"]).lstat()
            ino_rows = [
                row.partition(":")[2].strip()
                for row in fields.splitlines()
                if row.startswith("ino:")
            ]
            if (
                os.readlink(path) != lock["path"]
                or (metadata.st_dev, metadata.st_ino) != (lock["device"], lock["inode"])
                or len(rows) != 1
                or len(rows[0]) != 9
                or rows[0][2:6] != ["FLOCK", "ADVISORY", "WRITE", str(snapshot["owner"]["pid"])]
                or rows[0][-2:] != ["0", "EOF"]
                or rows[0][6] != expected_key
                or (final.st_dev, final.st_ino) != (lock["device"], lock["inode"])
                or (pathname.st_dev, pathname.st_ino) != (lock["device"], lock["inode"])
                or not stat.S_ISREG(pathname.st_mode)
                or ino_rows != [str(lock["inode"])]
            ):
                raise RuntimeError("Protected exclusive lock ownership changed")
        for protected_process in snapshot.get("processes", []):
            if identity(protected_process["pid"], deadline) != protected_process:
                raise RuntimeError("Protected supervisory process changed")
        if identity(snapshot["owner"]["pid"], deadline) != snapshot["owner"]:
            raise RuntimeError("Protected owner changed during inspection")
    if locks is not None and len(locks) != len(contract["locks"]):
        raise RuntimeError("Resource descriptor count differs")
    for index, resource in enumerate(contract["locks"]):
        remaining(deadline)
        metadata = Path(resource["path"]).lstat()
        expected = (resource["device"], resource["inode"])
        if not stat.S_ISREG(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != expected:
            raise RuntimeError("Selected resource path identity changed")
        if locks is not None:
            held = os.fstat(locks[index])
            if (held.st_dev, held.st_ino) != expected:
                raise RuntimeError("Held resource descriptor differs from its frozen path")
    remaining(deadline)


def cell_environment(cell: dict, gpu_uuid: str) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
        and not key.startswith(("PYTHON", "LD_", "CUDA_", "ORBITKV_", "MC_", "TENT_"))
    }
    env.update(cell["environment"])
    if (
        env.get("CUDA_VISIBLE_DEVICES") != gpu_uuid
        or env.get("ORBITKV_TEST_STRICT_SHUTDOWN") != "1"
        or env.get("MC_FORCE_TCP") != "1"
        or any(key.startswith("LD_") and key != "LD_LIBRARY_PATH" for key in env)
        or "PYTHONPYCACHEPREFIX" in env
        or "PYTHONOPTIMIZE" in env
    ):
        raise RuntimeError("Frozen installed TCP/strict/GPU environment is invalid")
    return env


def check_dlopen(contract: dict, deadline: float) -> dict:
    subreaper = ctypes.c_int()
    if ctypes.CDLL(None).prctl(37, ctypes.byref(subreaper), 0, 0, 0) != 0 or subreaper.value != 1:
        raise RuntimeError("Dynamic loading preflight requires a child subreaper")
    declaration = contract.get("runtime_dlopen")
    if not isinstance(declaration, dict) or set(declaration) != {"python", "helper", "library"}:
        raise RuntimeError("Installed Save readiness requires an actual frozen dlopen probe")
    for name in ("python", "helper", "library"):
        path = declaration[name]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise RuntimeError("Dynamic loading input is not absolute")
        if path not in contract["input_sha256"]:
            raise RuntimeError("Dynamic loading input has no frozen digest")
        if digest(Path(path), deadline) != contract["input_sha256"][path]:
            raise RuntimeError("Dynamic loading input changed before execution")
    expected = dict(contract["input_sha256"])
    for manifest in contract["runtime_manifests"]:
        raw = read_bytes(Path(manifest["path"]), MAX_RUNTIME_MANIFEST_BYTES, deadline)
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
            raise RuntimeError("Dynamic loading manifest changed")
        values = json.loads(raw)
        for item in values["files"]:
            path, sha = item["realpath"], item["sha256"]
            if path in expected and expected[path] != sha:
                raise RuntimeError("Conflicting dynamic loading input digests")
            expected[path] = sha
    probes = []
    for index, cell in enumerate(contract["cells"]):
        probe_deadline = time.monotonic() + min(30, remaining(deadline) / 2)
        initial_children = {
            (item["pid"], item["start_ticks"]) for item in descendants(probe_deadline)
        }
        output, errors = bytearray(), bytearray()
        finished = False
        probe = subprocess.Popen(
            [
                declaration["python"],
                "-I",
                "-S",
                "-B",
                declaration["helper"],
                declaration["library"],
            ],
            env=cell_environment(cell, contract["gpu_uuid"]),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            owner = identity(probe.pid)
            with selectors.DefaultSelector() as reader:
                for stream, data, limit in (
                    (probe.stdout, output, MAX_JSON_BYTES),
                    (probe.stderr, errors, MAX_LOG_BYTES),
                ):
                    reader.register(stream, selectors.EVENT_READ, (data, limit))
                terminal = None
                while reader.get_map() or terminal is None:
                    for key, _ in reader.select(min(0.05, remaining(probe_deadline))):
                        data, limit = key.data
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            reader.unregister(key.fileobj)
                            continue
                        if len(data) + len(chunk) > limit:
                            raise RuntimeError("Dynamic loading output exceeded its bound")
                        data.extend(chunk)
                    # Keep the direct child unreaped until its process group is
                    # stopped, so its PID cannot become an unrelated group ID.
                    terminal = os.waitid(os.P_PID, probe.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if any(
                    (item["pid"], item["start_ticks"]) not in initial_children
                    and item["pid"] != probe.pid
                    for item in descendants(probe_deadline)
                ):
                    raise RuntimeError("Dynamic loading probe left an unexpected child")
                finished = True
        finally:
            try:
                if not finished:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(probe.pid, signal.SIGKILL)
                code = probe.wait(timeout=remaining(deadline))
                # Adopted children may have left the original session. Only the
                # new descendants of this controller belong to this probe.
                while True:
                    children = [
                        item
                        for item in descendants(deadline)
                        if (item["pid"], item["start_ticks"]) not in initial_children
                    ]
                    if not children:
                        break
                    for child in children:
                        try:
                            fd = os.pidfd_open(child["pid"])
                        except ProcessLookupError:
                            continue
                        try:
                            if identity(child["pid"])["start_ticks"] != child["start_ticks"]:
                                continue
                            signal.pidfd_send_signal(fd, signal.SIGKILL)
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                        finally:
                            os.close(fd)
                        with contextlib.suppress(ChildProcessError):
                            os.waitpid(child["pid"], os.WNOHANG)
                    time.sleep(min(0.01, remaining(deadline)))
            finally:
                probe.stdout.close()
                probe.stderr.close()
        if code:
            detail = errors[:2048].decode(errors="replace")
            raise RuntimeError(f"Actual dynamic loading probe failed: {detail}")
        receipt = json.loads(output)
        if (
            receipt.get("schema") != "orbitkv.runtime-dlopen.v1"
            or type(receipt.get("pid")) is not int
            or receipt["pid"] != owner["pid"]
            or receipt.get("library") != declaration["library"]
            or receipt.get("engine_created") is not False
            or not isinstance(receipt.get("files"), dict)
            or not receipt["files"]
            or len(receipt["files"]) > MAX_EVIDENCE_FILES
            or str(Path(declaration["library"]).resolve()) not in receipt["files"]
        ):
            raise RuntimeError("Actual dynamic loading receipt is invalid")
        for path, sha in receipt["files"].items():
            if expected.get(path) != sha or digest(Path(path), deadline) != sha:
                raise RuntimeError(f"Actual dynamic loading has an unfrozen mapping: {path}")
        probes.append(dict(receipt, cell_index=index, owner=owner, actual_wait_exit=code))
    if not probes:
        raise RuntimeError("Dynamic loading preflight has no cell environments")
    return {"schema": "orbitkv.runtime-dlopen-preflight.v1", "probes": probes}


def validate_grant(grant: dict, ready: dict, ready_sha: str, phase: str, controller: dict) -> None:
    if (
        grant.get("approved") is not True
        or grant.get("phase") != phase
        or grant.get("contract_sha256") != ready["contract_sha256"]
        or grant.get("ready_sha256") != ready_sha
        or grant.get("controller_pid") != controller["pid"]
        or grant.get("controller_start_ticks") != controller["start_ticks"]
        or ready["controller"] != controller
        or not isinstance(grant.get("review_sha256"), str)
        or len(grant["review_sha256"]) != 64
    ):
        raise RuntimeError("Independent grant does not bind this READY/controller incarnation")


def quarantine(root: Path, reason: str, locks: list[int]) -> None:
    with contextlib.suppress(BaseException):
        write(
            root / "QUARANTINE.json",
            {
                "reason": reason,
                "controller": identity(os.getpid()),
                "locks": locks,
                "physical_termination_claimed": False,
            },
        )
    while True:
        with contextlib.suppress(BaseException):
            time.sleep(1)


def validate_python_origin(
    text: str, expected: dict, manager_pid: int, deadline: float | None = None
) -> None:
    values = {}
    for kind in ("runtime", "helper"):
        marker = f"CUDA registration Python {kind}: "
        records = [line.partition(marker)[2] for line in text.splitlines() if marker in line]
        count = expected.get("helper_records", 1) if kind == "helper" else 1
        if (
            type(count) is not int
            or not 1 <= count <= 64
            or len(records) != count
            or any(len(record.encode()) > MAX_JSON_BYTES for record in records)
        ):
            raise RuntimeError("Missing or duplicate Manager Python origin evidence")
        if len(set(records)) != 1:
            raise RuntimeError("Manager Python helper origin changed during registrations")
        remaining(deadline)
        value = json.loads(records[0])
        remaining(deadline)
        if type(value.get("pid")) is not int or value["pid"] != manager_pid:
            raise RuntimeError("Manager Python origin process identity differs")
        values[kind] = value
    runtime = values["runtime"]
    paths = runtime.get("sys_path")
    if (
        runtime.get("torch_file") != expected["torch_file"]
        or runtime.get("prefix") != expected["prefix"]
        or runtime.get("base_prefix") != expected["base_prefix"]
        or type(runtime.get("no_user_site")) is not int
        or runtime["no_user_site"] != 1
        or not isinstance(paths, list)
        or not paths
        or any(
            not isinstance(path, str) or path not in expected["allowed_sys_path"] for path in paths
        )
        or values["helper"].get("gpu_module_file") != expected["gpu_module_file"]
    ):
        raise RuntimeError("Manager Python importer leaves the frozen runtime")
    remaining(deadline)


def capture_process_runtime(pid: int, start_ticks: int, deadline: float) -> dict:
    """Capture one live process's actual native mappings and executable bytes."""
    owner = identity(pid, deadline)
    if owner["start_ticks"] != start_ticks:
        raise RuntimeError("Manager runtime process identity changed")
    raw = read_bytes(Path(f"/proc/{pid}/maps"), 1024 * 1024, deadline).decode()
    executable = Path(f"/proc/{pid}/exe").resolve(strict=True)
    paths = {}
    executable_regions = {}
    executable_bytes = 0
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            name = fields[5]
            if ".so" not in Path(name).name and name != str(executable):
                continue
            if name.endswith(" (deleted)"):
                raise RuntimeError("Manager has a deleted mapped runtime file")
            major, minor = (int(part, 16) for part in fields[3].split(":"))
            mapped_identity = (os.makedev(major, minor), int(fields[4]))
            if name in paths and paths[name] != mapped_identity:
                raise RuntimeError("Manager mapped runtime identities conflict")
            paths[name] = mapped_identity
            if "x" in fields[1]:
                start, end = (int(value, 16) for value in fields[0].split("-"))
                if end <= start:
                    raise RuntimeError("Invalid Manager executable mapping range")
                executable_bytes += end - start
                if executable_bytes > MAX_MAPPED_EXECUTABLE_BYTES:
                    raise RuntimeError("Mapped executable evidence exceeds byte bound")
                executable_regions.setdefault(name, []).append((start, end, int(fields[2], 16)))
    if not paths or len(paths) > 2048:
        raise RuntimeError("Manager mapped file inventory is empty or exceeds its bound")
    files = []
    for name in sorted(paths):
        remaining(deadline)
        path = Path(name)
        resolved = path.resolve(strict=True)
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_ino != paths[name][1]:
                raise RuntimeError("Manager mapped runtime path was replaced")
            code_regions = []
            if metadata.st_dev != paths[name][0]:
                regions = executable_regions.get(name, [])
                if not regions:
                    raise RuntimeError("Mapped runtime identity differs without executable proof")
                memory = os.open(f"/proc/{pid}/mem", os.O_RDONLY | os.O_NONBLOCK)
                try:
                    for start, end, offset in regions:
                        expected_hash = hashlib.sha256()
                        actual_hash = hashlib.sha256()
                        position = 0
                        while position < end - start:
                            remaining(deadline)
                            count = min(1024 * 1024, end - start - position)
                            expected = os.pread(fd, count, offset + position)
                            actual = os.pread(memory, count, start + position)
                            file_count = min(count, max(0, metadata.st_size - offset - position))
                            if len(expected) != file_count:
                                raise RuntimeError("Mapped executable file read was short")
                            # The last executable page can contain zero padding beyond the file.
                            expected += bytes(count - len(expected))
                            if len(actual) != count or actual != expected:
                                raise RuntimeError(
                                    "Actual mapped executable bytes differ from the file"
                                )
                            expected_hash.update(expected)
                            actual_hash.update(actual)
                            position += count
                        code_regions.append(
                            {
                                "start": start,
                                "end": end,
                                "file_offset": offset,
                                "file_sha256": expected_hash.hexdigest(),
                                "mapped_sha256": actual_hash.hexdigest(),
                            }
                        )
                finally:
                    os.close(memory)
            value = hashlib.sha256()
            while True:
                remaining(deadline)
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                value.update(chunk)
            after = os.fstat(fd)
            current = resolved.stat()
            if (
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
                or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
                or path.resolve(strict=True) != resolved
            ):
                raise RuntimeError("Manager mapped runtime changed during hashing")
        finally:
            os.close(fd)
        files.append(
            {
                "path": name,
                "realpath": str(resolved),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "bytes": metadata.st_size,
                "sha256": value.hexdigest(),
                "mapped_device": paths[name][0],
                "mapped_inode": paths[name][1],
                "executable_regions": code_regions,
            }
        )
    if (
        identity(pid, deadline) != owner
        or Path(f"/proc/{pid}/exe").resolve(strict=True) != executable
    ):
        raise RuntimeError("Manager changed during its runtime inventory")
    final_maps = read_bytes(Path(f"/proc/{pid}/maps"), 1024 * 1024, deadline).decode()

    def runtime_lines(value):
        return {
            line
            for line in value.splitlines()
            if len(fields := line.split(maxsplit=5)) == 6
            and fields[5].startswith("/")
            and (".so" in Path(fields[5]).name or fields[5] == str(executable))
        }

    if runtime_lines(raw) != runtime_lines(final_maps):
        raise RuntimeError("Manager mapped runtime changed during capture")
    return {"owner": owner, "maps": raw, "files": files, "physical_termination_claimed": False}


def process_trace_inventory(
    directory: Path, root_pid: int, deadline: float | None = None
) -> dict[int, dict]:
    tasks = {}
    births = {}
    waits = []
    for path in evidence_paths(directory, "process.trace.*", deadline):
        name = path.name.removeprefix("process.trace.")
        if not name.isdecimal() or int(name) in tasks:
            raise RuntimeError("Invalid or duplicate process trace identity")
        pid = int(name)
        events = []
        pending = {}
        terminals = []
        text = read_bytes(path, MAX_LOG_BYTES, deadline).decode(errors="strict")
        for line in text.splitlines():
            remaining(deadline)
            match = re.fullmatch(r"(\d+\.\d+) (.+)", line)
            if not match:
                raise RuntimeError("Malformed process trace event")
            stamp, event = match.groups()
            terminal = re.fullmatch(r"\+\+\+ exited with (\d+) \+\+\+", event)
            if terminal:
                terminals.append(int(terminal[1]))
                continue
            if event.startswith("+++ "):
                raise RuntimeError("Native task has no normal traced terminal state")
            unfinished = re.fullmatch(r"(\w+)\((.*)<unfinished \.\.\.>", event)
            resumed = re.fullmatch(r"<\.\.\. (\w+) resumed>(.*)", event)
            if unfinished:
                if unfinished[1] in pending:
                    raise RuntimeError("Duplicate unfinished process syscall")
                pending[unfinished[1]] = f"{unfinished[1]}({unfinished[2]}"
                continue
            if resumed:
                start = pending.pop(resumed[1], None)
                if start is None:
                    raise RuntimeError("Resumed process syscall has no start")
                event = start + resumed[2]
            events.append((stamp, event))
        if pending or len(terminals) != 1:
            raise RuntimeError("Process trace is incomplete or has conflicting exits")
        tasks[pid] = {"pid": pid, "exit_code": terminals[0]}
        for stamp, event in events:
            created = re.fullmatch(r"(clone3?|fork|vfork)\((.*)\)\s+= (\d+)", event)
            if created and int(created[3]) > 0:
                child = int(created[3])
                if (
                    child in births
                    or child == root_pid
                    or re.search(r"\bCLONE_PARENT\b", created[2])
                ):
                    raise RuntimeError("Unsupported/reused native task identity")
                births[child] = (pid, "CLONE_THREAD" in created[2], stamp)
            waited = re.fullmatch(
                r"wait4\([^,]+, \[\{WIFEXITED\(s\) && WEXITSTATUS\(s\) == (\d+)\}\],.*\)\s+= (\d+)",
                event,
            )
            if waited:
                waits.append((pid, int(waited[2]), int(waited[1])))
    if root_pid not in tasks or set(tasks) != {root_pid, *births}:
        raise RuntimeError("Complete process trace census differs from its creation events")
    tasks[root_pid].update(parent=None, group=root_pid, thread=False)
    unresolved = set(births)
    while unresolved:
        advanced = False
        for child in list(unresolved):
            parent, thread, _stamp = births[child]
            if parent not in tasks or "group" not in tasks[parent]:
                continue
            tasks[child].update(
                parent=tasks[parent]["group"],
                group=tasks[parent]["group"] if thread else child,
                thread=thread,
            )
            unresolved.remove(child)
            advanced = True
        if not advanced:
            raise RuntimeError("Process trace ancestry does not reach its leader")
    for waiter, child, code in waits:
        if (
            child not in tasks
            or tasks[child]["thread"]
            or tasks[child]["parent"] != tasks[waiter]["group"]
        ):
            raise RuntimeError("Traced wait does not belong to the child's actual parent")
        if tasks[child].get("waited_by") is not None or tasks[child]["exit_code"] != code:
            raise RuntimeError("Duplicate traced wait or conflicting child exit")
        tasks[child]["waited_by"] = waiter
    remaining(deadline)
    return tasks


def traced_child_wait(directory: Path, child: dict, deadline: float | None = None) -> int:
    tasks = process_trace_inventory(directory, child["parent"], deadline)
    task = tasks.get(child["pid"])
    if task is None or task.get("waited_by") is None:
        raise RuntimeError("Observed native child has no unique traced parent wait")
    return task["exit_code"]


def validate_cell(
    cell: dict,
    directory: Path,
    pytest_owner: dict,
    observed: list[dict] | None = None,
    deadline: float | None = None,
) -> None:
    if observed is None:
        raise RuntimeError("Observed native process ownership is required")
    known = {(row["pid"], row["start_ticks"]): row for row in observed}
    if len(known) != len(observed):
        raise RuntimeError("Duplicate observed process identities")
    primary = (pytest_owner["pid"], pytest_owner["start_ticks"])
    if primary not in known:
        raise RuntimeError("Waited pytest process was not observed")
    cases = list(
        ET.fromstring(read_bytes(directory / "junit.xml", MAX_JSON_BYTES, deadline)).iter(
            "testcase"
        )
    )
    remaining(deadline)
    if len(cases) != cell["passed"] or any(
        case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
    ):
        raise RuntimeError("Frozen pytest outcome differs")
    receipts = evidence_paths(directory, "*.cleanup.json", deadline)
    if len(receipts) != cell["cleanup_receipts"]:
        raise RuntimeError("Missing native service cleanup receipts")
    accounted = {primary}
    service_logs = set()
    for path in receipts:
        row = read_json(path, deadline)
        if (
            type(row.get("pid")) is not int
            or type(row.get("start_ticks")) is not int
            or row["pid"] <= 0
            or row["start_ticks"] <= 0
            or row.get("forced_kill") is not False
            or row.get("reaped") is not True
            or type(row.get("exit_code")) is not int
            or row.get("exit_code") not in cell["normal_exit_codes"]
            or row.get("remaining_processes") != []
            or row.get("inspection_errors") != []
        ):
            raise RuntimeError(f"Native cleanup is not established: {path}")
        key = (row["pid"], row["start_ticks"])
        if key not in known or key in accounted:
            raise RuntimeError("Native receipt owner is unobserved or duplicated")
        accounted.add(key)
        log = Path(str(path).removesuffix(".cleanup.json"))
        if log.suffix != ".log":
            raise RuntimeError("Native cleanup receipt has no associated log")
        service_logs.add(log)
    for path in evidence_paths(directory, "ADOPTED-*.json", deadline):
        row = read_json(path, deadline)
        key = (row["pid"], row["start_ticks"])
        if (
            key not in known
            or key in accounted
            or row.get("actual_wait") is not True
            or type(row.get("exit_code")) is not int
            or row["exit_code"] not in cell["adopted_exit_codes"]
        ):
            raise RuntimeError("Adopted native exit was invalid, unobserved or duplicated")
        accounted.add(key)
    if cell.get("process_trace_binary"):
        tasks = process_trace_inventory(directory, pytest_owner["pid"], deadline)
        by_pid = {}
        for key in known:
            if key[0] in by_pid:
                raise RuntimeError("One trace cannot account for reused process incarnations")
            by_pid[key[0]] = key
        for pid, task in tasks.items():
            if task["exit_code"] not in cell["adopted_exit_codes"]:
                raise RuntimeError("Traced native child exit is not accepted")
            key = by_pid.get(pid)
            if (
                pid != pytest_owner["pid"]
                and not task["thread"]
                and task.get("waited_by") is None
                and (key is None or key not in accounted)
            ):
                raise RuntimeError("Traced child has neither actual parent nor adopted wait")
            if key is not None and task.get("waited_by") is not None:
                if known[key].get("parent") != task["parent"]:
                    raise RuntimeError("Observed and traced parent identities differ")
                accounted.add(key)
    if accounted != set(known):
        raise RuntimeError("Observed native children have no terminal receipt")
    if not cell.get("log_sets"):
        raise RuntimeError("Expected native log set is required")
    expected_logs = set()
    for declaration in cell["log_sets"]:
        paths = evidence_paths(directory, declaration["pattern"], deadline)
        if len(paths) != declaration["count"]:
            raise RuntimeError("Expected native/worker logs are missing")
        if expected_logs.intersection(paths):
            raise RuntimeError("Declared native logs overlap")
        expected_logs.update(paths)
    logs = set(evidence_paths(directory, "*.log", deadline))
    if logs != expected_logs or not service_logs <= logs or directory / "pytest.log" not in logs:
        raise RuntimeError("Native log census or receipt association differs")
    for path in logs:
        text = read_bytes(path, MAX_LOG_BYTES, deadline).decode(errors="replace")
        if (
            "Traceback (most recent call last)" in text
            or "EngineDeadError" in text
            or "panicked at" in text
        ):
            raise RuntimeError(f"Native/worker failure remains in log: {path}")
    if cell["kind"] in ("native", "pending-save-bytes"):
        expected_origin = cell.get("manager_python")
        if not expected_origin:
            raise RuntimeError("Frozen Manager Python origin contract is required")
        starts = evidence_paths(directory, "*.manager-start.json", deadline)
        if len(starts) != len(receipts):
            raise RuntimeError("Missing actual Manager startup command records")
        runtimes = evidence_paths(directory, "*.manager-runtime.json", deadline)
        if len(runtimes) != len(receipts):
            raise RuntimeError("Missing actual Manager mapped runtime records")
        runtime_owners = set()
        frozen_files = cell.get("runtime_mapped_files", {})
        if not isinstance(frozen_files, dict) or not frozen_files:
            raise RuntimeError("Frozen Manager mapped runtime inputs are required")
        for path in runtimes:
            remaining(deadline)
            row = read_json(path, deadline)
            owner = row.get("owner", {})
            key = (owner.get("pid"), owner.get("start_ticks"))
            if (
                key not in known
                or key in runtime_owners
                or not row.get("maps")
                or not row.get("files")
            ):
                raise RuntimeError("Manager runtime record identity or mappings differ")
            runtime_owners.add(key)
            mapped_paths = {}
            executable_bytes = 0
            for line in row["maps"].splitlines():
                remaining(deadline)
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    name = fields[5]
                    if (
                        ".so" not in Path(name).name
                        and name != cell["environment"]["ORBITKV_CACHE_MANAGER_BINARY"]
                    ):
                        continue
                    major, minor = (int(part, 16) for part in fields[3].split(":"))
                    value = (os.makedev(major, minor), int(fields[4]))
                    if name in mapped_paths and mapped_paths[name] != value:
                        raise RuntimeError("Manager runtime mapping identities conflict")
                    mapped_paths[name] = value
                    if "x" in fields[1]:
                        start, end = (int(part, 16) for part in fields[0].split("-"))
                        if end <= start:
                            raise RuntimeError("Invalid Manager executable mapping range")
                        executable_bytes += end - start
                        if executable_bytes > MAX_MAPPED_EXECUTABLE_BYTES:
                            raise RuntimeError("Mapped executable evidence exceeds byte bound")
            if (
                not mapped_paths
                or set(mapped_paths) != set(frozen_files)
                or set(mapped_paths) != {mapped.get("path") for mapped in row["files"]}
                or len(row["files"]) != len(mapped_paths)
                or cell["environment"]["ORBITKV_CACHE_MANAGER_BINARY"] not in mapped_paths
                or row.get("physical_termination_claimed") is not False
            ):
                raise RuntimeError("Manager actual runtime mapping census differs")
            for mapped in row["files"]:
                expected = frozen_files.get(mapped.get("path"))
                if (
                    expected is None
                    or mapped.get("realpath") != expected.get("realpath")
                    or mapped.get("bytes") != expected.get("bytes")
                    or mapped.get("sha256") != expected.get("sha256")
                    or type(mapped.get("device")) is not int
                    or mapped["device"] < 0
                    or type(mapped.get("inode")) is not int
                    or mapped["inode"] <= 0
                    or mapped["inode"] != mapped_paths[mapped["path"]][1]
                    or (
                        mapped.get("mapped_device", mapped.get("device")),
                        mapped.get("mapped_inode", mapped.get("inode")),
                    )
                    != mapped_paths[mapped["path"]]
                ):
                    raise RuntimeError("Manager actually mapped an unfrozen runtime input")
                if mapped.get("device") != mapped_paths[mapped["path"]][0]:
                    expected_regions = []
                    for line in row["maps"].splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and fields[5] == mapped["path"] and "x" in fields[1]:
                            start, end = (int(value, 16) for value in fields[0].split("-"))
                            expected_regions.append((start, end, int(fields[2], 16)))
                    regions = mapped.get("executable_regions", [])
                    if (
                        not expected_regions
                        or [
                            (region.get("start"), region.get("end"), region.get("file_offset"))
                            for region in regions
                        ]
                        != expected_regions
                        or any(
                            not isinstance(region.get("mapped_sha256"), str)
                            or not re.fullmatch("[0-9a-f]{64}", region["mapped_sha256"])
                            or region["mapped_sha256"] != region.get("file_sha256")
                            for region in regions
                        )
                    ):
                        raise RuntimeError(
                            "Overlay runtime lacks complete actual executable-byte proof"
                        )
        startup = {}
        for path in starts:
            row = read_json(path, deadline)
            key = (row["pid"], row["start_ticks"])
            argv = row.get("argv")
            if (
                key in startup
                or key not in known
                or type(row["pid"]) is not int
                or type(row["start_ticks"]) is not int
                or not isinstance(argv, list)
                or not argv
                or any(not isinstance(argument, str) for argument in argv)
                or argv[0] != cell["environment"]["ORBITKV_CACHE_MANAGER_BINARY"]
                or row.get("environment", {}).get("PYTHONNOUSERSITE") != "1"
                or row.get("environment", {}).get("MC_FORCE_TCP") != "1"
                or row.get("environment", {}).get("CUDA_VISIBLE_DEVICES")
                != cell["environment"]["CUDA_VISIBLE_DEVICES"]
            ):
                raise RuntimeError("Manager startup identity/command/environment differs")
            required = {
                "--addr": f"127.0.0.1:{row['port']}",
                "--http-addr": f"127.0.0.1:{row['http_port']}",
                "--bootstrap-socket": row["bootstrap_socket"],
                "--channel-service": row["channel_service"],
                "--channel-session-epoch": str(row["channel_session_epoch"]),
            }
            for flag, value in required.items():
                if argv.count(flag) != 1:
                    raise RuntimeError("Manager startup lacks its declared listener/namespace")
                index = argv.index(flag)
                if index + 1 >= len(argv) or argv[index + 1] != value:
                    raise RuntimeError("Manager startup listener/namespace differs")
            startup[key] = row
        for path in receipts:
            row = read_json(path, deadline)
            log = Path(str(path).removesuffix(".cleanup.json"))
            start = startup.pop((row["pid"], row["start_ticks"]), None)
            if start is None or start["log_path"] != str(log.resolve()):
                raise RuntimeError("Manager startup command is not bound to its cleanup log")
            text = read_bytes(log, MAX_LOG_BYTES, deadline).decode(errors="replace")
            if row.get("sigterm_sent") is not True or "SIGTERM received, shutting down" not in text:
                raise RuntimeError("Missing actual normal Manager SIGTERM evidence")
            validate_python_origin(
                text,
                expected_origin,
                row["pid"],
                deadline,
            )
        if startup:
            raise RuntimeError("Manager startup has no matching cleanup")
        if runtime_owners != {
            (row["pid"], row["start_ticks"])
            for path in receipts
            for row in [read_json(path, deadline)]
        }:
            raise RuntimeError("Manager runtime records do not match actual cleanup owners")
        pending_bytes = cell["kind"] == "pending-save-bytes"
        credits = evidence_paths(
            directory,
            "round-*.credit-owner.json" if pending_bytes else "registration.credit-owner.json",
            deadline,
        )
        if len(credits) != (6 if pending_bytes else len(receipts)):
            raise RuntimeError("Missing original registration credit snapshots")
        if pending_bytes and {path.name for path in credits} != {
            f"round-{index}-{role}.credit-owner.json"
            for index in range(3)
            for role in ("source", "consumer")
        }:
            raise RuntimeError("Save registration credit round/role identities differ")
        if pending_bytes:
            final_credits = evidence_paths(directory, "round-*.credit-release.json", deadline)
            expected_final = {
                path.with_name(path.name.replace(".credit-owner.json", ".credit-release.json"))
                for path in credits
            }
            if len(final_credits) != 6 or set(final_credits) != expected_final:
                raise RuntimeError("Missing, duplicate or misplaced Save credit release receipts")
        generations = set()
        for path in credits:
            row = read_json(path, deadline)
            if row.get("owner") != pytest_owner:
                raise RuntimeError("Registration credit snapshot owner differs")
            evidence = row.get("evidence", {})
            if (
                not isinstance(evidence, dict)
                or evidence.get("schema") != "orbitkv.cuda-credit-observation.v1"
                or evidence.get("physical_termination_claimed") is not False
                or evidence.get("reclamation_authorized") is not False
                or not evidence.get("registration_generation")
                or not evidence.get("export_owner_generation")
                or not isinstance(evidence.get("credits"), list)
                or not 0 < len(evidence["credits"]) <= 4096
            ):
                raise RuntimeError("Registration handoff credit evidence is missing")
            for field in ("registration_generation", "export_owner_generation"):
                value = evidence[field]
                if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
                    raise RuntimeError("Registration credit generation is invalid")
                if pending_bytes:
                    marker = (field, value)
                    if marker in generations:
                        raise RuntimeError("Replayed Save registration generation")
                    generations.add(marker)
            identities = set()
            physical_slots = set()
            for credit in evidence["credits"]:
                if not isinstance(credit, dict) or not isinstance(credit.get("observation"), dict):
                    raise RuntimeError("Registration counter observation schema is invalid")
                observation = credit.get("observation", {})
                name, offset = credit.get("counter_name"), credit.get("counter_offset")
                normalized = name.removeprefix("/") if isinstance(name, str) else ""
                if (
                    not isinstance(name, str)
                    or not name
                    or not normalized
                    or normalized in (".", "..")
                    or "/" in normalized
                    or "\0" in name
                    or len(name.encode()) > 255
                    or type(offset) is not int
                    or not 0 <= offset < 10_000
                    or (normalized, offset) in identities
                    or credit.get("disposition") != "importer_owned"
                    or type(credit.get("exclusive_value_at_export")) is not int
                    or credit.get("exclusive_value_at_export") != 1
                    or credit.get("observation_error") is not None
                    or observation.get("same_object") is not True
                    or observation.get("path_error") is not None
                    or observation.get("read_error") is not None
                    or type(observation.get("observed_value")) is not int
                    or observation["observed_value"] not in (0, 1)
                    or type(observation.get("inode")) is not int
                    or observation["inode"] <= 0
                    or type(observation.get("device")) is not int
                    or observation["device"] < 0
                    or type(observation.get("path_inode")) is not int
                    or type(observation.get("path_device")) is not int
                    or observation["inode"] != observation.get("path_inode")
                    or observation["device"] != observation.get("path_device")
                ):
                    raise RuntimeError("Original registration counter observation is unproven")
                physical = (observation["device"], observation["inode"], offset)
                if physical in physical_slots:
                    raise RuntimeError("Duplicate physical registration counter slot")
                physical_slots.add(physical)
                identities.add((normalized, offset))
            if pending_bytes:
                final = read_json(
                    path.with_name(path.name.replace(".credit-owner.json", ".credit-release.json")),
                    deadline,
                )
                if (
                    final.get("schema") != "orbitkv.cuda-credit-release.v1"
                    or final.get("owner") != pytest_owner
                    or final.get("instance_id") != row.get("instance_id")
                    or final.get("handoff_sha256") != digest(path, deadline)
                    or final.get("physical_termination_claimed") is not False
                    or type(final.get("observed_mono_ns")) is not int
                    or final["observed_mono_ns"] <= 0
                    or any(
                        final.get(field) != evidence[field]
                        for field in ("registration_generation", "export_owner_generation")
                    )
                    or not isinstance(final.get("counters"), list)
                    or len(final["counters"]) != len(evidence["credits"])
                ):
                    raise RuntimeError("Original Save credit release identity is unproven")
                for before, after in zip(evidence["credits"], final["counters"], strict=True):
                    expected = {
                        "counter_name": before["counter_name"],
                        "counter_offset": before["counter_offset"],
                        "device": before["observation"]["device"],
                        "inode": before["observation"]["inode"],
                        "before": 1,
                        "after": 0,
                    }
                    if after != expected or any(
                        type(after.get(field)) is not int
                        for field in ("counter_offset", "device", "inode", "before", "after")
                    ):
                        raise RuntimeError("Original Save counter did not repay its exact slot")
        releases = evidence_paths(directory, "*.native-release.json", deadline)
        if len(releases) != (3 if pending_bytes else 4):
            raise RuntimeError("Missing complete native client release receipts")
        for path in releases:
            row = read_json(path, deadline)
            valid = (
                (
                    row.get("owner") == pytest_owner
                    and row.get("unregistered") is True
                    and row.get("physical_fault_termination_claimed") is False
                )
                if pending_bytes
                else (
                    row.get("pid") == pytest_owner["pid"]
                    and row.get("start_ticks") == pytest_owner["start_ticks"]
                    and row.get("closed") is True
                )
            )
            if not valid:
                raise RuntimeError("Native client release identity differs")
        if pending_bytes:
            closed = evidence_paths(directory, "clients.native-close.json", deadline)
            if len(closed) != 1:
                raise RuntimeError("Missing complete native Save client close receipt")
            close = read_json(closed[0], deadline)
            if (
                close.get("owner") != pytest_owner
                or close.get("closed") is not True
                or close.get("normal_unregister") is not True
            ):
                raise RuntimeError("Native Save client close was not established")
            summaries = evidence_paths(directory, "PENDING-SAVE-BYTES.json", deadline)
            if len(summaries) != 1:
                raise RuntimeError("Missing installed Save byte summary")
            rows = json.loads(read_bytes(summaries[0], MAX_JSON_BYTES, deadline))
            tier = cell.get("tier")
            if tier not in ("dram", "ssd-uring") or not isinstance(rows, list) or len(rows) != 3:
                raise RuntimeError("Invalid Save byte medium or round count")
            previous_release = 0
            previous_reads = {}
            save_operations = set()
            for index, row in enumerate(rows):
                if (
                    not isinstance(row, dict)
                    or type(row.get("round")) is not int
                    or row["round"] != index
                    or type(row.get("copied_bytes")) is not int
                    or row["copied_bytes"] != 16384
                    or type(row.get("observed_bytes")) is not int
                    or row["observed_bytes"] != 32768
                    or row.get("full_oracle") is not True
                    or row.get("mutation_rejected") is not True
                    or row.get("normal_unregister") is not True
                    or row.get("tier") != tier
                ):
                    raise RuntimeError("Installed Save byte oracle differs")
                source_id, consumer_id = row.get("source_identity"), row.get("consumer_identity")
                if (
                    any(
                        not isinstance(value, str) or not value
                        for value in (source_id, consumer_id)
                    )
                    or source_id == consumer_id
                ):
                    raise RuntimeError("Save source and consumer identities are invalid")
                if close.get("source") != source_id or close.get("consumer") != consumer_id:
                    raise RuntimeError("Save client close identity differs")
                admission = row.get("producer_admission")
                if not isinstance(admission, dict):
                    raise RuntimeError("Missing consumed producer Save admission evidence")
                operations = admission.get("operation_ids")
                if (
                    admission.get("api") != "prepare_save/submit_saves/cancel_save"
                    or not isinstance(operations, list)
                    or len(operations) != 3
                    or any(type(value) is not int or value <= 0 for value in operations)
                    or len(set(operations)) != 3
                    or set(operations) & save_operations
                    or admission.get("canceled_before_submit") is not True
                    or admission.get("canceled_done") is not True
                    or admission.get("canceled_success") is not False
                    or admission.get("physical_fault_termination_claimed") is not False
                    or any(
                        type(admission.get(field)) is not int or admission[field] != expected
                        for field, expected in (
                            ("pending_before_submit", 2),
                            ("submitted_cancel_refused", 2),
                            ("completed_successfully", 2),
                            ("prepare_copy_bytes", 0),
                            ("submitted_copy_bytes", 16384),
                        )
                    )
                    or type(admission.get("started_mono_ns")) is not int
                    or type(admission.get("completed_mono_ns")) is not int
                    or not previous_release
                    < admission["started_mono_ns"]
                    < admission["completed_mono_ns"]
                ):
                    raise RuntimeError("Prepared Save ownership/cancellation evidence differs")
                save_operations.update(operations)
                rejected = row.get("rejected_save")
                if (
                    not isinstance(rejected, dict)
                    or rejected.get("instance_id") != source_id
                    or rejected.get("exception") != "SaveNotSubmittedError"
                    or rejected.get("cause") != "ValueError"
                    or rejected.get("publish_count_claimed") is not False
                    or rejected.get("physical_termination_claimed") is not False
                    or type(rejected.get("started_mono_ns")) is not int
                    or type(rejected.get("ended_mono_ns")) is not int
                    or not (
                        admission["completed_mono_ns"]
                        < rejected["started_mono_ns"]
                        < rejected["ended_mono_ns"]
                    )
                ):
                    raise RuntimeError("Missing exact retired Save rejection observation")
                counters = rejected.get("io_before")
                after_rejection = rejected.get("io_after")
                expected_counters = {
                    "orbitkv_save_bytes_total",
                    "orbitkv_load_bytes_total",
                    "orbitkv_ssd_write_bytes_total",
                    "orbitkv_ssd_prefetch_bytes_total",
                    "orbitkv_ssd_uring_read_operations_total",
                }
                if (
                    not isinstance(counters, dict)
                    or not isinstance(after_rejection, dict)
                    or set(counters) != expected_counters
                    or counters != after_rejection
                    or any(
                        value is not None
                        and (
                            type(value) not in (int, float) or not math.isfinite(value) or value < 0
                        )
                        for snapshot in (counters, after_rejection)
                        for value in snapshot.values()
                    )
                    or counters["orbitkv_save_bytes_total"] is None
                    or counters["orbitkv_save_bytes_total"] < (index + 1) * 16384
                    or counters["orbitkv_load_bytes_total"] is None
                    or (
                        tier == "ssd-uring"
                        and (
                            counters["orbitkv_ssd_write_bytes_total"] is None
                            or counters["orbitkv_ssd_write_bytes_total"] < (index + 1) * 16384
                        )
                    )
                ):
                    raise RuntimeError("I/O observation across retired Save rejection differs")
                if tier == "ssd-uring":
                    reads_before, reads_after = row.get("read_before"), row.get("read_after")
                    if not isinstance(reads_before, dict) or not isinstance(reads_after, dict):
                        raise RuntimeError("Missing actual io_uring read evidence")
                    expected_read_keys = {
                        "orbitkv_ssd_prefetch_bytes_total",
                        "orbitkv_ssd_uring_read_operations_total",
                    }
                    if (
                        set(reads_before) != expected_read_keys
                        or set(reads_after) != expected_read_keys
                    ):
                        raise RuntimeError("Missing exact io_uring read counter fields")
                    for name in (
                        "orbitkv_ssd_prefetch_bytes_total",
                        "orbitkv_ssd_uring_read_operations_total",
                    ):
                        before, after = reads_before.get(name), reads_after.get(name)
                        if index and (before is None or before < previous_reads[name]):
                            raise RuntimeError(
                                "io_uring read counter interval regressed or disappeared"
                            )
                        if (
                            type(after) not in (int, float)
                            or not math.isfinite(after)
                            or after <= 0
                            or (
                                before is not None
                                and (
                                    type(before) not in (int, float)
                                    or not math.isfinite(before)
                                    or before < 0
                                    or after <= before
                                )
                            )
                        ):
                            raise RuntimeError("No advancing io_uring read evidence")
                        previous_reads[name] = after
                for role, instance in (("source", source_id), ("consumer", consumer_id)):
                    credit = read_json(
                        summaries[0].parent / f"round-{index}-{role}.credit-owner.json", deadline
                    )
                    if credit.get("instance_id") != instance:
                        raise RuntimeError("Save payload and registration identity differ")
                payload = summaries[0].parent / f"round-{index}.payload.bin"
                raw = read_bytes(payload, 32768, deadline)
                if len(raw) != 32768 or hashlib.sha256(raw).hexdigest() != row.get("sha256"):
                    raise RuntimeError("Installed Save payload archive differs")
                expected = bytearray([167] * 32768)
                for source, target in zip((0, 2, 4, 6), (1, 3, 5, 7), strict=True):
                    expected[target * 4096 : (target + 1) * 4096] = bytes(
                        (source * 4096 + offset + source * 37 + index * 71) % 251
                        for offset in range(4096)
                    )
                if raw != expected:
                    raise RuntimeError("Recomputed installed Save payload/sentinel oracle failed")
                release = read_json(
                    summaries[0].parent / f"round-{index}.native-release.json", deadline
                )
                if (
                    type(release.get("completed_mono_ns")) is not int
                    or release["completed_mono_ns"] <= 0
                ):
                    raise RuntimeError("Save release completion timestamp is invalid")
                if release["completed_mono_ns"] <= previous_release:
                    raise RuntimeError("Save release chronology regressed")
                if rejected["ended_mono_ns"] >= release["completed_mono_ns"]:
                    raise RuntimeError("Retired Save rejection follows its native release")
                for role in ("source", "consumer"):
                    credit_release = read_json(
                        summaries[0].parent / f"round-{index}-{role}.credit-release.json", deadline
                    )
                    if credit_release["observed_mono_ns"] > release["completed_mono_ns"]:
                        raise RuntimeError("Save credit repayment follows its release receipt")
                    if role == "source" and (
                        credit_release["observed_mono_ns"] > rejected["started_mono_ns"]
                    ):
                        raise RuntimeError(
                            "Retired Save rejection precedes source credit repayment"
                        )
                previous_release = release["completed_mono_ns"]
                if release.get("source") != row.get("source_identity") or release.get(
                    "consumer"
                ) != row.get("consumer_identity"):
                    raise RuntimeError("Save payload and release identities differ")
    if cell["kind"] == "sglang-pool":
        releases = evidence_paths(directory, "sglang-registration.native-release.json", deadline)
        if len(releases) != 2:
            raise RuntimeError("Missing SGLang pool lifetime receipts")
        for path in releases:
            row = read_json(path, deadline)
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("workers_joined") is not True
                or row.get("exporters_after_last_hold") != 0
                or row.get("offload_completed") is not True
                or row.get("registration_retired") is not True
                or row.get("client_closed") is not True
            ):
                raise RuntimeError("SGLang pool lifetime contract differs")
    if cell["kind"] == "credits":
        credits = evidence_paths(directory, "*.credit-state.json", deadline)
        if len(credits) != 2:
            raise RuntimeError("Missing explicit credit-state receipts")
        for path in credits:
            row = read_json(path, deadline)
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("credit_counter_exists_and_zero") is not True
                or row.get("physical_fault_drain_claimed") is not False
            ):
                raise RuntimeError("Credit receipt identity or scope differs")
        partial = read_json(
            evidence_paths(directory, "partial-export.credit-state.json", deadline)[0], deadline
        )
        normal = read_json(
            evidence_paths(directory, "duplicate-export.credit-state.json", deadline)[0], deadline
        )
        if (
            partial.get("framework_throw_retained") is not True
            or partial.get("unregister_refused") is not True
            or partial.get("native_registration_or_payload_submitted") is not False
            or partial.get("native_retained_attempts", 0) < 1
            or normal.get("export_calls") != 1
            or normal.get("duplicate_refused_before_export") is not True
            or normal.get("unregister_succeeded") is not True
        ):
            raise RuntimeError("Credit repayment/retention contract failed")
    remaining(deadline)


def supervise(contract_path: Path) -> None:
    deadline = time.monotonic() + 180
    root = contract_path.parent
    owner = identity(os.getpid(), deadline)
    write(root / "SUPERVISOR-STARTED.json", owner, deadline)
    for number in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal.SIG_IGN)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    argv = [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), str(contract_path)]
    with (root / "controller.log").open("x") as output:
        remaining(deadline)
        process = subprocess.Popen(
            argv,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        controller = {"pid": process.pid, "start_ticks": None}
        start_error = None
        try:
            controller = identity(process.pid, deadline)
            if controller["parent"] != owner["pid"]:
                raise RuntimeError("Controller parent identity differs from its wait owner")
            write(
                root / "CONTROLLER-STARTED.json",
                {"controller": controller, "supervisor": owner, "argv": argv},
                deadline,
            )
        except BaseException as error:
            start_error = repr(error)[:1024]
            print(
                f"Controller start evidence failed; wait owner retained: {start_error}", flush=True
            )
        wait_error = None
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                continue
            except BaseException as error:
                if wait_error is None:
                    wait_error = repr(error)[:1024]
                    print(f"Controller wait failed; owner retained: {wait_error}", flush=True)
                time.sleep(1)
        write(
            root / "CONTROLLER-WAIT.json",
            {
                "controller": controller,
                "supervisor": owner,
                "exit_code": code,
                "actual_wait": start_error is None and wait_error is None,
                "forced_cleanup": False,
                "start_error": start_error,
                "wait_error": wait_error,
                "waited_mono_ns": time.monotonic_ns(),
            },
            time.monotonic() + 30,
        )
    if code != 0 or start_error is not None or wait_error is not None:
        raise RuntimeError(f"Controller exit {code}; campaign not accepted")


def main() -> None:
    startup_deadline = time.monotonic() + 180
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized controller execution is forbidden")
    if os.environ.get("LD_AUDIT") or os.environ.get("PYTHONPYCACHEPREFIX"):
        raise RuntimeError("Controller startup contains an unfrozen loader/bytecode influence")
    if args.supervise:
        supervise(args.contract)
        return
    contract_bytes = read_bytes(args.contract, 16 * MAX_JSON_BYTES, startup_deadline)
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    contract = json.loads(contract_bytes)
    if contract.get("qualification") not in (None, "pending-save-bytes-v2"):
        raise RuntimeError("Unsupported qualification contract; historical freezes stay immutable")
    remaining(startup_deadline)
    root = args.contract.parent
    if contract["phase"] not in ("precheck", "formal"):
        raise RuntimeError("Unknown registration gate phase")
    if contract["schema"] != "orbitkv.explicit-registration-campaign.v1":
        raise RuntimeError("Unknown campaign schema")
    expected = (
        [("pending-save-bytes", 1), ("pending-save-bytes", 1)]
        if contract.get("qualification") == "pending-save-bytes-v2"
        else (
            [("native", 2)]
            if contract["phase"] == "precheck"
            else [("credits", 2), ("native", 2), ("vllm-worker", 1), ("sglang-pool", 2)]
        )
    )
    if [(cell["kind"], cell["passed"]) for cell in contract["cells"]] != expected:
        raise RuntimeError("Frozen registration gate cells differ")
    if contract.get("qualification") == "pending-save-bytes-v2":
        if [cell.get("tier") for cell in contract["cells"]] != ["dram", "ssd-uring"]:
            raise RuntimeError("Frozen installed Save medium order differs")
        for cell in contract["cells"]:
            argv = cell.get("argv", [])
            if (
                argv.count("--ssd-backend") != 1
                or argv[argv.index("--ssd-backend") + 1] != "uring"
                or argv.count("--ssd-read-path") != 1
                or argv[argv.index("--ssd-read-path") + 1] != "uring"
            ):
                raise RuntimeError("Installed Save gate requires explicit io_uring configuration")
    for field, maximum in (
        ("review_wait_seconds", 3600),
        ("campaign_seconds", 7200),
        ("cleanup_seconds", 180),
    ):
        if type(contract[field]) is not int or not 0 < contract[field] <= maximum:
            raise RuntimeError("Invalid bounded campaign/review/cleanup budget")
    if any(type(cell["passed"]) is not int for cell in contract["cells"]):
        raise RuntimeError("Invalid declared gate count")
    if identity(os.getpid())["pid"] != os.getpid():
        raise RuntimeError("Invalid controller process identity")
    locks = []
    active = False
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Cannot own orphaned native workers")
    for number in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal.SIG_IGN)
    try:
        check_inputs(contract, deadline=startup_deadline)
        if contract.get("qualification") == "pending-save-bytes-v2":
            write(
                root / "DLOPEN-PREFLIGHT.json",
                check_dlopen(contract, startup_deadline),
                startup_deadline,
            )
        remaining(startup_deadline)
        cpus = contract["cpus"]
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise RuntimeError("Frozen CPU affinity is unavailable")
        remaining(startup_deadline)
        os.sched_setaffinity(0, cpus)
        for resource in contract["locks"]:
            remaining(startup_deadline)
            fd = os.open(resource["path"], os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            remaining(startup_deadline)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(fd).st_ino != resource["inode"]:
                raise RuntimeError("Resource lock inode changed")
            remaining(startup_deadline)
        if Path(contract["quarantine"]).exists():
            raise RuntimeError("GPU resource is quarantined")
        check_inputs(contract, locks, startup_deadline)
        before = inventory(contract["gpu_uuid"], min(startup_deadline, time.monotonic() + 20))
        if before["memory_mib"] != 0 or before["compute"]:
            raise RuntimeError("GPU resource is in use")
        controller = identity(os.getpid())
        ready = {
            "controller": controller,
            "contract_sha256": contract_sha,
            "locks": [
                {"path": row["path"], "device": os.fstat(fd).st_dev, "inode": os.fstat(fd).st_ino}
                for row, fd in zip(contract["locks"], locks, strict=True)
            ],
            "inventory": before,
            "GPU_launches": 0,
            "phase": contract["phase"],
        }
        ready_path = root / "READY.json"
        write(ready_path, ready, startup_deadline)
        ready_sha = digest(ready_path, startup_deadline)
        deadline = time.monotonic() + contract["review_wait_seconds"]
        approval = root / "LAUNCH-APPROVED.json"
        while not approval.exists():
            if time.monotonic() >= deadline:
                write(
                    root / "NOT-LAUNCHED.json", {"GPU_launches": 0, "reason": "review wait expired"}
                )
                return
            time.sleep(0.25)
        if approval.is_symlink() or args.contract.read_bytes() != contract_bytes:
            raise RuntimeError("Launch grant or frozen contract changed")
        check_inputs(contract, locks, deadline)
        grant = read_json(approval, deadline)
        if digest(ready_path, deadline) != ready_sha or identity(os.getpid()) != controller:
            raise RuntimeError("READY or controller incarnation changed")
        validate_grant(grant, ready, ready_sha, contract["phase"], controller)
        fresh = inventory(contract["gpu_uuid"], deadline)
        if fresh["memory_mib"] != 0 or fresh["compute"]:
            raise RuntimeError("Resource is no longer quiet at grant admission")
        global_deadline = time.monotonic() + contract["campaign_seconds"]
        cleanup_spent = 0.0
        completed = []
        for cell in contract["cells"]:
            if time.monotonic() >= global_deadline or args.contract.read_bytes() != contract_bytes:
                raise RuntimeError("Campaign budget exhausted or contract changed before admission")
            check_inputs(contract, locks, global_deadline, cell.get("runtime_manifest_paths"))
            fresh = inventory(contract["gpu_uuid"], global_deadline)
            if fresh["memory_mib"] != 0 or fresh["compute"]:
                raise RuntimeError("GPU resource is in use before cell admission")
            if time.monotonic() >= global_deadline:
                raise RuntimeError("Campaign budget exhausted by input verification")
            directory = root / cell["name"]
            directory.mkdir()
            env = cell_environment(cell, contract["gpu_uuid"])
            env["ORBITKV_TEST_GPU_LOCK_FDS"] = json.dumps(locks)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with (directory / "pytest.log").open("x") as log:
                active = True
                argv = cell["argv"]
                if cell.get("process_trace_binary"):
                    argv = [
                        cell["process_trace_binary"],
                        "-D",
                        "-ff",
                        "-ttt",
                        "-s",
                        "256",
                        "-e",
                        "trace=process",
                        "-o",
                        str(directory / "process.trace"),
                        *argv,
                    ]
                process = subprocess.Popen(
                    argv,
                    cwd=cell["cwd"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(locks),
                )
                deadline = min(global_deadline, time.monotonic() + cell["seconds"])
                owner = identity(process.pid, deadline)
                if cell.get("process_trace_binary"):
                    exec_deadline = min(deadline, time.monotonic() + 10)
                    expected_executable = Path(cell["argv"][0]).resolve()
                    while Path(f"/proc/{process.pid}/exe").resolve() != expected_executable:
                        remaining(exec_deadline)
                        if process.poll() is not None:
                            raise RuntimeError("Traced pytest leader exited before its exec")
                        time.sleep(0.01)
                    write(
                        directory / "TRACED-EXEC.json",
                        {"owner": owner, "executable": str(expected_executable)},
                        deadline,
                    )
                observed = {(owner["pid"], owner["start_ticks"]): owner}
                write(directory / "STARTED.json", owner, deadline)
                while process.poll() is None:
                    for child in descendants(deadline):
                        observed[(child["pid"], child["start_ticks"])] = child
                        if len(observed) > MAX_OBSERVED_PROCESSES:
                            raise RuntimeError("Observed process ownership bound exceeded")
                        if child["parent"] == os.getpid() and child["pid"] != process.pid:
                            pid, status = os.waitpid(child["pid"], os.WNOHANG)
                            if pid:
                                write(
                                    directory / f"ADOPTED-{pid}.json",
                                    dict(
                                        child,
                                        exit_code=os.waitstatus_to_exitcode(status),
                                        actual_wait=True,
                                    ),
                                    deadline,
                                )
                    if time.monotonic() >= deadline:
                        quarantine(
                            root, "Native process deadline without terminal ownership", locks
                        )
                    time.sleep(0.1)
                cleanup_started = time.monotonic()
                cleanup_deadline = min(
                    global_deadline,
                    cleanup_started + max(0.0, contract["cleanup_seconds"] - cleanup_spent),
                )
                remaining(deadline)
                code = process.wait(timeout=remaining(cleanup_deadline))
                write(
                    directory / "WAIT.json",
                    dict(owner, exit_code=code, actual_wait=True),
                    cleanup_deadline,
                )
            while descendants(cleanup_deadline):
                for child in descendants(cleanup_deadline):
                    observed[(child["pid"], child["start_ticks"])] = child
                    if len(observed) > MAX_OBSERVED_PROCESSES:
                        raise RuntimeError("Observed process ownership bound exceeded")
                    if child["parent"] == os.getpid():
                        pid, status = os.waitpid(child["pid"], os.WNOHANG)
                        if pid:
                            write(
                                directory / f"ADOPTED-{pid}.json",
                                dict(
                                    child,
                                    exit_code=os.waitstatus_to_exitcode(status),
                                    actual_wait=True,
                                ),
                            )
                if time.monotonic() >= cleanup_deadline:
                    quarantine(
                        root, "Native descendants remain; cumulative cleanup exhausted", locks
                    )
                time.sleep(0.1)
            check_inputs(contract, locks, cleanup_deadline, cell.get("runtime_manifest_paths"))
            after = inventory(contract["gpu_uuid"], cleanup_deadline)
            if after["memory_mib"] != 0 or after["compute"]:
                quarantine(root, "GPU occupancy remains after owned-process waits", locks)
            write(
                directory / "OBSERVED.json",
                {"processes": list(observed.values())},
                cleanup_deadline,
            )
            validate_cell(cell, directory, owner, list(observed.values()), cleanup_deadline)
            remaining(cleanup_deadline)
            if code != 0:
                raise RuntimeError("First invalid formal cell; no retries or replacement")
            completed.append(cell["name"])
            write(
                directory / "COMPLETE.json",
                {"completed": True, "inventory": after},
                cleanup_deadline,
            )
            cleanup_spent += time.monotonic() - cleanup_started
            if cleanup_spent > contract["cleanup_seconds"]:
                quarantine(root, "Cumulative cleanup/oracle budget exceeded", locks)
        final_started = time.monotonic()
        final_deadline = min(
            global_deadline, final_started + max(0.0, contract["cleanup_seconds"] - cleanup_spent)
        )
        check_inputs(contract, locks, final_deadline)
        if descendants(final_deadline):
            raise RuntimeError("Native descendants remain before final integrity acceptance")
        final_inventory = inventory(contract["gpu_uuid"], final_deadline)
        if final_inventory["memory_mib"] != 0 or final_inventory["compute"]:
            raise RuntimeError("GPU occupancy remains at final postflight")
        if time.monotonic() >= global_deadline:
            raise RuntimeError("Campaign budget exhausted before final acceptance")
        check_inputs(contract, locks, final_deadline)
        write(
            root / "RESULT.json",
            {
                "state": "completed_pending_review",
                "cells": completed,
                "physical_fault_qualification": False,
                "forced_cleanup": False,
                "phase": contract["phase"],
                "formal_cells": len(completed) if contract["phase"] == "formal" else 0,
                "final_inventory": final_inventory,
            },
            final_deadline,
        )
        remaining(final_deadline)
        check_inputs(contract, locks, final_deadline)
        active = False
    except BaseException as error:
        if active:
            quarantine(root, repr(error), locks)
        write(root / "FAILURE.json", {"state": "fail_stop", "error": repr(error)})
        raise
    finally:
        for fd in locks:
            os.close(fd)


if __name__ == "__main__":
    main()
