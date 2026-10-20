"""Shared Linux process identity, mapped-runtime capture and input checks for tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_JSON_BYTES = 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024

MAX_EVIDENCE_FILES = 4096

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


def record_process_group_runtime(
    process: subprocess.Popen,
    start_ticks: int,
    log_path: Path,
    deadline: float,
) -> None:
    """Record a live service and its workers before the owner requests shutdown."""
    if process.poll() is not None:
        raise RuntimeError("Service exited before its runtime capture")
    owner = identity(process.pid, deadline)
    if owner["start_ticks"] != start_ticks or owner["group"] != process.pid:
        raise RuntimeError("Service runtime owner identity changed")

    def members():
        found = []
        for path in Path("/proc").glob("[0-9]*/stat"):
            remaining(deadline)
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == owner["group"]:
                    found.append(identity(int(path.parent.name), deadline))
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not found or len(found) > 64:
            raise RuntimeError("Service runtime process group is empty or exceeds its bound")
        return sorted(found, key=lambda item: item["pid"])

    before = members()
    if owner not in before or any(item["session"] != owner["session"] for item in before):
        raise RuntimeError("Service runtime process group ownership differs")
    records = []
    for member in before:
        pid = member["pid"]
        root = Path(f"/proc/{pid}")
        argv = read_bytes(root / "cmdline", MAX_JSON_BYTES, deadline).split(b"\0")[:-1]
        environment = dict(
            item.split(b"=", 1)
            for item in read_bytes(root / "environ", MAX_JSON_BYTES, deadline).split(b"\0")
            if b"=" in item
        )
        runtime = capture_process_runtime(pid, member["start_ticks"], deadline)
        if runtime["owner"] != member:
            raise RuntimeError("Service worker identity changed during runtime capture")
        records.append(
            {
                "argv": [value.decode() for value in argv],
                "environment": {
                    key: environment.get(key.encode(), b"").decode()
                    for key in (
                        "PYTHONPATH",
                        "LD_LIBRARY_PATH",
                        "CUDA_VISIBLE_DEVICES",
                        "ORBITKV_CHANNEL_CONFIG",
                        "VLLM_USE_V2_MODEL_RUNNER",
                    )
                },
                **runtime,
            }
        )
    if (
        process.poll() is not None
        or identity(process.pid, deadline) != owner
        or members() != before
    ):
        raise RuntimeError("Service runtime process group changed during capture")
    write(
        Path(f"{log_path}.{owner['pid']}-{start_ticks}.process-runtime.json"),
        {"owner": owner, "processes": records, "physical_termination_claimed": False},
        deadline,
    )


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
