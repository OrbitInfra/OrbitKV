"""Execute one frozen installed registration gate with durable resource ownership."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_JSON_BYTES = 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_EVIDENCE_FILES = 4096
MAX_OBSERVED_PROCESSES = 1024
MAX_RUNTIME_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_FILES = 200000


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
    for manifest in manifests:
        check_runtime_manifest(manifest, deadline)
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
        if len(records) != 1 or len(records[0].encode()) > MAX_JSON_BYTES:
            raise RuntimeError("Missing or duplicate Manager Python origin evidence")
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


def traced_child_wait(directory: Path, child: dict, deadline: float | None = None) -> int:
    path = directory / f"process.trace.{child['parent']}"
    text = read_bytes(path, MAX_LOG_BYTES, deadline).decode(errors="strict")
    matches = []
    for line in text.splitlines():
        remaining(deadline)
        match = re.search(
            r"wait4\([^,]+, \[\{WIFEXITED\(s\) && WEXITSTATUS\(s\) == (\d+)\}\],.*\) = (\d+)$",
            line,
        )
        if match and int(match[2]) == child["pid"]:
            matches.append(int(match[1]))
    if len(matches) != 1:
        raise RuntimeError("Observed native child has no unique traced parent wait")
    terminal = read_bytes(
        directory / f"process.trace.{child['pid']}", MAX_LOG_BYTES, deadline
    ).decode(errors="strict")
    if len(re.findall(rf"\+\+\+ exited with {matches[0]} \+\+\+", terminal)) != 1:
        raise RuntimeError("Traced child exit disagrees with its actual parent wait")
    remaining(deadline)
    return matches[0]


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
        for key in set(known) - accounted:
            code = traced_child_wait(directory, known[key], deadline)
            if code not in cell["adopted_exit_codes"]:
                raise RuntimeError("Traced native child exit is not accepted")
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
    if cell["kind"] == "native":
        expected_origin = cell.get("manager_python")
        if not expected_origin:
            raise RuntimeError("Frozen Manager Python origin contract is required")
        starts = evidence_paths(directory, "*.manager-start.json", deadline)
        if len(starts) != len(receipts):
            raise RuntimeError("Missing actual Manager startup command records")
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
        releases = evidence_paths(directory, "*.native-release.json", deadline)
        if len(releases) != 4:
            raise RuntimeError("Missing complete native client release receipts")
        for path in releases:
            row = read_json(path, deadline)
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("closed") is not True
            ):
                raise RuntimeError("Native client release identity differs")
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
    remaining(startup_deadline)
    root = args.contract.parent
    if contract["phase"] not in ("precheck", "formal"):
        raise RuntimeError("Unknown registration gate phase")
    if contract["schema"] != "orbitkv.explicit-registration-campaign.v1":
        raise RuntimeError("Unknown campaign schema")
    expected = (
        [("native", 2)]
        if contract["phase"] == "precheck"
        else [("credits", 2), ("native", 2), ("vllm-worker", 1), ("sglang-pool", 2)]
    )
    if [(cell["kind"], cell["passed"]) for cell in contract["cells"]] != expected:
        raise RuntimeError("Frozen registration gate cells differ")
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
