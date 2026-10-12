"""Keep dedicated GPU leases attached to identity-gate process lifetimes."""

import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


class UnissuedCudaIPCWrapper:
    """Metadata-only input proving a negative guard runs before tensor import."""

    def __init__(self, device_uuid: str):
        self.device_uuid = device_uuid

    def to_tensor(self):
        raise AssertionError("Unissued negative-control IPC handle reached tensor import")


def inherited_gpu_lock_fds() -> tuple[int, ...]:
    descriptors = tuple(json.loads(os.environ.get("ORBITKV_TEST_GPU_LOCK_FDS", "[]")))
    for descriptor in descriptors:
        if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
            raise ValueError("GPU lease descriptors must be nonnegative integers")
        fcntl.fcntl(descriptor, fcntl.F_GETFD)
    return descriptors


@contextmanager
def identity_gpu_locks(root: Path, devices: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    locks = []
    try:
        for device in sorted(devices):
            lock = (root / f"gpu-{device}.lock").open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            quarantine = root / f"gpu-{device}.quarantine.json"
            if quarantine.exists():
                raise RuntimeError(f"GPU {device} is quarantined: {quarantine}")
        yield tuple(lock.fileno() for lock in locks)
    finally:
        for lock in locks:
            lock.close()


def owned_process(process):
    if process is None:
        return None
    result = {"pid": process.pid, "exit_code": process.poll()}
    if result["exit_code"] is None:
        root = Path(f"/proc/{process.pid}")
        try:
            result["start_ticks"] = int((root / "stat").read_text().rsplit(")", 1)[1].split()[19])
            result["command"] = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError, IndexError) as error:
            result["inspection_error"] = str(error)
    return result


def identity_group_members(group: int):
    members, errors = [], []
    try:
        processes = list(Path("/proc").glob("[0-9]*/stat"))
    except OSError as error:
        return members, [str(error)]
    for path in processes:
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members.append(
                    {
                        "pid": int(path.parent.name),
                        "state": fields[0],
                        "start_ticks": int(fields[19]),
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            errors.append(f"{path}: {error}")
    return members, errors


def drain_identity_processes(client, server, release: Path, timeout: float = 60):
    """Release the tensor owner before requesting normal Manager drain.

    A timeout leaves the live process and its inherited flock descriptors intact.
    The caller persists quarantine before closing its own descriptors.
    """
    errors = []
    manager = server.process
    try:
        release.touch()
    except OSError as error:
        errors.append(f"Client release: {error}")
    if client is not None:
        try:
            code = client.wait(timeout=timeout)
            if code != 0:
                errors.append(f"Client normal exit code {code}")
        except subprocess.TimeoutExpired:
            errors.append("Client normal unregister/exit timed out")
        except OSError as error:
            errors.append(f"Client wait failed: {error}")
    client_members, inspection_errors = (
        identity_group_members(client.pid) if client is not None else ([], [])
    )
    errors.extend(inspection_errors)
    inspection_failed = bool(inspection_errors)
    if (
        (client is None or client.poll() is not None)
        and not client_members
        and not inspection_errors
        and server.process is not None
    ):
        try:
            code, _ = server.terminate_gracefully(timeout=timeout)
            if code != 0:
                errors.append(f"Manager normal exit code {code}")
        except subprocess.TimeoutExpired:
            errors.append("Manager normal drain timed out")
        except OSError as error:
            errors.append(f"Manager normal drain failed: {error}")
    processes = [owned_process(process) for process in (client, manager)]
    remaining = [process for process in processes if process and process["exit_code"] is None]
    for process in (client, manager):
        if process is None:
            continue
        deadline = time.monotonic() + timeout
        while True:
            members, inspection_errors = identity_group_members(process.pid)
            if not members or inspection_errors or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        errors.extend(inspection_errors)
        inspection_failed |= bool(inspection_errors)
        if members:
            remaining.extend(members)
            errors.append(f"Owned process group {process.pid} did not drain")
    for process in processes:
        if process and process.get("inspection_error"):
            inspection_failed = True
            errors.append(f"PID {process['pid']} inspection failed: {process['inspection_error']}")
    if inspection_failed:
        remaining.append({"ownership": "unknown", "inspection_errors": errors.copy()})
    return {"errors": errors, "processes": processes, "remaining_processes": remaining}


def require_clean_identity_log(log: str):
    failures = [
        line
        for line in log.splitlines()
        if "ERROR" in line
        or "panicked at" in line
        or "Producer process has been terminated" in line
        or "Exception ignored in" in line
        or "Traceback (most recent call last):" in line
    ]
    assert not failures, failures


def quarantine_identity_gpus(root: Path, devices: list[str], directory: Path, cleanup: dict):
    if not cleanup["remaining_processes"]:
        return
    for device in devices:
        path = root / f"gpu-{device}.quarantine.json"
        try:
            with path.open("x") as stream:
                json.dump({"evidence": str(directory), **cleanup}, stream, indent=2)
                stream.write("\n")
        except FileExistsError:
            continue
