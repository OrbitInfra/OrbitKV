"""Hold one host-local exclusive resource lock through campaign cleanup."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import stat
import time
from pathlib import Path

try:
    from .environment import direct_network_environment_evidence
    from .integrity import verify_inputs
except ImportError:
    from environment import direct_network_environment_evidence
    from integrity import verify_inputs


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    verify_inputs(args.root, args.approved_input_manifest_sha256)
    settings = json.loads((args.root / "inputs/settings.json").read_text())
    direct_network_environment_evidence(os.environ, settings["network_environment"])
    node = next(item for item in settings["nodes"] if item["name"] == args.node)
    if socket.gethostname() != node["hostname"]:
        raise RuntimeError("resource lock hostname mismatch")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot_id != node["boot_id"]:
        raise RuntimeError("resource lock boot ID mismatch")

    lock_path = Path(settings["resource_lock"]["path"])
    ready = args.root / f"{args.node}.resource-lock.json"
    stop = args.root / f"{args.node}.resource-lock.stop"
    released = args.root / f"{args.node}.resource-lock.released.json"
    if ready.exists() or stop.exists() or released.exists():
        raise FileExistsError("resource lock evidence already exists")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.with_suffix(".quarantine.json").exists():
        raise RuntimeError("resource lock has unresolved quarantine")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(fd, "a+")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError("resource lock file owner/type differs")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {
        "status": "locked",
        "task_id": settings["task_id"],
        "node": args.node,
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "pid": os.getpid(),
        "start_ticks": start_ticks(os.getpid()),
        "lock_path": str(lock_path),
        "lock_device": info.st_dev,
        "lock_inode": info.st_ino,
        "lock_uid": info.st_uid,
        "resources": settings["resource_lock"]["resources"],
        "approved_input_manifest_sha256": args.approved_input_manifest_sha256,
        "started_unix_ns": time.time_ns(),
    }
    atomic_json(ready, record)
    while not stop.exists():
        current = lock_path.lstat()
        if (
            current.st_dev != info.st_dev
            or current.st_ino != info.st_ino
            or current.st_uid != info.st_uid
        ):
            atomic_json(
                lock_path.with_suffix(".quarantine.json"),
                {**record, "status": "quarantined", "reason": "lock_identity_changed"},
            )
            while True:
                time.sleep(60)
        time.sleep(0.1)
    atomic_json(
        released,
        {**record, "status": "released", "finished_unix_ns": time.time_ns()},
    )


if __name__ == "__main__":
    main()
