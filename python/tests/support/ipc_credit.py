"""Keep original Torch counter descriptors through installed normal unregister."""

import hashlib
import json
import os
import stat
import struct
import time
from pathlib import Path


def hold_counters(stack, handoff, counter_root=Path("/dev/shm")):
    raw = handoff.read_bytes()
    row = json.loads(raw)
    held = []
    for credit in row["evidence"]["credits"]:
        name = credit["counter_name"].removeprefix("/")
        offset = credit["counter_offset"]
        if (
            not name
            or name in (".", "..")
            or "/" in name
            or "\0" in name
            or type(offset) is not int
            or not 0 <= offset < 10_000
        ):
            raise RuntimeError("Invalid original IPC counter identity")
        fd = os.open(counter_root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stack.callback(os.close, fd)
        metadata = os.fstat(fd)
        observed = credit["observation"]
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (observed["device"], observed["inode"])
            or metadata.st_size < 64 + (offset + 1) * 8
            or os.pread(fd, 8, 64 + offset * 8) != struct.pack("=q", 1)
        ):
            raise RuntimeError("Original IPC counter descriptor is not the borrowed slot")
        held.append((fd, credit))
    if not held:
        raise RuntimeError("No original IPC counter descriptors")
    return row, hashlib.sha256(raw).hexdigest(), held


def release_receipt(held):
    row, handoff_sha, counters = held
    deadline = time.monotonic() + 30
    while True:
        observed = []
        for fd, credit in counters:
            metadata = os.fstat(fd)
            original = credit["observation"]
            raw = os.pread(fd, 8, 64 + credit["counter_offset"] * 8)
            if (metadata.st_dev, metadata.st_ino) != (original["device"], original["inode"]) or len(
                raw
            ) != 8:
                raise RuntimeError("Original IPC counter descriptor changed or cannot be read")
            value = struct.unpack("=q", raw)[0]
            if value not in (0, 1):
                raise RuntimeError("Original IPC counter value is invalid")
            observed.append(
                {
                    "counter_name": credit["counter_name"],
                    "counter_offset": credit["counter_offset"],
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "before": 1,
                    "after": value,
                }
            )
        if all(item["after"] == 0 for item in observed):
            return {
                "schema": "orbitkv.cuda-credit-release.v1",
                "owner": row["owner"],
                "instance_id": row["instance_id"],
                "handoff_sha256": handoff_sha,
                "registration_generation": row["evidence"]["registration_generation"],
                "export_owner_generation": row["evidence"]["export_owner_generation"],
                "counters": observed,
                "observed_mono_ns": time.monotonic_ns(),
                "physical_termination_claimed": False,
            }
        if time.monotonic() >= deadline:
            raise RuntimeError("Original IPC counter repayment remains unproven")
        time.sleep(0.01)
