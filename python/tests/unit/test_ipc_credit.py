"""Actual CPU descriptors protect original-counter byte evidence."""

import contextlib
import json
import os
import struct

import pytest

from tests.support import ipc_credit


@pytest.mark.parametrize("control", ["normal", "unlink-replace", "short", "nonzero", "symlink"])
def test_original_descriptor_credit_transition(tmp_path, monkeypatch, control):
    counter = tmp_path / "counter"
    counter.write_bytes(bytes(64) + struct.pack("=q", 1))
    metadata = counter.stat()
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "owner": {"pid": os.getpid()},
                "instance_id": "source",
                "evidence": {
                    "registration_generation": "generation",
                    "export_owner_generation": "export",
                    "credits": [
                        {
                            "counter_name": "/counter",
                            "counter_offset": 0,
                            "observation": {"device": metadata.st_dev, "inode": metadata.st_ino},
                        }
                    ],
                },
            }
        )
    )
    with contextlib.ExitStack() as stack:
        if control == "symlink":
            counter.rename(tmp_path / "backing")
            counter.symlink_to(tmp_path / "backing")
            with pytest.raises(OSError):
                ipc_credit.hold_counters(stack, handoff, tmp_path)
            return
        held = ipc_credit.hold_counters(stack, handoff, tmp_path)
        with counter.open("r+b") as original:
            if control == "unlink-replace":
                counter.unlink()
                counter.write_bytes(bytes(64) + struct.pack("=q", 1))
            original.seek(64)
            original.write(struct.pack("=q", 1 if control == "nonzero" else 0))
            original.flush()
            if control == "short":
                original.truncate(64)
        if control == "nonzero":
            ticks = iter([0, 31])
            monkeypatch.setattr(ipc_credit.time, "monotonic", lambda: next(ticks))
        if control in ("short", "nonzero"):
            with pytest.raises(RuntimeError):
                ipc_credit.release_receipt(held)
        else:
            receipt = ipc_credit.release_receipt(held)
            assert receipt["counters"][0]["inode"] == metadata.st_ino
            assert receipt["counters"][0]["after"] == 0
            if control == "unlink-replace":
                assert counter.stat().st_ino != metadata.st_ino
                assert struct.unpack("=q", counter.read_bytes()[64:])[0] == 1
