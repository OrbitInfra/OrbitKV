"""Installed host-buffer TCP gate; independent of GPU or engine serving qualification."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.release_smoke, pytest.mark.integration]


def test_installed_tcp_payload_and_normal_registration(monkeypatch):
    import orbitkv.orbitkv as native

    extension = Path(native.__file__).resolve(strict=True)
    if "site-packages" not in extension.parts or "target" in extension.parts:
        pytest.fail(f"TCP artifact gate requires an installed extension: {extension}")
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    size, payload, stride = 32768, 1024, 4096
    source = (ctypes.c_ubyte * size)()
    destination = (ctypes.c_ubyte * size)()
    sender = native.MooncakeTransferEngine(bind_host="127.0.0.1")
    receiver = native.MooncakeTransferEngine(bind_host="127.0.0.1")
    for engine, allocation in [(sender, source), (receiver, destination)]:
        engine.register_memory(
            [{"addr": ctypes.addressof(allocation), "len": size, "location": "cpu:0"}]
        )
    for round_number in range(3):
        for operation in ["write", "read"]:
            source[:] = [(index * 17 + round_number * 31) % 251 for index in range(size)]
            destination[:] = [0x5A] * size
            engine, endpoint = (
                (sender, receiver.endpoint) if operation == "write" else (receiver, sender.endpoint)
            )
            local, remote = (source, destination) if operation == "write" else (destination, source)
            slices = [
                (ctypes.addressof(local) + offset, ctypes.addressof(remote) + offset, payload)
                for offset in range(0, size, stride)
            ]
            original_source = bytes(source)
            expected = bytearray([0x5A] * size)
            for offset in range(0, size, stride):
                expected[offset : offset + payload] = original_source[offset : offset + payload]
            completed = getattr(engine, operation)(endpoint, slices, timeout_s=5)
            assert bytes(source) == original_source
            assert completed == len(slices) * payload
            assert bytes(destination) == expected, (
                operation,
                round_number,
                hashlib.sha256(bytes(destination)).hexdigest(),
                hashlib.sha256(expected).hexdigest(),
            )
    maps = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ["libtent_shared", "libmooncake_common", "libasio"])
    }
    assert len(maps) == 3, maps
    assert all("site-packages" in Path(path).parts for path in maps), maps
    sender.unregister_memory([ctypes.addressof(source)])
    receiver.unregister_memory([ctypes.addressof(destination)])
