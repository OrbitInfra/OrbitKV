"""Installed native client/pidfd gate with an actual CPU process-channel peer.

Build crates/orbitkv-channel/tests/completion_process.rs first and pass its
executable in ORBITKV_COMPLETION_TEST_BINARY. Run from outside the checkout
with the final installed wheel. No CUDA operation or Manager is started.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.release_smoke]


def test_installed_peer_liveness_keeps_authenticated_process_identity(tmp_path):
    import orbitkv

    package = Path(orbitkv.__file__).resolve()
    assert "site-packages" in package.parts, f"installed artifact required: {package}"
    executable = Path(os.environ["ORBITKV_COMPLETION_TEST_BINARY"]).resolve(strict=True)
    directory = tmp_path / "peer"
    directory.mkdir()
    (directory / "liveness-only").touch()
    children, clients, exits = [], [], []
    try:
        for index in range(2):
            for name in ("ready", "exit"):
                (directory / name).unlink(missing_ok=True)
            log = (tmp_path / f"peer-{index}.log").open("wb")
            child = subprocess.Popen(
                [str(executable), "--exact", "shared_completion_child", "--nocapture"],
                env={**os.environ, "ORBITKV_COMPLETION_TEST_DIRECTORY": str(directory)},
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            log.close()
            children.append(child)
            deadline = time.monotonic() + 10
            while not (directory / "ready").exists():
                assert child.poll() is None, "process-channel peer exited before readiness"
                assert time.monotonic() < deadline, "process-channel peer readiness timeout"
                time.sleep(0.01)
            client = orbitkv.CacheManagerClient(str(directory / "cache.sock"))
            clients.append(client)
            for _ in range(8):
                assert client.peer_alive()
            if index:
                assert not clients[0].peer_alive(), "old client followed replacement peer"
            (directory / "exit").touch()
            exits.append(child.wait(timeout=10))
            assert exits[-1] == 0
            assert not client.peer_alive()
        (tmp_path / "process-exits.json").write_text(json.dumps(exits) + "\n")
    finally:
        for client in clients:
            client.close()
        for child in children:
            if child.poll() is None:
                (directory / "exit").touch(exist_ok=True)
                child.wait(timeout=10)
    assert exits == [0, 0]
