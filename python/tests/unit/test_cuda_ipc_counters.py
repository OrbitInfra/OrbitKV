"""Counter diagnostics must distinguish MapInfo from actual exported credits."""

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.trace_sglang_ipc import read_counter


@pytest.mark.parametrize("live_offset", [None, 0, 71], ids=["drained", "first-live", "last-live"])
def test_counter_slots_exclude_file_header(tmp_path, live_offset):
    path = tmp_path / "torch-counter"
    counts = [int(offset == live_offset) for offset in range(72)]
    path.write_bytes(struct.pack("=i", 1) + bytes(60) + struct.pack("=72q", *counts))

    assert [read_counter(path, offset) for offset in range(72)] == counts


@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["negative", "truncated-first", "missing-next"])
def test_invalid_counter_slot_is_not_a_zero(tmp_path, offset):
    path = tmp_path / "torch-counter"
    path.write_bytes(struct.pack("=i", 1) + bytes(60) + bytes(7))

    with pytest.raises(ValueError):
        read_counter(path, offset)


def test_spawned_worker_installs_trace_without_starting_server(tmp_path):
    helper = Path(__file__).parents[1] / "support" / "trace_sglang_ipc.py"
    script = """
import os
import runpy
import sys
from types import ModuleType

linker = ModuleType("orbitkv.sglang.linker")
original_close = lambda self: None
linker.OrbitKVLinker = type("Linker", (), {"close": original_close})
package = ModuleType("orbitkv")
package.__path__ = []
sglang = ModuleType("orbitkv.sglang")
sglang.__path__ = []
package.sglang = sglang
sglang.linker = linker
sys.modules.update({"orbitkv": package, "orbitkv.sglang": sglang, "orbitkv.sglang.linker": linker})
os.environ["ORBITKV_CLOSE_TRACE_DIRECTORY"] = sys.argv[2]
runpy.run_path(sys.argv[1], run_name="__mp_main__")
assert linker.OrbitKVLinker.close is not original_close
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(helper), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
