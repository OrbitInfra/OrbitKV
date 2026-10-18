"""Actual CPU dlopen proves that readiness checks its complete consumed closure."""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support import registration_campaign as campaign


@pytest.mark.parametrize(
    "control",
    ["normal", "missing-mapping", "unfrozen-helper", "changed", "helper-changed", "failed"],
)
def test_actual_cpu_dlopen_requires_all_mapped_bytes(tmp_path, control):
    ctypes.CDLL("libc.so.6")
    libc = next(
        Path(line.split(maxsplit=5)[5])
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].endswith("/libc.so.6")
    )
    helper = Path(__file__).parents[1] / "support" / "runtime_dlopen.py"
    executable = str(Path(sys.executable).resolve())
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="GPU-cpu-fixture",
        MC_FORCE_TCP="1",
        ORBITKV_TEST_STRICT_SHUTDOWN="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    probe = subprocess.run(
        [executable, "-I", "-S", "-B", str(helper), str(libc)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert probe.returncode == 0, probe.stderr
    receipt = json.loads(probe.stdout)
    frozen = dict(receipt["files"])
    for path in [str(helper), executable]:
        frozen[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if control == "missing-mapping":
        frozen.pop(str(libc))
    elif control == "unfrozen-helper":
        frozen.pop(str(helper))
    elif control == "changed":
        frozen[str(libc)] = "0" * 64
    elif control == "helper-changed":
        copied = tmp_path / "helper.py"
        copied.write_text("raise RuntimeError('Changed helper was executed')\n")
        frozen[str(copied)] = frozen[str(helper)]
        helper = copied
    declaration = {"python": executable, "helper": str(helper), "library": str(libc)}
    if control == "failed":
        invalid = tmp_path / "invalid.so"
        invalid.write_bytes(b"not an ELF library")
        declaration["library"] = str(invalid)
        frozen[str(invalid)] = hashlib.sha256(invalid.read_bytes()).hexdigest()
    contract = {
        "runtime_dlopen": declaration,
        "input_sha256": frozen,
        "runtime_manifests": [],
        "gpu_uuid": "GPU-cpu-fixture",
        "cells": [{"environment": env}],
    }
    if control == "normal":
        result = campaign.check_dlopen(contract, time.monotonic() + 10)
        assert result["actual_wait_exit"] == 0 and result["engine_created"] is False
        assert str(libc) in result["files"]
    else:
        with pytest.raises(RuntimeError):
            campaign.check_dlopen(contract, time.monotonic() + 10)
