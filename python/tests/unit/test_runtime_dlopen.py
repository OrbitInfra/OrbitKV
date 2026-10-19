"""Actual CPU dlopen proves that readiness checks its complete consumed closure."""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support import registration_campaign as campaign


@pytest.fixture
def dlopen_contract():
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
    frozen = json.loads(probe.stdout)["files"]
    for path in [str(helper), executable]:
        frozen[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    contract = {
        "runtime_dlopen": {"python": executable, "helper": str(helper), "library": str(libc)},
        "input_sha256": frozen,
        "runtime_manifests": [],
        "gpu_uuid": "GPU-cpu-fixture",
        "cells": [{"environment": env}],
    }
    state = ctypes.c_int()
    libc = ctypes.CDLL(None)
    assert libc.prctl(37, ctypes.byref(state), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        yield contract
    finally:
        assert libc.prctl(36, state.value, 0, 0, 0) == 0


@pytest.mark.parametrize(
    "control",
    ["normal", "missing-mapping", "unfrozen-helper", "changed", "helper-changed", "failed"],
)
def test_actual_cpu_dlopen_requires_all_mapped_bytes(tmp_path, control, dlopen_contract):
    contract = dlopen_contract
    declaration, frozen = contract["runtime_dlopen"], contract["input_sha256"]
    helper = Path(declaration["helper"])
    if control == "missing-mapping":
        frozen.pop(declaration["library"])
    elif control == "unfrozen-helper":
        frozen.pop(str(helper))
    elif control == "changed":
        frozen[declaration["library"]] = "0" * 64
    elif control == "helper-changed":
        copied = tmp_path / "helper.py"
        copied.write_text("raise RuntimeError('Changed helper was executed')\n")
        frozen[str(copied)] = frozen[str(helper)]
        declaration["helper"] = str(copied)
    if control == "failed":
        invalid = tmp_path / "invalid.so"
        invalid.write_bytes(b"not an ELF library")
        declaration["library"] = str(invalid)
        frozen[str(invalid)] = hashlib.sha256(invalid.read_bytes()).hexdigest()
    if control == "normal":
        result = campaign.check_dlopen(contract, time.monotonic() + 10)["probes"][0]
        assert result["actual_wait_exit"] == 0 and result["engine_created"] is False
        assert result["pid"] == result["owner"]["pid"] and result["cell_index"] == 0
        assert declaration["library"] in result["files"]
    else:
        with pytest.raises(RuntimeError):
            campaign.check_dlopen(contract, time.monotonic() + 10)


def test_each_cell_loader_environment_is_checked(tmp_path, dlopen_contract):
    contract = dlopen_contract
    library = next(Path(path) for path in contract["input_sha256"] if "/libm.so." in path)
    soname = library.name.split(".so.")[0] + ".so." + library.name.split(".so.")[1].split(".")[0]
    shutil.copyfile(library, tmp_path / soname)
    contract["cells"].append(
        {"environment": dict(contract["cells"][0]["environment"], LD_LIBRARY_PATH=str(tmp_path))}
    )
    with pytest.raises(RuntimeError, match="unfrozen mapping"):
        campaign.check_dlopen(contract, time.monotonic() + 10)


@pytest.mark.parametrize(
    "control",
    [
        "normal",
        "foreign-pid",
        "stderr",
        "fork-exit",
        "fork-timeout",
        "fork-session",
        "no-subreaper",
    ],
)
def test_probe_has_bounded_output_and_actual_child_wait(tmp_path, dlopen_contract, control):
    contract = dlopen_contract
    helper = tmp_path / "helper.py"
    original = contract["runtime_dlopen"]["helper"]
    prefix = "import os,sys,time,runpy,json,io,contextlib\n"
    baseline_helper = tmp_path / "baseline-helper.py"
    baseline_helper.write_text(prefix + f"runpy.run_path({original!r},run_name='__main__')\n")
    baseline = subprocess.run(
        [
            contract["runtime_dlopen"]["python"],
            "-I",
            "-S",
            "-B",
            str(baseline_helper),
            contract["runtime_dlopen"]["library"],
        ],
        env=contract["cells"][0]["environment"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert baseline.returncode == 0, baseline.stderr
    for path, digest in json.loads(baseline.stdout)["files"].items():
        assert contract["input_sha256"].get(path, digest) == digest
        contract["input_sha256"][path] = digest
    if control.startswith("fork-"):
        prefix += "ready_read,ready_write=os.pipe()\nif os.fork() == 0:\n os.close(ready_read)\n"
        if control == "fork-session":
            prefix += " os.setsid()\n"
        prefix += (
            " os.close(1);os.close(2);os.write(ready_write,b'1');os.close(ready_write)\n"
            " time.sleep(120);os._exit(0)\n"
            "os.close(ready_write)\n"
            "if os.read(ready_read,1)!=b'1':raise RuntimeError('child readiness failed')\n"
            "os.close(ready_read)\n"
        )
        if control == "fork-timeout":
            prefix += "time.sleep(120)\n"
    elif control == "stderr":
        prefix += "os.write(2,b'x'*65536);os.write(2,b'x'*65536)\n"
    if control == "foreign-pid":
        prefix += (
            "captured=io.StringIO()\n"
            "with contextlib.redirect_stdout(captured):\n"
            f" runpy.run_path({original!r},run_name='__main__')\n"
            "receipt=json.loads(captured.getvalue());receipt['pid']=1\n"
            "print(json.dumps(receipt),flush=True)\n"
        )
    else:
        prefix += f"runpy.run_path({original!r},run_name='__main__')\n"
    helper.write_text(prefix)
    contract["runtime_dlopen"]["helper"] = str(helper)
    contract["input_sha256"][str(helper)] = hashlib.sha256(helper.read_bytes()).hexdigest()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import ctypes,sys,json,time,os,signal\n"
        f"sys.path.insert(0,{str(Path(__file__).parents[2])!r})\n"
        "from tests.support import registration_campaign as c\n"
        f"if {control != 'no-subreaper'!r} and ctypes.CDLL(None).prctl(36,1,0,0,0):"
        "raise RuntimeError('subreaper failed')\n"
        "c.MAX_LOG_BYTES=65536\n"
        "try:\n"
        f" result=c.check_dlopen(json.load(open({str(contract_path)!r})),"
        f"time.monotonic()+{2 if control == 'fork-timeout' else 10})\n"
        " outcome={'accepted':True,'result':result}\n"
        "except BaseException as e:\n"
        " outcome={'accepted':False,'error':repr(e)}\n"
        "children=c.descendants(time.monotonic()+1)\n"
        "outcome['remaining_children']=children\n"
        "for child in children:\n"
        " os.kill(child['pid'],signal.SIGKILL)\n"
        " os.waitpid(child['pid'],0)\n"
        "print(json.dumps(outcome),flush=True)\n"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(driver)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    outcome = json.loads(process.stdout)
    assert outcome["remaining_children"] == [], outcome
    assert outcome["accepted"] == (control == "normal"), outcome
    if control == "normal":
        receipt = outcome["result"]["probes"][0]
        assert receipt["pid"] == receipt["owner"]["pid"]
        assert receipt["actual_wait_exit"] == 0
