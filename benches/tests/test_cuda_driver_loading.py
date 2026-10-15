"""Actual collector loading and exec inheritance, with stub CUPTI and no GPU."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    compiler = shutil.which("g++")
    include = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "include"
    if compiler is None or not (include / "cupti.h").is_file():
        pytest.skip("CPU collector loader control requires C++ and CUPTI headers")
    root = tmp_path_factory.mktemp("cpu-collector")
    library = root / "collector.so"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            "-I" + str(include),
            str(Path(__file__).with_name("cuda_driver_trace_loader.cpp")),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


PROGRAM = r"""
import ctypes, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import benches.sglang_kv_oracle as oracle

library, mode, output, config = sys.argv[1:]

def foreground(args):
    assert not any(name in sys.modules for name in ("torch", "sglang", "orbitkv"))
    assert str(library) in Path("/proc/self/maps").read_text()
    ctypes.CDLL(library).cpu_test_copy()
    child = subprocess.run(
        [sys.executable, "-c", "import json,os;print(json.dumps({'pid':os.getpid(),'parent':os.getppid()}))"],
        capture_output=True, text=True, check=True,
    )
    return {"status": "CPU_ONLY_SCOPE_CONTROL", "backend": "direct", "length": 4096,
            "pid": os.getpid(), "child": json.loads(child.stdout)}

oracle.run = foreground
if mode == "legacy-preload":
    print(json.dumps(foreground(SimpleNamespace())))
else:
    args = ["oracle", "--model-config", config, "--length", "4096", "--backend", "direct",
            "--manager-socket", "unused", "--manager-url", "unused", "--gpu-uuid", "unused",
            "--output", output]
    if mode != "env-only":
        args += ["--driver-collector", library]
    if mode == "preinitialized":
        sys.modules["torch"] = SimpleNamespace()
    sys.argv = args
    oracle.main()
"""


def launch(tmp_path, collector, mode):
    directory = tmp_path / "driver"
    directory.mkdir()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "num_hidden_layers": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        )
    )
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env["ORBITKV_CUDA_DRIVER_TRACE_DIR"] = str(directory)
    if mode in {"legacy-preload", "double-activation"}:
        env["LD_PRELOAD"] = str(collector)
    if mode == "missing-dir":
        env.pop("ORBITKV_CUDA_DRIVER_TRACE_DIR")
    if mode == "symlink-dir":
        real = tmp_path / "real-driver"
        directory.rename(real)
        directory.symlink_to(real, target_is_directory=True)
    library = collector
    if mode == "symlink-library":
        library = tmp_path / "collector-link.so"
        library.symlink_to(collector)
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(library), mode, str(tmp_path / "output"), str(config)],
        capture_output=True,
        text=True,
        env=env,
    )
    return result, directory


@pytest.mark.parametrize("mode", ["explicit", "legacy-preload"])
def test_actual_collector_exec_scope_and_matched_negative_control(tmp_path, collector, mode):
    process, directory = launch(tmp_path, collector, mode)
    assert process.returncode == 0, process.stderr
    if mode == "explicit":
        result = json.loads((tmp_path / "output/RESULT.json").read_text())
        assert result["driver_capture"] == {
            "loader": "explicit-dlopen",
            "pid": result["pid"],
            "library": str(collector),
            "sha256": hashlib.sha256(collector.read_bytes()).hexdigest(),
        }
    else:
        result = json.loads(process.stdout)
    assert result["child"]["parent"] == result["pid"]
    expected = {f"cuda-driver-{result['pid']}.jsonl"}
    if mode == "legacy-preload":
        expected.add(f"cuda-driver-{result['child']['pid']}.jsonl")
    assert {p.name for p in directory.iterdir()} == expected
    for path in directory.iterdir():
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events[-1]["complete"] is True
        assert events[-1]["calls"] == (1 if events[0]["pid"] == result["pid"] else 0)


@pytest.mark.parametrize(
    "mode,match",
    [
        ("env-only", "requires --driver-collector"),
        ("missing-dir", "requires a trace directory"),
        ("double-activation", "forbids inherited LD_PRELOAD"),
        ("preinitialized", "before Torch/engine/native"),
        ("symlink-library", "absolute regular library"),
        ("symlink-dir", "absolute, regular and empty"),
    ],
)
def test_ambiguous_activation_fails_before_foreground(tmp_path, collector, mode, match):
    process, _ = launch(tmp_path, collector, mode)
    assert process.returncode != 0
    failure = json.loads((tmp_path / "output/FAILURE.json").read_text())
    assert match in failure["error"]
    assert not (tmp_path / "output/RESULT.json").exists()
