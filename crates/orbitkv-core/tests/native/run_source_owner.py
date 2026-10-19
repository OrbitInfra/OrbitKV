"""Actual Rust source pool owners with CPU host-registration/C ABI fault controls.

This control uses real memfd mappings and frozen TENT CPU mechanics. CUDA host
registration is a test shim; it provides no GPU or physical-device qualification.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-binary", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--cuda-include", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registration-control", type=Path)
    args = parser.parse_args()
    binary = args.unit_binary.resolve(strict=True)
    source = args.source.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    cuda_include = args.cuda_include.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve(strict=True)
    fixture = Path(__file__).parent
    bundle = output / "runtime"
    bundle.mkdir()
    for name in ("libmooncake_common.so", "libasio.so"):
        shutil.copy2(runtime / name, bundle / name)
    actual = bundle / "libtent_actual.so"
    shutil.copy2(runtime / "libtent_shared.so", actual)
    subprocess.run(
        ["patchelf", "--set-soname", actual.name, str(actual)], check=True, timeout=10
    )
    commands = [
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-x",
            "c++",
            "-I",
            str(cuda_include),
            str(fixture / "source_owner_host_shim.cu"),
            "-o",
            str(bundle / "libcudart.so"),
        ],
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(source / "mooncake-transfer-engine/tent/include"),
            str(fixture / "source_owner_tent_shim.cpp"),
            "-L",
            str(bundle),
            "-Wl,--no-as-needed",
            "-l:libtent_actual.so",
            "-Wl,-rpath,$ORIGIN",
            "-ldl",
            "-o",
            str(bundle / "libtent_shared.so"),
        ],
    ]
    build_exits = []
    for index, command in enumerate(commands):
        with (output / f"compile-{index}.log").open("x") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=False
            )
        build_exits.append(result.returncode)
        if result.returncode:
            break
    scenarios = ["normal", "unsupported", "still-owned", "unregister-error", "partial"]
    registration_control = (
        args.registration_control.resolve(strict=True)
        if args.registration_control is not None
        else None
    )
    if registration_control is not None:
        scenarios.append("native-register-partial")
    record = {
        "commands": commands,
        "build_exits": build_exits,
        "unit_binary_sha256": digest(binary),
        "production_cpu_native_sha256": digest(runtime / "libtent_shared.so"),
        "fixture_sha256": {
            name: digest(fixture / name)
            for name in ("source_owner_host_shim.cu", "source_owner_tent_shim.cpp")
        },
        "runner_sha256": digest(Path(__file__)),
        "registration_control_sha256": (
            digest(registration_control) if registration_control is not None else None
        ),
        "scope": "Rust owner/memfd CPU controls; separate withheld-CABI cases and actual native partial-provider refusal; no GPU/physical termination qualification",
        "cells": [],
    }
    if build_exits == [0, 0]:
        record["test_bundle"] = {p.name: digest(p) for p in sorted(bundle.iterdir())}
        for scenario in scenarios:
            environment = os.environ.copy()
            environment.update(
                {
                    "MC_FORCE_TCP": "1",
                    "ORBITKV_MOONCAKE_LIB_DIR": str(bundle),
                    "LD_LIBRARY_PATH": str(bundle),
                    "ORBITKV_TEST_HOST_SHIM": str(bundle / "libcudart.so"),
                    "ORBITKV_TEST_SOURCE_OWNER_CASE": scenario,
                    "ORBITKV_TEST_OWNER_MAPS": str(output / f"{scenario}-maps.txt"),
                }
            )
            if registration_control is not None:
                environment["ORBITKV_REGISTER_CONTROL_LIBRARY"] = str(
                    registration_control
                )
            command = [
                str(binary),
                "peer::transport::tests::actual_source_owner_preserves_partial_and_unproven_resources",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]
            forced = False
            with (output / f"{scenario}.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                proc = Path("/proc") / str(child.pid)
                ticks = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
                try:
                    code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if (
                        int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
                        != ticks
                    ):
                        raise RuntimeError(
                            "owner control identity changed; refuse signal"
                        )
                    child.kill()
                    code = child.wait(timeout=5)
                    forced = True
            maps_path = output / f"{scenario}-maps.txt"
            maps = maps_path.read_text() if maps_path.is_file() else ""
            expected = [
                bundle / name
                for name in ("libcudart.so", "libtent_shared.so", "libtent_actual.so")
            ]
            if scenario == "native-register-partial":
                expected.append(registration_control)
            mapping = all(str(p) in maps for p in expected) and "libcuda.so" not in maps
            record["cells"].append(
                {
                    "scenario": scenario,
                    "pid": child.pid,
                    "start_ticks": ticks,
                    "exit_code": code,
                    "actual_wait_reap": child.returncode is not None,
                    "forced_cpu_cleanup": forced,
                    "mapping_verified": mapping,
                }
            )
    if (
        registration_control is not None
        and digest(registration_control) != record["registration_control_sha256"]
    ):
        raise RuntimeError("Native registration control changed during the gate")
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    if (
        build_exits != [0, 0]
        or len(record["cells"]) != len(scenarios)
        or any(
            cell["exit_code"] != 0
            or cell["forced_cpu_cleanup"]
            or not cell["mapping_verified"]
            for cell in record["cells"]
        )
    ):
        raise RuntimeError("CPU owner control failed; preserve all evidence")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
