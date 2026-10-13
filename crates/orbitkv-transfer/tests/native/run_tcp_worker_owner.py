"""Run controlled native TCP worker lifetime gates against frozen libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build, runtime = (
        args.native_build.resolve(strict=True),
        args.runtime.resolve(strict=True),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve(strict=True)
    flags = dict(
        line.split(" = ", 1)
        for line in (build / "tent/src/CMakeFiles/tent_shared.dir/flags.make")
        .read_text()
        .splitlines()
        if " = " in line
    )
    source = Path(__file__).with_name("tcp_worker_owner.cpp").resolve(strict=True)
    binary = output / "tcp-worker-owner"
    command = [
        "c++",
        "-std=c++20",
        "-O2",
        *shlex.split(flags["CXX_INCLUDES"]),
        str(source),
        "-L",
        str(runtime),
        f"-Wl,-rpath,{runtime}",
        "-Wl,--export-dynamic",
        "-ltent_shared",
        "-lmooncake_common",
        "-lasio",
        "-lglog",
        "-lnuma",
        "-pthread",
        "-o",
        str(binary),
    ]
    with (output / "compile.log").open("x") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False
        )
    if result.returncode:
        raise RuntimeError(f"TCP owner control compile failed: {result.returncode}")
    exits = []
    for name, mode in [("callback-and-exception", []), ("exception", ["exception"])]:
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(
                [str(binary), *mode],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
        exits.append(result.returncode)
    record = {
        "compile_command": command,
        "run_exits": exits,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(
            (runtime / "libtent_shared.so").read_bytes()
        ).hexdigest(),
        "scope": "CPU TCP worker callback and exception ownership; no DMA failure reclamation",
    }
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    raise SystemExit(any(exits))


if __name__ == "__main__":
    main()
