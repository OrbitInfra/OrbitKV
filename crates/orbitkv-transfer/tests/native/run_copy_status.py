"""Build/run the controlled TENT TCP/C ABI gate against one frozen runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def run(command: list[str], output: Path, cwd: Path, timeout: int) -> int:
    with output.open("x") as log:
        result = subprocess.run(
            command,
            check=False,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return result.returncode


def main() -> None:
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
    flags_file = build / "tent/src/CMakeFiles/tent_shared.dir/flags.make"
    flags = dict(
        line.split(" = ", 1)
        for line in flags_file.read_text().splitlines()
        if " = " in line
    )
    includes = shlex.split(flags["CXX_INCLUDES"])
    source = Path(__file__).with_name("tcp_copy_status.cpp").resolve(strict=True)
    binary = output / "tcp-copy-status"
    command = [
        "c++",
        "-std=c++20",
        "-O2",
        *includes,
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
    compile_exit = run(command, output / "compile.log", source.parent, 120)
    if compile_exit:
        raise RuntimeError(
            f"Control build failed: {compile_exit}; {output / 'compile.log'}"
        )
    test_exit = run([str(binary)], output / "run.log", source.parent, 30)
    record = {
        "compile_command": command,
        "compile_exit": compile_exit,
        "run_command": [str(binary)],
        "run_exit": test_exit,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "runtime": {
            name: hashlib.sha256((runtime / name).read_bytes()).hexdigest()
            for name in ["libtent_shared.so", "libmooncake_common.so", "libasio.so"]
        },
        "scope": "CPU controlled Platform copy Status -> TCP handlers -> C ABI; no GPU/fault drain qualification",
    }
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    raise SystemExit(test_exit)


if __name__ == "__main__":
    main()
