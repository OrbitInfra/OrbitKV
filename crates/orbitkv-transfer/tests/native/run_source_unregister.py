"""Reproduce logical unregister returning before an admitted TCP source read."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve(strict=True)
    library = runtime / "libtent_shared.so"
    closure = subprocess.run(
        ["ldd", str(library)], capture_output=True, text=True, timeout=10, check=False
    )
    (output / "runtime-ldd.txt").write_text(closure.stdout + closure.stderr)
    if (
        closure.returncode
        or "not found" in closure.stdout
        or any(
            name in closure.stdout for name in ("libcudart", "libcuda.so", "libcufile")
        )
    ):
        raise RuntimeError("this control requires a complete CPU-only TENT runtime")
    helper = Path(__file__).with_name("source_unregister_not_fence.cpp")
    binary = output / "control"
    command = [
        "c++",
        "-std=c++20",
        "-O2",
        "-pthread",
        "-I",
        str(source / "mooncake-transfer-engine/tent/include"),
        str(helper),
        "-L",
        str(runtime),
        f"-Wl,-rpath,{runtime}",
        "-Wl,--export-dynamic",
        "-ltent_shared",
        "-lmooncake_common",
        "-lasio",
        "-lglog",
        "-lnuma",
        "-o",
        str(binary),
    ]
    with (output / "compile.log").open("x") as log:
        compiled = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False
        )
    if compiled.returncode:
        raise RuntimeError(
            f"source unregister control compilation failed: {compiled.returncode}"
        )
    timed_out = False
    with (output / "raw.log").open("x") as log:
        try:
            run = subprocess.run(
                [str(binary)],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            exit_code = run.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
    record = {
        "command": command,
        "compile_exit": compiled.returncode,
        "run_exit": exit_code,
        "control_timeout_forced_cleanup": timed_out,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "scope": "Valid persistent CPU buffers and a test-only CPU copy barrier; real TENT TCP/C ABI; no GPU, RDMA, source-page reclamation or physical-fence acceptance",
    }
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    if timed_out or exit_code:
        raise RuntimeError("source unregister control did not finish normally")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
