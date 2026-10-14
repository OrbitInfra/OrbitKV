"""Deterministic CUDA API errors against the consumed CudaPlatform copy owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cuda-include", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, runtime = (
        args.source.resolve(strict=True),
        args.runtime.resolve(strict=True),
    )
    cuda_include = args.cuda_include.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve(strict=True)
    helper = Path(__file__).with_name("cuda_copy_errors.cu").resolve(strict=True)
    allocator = source / "src/platform/cuda/cuda_allocator.cpp"
    binary = output / "cuda-copy-errors"
    command = [
        "c++",
        "-x",
        "c++",
        "-std=c++20",
        "-O2",
        "-DUSE_CUDA",
        "-I",
        str(source / "include"),
        "-I",
        str(cuda_include),
        str(helper),
        str(allocator),
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
        raise RuntimeError(f"CUDA control build failed: {compiled.returncode}")
    exits = []
    for name, arguments in [
        ("copy", []),
        ("driver-init", ["driver-init"]),
        ("admission", ["admission"]),
        ("context-admission", ["context-admission"]),
        ("byte-admission", ["byte-admission"]),
        ("repeated-errors", ["repeated-errors"]),
    ]:
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(
                [str(binary), *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )
        exits.append(result.returncode)
    # These task-owned children use only interposed software CUDA APIs. Their
    # final kill/reap is synthetic-control teardown, never a DMA drain proof.
    retained_modes = []
    for mode in ["copy", "synchronize", "restore", "set-device-restore"]:
        path = output / f"{mode}-retention.log"
        with path.open("x") as log:
            child = subprocess.Popen(
                [str(binary), mode], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                deadline = time.monotonic() + 3
                while not any(
                    marker in path.read_text()
                    for marker in ["termination unproven", "context restoration failed"]
                ):
                    if child.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"No retained CUDA owner: {mode}")
                    time.sleep(0.01)
                time.sleep(0.05)
                text = path.read_text()
                if (
                    child.poll() is not None
                    or "scratch_owner_dropped" in text
                    or (mode != "restore" and "stream_owner_released" in text)
                    or "unproven_copy_returned" in text
                ):
                    raise RuntimeError(f"Premature CUDA scratch release: {mode}")
                retained_modes.append(mode)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=3)
    record = {
        "command": command,
        "compile_exit": compiled.returncode,
        "run_exits": exits,
        "retained_owner_controls": retained_modes,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "allocator_sha256": hashlib.sha256(allocator.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "scope": "Controlled CUDA API software errors; no real GPU or native DMA termination proof",
    }
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    raise SystemExit(any(exits))


if __name__ == "__main__":
    main()
