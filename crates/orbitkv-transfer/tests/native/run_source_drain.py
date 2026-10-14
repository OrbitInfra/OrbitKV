"""Run the CPU/TCP source admission, physical handler drain and restart gate."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--ylt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate", choices=("source", "dispatch", "grant"), default="source"
    )
    parser.add_argument("--mutation", choices=("stale-worker-cache",))
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    ylt = args.ylt.resolve(strict=True)
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
        raise RuntimeError("a complete CPU-only TENT closure is required")
    helper = Path(__file__).with_name(
        {
            "source": "source_admission_drain.cpp",
            "dispatch": "rpc_dispatch_drain.cpp",
            "grant": "source_grant_epoch.cpp",
        }[args.gate]
    )
    scenarios = (
        ("delayed", "cancel", "partial", "restart", "delegate", "entropy", "staging")
        if args.gate == "source"
        else ("queued", "running", "throw")
    )
    if args.gate == "grant":
        scenarios = (
            "warm",
            "lost-push",
            "preflight-race",
            "preflight-race-clear",
            "abi-mixed",
        )
    binary = output / "source-drain"
    command = [
        "c++",
        "-std=c++20",
        "-O2",
        "-pthread",
        "-I",
        str(source / "mooncake-transfer-engine/tent/include"),
        str(helper),
        "-I",
        str(ylt / "include"),
        "-I",
        str(ylt / "include/ylt/thirdparty"),
        "-I",
        str(ylt / "include/ylt/standalone"),
        "-DASIO_DYN_LINK",
        "-DASIO_SEPARATE_COMPILATION",
        "-DYLT_ENABLE_DL",
        "-L",
        str(runtime),
        f"-Wl,-rpath,{runtime}",
        "-Wl,--export-dynamic",
        "-ltent_shared",
        "-lmooncake_common",
        "-lasio",
        "-lglog",
        "-lnuma",
        "-ldl",
        "-o",
        str(binary),
    ]
    if args.mutation:
        if args.gate != "grant":
            raise ValueError("worker-cache mutation requires the grant gate")
        command.insert(1, "-DBASELINE_WORKER_CACHE")
    with (output / "compile.log").open("x") as log:
        built = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False
        )
    record = {
        "compile_command": command,
        "gate": args.gate,
        "mutation": args.mutation,
        "compile_exit": built.returncode,
        "runtime_sha256": digest(library),
        "helper_sha256": digest(helper),
        "runner_sha256": digest(Path(__file__)),
        "cells": [],
        "qualification": "CPU/TCP source handler closure only; no CUDA/RDMA, individual ticket reclamation or adapter acceptance",
    }
    if built.returncode == 0:
        record["binary_sha256"] = digest(binary)
        for scenario in scenarios:
            timed_out = False
            with (output / f"{scenario}.log").open("x") as log:
                run = subprocess.Popen(
                    [str(binary), scenario],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process_root = Path("/proc") / str(run.pid)
                start_ticks = int(
                    (process_root / "stat").read_text().rsplit(")", 1)[1].split()[19]
                )
                expected_maps = {
                    str((runtime / name).resolve(strict=True))
                    for name in (
                        "libtent_shared.so",
                        "libmooncake_common.so",
                        "libasio.so",
                    )
                }
                maps = ""
                mapped_paths = set()
                maps_deadline = time.monotonic() + 2
                while run.poll() is None and time.monotonic() < maps_deadline:
                    try:
                        maps = (process_root / "maps").read_text()
                    except FileNotFoundError:
                        break
                    mapped_paths = {
                        row.split()[-1] for row in maps.splitlines() if "/" in row
                    }
                    if expected_maps <= mapped_paths:
                        break
                    time.sleep(0.001)
                (output / f"{scenario}-maps.txt").write_text(maps)
                process_group = os.getpgid(run.pid)
                try:
                    code = run.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    current_ticks = int(
                        (process_root / "stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[19]
                    )
                    if current_ticks != start_ticks:
                        raise RuntimeError(
                            "CPU control process identity changed; refuse signal"
                        )
                    run.kill()
                    code = run.wait(timeout=5)
                    timed_out = True
            still_owned = (
                process_root.exists()
                and int(
                    (process_root / "stat").read_text().rsplit(")", 1)[1].split()[19]
                )
                == start_ticks
            )
            record["cells"].append(
                {
                    "scenario": scenario,
                    "exit_code": code,
                    "pid": run.pid,
                    "start_ticks": start_ticks,
                    "process_group": process_group,
                    "runtime_mapping_verified": expected_maps <= mapped_paths,
                    "wait_reaped": run.returncode is not None,
                    "owned_process_alive_after_wait": still_owned,
                    "forced_cpu_control_cleanup": timed_out,
                    "safe_retention_to_cpu_process_exit": args.gate == "grant"
                    and scenario in ("preflight-race", "abi-mixed"),
                    "normal_teardown": code == 0
                    and not timed_out
                    and not still_owned
                    and not (
                        args.gate == "grant"
                        and scenario in ("preflight-race", "abi-mixed")
                    ),
                }
            )
    (output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    if (
        built.returncode
        or len(record["cells"]) != len(scenarios)
        or any(
            cell["exit_code"] != 0
            or cell["forced_cpu_control_cleanup"]
            or cell["owned_process_alive_after_wait"]
            or not cell["runtime_mapping_verified"]
            for cell in record["cells"]
        )
    ):
        raise RuntimeError("CPU/TCP source drain control failed; preserve its evidence")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
