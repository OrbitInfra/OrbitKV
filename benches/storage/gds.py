"""Qualify native GPU storage on a bare-metal NVMe host; never accepts CPU fallback.

Uses prebuilt artifacts so Cargo cannot restage libraries under running services.
Run from the repository root: python -m benches.storage.gds --help.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from benches.common.artifacts import external_path
from benches.common.launch import STORAGE_CODECS, storage_codec_budget
from benches.common.report import collect_run, compare_outputs

ROOT = Path(__file__).resolve().parents[2]


def native_io_stats(output: str) -> dict:
    """Require explicit per-GPU read/write evidence; unknown formats fail closed."""
    operations = {"read": 0, "write": 0}
    rows = re.findall(r"GPU\s+\d+\s+Read:\s*(.*?)\s+Write:\s*(.*?)(?:BufRegister:|$)", output, re.M)
    if not rows:
        raise ValueError("gds_stats did not report per-GPU counters; enable profile.cufile_stats=3")
    for row in rows:
        for direction, section in zip(operations, row, strict=True):
            counters = dict(re.findall(r"(\w+)=([\d.]+)", section))
            if not {"n", "posix", "err", "unalign"} <= counters.keys():
                raise ValueError(f"Incomplete {direction} counters: {section}")
            if any(
                float(counters.get(name, 0)) != 0
                for name in ("posix", "err", "unalign", "r_sparse", "r_inline")
            ):
                raise ValueError(
                    f"Native qualification rejected fallback/unaligned/failed I/O: {section}"
                )
            operations[direction] += int(counters["n"])
    if not all(operations.values()):
        raise ValueError(f"Both native reads and writes must execute: {operations}")
    return {"operations": operations, "posix_operations": 0, "errors": 0}


def require_bare_metal(ssd_dir: Path) -> dict:
    cgroup = Path("/proc/1/cgroup").read_text()
    root_fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "/"], text=True).strip()
    if (
        root_fs == "overlay"
        or any(Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv"))
        or re.search(r"kubepods|docker|containerd|lxc", cgroup)
    ):
        raise RuntimeError(
            "This gate requires a bare-metal host; container correctness uses docs/gds.md"
        )
    detector = shutil.which("systemd-detect-virt")
    if (
        detector
        and subprocess.run([detector, "--container", "--quiet"], check=False).returncode == 0
    ):
        raise RuntimeError("Container detected; native host qualification was not run")
    mount = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(ssd_dir), "--output", "SOURCE,FSTYPE,TARGET"],
            text=True,
        )
    )["filesystems"][0]
    if mount["fstype"] not in ("ext4", "xfs"):
        raise RuntimeError(f"This local-NVMe gate requires ext4 or xfs: {mount}")
    source = mount["source"].split("[", 1)[0]
    devices = json.loads(
        subprocess.check_output(
            ["lsblk", "--inverse", "--json", "--output", "NAME,TYPE,TRAN", source], text=True
        )
    )
    pending = list(devices["blockdevices"])
    disks = []
    while pending:
        device = pending.pop()
        pending.extend(device.get("children", []))
        if device["type"] == "disk":
            disks.append(device)
    if not disks or any(disk.get("tran") != "nvme" for disk in disks):
        raise RuntimeError(f"Could not establish NVMe backing: {devices}")
    return {"mount": mount, "devices": devices}


def benchmark_commands(args, output: Path):
    """Keep request recipes and budgets equal across all codec/tier controls."""
    for engine in ("vllm", "sglang"):
        for workload in args.workloads:
            for tier, backend in (
                ("dram", "uring"),
                ("ssd", "uring"),
                ("ssd", "auto"),
                ("ssd", "cufile"),
            ):
                for codec in args.storage_codecs:
                    name = f"{engine}-{workload}-{tier}-{backend}-{codec}"
                    command = [
                        str(getattr(args, f"{engine}_python")),
                        "-m",
                        "benches.serving.single_node",
                        "--engine",
                        engine,
                        "--backend",
                        "orbitkv",
                        "--model",
                        str(args.model),
                        "--output",
                        str(output / name),
                        "--host-gib",
                        str(args.host_gib),
                        "--ssd-gib",
                        str(args.ssd_gib if tier == "ssd" else 0),
                        "--ssd-backend",
                        backend,
                        "--storage-codec",
                        codec,
                        "--storage-codec-budget",
                        str(args.storage_codec_budget),
                        "--workload",
                        workload,
                        "--working-set",
                        str(args.working_set),
                        "--lengths",
                        str(args.length),
                        "--duration-seconds",
                        str(args.duration_seconds),
                        "--max-requests",
                        str(args.max_requests),
                        "--gpu-tokens",
                        str(args.gpu_tokens),
                        "--prefill-tokens",
                        str(args.prefill_tokens),
                        "--output-tokens",
                        str(args.output_tokens),
                        "--seed",
                        str(args.seed),
                        "--repeats",
                        "3",
                        "--concurrencies",
                        *map(str, args.concurrencies),
                    ]
                    if tier == "ssd" and backend != "uring":
                        command += ["--gds-stats", str(args.gds_tools / "gds_stats")]
                    yield name, command


def qualify_benchmark(run: dict, manager_usage: dict) -> dict:
    args = run["manifest"]["arguments"]
    if args.get("ssd_read_path") is not None and (
        args.get("queue_warmup") == "on" or args.get("prepare_requests") == "on"
    ):
        raise ValueError(
            "Explicit --ssd-read-path qualification requires --queue-warmup off and "
            "--prepare-requests off; aggregate counters cannot separate demand and preparation reads"
        )
    summary = run["summary"]
    serial = args["workload"] == "serial"
    load_key = "orbitkv_load_bytes" if serial else "orbitkv_load_bytes_total"
    read_path = args.get("ssd_read_path") or args["ssd_backend"]
    read_key = (
        "orbitkv_ssd_prefetch_bytes_total"
        if read_path == "uring"
        else "orbitkv_ssd_cufile_read_bytes_total"
    )
    evidence = {
        "measured_gpu_load_bytes": sum(row.get(load_key, 0) for row in summary),
        "measured_ssd_read_bytes": sum(row.get(read_key, 0) for row in summary),
        "ssd_read_path_requested": args.get("ssd_read_path"),
        "ssd_read_counter": read_key if args["ssd_gib"] else None,
        "native_read_qualified": False,
    }
    if args["ssd_gib"] and not all(
        evidence[key] > 0 for key in ("measured_gpu_load_bytes", "measured_ssd_read_bytes")
    ):
        raise ValueError("Measured workload lacks SSD read/GPU restore evidence")
    counters = manager_usage.get("manager_delta", {})
    if args["storage_codec"] != "none":
        logical = counters.get("orbitkv_storage_codec_bytes_total_logical", 0)
        stored = counters.get("orbitkv_storage_codec_bytes_total_stored", 0)
        if not 0 < stored < logical:
            raise ValueError("Codec run lacks encoded-publication evidence")
        if counters.get("orbitkv_storage_codec_decode_failures_total", 0):
            raise ValueError("Codec run reported decode failures")
    if args["ssd_gib"] and args.get("ssd_read_path") is not None:
        other_key = (
            "orbitkv_ssd_cufile_read_bytes_total"
            if read_path == "uring"
            else "orbitkv_ssd_prefetch_bytes_total"
        )
        if any(row.get(other_key, 0) for row in summary):
            raise ValueError("Measured reads used a different path from --ssd-read-path")
    if args["ssd_gib"] and args["ssd_backend"] == "cufile" and read_path == "uring":
        if counters.get("orbitkv_ssd_cufile_write_bytes_total", 0) <= 0:
            raise ValueError("cuFile backend with host reads lacks cuFile write evidence")
        evidence["native_io_scope"] = (
            "io_uring host-read control; native GDS reads are not qualified"
        )
    if args["ssd_gib"] and read_path != "uring":
        if args["ssd_backend"] == "uring":
            raise ValueError("cuFile read qualification requires a cuFile-capable backend")
        native = json.loads((Path(run["directory"]) / "native-io.json").read_text())
        if (
            not all(native["operations"][key] > 0 for key in ("read", "write"))
            or any(native[key] != 0 for key in ("posix_operations", "errors", "backend_fallbacks"))
            or counters.get("orbitkv_ssd_backend_fallbacks_total", 0)
        ):
            raise ValueError("Missing native reads/writes or fallback observed")
        evidence["native_io"] = native
        evidence["native_read_qualified"] = True
        evidence["native_io_scope"] = (
            "Per-process native I/O evidence; aggregate cuFile counters do not attribute bytes to individual storage representations"
        )
    return evidence


def correctness_result(path: Path, exit_code: int) -> dict:
    cases = list(ET.parse(path).iter("testcase"))
    counts = {
        "tests": len(cases),
        **{
            tag: sum(case.find(tag) is not None for case in cases)
            for tag in ("failure", "error", "skipped")
        },
    }
    counts["passed"] = counts["tests"] - sum(counts[tag] for tag in ("failure", "error", "skipped"))
    return {
        "exit_code": exit_code,
        **counts,
        "qualified": exit_code == 0
        and counts["passed"] > 0
        and counts["failure"] == counts["error"] == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssd-dir",
        type=external_path,
        required=True,
        help="Writable NVMe directory; only a new private subdirectory is used",
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Local dense Qwen3 model, e.g. Qwen3-8B"
    )
    parser.add_argument(
        "--manager", type=Path, required=True, help="Prebuilt release orbitkv-cache-manager-py"
    )
    parser.add_argument(
        "--fault-manager",
        type=Path,
        required=True,
        help="Separate prebuilt Manager with test-hooks",
    )
    parser.add_argument(
        "--core-test",
        type=Path,
        required=True,
        help="Prebuilt Rust cufile integration-test executable",
    )
    parser.add_argument("--vllm-python", type=Path, default=ROOT / ".venv/vllm-release/bin/python")
    parser.add_argument(
        "--sglang-python", type=Path, default=ROOT / ".venv/sglang-release/bin/python"
    )
    parser.add_argument(
        "--gds-tools",
        type=Path,
        required=True,
        help="NVIDIA directory containing gdscheck.py, gdsio and gds_stats",
    )
    parser.add_argument(
        "--cufile-library", type=Path, default=Path("/usr/local/cuda/lib64/libcufile.so.0")
    )
    parser.add_argument("--host-gib", type=int, default=1)
    parser.add_argument("--ssd-gib", type=int, default=16)
    parser.add_argument("--working-set", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument(
        "--storage-codecs", choices=STORAGE_CODECS, nargs="+", default=list(STORAGE_CODECS)
    )
    parser.add_argument(
        "--storage-codec-budget",
        type=storage_codec_budget,
        default=64 * 1024**2,
        help="Benchmark scratch per worker; correctness fixtures use their default 64mb",
    )
    parser.add_argument(
        "--workloads", choices=("serial", "sustained"), nargs="+", default=["serial", "sustained"]
    )
    parser.add_argument("--gpu-tokens", type=int, default=16384)
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if (
        len(set(args.storage_codecs)) != len(args.storage_codecs)
        or args.storage_codecs[0] != "none"
    ):
        parser.error("--storage-codecs must be distinct and start with the none control")
    if len(set(args.workloads)) != len(args.workloads):
        parser.error("--workloads must be distinct")
    if (
        args.gpu_tokens <= 0
        or args.gpu_tokens % 64
        or args.prefill_tokens < 64
        or args.prefill_tokens % 64
        or not 1 <= args.output_tokens <= 64
        or not 1 <= args.max_requests <= 100000
        or len(set(args.concurrencies)) != len(args.concurrencies)
        or any(c not in (1, 4, 8) for c in args.concurrencies)
    ):
        parser.error(
            "use page-aligned token capacities, 1–64 outputs, 1–100000 requests, and distinct concurrencies from 1/4/8"
        )
    args.ssd_dir = args.ssd_dir.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    if (
        min(args.host_gib, args.duration_seconds, args.working_set) < 1
        or args.ssd_gib <= args.host_gib
    ):
        parser.error("require positive host/time/working-set and SSD capacity greater than DRAM")
    if not 64 <= args.length < args.gpu_tokens * 3 // 4 or not 1 <= args.working_set <= 1024:
        parser.error(
            "--length must be between 64 and 3/4 of GPU tokens; working set must be 1–1024"
        )
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        parser.error("the matched-capacity benchmark requires dense Qwen3")
    bytes_per_token = (
        4 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    )
    working_bytes = args.working_set * args.length * bytes_per_token
    if working_bytes <= max(args.host_gib * 1024**3, args.gpu_tokens * bytes_per_token):
        parser.error("working set must exceed both Manager DRAM and engine HBM KV capacities")
    for name in ("manager", "fault_manager", "core_test", "vllm_python", "sglang_python"):
        # Keep venv interpreter symlinks intact so Python finds its pyvenv.cfg.
        value = Path(os.path.abspath(getattr(args, name)))
        if not value.is_file() or not os.access(value, os.X_OK):
            parser.error(f"{name} is not an executable file: {value}")
        setattr(args, name, value)
    args.gds_tools = args.gds_tools.resolve(strict=True)
    for tool in ("gdscheck.py", "gdsio", "gds_stats"):
        if not os.access(args.gds_tools / tool, os.X_OK):
            parser.error(f"missing executable GDS tool: {args.gds_tools / tool}")
    args.cufile_library = args.cufile_library.resolve(strict=True)
    host = require_bare_metal(args.ssd_dir)
    if shutil.disk_usage(args.ssd_dir).free < (max(args.ssd_gib, 8) + 4) * 1024**3:
        parser.error(
            "NVMe directory needs max(SSD capacity, 8 GiB correctness cache) plus 4 GiB free"
        )
    output = Path(tempfile.mkdtemp(prefix="orbitkv-gds-", dir=args.ssd_dir))
    print(f"Qualification artifacts: {output}", flush=True)
    temporary = output / "tmp"
    temporary.mkdir()
    configuration = output / "cufile.json"
    configuration.write_text(
        json.dumps(
            {
                "profile": {"cufile_stats": 3},
                "properties": {"allow_compat_mode": False},
                "fs": {"generic": {"posix_unaligned_writes": False}},
                "logging": {"dir": str(output), "level": "ERROR"},
            },
            indent=2,
        )
        + "\n"
    )
    env = {
        **os.environ,
        "CUFILE_ALLOW_COMPAT_MODE": "false",
        "CUFILE_FORCE_COMPAT_MODE": "false",
        "CUFILE_ENV_PATH_JSON": str(configuration),
        "TMPDIR": str(temporary),
        "LD_PRELOAD": str(args.cufile_library),
        "ORBITKV_CACHE_MANAGER_BINARY": str(args.manager),
        "TRITON_CACHE_DIR": str(output / "kernels/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(output / "kernels/inductor"),
        "VLLM_CACHE_ROOT": str(output / "kernels/vllm"),
    }
    report = {
        "status": "running",
        "host": host,
        "working_set_bytes": working_bytes,
        "stages": [],
        "storage_codecs": args.storage_codecs,
        "storage_codec_budget_bytes_per_worker": args.storage_codec_budget,
        "quality_scope": "Correctness gates retain strict assertions; synthetic serving output differences remain diagnostics. Failed correctness gates fail qualification even if throughput measurements complete.",
    }

    def run(name, command, *, cwd=ROOT, overrides=None, timeout=3600, required=True):
        print(f"Running {name}", flush=True)
        started = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                [str(v) for v in command],
                cwd=cwd,
                env=env | (overrides or {}),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
            finally:
                if process.poll() is None:
                    # Let pytest fixtures / benchmark ExitStack stop their services.
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                report["stages"].append(
                    {
                        "name": name,
                        "command": [str(v) for v in command],
                        "cwd": str(cwd),
                        "exit_code": process.returncode,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                )
                (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        if returncode and required:
            raise RuntimeError(f"{name} failed; see {output / f'{name}.log'}")
        return returncode

    try:
        run("topology", ["nvidia-smi", "topo", "-m"], timeout=30)
        run("gdscheck", [args.gds_tools / "gdscheck.py", "-p"], timeout=120)
        probe = output / "probe.bin"
        try:
            for direction, operation in (("write", 1), ("read", 0)):
                run(
                    f"gdsio-{direction}",
                    [
                        args.gds_tools / "gdsio",
                        "-f",
                        probe,
                        "-d",
                        "0",
                        "-w",
                        "1",
                        "-s",
                        "64M",
                        "-i",
                        "1M",
                        "-x",
                        "0",
                        "-I",
                        str(operation),
                    ],
                    timeout=120,
                )
        finally:
            probe.unlink(missing_ok=True)
        run("core", [args.core_test, "--ignored", "--test-threads=1"], timeout=300)
        run(
            "faults",
            [
                args.vllm_python,
                "-m",
                "pytest",
                "-m",
                "integration",
                "tests/integration/test_cache_faults.py",
                "--ssd-backend",
                "cufile",
                f"--basetemp={temporary / 'faults'}",
            ],
            cwd=ROOT / "python",
            overrides={
                "ORBITKV_CACHE_MANAGER_BINARY": str(args.fault_manager),
                "ORBITKV_FAULT_TESTS": "1",
            },
            timeout=900,
        )
        failed = []
        for engine in ("vllm", "sglang"):
            for tier in ("dram", "ssd"):
                for codec in args.storage_codecs:
                    name = f"{engine}-correctness-{tier}-{codec}"
                    e2e = [
                        getattr(args, f"{engine}_python"),
                        "-m",
                        "pytest",
                        "-m",
                        "e2e",
                        "--model",
                        args.model,
                        "--ssd-backend",
                        "cufile",
                        "--storage-codec",
                        codec,
                        f"--basetemp={temporary / name}",
                        f"--junitxml={output / name}.xml",
                    ]
                    if engine == "vllm":
                        e2e += [
                            "tests/e2e/test_vllm_e2e_correctness.py",
                            "--vllm-cache-tier",
                            tier,
                            "--max-model-len",
                            "4096",
                            "--orbitkv-pool-size",
                            f"{args.host_gib}gb",
                        ]
                    else:
                        e2e += ["tests/e2e/test_sglang_direct_e2e.py", "-k", tier]
                    # Preserve strict quality failures but still collect the matched serving matrix.
                    exit_code = run(name, e2e, cwd=ROOT / "python", required=False)
                    try:
                        quality = correctness_result(output / f"{name}.xml", exit_code)
                    except (OSError, ET.ParseError) as error:
                        quality = {"qualified": False, "exit_code": exit_code, "error": str(error)}
                    report.setdefault("correctness", {})[name] = quality
                    if not quality["qualified"]:
                        failed.append(name)
                    # Each stopped correctness fixture owns an 8 GiB disposable payload.
                    # Retain its logs/JUnit evidence without accumulating every codec's cache.
                    for cache in (temporary / name).rglob("cache.bin"):
                        cache.unlink()
        controls = {}
        for name, command in benchmark_commands(args, output):
            if run(name, command, required=False):
                failed.append(name)
                continue
            try:
                result = collect_run(output / name)
                configuration = result["manifest"]["arguments"]
                usage = json.loads((output / name / "manager-usage.json").read_text())
                result["manager_usage"] = usage
                report.setdefault("benchmarks", {})[name] = result
                result["storage_evidence"] = qualify_benchmark(result, usage)
                key = tuple(
                    configuration[k] for k in ("engine", "workload", "ssd_gib", "ssd_backend")
                )
                if configuration["storage_codec"] == "none":
                    controls[key] = result
                elif key in controls:
                    result["output_reference"] = compare_outputs(result, controls[key])
            except (ValueError, OSError, KeyError) as error:
                report.setdefault("evidence_failures", {})[name] = str(error)
                failed.append(name)
        if failed:
            report["failed_stages"] = failed
            raise RuntimeError(
                f"Qualification failed in {len(failed)} stages; see per-stage evidence"
            )
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status']}: {output / 'qualification.json'}", flush=True)


if __name__ == "__main__":
    main()
