"""CPU-only same-process trace calibration with complete post-drain validation.

Run only from an independently reviewed frozen input index. Neither captured
addresses nor CUDA APIs are used. Every failed cell is retained and stops the
cohort; a valid guard failure does not authorize a replacement.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import external_path
from .local_restore_shapes import COPY_FIELDS, _collect, _decode_copies

NAME = "transfer::local::trace::profile::captured_geometry_profile"
DYNAMIC = {"pid", "trace_id", "tid", "begin_ns", "submitted_ns", "drained_ns"}
METRICS = ("total_ns", "observed_ns")
TIMINGS = (
    "open_ns",
    "geometry_ns",
    "batch_capture_ns",
    "finish_ns",
    "encoding_ns",
    "publish_ns",
    "flush_ns",
    *METRICS,
)
CELL_FILES = {
    "PROFILE.stdout",
    "PROFILE.stderr",
    "PROFILE-WAIT.json",
    "SAMPLES.json",
    "GEOMETRY.json",
    "COMPILER.stdout",
    "COMPILER.stderr",
    "COMPILER-WAIT.json",
    "COMPILED.json",
    "RESULT.json",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def records(path):
    return [
        json.loads(line.split("local_raw_copy ", 1)[1])
        for line in Path(path).read_text().splitlines()
        if "local_raw_copy " in line
    ]


class ChildDeadline(Exception):
    pass


def snapshot():
    paths = [
        "/proc/loadavg",
        "/proc/meminfo",
        "/proc/pressure/cpu",
        "/proc/pressure/memory",
        "/proc/pressure/io",
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
        "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        "/sys/fs/cgroup/memory/memory.stat",
        "/sys/fs/cgroup/memory/memory.numa_stat",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]
    data = {"monotonic_ns": time.monotonic_ns(), "files": {}}
    for name in paths:
        try:
            data["files"][name] = {"text": Path(name).read_text()}
        except OSError as error:
            data["files"][name] = {"unavailable": repr(error)}
    data["cpu_jiffies"] = [
        line
        for line in Path("/proc/stat").read_text().splitlines()
        if line.split()[0] in {"cpu0", "cpu1", "cpu2", "cpu3"}
    ]
    return data


def identity(pid):
    tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(tail[19])


def deadline_wait(child, seconds):
    def alarm(signum, frame):
        raise ChildDeadline("direct child deadline expired")

    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("Existing timer forbids replacing the deadline")
    signal.signal(signal.SIGALRM, alarm)
    result = None
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        result = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(result[1])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return result


def run(cmd, env, cell, label, timeout=120, cpu=None):
    before = snapshot()
    started = time.monotonic()
    errors = []
    forced = False
    waited = None
    ticks = None
    child = None
    live = None
    pidfd = None
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("preexec affinity requires a single-threaded controller")

    def affinity():
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})

    with (
        (cell / (label + ".stdout")).open("xb") as out,
        (cell / (label + ".stderr")).open("xb") as err,
    ):
        try:
            child = subprocess.Popen(
                cmd,
                env=env,
                stdout=out,
                stderr=err,
                preexec_fn=affinity if cpu is not None else None,
            )
            pidfd = os.pidfd_open(child.pid, 0)
            ticks = identity(child.pid)
            actual_exe = Path(f"/proc/{child.pid}/exe").resolve(strict=True)
            live = {
                "pid": child.pid,
                "start_ticks": ticks,
                "executable": str(actual_exe),
                "executable_sha256": sha(actual_exe),
                "affinity": sorted(os.sched_getaffinity(child.pid)),
                "status": [
                    line
                    for line in Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    if line.startswith(("NSpid:", "Cpus_allowed_list:", "Mems_allowed_list:"))
                ],
            }
            assert actual_exe == Path(cmd[0]).resolve(), live
            if cpu is not None:
                assert live["affinity"] == [cpu], live
            waited = deadline_wait(child, timeout)
        except BaseException as error:
            errors.append({"stage": "launch_identity_or_wait", "error": repr(error)})
        finally:
            if child is not None and child.returncode is None:
                # Only cleanup our still-unreaped direct child, never a reused PID.
                for sig, seconds in [(signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
                    try:
                        if pidfd is None:
                            errors.append(
                                {
                                    "stage": "cleanup",
                                    "error": "no pidfd: no numeric PID signal permitted",
                                }
                            )
                            break
                        signal.pidfd_send_signal(pidfd, sig, None, 0)
                        forced = True
                    except ProcessLookupError:
                        pass
                    except OSError as error:
                        errors.append({"stage": "cleanup_signal", "error": repr(error)})
                    end = time.monotonic() + seconds
                    while time.monotonic() < end:
                        try:
                            item = os.wait4(child.pid, os.WNOHANG)
                        except ChildProcessError as error:
                            errors.append({"stage": "cleanup_wait", "error": repr(error)})
                            break
                        if item[0]:
                            waited = item
                            child.returncode = os.waitstatus_to_exitcode(item[1])
                            break
                        time.sleep(0.01)
                    if child.returncode is not None:
                        break
    if pidfd is not None:
        os.close(pidfd)
    code = None if child is None else child.returncode
    usage = None
    if waited is not None:
        assert waited[0] == child.pid
        ru = waited[2]
        usage = {
            k: getattr(ru, k)
            for k in [
                "ru_utime",
                "ru_stime",
                "ru_maxrss",
                "ru_minflt",
                "ru_majflt",
                "ru_inblock",
                "ru_oublock",
                "ru_nvcsw",
                "ru_nivcsw",
            ]
        }
    ended = True
    if child is not None:
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            ended = ticks is not None and identity(child.pid) != ticks
    receipt = {
        "argv": cmd,
        "pid": None if child is None else child.pid,
        "start_ticks": ticks,
        "actual_wait_returncode": code,
        "actual_wait4_observed": waited is not None,
        "actual_wait4_raw_status": None if waited is None else waited[1],
        "forced_cleanup": forced,
        "identity_ended": ended,
        "status": "NORMAL"
        if code == 0 and waited and not forced and not errors and ended
        else "INVALID",
        "seconds": time.monotonic() - started,
        "rusage": usage,
        "live_identity": live,
        "errors": errors,
        "before": before,
        "after": snapshot(),
        "stdout_sha256": sha(cell / (label + ".stdout")),
        "stderr_sha256": sha(cell / (label + ".stderr")),
        "scope": "rusage covers complete child incl startup/input/output; host/cgroup counters shared, not timed-path attribution",
    }
    if code is None:
        receipt["status"] = "INVALID_UNREAPED_OR_WAIT_UNVERIFIED"
    write(cell / (label + "-WAIT.json"), receipt)
    if receipt["status"] != "NORMAL":
        raise RuntimeError("Invalid child: " + str(cell) + " " + label)
    return receipt


def validate_probe(report):
    assert report["probe"] == "per-sequence"
    enabled = report["contrast"] == "ab"
    resolution = report["thread_cpu_resolution_ns"]
    assert (type(resolution) is int and resolution > 0) if enabled else resolution is None
    assert type(report["profile_tid"]) is int and report["profile_tid"] > 0
    for row in report["samples"]:
        for key in [
            "total_ns",
            "observed_ns",
            "total_begin_ns",
            "observed_begin_ns",
            "observed_end_ns",
        ]:
            assert type(row[key]) is int and row[key] >= 0
        assert row["observed_ns"] == row["observed_end_ns"] - row["observed_begin_ns"]
        assert row["observed_begin_ns"] <= row["total_begin_ns"]
        assert row["total_begin_ns"] + row["total_ns"] <= row["observed_end_ns"]
        assert row["observed_ns"] >= row["total_ns"] > 0
        probe = row["thread"]
        if row["probe"] == "off":
            assert probe is None
            continue
        assert set(probe) == {
            "stamps",
            "intervals",
            "whole",
            "usage_before",
            "usage_after",
            "usage_delta",
        }
        stamps = probe["stamps"]
        assert len(stamps) == 6
        for stamp in stamps:
            assert set(stamp) == {"wall_before_ns", "cpu_ns", "wall_after_ns"}
            assert all(type(x) is int and x >= 0 for x in stamp.values())
            assert 0 <= stamp["wall_before_ns"] <= stamp["wall_after_ns"] <= row["total_ns"]

        def interval(start, end):
            assert (
                start["wall_after_ns"] <= end["wall_before_ns"] and start["cpu_ns"] <= end["cpu_ns"]
            )
            cpu = end["cpu_ns"] - start["cpu_ns"]
            lower = end["wall_before_ns"] - start["wall_after_ns"]
            upper = end["wall_after_ns"] - start["wall_before_ns"]
            return {
                "cpu_ns": cpu,
                "wall_lower_ns": lower,
                "wall_upper_ns": upper,
                "residual_lower_ns": lower - cpu,
                "residual_upper_ns": upper - cpu,
            }

        expected = [interval(a, b) for a, b in zip(stamps, stamps[1:], strict=False)]
        assert probe["intervals"] == expected and probe["whole"] == interval(stamps[0], stamps[-1])
        assert probe["whole"]["cpu_ns"] == sum(x["cpu_ns"] for x in expected)
        keys = {"minor_faults", "major_faults", "voluntary_switches", "involuntary_switches"}
        for name in ["usage_before", "usage_after", "usage_delta"]:
            assert set(probe[name]) == keys
            assert all(type(v) is int and v >= 0 for v in probe[name].values())
        assert probe["usage_delta"] == {
            k: probe["usage_after"][k] - probe["usage_before"][k] for k in keys
        }


def design_cells():
    groups = (
        ((0, "aa"), (2, "ab"), (0, "ab"), (2, "aa")),
        ((2, "aa"), (0, "ab"), (2, "ab"), (0, "aa")),
        ((0, "ab"), (2, "aa"), (0, "aa"), (2, "ab")),
        ((2, "ab"), (0, "aa"), (2, "aa"), (0, "ab")),
    )
    return [
        {
            "cell": 4 * repeat + position,
            "cpu": cpu,
            "contrast": contrast,
            "repeat": repeat,
            "order": "ABBABAAB" if repeat in (0, 3) else "BAABABBA",
        }
        for repeat, group in enumerate(groups)
        for position, (cpu, contrast) in enumerate(group)
    ]


def validate_samples(report, cell):
    assert report["schema"] == 3 and report["mode"] == "whole"
    assert report["contrast"] == cell["contrast"] and report["order"] == cell["order"]
    rows = report["samples"]
    assert [(row["sequence"], row["part"]) for row in rows] == [
        (s, p) for s in range(12) for p in range(3)
    ]
    labels = cell["order"][:4] + cell["order"]
    for row in rows:
        sequence = row["sequence"]
        assert type(sequence) is int and type(row["part"]) is int
        assert type(row["warmup"]) is bool and row["warmup"] == (sequence < 4)
        arm = labels[sequence]
        assert row["arm"] == arm
        assert row["probe"] == ("thread" if cell["contrast"] == "ab" and arm == "B" else "off")
        assert all(
            type(row[k]) is int and row[k] >= 0
            for k in TIMINGS + ("records", "descriptors", "encoded_bytes")
        )
        assert row["total_ns"] > 0 and row["records"] == 131 and row["descriptors"] == 8064
        assert type(row["publication_bytes"]) is int and row["publication_bytes"] == 65536
        assert "encoding_workspace_bytes" not in row and "encoding_workspace_len" not in row
        assert (
            sum(
                row[k]
                for k in ("open_ns", "geometry_ns", "batch_capture_ns", "finish_ns", "flush_ns")
            )
            <= row["total_ns"]
        )
        assert row["encoding_ns"] == row["publish_ns"] == row["encoded_bytes"] == 0
    validate_probe(report)
    assert type(report["clock_control_median_ns"]) is int
    assert type(report["clock_control_p99_ns"]) is int
    assert 0 <= report["clock_control_median_ns"] <= report["clock_control_p99_ns"]
    result = {}
    for metric in METRICS:
        medians = [
            statistics.median(row[metric] for row in rows if row["sequence"] == s)
            for s in range(4, 12)
        ]
        ratios = []
        for i in range(0, 8, 2):
            a, b = (
                (medians[i], medians[i + 1])
                if cell["order"][i] == "A"
                else (medians[i + 1], medians[i])
            )
            ratios.append(b / a)
        result[metric] = {
            "sequence_medians": medians,
            "adjacent_ratios": ratios,
            "process_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        }
    return result


def summarize(results, seed=20261017, draws=20000):
    assert [row["cell"] for row in results] == design_cells()
    summaries = {}
    for cpu in (0, 2):
        for contrast in ("aa", "ab"):
            selected = [
                row
                for row in results
                if row["cell"]["cpu"] == cpu and row["cell"]["contrast"] == contrast
            ]
            metrics = {}
            for metric in METRICS:
                ratios = [row["metrics"][metric]["process_ratio"] for row in selected]
                adjacent = [
                    r for row in selected for r in row["metrics"][metric]["adjacent_ratios"]
                ]
                assert len(ratios) == 4 and len(adjacent) == 16
                assert all(
                    type(r) in (int, float) and math.isfinite(r) and r > 0
                    for r in ratios + adjacent
                )
                rng = random.Random(seed)
                samples = sorted(
                    math.exp(sum(math.log(rng.choice(ratios)) for _ in ratios) / len(ratios))
                    for _ in range(draws)
                )
                interval = [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]
                lower = 1 / 1.05 if contrast == "aa" else 0
                guards = {
                    "all_adjacent": all(lower <= r <= 1.05 for r in adjacent),
                    "all_process": all(lower <= r <= 1.05 for r in ratios),
                    "bootstrap": lower <= interval[0] <= interval[1] <= 1.05,
                }
                metrics[metric] = {
                    "geometric_mean": math.exp(statistics.mean(map(math.log, ratios))),
                    "bootstrap_95_ci": interval,
                    "process_ratios": ratios,
                    "adjacent_ratios": adjacent,
                    "guards": guards,
                    "guard_passed": all(guards.values()),
                    "bootstrap_unit": "one complete independently restarted process",
                    "processes": 4,
                    "draws": draws,
                    "seed": seed,
                }
            summaries[f"cpu{cpu}_{contrast}"] = metrics
    aa_passed = all(summaries[f"cpu{cpu}_aa"][m]["guard_passed"] for cpu in (0, 2) for m in METRICS)
    ab_passed = all(summaries[f"cpu{cpu}_ab"][m]["guard_passed"] for cpu in (0, 2) for m in METRICS)
    return {
        "by_stratum": summaries,
        "aa_passed": aa_passed,
        "ab_raw_guards_passed": ab_passed,
        "method_usable": aa_passed and ab_passed,
        "hardware_qualification": False,
        "scope": "Median3-part projection only; no per-part, whole-sequence, process, causal or production overhead bound; old failures unchanged",
    }


def stable(event, named=None):
    out = {key: value for key, value in event.items() if key not in DYNAMIC}
    schema = out.pop("schema")
    assert type(schema) is int and schema in (2, 3, 4)
    if schema in (3, 4):
        if out["stage"] == "local_raw_copy_open":
            assert out.pop("copy_fields") == list(COPY_FIELDS)
        elif out["stage"] == "local_raw_copy_batch":
            if named is None:
                raw = copy.copy(event)
                _decode_copies(raw)
                out["copies"] = raw["copies"]
            else:
                out["copies"] = named
            if schema == 4:
                assert out.pop("copy_rows") == out["descriptors"]
                assert out.pop("copies_encoding") == "uvarint-rle-base64-v1"
    return out


def validate_log(log, expected, canonical, waited, report):
    decoded = _collect(log)
    assert (
        decoded["parts"] == 36 and decoded["batches"] == 4608 and decoded["descriptors"] == 290304
    )
    actual = records(log)
    assert len(expected) == len(canonical) == 393 and len(actual) == 4716
    geometry = {
        (
            trace["header"]["pid"],
            trace["header"]["trace_id"],
            batch["part_id"],
            batch["batch_id"],
        ): batch["copies"]
        for trace in decoded["traces"]
        for batch in trace["batches"]
    }
    expected = [stable(event) for event in expected]
    for index, event in enumerate(actual):
        named = (
            geometry[(event["pid"], event["trace_id"], event["part_id"], event["batch_id"])]
            if event["stage"] == "local_raw_copy_batch"
            else None
        )
        assert stable(event, named) == expected[index % 393], (
            index,
            "stable scalar/geometry mismatch",
        )
        assert event["schema"] == 4
        if named is not None:
            assert event["copies"] == canonical[index % 393]["copies"], (
                index,
                "canonical byte mismatch",
            )
    trace_ids = set()
    for sequence in range(12):
        events = actual[sequence * 393 : (sequence + 1) * 393]
        ids = {(v["pid"], v["trace_id"]) for v in events}
        assert len(ids) == 1
        pid, trace_id = ids.pop()
        assert (
            pid == waited["pid"]
            and type(trace_id) is int
            and trace_id > 0
            and trace_id not in trace_ids
        )
        trace_ids.add(trace_id)
        tids = set()
        for part_id in range(3):
            part = events[part_id * 131 : (part_id + 1) * 131]
            assert part[0]["stage"] == "local_raw_copy_open" and part[-1]["complete"] is True
            drained = part[-2]["drained_ns"]
            assert type(drained) is int and drained > 0
            for batch in part[1:-2]:
                assert type(batch["tid"]) is int and batch["tid"] > 0
                tids.add(batch["tid"])
                assert all(type(batch[k]) is int for k in ("begin_ns", "submitted_ns"))
                assert 0 < batch["begin_ns"] <= batch["submitted_ns"] <= drained
        assert tids == {report["profile_tid"]}
    return decoded


def census(directory, expected):
    assert {entry.name for entry in directory.iterdir()} == set(expected), directory
    for entry in directory.iterdir():
        assert not entry.is_symlink(), entry
        assert entry.stat().st_nlink == 1 and entry.is_file(), entry


def main(root, index_sha):
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized Python is forbidden before admission")
    root = external_path(str(root))
    index = root / "INPUTS.json"
    assert sha(index) == index_sha
    inputs = json.loads(index.read_text())

    def unchanged():
        assert sha(index) == index_sha
        for item in inputs["files"]:
            assert sha(item["path"]) == item["sha256"], item["path"]

    unchanged()
    authorization = json.loads((root / "LAUNCH-AUTHORIZATION.json").read_text())
    assert authorization["input_index_sha256"] == index_sha
    assert authorization["scope"] == "one16cellCPU_INTERLEAVED_only"
    receipt = Path(authorization["review_receipt_path"])
    assert sha(receipt) == authorization["review_receipt_sha256"]
    review = json.loads(receipt.read_text())
    assert review["CPU_launch_approved"] is True and review["cpu_input_index_sha256"] == index_sha
    assert (root / "FORMAL-STARTED").read_text() == index_sha + "\n"
    contract = json.loads((root / "CONTRACT.json").read_text())
    assert contract["cells"] == design_cells()
    assert (contract["warmup_sequences"], contract["measured_sequences"], contract["parts"]) == (
        4,
        8,
        3,
    )
    assert contract["metrics"] == list(METRICS) and contract["upper"] == 1.05
    assert contract["seed"] == 20261017 and contract["bootstrap_draws"] == 20000
    assert {0, 2} <= os.sched_getaffinity(0)
    for cpu, fields in inputs["topology"].items():
        for field, value in fields.items():
            assert (
                Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/{field}").read_text().strip()
                == value
            )
    topology = inputs["topology"]
    assert (topology["0"]["physical_package_id"], topology["0"]["core_id"]) != (
        topology["2"]["physical_package_id"],
        topology["2"]["core_id"],
    )
    frozen = root / "freeze"
    for name in (
        "benches",
        "benches.native_trace_calibration",
        "benches.artifacts",
        "benches.local_restore_shapes",
        "benches.raw_copy_shapes",
    ):
        expected_path = (
            frozen
            / "reader"
            / ("benches/__init__.py" if name == "benches" else name.replace(".", "/") + ".py")
        )
        assert Path(sys.modules[name].__file__).resolve() == expected_path.resolve()
        assert any(
            item["path"] == str(expected_path) and item["sha256"] == sha(expected_path)
            for item in inputs["files"]
        )
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RUST_LOG": "info",
    }
    source = frozen / "INPUT-native.log"
    expected, canonical = records(source), records(frozen / "ACCEPTED-schema4-native.log")
    assert all(event["schema"] == 4 for event in canonical)
    baseline = _collect(source)
    assert (baseline["parts"], baseline["batches"], baseline["descriptors"]) == (3, 384, 24192)
    output = root / "cpu"
    output.mkdir()
    write(
        output / "MACHINE.json",
        {
            "allowed_cpus": sorted(os.sched_getaffinity(0)),
            "topology": topology,
            "platform": platform.platform(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cpuinfo": Path("/proc/cpuinfo").read_text(),
            "controller_pid": os.getpid(),
            "inputs_sha256": index_sha,
        },
    )
    results = []
    for cell_spec in contract["cells"]:
        assert {entry.name for entry in output.iterdir()} == {
            "MACHINE.json",
            *(f"cell-{i:02d}" for i in range(cell_spec["cell"])),
        }
        unchanged()
        cell = output / f"cell-{cell_spec['cell']:02d}"
        cell.mkdir()
        child_env = env | {
            "ORBITKV_TRACE_PROFILE_INPUT": str(source),
            "ORBITKV_TRACE_PROFILE_OUTPUT": str(cell / "SAMPLES.json"),
            "ORBITKV_TRACE_PROFILE_MODE": "whole",
            "ORBITKV_TRACE_PROFILE_PROBE": "off",
            "ORBITKV_TRACE_PROFILE_CONTRAST": cell_spec["contrast"],
            "ORBITKV_TRACE_PROFILE_ORDER": cell_spec["order"],
        }
        waited = run(
            [
                str(frozen / "native-trace-tests"),
                "--exact",
                NAME,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
            child_env,
            cell,
            "PROFILE",
            cpu=cell_spec["cpu"],
        )
        report = json.loads((cell / "SAMPLES.json").read_text())
        assert report["input"] == str(source)
        metrics = validate_samples(report, cell_spec)
        decoded = validate_log(cell / "PROFILE.stderr", expected, canonical, waited, report)
        write(cell / "GEOMETRY.json", decoded)
        compiler_wait = run(
            [
                str(frozen / "native-copy-layout-validator"),
                "--validate-local-geometry",
                str(cell / "GEOMETRY.json"),
                "--output",
                str(cell / "COMPILED.json"),
            ],
            env,
            cell,
            "COMPILER",
        )
        compiled = json.loads((cell / "COMPILED.json").read_text())
        assert compiled["schema"] == 1 and compiled["production_compiler_verified"] is True
        assert all(
            type(compiled[k]) is int and compiled[k] == decoded[k]
            for k in ("executors", "batches", "descriptors", "dma_ranges")
        )
        result = {
            "cell": cell_spec,
            "metrics": metrics,
            "profile_wait": waited,
            "compiler_wait": compiler_wait,
            "complete_geometry_and_canonical_bytes": True,
            "native_qualification": False,
            "samples_sha256": sha(cell / "SAMPLES.json"),
            "log_sha256": sha(cell / "PROFILE.stderr"),
            "geometry_sha256": sha(cell / "GEOMETRY.json"),
            "compiler_receipt_sha256": sha(cell / "COMPILED.json"),
        }
        write(cell / "RESULT.json", result)
        census(cell, CELL_FILES)
        results.append(result)
        unchanged()
        print(json.dumps({"cell": cell_spec, "metrics": metrics}), flush=True)
    result = {
        "schema": 1,
        "status": "VALID_CPU_INTERLEAVED_COMPLETE",
        "cells": 16,
        "measured_parts": 384,
        "summary": summarize(results),
        "inputs_sha256": index_sha,
        "all_normal_waits": True,
        "inputs_unchanged": True,
        "prior_failures_unchanged": True,
        "next_campaign_authorized": False,
    }
    unchanged()
    assert {entry.name for entry in output.iterdir()} == {
        "MACHINE.json",
        *(f"cell-{i:02d}" for i in range(16)),
    }
    for i in range(16):
        directory = output / f"cell-{i:02d}"
        assert directory.is_dir() and not directory.is_symlink()
        census(directory, CELL_FILES)
    assert not (output / "MACHINE.json").is_symlink()
    write(output / "RESULT.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("input_index_sha256")
    args = parser.parse_args()
    main(args.root, args.input_index_sha256)
