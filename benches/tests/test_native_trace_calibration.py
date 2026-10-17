"""Protect interleaved labels, independent units and real direct-child waits."""

import copy
import json
import math
import os
import subprocess
import sys

import pytest

from benches import native_trace_calibration as calibration


def report(contrast="ab", order="ABBABAAB"):
    rows = []
    for sequence, arm in enumerate(order[:4] + order):
        for part in range(3):
            begin = len(rows) * 2000
            row = dict(
                sequence=sequence,
                part=part,
                warmup=sequence < 4,
                arm=arm,
                probe="thread" if contrast == "ab" and arm == "B" else "off",
                **dict.fromkeys(calibration.TIMINGS, 0),
                total_begin_ns=begin + 10,
                observed_begin_ns=begin,
                observed_end_ns=begin + 1020,
                records=131,
                descriptors=8064,
                publication_bytes=65536,
                encoded_bytes=0,
                thread=None,
            )
            row.update(total_ns=1000, observed_ns=1020)
            if row["probe"] == "thread":
                stamps = [
                    {
                        "wall_before_ns": 10 + i * 150,
                        "wall_after_ns": 15 + i * 150,
                        "cpu_ns": 100 + i * 150,
                    }
                    for i in range(6)
                ]

                def interval(a, b):
                    cpu = b["cpu_ns"] - a["cpu_ns"]
                    lower = b["wall_before_ns"] - a["wall_after_ns"]
                    upper = b["wall_after_ns"] - a["wall_before_ns"]
                    return {
                        "cpu_ns": cpu,
                        "wall_lower_ns": lower,
                        "wall_upper_ns": upper,
                        "residual_lower_ns": lower - cpu,
                        "residual_upper_ns": upper - cpu,
                    }

                before = dict.fromkeys(
                    ("minor_faults", "major_faults", "voluntary_switches", "involuntary_switches"),
                    0,
                )
                after = dict.fromkeys(before, 2)
                row["thread"] = {
                    "stamps": stamps,
                    "intervals": [interval(a, b) for a, b in zip(stamps, stamps[1:], strict=False)],
                    "whole": interval(stamps[0], stamps[-1]),
                    "usage_before": before,
                    "usage_after": after,
                    "usage_delta": after.copy(),
                }
            rows.append(row)
    return {
        "schema": 3,
        "mode": "whole",
        "probe": "per-sequence",
        "contrast": contrast,
        "order": order,
        "profile_tid": 23,
        "thread_cpu_resolution_ns": 1 if contrast == "ab" else None,
        "clock_control_median_ns": 10,
        "clock_control_p99_ns": 20,
        "samples": rows,
    }


@pytest.mark.parametrize("contrast", ["aa", "ab"])
@pytest.mark.parametrize("order", ["ABBABAAB", "BAABABBA"])
def test_complete_sequence_labels_restart_after_warmup_and_keep_signed_residuals(contrast, order):
    value = report(contrast, order)
    metrics = calibration.validate_samples(value, {"contrast": contrast, "order": order})
    assert all(metrics[m]["process_ratio"] == 1 for m in calibration.METRICS)
    assert value["samples"][12]["arm"] == order[0]
    if contrast == "ab":
        assert any(
            row["thread"]["intervals"][0]["residual_lower_ns"] < 0
            for row in value["samples"]
            if row["thread"]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "arm",
        "probe",
        "part",
        "warmup",
        "envelope",
        "stamp",
        "cpu_regression",
        "residual",
        "counter",
        "cap",
        "duration",
    ],
)
def test_report_corruption_cannot_be_accepted(mutation):
    value = report()
    row = next(row for row in value["samples"] if row["probe"] == "thread")
    if mutation in {"arm", "probe"}:
        row[mutation] = "A" if mutation == "arm" else "off"
    elif mutation == "part":
        row["part"] = 2
    elif mutation == "warmup":
        row["warmup"] = not row["warmup"]
    elif mutation == "envelope":
        row["observed_ns"] -= 1
    elif mutation == "stamp":
        row["thread"]["stamps"].pop()
    elif mutation == "cpu_regression":
        row["thread"]["stamps"][1]["cpu_ns"] = 0
    elif mutation == "residual":
        row["thread"]["intervals"][0]["residual_lower_ns"] = 0
    elif mutation == "counter":
        row["thread"]["usage_delta"]["minor_faults"] = -1
    elif mutation == "cap":
        row["publication_bytes"] = 65537
    elif mutation == "duration":
        row["total_ns"] = float("nan")
    with pytest.raises((AssertionError, IndexError)):
        calibration.validate_samples(value, {"contrast": "ab", "order": "ABBABAAB"})


def cohort():
    return [
        {
            "cell": cell,
            "metrics": calibration.validate_samples(report(cell["contrast"], cell["order"]), cell),
        }
        for cell in calibration.design_cells()
    ]


def test_strict_guards_require_both_cpus_metrics_and_all_aa_controls():
    results = cohort()
    summary = calibration.summarize(results)
    assert summary["method_usable"]
    assert all(m["processes"] == 4 for s in summary["by_stratum"].values() for m in s.values())
    # A single slow adjacent contrast cannot be averaged away by three fast ones.
    cell = next(c for c in results if c["cell"]["cpu"] == 0 and c["cell"]["contrast"] == "aa")
    cell["metrics"]["observed_ns"]["adjacent_ratios"] = [1.06, 1 / 1.06, 1, 1]
    summary = calibration.summarize(results)
    assert summary["ab_raw_guards_passed"] and not summary["aa_passed"]
    assert not summary["method_usable"]
    assert not summary["by_stratum"]["cpu0_aa"]["observed_ns"]["guards"]["all_adjacent"]


@pytest.mark.parametrize(
    "contrast,ratio,passed",
    [("aa", 0.94, False), ("aa", 1.06, False), ("ab", 0.94, True), ("ab", 1.06, False)],
)
def test_symmetric_aa_and_one_sided_ab_guards(contrast, ratio, passed):
    results = cohort()
    for cell in results:
        if cell["cell"]["contrast"] == contrast and cell["cell"]["cpu"] == 2:
            for metric in calibration.METRICS:
                cell["metrics"][metric]["process_ratio"] = ratio
                cell["metrics"][metric]["adjacent_ratios"] = [ratio] * 4
    summary = calibration.summarize(results)
    assert summary["by_stratum"][f"cpu2_{contrast}"]["total_ns"]["guard_passed"] is passed


def test_sample_aggregation_uses_median_parts_not_a_sum_or_pooled_observations():
    value = report("aa")
    for row in value["samples"]:
        ns = (100, 500, 900)[row["part"]] if row["arm"] == "A" else (300, 350, 1000)[row["part"]]
        row["total_ns"] = ns
    metrics = calibration.validate_samples(value, {"contrast": "aa", "order": "ABBABAAB"})
    assert math.isclose(metrics["total_ns"]["process_ratio"], 0.7)
    assert metrics["observed_ns"]["process_ratio"] == 1
    bad = cohort()[:-1]
    with pytest.raises(AssertionError):
        calibration.summarize(bad)


@pytest.mark.parametrize(
    "code,ignore_term,timeout,normal",
    [
        (0, False, 5, True),
        (7, False, 5, False),
        (None, False, 0.2, False),
        (None, True, 0.2, False),
    ],
)
def test_direct_child_actual_wait_and_bounded_cleanup(tmp_path, code, ignore_term, timeout, normal):
    script = "import signal,time; " + (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else ""
    )
    script += "time.sleep(.05); " + (
        "time.sleep(20)" if code is None else f"raise SystemExit({code})"
    )
    cpu = min(os.sched_getaffinity(0))
    # Other benchmark tests may import BLAS and create threads. The real
    # calibration controller deliberately rejects preexec in that process.
    controller = """
import json, os, sys
from pathlib import Path
from benches.native_trace_calibration import run
script, destination, timeout, cpu, normal = json.loads(sys.argv[1])
try:
    run([sys.executable, "-c", script], {"PATH": "/usr/bin:/bin"},
        Path(destination), "CHILD", timeout=timeout, cpu=cpu)
    assert normal
except RuntimeError as error:
    assert not normal and "Invalid child" in str(error)
receipt = json.loads((Path(destination) / "CHILD-WAIT.json").read_text())
try:
    os.waitpid(receipt["pid"], os.WNOHANG)
except ChildProcessError:
    pass
else:
    raise AssertionError("direct child was not already reaped")
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            controller,
            json.dumps([script, str(tmp_path), timeout, cpu, normal]),
        ],
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    receipt = json.loads((tmp_path / "CHILD-WAIT.json").read_text())
    assert receipt["actual_wait4_observed"] and receipt["identity_ended"]
    assert receipt["live_identity"]["affinity"] == [cpu]
    assert receipt["forced_cleanup"] is (code is None)
    assert (receipt["status"] == "NORMAL") is normal


def test_stable_geometry_preserves_all_nonclock_fields_and_exact_census(tmp_path):
    event = {
        "schema": 2,
        "stage": "local_raw_copy_batch",
        "copies": [{"size": 4096}],
        "pid": 1,
        "trace_id": 2,
        "begin_ns": 3,
        "descriptors": 1,
    }
    changed = copy.deepcopy(event)
    changed.update(pid=100, trace_id=200, begin_ns=300)
    assert calibration.stable(event) == calibration.stable(changed)
    changed["copies"][0]["size"] = 8192
    assert calibration.stable(event) != calibration.stable(changed)
    path = tmp_path / "a"
    path.write_text("ok")
    calibration.census(tmp_path, {"a"})
    with pytest.raises(AssertionError):
        calibration.census(tmp_path, {"missing"})
    link = tmp_path / "b"
    link.symlink_to(path)
    with pytest.raises(AssertionError):
        calibration.census(tmp_path, {"a", "b"})
