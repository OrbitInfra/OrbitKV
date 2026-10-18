"""Source-only timing and process observations for live-store experiments."""

from __future__ import annotations

import bisect
import json
import os
import statistics
import time
from pathlib import Path

MEASUREMENT_CONTRACT = "s2.10-performance-v2"


def _summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "samples": len(ordered),
        "p50": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _process_sample(pids):
    sample_started = time.monotonic_ns()
    cpu_started = time.thread_time_ns()
    result = {
        "thread_context_switches": {},
        "thread_affinity": {},
        "thread_cpu_ticks": {},
        "thread_schedstat": {},
        "context_switch_scope": "snapshot sum of live threads; exited threads excluded",
        "cpu_ticks": 0,
        "rss_kib": 0,
        "hwm_kib": 0,
        "voluntary_context_switches": 0,
        "nonvoluntary_context_switches": 0,
        "cpu_affinity": [],
        "memory_affinity": [],
        "processes": 0,
    }
    cpu_affinity = set()
    memory_affinity = set()
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, IndexError):
            continue
        values = {}
        for line in status:
            key, _, value = line.partition(":")
            if key in {
                "VmRSS",
                "VmHWM",
                "voluntary_ctxt_switches",
                "nonvoluntary_ctxt_switches",
            }:
                values[key] = int(value.split()[0])
            elif key == "Cpus_allowed_list":
                cpu_affinity.add(value.strip())
            elif key == "Mems_allowed_list":
                memory_affinity.add(value.strip())
        result["cpu_ticks"] += int(stat[11]) + int(stat[12])
        result["rss_kib"] += values.get("VmRSS", 0)
        result["hwm_kib"] += values.get("VmHWM", 0)
        for thread in Path(f"/proc/{pid}/task").iterdir():
            try:
                fields = dict(
                    line.split(":", 1) for line in (thread / "status").read_text().splitlines()
                )
                thread_stat = (thread / "stat").read_text().rsplit(") ", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            try:
                schedstat = [int(value) for value in (thread / "schedstat").read_text().split()]
            except (FileNotFoundError, ProcessLookupError, ValueError):
                schedstat = []
            identity = f"{pid}:{thread.name}:{thread_stat[19]}"
            counters = [
                int(fields[name])
                for name in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
            ]
            result["thread_context_switches"][identity] = counters
            result["thread_cpu_ticks"][identity] = [
                int(thread_stat[11]),
                int(thread_stat[12]),
            ]
            if len(schedstat) >= 3:
                result["thread_schedstat"][identity] = {
                    "runtime_ns": schedstat[0],
                    "runqueue_wait_ns": schedstat[1],
                    "timeslices": schedstat[2],
                }
            result["thread_affinity"][identity] = {
                "cpus_allowed": fields["Cpus_allowed_list"].strip(),
                "mems_allowed": fields["Mems_allowed_list"].strip(),
            }
            result["voluntary_context_switches"] += counters[0]
            result["nonvoluntary_context_switches"] += counters[1]
        result["processes"] += 1
    result["cpu_affinity"] = sorted(cpu_affinity)
    result["memory_affinity"] = sorted(memory_affinity)
    result["sample_cpu_ns"] = time.thread_time_ns() - cpu_started
    result["sample_elapsed_ns"] = time.monotonic_ns() - sample_started
    return result


def _clock_domain(pids):
    pids = [os.getpid(), *pids]
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    clocks = {}
    for pid in pids:
        clocks[str(pid)] = {
            "time_namespace": os.readlink(f"/proc/{pid}/ns/time"),
            "time_offsets": Path(f"/proc/{pid}/timens_offsets").read_text(),
        }
    assert len({(row["time_namespace"], row["time_offsets"]) for row in clocks.values()}) == 1, (
        clocks
    )
    return {"clock": "CLOCK_MONOTONIC", "boot_id": boot_id, "processes": clocks}


def _installation_sample(targets, installed, harness_completed_ns):
    owners = []
    for node, target in targets.items():
        row = installed[node]
        install_ns = row["installed_mono_ns"]
        assert row["applied_sequence"] == target["sequence"], (target, row)
        assert row["view_id"] == target["view_id"], (target, row)
        assert target["published_mono_ns"] > 0
        assert target["published_mono_ns"] <= install_ns <= harness_completed_ns
        owners.append(
            {
                **target,
                "installed_sequence": row["applied_sequence"],
                "installed_view_id": row["view_id"],
                "installed_mono_ns": install_ns,
                "publication_to_install_ms": (install_ns - target["published_mono_ns"]) / 1_000_000,
                "save_start_to_install_ms": (install_ns - target["save_start_mono_ns"]) / 1_000_000,
                "install_to_harness_ms": (harness_completed_ns - install_ns) / 1_000_000,
            }
        )
    return {
        "owners": owners,
        "publication_to_install_ms": max(row["publication_to_install_ms"] for row in owners),
        "save_start_to_install_ms": max(row["save_start_to_install_ms"] for row in owners),
        "install_to_harness_ms": max(row["install_to_harness_ms"] for row in owners),
    }


def _context_switch_delta(before, after):
    initial = before["thread_context_switches"]
    final = after["thread_context_switches"]
    surviving = initial.keys() & final.keys()
    return {
        "scope": "surviving thread identities only; exited threads are not a measured zero",
        "surviving_threads": len(surviving),
        "exited_threads": len(initial.keys() - final.keys()),
        "new_threads": len(final.keys() - initial.keys()),
        "voluntary": sum(final[key][0] - initial[key][0] for key in surviving),
        "nonvoluntary": sum(final[key][1] - initial[key][1] for key in surviving),
    }


def _pressure_exposure(
    output,
    condition,
    expected_rounds,
    pressure_cadence_ms,
    foreground_cadence_ms,
    shift,
    observer_sample_ms=None,
):
    def rows(name):
        with (output / name).open() as file:
            for line in file:
                yield json.loads(line)

    all_foreground = list(rows("samples.jsonl"))
    foreground = [row for row in all_foreground if row["measured"]]
    bursts = []
    actual = []
    scheduling_lag = []
    publication_lag = []
    for row in rows("pressure-samples.jsonl"):
        actual.append(row["actual_seconds"])
        scheduling_lag.append((row["actual_seconds"] - row["scheduled_seconds"]) * 1000)
        if condition == "pressure":
            assert row["inventory"]["sequence"] - row["sequence_before"] == shift * 2
            first, last = row["first_publication_mono_ns"], row["last_publication_mono_ns"]
            assert 0 < first <= last
            assert row["scheduled_mono_ns_lower"] <= row["scheduled_mono_ns_upper"] <= first
            publication_lag.append((last - row["scheduled_mono_ns_lower"]) / 1e6)
            bursts.append((first, last))
        else:
            assert row["inventory"]["sequence"] == row["sequence_before"]
    assert len(actual) == expected_rounds
    interval_ms = [
        (later - earlier) * 1000 for earlier, later in zip(actual, actual[1:], strict=False)
    ]
    installed = {}
    poll_ms, poll_gaps, sampler_cpu = [], [], 0
    observer_scheduling_deviation_ms = []
    observer_response_bytes = []
    scheduled_poll_starts = []
    observer_rows = []
    previous_poll = None
    for row in rows("observer-samples.jsonl"):
        observer_rows.append(row)
        owner = row["owner"]
        key = (owner["owner"], owner["view_id"], owner["applied_sequence"])
        timestamp = owner["installed_mono_ns"]
        assert installed.setdefault(key, timestamp) == timestamp
        poll_ms.append((row["poll_end_mono_ns"] - row["poll_start_mono_ns"]) / 1e6)
        sampler_cpu += row["sampler_cpu_ns"]
        if "scheduled_poll_start_mono_ns" in row:
            scheduled_poll_starts.append(row["scheduled_poll_start_mono_ns"])
            observer_scheduling_deviation_ms.append(row["scheduling_deviation_ns"] / 1e6)
            observer_response_bytes.append(row["response_bytes"])
        if previous_poll is not None:
            poll_gaps.append((row["poll_start_mono_ns"] - previous_poll) / 1e6)
        previous_poll = row["poll_start_mono_ns"]
    install_times = sorted(installed.values())
    starts = [first for first, _ in bursts]
    ends = [last for _, last in bursts]
    phases = set()
    windows_without_install = []
    direct_overlap = {"save": 0, "query": 0}
    offered_overlap = {"save": 0, "query": 0}
    for row in foreground:
        window_start = row["round_start_mono_ns"]
        window_end = window_start + foreground_cadence_ms * 1_000_000
        pos = bisect.bisect_left(install_times, window_start)
        if pos == len(install_times) or install_times[pos] >= window_end:
            windows_without_install.append(row["round"])
        if starts:
            phase_index = bisect.bisect_right(starts, row["save_start_mono_ns"]) - 1
            assert phase_index >= 0
            phase = (row["save_start_mono_ns"] - starts[phase_index]) / (pressure_cadence_ms * 1e6)
            phases.add(min(3, int(phase * 4)))
        for operation in direct_overlap:
            start, end = row[f"{operation}_start_mono_ns"], row[f"{operation}_end_mono_ns"]
            pos = bisect.bisect_left(install_times, start)
            if pos < len(install_times) and install_times[pos] <= end:
                direct_overlap[operation] += 1
            pos = bisect.bisect_left(ends, start)
            if pos < len(starts) and starts[pos] <= end:
                offered_overlap[operation] += 1
    pressure_result = json.loads((output / "pressure-source-result.json").read_text())
    rate_ratio = len(actual) * pressure_cadence_ms / 1000 / pressure_result["elapsed_seconds"]
    publication_gaps = [
        (later[0] - earlier[0]) / 1e6 for earlier, later in zip(bursts, bursts[1:], strict=False)
    ]
    foreground_scheduling_deviation_ms = [
        row["foreground_start_deviation_ns"] / 1e6
        for row in all_foreground
        if "foreground_start_deviation_ns" in row
    ]
    scheduled_poll_gaps = [
        (later - earlier) / 1e6
        for earlier, later in zip(scheduled_poll_starts, scheduled_poll_starts[1:], strict=False)
    ]
    phase = None
    if scheduled_poll_starts and all_foreground:
        scheduled_phase_ms = (
            all_foreground[0]["scheduled_foreground_start_mono_ns"] - scheduled_poll_starts[0]
        ) / 1e6
        actual_phase_ms = (
            all_foreground[0]["round_start_mono_ns"] - observer_rows[0]["poll_start_mono_ns"]
        ) / 1e6
        phase = {
            "scheduled_ms": scheduled_phase_ms,
            "actual_ms": actual_phase_ms,
            "deviation_ms": actual_phase_ms - scheduled_phase_ms,
        }
    report = {
        "condition": condition,
        "publication_lateness_upper_ms": _summary(publication_lag),
        "publication_gap_ms": _summary(publication_gaps),
        "foreground_samples": len(foreground),
        "source_rounds": len(actual),
        "source_rate_ratio": rate_ratio,
        "source_scheduling_lag_ms": _summary(scheduling_lag),
        "source_interval_ms": _summary(interval_ms),
        "minimum_source_interval_ms": min(interval_ms),
        "observer_poll_ms": _summary(poll_ms),
        "observer_poll_gap_ms": _summary(poll_gaps),
        "observer_scheduled_gap_ms": _summary(scheduled_poll_gaps),
        "observer_scheduling_deviation_ms": _summary(observer_scheduling_deviation_ms),
        "observer_response_bytes": _summary(observer_response_bytes),
        "observer_poll_count": len(observer_rows),
        "observer_sampler_cpu_ms": sampler_cpu / 1e6,
        "foreground_scheduling_deviation_ms": _summary(foreground_scheduling_deviation_ms),
        "observer_foreground_phase": phase,
        "observed_unique_installs": len(install_times),
        "offered_phase_quarters": sorted(phases),
        "windows_without_observed_install": windows_without_install,
        "observed_install_overlap_counts": direct_overlap,
        "source_publication_overlap_counts": offered_overlap,
        "unobserved_installs": "unknown; polling can miss superseded latest watermarks",
    }
    (output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n")
    assert 0.95 <= rate_ratio <= 1.05, report
    assert max(interval_ms) <= 100 and min(interval_ms) >= pressure_cadence_ms / 2, report
    assert max(scheduling_lag) <= 100, report
    assert max(poll_gaps) <= 100, report
    if scheduled_poll_gaps:
        assert observer_sample_ms is not None, report
        assert all(gap == observer_sample_ms for gap in scheduled_poll_gaps), report
        assert max(observer_scheduling_deviation_ms) <= 100, report
        assert max(foreground_scheduling_deviation_ms) <= 100, report
        assert abs(phase["deviation_ms"]) <= 100, report
    if condition == "pressure":
        assert max(publication_lag) <= 100, report
        assert max(publication_gaps) <= 100 and min(publication_gaps) >= pressure_cadence_ms / 2, (
            report
        )
        assert not windows_without_install, report
        assert phases == {0, 1, 2, 3}, report
        assert starts[0] <= foreground[0]["save_start_mono_ns"], report
        assert ends[-1] >= foreground[-1]["gpu_completed_mono_ns"], report
    else:
        assert len(install_times) == 1, report
    return report


def _payload_header(generation, block):
    if not 0 <= generation < 2**64 or not 0 <= block < 2**64:
        raise ValueError("generation and block index must fit unsigned 64-bit fields")
    return generation.to_bytes(8, "little") + block.to_bytes(8, "little")


def _payload(torch, pages, block_bytes, generation):
    if pages <= 0 or block_bytes < 16:
        raise ValueError("payload requires positive pages and at least 16 bytes per block")
    headers = b"".join(_payload_header(generation, block) for block in range(pages))
    values = torch.arange(pages * block_bytes, device="cuda", dtype=torch.int64).reshape(
        pages, block_bytes
    )
    offsets = torch.arange(pages, device="cuda", dtype=torch.int64).unsqueeze(1) * 17
    payload = ((values + offsets + (generation % 251) * 31) % 251).to(torch.uint8)
    payload[:, :16] = torch.tensor(list(headers), device="cuda", dtype=torch.uint8).reshape(
        pages, 16
    )
    return payload.flatten()
