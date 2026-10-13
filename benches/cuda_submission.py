"""Attribute native fixture wall intervals without inventing Rust/GPU stage costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCOPE = (
    "Diagnostic native fixture DRIVER_API wall intervals, including callback overhead. "
    "Remaining host wall time is unattributed, not proven packing or scheduling cost. "
    "This is not GPU execution, model forward admission, inference gain or an overhead qualification."
)


def _integer(row: dict, key: str, minimum: int = 0) -> int:
    value = row.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {key}")
    return value


def _rows(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input must be a regular file")
    with path.open() as source:
        rows = [json.loads(line) for line in source]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Missing or malformed records")
    return rows


def _union(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for left, right in sorted(intervals):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def analyze(trace_path: Path, fixture_path: Path) -> dict:
    trace = _rows(trace_path)
    header, footer = trace[0], trace[-1]
    if (
        header.get("stage") != "cuda_driver_open"
        or header.get("schema") != 1
        or header.get("clock") != "CLOCK_MONOTONIC"
        or footer.get("stage") != "cuda_driver_complete"
        or footer.get("complete") is not True
    ):
        raise ValueError("Missing complete driver capture")
    pid = _integer(header, "pid", 1)
    capacity = _integer(header, "capacity", 1)
    calls = _integer(footer, "calls", 1)
    if footer.get("pid") != pid or calls != len(trace) - 2 or calls > capacity:
        raise ValueError("Driver PID/count/capacity mismatch")
    by_thread = defaultdict(list)
    for index, row in enumerate(trace[1:-1]):
        if row.get("stage") != "cuda_driver_api" or _integer(row, "id") != index:
            raise ValueError("Driver call sequence mismatch")
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith("cu"):
            raise ValueError("Invalid driver function name")
        begin, end = _integer(row, "begin_ns", 1), _integer(row, "end_ns", 1)
        _integer(row, "context")
        _integer(row, "correlation")
        _integer(row, "result")
        if end < begin:
            raise ValueError("Reversed driver clocks")
        by_thread[_integer(row, "tid", 1)].append(row)
    fixture = _rows(fixture_path)
    groups = []
    all_windows = defaultdict(list)
    directions = set()
    cases = set()
    identities = set()
    for row in fixture:
        if row.get("schema") != 1 or row.get("native_pid") != pid:
            raise ValueError("Fixture PID/schema mismatch")
        oracle = row.get("byte_source_guard_oracle", "")
        if not isinstance(oracle, str) or not oracle.startswith("passed before"):
            raise ValueError("Missing byte/source guard control")
        direction = row.get("direction")
        if direction not in ("h2d", "d2h"):
            raise ValueError("Invalid transfer direction")
        case = row.get("case")
        uuid = row.get("cuda_device_uuid")
        if (
            not isinstance(case, str)
            or not case
            or not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
        ):
            raise ValueError("Missing case/device identity")
        if (case, direction) in directions:
            raise ValueError("Duplicate fixture direction")
        directions.add((case, direction))
        cases.add(case)
        identities.add(uuid)
        samples = _integer(row, "samples", 1)
        _integer(row, "bytes", 1)
        _integer(row, "descriptors", 1)
        ranges = _integer(row, "dma_ranges", 1)
        for backend in ("kernel", "dma"):
            for mode in ("event_off", "event_on"):
                clocks = row.get(backend, {}).get(mode, {})
                windows = clocks.get("submission_windows")
                if not isinstance(windows, list) or len(windows) != samples:
                    raise ValueError("Missing complete submission windows")
                for field in ("call_to_return_ms", "cpu_submission_ms", "host_wait_ms"):
                    values = clocks.get(field)
                    if (
                        not isinstance(values, list)
                        or len(values) != samples
                        or any(
                            type(v) not in (float, int) or not math.isfinite(v) or v < 0
                            for v in values
                        )
                    ):
                        raise ValueError("Missing fixture timing arrays")
                attributed = []
                for window in windows:
                    if not isinstance(window, dict):
                        raise ValueError("Malformed submission window")
                    left = _integer(window, "begin_ns", 1)
                    right = _integer(window, "submitted_ns", 1)
                    drained = _integer(window, "drained_ns", 1)
                    tid = _integer(window, "tid", 1)
                    if not left < right <= drained:
                        raise ValueError("Reversed submission/drain clocks")
                    all_windows[tid].append((left, drained))
                    matching = []
                    for api in by_thread[tid]:
                        begin, end = api["begin_ns"], api["end_ns"]
                        if begin < right and end > left:
                            if begin < left or end > right:
                                raise ValueError("Driver call crosses submission boundary")
                            matching.append(api)
                    if not matching:
                        raise ValueError("No driver calls in submission window")
                    copies = [api for api in matching if api["name"].startswith("cuMemcpy")]
                    launches = [api for api in matching if api["name"].startswith("cuLaunchKernel")]
                    if backend == "dma" and (len(copies) != ranges or launches):
                        raise ValueError("DMA driver count differs from compiled geometry")
                    if backend == "kernel" and (len(copies) != 1 or len(launches) != 1):
                        raise ValueError("Missing kernel descriptor upload/launch")
                    if any(api["result"] for api in copies + launches):
                        raise ValueError("Copy/launch driver failure")
                    union = _union([(api["begin_ns"], api["end_ns"]) for api in matching])
                    attributed.append(
                        {
                            "begin_ns": left,
                            "submitted_ns": right,
                            "drained_ns": drained,
                            "tid": tid,
                            "submission_wall_ns": right - left,
                            "driver_api_union_wall_ns": union,
                            "unattributed_host_wall_ns": right - left - union,
                            "driver_calls": dict(Counter(api["name"] for api in matching)),
                        }
                    )
                groups.append(
                    {
                        "case": case,
                        "direction": direction,
                        "backend": backend,
                        "mode": mode,
                        "samples": attributed,
                    }
                )
    if len(identities) != 1 or any(
        (case, direction) not in directions for case in cases for direction in ("h2d", "d2h")
    ):
        raise ValueError("Incomplete directional/device controls")
    for windows in all_windows.values():
        ordered = sorted(windows)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("Overlapping fixture operations")
    return {
        "schema": 1,
        "scope": SCOPE,
        "pid": pid,
        "cuda_device_uuid": identities.pop(),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "captured_driver_calls": calls,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.fixture), indent=2))


if __name__ == "__main__":
    main()
