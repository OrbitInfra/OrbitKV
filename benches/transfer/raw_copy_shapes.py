"""Read finalized, bounded Manager raw-copy diagnostics after normal shutdown."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benches.common.metrics import percentile

SIZE_LIMITS = (256, 4096, 16384, 65536, 262144, 1048576, 4194304)


def _integer(event: dict, field: str, minimum: int = 0) -> int:
    value = event.get(field)
    if type(value) is not int or not minimum <= value <= 2**64 - 1:
        raise ValueError(f"Invalid {field} in Manager timeline")
    return value


def _validate_shape(
    event: dict,
    *,
    timing_fields: tuple[str, ...] = (
        "device",
        "cpu_submit_ns",
        "host_drain_ns",
        "diagnostic_event",
        "monotonic_ns",
    ),
) -> None:
    if event.get("backend") not in {"direct", "kernel"} or event.get("direction") not in {
        "h2d",
        "d2h",
    }:
        raise ValueError("Unknown raw-copy backend/direction")
    if event.get("success") is not True:
        raise ValueError("Unsuccessful raw-copy submission; retain the failure evidence")
    for field in timing_fields:
        _integer(event, field)
    descriptors = _integer(event, "descriptors", 1)
    ranges = _integer(event, "dma_ranges", 1)
    size = _integer(event, "bytes", 1)
    zero = _integer(event, "zero_descriptors")
    smallest = _integer(event, "min_bytes")
    largest = _integer(event, "max_bytes", 1)
    histogram = event.get("size_histogram")
    if (
        not isinstance(histogram, list)
        or len(histogram) != len(SIZE_LIMITS) + 1
        or any(type(count) is not int or not 0 <= count <= descriptors for count in histogram)
        or sum(histogram) != descriptors
        or not ranges <= descriptors
        or not zero < descriptors
        or zero > histogram[0]
        or (smallest == 0) != (zero > 0)
        or not smallest <= largest <= size
        or size > descriptors * largest
        or size < (descriptors - zero) * smallest
    ):
        raise ValueError("Inconsistent raw-copy shape counters")
    remaining = histogram.copy()
    remaining[0] -= zero
    fixed_bytes = 0
    for extreme in {smallest, largest} - {0}:
        bucket = sum(extreme > limit for limit in SIZE_LIMITS)
        if remaining[bucket] == 0:
            raise ValueError("Raw-copy extrema disagree with the fragment histogram")
        remaining[bucket] -= 1
        fixed_bytes += extreme
    lower = upper = fixed_bytes
    for bucket, count in enumerate(remaining):
        low = max(smallest, 1 if bucket == 0 else SIZE_LIMITS[bucket - 1] + 1)
        high = min(largest, SIZE_LIMITS[bucket] if bucket < len(SIZE_LIMITS) else largest)
        if count and low > high:
            raise ValueError("Raw-copy histogram lies outside the observed extrema")
        lower += count * low
        upper += count * high
    if not lower <= size <= upper:
        raise ValueError("Raw-copy bytes disagree with the fragment histogram and extrema")


def collect(log_path: Path) -> dict:
    counts = Counter()
    complete = {}
    indices = defaultdict(set)
    copies = []
    with log_path.open() as source:
        for line in source:
            _, marker, payload = line.partition("cache_timeline ")
            if not marker:
                continue
            event = json.loads(payload)
            if not isinstance(event, dict) or not isinstance(event.get("stage"), str):
                raise ValueError("Malformed Manager timeline record")
            pid = _integer(event, "pid", 1)
            stage = event["stage"]
            if stage == "diagnostic_timeline_limit":
                raise ValueError("Manager timeline overflow/incomplete/late record")
            if stage == "diagnostic_timeline_complete":
                reserved = _integer(event, "reserved")
                if pid in complete or _integer(event, "ready") != reserved:
                    raise ValueError("Duplicate/inconsistent Manager timeline completion")
                complete[pid] = reserved
                continue
            if pid in complete:
                raise ValueError("Manager timeline record after completion")
            counts[pid] += 1
            if "diagnostic_event" in event:
                index = _integer(event, "diagnostic_event")
                if index in indices[pid]:
                    raise ValueError("Duplicate Manager diagnostic slot")
                indices[pid].add(index)
            if stage == "manager_raw_copy_complete":
                _validate_shape(event)
                copies.append(event)
    if not complete or set(counts) - complete.keys():
        raise ValueError("Missing normal Manager timeline flush; live/partial logs are invalid")
    for pid, reserved in complete.items():
        if counts[pid] != reserved or any(index >= reserved for index in indices[pid]):
            raise ValueError("Manager timeline record count disagrees with completed buffer")
    if not copies:
        raise ValueError(
            "No consumed Manager raw-copy batches; engine-local copies are outside scope"
        )

    groups = defaultdict(list)
    for event in copies:
        groups[(event["backend"], event["direction"], event["device"])].append(event)
    summaries = []
    for (backend, direction, device), events in sorted(groups.items()):
        timings = {}
        for field in ("cpu_submit_ns", "host_drain_ns"):
            values = [event[field] / 1e6 for event in events]
            timings[field.removesuffix("_ns") + "_ms"] = {
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
        summaries.append(
            {
                "backend": backend,
                "direction": direction,
                "device": device,
                "batches": len(events),
                "bytes": sum(event["bytes"] for event in events),
                "descriptors": sum(event["descriptors"] for event in events),
                "dma_ranges": sum(event["dma_ranges"] for event in events),
                "size_histogram": [
                    sum(event["size_histogram"][bucket] for event in events) for bucket in range(8)
                ],
                "timings": timings,
            }
        )
    return {
        "scope": "Manager worker raw-copy batches only; engine-local asynchronous restore excluded",
        "processes": len(complete),
        "buffered_records": sum(complete.values()),
        "raw_copy_batches": len(copies),
        "histogram_upper_bounds_bytes": [*SIZE_LIMITS, None],
        "groups": summaries,
        "batches": copies,
        "notes": (
            "Analyze only after normal Manager exit and owned-process cleanup are independently "
            "verified. Completion proves this bounded log's record coverage, not resource drain. "
            "DMA ranges use the actual legal compiler and maximum pitch even for the kernel backend; "
            "they are a shape comparison, not that kernel's CUDA call count. CPU submission and "
            "host stream synchronization are measured separately within one process. Waiting may "
            "include earlier stream work and scheduling; neither is isolated device time or TTFT. "
            "Batch quantiles include all recorded batches, including cold startup; no warmup filter, "
            "request linkage, confidence interval or performance advantage is inferred."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.manager_log)
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
