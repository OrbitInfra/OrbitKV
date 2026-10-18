"""Extract measured-request transfer observations without mixing host clocks."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from benches.common.metrics import percentile


def collect(directory: Path, samples: list[dict]) -> dict:
    request_ids = {sample["request_id"] for sample in samples if sample.get("request_id")}
    observed = []
    by_request = defaultdict(list)
    for filename in ("manager.log", "engine.log"):
        for line in (directory / filename).read_text().splitlines():
            _, marker, payload = line.partition("cache_timeline ")
            if not marker:
                continue
            event, _ = json.JSONDecoder().raw_decode(payload)
            event["source"] = filename
            observed.append(event)

    def match_request(rid):
        # vLLM appends a completion index and an engine-unique suffix.
        while rid not in request_ids and "-" in rid:
            rid = rid.rsplit("-", 1)[0]
        return rid if rid in request_ids else None

    restores = defaultdict(set)
    for event in observed:
        if event["stage"] == "restore_link" and (rid := match_request(event["request_id"])):
            restores[event["restore_key"]].add(rid)
    events = []
    for event in observed:
        rid = match_request(event.get("request_id", ""))
        owners = {rid} if rid else restores.get(event.get("restore_key"), set())
        if not owners:
            continue
        events.append(event)
        for owner in owners:
            by_request[owner].append(event)

    if request_ids and not by_request:
        raise ValueError("No transfer timeline records matched the measured request IDs")

    intervals = defaultdict(list)
    for group in by_request.values():
        for start, end, label in (
            ("queued", "first_use", "queued_to_first_use_ms"),
            ("restore_submit", "gpu_ready", "restore_ms"),
        ):
            starts = {}
            for event in group:
                if "monotonic_ns" not in event:
                    continue
                clock = (event["source"], event["pid"])
                now = event["monotonic_ns"]
                if event["stage"] == start:
                    starts[clock] = now
                elif event["stage"] == end and clock in starts:
                    intervals[label].append((now - starts.pop(clock)) / 1e6)
    for event in events:
        if event["stage"] == "local_restore_complete":
            if not event["success"]:
                continue
            intervals["native_restore_ms"].append(event["drained_ns"] / 1e6)
            previous = 0
            for field, label in (
                ("readiness_ns", "native_readiness_ms"),
                ("dispatched_ns", "native_dispatch_ms"),
                ("dequeued_ns", "native_queue_ms"),
                ("claimed_ns", "native_grant_wait_ms"),
                ("submitted_ns", "native_plan_submit_ms"),
                ("drained_ns", "native_drain_wait_ms"),
            ):
                current = event[field]
                intervals[label].append((current - previous) / 1e6)
                previous = current
        elif event["stage"] == "local_restore_observed":
            if event["success"]:
                intervals["native_consumer_wait_ms"].append(event["elapsed_ns"] / 1e6)
        elif event["stage"] == "source_ready":
            label = (
                "prepared_read_ms"
                if event.get("prepare")
                else "warmup_prepare_ms"
                if event["warmup"]
                else "demand_prepare_ms"
            )
            intervals[label].append(event["elapsed_us"] / 1000)
        elif event["stage"] in (
            "discovery_ready",
            "restore_complete",
            "restore_notification",
        ):
            labels = {
                "discovery_ready": "candidate_discovery_ms",
                "restore_complete": "manager_restore_ms",
                "restore_notification": "completion_signal_ms",
            }
            intervals[labels[event["stage"]]].append(event["elapsed_us"] / 1000)

    for label in (
        "restore_ms",
        "manager_restore_ms",
        "completion_signal_ms",
        "native_restore_ms",
        "native_consumer_wait_ms",
    ):
        intervals.setdefault(label, [])
    stage_counts = Counter(event["stage"] for event in events)
    completion_batches = {
        stage: {
            event["restore_key"]
            for event in events
            if event["stage"] == stage and event.get("restore_key") in restores
        }
        for stage in ("restore_complete", "restore_notification", "local_restore_complete")
    }

    terminal = completion_batches["restore_complete"] | completion_batches["local_restore_complete"]
    manager_notifications = restores.keys() - completion_batches["local_restore_complete"]

    with (directory / "timeline.jsonl").open("w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")
    summary = {
        "measured_requests": len(samples),
        "traced_requests": len(by_request),
        "stage_counts": dict(stage_counts),
        "completion_coverage": {
            "linked_restore_batches": len(restores),
            "batches_with_worker_terminal": len(terminal),
            "batches_with_local_terminal": len(completion_batches["local_restore_complete"]),
            "batches_with_notification": len(completion_batches["restore_notification"]),
            "batches_without_worker_terminal": len(restores.keys() - terminal),
            "batches_without_notification": len(
                manager_notifications - completion_batches["restore_notification"]
            ),
            "client_restore_submissions": sum(
                event["stage"] == "restore_submit" and event["source"] == "engine.log"
                for event in events
            ),
            "client_restore_intervals": len(intervals["restore_ms"]),
        },
        "intervals": {
            label: {
                "count": len(values),
                "p50": percentile(values, 0.5) if values else None,
                "p95": percentile(values, 0.95) if values else None,
                "p99": percentile(values, 0.99) if values else None,
            }
            for label, values in intervals.items()
        },
        "notes": (
            "Durations use one process's monotonic clock, transported native elapsed_ns, or Manager-local elapsed_us. "
            "Native stages partition caller-to-drain: readiness includes caller lock/admission; dispatch "
            "ends before native job construction; queue includes enqueue handoff until the worker dequeues; grant_wait includes "
            "grant availability, worker scheduling and plan consumption; plan_submit includes validation "
            "and CUDA enqueue; drain_wait ends after stream synchronization. Successful local samples "
            "exclude failures. Native consumer wait runs from GPU drain to native poll/wait consumption. "
            "Local terminal coverage needs no Manager notification. Manager retirement is excluded. "
            "Manager restore includes submission/worker queue and synchronized copy; completion "
            "signal starts at the worker's terminal timestamp and measures the notification attempt. "
            "Shared-memory completion consumption has no isolated delivery timer. "
            "Use restore_ms for the engine's submit-to-gpu_ready observation, including submission, "
            "restore work and consumer scheduling; it is not a pure delivery interval. "
            "Manager batches are counted once even when several requests share them; client restore "
            "intervals are per request and process. Engine callback observations are not GPU kernel "
            "timestamps. Dense lookup can fuse discovery with preparation. Missing interval samples "
            "have count 0 and null quantiles, never zero latency. Do not subtract interval quantiles "
            "to infer completion latency or add overlapping intervals to obtain TTFT."
        ),
    }
    (directory / "timeline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
