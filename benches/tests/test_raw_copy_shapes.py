"""Finalized raw-copy evidence rejects missing, failed and inconsistent observations."""

import json

import pytest

from benches.raw_copy_shapes import collect


def shape(**changes):
    return {
        "stage": "manager_raw_copy_complete",
        "pid": 11,
        "diagnostic_event": 1,
        "monotonic_ns": 13,
        "device": 0,
        "backend": "direct",
        "direction": "d2h",
        "bytes": 8192,
        "descriptors": 2,
        "dma_ranges": 1,
        "zero_descriptors": 0,
        "min_bytes": 4096,
        "max_bytes": 4096,
        "size_histogram": [0, 2, 0, 0, 0, 0, 0, 0],
        "cpu_submit_ns": 100000,
        "host_drain_ns": 300000,
        "success": True,
        **changes,
    }


def finalized(event=None):
    return [
        {"stage": "restore_complete", "pid": 11},
        event or shape(),
        {"stage": "diagnostic_timeline_complete", "pid": 11, "reserved": 2, "ready": 2},
    ]


def write_log(tmp_path, events):
    log = tmp_path / "manager.log"
    log.write_text("\n".join("prefix cache_timeline " + json.dumps(event) for event in events))
    return log


def test_counts_actual_batches_separately_from_other_stages_and_processes(tmp_path):
    events = finalized()
    events += [
        shape(pid=17, backend="kernel", direction="h2d", diagnostic_event=0),
        {"stage": "diagnostic_timeline_complete", "pid": 17, "reserved": 1, "ready": 1},
    ]
    result = collect(write_log(tmp_path, events))
    assert result["processes"] == 2
    assert result["buffered_records"] == 3
    assert result["raw_copy_batches"] == 2
    assert len(result["groups"]) == 2
    assert result["groups"][0]["timings"]["cpu_submit_ms"]["p50"] == 0.1
    assert result["groups"][0]["dma_ranges"] == 1
    assert result["batches"] == [events[1], events[3]]


@pytest.mark.parametrize(
    "changes",
    [
        {"success": False},
        {"success": 1},
        {"descriptors": True},
        {"descriptors": 0},
        {"dma_ranges": 3},
        {"bytes": 1},
        {"bytes": 20000},
        {"zero_descriptors": 1},
        {"min_bytes": 0},
        {"max_bytes": 10000},
        {"size_histogram": [0, 1, 0, 0, 0, 0, 0, 0]},
        {"size_histogram": [2, 0, 0, 0, 0, 0, 0, 0]},
        {"size_histogram": [0, True, 0, 0, 0, 0, 0, 0]},
        {"cpu_submit_ns": -1},
        {"host_drain_ns": 2**64},
        {"backend": "auto"},
        {"direction": "peer"},
    ],
)
def test_malformed_or_failed_shapes_invalidate_evidence(tmp_path, changes):
    with pytest.raises(ValueError):
        collect(write_log(tmp_path, finalized(shape(**changes))))


@pytest.mark.parametrize(
    "case",
    [
        "no_flush",
        "lost_record",
        "duplicate_flush",
        "late_record",
        "overflow",
        "duplicate_slot",
        "slot_outside_buffer",
    ],
)
def test_partial_duplicate_and_overflow_buffers_invalidate_evidence(tmp_path, case):
    events = finalized()
    if case == "no_flush":
        events.pop()
    elif case == "lost_record":
        events.pop(0)
    elif case == "duplicate_flush":
        events.append(events[-1])
    elif case == "late_record":
        events.append(shape())
    elif case == "overflow":
        events.append({"stage": "diagnostic_timeline_limit", "pid": 11})
    elif case == "duplicate_slot":
        events[0]["diagnostic_event"] = 1
    else:
        events[1]["diagnostic_event"] = 2
    with pytest.raises(ValueError):
        collect(write_log(tmp_path, events))


def test_complete_buffer_without_manager_raw_work_is_not_a_restore_measurement(tmp_path):
    with pytest.raises(ValueError, match="No consumed Manager"):
        collect(
            write_log(
                tmp_path,
                [
                    finalized()[0],
                    {"stage": "diagnostic_timeline_complete", "pid": 11, "reserved": 1, "ready": 1},
                ],
            )
        )
