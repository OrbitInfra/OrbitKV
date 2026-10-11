import copy
import json

import pytest

from benches.local_restore_shapes import _collect


def records():
    base = {"schema": 1, "pid": 17, "trace_id": 1, "device": 0}
    geometry = {
        "device": 0x10000,
        "host": 0x100100,
        "host_device": 0x200100,
        "size": 4096,
        "device_allocation": 7,
        "host_registration": 0x100000,
        "registered_device_start": 0x10000,
        "registered_device_bytes": 65536,
        "destination_offset": 0,
        "arena_id": 2,
        "arena_bytes": 65536,
        "source_allocation_id": 3,
        "source_allocation_offset": 256,
        "source_allocation_bytes": 16384,
        "source_offset": 256,
    }
    return [
        {**base, "stage": "local_raw_copy_open", "limit": 100, "max_pitch": 65536},
        {
            **base,
            "stage": "local_raw_copy_batch",
            "part_id": 0,
            "batch_id": 0,
            "backend": "direct",
            "direction": "h2d",
            "bytes": 4096,
            "descriptors": 1,
            "dma_ranges": 1,
            "zero_descriptors": 0,
            "min_bytes": 4096,
            "max_bytes": 4096,
            "size_histogram": [0, 1, 0, 0, 0, 0, 0, 0],
            "cpu_submit_ns": 19,
            "success": True,
            "copies": [geometry],
        },
        {
            **base,
            "stage": "local_raw_copy_part",
            "part_id": 0,
            "final_part": True,
            "planned_descriptors": 1,
            "submitted_descriptors": 1,
            "batches": 1,
            "host_drain_ns": 31,
            "success": True,
        },
        {
            **base,
            "stage": "local_raw_copy_complete",
            "parts": 1,
            "batches": 1,
            "descriptors": 1,
            "records_reserved": 3,
            "invalid_reason": None,
            "complete": True,
        },
    ]


def log(tmp_path, events):
    path = tmp_path / "engine.log"
    path.write_text("\n".join("prefix local_raw_copy " + json.dumps(event) for event in events))
    return path


def test_actual_geometry_and_shared_part_wait_are_reported_separately(tmp_path):
    events = records()
    result = _collect(log(tmp_path, events))
    assert result["geometry_validation"] == "scalar_consistency_only_not_qualified"
    assert result["bytes"] == 4096
    assert result["cpu_submit_ns"] == 19
    assert result["host_drain_ns"] == 31
    assert result["traces"][0]["batches"][0]["copies"] == events[1]["copies"]


@pytest.mark.parametrize(
    "case",
    [
        "missing_footer",
        "overflow",
        "missing_part",
        "bad_device",
        "duplicate_header",
        "late_batch",
        "bad_bounds",
        "bad_generation",
        "size_disagrees",
        "failed_batch",
        "lost_descriptor",
        "wrong_part",
        "boolean_clock",
        "wrong_reservation",
        "empty_only",
        "mapped_span_overflow",
        "mapped_base_zero",
        "changed_arena_next_part",
        "changed_tensor_next_part",
        "changed_generation_next_part",
    ],
)
def test_partial_failed_or_inconsistent_capture_is_rejected(tmp_path, case):
    events = copy.deepcopy(records())
    if case == "missing_footer":
        events.pop()
    elif case == "overflow":
        events[-1].update(complete=False, invalid_reason="capacity_exceeded")
    elif case == "missing_part":
        events.pop(2)
    elif case == "bad_device":
        events[1]["device"] = 1
    elif case == "duplicate_header":
        events.insert(1, events[0])
    elif case == "late_batch":
        events.append(events[1])
    elif case == "bad_bounds":
        events[1]["copies"][0]["destination_offset"] = 65536
    elif case == "bad_generation":
        events[1]["copies"][0]["source_allocation_bytes"] = 1
    elif case == "size_disagrees":
        events[1]["copies"][0]["size"] = 2048
    elif case == "failed_batch":
        events[1]["success"] = False
    elif case == "lost_descriptor":
        events[2]["planned_descriptors"] = 2
    elif case == "wrong_part":
        events[1]["part_id"] = 1
    elif case == "boolean_clock":
        events[2]["host_drain_ns"] = True
    elif case == "wrong_reservation":
        events[-1]["records_reserved"] = 2
    elif case == "empty_only":
        events.pop(1)
        events[1].update(planned_descriptors=0, submitted_descriptors=0, batches=0)
        events[-1].update(batches=0, descriptors=0, records_reserved=1)
    elif case in ("mapped_span_overflow", "mapped_base_zero"):
        events[1]["copies"][0]["host_device"] = (
            2**64 - 8192 if case == "mapped_span_overflow" else 256
        )
    else:
        second = copy.deepcopy(events[1:3])
        second[0]["part_id"] = second[1]["part_id"] = 1
        if case == "changed_arena_next_part":
            second[0]["copies"][0]["host_device"] += 65536
        elif case == "changed_tensor_next_part":
            second[0]["copies"][0]["registered_device_bytes"] += 65536
        else:
            second[0]["copies"][0]["source_allocation_bytes"] -= 4096
        events[2]["final_part"] = False
        events[3:3] = second
        events[-1].update(parts=2, batches=2, descriptors=2, records_reserved=6)
    with pytest.raises(ValueError):
        _collect(log(tmp_path, events))
