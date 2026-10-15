import copy
import json

import pytest

from benches.local_restore_shapes import COPY_FIELDS, _collect


def records():
    base = {"schema": 2, "pid": 17, "trace_id": 1, "device": 0, "sequence": 0}
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
        {
            **base,
            "stage": "local_raw_copy_open",
            "limit": 100,
            "max_pitch": 65536,
            "records_before": 0,
        },
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


def wire_records(events, schema):
    events = [copy.deepcopy(event) for event in events]
    if schema == 3:
        for event in events:
            event["schema"] = 3
            if event["stage"] == "local_raw_copy_open":
                event["copy_fields"] = list(COPY_FIELDS)
            elif event["stage"] == "local_raw_copy_batch":
                event["copies"] = [[row[field] for field in COPY_FIELDS] for row in event["copies"]]
    return events


def log(tmp_path, events, schema=2):
    events = wire_records(events, schema)
    path = tmp_path / "engine.log"
    path.write_text("\n".join("prefix local_raw_copy " + json.dumps(event) for event in events))
    return path


@pytest.mark.parametrize("schema", [2, 3])
def test_actual_geometry_and_shared_part_wait_are_reported_separately(tmp_path, schema):
    events = records()
    result = _collect(log(tmp_path, events, schema))
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
        "missing_previous_footer",
        "replenished_budget",
        "skipped_sequence",
        "changed_limit",
        "changed_pitch",
    ],
)
@pytest.mark.parametrize("schema", [2, 3])
def test_partial_failed_or_inconsistent_capture_is_rejected(tmp_path, case, schema):
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
        second = copy.deepcopy(events)
        for event in second:
            event["sequence"] = 1
        second[0]["records_before"] = 3
        second[1]["part_id"] = second[2]["part_id"] = 1
        second[-1].update(parts=2, batches=2, descriptors=2, records_reserved=6)
        if case == "changed_arena_next_part":
            second[1]["copies"][0]["host_device"] += 65536
        elif case == "changed_tensor_next_part":
            second[1]["copies"][0]["registered_device_bytes"] += 65536
        elif case == "changed_generation_next_part":
            second[1]["copies"][0]["source_allocation_bytes"] -= 4096
        elif case == "missing_previous_footer":
            events.pop()
        elif case == "replenished_budget":
            second[0]["records_before"] = 0
        elif case == "skipped_sequence":
            second[0]["sequence"] = 2
        elif case == "changed_limit":
            second[0]["limit"] += 1
        elif case == "changed_pitch":
            second[0]["max_pitch"] += 1
        events[2]["final_part"] = False
        events.extend(second)
    with pytest.raises(ValueError):
        _collect(log(tmp_path, events, schema))


@pytest.mark.parametrize("schema", [2, 3])
def test_drained_checkpoints_share_executor_geometry_and_cumulative_budget(tmp_path, schema):
    events = records()
    events[2]["final_part"] = False
    second = copy.deepcopy(records())
    for event in second:
        event["sequence"] = 1
    second[0]["records_before"] = 3
    second[1]["part_id"] = second[2]["part_id"] = 1
    second[-1].update(parts=2, batches=2, descriptors=2, records_reserved=6)
    events.extend(second)
    result = _collect(log(tmp_path, events, schema))
    assert (result["executors"], result["parts"], result["bytes"]) == (1, 2, 8192)
    wire = wire_records(events, schema)
    assert result["traces"][0]["checkpoints"] == [wire[3], wire[7]]
    assert result["traces"][0]["headers"] == [wire[0], wire[4]]


@pytest.mark.parametrize(
    "case",
    [
        "missing_fields",
        "reordered_fields",
        "short",
        "long",
        "object",
        "boolean",
        "float",
        "negative",
        "overflow",
        "mixed_batch_schema",
        "mixed_next_part_schema",
    ],
)
def test_compact_geometry_requires_exact_order_width_unsigned_values_and_schema(tmp_path, case):
    events = wire_records(records(), 3)
    if case == "missing_fields":
        del events[0]["copy_fields"]
    elif case == "reordered_fields":
        events[0]["copy_fields"].reverse()
    elif case == "short":
        events[1]["copies"][0].pop()
    elif case == "long":
        events[1]["copies"][0].append(0)
    elif case == "object":
        events[1]["copies"] = records()[1]["copies"]
    elif case in ("boolean", "float", "negative", "overflow"):
        events[1]["copies"][0][8] = {
            "boolean": False,
            "float": 0.0,
            "negative": -1,
            "overflow": 2**64,
        }[case]
    elif case == "mixed_batch_schema":
        events[1]["schema"] = 2
        events[1]["copies"] = records()[1]["copies"]
    else:
        second = records()
        for event in second:
            event["sequence"] = 1
        second[0]["records_before"] = 3
        second[1]["part_id"] = second[2]["part_id"] = 1
        second[-1].update(parts=2, batches=2, descriptors=2, records_reserved=6)
        events.extend(second)
    with pytest.raises(ValueError):
        _collect(log(tmp_path, events))


def test_compact_geometry_retains_all_unsigned_bits_before_native_validation(tmp_path):
    events = records()
    for row in events[1]["copies"]:
        row["device"] += 2**63
        row["registered_device_start"] += 2**63
        row["host"] += 2**63
        row["host_device"] += 2**63
        row["host_registration"] += 2**63
        row["arena_id"] = 2**64 - 1
        row["source_allocation_id"] = 2**64 - 2
    result = _collect(log(tmp_path, events, 3))
    assert result["traces"][0]["batches"][0]["copies"] == events[1]["copies"]
