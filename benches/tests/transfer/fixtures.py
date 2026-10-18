"""Shared deterministic inputs for this benchmark family."""

import base64
import copy
import json
from itertools import groupby

from benches.transfer.local_restore_shapes import COPY_FIELDS


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


def encoded_rows(rows):
    data = bytearray()
    for column in zip(*rows, strict=True):
        for value, group in groupby(column):
            for number in (sum(1 for _ in group), value):
                while number >= 128:
                    data.append((number & 127) | 128)
                    number >>= 7
                data.append(number)
    return base64.b64encode(data).decode("ascii")


def wire_records(events, schema):
    events = [copy.deepcopy(event) for event in events]
    if schema in (3, 4):
        for event in events:
            event["schema"] = schema
            if event["stage"] == "local_raw_copy_open":
                event["copy_fields"] = list(COPY_FIELDS)
            elif event["stage"] == "local_raw_copy_batch":
                event["copies"] = [[row[field] for field in COPY_FIELDS] for row in event["copies"]]
                if schema == 4:
                    event["copy_rows"] = len(event["copies"])
                    event["copies_encoding"] = "uvarint-rle-base64-v1"
                    event["copies"] = encoded_rows(event["copies"])
    return events


def log(tmp_path, events, schema=2):
    events = wire_records(events, schema)
    path = tmp_path / "engine.log"
    path.write_text("\n".join("prefix local_raw_copy " + json.dumps(event) for event in events))
    return path
