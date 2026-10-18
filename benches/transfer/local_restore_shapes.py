"""Validate bounded engine-local raw H2D observations after each physical part drain."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from benches.transfer.raw_copy_shapes import SIZE_LIMITS, _integer, _validate_shape

COPY_FIELDS = (
    "device",
    "host",
    "host_device",
    "size",
    "device_allocation",
    "host_registration",
    "registered_device_start",
    "registered_device_bytes",
    "destination_offset",
    "arena_id",
    "arena_bytes",
    "source_allocation_id",
    "source_allocation_offset",
    "source_allocation_bytes",
    "source_offset",
)


def _decode_integer(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    for digit in range(10):
        if cursor == len(data):
            raise ValueError("Truncated local H2D descriptor integer")
        byte = data[cursor]
        cursor += 1
        if digit == 9 and byte > 1:
            raise ValueError("Local H2D descriptor integer exceeds u64")
        value |= (byte & 127) << (digit * 7)
        if byte < 128:
            if digit and byte == 0:
                raise ValueError("Noncanonical local H2D descriptor integer")
            return value, cursor
    raise ValueError("Unterminated local H2D descriptor integer")


def _decode_copies(batch: dict) -> None:
    copies = batch.get("copies")
    if batch["schema"] == 4:
        rows = batch.get("copy_rows")
        if (
            type(rows) is not int
            or rows != batch["descriptors"]
            or not 1 <= rows <= 65536
            or batch.get("copies_encoding") != "uvarint-rle-base64-v1"
            or not isinstance(copies, str)
        ):
            raise ValueError("Malformed encoded local H2D descriptor frame")
        max_bytes = rows * len(COPY_FIELDS) * 11
        if len(copies) > 4 * ((max_bytes + 2) // 3):
            raise ValueError("Encoded local H2D geometry exceeds its bounded frame")
        data = base64.b64decode(copies, validate=True)
        if len(data) > max_bytes or base64.b64encode(data).decode("ascii") != copies:
            raise ValueError("Noncanonical local H2D descriptor encoding")
        copies = [[0] * len(COPY_FIELDS) for _ in range(rows)]
        cursor = 0
        for column in range(len(COPY_FIELDS)):
            next_row = 0
            previous = None
            while next_row < rows:
                run, cursor = _decode_integer(data, cursor)
                if not 1 <= run <= rows - next_row:
                    raise ValueError("Local H2D descriptor run exceeds its column")
                value, cursor = _decode_integer(data, cursor)
                if value == previous:
                    raise ValueError("Noncanonical adjacent local H2D descriptor runs")
                for index in range(next_row, next_row + run):
                    copies[index][column] = value
                next_row += run
                previous = value
        if cursor != len(data):
            raise ValueError("Extra local H2D descriptor bytes")
    if not isinstance(copies, list) or len(copies) != batch["descriptors"]:
        raise ValueError("Missing local H2D descriptor geometry")
    if any(not isinstance(row, list) or len(row) != len(COPY_FIELDS) for row in copies):
        raise ValueError("Malformed compact local H2D descriptor")
    batch["copies"] = [dict(zip(COPY_FIELDS, row, strict=True)) for row in copies]


def _geometry(batch: dict) -> None:
    copies = batch.get("copies")
    if not isinstance(copies, list) or len(copies) != batch["descriptors"]:
        raise ValueError("Missing local H2D descriptor geometry")
    histogram = [0] * 8
    sizes = []
    allocations = {}
    for copy in copies:
        if not isinstance(copy, dict):
            raise ValueError("Malformed local H2D descriptor")
        values = {
            field: _integer(copy, field, minimum)
            for field, minimum in (
                ("device", 1),
                ("host", 1),
                ("host_device", 1),
                ("size", 1),
                ("device_allocation", 0),
                ("host_registration", 1),
                ("registered_device_start", 1),
                ("registered_device_bytes", 1),
                ("destination_offset", 0),
                ("arena_id", 1),
                ("arena_bytes", 1),
                ("source_allocation_id", 1),
                ("source_allocation_offset", 0),
                ("source_allocation_bytes", 1),
                ("source_offset", 0),
            )
        }
        size = values["size"]
        end = values["source_allocation_offset"] + values["source_allocation_bytes"]
        if (
            values["registered_device_start"] + values["destination_offset"] != values["device"]
            or values["destination_offset"] + size > values["registered_device_bytes"]
            or values["registered_device_start"] + values["registered_device_bytes"] > 2**64 - 1
            or values["host_registration"] + values["source_offset"] != values["host"]
            or values["host_registration"] + values["arena_bytes"] > 2**64 - 1
            or values["host_device"] < values["source_offset"]
            or values["host_device"] - values["source_offset"] == 0
            or values["host_device"] - values["source_offset"] + values["arena_bytes"] > 2**64 - 1
            or values["host_device"] + size > 2**64 - 1
            or end > values["arena_bytes"]
            or values["source_offset"] < values["source_allocation_offset"]
            or values["source_offset"] + size > end
        ):
            raise ValueError("Local H2D copy exceeds its registered/granted span")
        key = values["arena_id"], values["source_allocation_id"]
        bounds = values["source_allocation_offset"], values["source_allocation_bytes"]
        if allocations.setdefault(key, bounds) != bounds:
            raise ValueError("Inconsistent local H2D source generation bounds")
        sizes.append(size)
        histogram[sum(size > limit for limit in SIZE_LIMITS)] += 1
    if (
        sum(sizes) != batch["bytes"]
        or min(sizes) != batch["min_bytes"]
        or max(sizes) != batch["max_bytes"]
        or histogram != batch["size_histogram"]
    ):
        raise ValueError("Local H2D scalar shape disagrees with actual geometry")


def _part_geometry(batches: list[dict], arenas: dict, tensors: dict, allocations: dict) -> None:
    targets = []
    for batch in batches:
        for copy in batch["copies"]:
            key = copy["arena_id"], copy["source_allocation_id"]
            bounds = copy["source_allocation_offset"], copy["source_allocation_bytes"]
            if allocations.setdefault(key, bounds) != bounds:
                raise ValueError("Inconsistent source generation across local H2D batches")
            arena = (
                copy["host_registration"],
                copy["arena_bytes"],
                copy["host_device"] - copy["source_offset"],
            )
            if arenas.setdefault(copy["arena_id"], arena) != arena:
                raise ValueError("Inconsistent registered source arena across local H2D batches")
            tensor = copy["device_allocation"], copy["registered_device_start"]
            if (
                tensors.setdefault(tensor, copy["registered_device_bytes"])
                != copy["registered_device_bytes"]
            ):
                raise ValueError(
                    "Inconsistent registered destination span across local H2D batches"
                )
            targets.append((copy["device"], copy["size"]))
    targets.sort()
    if any(a + size > b for (a, size), (b, _) in zip(targets, targets[1:], strict=False)):
        raise ValueError("Overlapping local H2D destination ranges")


def _collect(log_path: Path) -> dict:
    traces = {}
    for line in log_path.open():
        _, marker, payload = line.partition("local_raw_copy ")
        if not marker:
            continue
        event = json.loads(payload)
        if (
            not isinstance(event, dict)
            or type(event.get("schema")) is not int
            or event["schema"] not in (2, 3, 4)
        ):
            raise ValueError("Unknown local H2D observation schema")
        key = _integer(event, "pid", 1), _integer(event, "trace_id", 1)
        device = _integer(event, "device")
        sequence = _integer(event, "sequence")
        stage = event.get("stage")
        if stage == "local_raw_copy_open":
            if event["schema"] in (3, 4) and event.get("copy_fields") != list(COPY_FIELDS):
                raise ValueError("Unknown compact local H2D descriptor order")
            if not 1 <= _integer(event, "limit") <= 65536:
                raise ValueError("Invalid local H2D capture budget")
            _integer(event, "max_pitch", 1)
            records_before = _integer(event, "records_before")
            trace = traces.get(key)
            if trace is None:
                if sequence != 0 or records_before != 0:
                    raise ValueError("Missing initial local H2D part")
                trace = traces[key] = {
                    "header": event,
                    "headers": [],
                    "batches": [],
                    "parts": [],
                    "checkpoints": [],
                    "complete": None,
                    "arenas": {},
                    "tensors": {},
                    "allocations": {},
                }
            elif (
                trace["pending_header"] is not None
                or sequence != len(trace["parts"])
                or records_before != trace["complete"]["records_reserved"]
                or any(
                    event[field] != trace["header"][field]
                    for field in ("schema", "device", "limit", "max_pitch")
                )
            ):
                raise ValueError("Noncontiguous or inconsistent local H2D part opening")
            trace["pending_header"] = event
            trace["complete"] = None
            trace["headers"].append(event)
            continue
        trace = traces.get(key)
        if (
            trace is None
            or trace["pending_header"] is None
            or sequence != trace["pending_header"]["sequence"]
            or device != trace["header"]["device"]
            or event["schema"] != trace["header"]["schema"]
        ):
            raise ValueError("Local H2D record outside an open matching trace")
        if stage == "local_raw_copy_batch":
            _validate_shape(event, timing_fields=("device", "cpu_submit_ns"))
            if event.get("direction") != "h2d":
                raise ValueError("Engine-local copy direction must be H2D")
            part_id = _integer(event, "part_id")
            batch_id = _integer(event, "batch_id")
            current = [batch for batch in trace["batches"] if batch["part_id"] == part_id]
            if part_id != len(trace["parts"]) or batch_id != len(current):
                raise ValueError("Noncontiguous local H2D part/batch sequence")
            retained = sum(batch["descriptors"] for batch in trace["batches"])
            if retained + event["descriptors"] > trace["header"]["limit"]:
                raise ValueError("Local H2D geometry exceeds capture budget")
            if event["schema"] in (3, 4):
                _decode_copies(event)
            _geometry(event)
            trace["batches"].append(event)
        elif stage == "local_raw_copy_part":
            part_id = _integer(event, "part_id")
            if part_id != len(trace["parts"]) or type(event.get("final_part")) is not bool:
                raise ValueError("Invalid local H2D part completion")
            if event.get("success") is not True:
                raise ValueError("Unsuccessful local H2D part")
            _integer(event, "host_drain_ns")
            batches = [batch for batch in trace["batches"] if batch["part_id"] == part_id]
            count = sum(batch["descriptors"] for batch in batches)
            if (
                _integer(event, "planned_descriptors") != count
                or _integer(event, "submitted_descriptors") != count
                or _integer(event, "batches") != len(batches)
            ):
                raise ValueError("Incomplete local H2D part descriptor coverage")
            _part_geometry(batches, trace["arenas"], trace["tensors"], trace["allocations"])
            trace["parts"].append(event)
        elif stage == "local_raw_copy_complete":
            if event.get("complete") is not True or event.get("invalid_reason") is not None:
                raise ValueError("Incomplete/overflow/failed local H2D capture")
            count = sum(batch["descriptors"] for batch in trace["batches"])
            reserved = 2 * sum(part["planned_descriptors"] for part in trace["parts"]) + len(
                trace["parts"]
            )
            if (
                _integer(event, "parts") != len(trace["parts"])
                or _integer(event, "batches") != len(trace["batches"])
                or _integer(event, "descriptors") != count
                or _integer(event, "records_reserved") != reserved
                or reserved > trace["header"]["limit"]
                or sequence + 1 != len(trace["parts"])
                or any(batch["part_id"] >= len(trace["parts"]) for batch in trace["batches"])
            ):
                raise ValueError("Local H2D footer disagrees with bounded record coverage")
            trace["complete"] = event
            trace["pending_header"] = None
            trace["checkpoints"].append(event)
        else:
            raise ValueError("Unknown local H2D record stage")
    if not traces or any(trace["complete"] is None for trace in traces.values()):
        raise ValueError(
            "Missing physical part-drain completion; partial local H2D logs are invalid"
        )
    batches = [batch for trace in traces.values() for batch in trace["batches"]]
    if not batches:
        raise ValueError(
            "No actual engine-local H2D batches; managed empty parts do not supply geometry"
        )
    parts = [part for trace in traces.values() for part in trace["parts"]]
    for trace in traces.values():
        del trace["arenas"], trace["tensors"], trace["allocations"], trace["pending_header"]
    return {
        "geometry_validation": "scalar_consistency_only_not_qualified",
        "scope": "Actual engine-local raw H2D submissions; registered tensor spans are not full physical allocation extents",
        "traces": list(traces.values()),
        "executors": len(traces),
        "batches": len(batches),
        "descriptors": sum(batch["descriptors"] for batch in batches),
        "bytes": sum(batch["bytes"] for batch in batches),
        "dma_ranges": sum(batch["dma_ranges"] for batch in batches),
        "parts": len(parts),
        "cpu_submit_ns": sum(batch["cpu_submit_ns"] for batch in batches),
        "host_drain_ns": sum(part["host_drain_ns"] for part in parts),
        "notes": "CPU submission belongs to each batch; stream wait belongs once to its whole part. Neither isolates GPU duration. Each checkpoint seals the cumulatively bounded geometry through its drained part, not executor/process shutdown or operation completion. Every opened part requires a footer. Independent process/GPU/lease cleanup is required. JSON/logging contributes diagnostic operation latency; instrumentation overhead, inference performance and source-operation linkage remain unqualified.",
    }


def collect(log_path: Path, native_fixture: Path) -> dict:
    summary = _collect(log_path)
    fixture_hash = hashlib.sha256(native_fixture.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="orbitkv-h2d-geometry-") as directory:
        root = Path(directory)
        source = root / "geometry.json"
        output = root / "receipt.json"
        source.write_text(json.dumps(summary))
        subprocess.run(
            [
                str(native_fixture.resolve()),
                "--validate-local-geometry",
                str(source),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        receipt = json.loads(output.read_text())
        if (
            type(receipt.get("schema")) is not int
            or receipt["schema"] != 1
            or receipt.get("production_compiler_verified") is not True
            or any(
                type(receipt.get(field)) is not int or receipt[field] != summary[field]
                for field in (
                    "executors",
                    "batches",
                    "descriptors",
                    "dma_ranges",
                )
            )
        ):
            raise ValueError("Native production compiler receipt disagrees with capture")
        if hashlib.sha256(native_fixture.read_bytes()).hexdigest() != fixture_hash:
            raise ValueError("Native geometry fixture changed during capture validation")
        summary["geometry_validation"] = "production_compiler_verified"
        summary["native_geometry_validation"] = {
            "fixture_sha256": fixture_hash,
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "receipt": receipt,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-log", type=Path, required=True)
    parser.add_argument("--native-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.engine_log, args.native_fixture)
    with args.output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
