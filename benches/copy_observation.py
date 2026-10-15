"""Evaluate matched diagnostic overhead on fully checked native SGLang restores."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .cuda_submission import _driver_capture, analyze_engine
from .kernel_pairs import _ratio_summary, _service
from .local_restore_shapes import collect as collect_native
from .metrics import validate_ownership
from .sglang_kv_oracle import PAGE_BYTES, PAGES, PLANES, page_maps

VARIANTS = {
    "off": {"trace": False, "driver": False, "limit": 0},
    "native": {"trace": True, "driver": False, "limit": 65536},
    "driver": {"trace": False, "driver": True, "limit": 0},
    "on": {"trace": True, "driver": True, "limit": 65536},
}
PILOT_ORDER = ("off", "on", "on", "off", "off", "on", "on", "off")
COMPONENT_ORDER = (
    "off",
    "native",
    "on",
    "driver",
    "driver",
    "on",
    "native",
    "off",
    "native",
    "driver",
    "off",
    "on",
    "on",
    "off",
    "driver",
    "native",
)


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular observation evidence: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Malformed observation evidence: {path}")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular observation evidence: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_file(cell: Path, mode: str, pid: int) -> Path | None:
    """Require the exact process capture before accepting a cell or its analysis."""
    if mode not in VARIANTS or type(pid) is not int or pid <= 0:
        raise ValueError("Invalid capture mode or process identity")
    directory = cell / "driver"
    if not VARIANTS[mode]["driver"]:
        if directory.exists() or directory.is_symlink():
            raise ValueError(f"Disabled diagnostic unexpectedly emitted capture: {cell.name}")
        return None
    trace = directory / f"cuda-driver-{pid}.jsonl"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or list(directory.iterdir()) != [trace]
        or trace.is_symlink()
        or not trace.is_file()
    ):
        raise ValueError(f"Missing or foreign driver capture: {cell.name}")
    return trace


def validate_activation(result: dict, contract: dict, mode: str, pid: int) -> None:
    """Bind explicit loading to the frozen library and the waited oracle process."""
    if mode not in VARIANTS or type(pid) is not int or pid <= 0:
        raise ValueError("Invalid capture activation identity")
    expected = (
        {
            "loader": "explicit-dlopen",
            "pid": pid,
            "library": contract["collector"],
            "sha256": contract["collector_sha256"],
        }
        if VARIANTS[mode]["driver"]
        else None
    )
    activation = result.get("driver_capture")
    if activation != expected or (
        VARIANTS[mode]["driver"] and type(activation.get("pid")) is not int
    ):
        raise ValueError("Driver collector activation differs from frozen profile")


def validate_observations(result: dict, contract: dict) -> list[int]:
    """Require the exact warm-up, physical terminal and full-byte controls."""
    count = contract["warmup"] + contract["samples"]
    sources, targets = page_maps(contract["length"], 0)
    payload = len(sources) * PAGE_BYTES * PLANES
    expected_controls = {
        ("last-payload-byte", (targets[-1] + 1) * PAGE_BYTES - 1),
        ("last-sentinel-byte", PAGES * PAGE_BYTES - 1),
    }
    if (
        result.get("status") != "VALID_FULL_LAYOUT_OBSERVATION"
        or result.get("backend") != "direct"
        or result.get("length") != contract["length"]
        or result.get("model_config_sha256") != contract["model_config_sha256"]
        or result.get("gpu_uuid") != contract["gpu_uuid"]
        or result.get("sglang") != "0.5.21"
        or result.get("physical_planes") != PLANES
        or result.get("pages_per_plane") != PAGES
        or result.get("bytes_per_page_per_plane") != PAGE_BYTES
        or result.get("unregistered") is not True
        or result.get("client_closed") is not True
        or result.get("observation_design")
        != {"warmup": contract["warmup"], "samples": contract["samples"]}
        or len(result.get("rounds", [])) != count
    ):
        raise ValueError("Incomplete or mismatched observation byte gate")
    elapsed = []
    keys = set()
    epoch = None
    last_drained = 0
    for index, row in enumerate(result["rounds"]):
        controls = row.get("negative_controls", [])
        if (
            row.get("generation") != 0
            or row.get("seed") != contract["seed"]
            or row.get("source_pages") != sources
            or row.get("destination_pages") != targets
            or row.get("verified_pool_bytes") != PAGES * PAGE_BYTES * PLANES
            or row.get("verified_payload_bytes") != payload
            or row.get("verified_sentinel_bytes") != PAGES * PAGE_BYTES * PLANES - payload
            or row.get("manager_restored_bytes") != payload
            or row.get("plane_sha256") != contract["expected_plane_sha256"]
            or len(controls) != 2
            or {(c.get("region"), c.get("position")) for c in controls} != expected_controls
            or any(c.get("rejected") is not True for c in controls)
        ):
            raise ValueError(f"Payload/sentinel/corruption coverage differs: operation {index}")
        validate_ownership(row["manager_after"], drained=True)
        observation = row.get("observation", {})
        clocks = [observation.get(k) for k in ("started_ns", "enqueued_ns", "drained_ns")]
        key = (observation.get("session_epoch"), observation.get("operation_id"))
        if (
            observation.get("operation") != index
            or observation.get("phase") != ("warmup" if index < contract["warmup"] else "measured")
            or observation.get("clock") != "CLOCK_MONOTONIC"
            or observation.get("terminal_done") is not True
            or observation.get("terminal_success") is not True
            or any(type(t) is not int or t <= 0 for t in clocks)
            or not last_drained < clocks[0] <= clocks[1] <= clocks[2]
            or clocks[0] == clocks[2]
            or any(type(k) is not int or k <= 0 for k in key)
            or not isinstance(observation.get("operation_key"), str)
            or not observation["operation_key"]
            or key in keys
            or (epoch is not None and epoch != key[0])
        ):
            raise ValueError(f"Invalid or duplicate native observation: operation {index}")
        epoch = key[0]
        keys.add(key)
        last_drained = clocks[2]
        if index >= contract["warmup"]:
            elapsed.append(clocks[2] - clocks[0])
    return elapsed


def validate_capture(
    path: Path,
    result: dict,
    wait: dict,
    contract: dict,
    native_fixture: Path,
    mode: str,
) -> dict | None:
    """Reject an invalid mode capture before admitting another hardware cell."""
    name = path.name
    driver = capture_file(path, mode, wait.get("pid"))
    inputs = {
        path / "oracle.stderr": _sha256(path / "oracle.stderr"),
        native_fixture: _sha256(native_fixture),
    }
    if inputs[native_fixture] != contract["native_fixture_sha256"]:
        raise ValueError("Native capture compiler differs from frozen profile")
    if driver is not None:
        inputs[driver] = _sha256(driver)
    validate_activation(result, contract, mode, wait["pid"])
    capture = None
    if VARIANTS[mode]["trace"]:
        if mode == "on":
            capture = analyze_engine(
                path / "driver" / f"cuda-driver-{wait['pid']}.jsonl",
                path / "oracle.stderr",
                native_fixture,
            )
        else:
            native = collect_native(path / "oracle.stderr", native_fixture)
            capture = {
                "native_geometry_validation": native["native_geometry_validation"],
                "pid": wait["pid"],
                "batches": [],
            }
            for trace in native["traces"]:
                if trace["header"]["pid"] != wait["pid"]:
                    raise ValueError(f"Foreign native capture: {name}")
                parts = {part["part_id"]: part for part in trace["parts"]}
                for batch in trace["batches"]:
                    capture["batches"].append(
                        {
                            **batch,
                            "trace_id": trace["header"]["trace_id"],
                            "drained_ns": parts[batch["part_id"]]["drained_ns"],
                        }
                    )
        expected = (contract["warmup"] + contract["samples"]) * PLANES
        batches = capture["batches"]
        if len(batches) != expected or capture["pid"] != wait["pid"]:
            raise ValueError(f"Incomplete or foreign native batch capture: {name}")
        parts = sorted({(b["trace_id"], b["part_id"]) for b in batches})
        if (
            len(parts) != contract["warmup"] + contract["samples"]
            or len({t for t, _ in parts}) != 1
        ):
            raise ValueError(f"Incorrect operation/trace coverage: {name}")
        for operation, part in zip(result["rounds"], parts, strict=True):
            observed = operation["observation"]
            group = [b for b in batches if (b["trace_id"], b["part_id"]) == part]
            if len(group) != PLANES or any(
                b["backend"] != "direct"
                or not observed["started_ns"]
                <= b["begin_ns"]
                < b["submitted_ns"]
                <= observed["enqueued_ns"]
                or b["drained_ns"] > observed["drained_ns"]
                for b in group
            ):
                raise ValueError(f"Driver windows differ from native operation: {name}")
    elif any(
        marker in (path / "oracle.stderr").read_text()
        for marker in ("cache_timeline ", "local_raw_copy ")
    ):
        raise ValueError(f"Disabled diagnostic emitted native timeline records: {name}")
    if mode == "driver":
        pid, calls, threads = _driver_capture(path / "driver" / f"cuda-driver-{wait['pid']}.jsonl")
        events = [event for thread in threads.values() for event in thread]
        if pid != wait["pid"] or any(
            not wait["launched_mono_ns"]
            <= event["begin_ns"]
            <= event["end_ns"]
            <= wait["waited_mono_ns"]
            for event in events
        ):
            raise ValueError(f"Foreign driver-only process/window: {name}")
        operation_calls = []
        for row in result["rounds"]:
            clocks = row["observation"]
            copies = [
                event
                for event in events
                if event["name"].startswith("cuMemcpy")
                and event["begin_ns"] < clocks["drained_ns"]
                and event["end_ns"] > clocks["started_ns"]
            ]
            if not copies or any(
                not clocks["started_ns"]
                <= event["begin_ns"]
                <= event["end_ns"]
                <= clocks["enqueued_ns"]
                or event["name"] not in {"cuMemcpyHtoDAsync_v2", "cuMemcpy2DAsync_v2"}
                or event["result"] != 0
                for event in copies
            ):
                raise ValueError(f"Missing or failed driver-only operation capture: {name}")
            operation_calls.append(len(copies))
        capture = {
            "pid": pid,
            "captured_driver_calls": calls,
            "operation_copy_calls": operation_calls,
        }
    capture_file(path, mode, wait["pid"])
    if any(_sha256(file) != expected for file, expected in inputs.items()):
        raise ValueError("Consumed capture evidence changed during cell validation")
    return capture


def collect(directory: Path, contract_path: Path, native_fixture: Path) -> dict:
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("Missing regular observation contract")
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if not isinstance(contract, dict):
        raise ValueError("Malformed observation contract")
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    compiler_sha = _sha256(native_fixture)
    components = contract.get("schema") == "s4.copy-observation-components.v1"
    order = COMPONENT_ORDER if components else PILOT_ORDER
    width = 4 if components else 2
    if (
        contract.get("schema")
        not in {"s4.copy-observation-pilot.v2", "s4.copy-observation-components.v1"}
        or type(contract.get("warmup")) is not int
        or type(contract.get("samples")) is not int
        or contract.get("warmup") != 1
        or contract.get("samples") != 2
        or contract.get("length") != 4096
        or contract.get("max_ratio") != 1.05
        or len(contract.get("cells", [])) != len(order)
        or len(contract.get("expected_plane_sha256", [])) != PLANES
        or len(set(contract["expected_plane_sha256"])) != PLANES
        or any(
            not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in contract["expected_plane_sha256"]
        )
        or contract.get("variant_envs") != {mode: VARIANTS[mode] for mode in set(order)}
        or compiler_sha != contract.get("native_fixture_sha256")
        or not isinstance(contract.get("collector"), str)
        or not Path(contract["collector"]).is_absolute()
        or not isinstance(contract.get("collector_sha256"), str)
        or len(contract["collector_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in contract["collector_sha256"])
    ):
        raise ValueError("Observation design differs from the frozen contract")
    rows = []
    names = set()
    owners = set()
    inputs = {contract_path: contract_sha, native_fixture: compiler_sha}
    driver_files = {}
    paths = []
    for index, cell in enumerate(contract["cells"]):
        name, mode = cell["name"], cell["mode"]
        expected_mode = order[index]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or Path(name).name != name
            or name in {".", ".."}
            or mode != expected_mode
            or cell.get("pair") != index // width
        ):
            raise ValueError("Unbalanced or invalid observation pair order")
        names.add(name)
        path = directory / name
        paths.append(path)
        for file in (
            "CELL.json",
            "ORACLE-WAIT.json",
            "manager.cleanup.json",
            "bytes/RESULT.json",
            "oracle.stderr",
        ):
            inputs[path / file] = _sha256(path / file)
        wait = _read(path / "ORACLE-WAIT.json")
        trace = capture_file(path, mode, wait.get("pid"))
        if trace is not None:
            inputs[trace] = _sha256(trace)
        driver_files[path] = (mode, wait["pid"])
    previous_stop = previous_start_ticks = 0
    for cell, path in zip(contract["cells"], paths, strict=True):
        name, mode = cell["name"], cell["mode"]
        receipt = _read(path / "CELL.json")
        wait = _read(path / "ORACLE-WAIT.json")
        if (
            receipt.get("contract_sha256") != contract_sha
            or receipt.get("mode") != mode
            or receipt.get("variant_env") != contract["variant_envs"][mode]
            or receipt.get("frozen_inputs_unchanged") is not True
            or receipt.get("installed_files_unchanged") is not True
            or receipt.get("physical_postflight_empty") is not True
            or type(wait.get("pid")) is not int
            or wait["pid"] <= 0
            or type(wait.get("start_ticks")) is not int
            or wait["start_ticks"] <= 0
            or (wait["pid"], wait["start_ticks"]) in owners
            or type(wait.get("exit_code")) is not int
            or wait["exit_code"] != 0
            or wait.get("normal_wait") is not True
            or wait.get("forced_cleanup") is not False
            or any(
                type(wait.get(k)) is not int or wait[k] <= 0
                for k in ("launched_mono_ns", "waited_mono_ns")
            )
            or wait["start_ticks"] <= previous_start_ticks
            or wait["launched_mono_ns"] < previous_stop
        ):
            raise ValueError(f"Invalid runtime/ownership receipt: {name}")
        owners.add((wait["pid"], wait["start_ticks"]))
        if type(_read(path / "manager.cleanup.json").get("exit_code")) is not int:
            raise ValueError(f"Invalid Manager wait status: {name}")
        cleanup = _service(path, "manager")
        result = _read(path / "bytes" / "RESULT.json")
        elapsed = validate_observations(result, contract)
        if not (
            wait["launched_mono_ns"]
            <= result["rounds"][0]["observation"]["started_ns"]
            < result["rounds"][-1]["observation"]["drained_ns"]
            <= wait["waited_mono_ns"]
            <= cleanup["stop_started_mono_ns"]
        ):
            raise ValueError(f"Native clocks lie outside live process ownership: {name}")
        previous_stop = cleanup["stop_finished_mono_ns"]
        previous_start_ticks = wait["start_ticks"]
        capture = validate_capture(path, result, wait, contract, native_fixture, mode)
        rows.append(
            {
                "name": name,
                "mode": mode,
                "pair": cell["pair"],
                "elapsed_ns": elapsed,
                "median_ns": statistics.median(elapsed),
                "capture": capture,
            }
        )
    contrasts = {}
    for mode in ("native", "driver", "on") if components else ("on",):
        ratios = []
        for pair in range(4):
            off, observed = (
                next(r for r in rows if r["pair"] == pair and r["mode"] == selected)
                for selected in ("off", mode)
            )
            ratios.append(observed["median_ns"] / off["median_ns"])
        summary = _ratio_summary(ratios, seed=20261014, draws=20000)
        summary["bootstrap_unit"] = (
            f"One freshly restarted off/{mode} process pair within an independent block; requests are not replicates"
            if components
            else "One freshly restarted off/on process pair; requests are not replicates"
        )
        contrasts[mode] = summary
    summary = contrasts["on"]
    ratios = summary["ratios"]
    passed = (
        max(ratios) <= contract["max_ratio"]
        and summary["bootstrap_95_ci"][1] <= contract["max_ratio"]
    )
    for path, (mode, pid) in driver_files.items():
        try:
            capture_file(path, mode, pid)
        except ValueError as error:
            raise ValueError(
                "Diagnostic directory inventory changed during cohort analysis"
            ) from error
    if any(_sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("Consumed observation evidence changed during cohort analysis")
    return {
        "input_sha256": {str(path): expected for path, expected in inputs.items()},
        "status": "PILOT_OVERHEAD_PASS" if passed else "PILOT_OVERHEAD_FAIL",
        "contract_sha256": contract_sha,
        "cells": rows,
        **(
            {
                "component_contrasts": contrasts,
                "component_scope": "Paired mode costs; native/driver effects and interaction are not assumed additive",
            }
            if components
            else {}
        ),
        "pairs": summary,
        "eligible_for_separate_contention_contract": passed,
        "scope": "Isolated complete-restoration host wall pilot; not p99, device-only time, model serving gain, native fault reclamation or a copy default.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-fixture", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.directory, args.contract, args.native_fixture), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
