"""Evaluate fresh SGLang DMA/kernel pairs after the actual-layout byte gate.

Resample independent process pairs, never their individual requests. Native
shutdown diagnostics remain separate from ordinary outer-service receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from .artifacts import external_path
from .metrics import percentile, validate_ownership
from .report import collect_run, compare_outputs
from .sglang_kv_oracle import PAGE_BYTES, PAGES, PLANES, page_maps
from .shutdown import validate_engine_shutdown


def validate_oracle(result: dict, cell: dict, config_sha256: str) -> None:
    if (
        result.get("status") != "VALID_FULL_LAYOUT_BYTES"
        or result.get("backend") != cell["variant"]
        or result.get("length") != cell["length"]
        or result.get("model_config_sha256") != config_sha256
        or result.get("physical_planes") != PLANES
        or result.get("pages_per_plane") != PAGES
        or result.get("bytes_per_page_per_plane") != PAGE_BYTES
        or result.get("unregistered") is not True
        or result.get("client_closed") is not True
        or len(result.get("rounds", [])) != 3
    ):
        raise ValueError(f"Incomplete or mismatched actual-layout byte gate: {cell['name']}")
    for generation, row in enumerate(result["rounds"]):
        sources, targets = page_maps(cell["length"], generation)
        payload = sum(page is not None for page in targets) * PAGE_BYTES * PLANES
        digests = row.get("plane_sha256", [])
        controls = row.get("negative_controls", [])
        expected_controls = (
            {
                ("last-payload-byte", (targets[-1] + 1) * PAGE_BYTES - 1),
                ("last-sentinel-byte", PAGES * PAGE_BYTES - 1),
            }
            if generation == 0
            else set()
        )
        if (
            row.get("generation") != generation
            or row.get("source_pages") != sources
            or row.get("destination_pages") != targets
            or row.get("verified_pool_bytes") != PAGES * PAGE_BYTES * PLANES
            or row.get("verified_payload_bytes") != payload
            or row.get("verified_sentinel_bytes") != PAGES * PAGE_BYTES * PLANES - payload
            or row.get("manager_restored_bytes") != payload
            or len(digests) != PLANES
            or len(set(digests)) != PLANES
            or any(
                not isinstance(d, str)
                or len(d) != 64
                or any(c not in "0123456789abcdef" for c in d)
                for d in digests
            )
            or len(controls) != len(expected_controls)
            or {(c.get("region"), c.get("position")) for c in controls} != expected_controls
            or any(c.get("rejected") is not True for c in controls)
        ):
            raise ValueError(f"Byte/sentinel/control coverage differs: {cell['name']}/{generation}")
        validate_ownership(row["manager_after"], drained=True)


def _service(directory: Path, name: str) -> dict:
    record = json.loads((directory / f"{name}.cleanup.json").read_text())
    if (
        record.get("exit_code") != 0
        or record.get("was_running") is not True
        or record.get("reaped") is not True
        or record.get("forced_kill") is not False
        or record.get("interrupted") is not False
        or record.get("remaining_group") != []
        or record.get("process_group_errors") != 0
        or "shutdown_error" in record
        or any(
            type(record.get(k)) is not int or record[k] <= 0
            for k in ("pid", "stop_started_mono_ns", "stop_finished_mono_ns")
        )
        or record["stop_finished_mono_ns"] < record["stop_started_mono_ns"]
    ):
        raise ValueError(f"Incomplete outer-service shutdown receipt: {directory}/{name}")
    if name == "engine":
        start, end = record["shutdown_log_start_bytes"], record["shutdown_log_end_bytes"]
        with (directory / "engine.log").open("rb") as source:
            source.seek(start)
            text = source.read(end - start)
        if len(text) != end - start or record.get("shutdown_log_truncated") is not False:
            raise ValueError("Missing exact native shutdown log slice")
        validate_engine_shutdown(text.decode(errors="replace"), "sglang", pid=record["pid"])
    return record


def _ratio_summary(ratios: list[float], seed: int, draws: int) -> dict:
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("Pair ratios must be positive finite measurements")
    rng = random.Random(seed)
    bootstrap = sorted(
        math.exp(statistics.fmean(math.log(rng.choice(ratios)) for _ in ratios))
        for _ in range(draws)
    )
    return {
        "independent_pairs": len(ratios),
        "geometric_mean_ratio": math.exp(statistics.fmean(map(math.log, ratios))),
        "bootstrap_95_ci": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "ratios": ratios,
        "bootstrap_unit": "One freshly restarted DMA/kernel process pair; requests are not replicates",
    }


def analyze(root: Path, contract: dict) -> dict:
    design = contract["paired_design"]
    pairs = design["pairs"]
    cells = contract["cells"]
    names = [cell["name"] for cell in cells]
    if (
        len(pairs) != 4
        or [pair["id"] for pair in pairs] != [1, 2, 3, 4]
        or any(type(pair["id"]) is not int for pair in pairs)
        or len(set(names)) != len(names)
        or design["lengths"] != [4096, 8192]
        or design["repeats"] != 5
        or design["bootstrap_draws"] != 10000
        or design["pressure8k_ci_upper"] != 0.97
        or design["no_regression_ci_upper"] != 1.05
    ):
        raise ValueError("Changed frozen independent-pair design")
    terminal = json.loads((root / "RESULT.json").read_text())
    if (
        terminal.get("state") != "COLLECTION_VALID_PENDING_REVIEW"
        or terminal.get("error") is not None
        or terminal.get("expected_cells") != len(cells)
        or [row.get("name") for row in terminal.get("cells", [])] != names
        or any(
            row.get("state") != "COLLECTION_VALID" or row.get("exit_code") != 0
            for row in terminal["cells"]
        )
    ):
        raise ValueError("Invalid, reordered or incomplete cohort cannot establish a direction")
    oracles = [cell for cell in cells if cell["kind"] == "oracle"]
    if {(cell["variant"], cell["length"]) for cell in oracles} != {
        (backend, length) for backend in ("direct", "kernel") for length in (4096, 8192)
    } or len(oracles) != 4:
        raise ValueError("Every actual-layout byte cell is required")
    for cell in oracles:
        path = root / "cells" / cell["name"] / "RESULT.json"
        validate_oracle(json.loads(path.read_text()), cell, contract["model_config_sha256"])
    model_names = {cell["name"] for cell in cells if cell["kind"] == "model"}
    expected_names = {design["native_control"]}
    for pair in pairs:
        expected_names.update((pair["direct"], pair["kernel"]))
    if len(expected_names) != 9 or model_names != expected_names:
        raise ValueError("Missing, duplicate or additional model cells")
    runs = {}
    engines = {}
    for name in model_names:
        directory = root / "cells" / name
        data = collect_run(directory)
        args = data["manifest"]["arguments"]
        if (
            args.get("engine") != "sglang"
            or args.get("workload") != "serial"
            or args.get("lengths") != design["lengths"]
            or args.get("repeats") != 5
            or args.get("sglang_profile") is not False
            or args.get("trace_transfers") is not False
            or args.get("observation_mode") != "boundary"
            or args.get("installed_artifact") is not True
            or args.get("enforce_eager") is not True
            or args.get("deterministic_inference") is not True
            or data["manifest"]["git_commit"] != contract["source_commit"]
            or data["manifest"]["git_diff"]
            or data["manifest"]["kv_bytes_per_token"] != 262144
            or data["manifest"]["packages"].get("sglang") != "0.5.21"
            or args.get("gpu_tokens") != 16384
            or args.get("prefill_tokens") != 8192
            or args.get("host_gib") != 16
            or args.get("output_tokens") != 16
            or args.get("seed") != 20260920
            or any(
                row["profiling_mode"] != "off" or row["n"] != 5 or row["output_mismatches"]
                for row in data["summary"]
            )
        ):
            raise ValueError(f"Mismatched model/profiling contract: {name}")
        engine = _service(directory, "engine")
        engines[name] = engine
        if name == design["native_control"]:
            if args.get("backend") != "native":
                raise ValueError("Fresh output reference must be the native engine")
        else:
            backend = next(
                variant
                for pair in pairs
                for variant in ("direct", "kernel")
                if pair[variant] == name
            )
            if args.get("backend") != "orbitkv" or args.get("orbitkv_transfer_backend") != backend:
                raise ValueError(f"Unconsumed kernel/DMA configuration: {name}")
            manager = _service(directory, "manager")
            if engine["stop_finished_mono_ns"] > manager["stop_started_mono_ns"]:
                raise ValueError(f"Manager unavailable for engine unregistration: {name}")
            samples = [
                json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()
            ]
            for row in samples:
                if row.get("profiling", {}).get("mode") != "off":
                    raise ValueError("Missing explicit unprofiled sample label")
                if row["phase"] == "after_pressure":
                    tokens = row["length"] - 64
                    if (
                        row["usage"].get("cached_tokens_details") != {"device": 0, "host": tokens}
                        or row["manager_delta"].get("orbitkv_load_bytes_total") != tokens * 262144
                    ):
                        raise ValueError(f"Actual restore coverage differs: {name}")
                elif row["phase"] == "hbm_hit":
                    if row["usage"].get("cached_tokens_details") != {
                        "device": row["length"] - 64,
                        "host": 0,
                    } or row["manager_delta"].get("orbitkv_load_bytes_total", 0):
                        raise ValueError(f"Resident phase is not pure native HBM: {name}")
                elif row["phase"] == "cold":
                    if row["usage"].get("cached_tokens", 0) or row["manager_delta"].get(
                        "orbitkv_load_bytes_total", 0
                    ):
                        raise ValueError(
                            f"Cold phase unexpectedly reused external/native state: {name}"
                        )
        runs[name] = data
    if len({record["pid"] for record in engines.values()}) != 9:
        raise ValueError("Independent cells must restart their engine process")
    for name, data in runs.items():
        parity = compare_outputs(data, runs[design["native_control"]])
        if parity["compared_requests"] != 30 or any(
            parity[k]
            for k in (
                "output_mismatches",
                "unverified_input_requests",
                "run_uncompared_requests",
                "reference_uncompared_requests",
            )
        ):
            raise ValueError(f"Fresh native input/output parity differs: {name}")
    comparisons = []
    for index, pair in enumerate(pairs):
        order = ["direct", "kernel"] if index % 2 == 0 else ["kernel", "direct"]
        actual_order = sorted((pair[v] for v in order), key=names.index)
        if pair.get("order") != order or actual_order != [pair[v] for v in order]:
            raise ValueError("Pair order does not match the frozen balanced sequence")
        first, second = (engines[pair[v]] for v in order)
        if first["stop_finished_mono_ns"] > second["stop_started_mono_ns"]:
            raise ValueError("Actual engine stop boundaries contradict pair order")
        for length in design["lengths"]:
            for phase in ("cold", "hbm_hit", "after_pressure"):
                medians = {
                    v: next(
                        row["ttft_p50_ms"]
                        for row in runs[pair[v]]["summary"]
                        if row["length"] == length and row["phase"] == phase
                    )
                    for v in ("direct", "kernel")
                }
                if any(not math.isfinite(value) or value <= 0 for value in medians.values()):
                    raise ValueError("Pair medians must be positive finite measurements")
                comparisons.append(
                    {
                        "pair": pair["id"],
                        "order": order,
                        "length": length,
                        "phase": phase,
                        "median_ttft_ms": medians,
                        "kernel_over_direct": medians["kernel"] / medians["direct"],
                        "kernel_minus_direct_ms": medians["kernel"] - medians["direct"],
                    }
                )
    groups = []
    for length in design["lengths"]:
        for phase in ("cold", "hbm_hit", "after_pressure"):
            ratios = [
                row["kernel_over_direct"]
                for row in comparisons
                if row["length"] == length and row["phase"] == phase
            ]
            summary = _ratio_summary(ratios, design["bootstrap_seed"], design["bootstrap_draws"])
            gate = (
                design["pressure8k_ci_upper"]
                if (length, phase) == (8192, "after_pressure")
                else design["no_regression_ci_upper"]
            )
            passed = summary["bootstrap_95_ci"][1] <= gate
            if (length, phase) == (8192, "after_pressure"):
                passed &= all(ratio < 1 for ratio in ratios)
            groups.append(
                {
                    "length": length,
                    "phase": phase,
                    **summary,
                    "required_ci_upper": gate,
                    "passed": bool(passed),
                }
            )
    return {
        "status": "VALID_PAIRED_DIRECTION_EVALUATION",
        "pairs": comparisons,
        "groups": groups,
        "all_frozen_guards_passed": all(row["passed"] for row in groups),
        "production_default_selected": False,
        "native_shutdown_diagnostics": [
            {
                "cell": name,
                "leader_exit_code": record["exit_code"],
                "adopted_children": record.get("adopted_children", []),
                "CUDA_IPC_producer_warnings": (root / "cells" / name / "engine.log")
                .read_text(errors="replace")
                .count(
                    "Producer process has been terminated before all shared CUDA tensors released"
                ),
                "all_native_children_graceful_proven": False,
            }
            for name, record in sorted(engines.items())
        ],
        "scope": "Four independent actual-model pairs plus released-pool byte controls; no p99, throughput, computed model KV dump, native-child graceful/IPC retirement or S3/support qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    contract_bytes = args.contract.read_bytes()
    result = analyze(args.runs, json.loads(contract_bytes))
    result["contract_sha256"] = hashlib.sha256(contract_bytes).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "ANALYSIS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_frozen_guards_passed": result["all_frozen_guards_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
