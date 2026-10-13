"""CPU artifact contracts; generated receipts are not hardware evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benches.kernel_pairs import analyze
from benches.metrics import REQUIRED_OWNERSHIP_GAUGES
from benches.sglang_kv_oracle import PAGE_BYTES, PAGES, PLANES, page_maps


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def cohort(root: Path) -> dict:
    cells = [
        {
            "name": f"bytes-{backend}-{length}",
            "kind": "oracle",
            "variant": backend,
            "length": length,
        }
        for backend, length in (
            ("direct", 4096),
            ("kernel", 4096),
            ("kernel", 8192),
            ("direct", 8192),
        )
    ]
    cells.append({"name": "native", "kind": "model"})
    pairs = []
    for index in range(4):
        order = ["direct", "kernel"] if index % 2 == 0 else ["kernel", "direct"]
        pair = {
            "id": index + 1,
            "order": order,
            "direct": f"pair{index + 1}-direct",
            "kernel": f"pair{index + 1}-kernel",
        }
        pairs.append(pair)
        cells.extend(
            {"name": pair[variant], "kind": "model", "variant": variant} for variant in order
        )
    contract = {
        "cells": cells,
        "source_commit": "reviewed-source",
        "model_config_sha256": "a" * 64,
        "paired_design": {
            "pairs": pairs,
            "native_control": "native",
            "lengths": [4096, 8192],
            "repeats": 5,
            "bootstrap_seed": 42,
            "bootstrap_draws": 10000,
            "pressure8k_ci_upper": 0.97,
            "no_regression_ci_upper": 1.05,
        },
    }
    write(
        root / "RESULT.json",
        {
            "state": "COLLECTION_VALID_PENDING_REVIEW",
            "error": None,
            "expected_cells": len(cells),
            "cells": [
                {"name": c["name"], "state": "COLLECTION_VALID", "exit_code": 0} for c in cells
            ],
        },
    )
    quiet = dict.fromkeys(REQUIRED_OWNERSHIP_GAUGES, 0)
    for index, cell in enumerate(cells):
        path = root / "cells" / cell["name"]
        path.mkdir(parents=True)
        if cell["kind"] == "oracle":
            rounds = []
            for generation in range(3):
                sources, targets = page_maps(cell["length"], generation)
                payload = sum(p is not None for p in targets) * PAGE_BYTES * PLANES
                rounds.append(
                    {
                        "generation": generation,
                        "source_pages": sources,
                        "destination_pages": targets,
                        "verified_pool_bytes": PAGES * PAGE_BYTES * PLANES,
                        "verified_payload_bytes": payload,
                        "verified_sentinel_bytes": PAGES * PAGE_BYTES * PLANES - payload,
                        "manager_restored_bytes": payload,
                        "plane_sha256": [
                            hashlib.sha256(str(i).encode()).hexdigest() for i in range(PLANES)
                        ],
                        "manager_after": quiet,
                        "negative_controls": [
                            {
                                "region": "last-payload-byte",
                                "position": (targets[-1] + 1) * PAGE_BYTES - 1,
                                "rejected": True,
                            },
                            {
                                "region": "last-sentinel-byte",
                                "position": PAGES * PAGE_BYTES - 1,
                                "rejected": True,
                            },
                        ]
                        if generation == 0
                        else [],
                    }
                )
            write(
                path / "RESULT.json",
                {
                    "status": "VALID_FULL_LAYOUT_BYTES",
                    "backend": cell["variant"],
                    "length": cell["length"],
                    "model_config_sha256": "a" * 64,
                    "physical_planes": PLANES,
                    "pages_per_plane": PAGES,
                    "bytes_per_page_per_plane": PAGE_BYTES,
                    "unregistered": True,
                    "client_closed": True,
                    "rounds": rounds,
                },
            )
            continue
        native = cell["name"] == "native"
        args = {
            "engine": "sglang",
            "model": "frozen-qwen",
            "backend": "native" if native else "orbitkv",
            "workload": "serial",
            "lengths": [4096, 8192],
            "repeats": 5,
            "sglang_profile": False,
            "trace_transfers": False,
            "observation_mode": "boundary",
            "installed_artifact": True,
            "enforce_eager": True,
            "deterministic_inference": True,
            "gpu_tokens": 16384,
            "prefill_tokens": 8192,
            "host_gib": 16,
            "output_tokens": 16,
            "seed": 20260920,
            "storage_codec": "none",
            "ssd_gib": 0,
            "orbitkv_transfer_backend": cell.get("variant"),
        }
        write(
            path / "manifest.json",
            {
                "arguments": args,
                "git_commit": "reviewed-source",
                "git_diff": "",
                "observation_mode": "boundary",
                "manager_present": not native,
                "manager_backend": None if native else "orbitkv",
                "model_revision": None,
                "kv_bytes_per_token": 262144,
                "gpu": "CPU fixture",
                "python": "same",
                "cpu_affinity": [1],
                "packages": {"sglang": "0.5.21", "torch": "same", "transformers": "same"},
            },
        )
        rows = []
        for length in (4096, 8192):
            for repeat in range(5):
                for phase in ("cold", "hbm_hit", "after_pressure"):
                    tokens = length - 64
                    host = tokens if not native and phase == "after_pressure" else 0
                    device = tokens if phase == "hbm_hit" else 0
                    ttft = (
                        90
                        if cell.get("variant") == "kernel"
                        and (length, phase) == (8192, "after_pressure")
                        else 100
                    )
                    rows.append(
                        {
                            "length": length,
                            "repeat": repeat,
                            "phase": phase,
                            "ttft_ms": ttft,
                            "e2e_ms": ttft + 16,
                            "prompt_sha256": f"{length}:{repeat}",
                            "text": "same",
                            "matches_cold_output": True,
                            "profiling": {"mode": "off"},
                            "observation_mode": "boundary",
                            "manager_present": not native,
                            "manager_backend": None if native else "orbitkv",
                            "memory_samples": 0,
                            "sampled_peak_bytes": None,
                            "usage": {
                                "cached_tokens": host + device,
                                "cached_tokens_details": {"host": host, "device": device},
                            },
                            "manager_before": quiet if not native else {},
                            "manager_after": quiet if not native else {},
                            "manager_delta": {"orbitkv_load_bytes_total": host * 262144},
                        }
                    )
        (path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        pid = 1000 + index
        text = f"SIGTERM received.\nRemaining number of requests 0.\nFinished server process [{pid}]\ninclude_parent=False\n"
        (path / "engine.log").write_text(text)
        stop = {
            "pid": pid,
            "exit_code": 0,
            "was_running": True,
            "reaped": True,
            "forced_kill": False,
            "interrupted": False,
            "remaining_group": [],
            "process_group_errors": 0,
            "adopted_children": [{"pid": pid + 1000, "exit_code": -9}],
            "stop_started_mono_ns": 1000 + index * 100,
            "stop_finished_mono_ns": 1020 + index * 100,
            "shutdown_log_start_bytes": 0,
            "shutdown_log_end_bytes": len(text.encode()),
            "shutdown_log_truncated": False,
        }
        write(path / "engine.cleanup.json", stop)
        if not native:
            write(
                path / "manager.cleanup.json",
                {
                    **stop,
                    "pid": pid + 2000,
                    "stop_started_mono_ns": stop["stop_finished_mono_ns"] + 1,
                    "stop_finished_mono_ns": stop["stop_finished_mono_ns"] + 2,
                },
            )
    return contract


def test_pairs_use_four_independent_units_and_preserve_native_limits(tmp_path):
    contract = cohort(tmp_path)
    result = analyze(tmp_path, contract)
    assert result["all_frozen_guards_passed"]
    pressure = next(
        r for r in result["groups"] if (r["length"], r["phase"]) == (8192, "after_pressure")
    )
    assert pressure["independent_pairs"] == 4
    assert pressure["geometric_mean_ratio"] == pytest.approx(0.9)
    assert pressure["bootstrap_95_ci"] == pytest.approx([0.9, 0.9])
    assert not result["production_default_selected"]
    assert all(
        not r["all_native_children_graceful_proven"] for r in result["native_shutdown_diagnostics"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_pair",
        "duplicate_cell",
        "wrong_order",
        "profile",
        "trace",
        "changed_model",
        "unmatched_prompt",
        "output",
        "failed_run",
        "wrong_bytes",
        "resident_mixed",
        "forced_stop",
        "stop_reversed",
        "oracle_truncated",
        "missing_control",
        "changed_guard",
    ],
)
def test_invalid_cells_or_pairing_cannot_establish_performance(tmp_path, change):
    contract = cohort(tmp_path)
    cell = tmp_path / "cells/pair1-kernel"
    if change == "missing_pair":
        contract["paired_design"]["pairs"].pop()
    elif change == "duplicate_cell":
        contract["cells"].append(copy.deepcopy(contract["cells"][-1]))
    elif change == "wrong_order":
        contract["paired_design"]["pairs"][0]["order"].reverse()
    elif change in {"profile", "trace", "changed_model"}:
        data = json.loads((cell / "manifest.json").read_text())
        key = {"profile": "sglang_profile", "trace": "trace_transfers", "changed_model": "model"}[
            change
        ]
        data["arguments"][key] = "other" if change == "changed_model" else True
        write(cell / "manifest.json", data)
    elif change == "failed_run":
        write(cell / "failure.json", {"error": "retained failure"})
    elif change in {"forced_stop", "stop_reversed"}:
        p = cell / "manager.cleanup.json"
        record = json.loads(p.read_text())
        record["forced_kill"] = change == "forced_stop"
        if change == "stop_reversed":
            record["stop_started_mono_ns"] -= 40
            record["stop_finished_mono_ns"] -= 40
        write(p, record)
    elif change in {"oracle_truncated", "missing_control"}:
        p = tmp_path / "cells/bytes-kernel-8192/RESULT.json"
        record = json.loads(p.read_text())
        record["rounds"][0][
            "plane_sha256" if change == "oracle_truncated" else "negative_controls"
        ].pop()
        write(p, record)
    elif change == "changed_guard":
        contract["paired_design"]["pressure8k_ci_upper"] = 1.01
    else:
        rows = [json.loads(line) for line in (cell / "samples.jsonl").read_text().splitlines()]
        row = next(
            r
            for r in rows
            if r["phase"] == ("hbm_hit" if change == "resident_mixed" else "after_pressure")
        )
        if change == "unmatched_prompt":
            row["prompt_sha256"] = "wrong"
        elif change == "output":
            row["text"] = "wrong"
        elif change == "wrong_bytes":
            row["manager_delta"]["orbitkv_load_bytes_total"] -= 1
        elif change == "resident_mixed":
            row["manager_delta"]["orbitkv_load_bytes_total"] = PAGE_BYTES
        (cell / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError):
        analyze(tmp_path, contract)


def test_valid_slow_candidate_is_reported_as_failed_guard_not_discarded(tmp_path):
    contract = cohort(tmp_path)
    path = tmp_path / "cells/pair4-kernel/samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if (row["length"], row["phase"]) == (8192, "after_pressure"):
            row["ttft_ms"] = 130
            row["e2e_ms"] = 150
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")
    result = analyze(tmp_path, contract)
    assert result["status"] == "VALID_PAIRED_DIRECTION_EVALUATION"
    assert not result["all_frozen_guards_passed"]
    assert (
        next(r for r in result["groups"] if (r["length"], r["phase"]) == (8192, "after_pressure"))[
            "ratios"
        ][-1]
        == 1.3
    )
