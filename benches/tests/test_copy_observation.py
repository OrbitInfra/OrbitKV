"""Malformed observations and reused processes cannot qualify diagnostic overhead."""

import copy
import hashlib
import json

import pytest

from benches import copy_observation as observation
from benches.metrics import DRAIN_GAUGES, REQUIRED_OWNERSHIP_GAUGES
from benches.sglang_kv_oracle import PAGE_BYTES, PAGES, PLANES, page_maps


def result(contract, factor=1):
    sources, targets = page_maps(4096, 0)
    payload = len(sources) * PAGE_BYTES * PLANES
    rounds = []
    for index in range(5):
        start = 1_000_000 + index * 1_000_000
        duration = int(400_000 * factor)
        rounds.append(
            {
                "generation": 0,
                "seed": contract["seed"],
                "source_pages": sources,
                "destination_pages": targets,
                "verified_pool_bytes": PAGES * PAGE_BYTES * PLANES,
                "verified_payload_bytes": payload,
                "verified_sentinel_bytes": PAGES * PAGE_BYTES * PLANES - payload,
                "manager_restored_bytes": payload,
                "plane_sha256": contract["expected_plane_sha256"].copy(),
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
                ],
                "manager_after": dict.fromkeys(
                    set(DRAIN_GAUGES) | set(REQUIRED_OWNERSHIP_GAUGES), 0
                ),
                "observation": {
                    "operation": index,
                    "phase": "warmup" if index == 0 else "measured",
                    "clock": "CLOCK_MONOTONIC",
                    "started_ns": start,
                    "enqueued_ns": start + duration // 2,
                    "drained_ns": start + duration,
                    "terminal_done": True,
                    "terminal_success": True,
                    "session_epoch": 3,
                    "operation_id": index + 1,
                    "operation_key": f"3:{index + 1}",
                },
            }
        )
    return {
        "status": "VALID_FULL_LAYOUT_OBSERVATION",
        "backend": "direct",
        "length": 4096,
        "model_config_sha256": contract["model_config_sha256"],
        "gpu_uuid": contract["gpu_uuid"],
        "sglang": "0.5.21",
        "physical_planes": PLANES,
        "pages_per_plane": PAGES,
        "bytes_per_page_per_plane": PAGE_BYTES,
        "unregistered": True,
        "client_closed": True,
        "observation_design": {"warmup": 1, "samples": 4},
        "rounds": rounds,
    }


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.write_bytes(b"Frozen compiled geometry control")
    contract = {
        "schema": "s4.copy-observation-pilot.v1",
        "warmup": 1,
        "samples": 4,
        "length": 4096,
        "max_ratio": 1.05,
        "seed": 20260920,
        "model_config_sha256": "a" * 64,
        "gpu_uuid": "GPU-26b6f4ae-dd04-cfdd-3e78-0150a85ae9fb",
        "native_fixture_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "expected_plane_sha256": [f"{i:064x}" for i in range(PLANES)],
        "variant_envs": {
            "off": {"trace": False, "driver": False, "limit": 0},
            "on": {"trace": True, "driver": True, "limit": 65536},
        },
        "cells": [],
    }
    for i, mode in enumerate(("off", "on", "on", "off", "off", "on", "on", "off")):
        contract["cells"].append({"name": f"cell-{i}-{mode}", "mode": mode, "pair": i // 2})
    cp = tmp_path / "contract.json"
    cp.write_text(json.dumps(contract))
    sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    root = tmp_path / "cells"
    root.mkdir()

    def put(path, value):
        path.write_text(json.dumps(value))

    for i, cell in enumerate(contract["cells"]):
        path = root / cell["name"]
        (path / "bytes").mkdir(parents=True)
        put(path / "bytes/RESULT.json", result(contract))
        put(
            path / "CELL.json",
            {
                "mode": cell["mode"],
                "contract_sha256": sha,
                "variant_env": contract["variant_envs"][cell["mode"]],
                "frozen_inputs_unchanged": True,
                "installed_files_unchanged": True,
                "physical_postflight_empty": True,
            },
        )
        put(
            path / "ORACLE-WAIT.json",
            {
                "pid": 100 + i,
                "start_ticks": 1000 + i,
                "exit_code": 0,
                "normal_wait": True,
                "forced_cleanup": False,
            },
        )
        put(
            path / "manager.cleanup.json",
            {
                "pid": 500 + i,
                "exit_code": 0,
                "was_running": True,
                "reaped": True,
                "forced_kill": False,
                "interrupted": False,
                "remaining_group": [],
                "process_group_errors": 0,
                "stop_started_mono_ns": 1,
                "stop_finished_mono_ns": 2,
            },
        )
        if cell["mode"] == "on":
            (path / "driver").mkdir()
            (path / "oracle.stderr").write_text("")

    def analyze(trace, log, validator):
        data = json.loads((log.parent / "bytes/RESULT.json").read_text())
        pid = json.loads((log.parent / "ORACLE-WAIT.json").read_text())["pid"]
        return {
            "pid": pid,
            "batches": [
                {
                    "trace_id": 1,
                    "part_id": i,
                    "backend": "direct",
                    "begin_ns": row["observation"]["started_ns"] + 10 + j * 1000,
                    "submitted_ns": row["observation"]["started_ns"] + 510 + j * 1000,
                    "drained_ns": row["observation"]["drained_ns"] - 10,
                }
                for i, row in enumerate(data["rounds"])
                for j in range(PLANES)
            ],
        }

    monkeypatch.setattr(observation, "analyze_engine", analyze)
    return root, cp, native, contract


def test_all_samples_and_independent_pairs_are_retained(cohort):
    root, cp, native, contract = cohort
    row = result(contract)
    row["rounds"][0]["observation"]["drained_ns"] += 500_000
    row["rounds"][4]["observation"]["drained_ns"] += 100_000
    path = root / contract["cells"][0]["name"] / "bytes/RESULT.json"
    path.write_text(json.dumps(row))
    data = observation.collect(root, cp, native)
    assert data["status"] == "PILOT_OVERHEAD_PASS"
    assert data["cells"][0]["elapsed_ns"] == [400_000, 400_000, 400_000, 500_000]
    assert data["pairs"]["independent_pairs"] == 4
    assert (
        "requests are not replicates" in data["pairs"]["bootstrap_unit"]
        or "process pair" in data["pairs"]["bootstrap_unit"]
    )


def test_one_failed_pair_blocks_further_contention(cohort):
    root, cp, native, contract = cohort
    cell = contract["cells"][1]
    (root / cell["name"] / "bytes/RESULT.json").write_text(json.dumps(result(contract, 1.08)))
    data = observation.collect(root, cp, native)
    assert data["status"] == "PILOT_OVERHEAD_FAIL"
    assert data["eligible_for_separate_contention_contract"] is False
    assert data["pairs"]["ratios"][0] == pytest.approx(1.08)


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "plane",
        "corruption",
        "phase",
        "clock",
        "terminal",
        "epoch",
        "operation",
        "unregister",
        "ownership",
    ],
)
def test_incomplete_byte_or_native_boundaries_cannot_qualify(cohort, fault):
    _, _, _, contract = cohort
    data = result(contract)
    row = data["rounds"][2]
    if fault == "short":
        data["rounds"].pop()
    elif fault == "plane":
        row["plane_sha256"][-1] = "f" * 64
    elif fault == "corruption":
        row["negative_controls"].pop()
    elif fault == "phase":
        row["observation"]["phase"] = "warmup"
    elif fault == "clock":
        row["observation"]["drained_ns"] = row["observation"]["started_ns"]
    elif fault == "terminal":
        row["observation"]["terminal_done"] = False
    elif fault == "epoch":
        row["observation"]["session_epoch"] = 4
    elif fault == "operation":
        row["observation"]["operation_id"] = 1
    elif fault == "unregister":
        data["unregistered"] = False
    elif fault == "ownership":
        row["manager_after"][next(iter(REQUIRED_OWNERSHIP_GAUGES))] = 1
    with pytest.raises(ValueError):
        observation.validate_observations(data, contract)


@pytest.mark.parametrize(
    "fault", ["mode", "wait", "signal", "owner", "compiler", "baseline_capture"]
)
def test_bad_runtime_and_process_identity_cannot_form_a_pair(cohort, fault):
    root, cp, native, contract = cohort
    path = root / contract["cells"][0]["name"]
    if fault == "compiler":
        native.write_bytes(b"different compiler")
    elif fault == "baseline_capture":
        (path / "driver").mkdir()
    elif fault == "mode":
        file = path / "CELL.json"
        data = json.loads(file.read_text())
        data["variant_env"] = contract["variant_envs"]["on"]
        file.write_text(json.dumps(data))
    else:
        file = path / "ORACLE-WAIT.json"
        data = json.loads(file.read_text())
        if fault == "wait":
            data["normal_wait"] = False
        elif fault == "signal":
            data["exit_code"] = -9
        elif fault == "owner":
            data["pid"] = 101
            data["start_ticks"] = 1001
        file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        observation.collect(root, cp, native)


@pytest.mark.parametrize("fault", ["pid", "short", "outside"])
def test_driver_capture_must_cover_this_process_and_operation(cohort, monkeypatch, fault):
    root, cp, native, _ = cohort
    original = observation.analyze_engine

    def wrong(*args):
        value = copy.deepcopy(original(*args))
        if fault == "pid":
            value["pid"] += 1
        elif fault == "short":
            value["batches"].pop()
        else:
            value["batches"][0]["begin_ns"] = 1
        return value

    monkeypatch.setattr(observation, "analyze_engine", wrong)
    with pytest.raises(ValueError):
        observation.collect(root, cp, native)
