"""Malformed observations and reused processes cannot qualify diagnostic overhead."""

import copy
import hashlib
import json

import pytest

from benches import copy_observation as observation
from benches.metrics import DRAIN_GAUGES, REQUIRED_OWNERSHIP_GAUGES
from benches.sglang_kv_oracle import PAGE_BYTES, PAGES, PLANES, page_maps


def result(contract, factor=1, offset=0, mode="off"):
    sources, targets = page_maps(4096, 0)
    payload = len(sources) * PAGE_BYTES * PLANES
    rounds = []
    for index in range(contract["warmup"] + contract["samples"]):
        start = 10_000_000_000 + offset + 1_000_000 + index * 1_000_000
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
        "driver_capture": {
            "loader": "explicit-dlopen",
            "pid": 100 + offset // 1_000_000_000,
            "library": contract["collector"],
            "sha256": contract["collector_sha256"],
        }
        if mode == "on"
        else None,
        "observation_design": {"warmup": contract["warmup"], "samples": contract["samples"]},
        "rounds": rounds,
    }


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.write_bytes(b"Frozen compiled geometry control")
    contract = {
        "schema": "s4.copy-observation-pilot.v2",
        "warmup": 1,
        "samples": 2,
        "length": 4096,
        "max_ratio": 1.05,
        "seed": 20260920,
        "model_config_sha256": "a" * 64,
        "gpu_uuid": "GPU-26b6f4ae-dd04-cfdd-3e78-0150a85ae9fb",
        "native_fixture_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "collector": str(tmp_path / "collector.so"),
        "collector_sha256": "c" * 64,
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
        put(
            path / "bytes/RESULT.json",
            result(contract, offset=i * 1_000_000_000, mode=cell["mode"]),
        )
        (path / "oracle.stderr").write_text("")
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
                "start_ticks": 1000 + i * 100,
                "launched_mono_ns": 10_000_000_000 + i * 1_000_000_000,
                "waited_mono_ns": 10_000_000_000 + i * 1_000_000_000 + 4_000_000,
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
                "stop_started_mono_ns": 10_000_000_000 + i * 1_000_000_000 + 5_000_000,
                "stop_finished_mono_ns": 10_000_000_000 + i * 1_000_000_000 + 6_000_000,
            },
        )
        if cell["mode"] == "on":
            (path / "driver").mkdir()
            (path / "driver" / f"cuda-driver-{100 + i}.jsonl").write_text(
                "Frozen capture fixture\n"
            )

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
    row["rounds"][2]["observation"]["drained_ns"] += 100_000
    path = root / contract["cells"][0]["name"] / "bytes/RESULT.json"
    path.write_text(json.dumps(row))
    data = observation.collect(root, cp, native)
    assert data["status"] == "PILOT_OVERHEAD_PASS"
    assert data["cells"][0]["elapsed_ns"] == [400_000, 500_000]
    assert data["pairs"]["independent_pairs"] == 4
    assert (
        "requests are not replicates" in data["pairs"]["bootstrap_unit"]
        or "process pair" in data["pairs"]["bootstrap_unit"]
    )


def test_one_failed_pair_blocks_further_contention(cohort):
    root, cp, native, contract = cohort
    cell = contract["cells"][1]
    (root / cell["name"] / "bytes/RESULT.json").write_text(
        json.dumps(result(contract, 1.08, offset=1_000_000_000, mode="on"))
    )
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
    "fault",
    [
        "mode",
        "wait",
        "signal",
        "owner",
        "compiler",
        "baseline_capture",
        "order",
        "shutdown",
        "trace_off",
    ],
)
def test_bad_runtime_and_process_identity_cannot_form_a_pair(cohort, fault):
    root, cp, native, contract = cohort
    path = root / contract["cells"][0]["name"]
    if fault == "compiler":
        native.write_bytes(b"different compiler")
    elif fault == "baseline_capture":
        (path / "driver").mkdir()
    elif fault == "trace_off":
        (path / "oracle.stderr").write_text('cache_timeline {"stage":"local_raw_copy_open"}\n')
    elif fault == "shutdown":
        file = path / "manager.cleanup.json"
        data = json.loads(file.read_text())
        data["stop_started_mono_ns"] = 1
        data["stop_finished_mono_ns"] = 2
        file.write_text(json.dumps(data))
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
            data["start_ticks"] = 1100
        elif fault == "order":
            data["start_ticks"] = 1200
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


def test_consumed_byte_evidence_cannot_change_during_analysis(cohort, monkeypatch):
    root, cp, native, contract = cohort
    original = observation.analyze_engine
    path = root / contract["cells"][0]["name"] / "bytes/RESULT.json"

    def mutate(*args):
        data = json.loads(path.read_text())
        data["rounds"][0]["observation"]["terminal_success"] = False
        path.write_text(json.dumps(data))
        return original(*args)

    monkeypatch.setattr(observation, "analyze_engine", mutate)
    with pytest.raises(ValueError, match="evidence changed"):
        observation.collect(root, cp, native)


@pytest.mark.parametrize("fault", ["compiler_after_first_hash", "late_off_driver"])
def test_initial_compiler_binding_and_disabled_inventory_survive_analysis(
    cohort, monkeypatch, fault
):
    root, cp, native, contract = cohort
    if fault == "compiler_after_first_hash":
        original = observation._sha256
        changed = False

        def replace_after_hash(path):
            nonlocal changed
            digest = original(path)
            if path == native and not changed:
                native.write_bytes(b"Replaced after initial compiler validation")
                changed = True
            return digest

        monkeypatch.setattr(observation, "_sha256", replace_after_hash)
    else:
        original = observation.analyze_engine

        def add_capture(*args):
            (root / contract["cells"][0]["name"] / "driver").mkdir(exist_ok=True)
            return original(*args)

        monkeypatch.setattr(observation, "analyze_engine", add_capture)
    with pytest.raises(ValueError, match="changed during cohort analysis"):
        observation.collect(root, cp, native)


def test_file_integrity_checks_work_without_python311_file_digest(cohort, monkeypatch):
    root, cp, native, _ = cohort
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert observation.collect(root, cp, native)["status"] == "PILOT_OVERHEAD_PASS"


@pytest.mark.parametrize(
    "fault", ["foreign-zero-call", "missing-main", "symlink-main", "symlink-dir", "off-dangling"]
)
def test_per_cell_capture_census_rejects_invalid_files_before_analysis(tmp_path, fault):
    directory = tmp_path / "driver"
    directory.mkdir()
    trace = directory / "cuda-driver-123.jsonl"
    trace.write_text("main capture")
    if fault == "foreign-zero-call":
        (directory / "cuda-driver-124.jsonl").write_text(
            '{"stage":"cuda_driver_complete","pid":124,"calls":0,"complete":true}\n'
        )
    elif fault == "missing-main":
        trace.unlink()
    elif fault == "symlink-main":
        target = tmp_path / "external.jsonl"
        trace.rename(target)
        trace.symlink_to(target)
    elif fault == "symlink-dir":
        target = tmp_path / "external-driver"
        directory.rename(target)
        directory.symlink_to(target, target_is_directory=True)
    else:
        trace.unlink()
        directory.rmdir()
        directory.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="capture"):
        observation.capture_file(tmp_path, "off" if fault == "off-dangling" else "on", 123)


@pytest.mark.parametrize(
    "fault", ["legacy-loader", "wrong-pid", "wrong-library", "wrong-sha", "off-loaded"]
)
def test_frozen_collector_activation_is_consumed(cohort, fault):
    root, cp, native, contract = cohort
    i = 0 if fault == "off-loaded" else 1
    path = root / contract["cells"][i]["name"] / "bytes/RESULT.json"
    value = json.loads(path.read_text())
    if fault == "off-loaded":
        value["driver_capture"] = {"loader": "explicit-dlopen"}
    else:
        field, changed = {
            "legacy-loader": ("loader", "LD_PRELOAD"),
            "wrong-pid": ("pid", 999),
            "wrong-library": ("library", "/other/collector.so"),
            "wrong-sha": ("sha256", "d" * 64),
        }[fault]
        value["driver_capture"][field] = changed
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="activation"):
        observation.collect(root, cp, native)
