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
        if observation.VARIANTS[mode]["driver"]
        else None,
        "observation_design": {"warmup": contract["warmup"], "samples": contract["samples"]},
        "rounds": rounds,
    }


@pytest.fixture(params=("pilot", "components"))
def cohort(tmp_path, monkeypatch, request):
    native = tmp_path / "native"
    native.write_bytes(b"Frozen compiled geometry control")
    contract = {
        "schema": "s4.copy-observation-components.v1"
        if request.param == "components"
        else "s4.copy-observation-pilot.v2",
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
    order = (
        observation.COMPONENT_ORDER if request.param == "components" else observation.PILOT_ORDER
    )
    width = 4 if request.param == "components" else 2
    contract["variant_envs"] = {mode: observation.VARIANTS[mode].copy() for mode in set(order)}
    for i, mode in enumerate(order):
        contract["cells"].append({"name": f"cell-{i}-{mode}", "mode": mode, "pair": i // width})
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
        if observation.VARIANTS[cell["mode"]]["driver"]:
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

    def native_collect(log, validator):
        value = analyze(None, log, validator)
        data = json.loads((log.parent / "bytes/RESULT.json").read_text())
        return {
            "native_geometry_validation": {"fixture_sha256": contract["native_fixture_sha256"]},
            "traces": [
                {
                    "header": {"pid": value["pid"], "trace_id": 1},
                    "headers": [
                        {"pid": value["pid"], "clock": "CLOCK_MONOTONIC"} for _ in data["rounds"]
                    ],
                    "parts": [
                        {"part_id": i, "drained_ns": row["observation"]["drained_ns"] - 10}
                        for i, row in enumerate(data["rounds"])
                    ],
                    "batches": value["batches"],
                }
            ],
        }

    def driver_capture(trace):
        data = json.loads((trace.parent.parent / "bytes/RESULT.json").read_text())
        pid = json.loads((trace.parent.parent / "ORACLE-WAIT.json").read_text())["pid"]
        calls = [
            {
                "name": "cuMemcpyHtoDAsync_v2",
                "result": 0,
                "begin_ns": row["observation"]["started_ns"] + 10,
                "end_ns": row["observation"]["started_ns"] + 100,
            }
            for row in data["rounds"]
        ]
        return pid, len(calls), {7: calls}

    monkeypatch.setattr(observation, "collect_native", native_collect)
    monkeypatch.setattr(observation, "_driver_capture", driver_capture)
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
    cell = next(cell for cell in contract["cells"] if cell["mode"] == "on")
    (root / cell["name"] / "bytes/RESULT.json").write_text(
        json.dumps(
            result(contract, 1.08, offset=contract["cells"].index(cell) * 1_000_000_000, mode="on")
        )
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
    message = (
        "compiler differs from frozen profile"
        if fault == "compiler_after_first_hash"
        else "changed during cohort analysis"
    )
    with pytest.raises(ValueError, match=message):
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
    i = (
        0
        if fault == "off-loaded"
        else next(i for i, cell in enumerate(contract["cells"]) if cell["mode"] == "on")
    )
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


@pytest.mark.parametrize("cohort", ["components"], indirect=True)
@pytest.mark.parametrize(
    "fault",
    [
        "native_pid",
        "driver_pid",
        "driver_window",
        "driver_result",
        "driver_direction",
        "driver_after_enqueued",
        "driver_crossing_start",
        "driver_crossing_end",
        "missing_operation",
        "native_when_disabled",
    ],
)
def test_component_cost_requires_actual_mode_and_operation_coverage(cohort, monkeypatch, fault):
    root, cp, native, contract = cohort
    if fault == "native_when_disabled":
        cell = next(cell for cell in contract["cells"] if cell["mode"] == "driver")
        (root / cell["name"] / "oracle.stderr").write_text("local_raw_copy {}\n")
    elif fault == "native_pid":
        original = observation.collect_native

        def change(*args):
            value = copy.deepcopy(original(*args))
            value["traces"][0]["headers"][0]["pid"] += 1
            return value

        monkeypatch.setattr(observation, "collect_native", change)
    else:
        original = observation._driver_capture

        def change(*args):
            pid, calls, threads = copy.deepcopy(original(*args))
            if fault == "driver_pid":
                pid += 1
            elif fault == "driver_window":
                threads[7][0]["begin_ns"] = 1
            elif fault == "driver_result":
                threads[7][0]["result"] = 1
            elif fault == "driver_direction":
                threads[7][0]["name"] = "cuMemcpyDtoHAsync_v2"
            elif fault == "driver_after_enqueued":
                for event in threads[7]:
                    event["begin_ns"] += 200_000
                    event["end_ns"] += 200_000
            elif fault == "driver_crossing_start":
                event = threads[7][0].copy()
                event["begin_ns"] -= 20
                event["name"] = "cuMemcpyDtoHAsync_v2"
                threads[7].append(event)
            elif fault == "driver_crossing_end":
                event = threads[7][0].copy()
                event["begin_ns"] += 399_980
                event["end_ns"] += 400_000
                threads[7].append(event)
            else:
                threads[7].pop()
            return pid, calls, threads

        monkeypatch.setattr(observation, "_driver_capture", change)
    with pytest.raises(ValueError):
        observation.collect(root, cp, native)


@pytest.mark.parametrize("cohort", ["components"], indirect=True)
def test_component_contrasts_do_not_gate_on_a_single_faster_component(cohort):
    root, cp, native, contract = cohort
    for index, cell in enumerate(contract["cells"]):
        factor = {"off": 1, "native": 1.01, "driver": 1.20, "on": 1.22}[cell["mode"]]
        (root / cell["name"] / "bytes/RESULT.json").write_text(
            json.dumps(
                result(contract, factor=factor, offset=index * 1_000_000_000, mode=cell["mode"])
            )
        )
    data = observation.collect(root, cp, native)
    assert data["status"] == "PILOT_OVERHEAD_FAIL"
    assert data["eligible_for_separate_contention_contract"] is False
    assert set(data["component_contrasts"]) == {"native", "driver", "on"}
    assert data["component_contrasts"]["native"]["geometric_mean_ratio"] == pytest.approx(1.01)
    assert data["component_contrasts"]["driver"]["geometric_mean_ratio"] == pytest.approx(1.20)
    assert data["pairs"]["geometric_mean_ratio"] == pytest.approx(1.22)


@pytest.mark.parametrize("cohort", ["components"], indirect=True)
@pytest.mark.parametrize("fault", ["native_log", "driver_trace", "combined_trace", "compiler"])
def test_live_cell_capture_cannot_change_before_next_admission(cohort, monkeypatch, fault):
    root, cp, native, contract = cohort
    mode, owner = {
        "native_log": ("native", "collect_native"),
        "driver_trace": ("driver", "_driver_capture"),
        "combined_trace": ("on", "analyze_engine"),
        "compiler": ("native", "collect_native"),
    }[fault]
    cell = next(cell for cell in contract["cells"] if cell["mode"] == mode)
    path = root / cell["name"]
    wait = json.loads((path / "ORACLE-WAIT.json").read_text())
    data = json.loads((path / "bytes/RESULT.json").read_text())
    changed = (
        native
        if fault == "compiler"
        else path / "oracle.stderr"
        if fault == "native_log"
        else path / "driver" / f"cuda-driver-{wait['pid']}.jsonl"
    )
    original = getattr(observation, owner)

    def modify_after_parse(*args):
        value = original(*args)
        changed.write_bytes(b"changed after actual parser return")
        return value

    monkeypatch.setattr(observation, owner, modify_after_parse)
    with pytest.raises(ValueError, match="changed during cell validation"):
        observation.validate_capture(path, data, wait, contract, native, mode)


@pytest.mark.parametrize("cohort", ["components"], indirect=True)
@pytest.mark.parametrize(
    "fault", [None, "duration_as_clock", "early", "late", "missing", "float", "clock"]
)
def test_native_only_consumes_absolute_part_completion_from_real_parser(cohort, monkeypatch, fault):
    from benches.local_restore_shapes import _collect
    from benches.tests.test_local_restore_shapes import records

    root, _, native, contract = cohort
    cell = next(cell for cell in contract["cells"] if cell["mode"] == "native")
    path = root / cell["name"]
    wait = json.loads((path / "ORACLE-WAIT.json").read_text())
    data = json.loads((path / "bytes/RESULT.json").read_text())
    events = []
    for part, row in enumerate(data["rounds"]):
        header, batch, drained, footer = records()
        for event in (header, batch, drained, footer):
            event.update(pid=wait["pid"], sequence=part, clock="CLOCK_MONOTONIC")
        reserved = 2 * PLANES + 1
        header.update(limit=65536, records_before=part * reserved)
        if fault == "clock" and part == 1:
            header["clock"] = "CLOCK_REALTIME"
        events.append(header)
        for plane in range(PLANES):
            item = copy.deepcopy(batch)
            item.update(
                part_id=part,
                batch_id=plane,
                begin_ns=row["observation"]["started_ns"] + 10 + plane * 1000,
                submitted_ns=row["observation"]["started_ns"] + 510 + plane * 1000,
            )
            item["copies"][0]["device"] += plane * 65536
            item["copies"][0]["registered_device_start"] += plane * 65536
            events.append(item)
        drained.update(
            part_id=part,
            batches=PLANES,
            planned_descriptors=PLANES,
            submitted_descriptors=PLANES,
            host_drain_ns=10,
            drained_ns=row["observation"]["drained_ns"] - 10,
        )
        if part == 0:
            if fault == "duration_as_clock":
                drained["drained_ns"] = drained["host_drain_ns"]
            elif fault == "early":
                drained["drained_ns"] = row["observation"]["started_ns"] + 1000
            elif fault == "late":
                drained["drained_ns"] = row["observation"]["drained_ns"] + 1
            elif fault == "missing":
                drained.pop("drained_ns")
            elif fault == "float":
                drained["drained_ns"] = float(drained["drained_ns"])
        footer.update(
            parts=part + 1,
            batches=(part + 1) * PLANES,
            descriptors=(part + 1) * PLANES,
            records_reserved=(part + 1) * reserved,
        )
        events.extend((drained, footer))
    (path / "oracle.stderr").write_text(
        "".join("local_raw_copy " + json.dumps(event) + "\n" for event in events)
    )

    def native_collect(log, fixture):
        # This CPU contract test consumes real schema records; native compiler
        # execution is exercised separately by installed-artifact review.
        return _collect(log) | {"native_geometry_validation": {"fixture": str(fixture)}}

    monkeypatch.setattr(observation, "collect_native", native_collect)
    if fault is None:
        value = observation.validate_capture(path, data, wait, contract, native, "native")
        assert len(value["batches"]) == 3 * PLANES
        assert (
            value["batches"][0]["drained_ns"] == data["rounds"][0]["observation"]["drained_ns"] - 10
        )
    else:
        with pytest.raises(ValueError, match="clock|windows|drained_ns"):
            observation.validate_capture(path, data, wait, contract, native, "native")


@pytest.mark.parametrize(
    "fault", ["extra-directory", "extra-file", "extra-symlink", "cell-symlink", "root-symlink"]
)
def test_cell_census_rejects_foreign_or_linked_entries_before_replay(cohort, monkeypatch, fault):
    root, cp, native, contract = cohort
    if fault == "extra-directory":
        (root / "foreign").mkdir()
    elif fault == "extra-file":
        (root / "foreign").write_text("unlisted evidence")
    elif fault == "extra-symlink":
        (root / "foreign").symlink_to(root / "missing")
    elif fault == "cell-symlink":
        cell = root / contract["cells"][0]["name"]
        target = root.parent / "moved-cell"
        cell.rename(target)
        cell.symlink_to(target, target_is_directory=True)
    else:
        target = root.parent / "moved-cohort"
        root.rename(target)
        root.symlink_to(target, target_is_directory=True)

    def unexpected_replay(*args, **kwargs):
        pytest.fail("Native replay ran before cell census was validated")

    monkeypatch.setattr(observation, "validate_capture", unexpected_replay)
    with pytest.raises(ValueError, match="cell directory"):
        observation.collect(root, cp, native)


def test_cell_added_during_replay_cannot_qualify(cohort, monkeypatch):
    root, cp, native, _ = cohort
    original = observation.validate_capture

    def add_cell(*args, **kwargs):
        (root / "foreign").mkdir(exist_ok=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(observation, "validate_capture", add_cell)
    with pytest.raises(ValueError, match="cell directory inventory"):
        observation.collect(root, cp, native)
