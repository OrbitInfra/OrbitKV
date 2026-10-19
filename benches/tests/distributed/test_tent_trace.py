"""Reject incomplete native evidence and preserve concurrent stage timing."""

import copy
import hashlib
import json

import pytest

from benches.distributed import tent_trace


def trace():
    events = [
        {"stage": stage, "begin_mono_ns": 100, "end_mono_ns": 300, "tid": index + 1, "status": 0}
        for index, stage in enumerate(sorted(tent_trace.STAGES))
    ]
    events.append({**events[0], "begin_mono_ns": 200, "end_mono_ns": 400, "tid": 5})
    return {
        "schema": "tent.native-trace.v1",
        "clock": "CLOCK_MONOTONIC",
        "pid": 42,
        "boot_id": "0f7f8389-3d2e-424f-84b8-c52d59f20908",
        "start_ticks": 111,
        "limit": 4096,
        "seen": len(events),
        "overflow": False,
        "events": events,
    }


def test_stage_union_does_not_add_concurrent_bootstrap_calls():
    sample = dict(
        zip(
            ("begin_mono_ns", "submit_return_mono_ns", "terminal_mono_ns", "freed_mono_ns"),
            (150, 160, 350, 360),
            strict=True,
        )
    )
    summary = tent_trace.summarize(trace(), "consumer", 42, sample)
    assert summary["stages"]["bootstrap_rpc"]["overlap_union_ms"] == 300 / 1e6
    cold = summary["cold_read"]["stages"]["bootstrap_rpc"]
    assert cold["calls"] == 2
    assert cold["overlap_union_ms"] == 210 / 1e6
    assert cold["duration_max_ms"] == 160 / 1e6


@pytest.mark.parametrize(
    ("field", "value"),
    [("pid", 43), ("clock", "foreign clock"), ("overflow", True), ("seen", 4), ("limit", 8192)],
)
def test_rejects_wrong_process_clock_overflow_and_missing_events(field, value):
    evidence = trace()
    evidence[field] = value
    with pytest.raises(ValueError):
        tent_trace.summarize(evidence, "consumer", 42)


@pytest.mark.parametrize(
    "mutation",
    [
        {"stage": "unknown"},
        {"status": 1},
        {"begin_mono_ns": 301},
        {"end_mono_ns": 300.0},
        {"tid": 0},
    ],
)
def test_rejects_failed_or_malformed_native_intervals(mutation):
    evidence = trace()
    evidence["events"][1].update(mutation)
    with pytest.raises(ValueError):
        tent_trace.summarize(evidence, "consumer", 42)


def test_source_cannot_borrow_consumer_clock_or_claim_full_decomposition():
    evidence = trace()
    evidence["events"] = [row for row in evidence["events"] if row["stage"] == "endpoint_construct"]
    evidence["seen"] = len(evidence["events"])
    assert (
        tent_trace.summarize(evidence, "source", 42)["stages"]["endpoint_construct"]["calls"] == 1
    )
    with pytest.raises(ValueError, match="required stage"):
        tent_trace.summarize(evidence, "consumer", 42)
    sample = {
        "begin_mono_ns": 0,
        "submit_return_mono_ns": 1,
        "terminal_mono_ns": 2,
        "freed_mono_ns": 3,
    }
    with pytest.raises(ValueError, match="source trace"):
        tent_trace.summarize(evidence, "source", 42, sample)
    with pytest.raises(ValueError, match="do not overlap"):
        tent_trace.summarize(copy.deepcopy(trace()), "consumer", 42, sample)


@pytest.mark.parametrize("exit_code", [0, 1, -15])
def test_cli_requires_actual_zero_exit_even_with_a_drained_stop(tmp_path, monkeypatch, exit_code):
    native = tmp_path / "native.json"
    native.write_text(json.dumps(trace()))
    stdout = tmp_path / "endpoint.stdout"
    sample = {
        "begin_mono_ns": 0,
        "submit_return_mono_ns": 1,
        "terminal_mono_ns": 500,
        "freed_mono_ns": 501,
    }
    records = [
        {"event": "ready", "pid": 42},
        {"event": "read_complete", "samples": [sample]},
        {"event": "stopped", "registered_regions": 0, "active_batches": 0},
    ]
    stdout.write_text("\n".join(json.dumps({"tent_stage_probe": True, **row}) for row in records))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tent_trace",
            "--trace",
            str(native),
            "--endpoint-stdout",
            str(stdout),
            "--endpoint-exit-code",
            str(exit_code),
            "--role",
            "consumer",
            "--output",
            str(output),
        ],
    )
    if exit_code:
        with pytest.raises(ValueError, match="must exit zero"):
            tent_trace.main()
        assert not output.exists()
    else:
        tent_trace.main()
        assert json.loads(output.read_text())["pid"] == 42


def manager_cell():
    boot = "0f7f8389-3d2e-424f-84b8-c52d59f20908"
    ready = [
        {
            "host": "h1",
            "name": name,
            "cell": "cold-warm",
            "pid": pid,
            "start_ticks": 111 if name == "manager" else 222,
            "boot_id": boot,
            "exit_code": None,
            "actual_wait_reap": False,
        }
        for name, pid in (("manager", 42), ("fixture", 43))
    ]
    return {
        "status": "PASS_DESCRIPTIVE_PROFILE",
        "cell": {"id": "cold-warm", "target_host": "h1", "source_host": "h0"},
        "before": {"h1": {"boot_id": boot}},
        "after": {"h1": {"boot_id": boot}},
        "readiness": ready,
        "exits": [
            {
                **row,
                "exit_code": 0 if row["name"] == "manager" else 143,
                "actual_wait_reap": True,
                "forced_cleanup": False,
            }
            for row in ready
        ],
        "evidence": {
            "status": "PASS_DESCRIPTIVE_PROFILE",
            "samples": [
                {
                    "iteration": iteration,
                    "phase": phase,
                    "query_id": f"q-{iteration}",
                    "timing": {"query_ms": (end - begin) / 1e6},
                    "query_window": {
                        "clock": "CLOCK_MONOTONIC",
                        "boot_id": boot,
                        "pid": 43,
                        "start_ticks": 222,
                        "instance": "consumer",
                        "namespace": "profile",
                        "begin_mono_ns": begin,
                        "ready_mono_ns": end,
                    },
                }
                for iteration, (phase, begin, end) in enumerate(
                    (("cold", 150, 500), ("warmup", 1000, 1100), ("measured", 2000, 2100))
                )
            ],
        },
    }


def test_manager_windows_link_distinct_local_actor_and_allow_zero_warm_setup(tmp_path, monkeypatch):
    cell = manager_cell()
    # PIDs on another host are independent and must not select the trace owner.
    cell["readiness"].append({**cell["readiness"][0], "host": "h0", "pid": 900})
    summary = tent_trace.summarize_manager(trace(), cell)
    assert summary["pid"] == 42 and summary["host"] == "h1"
    cold, warmup, measured = summary["client_query_windows"]
    assert cold["actor"]["pid"] == 43
    assert cold["stages"]["bootstrap_rpc"]["overlap_union_ms"] == 250 / 1e6
    for window in (warmup, measured):
        assert all(row["calls"] == 0 for row in window["stages"].values())
    assert "cold_read" not in summary
    native = tmp_path / "trace.json"
    native.write_text(json.dumps(trace()))
    result = tmp_path / "cell.json"
    result.write_text(json.dumps(cell))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tent_trace",
            "--trace",
            str(native),
            "--manager-cell",
            str(result),
            "--role",
            "consumer",
            "--output",
            str(output),
        ],
    )
    tent_trace.main()
    assert json.loads(output.read_text()) == summary


@pytest.mark.parametrize(
    "section,index,field,value",
    [
        ("readiness", 0, "host", "h0"),
        ("readiness", 0, "cell", "old"),
        ("readiness", 0, "boot_id", "old-boot"),
        ("readiness", 0, "actual_wait_reap", True),
        ("exits", 0, "pid", 99),
        ("exits", 0, "start_ticks", 110),
        ("exits", 0, "exit_code", False),
        ("exits", 0, "exit_code", -9),
        ("exits", 1, "exit_code", 0),
        ("exits", 1, "actual_wait_reap", False),
        ("exits", 1, "forced_cleanup", True),
    ],
)
def test_manager_trace_rejects_foreign_or_unreaped_owner(section, index, field, value):
    cell = manager_cell()
    cell[section][index][field] = value
    with pytest.raises(ValueError):
        tent_trace.summarize_manager(trace(), cell)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("query_window", "pid", 42),
        ("query_window", "start_ticks", 111),
        ("query_window", "boot_id", "old-boot"),
        ("query_window", "namespace", "another"),
        ("query_window", "begin_mono_ns", 200),
        ("sample", "query_id", "q-0"),
        ("sample", "phase", "cold"),
        ("sample", "iteration", 0),
    ],
)
def test_manager_trace_rejects_foreign_or_repeated_query_window(section, field, value):
    cell = manager_cell()
    sample = cell["evidence"]["samples"][1]
    target = sample if section == "sample" else sample[section]
    target[field] = value
    window = sample["query_window"]
    sample["timing"]["query_ms"] = (window["ready_mono_ns"] - window["begin_mono_ns"]) / 1e6
    with pytest.raises(ValueError):
        tent_trace.summarize_manager(trace(), cell)


@pytest.mark.parametrize("field,value", [("boot_id", "old-boot"), ("start_ticks", 110)])
def test_manager_trace_cannot_reuse_a_pid_from_another_boot_or_process(field, value):
    native = trace()
    native[field] = value
    with pytest.raises(ValueError, match="boot and process start"):
        tent_trace.summarize_manager(native, manager_cell())


def observer_pilot():
    contract = {
        "observer_pilot": "manager-native-observer.v1",
        "warmup": 5,
        "samples": 50,
        "hosts": {"h0": {}, "h1": {}},
        "required_idle_metrics": ["cpu_control_idle"],
        "launches": {"h0": {}, "h1": {}},
        "cells": [],
    }
    results = []
    for index in range(8):
        source, target = ("h0", "h1") if index < 4 else ("h1", "h0")
        mode = ("off", "on", "on", "off")[index % 4]
        cell = {
            "id": f"c{index + 1}-{mode}",
            "ordinal": index,
            "kind": "manager",
            "source_host": source,
            "target_host": target,
            "pair_id": f"{source}-{target}-p{1 + index % 4 // 2}",
            "observer": mode,
            "payload": 8 << 20,
            "segment_bytes": 65536,
        }
        contract["cells"].append(cell)
        result = manager_cell()
        result.update(cell=cell, readiness=[], exits=[], native_traces={}, runtime_admission={})
        boot = trace()["boot_id"]
        for host_index, host in enumerate((source, target)):
            contract["launches"][host][cell["id"]] = {}
            for name, offset in (("manager", 0), ("fixture", 1)):
                enabled = mode == "on" and name == "manager"
                planned = {
                    "kind": name,
                    "runtime": {"observer_enabled": enabled},
                    "environment": {"TENT_NATIVE_TRACE_PATH": "/cpu-control/manager.json"}
                    if enabled
                    else {},
                }
                contract["launches"][host][cell["id"]][name] = planned
                ready = {
                    "host": host,
                    "cell": cell["id"],
                    "name": name,
                    "pid": 42 + index * 10 + host_index * 2 + offset,
                    "start_ticks": 111 + index * 10 + host_index * 2 + offset,
                    "boot_id": boot,
                    "exit_code": None,
                    "actual_wait_reap": False,
                }
                result["readiness"].append(ready)
                stopped = {
                    **ready,
                    "exit_code": 0 if name == "manager" else 143,
                    "actual_wait_reap": True,
                    "was_running": True,
                    "forced_cleanup": False,
                }
                result["exits"].append(stopped)
                if name == "manager":
                    native = trace()
                    native.update({key: ready[key] for key in ("pid", "boot_id", "start_ticks")})
                    content = json.dumps(native) if enabled else None
                    result["native_traces"][host] = {
                        "host": host,
                        "cell": cell["id"],
                        "observer_enabled": enabled,
                        "stop": stopped,
                        "content": content,
                        "sha256": hashlib.sha256(content.encode()).hexdigest() if enabled else None,
                    }
            result["before"][host] = result["after"][host] = {"boot_id": boot}
            result["runtime_admission"][host] = {
                "cell": cell["id"],
                "host": host,
                "owners": {
                    row["name"]: {key: row[key] for key in ("pid", "start_ticks")}
                    for row in result["readiness"]
                    if row["host"] == host
                },
            }
        actor = result["readiness"][-1]
        result["evidence"].update(payload_bytes=cell["payload"], samples=[])
        for iteration in range(56):
            begin = 150 + iteration * 10_000_000
            result["evidence"]["samples"].append(
                {
                    "iteration": iteration,
                    "phase": "cold"
                    if iteration == 0
                    else "warmup"
                    if iteration <= 5
                    else "measured",
                    "query_id": f"q-{iteration}",
                    "sha256": ["a" * 64] * 4,
                    "remote_bytes": cell["payload"],
                    "h2d_bytes": cell["payload"],
                    "manager_fetch_ms": 1.0,
                    "timing": {"query_ms": 1.0, "query_restore_ms": 2.0},
                    "query_window": {
                        "clock": "CLOCK_MONOTONIC",
                        "boot_id": boot,
                        "pid": actor["pid"],
                        "start_ticks": actor["start_ticks"],
                        "instance": "consumer",
                        "namespace": "profile",
                        "begin_mono_ns": begin,
                        "ready_mono_ns": begin + 1_000_000,
                    },
                }
            )
        result["pre_stop_idle"] = result["cleanup_idle"] = [{"cpu_control_idle": 0}] * 2
        results.append(result)
    return contract, results


def test_observer_overhead_uses_restarted_pairs_and_excludes_cold_and_warmup():
    contract, results = observer_pilot()
    for result in results:
        for sample in result["evidence"]["samples"][:6]:
            sample["manager_fetch_ms"] = 9000
            sample["timing"]["query_restore_ms"] = 10000
    report = tent_trace.observer_overhead(contract, results)
    assert report["status"] == "MEASUREMENT_GUARDS_PASS"
    assert len(report["pairs"]) == 4
    for direction in report["directions"].values():
        assert all(metric["pair_bootstrap_95_ci"] == [1.0, 1.0] for metric in direction.values())
    # A within-cell outlier affects the separate max guard, not the pair median.
    results[1]["evidence"]["samples"][-1]["manager_fetch_ms"] = 1.26
    report = tent_trace.observer_overhead(contract, results[:2])
    assert report["status"] == "OVERHEAD_FAIL"
    assert report["pairs"][0]["metrics"]["manager_fetch_ms"]["median_ratio"] == 1
    assert report["pairs"][0]["metrics"]["manager_fetch_ms"]["tail_difference_ms"] == pytest.approx(
        0.26
    )


@pytest.mark.parametrize(
    "mutation",
    ["reordered", "short", "foreign_trace", "off_trace", "local_hit", "nonfinite", "wrong_mode"],
)
def test_observer_pilot_rejects_invalid_prefix_instead_of_dropping_cells(mutation):
    contract, results = observer_pilot()
    if mutation == "reordered":
        results[0], results[1] = results[1], results[0]
    elif mutation == "short":
        results[0]["evidence"]["samples"].pop()
    elif mutation == "foreign_trace":
        receipt = results[1]["native_traces"]["h0"]
        native = json.loads(receipt["content"])
        native["start_ticks"] += 1
        receipt.update(
            content=json.dumps(native),
            sha256=hashlib.sha256(json.dumps(native).encode()).hexdigest(),
        )
    elif mutation == "off_trace":
        results[0]["native_traces"]["h0"]["content"] = "{}"
    elif mutation == "local_hit":
        results[0]["evidence"]["samples"][0]["remote_bytes"] = 0
    elif mutation == "nonfinite":
        results[0]["evidence"]["samples"][-1]["manager_fetch_ms"] = float("nan")
    else:
        contract["launches"]["h0"][contract["cells"][0]["id"]]["manager"]["runtime"][
            "observer_enabled"
        ] = True
    with pytest.raises(ValueError):
        tent_trace.observer_overhead(contract, results)


@pytest.mark.parametrize("failure", ["trace", "overhead"])
def test_campaign_fail_stop_blocks_next_cell_and_finishes_known_owners(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    from benches.distributed import rdma_profile

    contract, results = observer_pilot()
    contract.update(
        source_hashes={},
        owner_commands={"h0": ["cpu-only"], "h1": ["cpu-only"]},
        etcd_url="http://cpu-control",
    )
    contract["launches"]["h0"]["campaign"] = {"etcd": {"argv": [], "environment": {}}}
    for sample in results[1]["evidence"]["samples"][6:]:
        sample["manager_fetch_ms"] = 1.2
    path = tmp_path / "CONTRACT.json"
    raw = json.dumps(contract).encode()
    path.write_bytes(raw)
    inputs = json.dumps({path.name: hashlib.sha256(raw).hexdigest()}).encode()
    (tmp_path / "INPUTS.json").write_bytes(inputs)
    (tmp_path / "LAUNCH-ACCEPTED.json").write_text(
        json.dumps({"accepted": True, "inputs_sha256": hashlib.sha256(inputs).hexdigest()})
    )
    launched, closed = [], []

    class CPUProtocolControl:
        uncertain = finished = False
        process = None

        def __init__(self, command, host, output):
            self.host, self.command = host, command

        def start(self):
            pass

        def rpc(self, operation, **fields):
            assert operation in ("launch", "stop")
            return {"exit_code": -15}

        def finish(self):
            closed.append(self.host)
            self.finished = True
            return {"cpu_only_control": True}

    def cell(self, cell, output):
        launched.append(cell["id"])
        if failure == "trace":
            raise ValueError("controlled invalid trace after cell cleanup")
        return results[cell["ordinal"]]

    monkeypatch.setattr(rdma_profile, "HostConnection", CPUProtocolControl)
    monkeypatch.setattr(rdma_profile.Campaign, "manager_cell", cell)
    monkeypatch.setattr(rdma_profile.requests, "get", lambda *a, **kw: SimpleNamespace(ok=True))
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        "sys.argv", ["rdma_profile", "--contract", str(path), "--output", str(output)]
    )
    with pytest.raises((RuntimeError, ValueError)):
        rdma_profile.main()
    status = json.loads((output / "STATUS.json").read_text())
    assert len(launched) == (1 if failure == "trace" else 2)
    assert closed == ["h0", "h1"]
    assert status["cleanup_etcd_exit"]["exit_code"] == -15
    assert status["status"] == (
        "INVALID_OR_FAILED_FAIL_STOP" if failure == "trace" else "VALID_OBSERVER_OVERHEAD_FAIL"
    )
