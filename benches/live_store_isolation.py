"""Matched local Manager latency under quiet or metadata-only stream pressure."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from tests.integration.test_distributed_cache import (
    _cleanup_dram,
    _discover_storage_namespaces,
    _etcd_keys,
    _metadata,
    _owner_status,
    _query_ready,
    _until,
    _wait_for_ssd_write,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

from .artifacts import external_path
from .live_store_measurements import (
    MEASUREMENT_CONTRACT,
    _clock_domain,
    _payload,
    _pressure_exposure,
    _process_sample,
    _summary,
)
from .observer_polling import OBSERVER_PROTOCOL, ObserverSession
from .scoped_metadata import _etcd_revision

PRESSURE_TEST = "cluster::inventory::tests::pressure::external_metadata_only_pressure_source"
DIAGNOSTIC_CONTRACT = "s2.10-isolation-diagnosis-v1"


def _channel_observation(observation):
    return {
        "request_id": observation.request_id,
        "session_epoch": observation.session_epoch,
        "session_token": observation.session_token,
        "submitted_mono_ns": observation.submitted_mono_ns,
        "returned_mono_ns": observation.returned_mono_ns,
    }


def _query_ready_diagnostic(client, instance, hashes, request, expected, managers):
    from orbitkv import BlockHashes, QueryReady

    batch = BlockHashes(hashes)
    observations = []

    def poll():
        result, observation = client.query_prefetch_diagnostic(instance, batch, request)
        observations.append(_channel_observation(observation))
        if not isinstance(result, QueryReady):
            return None
        if result.num_hit_blocks == expected:
            return result
        if result.lease:
            client.release(result.lease)
        if expected == 0:
            raise AssertionError(
                f"expected no hits for {request}, observed {result.num_hit_blocks}"
            )
        return None

    return _until(poll, managers), observations


def _collect_diagnostic_events(output, samples, medium):
    events = []
    for line in (output / "foreground-manager.log").read_text().splitlines():
        _, marker, payload = line.partition("cache_timeline ")
        if not marker:
            continue
        event, _ = json.JSONDecoder().raw_decode(payload)
        if "diagnostic_event" in event or event["stage"] == "diagnostic_timeline_limit":
            events.append(event)
    assert not any(event["stage"] == "diagnostic_timeline_limit" for event in events), events[-1:]

    by_operation = {}
    for event in events:
        key = (event["session_epoch"], event["session_token"], event["request_id"])
        by_operation.setdefault(key, []).append(event["stage"])

    save_required = {
        "publish_manager_receive",
        "publish_manager_process_start",
        "publish_manager_process_complete",
        "publish_manager_response_publish",
        "publish_storage_enqueue",
        "publish_storage_dequeue",
        "publish_storage_complete",
    }
    if medium == "ssd":
        save_required.update({"publish_ssd_enqueue", "publish_ssd_dequeue", "publish_ssd_complete"})
    save_operations = []
    query_operations = []
    for sample in samples:
        observation = sample["save_channel"]
        key = (
            observation["session_epoch"],
            observation["session_token"],
            observation["request_id"],
        )
        stages = set(by_operation.get(key, []))
        assert save_required <= stages, (key, save_required - stages)
        save_operations.append(key)
        for observation in sample["query_channels"]:
            key = (
                observation["session_epoch"],
                observation["session_token"],
                observation["request_id"],
            )
            stages = set(by_operation.get(key, []))
            assert {"query_manager_receive", "query_manager_process_complete"} <= stages, (
                key,
                stages,
            )
            query_operations.append(key)

    with (output / "diagnostic-events.jsonl").open("w") as event_file:
        for event in events:
            event_file.write(json.dumps(event) + "\n")
    return {
        "events": len(events),
        "save_operations": len(save_operations),
        "query_operations": len(query_operations),
        "unique_save_operations": len(set(save_operations)),
        "unique_query_operations": len(set(query_operations)),
        "stage_counts": {
            stage: sum(event["stage"] == stage for event in events)
            for stage in sorted({event["stage"] for event in events})
        },
    }


def _start_pressure_source(
    output: Path,
    endpoint: str,
    cluster: str,
    namespace: str,
    condition: str,
    rounds: int,
    cadence_ms: int,
    keys: int,
    active_keys: int,
    window_shift: int,
):
    binary = os.environ["ORBITKV_SERVER_TEST_BINARY"]
    go = output / "pressure.go"
    stop = output / "pressure.stop"
    environment = os.environ.copy()
    environment.update(
        {
            "ORBITKV_PRESSURE_ETCD_ENDPOINTS": endpoint,
            "ORBITKV_PRESSURE_CLUSTER": cluster,
            "ORBITKV_PRESSURE_NODE": "metadata-pressure",
            "ORBITKV_PRESSURE_NAMESPACE": namespace,
            "ORBITKV_PRESSURE_ARTIFACT_DIR": str(output),
            "ORBITKV_PRESSURE_GO_FILE": str(go),
            "ORBITKV_PRESSURE_STOP_FILE": str(stop),
            "ORBITKV_PRESSURE_MODE": condition,
            "ORBITKV_PRESSURE_ROUNDS": str(rounds),
            "ORBITKV_PRESSURE_CADENCE_MS": str(cadence_ms),
            "ORBITKV_PRESSURE_KEYS": str(keys),
            "ORBITKV_PRESSURE_ACTIVE_KEYS": str(active_keys),
            "ORBITKV_PRESSURE_WINDOW_SHIFT": str(window_shift),
        }
    )
    log = (output / "pressure-source.log").open("wb")
    process = subprocess.Popen(
        [binary, PRESSURE_TEST, "--exact", "--ignored", "--nocapture"],
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log, go, stop


def _stop_pressure_source(process, log, stop):
    stop.touch(exist_ok=True)
    if process.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)
    if process.poll() is None:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)
    log.close()


def run(
    output: Path,
    condition: str,
    medium: str,
    seed: str,
    warmup_rounds: int,
    samples: int,
    cadence_ms: int,
    pressure_keys: int,
    pressure_active_keys: int,
    pressure_window_shift: int,
    order_index: int,
    pressure_cadence_ms: int,
    observer_sample_ms: int,
    diagnostic_timeline_limit: int,
    observer_mode: str,
    observer_phase_offset_ms: float,
    foreground_phase_offset_ms: float,
    observer_start_lead_ms: int,
):
    if diagnostic_timeline_limit:
        os.environ["ORBITKV_TRACE_TRANSFERS"] = "1"
        os.environ["ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT"] = str(diagnostic_timeline_limit)
    elif os.environ.get("ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT", "0") != "0":
        raise ValueError("diagnostic timeline requires --diagnostic-timeline-limit")
    import torch

    import orbitkv
    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    identity = f"s2.10:isolation:{medium}:{seed}"
    installed_python_path = str(Path(native.__file__).resolve().parents[1])
    manager_python_paths = (installed_python_path,)
    local_namespace, pressure_namespace = _discover_storage_namespaces(
        output,
        [identity, "s2.10:isolation:metadata-only-pressure:v1"],
        pages,
        block_bytes,
        manager_python_paths=manager_python_paths,
    )
    assert local_namespace != pressure_namespace
    pressure_rounds = math.ceil((warmup_rounds + samples + 1) * cadence_ms / pressure_cadence_ms)
    observer_poll_count = math.ceil((warmup_rounds + samples + 1) * cadence_ms / observer_sample_ms)
    cluster = f"s210-isolation-{uuid.uuid4().hex[:12]}"
    result = {
        "measurement_contract": (
            DIAGNOSTIC_CONTRACT if diagnostic_timeline_limit else MEASUREMENT_CONTRACT
        ),
        "payload_schema": "u64-generation-u64-block-le-v1",
        "condition": condition,
        "pressure_profile": f"sustained-{pressure_cadence_ms}ms-v1",
        "medium": medium,
        "seed": seed,
        "warmup_rounds": warmup_rounds,
        "samples": samples,
        "cadence_ms": cadence_ms,
        "cluster": cluster,
        "local_namespace": local_namespace,
        "pressure_namespace": pressure_namespace,
        "pressure_keys": pressure_keys,
        "pressure_active_keys": pressure_active_keys,
        "pressure_window_shift": pressure_window_shift,
        "pressure_rounds": pressure_rounds,
        "pressure_cadence_ms": pressure_cadence_ms,
        "observer_sample_ms": observer_sample_ms,
        "observer_mode": observer_mode,
        "observer_contract": OBSERVER_PROTOCOL,
        "observer_poll_count": observer_poll_count,
        "observer_phase_offset_ms": observer_phase_offset_ms,
        "foreground_phase_offset_ms": foreground_phase_offset_ms,
        "observer_start_lead_ms": observer_start_lead_ms,
        "order_index": order_index,
        "budgets": {
            "dram_bytes": 67108864,
            "ssd_bytes": 67108864 if medium == "ssd" else 0,
            "index_bytes": 16777216,
            "journal_bytes": 16777216,
            "coalescing_ms": 2,
        },
        "latency_endpoints": {
            "payload_prepare_ms": "tensor copy through CUDA synchronization",
            "manager_save_submit_ms": "CacheManagerClient.save call only",
            "save_to_query_ready_ms": "save call start through local QueryReady",
            "local_query_ms": "query/prefetch call through QueryReady; excludes restore",
            "save_post_return_ms": "last native save returned_mono_ns through Python outer timer",
            "query_post_return_ms": "last native query returned_mono_ns through Python outer timer",
            "destination_prepare_ms": "destination zero through CUDA synchronization",
            "restore_complete_ms": "start_restore through native wait_restore completion",
            "gpu_consumable_ms": "start_restore through final CUDA synchronization",
        },
        "ssd_read_counter": "orbitkv_ssd_prefetch_bytes_total (io_uring reader completed bytes)",
        "scope": (
            "same-host A100, one full foreground Manager plus one test-owned "
            "production inventory-stream source and a fixed-location observer, forced TCP"
        ),
        "python_runtime": {
            "client_package": str(Path(orbitkv.__file__).resolve()),
            "client_extension": str(Path(native.__file__).resolve()),
            "manager_python_paths": list(manager_python_paths),
        },
        "diagnostic": {
            "enabled": bool(diagnostic_timeline_limit),
            "timeline_limit": diagnostic_timeline_limit,
            "clock": "CLOCK_MONOTONIC",
            "perf_sched_available": shutil.which("perf") is not None,
            "thread_schedstat": "optional per-thread /proc schedstat snapshot deltas",
        },
    }
    artifacts = {
        "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
        "server_tests": os.environ["ORBITKV_SERVER_TEST_BINARY"],
        "etcd": os.environ["ETCD_BIN"],
        "extension": native.__file__,
        "tent": str(Path(os.environ["ORBITKV_MOONCAKE_LIB_DIR"]) / "libtent_shared.so"),
    }
    result["artifacts"] = {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in artifacts.items()
    }

    raw = (output / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        port = find_available_port()
        ssd_enabled = medium == "ssd"
        manager = CacheManagerProcess(
            port,
            pool_size="64mb",
            http_port=find_available_port(),
            bootstrap_socket=f"/tmp/orbitkv-s210-isolation-{port}.sock",
            ssd_cache_path=output / "foreground-ssd" if ssd_enabled else None,
            ssd_cache_capacity="64mb",
            ssd_backend="uring",
            ssd_read_path="uring" if ssd_enabled else None,
            log_path=output / "foreground-manager.log",
            extra_args=(
                "--etcd-endpoints",
                endpoint,
                "--node-id",
                "foreground",
                "--cluster-name",
                cluster,
                "--index-budget",
                "16mb",
                "--inventory-journal-bytes",
                "16777216",
                "--membership-ttl-secs",
                "120",
                "--metadata-namespace",
                pressure_namespace,
                "--inventory-stream-coalesce-ms",
                "2",
                *(() if ssd_enabled else ("--enable-prometheus",)),
            ),
            runtime_python_paths=manager_python_paths,
        )
        stack.callback(manager.stop)
        assert manager.start(), manager.read_logs()
        client = CacheManagerClient(manager.bootstrap_socket)
        stack.callback(client.close)
        tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
        client.start_session_watcher("foreground", identity, 1, 1)
        registration = client.register_context_batch(
            "foreground",
            identity,
            0,
            0,
            1,
            1,
            resolve_device_id(),
            ["kv:0"],
            [pages],
            [block_bytes],
            [0],
            [1],
            "direct",
            False,
            tensors=[tensor],
        )

        pressure, pressure_log, go, stop = _start_pressure_source(
            output,
            endpoint,
            cluster,
            pressure_namespace,
            condition,
            pressure_rounds,
            pressure_cadence_ms,
            pressure_keys,
            pressure_active_keys,
            pressure_window_shift,
        )
        stack.callback(_stop_pressure_source, pressure, pressure_log, stop)
        ready_path = output / "pressure-source-ready.json"
        _until(
            lambda: ready_path.exists() or pressure.poll() is not None,
            [manager],
            timeout=30,
        )
        assert pressure.poll() is None, (output / "pressure-source.log").read_text()
        ready = json.loads(ready_path.read_text())
        owner = ready["owner"]["incarnation"]
        _until(
            lambda: (
                (status := _owner_status(manager, owner))
                and status["fresh"]
                and status["applied_sequence"] >= ready["initial_sequence"]
            ),
            [manager],
            timeout=30,
        )
        metadata = _metadata(manager)
        assert metadata["stream"]["scope_digest"] == ready["scope_digest"]
        result["foreground_binding"] = {
            "node": metadata["stream"]["source_node"],
            "epoch": metadata["stream"]["source_node_epoch"],
            "incarnation": metadata["stream"]["source_incarnation"],
            "scope_digest": metadata["stream"]["scope_digest"],
        }
        result["pressure_binding"] = ready
        initial_view = _owner_status(manager, owner)["view_id"]
        observer = ObserverSession(
            mode=observer_mode,
            endpoint=f"http://127.0.0.1:{manager.http_port}",
            incarnation=owner,
            expected_view=initial_view,
            output=output,
            cadence_ns=observer_sample_ms * 1_000_000,
            poll_count=observer_poll_count,
        )
        stack.callback(observer.abort)
        observer_ready = observer.launch()
        result["observer_ready"] = observer_ready
        observed_pids = [os.getpid(), manager.process.pid, pressure.pid]
        if observer.pid is not None:
            observed_pids.append(observer.pid)
        result["clock_domain"] = _clock_domain(
            [manager.process.pid, pressure.pid, *([observer.pid] if observer.pid else [])]
        )

        latency_names = [
            "payload_prepare_ms",
            "cleanup_ms",
            "manager_save_submit_ms",
            "save_to_query_ready_ms",
            "local_query_ms",
            "destination_prepare_ms",
            "restore_complete_ms",
            "gpu_consumable_ms",
        ]
        if diagnostic_timeline_limit:
            latency_names.extend(("save_post_return_ms", "query_post_return_ms"))
        latencies = {name: [] for name in latency_names}
        ssd_written = 0
        payload_oracle = hashlib.sha256()
        key_oracle = hashlib.sha256()
        resource_samples = (output / "resources.jsonl").open("w", buffering=1)
        stack.callback(resource_samples.close)

        def foreground_round(
            round_id,
            measured,
            scheduled_seconds,
            actual_seconds,
            scheduled_mono_ns,
        ):
            nonlocal ssd_written
            round_start_mono_ns = time.monotonic_ns()
            if round_id > 0:
                cleanup_started = time.monotonic_ns()
                cleaned = _cleanup_dram(manager)
                cleanup_ms = (time.monotonic_ns() - cleanup_started) / 1_000_000
                assert cleaned["evicted_blocks"] == (0 if ssd_enabled else pages), cleaned
            else:
                cleanup_ms = 0.0
            expected = _payload(torch, pages, block_bytes, round_id + 1)
            hashes = [
                hashlib.sha256(
                    f"s2.10:isolation:{medium}:{seed}:{round_id}:{block}".encode()
                ).digest()
                for block in range(pages)
            ]
            prepare_started = time.monotonic_ns()
            tensor.copy_(expected)
            torch.cuda.synchronize()
            payload_prepare_ms = (time.monotonic_ns() - prepare_started) / 1_000_000
            save_started = time.monotonic_ns()
            save_channel = None
            if diagnostic_timeline_limit:
                ok, message, observation = client.save_diagnostic(
                    registration,
                    [("kv:0", list(range(pages)), hashes)],
                )
                save_channel = _channel_observation(observation)
            else:
                ok, message = client.save(
                    registration,
                    [("kv:0", list(range(pages)), hashes)],
                )
            save_completed = time.monotonic_ns()
            manager_save_submit_ms = (save_completed - save_started) / 1_000_000
            assert ok, message
            if ssd_enabled:
                ssd_written += payload_bytes
                _wait_for_ssd_write(manager, ssd_written, [manager])
                cleaned = _cleanup_dram(manager)
                assert cleaned["evicted_blocks"] == pages
            metrics_before = fetch_orbitkv_metrics(manager.http_port)
            query_started = time.monotonic_ns()
            if diagnostic_timeline_limit:
                query, query_channels = _query_ready_diagnostic(
                    client,
                    "foreground",
                    hashes,
                    f"isolation-{condition}-{round_id}",
                    pages,
                    [manager],
                )
            else:
                query = _query_ready(
                    client,
                    "foreground",
                    hashes,
                    f"isolation-{condition}-{round_id}",
                    pages,
                    [manager],
                )
                query_channels = []
            query_completed = time.monotonic_ns()
            destination_started = query_completed
            tensor.zero_()
            torch.cuda.synchronize()
            destination_completed = time.monotonic_ns()
            restore_started = destination_completed
            operation = client.start_restore(
                "foreground",
                0,
                tensor.device.index or 0,
                [["kv:0"]],
                [(query.lease, [list(range(pages))])],
                ready_stream=torch.cuda.current_stream(tensor.device).cuda_stream,
            )
            status = client.wait_restore(operation, timeout=20)
            restore_completed = time.monotonic_ns()
            assert status.success, status
            torch.cuda.synchronize()
            gpu_completed = time.monotonic_ns()
            assert torch.equal(tensor, expected)
            metrics_after = fetch_orbitkv_metrics(manager.http_port)
            read_delta = metrics_after.get(
                "orbitkv_ssd_prefetch_bytes_total", 0
            ) - metrics_before.get("orbitkv_ssd_prefetch_bytes_total", 0)
            if ssd_enabled:
                assert read_delta >= payload_bytes, metrics_after
            else:
                assert read_delta == 0, metrics_after
            assert metrics_after.get("orbitkv_remote_fetch_bytes_total", 0) == 0
            sample = {
                "round": round_id,
                "round_start_mono_ns": round_start_mono_ns,
                "save_start_mono_ns": save_started,
                "save_end_mono_ns": save_completed,
                "query_start_mono_ns": query_started,
                "query_end_mono_ns": query_completed,
                "gpu_completed_mono_ns": gpu_completed,
                "measured": measured,
                "scheduled_seconds": scheduled_seconds,
                "actual_seconds": actual_seconds,
                "scheduled_foreground_start_mono_ns": scheduled_mono_ns,
                "foreground_start_deviation_ns": round_start_mono_ns - scheduled_mono_ns,
                "ssd_read_bytes": read_delta,
                "keys_sha256": hashlib.sha256(b"".join(hashes)).hexdigest(),
                "payload_prepare_ms": payload_prepare_ms,
                "cleanup_ms": cleanup_ms,
                "manager_save_submit_ms": manager_save_submit_ms,
                "save_to_query_ready_ms": (query_completed - save_started) / 1_000_000,
                "local_query_ms": (query_completed - query_started) / 1_000_000,
                "destination_prepare_ms": (destination_completed - destination_started) / 1_000_000,
                "restore_complete_ms": (restore_completed - restore_started) / 1_000_000,
                "gpu_consumable_ms": (gpu_completed - restore_started) / 1_000_000,
                "payload_sha256": hashlib.sha256(expected.cpu().numpy().tobytes()).hexdigest(),
            }
            if diagnostic_timeline_limit:
                assert save_started <= save_channel["submitted_mono_ns"]
                assert save_channel["returned_mono_ns"] <= save_completed
                assert query_channels
                assert query_channels[-1]["returned_mono_ns"] <= query_completed
                sample["save_channel"] = save_channel
                sample["query_channels"] = query_channels
                sample["save_post_return_ms"] = (
                    save_completed - save_channel["returned_mono_ns"]
                ) / 1_000_000
                sample["query_post_return_ms"] = (
                    query_completed - query_channels[-1]["returned_mono_ns"]
                ) / 1_000_000
            if measured:
                for name in latencies:
                    latencies[name].append(sample[name])
                payload_oracle.update(bytes.fromhex(sample["payload_sha256"]))
                key_oracle.update(bytes.fromhex(sample["keys_sha256"]))
            raw.write(json.dumps(sample) + "\n")

        epoch_mono_ns = time.monotonic_ns() + observer_start_lead_ms * 1_000_000
        observer_start = observer.start(
            epoch_mono_ns=epoch_mono_ns,
            observer_phase_offset_ns=round(observer_phase_offset_ms * 1_000_000),
            foreground_phase_offset_ns=round(foreground_phase_offset_ms * 1_000_000),
        )
        result["observer_start"] = observer_start
        go.touch()
        started_path = output / "pressure-started.json"
        _until(lambda: started_path.exists(), [manager])
        result["pressure_start"] = json.loads(started_path.read_text())
        foreground_epoch_ns = epoch_mono_ns + observer_start["foreground_phase_offset_ns"]
        result["schedule"] = {
            "epoch_mono_ns": epoch_mono_ns,
            "observer_first_scheduled_mono_ns": epoch_mono_ns
            + observer_start["observer_phase_offset_ns"],
            "foreground_first_scheduled_mono_ns": foreground_epoch_ns,
            "missed_deadline_policy": observer_start["missed_deadline_policy"],
        }
        result["foreground_started_unix_ns"] = time.time_ns()
        before_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        measured_started = None
        for round_id in range(warmup_rounds + samples):
            scheduled = round_id * cadence_ms / 1000
            scheduled_mono_ns = foreground_epoch_ns + round_id * cadence_ms * 1_000_000
            remaining_ns = scheduled_mono_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1_000_000_000)
            if round_id == warmup_rounds:
                process_before = _process_sample(observed_pids)
                metadata_before = _metadata(manager)
                measured_started = time.monotonic()
            assert pressure.poll() is None, (output / "pressure-source.log").read_text()
            observer.check()
            foreground_round(
                round_id,
                round_id >= warmup_rounds,
                scheduled,
                (time.monotonic_ns() - foreground_epoch_ns) / 1_000_000_000,
                scheduled_mono_ns,
            )
            resource_metadata = _metadata(manager)
            assert resource_metadata["index"]["accounted_bytes"] <= result["budgets"]["index_bytes"]
            assert (
                resource_metadata["inventory_journal_bytes"] <= result["budgets"]["journal_bytes"]
            )
            resource_samples.write(
                json.dumps(
                    {
                        "round": round_id,
                        "process": _process_sample(observed_pids),
                        "metadata": resource_metadata,
                    }
                )
                + "\n"
            )
        foreground_end_ns = foreground_epoch_ns + (warmup_rounds + samples) * cadence_ms * 1_000_000
        remaining_ns = foreground_end_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000)
        wall_seconds = time.monotonic() - measured_started
        process_after = _process_sample(observed_pids)
        result_path = output / "pressure-source-result.json"
        _until(lambda: result_path.exists(), [manager])
        pressure_result = json.loads(result_path.read_text())
        total_rounds = pressure_rounds
        expected_changes = (
            total_rounds * pressure_window_shift * 2 if condition == "pressure" else 0
        )
        assert pressure_result["changes"] == expected_changes
        assert pressure_result["owner"] == ready["owner"]
        assert pressure_result["final_sequence"] == ready["initial_sequence"] + expected_changes
        _until(
            lambda: (
                (row := _owner_status(manager, owner))
                and row["fresh"]
                and row["applied_sequence"] == pressure_result["final_sequence"]
            ),
            [manager],
        )
        final_owner = _owner_status(manager, owner)
        assert final_owner["records"] == pressure_active_keys
        final_records = pressure_result["final_records"]
        inserted = total_rounds * pressure_window_shift if condition == "pressure" else 0
        expected_records = []
        for offset in range(pressure_active_keys):
            key = (inserted + offset) % pressure_keys
            last_insert = (
                inserted - 1 - ((inserted - 1 - (key - pressure_active_keys)) % pressure_keys)
            )
            sequence = (
                key + 1
                if last_insert < 0
                else (
                    pressure_active_keys
                    + (last_insert // pressure_window_shift) * 2 * pressure_window_shift
                    + pressure_window_shift
                    + last_insert % pressure_window_shift
                    + 1
                )
            )
            expected_records.append((key, sequence))
        actual_records = []
        for record in final_records:
            assert record["key"]["namespace"] == pressure_namespace
            assert record["present"]
            assert record["metadata"] == {
                "medium": "Dram",
                "representation": "Raw",
                "stored_bytes": 4096,
            }
            actual_records.append(
                (int.from_bytes(bytes(record["key"]["hash"]), "little"), record["sequence"])
            )
        assert sorted(actual_records) == sorted(expected_records)
        metadata_after = _metadata(manager)
        assert metadata_after["index"]["coverage"] == "complete_at_watermarks"
        assert (
            metadata_after["stream"]["source_incarnation"]
            == result["foreground_binding"]["incarnation"]
        )
        after_revision = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        assert before_revision == after_revision
        result["etcd_revision_delta"] = after_revision - before_revision
        result["payload_oracle_sha256"] = payload_oracle.hexdigest()
        result["key_oracle_sha256"] = key_oracle.hexdigest()
        result["observer_exit"] = observer.finish()
        validation_error = None
        try:
            result["exposure"] = _pressure_exposure(
                output,
                condition,
                pressure_rounds,
                pressure_cadence_ms,
                cadence_ms,
                pressure_window_shift,
                observer_sample_ms,
            )
        except AssertionError as error:
            validation_error = error
            result["validation_error"] = str(error)
            if (output / "exposure.json").exists():
                result["exposure"] = json.loads((output / "exposure.json").read_text())
        stop.touch()
        pressure_exit = pressure.wait(timeout=30)
        pressure_log.close()
        assert pressure_exit == 0, (output / "pressure-source.log").read_text()
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)

        def local_drained():
            metrics = fetch_orbitkv_metrics(manager.http_port)
            if any(
                metrics.get(name, 0)
                for name in (
                    "orbitkv_query_reserved_bytes",
                    "orbitkv_ssd_read_pinned_bytes",
                    "orbitkv_transfer_completion_outstanding",
                )
            ):
                return None
            return metrics

        final_metrics = _until(local_drained, [manager])
        result["final_drain_metrics"] = final_metrics
        ok, message = client.unregister_context("foreground")
        assert ok, message
        client.close()
        registration = None
        tensor = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        exit_code, shutdown_seconds = manager.terminate_gracefully(timeout=10)
        assert exit_code == 0 and shutdown_seconds <= 10
        result.update(
            {
                "status": "passed" if validation_error is None else "invalid_measurement",
                "wall_seconds": wall_seconds,
                "achieved_samples_per_second": samples / wall_seconds,
                "latencies": {name: _summary(values) for name, values in latencies.items()},
                "process_before": process_before,
                "process_after": process_after,
                "metadata_before": metadata_before,
                "metadata_after": metadata_after,
                "pressure_result": pressure_result,
                "final_owner": final_owner,
                "etcd_keys": keys,
                "pressure_exit": pressure_exit,
                "manager_exit": exit_code,
                "manager_shutdown_seconds": shutdown_seconds,
            }
        )
        if diagnostic_timeline_limit:
            raw.flush()
            diagnostic_samples = [
                json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            result["diagnostic"].update(
                _collect_diagnostic_events(output, diagnostic_samples, medium)
            )
        (output / "isolation-result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        if validation_error is not None:
            raise validation_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("quiet", "pressure"), required=True)
    parser.add_argument("--medium", choices=("dram", "ssd"), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--warmup-rounds", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--cadence-ms", type=int, default=1000)
    parser.add_argument("--pressure-cadence-ms", type=int, default=17)
    parser.add_argument("--observer-sample-ms", type=int, default=25)
    parser.add_argument(
        "--observer-mode",
        choices=("in-process", "helper-process"),
        default="in-process",
    )
    parser.add_argument("--observer-phase-offset-ms", type=float, default=0.0)
    parser.add_argument("--foreground-phase-offset-ms", type=float, default=12.5)
    parser.add_argument("--observer-start-lead-ms", type=int, default=500)
    parser.add_argument("--diagnostic-timeline-limit", type=int, default=0)
    parser.add_argument("--pressure-keys", type=int, default=2048)
    parser.add_argument("--pressure-active-keys", type=int, default=1536)
    parser.add_argument("--pressure-window-shift", type=int, default=128)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds cannot be negative")
    if args.samples <= 0 or args.cadence_ms <= 0:
        parser.error("--samples and --cadence-ms must be positive")
    if args.order_index < 0:
        parser.error("--order-index cannot be negative")
    if args.pressure_cadence_ms <= 0 or args.observer_sample_ms <= 0:
        parser.error("pressure and observer cadences must be positive")
    if args.observer_phase_offset_ms < 0 or args.foreground_phase_offset_ms < 0:
        parser.error("observer and foreground phase offsets cannot be negative")
    if args.observer_start_lead_ms < 100:
        parser.error("--observer-start-lead-ms must be at least 100")
    if not 0 <= args.diagnostic_timeline_limit <= 65_536:
        parser.error("diagnostic timeline limit must be between 0 and 65536")
    if args.pressure_active_keys <= args.pressure_window_shift:
        parser.error("pressure active keys must exceed the window shift")
    if args.pressure_keys < args.pressure_active_keys + args.pressure_window_shift:
        parser.error("pressure keys must cover the active window plus one shift")
    for variable in (
        "ETCD_BIN",
        "ORBITKV_CACHE_MANAGER_BINARY",
        "ORBITKV_MOONCAKE_LIB_DIR",
        "ORBITKV_SERVER_TEST_BINARY",
    ):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(
            args.output,
            args.condition,
            args.medium,
            args.seed,
            args.warmup_rounds,
            args.samples,
            args.cadence_ms,
            args.pressure_keys,
            args.pressure_active_keys,
            args.pressure_window_shift,
            args.order_index,
            args.pressure_cadence_ms,
            args.observer_sample_ms,
            args.diagnostic_timeline_limit,
            args.observer_mode,
            args.observer_phase_offset_ms,
            args.foreground_phase_offset_ms,
            args.observer_start_lead_ms,
        )
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
