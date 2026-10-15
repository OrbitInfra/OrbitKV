"""Two-Manager DRAM/io_uring live-store fault soak with exact GPU-byte oracles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import requests

from tests.integration.test_distributed_cache import (
    _await_fence,
    _cleanup_dram,
    _discover_storage_namespaces,
    _etcd_keys,
    _metadata,
    _owner_status,
    _payload,
    _query_ready,
    _restore,
    _sync,
    _until,
    _wait_for_remote_drain,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import TcpGate, etcd_server
from tests.support.metrics import fetch_orbitkv_metrics

from .artifacts import external_path
from .live_store_measurements import _process_sample, _summary

FULL_DURATION_SECONDS = 2 * 60 * 60
FAULTS = (
    (10 * 60, "peer_partition"),
    (12 * 60, "peer_heal"),
    (25 * 60, "slow_subscriber"),
    (27 * 60, "slow_release"),
    (40 * 60, "receiver_restart"),
    (55 * 60, "overflow_partition"),
    (58 * 60, "overflow_heal"),
    (75 * 60, "source_restart"),
    (95 * 60, "source_etcd_isolate"),
    (98 * 60, "source_etcd_heal_restart"),
    (115 * 60, "final_reconcile"),
)


def _etcd_json(endpoint: str, key: str):
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={"key": base64.b64encode(key.encode()).decode()},
        timeout=5,
    )
    response.raise_for_status()
    rows = response.json().get("kvs", [])
    assert len(rows) <= 1, rows
    return json.loads(base64.b64decode(rows[0]["value"])) if rows else None


def run(output: Path, duration_seconds: int, schedule_scale: float):
    import torch

    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes = 8, 4096
    payload_bytes = pages * block_bytes
    identities = [f"s2.10:soak:{label}" for label in ("dram", "ssd", "outside")]
    scopes = _discover_storage_namespaces(output, identities, pages, block_bytes)
    selected_scopes = scopes[:2]
    cluster = f"s210-soak-{uuid.uuid4().hex[:12]}"
    result = {
        "duration_seconds": duration_seconds,
        "schedule_scale": schedule_scale,
        "cluster": cluster,
        "storage_namespaces": scopes,
        "selected_scopes": selected_scopes,
        "topology": "same-host A100, two full Managers, forced TCP",
        "fault_schedule_seconds": [
            {"at": at * schedule_scale, "action": action} for at, action in FAULTS
        ],
    }
    artifacts = {
        "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
        "etcd": os.environ["ETCD_BIN"],
        "extension": native.__file__,
        "tent": str(Path(os.environ["ORBITKV_MOONCAKE_LIB_DIR"]) / "libtent_shared.so"),
    }
    result["artifacts"] = {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in artifacts.items()
    }

    raw = (output / "samples.jsonl").open("w", buffering=1)
    timeline = (output / "fault-timeline.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        stack.callback(timeline.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        source_etcd_gate = TcpGate(endpoint)
        stack.callback(source_etcd_gate.close)
        source_port = find_available_port()
        source_peer_gate = TcpGate(f"http://127.0.0.1:{source_port}")
        stack.callback(source_peer_gate.close)
        ports = {"source": source_port, "consumer": find_available_port()}
        runs = {"source": 0, "consumer": 0}
        managers = {}
        clients = {}
        tensors = {}
        gpu_registrations = {}
        registrations = {}
        membership_history = []

        def member_key(node):
            return f"/orbitkv/v2/{cluster}/members/{node}"

        def manager_args(node):
            args = [
                "--etcd-endpoints",
                source_etcd_gate.endpoint if node == "source" else endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "12",
                "--inventory-journal-bytes",
                str(256 * 1024),
                "--inventory-stream-coalesce-ms",
                "2",
                "--index-budget",
                "16mb",
                "--metadata-namespace",
                selected_scopes[0],
                "--metadata-namespace",
                selected_scopes[1],
            ]
            if node == "source":
                args.extend(["--peer-advertise-addr", source_peer_gate.endpoint.rsplit("//", 1)[1]])
            else:
                args.append("--enable-prometheus")
            return tuple(args)

        def start_node(node):
            previous = registrations.get(node)
            assert _etcd_json(endpoint, member_key(node)) is None
            run_id = runs[node]
            runs[node] += 1
            manager = CacheManagerProcess(
                ports[node],
                pool_size="96mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s210-soak-{ports[node]}.sock",
                ssd_cache_path=output / "source-ssd" if node == "source" else None,
                ssd_cache_capacity="128mb",
                ssd_backend="uring",
                ssd_read_path="uring" if node == "source" else None,
                query_budget="32mb",
                log_path=output / f"{node}-manager-{run_id}.log",
                extra_args=manager_args(node),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers[node] = manager
            current = _until(
                lambda: _etcd_json(endpoint, member_key(node)), list(managers.values())
            )
            if previous is not None:
                assert current["epoch"] > previous["epoch"]
                assert current["owner"]["incarnation"] != previous["owner"]["incarnation"]
            registrations[node] = current
            membership_history.append(
                {"node": node, "run": run_id, "previous": previous, "current": current}
            )
            for label, identity in zip(("dram", "ssd", "outside"), identities, strict=True):
                client = CacheManagerClient(manager.bootstrap_socket)
                clients[(node, label)] = client
                stack.callback(client.close)
                tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
                tensors[(node, label)] = tensor
                instance = f"{node}-{label}"
                client.start_session_watcher(instance, identity, 1, 1)
                gpu_registrations[(node, label)] = client.register_context_batch(
                    instance,
                    identity,
                    0,
                    0,
                    1,
                    1,
                    resolve_device_id(),
                    ["kv:0"],
                    [serialize_gpu_buffer(tensor)],
                    [pages],
                    [block_bytes],
                    [0],
                    [1],
                    "direct",
                    False,
                    tensors=[tensor],
                )
            return previous, current

        def close_node_clients(node):
            for label in ("dram", "ssd", "outside"):
                client = clients.get((node, label))
                if client is not None:
                    ok, message = client.unregister_context(f"{node}-{label}")
                    assert ok, message
                    client.close()
                    clients.pop((node, label))
                    gpu_registrations.pop((node, label))
                tensors.pop((node, label), None)
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()

        def stop_node(node):
            close_node_clients(node)
            exit_code, seconds = managers[node].terminate_gracefully(timeout=10)
            assert exit_code == 0
            assert _etcd_json(endpoint, member_key(node)) is None
            return seconds

        start_node("consumer")
        start_node("source")
        current_hashes = {}
        current_payloads = {}
        async_visibility = []
        barrier_visibility = []
        save_latency = []
        local_query_latency = []
        quiet_save_latency = []
        pressure_save_latency = []
        quiet_query_latency = []
        pressure_query_latency = []
        remote_dram_bytes = 0
        remote_ssd_bytes = 0
        ssd_read_bytes = 0
        wrong_bytes = 0
        stale_authorizations = 0
        expected_remote = True
        peer_partitioned = False
        source_isolated = False
        slow_subscriber = False
        fault_index = 0
        schedule = [(at * schedule_scale, action) for at, action in FAULTS]
        started = time.monotonic()
        process_before = _process_sample(manager.process.pid for manager in managers.values())
        revision_before = _metadata(managers["source"])["index"]["membership_revision"]

        def hashes(label, cycle):
            return [
                hashlib.sha256(f"s2.10:soak:{label}:{cycle}:{block}".encode()).digest()
                for block in range(pages)
            ]

        def reconcile(reason):
            nonlocal expected_remote
            fence = _sync(managers["source"])
            repair_started = time.monotonic()
            _await_fence(managers["consumer"], fence, list(managers.values()), timeout=30)
            elapsed_ms = (time.monotonic() - repair_started) * 1000
            expected_remote = True
            timeline.write(
                json.dumps(
                    {
                        "elapsed": time.monotonic() - started,
                        "action": reason,
                        "repair_ms": elapsed_ms,
                    }
                )
                + "\n"
            )
            return elapsed_ms

        repair_times = []
        shutdown_times = []

        def apply_fault(action):
            nonlocal expected_remote, peer_partitioned, slow_subscriber, source_isolated
            event = {"elapsed": time.monotonic() - started, "action": action}
            if action in ("peer_partition", "overflow_partition"):
                source_peer_gate.partition()
                peer_partitioned = True
                expected_remote = False
            elif action in ("peer_heal", "overflow_heal"):
                source_peer_gate.heal()
                peer_partitioned = False
                repair_times.append(reconcile(action))
            elif action == "slow_subscriber":
                source_peer_gate.set_delay(0.05)
                slow_subscriber = True
            elif action == "slow_release":
                source_peer_gate.set_delay(0)
                slow_subscriber = False
                repair_times.append(reconcile(action))
            elif action == "receiver_restart":
                old_owner = registrations["consumer"]["owner"]["incarnation"]
                seconds = stop_node("consumer")
                shutdown_times.append(seconds)
                previous, current = start_node("consumer")
                assert previous["owner"]["incarnation"] == old_owner
                event.update({"shutdown_seconds": seconds, "old": previous, "new": current})
                repair_times.append(reconcile(action))
            elif action == "source_restart":
                old_owner = registrations["source"]["owner"]["incarnation"]
                seconds = stop_node("source")
                shutdown_times.append(seconds)
                previous, current = start_node("source")
                assert previous["owner"]["incarnation"] == old_owner
                event.update({"shutdown_seconds": seconds, "old": previous, "new": current})
                expected_remote = False
            elif action == "source_etcd_isolate":
                source_etcd_gate.partition()
                source_isolated = True
            elif action == "source_etcd_heal_restart":
                source_etcd_gate.heal()
                _until(
                    lambda: _etcd_json(endpoint, member_key("source")) is None,
                    list(managers.values()),
                    timeout=30,
                )
                assert not _metadata(managers["source"])["index"]["registration_valid"]
                seconds = stop_node("source")
                shutdown_times.append(seconds)
                previous, current = start_node("source")
                event.update({"shutdown_seconds": seconds, "old": previous, "new": current})
                source_isolated = False
                expected_remote = False
            elif action == "final_reconcile":
                repair_times.append(reconcile(action))
            else:
                raise AssertionError(action)
            timeline.write(json.dumps(event) + "\n")

        cycle = 0
        while time.monotonic() - started < duration_seconds:
            cycle_started = time.monotonic()
            elapsed = cycle_started - started
            while fault_index < len(schedule) and elapsed >= schedule[fault_index][0]:
                apply_fault(schedule[fault_index][1])
                fault_index += 1

            source_manager = managers["source"]
            consumer_manager = managers["consumer"]
            source_valid = _metadata(source_manager)["index"]["registration_valid"]
            write_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                "orbitkv_ssd_write_bytes_total", 0
            )
            save_started = time.monotonic()
            for label, payload_round in (("dram", cycle * 2 + 1), ("ssd", cycle * 2 + 2)):
                block_hashes = hashes(label, cycle)
                payload = _payload(torch, pages, block_bytes, payload_round)
                current_hashes[label] = block_hashes
                current_payloads[label] = payload
                tensor = tensors[("source", label)]
                tensor.copy_(payload)
                torch.cuda.synchronize()
                ok, message = clients[("source", label)].save(
                    gpu_registrations[("source", label)],
                    [("kv:0", list(range(pages)), block_hashes)],
                )
                assert ok, message
            save_ms = (time.monotonic() - save_started) * 1000
            save_latency.append(save_ms)
            pressure_window = peer_partitioned or slow_subscriber
            (pressure_save_latency if pressure_window else quiet_save_latency).append(save_ms)
            expected_write = write_before + payload_bytes * 2
            _until(
                lambda source_manager=source_manager, expected_write=expected_write: (
                    metrics
                    if (metrics := fetch_orbitkv_metrics(source_manager.http_port)).get(
                        "orbitkv_ssd_write_bytes_total", 0
                    )
                    >= expected_write
                    and not any(
                        metrics.get(name, 0)
                        for name in (
                            "orbitkv_ssd_write_inflight",
                            "orbitkv_ssd_write_queue_pending",
                            "orbitkv_inflight_bytes",
                        )
                    )
                    else None
                ),
                list(managers.values()),
            )

            local_started = time.monotonic()
            local = _query_ready(
                clients[("source", "dram")],
                "source-dram",
                current_hashes["dram"],
                f"local-{cycle}",
                pages,
                list(managers.values()),
            )
            local_query_latency.append((time.monotonic() - local_started) * 1000)
            (pressure_query_latency if pressure_window else quiet_query_latency).append(
                local_query_latency[-1]
            )
            clients[("source", "dram")].release(local.lease)

            can_barrier = source_valid and not peer_partitioned and not source_isolated
            if can_barrier:
                visible_started = time.monotonic()
                if cycle % 2:
                    fence = _sync(source_manager)
                    _await_fence(consumer_manager, fence, list(managers.values()), timeout=30)
                else:
                    target = _metadata(source_manager)["inventory_sequence"]
                    source_owner = registrations["source"]["owner"]["incarnation"]
                    _until(
                        lambda target=target, source_owner=source_owner, consumer_manager=consumer_manager: (
                            (status := _owner_status(consumer_manager, source_owner))
                            and status["fresh"]
                            and status["applied_sequence"] >= target
                        ),
                        list(managers.values()),
                        timeout=30,
                    )
                visibility = (time.monotonic() - visible_started) * 1000
                (barrier_visibility if cycle % 2 else async_visibility).append(visibility)
                expected_remote = True
            else:
                visibility = None

            dram_remote = ssd_remote = False
            if expected_remote and source_valid:
                _cleanup_dram(consumer_manager)
                dram_ssd_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                before = fetch_orbitkv_metrics(consumer_manager.http_port).get(
                    "orbitkv_remote_fetch_bytes_total", 0
                )
                try:
                    _restore(
                        clients[("consumer", "dram")],
                        "consumer-dram",
                        tensors[("consumer", "dram")],
                        current_hashes["dram"],
                        f"remote-dram-{cycle}",
                        current_payloads["dram"],
                        list(managers.values()),
                    )
                    dram_remote = True
                except AssertionError:
                    wrong_bytes += 1
                    raise
                _wait_for_remote_drain(source_manager, consumer_manager, list(managers.values()))
                after = fetch_orbitkv_metrics(consumer_manager.http_port)[
                    "orbitkv_remote_fetch_bytes_total"
                ]
                dram_ssd_after = fetch_orbitkv_metrics(source_manager.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                assert after >= before + payload_bytes
                assert dram_ssd_after == dram_ssd_before
                remote_dram_bytes += after - before

            cleaned = _cleanup_dram(source_manager)
            assert cleaned["still_referenced_blocks"] == 0
            if expected_remote and source_valid:
                _cleanup_dram(consumer_manager)
                source_ssd_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                before = fetch_orbitkv_metrics(consumer_manager.http_port).get(
                    "orbitkv_remote_fetch_bytes_total", 0
                )
                _restore(
                    clients[("consumer", "ssd")],
                    "consumer-ssd",
                    tensors[("consumer", "ssd")],
                    current_hashes["ssd"],
                    f"remote-ssd-{cycle}",
                    current_payloads["ssd"],
                    list(managers.values()),
                )
                ssd_remote = True
                _wait_for_remote_drain(source_manager, consumer_manager, list(managers.values()))
                after = fetch_orbitkv_metrics(consumer_manager.http_port)[
                    "orbitkv_remote_fetch_bytes_total"
                ]
                source_ssd_after = fetch_orbitkv_metrics(source_manager.http_port)[
                    "orbitkv_ssd_prefetch_bytes_total"
                ]
                assert after >= before + payload_bytes
                assert source_ssd_after >= source_ssd_before + payload_bytes
                remote_ssd_bytes += after - before
                ssd_read_bytes += source_ssd_after - source_ssd_before
            elif not source_valid:
                miss = _query_ready(
                    clients[("consumer", "ssd")],
                    "consumer-ssd",
                    current_hashes["ssd"],
                    f"fenced-source-{cycle}",
                    0,
                    list(managers.values()),
                )
                if miss.lease:
                    stale_authorizations += 1
                    clients[("consumer", "ssd")].release(miss.lease)
                assert not miss.lease

            source_status = _metadata(source_manager)
            consumer_status = _metadata(consumer_manager)
            process = _process_sample(manager.process.pid for manager in managers.values())
            sample = {
                "cycle": cycle,
                "elapsed": time.monotonic() - started,
                "save_ms": save_ms,
                "local_query_ms": local_query_latency[-1],
                "visibility_ms": visibility,
                "source_valid": source_valid,
                "peer_partitioned": peer_partitioned,
                "slow_subscriber": slow_subscriber,
                "source_isolated": source_isolated,
                "coverage": consumer_status["index"]["coverage"],
                "dram_remote": dram_remote,
                "ssd_remote": ssd_remote,
                "source_index": source_status["index"],
                "consumer_index": consumer_status["index"],
                "source_stream": source_status["stream"],
                "consumer_stream": consumer_status["stream"],
                "journal_bytes": source_status["inventory_journal_bytes"],
                "journal_gaps": source_status["inventory_history_gaps"],
                "process": process,
            }
            raw.write(json.dumps(sample) + "\n")
            cycle += 1
            remaining = 1 - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)

        wall_seconds = time.monotonic() - started
        assert wall_seconds >= duration_seconds
        if peer_partitioned:
            source_peer_gate.heal()
        source_peer_gate.set_delay(0)
        if source_isolated:
            source_etcd_gate.heal()
        final_repair_ms = reconcile("soak_complete")
        repair_times.append(final_repair_ms)
        source_manager = managers["source"]
        consumer_manager = managers["consumer"]
        for label in ("dram", "ssd"):
            if label == "ssd":
                _cleanup_dram(source_manager)
            _cleanup_dram(consumer_manager)
            _restore(
                clients[("consumer", label)],
                f"consumer-{label}",
                tensors[("consumer", label)],
                current_hashes[label],
                f"final-{label}",
                current_payloads[label],
                list(managers.values()),
            )
            _wait_for_remote_drain(source_manager, consumer_manager, list(managers.values()))

        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        final_source = _metadata(source_manager)
        final_consumer = _metadata(consumer_manager)
        process_after = _process_sample(manager.process.pid for manager in managers.values())
        assert final_consumer["index"]["coverage"] == "complete_at_watermarks"
        assert wrong_bytes == 0
        assert stale_authorizations == 0
        quiet_save = _summary(quiet_save_latency)
        pressure_save = _summary(pressure_save_latency)
        quiet_query = _summary(quiet_query_latency)
        pressure_query = _summary(pressure_query_latency)
        save_regression = pressure_save["p99"] / quiet_save["p99"] - 1
        query_regression = pressure_query["p99"] / quiet_query["p99"] - 1
        source_metrics = fetch_orbitkv_metrics(source_manager.http_port)
        consumer_metrics = fetch_orbitkv_metrics(consumer_manager.http_port)
        assert not source_metrics.get("orbitkv_transfer_lock_active", 0)
        assert not consumer_metrics.get("orbitkv_transfer_completion_outstanding", 0)
        result.update(
            {
                "status": "passed",
                "wall_seconds": wall_seconds,
                "cycles": cycle,
                "membership_history": membership_history,
                "shutdown_seconds": shutdown_times,
                "repair_ms": _summary(repair_times),
                "async_visibility_ms": _summary(async_visibility),
                "barrier_visibility_ms": _summary(barrier_visibility),
                "save_latency_ms": _summary(save_latency),
                "local_query_latency_ms": _summary(local_query_latency),
                "quiet_save_latency_ms": quiet_save,
                "pressure_save_latency_ms": pressure_save,
                "quiet_query_latency_ms": quiet_query,
                "pressure_query_latency_ms": pressure_query,
                "save_p99_regression": save_regression,
                "query_p99_regression": query_regression,
                "remote_dram_bytes": remote_dram_bytes,
                "remote_ssd_bytes": remote_ssd_bytes,
                "ssd_read_bytes": ssd_read_bytes,
                "wrong_bytes": wrong_bytes,
                "stale_authorizations": stale_authorizations,
                "process_before": process_before,
                "process_after": process_after,
                "final_source": final_source,
                "final_consumer": final_consumer,
                "etcd_keys": keys,
                "initial_membership_revision": revision_before,
            }
        )
        (output / "soak-result.json").write_text(json.dumps(result, indent=2) + "\n")

        for node in ("source", "consumer"):
            stop_node(node)
        current_payloads.clear()
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=FULL_DURATION_SECONDS)
    parser.add_argument("--schedule-scale", type=float, default=1.0)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    if not 0 < args.schedule_scale <= 1:
        parser.error("--schedule-scale must be in (0, 1]")
    last_fault = FAULTS[-1][0] * args.schedule_scale
    if args.duration_seconds <= last_fault:
        parser.error("duration must extend beyond the final scheduled fault")
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY", "ORBITKV_MOONCAKE_LIB_DIR"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output, args.duration_seconds, args.schedule_scale)
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
