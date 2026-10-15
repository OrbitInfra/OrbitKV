"""Matched full-Manager all/scoped DRAM or io_uring live-store experiment."""

from __future__ import annotations

import argparse
import base64
import contextlib
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

DEFAULT_JOURNAL_BYTES = 256 * 1024


def _etcd_revision(endpoint, prefix):
    key = prefix.encode()
    end = bytearray(key)
    end[-1] += 1
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(bytes(end)).decode(),
            "limit": "1",
        },
        timeout=5,
    )
    response.raise_for_status()
    return int(response.json()["header"]["revision"])


def run(
    output: Path,
    subscription: str,
    coalesce_ms: int,
    seed: str,
    medium: str,
    duration_seconds: int,
    cycles: int,
    journal_bytes: int,
):
    import torch

    import orbitkv.orbitkv as native
    from orbitkv import CacheManagerClient
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    assert torch.cuda.is_available()
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes, namespace_count = 8, 4096, 4
    payload_bytes = pages * block_bytes
    identities = [f"s2.10:{medium}:{seed}:{index}" for index in range(namespace_count)]
    storage_namespaces = _discover_storage_namespaces(output, identities, pages, block_bytes)
    selected_scope = storage_namespaces[0]

    def hashes_for_round(round_id):
        key_round = round_id if medium == "ssd" else 0
        return [
            hashlib.sha256(f"s2.10:{medium}:{seed}:{key_round}:{block}".encode()).digest()
            for block in range(pages)
        ]

    hashes = hashes_for_round(0)
    payloads = [
        _payload(torch, pages, block_bytes, 120 + index) for index in range(namespace_count)
    ]
    cluster = f"s29-{subscription}-{uuid.uuid4().hex[:12]}"
    result = {
        "subscription": subscription,
        "seed": seed,
        "medium": medium,
        "coalesce_ms": coalesce_ms,
        "inventory_journal_bytes": journal_bytes,
        "namespace_count": namespace_count,
        "selected_scope": selected_scope,
        "storage_namespaces": storage_namespaces,
        "cycles": cycles,
        "duration_seconds": duration_seconds,
        "pages": pages,
        "block_bytes": block_bytes,
        "inference_latency": None,
        "scope": "same-host A100, two full Managers, forced TCP, metadata/client workload",
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
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(output))
        source_backend = find_available_port()
        gate = TcpGate(f"http://127.0.0.1:{source_backend}")
        stack.callback(gate.close)
        source_advertise = gate.endpoint.rsplit("//", 1)[1]
        managers = []
        clients = {}
        tensors = {}
        gpu_registrations = {}
        for node in ("source", "observer"):
            source = node == "source"
            port = source_backend if source else find_available_port()
            ssd_enabled = source and medium == "ssd"
            args = [
                "--etcd-endpoints",
                endpoint,
                "--node-id",
                node,
                "--cluster-name",
                cluster,
                "--membership-ttl-secs",
                "120",
                "--inventory-journal-bytes",
                str(journal_bytes),
                "--inventory-stream-coalesce-ms",
                str(coalesce_ms),
                "--index-budget",
                "16mb",
            ]
            if not ssd_enabled:
                args.append("--enable-prometheus")
            if subscription == "scoped":
                args.extend(["--metadata-namespace", selected_scope])
            if source:
                args.extend(["--peer-advertise-addr", source_advertise])
            manager = CacheManagerProcess(
                port,
                pool_size="96mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s29-bench-{port}.sock",
                ssd_cache_path=output / "source-ssd" if ssd_enabled else None,
                ssd_cache_capacity="128mb",
                ssd_backend="uring",
                ssd_read_path="uring" if ssd_enabled else None,
                log_path=output / f"{node}-manager.log",
                extra_args=tuple(args),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)
            for index, identity in enumerate(identities):
                instance = f"{node}-{index}"
                client = CacheManagerClient(manager.bootstrap_socket)
                stack.callback(client.close)
                clients[(node, index)] = client
                tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
                tensors[(node, index)] = tensor
                client.start_session_watcher(instance, identity, 1, 1)
                gpu_registrations[(node, index)] = client.register_context_batch(
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

        source_manager, observer_manager = managers

        def save_all(round_id):
            round_hashes = hashes_for_round(round_id)
            write_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                "orbitkv_ssd_write_bytes_total", 0
            )
            started = time.monotonic()
            for index, payload in enumerate(payloads):
                tensor = tensors[("source", index)]
                tensor.copy_((payload + round_id) % 251)
                torch.cuda.synchronize()
                ok, message = clients[("source", index)].save(
                    gpu_registrations[("source", index)],
                    [("kv:0", list(range(pages)), round_hashes)],
                )
                assert ok, message
            save_ms = (time.monotonic() - started) * 1000
            if medium == "ssd":
                expected_write = write_before + payload_bytes * namespace_count
                _until(
                    lambda: (
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
                    managers,
                )
                cleaned = _cleanup_dram(source_manager)
                assert cleaned["evicted_blocks"] == pages * namespace_count
                assert cleaned["still_referenced_blocks"] == 0
            return save_ms, round_hashes

        _, hashes = save_all(0)
        bootstrap_started = time.monotonic()
        initial_fence = _sync(source_manager)
        _await_fence(observer_manager, initial_fence, managers)
        bootstrap_ms = (time.monotonic() - bootstrap_started) * 1000
        before_source = _metadata(source_manager)
        before_observer = _metadata(observer_manager)
        process_before = _process_sample(manager.process.pid for manager in managers)
        revision_before = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        workload_started = time.monotonic()
        async_visibility = []
        barrier_visibility = []
        save_latency = []
        local_query_latency = []
        ssd_read_bytes = 0
        remote_fetch_bytes = 0
        cycle = 0
        while (
            time.monotonic() - workload_started < duration_seconds
            if duration_seconds
            else cycle < cycles
        ):
            cycle_started = time.monotonic()
            cleaned = _cleanup_dram(source_manager)
            if medium == "dram":
                assert cleaned["evicted_blocks"] == pages * namespace_count
            assert cleaned["still_referenced_blocks"] == 0
            sequence_before = _metadata(source_manager)["inventory_sequence"]
            started = time.monotonic()
            save_ms, hashes = save_all(cycle + 1)
            save_latency.append(save_ms)
            if cycle % 2 == 0:
                expected = sequence_before + pages * namespace_count
                _until(
                    lambda expected=expected: (
                        _metadata(source_manager)["inventory_sequence"] >= expected
                    ),
                    managers,
                )
                target = _metadata(source_manager)["inventory_sequence"]
                _until(
                    lambda target=target: (
                        (
                            status := _owner_status(
                                observer_manager, initial_fence["source_incarnation"]
                            )
                        )
                        and status["fresh"]
                        and status["applied_sequence"] >= target
                    ),
                    managers,
                )
                visibility = (time.monotonic() - started) * 1000
                async_visibility.append(visibility)
                mode = "async"
            else:
                fence = _sync(source_manager)
                _await_fence(observer_manager, fence, managers)
                visibility = (time.monotonic() - started) * 1000
                barrier_visibility.append(visibility)
                mode = "barrier"

            local_ssd_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                "orbitkv_ssd_prefetch_bytes_total", 0
            )
            local_expected = (payloads[0] + cycle + 1) % 251
            local_tensor = tensors[("source", 0)]
            local_tensor.zero_()
            torch.cuda.synchronize()
            query_started = time.monotonic()
            local = _query_ready(
                clients[("source", 0)],
                "source-0",
                hashes,
                f"source-local-{cycle}",
                pages,
                managers,
            )
            local_query_latency.append((time.monotonic() - query_started) * 1000)
            operation = clients[("source", 0)].start_restore(
                "source-0",
                0,
                local_tensor.device.index or 0,
                [["kv:0"]],
                [(local.lease, [list(range(pages))])],
                ready_stream=torch.cuda.current_stream(local_tensor.device).cuda_stream,
            )
            status = clients[("source", 0)].wait_restore(operation, timeout=20)
            assert status.success, status
            torch.cuda.synchronize()
            assert torch.equal(local_tensor, local_expected)
            if medium == "ssd":
                local_ssd_after = fetch_orbitkv_metrics(source_manager.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                assert local_ssd_after >= local_ssd_before + payload_bytes
                ssd_read_bytes += local_ssd_after - local_ssd_before
                cleaned = _cleanup_dram(source_manager)
                assert cleaned["still_referenced_blocks"] == 0

            _cleanup_dram(observer_manager)
            expected_payload = (payloads[0] + cycle + 1) % 251
            source_ssd_before = fetch_orbitkv_metrics(source_manager.http_port).get(
                "orbitkv_ssd_prefetch_bytes_total", 0
            )
            remote_before = fetch_orbitkv_metrics(observer_manager.http_port).get(
                "orbitkv_remote_fetch_bytes_total", 0
            )
            _restore(
                clients[("observer", 0)],
                "observer-0",
                tensors[("observer", 0)],
                hashes,
                f"selected-{cycle}",
                expected_payload,
                managers,
            )
            _wait_for_remote_drain(source_manager, observer_manager, managers)
            remote_after = fetch_orbitkv_metrics(observer_manager.http_port)[
                "orbitkv_remote_fetch_bytes_total"
            ]
            assert remote_after >= remote_before + payload_bytes
            remote_fetch_bytes += remote_after - remote_before
            if medium == "ssd":
                source_ssd_after = fetch_orbitkv_metrics(source_manager.http_port)[
                    "orbitkv_ssd_prefetch_bytes_total"
                ]
                assert source_ssd_after >= source_ssd_before + payload_bytes
                ssd_read_bytes += source_ssd_after - source_ssd_before
            if subscription == "scoped":
                miss = _query_ready(
                    clients[("observer", 1)],
                    "observer-1",
                    hashes,
                    f"outside-{cycle}",
                    0,
                    managers,
                )
                assert not miss.lease
                if cycle % 10 == 0:
                    outside = tensors[("observer", 1)]
                    outside_payload = (payloads[1] + cycle + 1) % 251
                    outside.copy_(outside_payload)
                    torch.cuda.synchronize()
                    ok, message = clients[("observer", 1)].save(
                        gpu_registrations[("observer", 1)], [("kv:0", list(range(pages)), hashes)]
                    )
                    assert ok, message
                    _restore(
                        clients[("observer", 1)],
                        "observer-1",
                        outside,
                        hashes,
                        f"outside-local-{cycle}",
                        outside_payload,
                        managers,
                    )
                    _cleanup_dram(observer_manager)
                    miss = _query_ready(
                        clients[("observer", 1)],
                        "observer-1",
                        hashes,
                        f"outside-after-local-{cycle}",
                        0,
                        managers,
                    )
                    assert not miss.lease
            raw.write(
                json.dumps(
                    {
                        "cycle": cycle,
                        "mode": mode,
                        "visibility_ms": visibility,
                        "save_ms": save_latency[-1],
                        "local_query_ms": local_query_latency[-1],
                    }
                )
                + "\n"
            )
            cycle += 1
            if duration_seconds:
                remaining = 1 - (time.monotonic() - cycle_started)
                if remaining > 0:
                    time.sleep(remaining)

        actual_cycles = cycle
        steady_elapsed_ms = (time.monotonic() - workload_started) * 1000
        if duration_seconds:
            assert steady_elapsed_ms >= duration_seconds * 1000

        steady_source = _metadata(source_manager)
        steady_observer = _metadata(observer_manager)
        steady_history_gaps = (
            steady_source["inventory_history_gaps"] - before_source["inventory_history_gaps"]
        )
        assert steady_history_gaps == 0
        gate.partition()
        repair_rounds = max(10, journal_bytes // 4096 + 10)
        for repair in range(repair_rounds):
            _cleanup_dram(source_manager)
            _, hashes = save_all(actual_cycles + repair + 1)
        repair_fence = _sync(source_manager)
        repair_started = time.monotonic()
        gate.heal()
        _await_fence(observer_manager, repair_fence, managers, timeout=20)
        repair_ms = (time.monotonic() - repair_started) * 1000
        _cleanup_dram(observer_manager)
        final_payload = (payloads[0] + actual_cycles + repair_rounds) % 251
        _restore(
            clients[("observer", 0)],
            "observer-0",
            tensors[("observer", 0)],
            hashes,
            "repair-selected",
            final_payload,
            managers,
        )
        _wait_for_remote_drain(source_manager, observer_manager, managers)

        after_source = _metadata(source_manager)
        after_observer = _metadata(observer_manager)
        process_after = _process_sample(manager.process.pid for manager in managers)
        revision_after = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        owner = _owner_status(observer_manager, initial_fence["source_incarnation"])
        retained_rounds = 1
        if medium == "ssd":
            capacity_rounds = (128 * 1024**2) // (payload_bytes * namespace_count)
            retained_rounds = min(1 + actual_cycles + repair_rounds, capacity_rounds)
        expected_records = (
            retained_rounds * pages * (namespace_count if subscription == "all" else 1)
        )
        assert owner and owner["records"] == expected_records
        oldest_evicted = medium == "ssd" and 1 + actual_cycles + repair_rounds > retained_rounds
        if oldest_evicted:
            _cleanup_dram(observer_manager)
            miss = _query_ready(
                clients[("observer", 0)],
                "observer-0",
                hashes_for_round(0),
                "evicted-oldest-ssd-round",
                0,
                managers,
            )
            assert not miss.lease
        source_metrics = fetch_orbitkv_metrics(source_manager.http_port)
        observer_metrics = fetch_orbitkv_metrics(observer_manager.http_port)
        result.update(
            {
                "cluster": cluster,
                "manager_commands": [manager.command for manager in managers],
                "bootstrap_ms": bootstrap_ms,
                "repair_ms": repair_ms,
                "repair_rounds": repair_rounds,
                "workload_ms": (time.monotonic() - workload_started) * 1000,
                "steady_elapsed_ms": steady_elapsed_ms,
                "actual_cycles": actual_cycles,
                "achieved_cycles_per_second": actual_cycles / (steady_elapsed_ms / 1000),
                "async_visibility_ms": _summary(async_visibility),
                "barrier_visibility_ms": _summary(barrier_visibility),
                "save_latency_ms": _summary(save_latency),
                "local_query_latency_ms": _summary(local_query_latency),
                "ssd_read_bytes": ssd_read_bytes,
                "remote_fetch_bytes": remote_fetch_bytes,
                "owner_records": owner["records"],
                "expected_owner_records": expected_records,
                "retained_rounds": retained_rounds,
                "oldest_ssd_round_evicted": oldest_evicted,
                "final_hashes": [value.hex() for value in hashes],
                "final_payload_sha256": hashlib.sha256(
                    final_payload.cpu().numpy().tobytes()
                ).hexdigest(),
                "stream_bytes_sent": after_source["stream"]["encoded_bytes_sent"]
                - before_source["stream"]["encoded_bytes_sent"],
                "stream_bytes_received": after_observer["stream"]["encoded_bytes_received"]
                - before_observer["stream"]["encoded_bytes_received"],
                "steady_stream_bytes_sent": steady_source["stream"]["encoded_bytes_sent"]
                - before_source["stream"]["encoded_bytes_sent"],
                "steady_stream_bytes_received": steady_observer["stream"]["encoded_bytes_received"]
                - before_observer["stream"]["encoded_bytes_received"],
                "stream_frames_sent": after_source["stream"]["frames_sent"]
                - before_source["stream"]["frames_sent"],
                "stream_frames_received": after_observer["stream"]["frames_received"]
                - before_observer["stream"]["frames_received"],
                "steady_history_gaps": steady_history_gaps,
                "repair_history_gaps": after_source["inventory_history_gaps"]
                - steady_source["inventory_history_gaps"],
                "stream_resets": after_source["stream"]["resets"]
                - before_source["stream"]["resets"],
                "inventory_journal_bytes_peak": after_source["inventory_journal_bytes_peak"],
                "delta_input_records": after_source["inventory_delta_input_records"]
                - before_source["inventory_delta_input_records"],
                "delta_output_records": after_source["inventory_delta_output_records"]
                - before_source["inventory_delta_output_records"],
                "delta_frames": after_source["inventory_delta_frames"]
                - before_source["inventory_delta_frames"],
                "filter_input_records": after_source["inventory_scope_filter_input_records"]
                - before_source["inventory_scope_filter_input_records"],
                "filter_output_records": after_source["inventory_scope_filter_output_records"]
                - before_source["inventory_scope_filter_output_records"],
                "filter_micros": after_source["inventory_scope_filter_micros"]
                - before_source["inventory_scope_filter_micros"],
                "index": after_observer["index"],
                "queue_peak_bytes": after_source["stream"]["outbound_queue_bytes_peak"],
                "source_sessions_peak": after_source["stream"]["source_sessions_peak"],
                "receiver_sessions_peak": after_observer["stream"]["receiver_sessions_peak"],
                "process_before": process_before,
                "process_after": process_after,
                "source_storage_metrics": {
                    name: source_metrics.get(name, 0)
                    for name in (
                        "orbitkv_ssd_write_bytes_total",
                        "orbitkv_ssd_prefetch_bytes_total",
                        "orbitkv_ssd_write_inflight",
                        "orbitkv_ssd_write_queue_pending",
                        "orbitkv_inflight_bytes",
                        "orbitkv_transfer_lock_active",
                    )
                },
                "observer_transfer_metrics": {
                    name: observer_metrics.get(name, 0)
                    for name in (
                        "orbitkv_remote_fetch_bytes_total",
                        "orbitkv_transfer_completion_outstanding",
                        "orbitkv_query_reserved_bytes",
                    )
                },
                "etcd_revision_delta": revision_after - revision_before,
                "etcd_keys": keys,
            }
        )
        assert result["filter_input_records"] >= result["filter_output_records"]
        assert result["repair_history_gaps"] >= 1
        assert result["stream_resets"] >= 1
        assert result["remote_fetch_bytes"] >= payload_bytes * actual_cycles
        if medium == "ssd":
            assert result["ssd_read_bytes"] >= payload_bytes * actual_cycles * 2
        assert after_observer["index"]["coverage"] == "complete_at_watermarks"
        assert after_source["stream"]["source_sessions_peak"] >= 1
        assert after_observer["stream"]["receiver_sessions_peak"] >= 1
        assert not any(
            source_metrics.get(name, 0)
            for name in (
                "orbitkv_ssd_write_inflight",
                "orbitkv_ssd_write_queue_pending",
                "orbitkv_inflight_bytes",
                "orbitkv_transfer_lock_active",
            )
        )
        assert not observer_metrics.get("orbitkv_transfer_completion_outstanding", 0)
        (output / "scoped-result.json").write_text(json.dumps(result, indent=2) + "\n")
        for (node, index), client in clients.items():
            with contextlib.suppress(Exception):
                client.unregister_context(f"{node}-{index}")
            client.close()
        tensor = outside = local_tensor = None
        expected_payload = local_expected = final_payload = outside_payload = None
        payloads.clear()
        tensors.clear()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", choices=("all", "scoped"), required=True)
    parser.add_argument("--medium", choices=("dram", "ssd"), default="dram")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--coalesce-ms", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--duration-seconds", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--journal-bytes", type=int, default=DEFAULT_JOURNAL_BYTES)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.duration_seconds < 0:
        parser.error("--duration-seconds cannot be negative")
    if args.cycles <= 0:
        parser.error("--cycles must be positive")
    if args.journal_bytes < 32 * 1024:
        parser.error("--journal-bytes must be at least 32 KiB")
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY", "ORBITKV_MOONCAKE_LIB_DIR"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(
            args.output,
            args.subscription,
            args.coalesce_ms,
            args.seed,
            args.medium,
            args.duration_seconds,
            args.cycles,
            args.journal_bytes,
        )
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
