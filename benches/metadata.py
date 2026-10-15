"""Full-Manager inventory-stream churn and visibility experiment.

Run with a frozen package and test support on PYTHONPATH. The workload performs
no build and writes every result to the required external output directory.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import statistics
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

import requests

from tests.integration.test_distributed_cache import (
    _await_fence,
    _cleanup_dram,
    _metadata,
    _payload,
    _prefix_end,
    _query_ready,
    _restore,
    _sync,
    _until,
)
from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.cluster import etcd_server

from .artifacts import external_path


def _process_sample(pids):
    result = {"cpu_ticks": 0, "rss_kib": 0, "hwm_kib": 0, "processes": 0}
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, IndexError):
            continue
        values = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in status
            if line.startswith(("VmRSS:", "VmHWM:"))
        }
        result["cpu_ticks"] += int(stat[11]) + int(stat[12])
        result["rss_kib"] += values.get("VmRSS", 0)
        result["hwm_kib"] += values.get("VmHWM", 0)
        result["processes"] += 1
    return result


def _etcd_pid(endpoint):
    encoded = endpoint.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        if Path(os.fsdecode(command.split(b"\0", 1)[0])).name == "etcd" and encoded in command:
            return int(entry.name)
    raise AssertionError(f"cannot find test-owned etcd for {endpoint}")


def _etcd_metrics(endpoint):
    response = requests.get(f"{endpoint}/metrics", timeout=5)
    response.raise_for_status()
    wanted = {
        "etcd_network_client_grpc_received_bytes_total",
        "etcd_network_client_grpc_sent_bytes_total",
    }
    result = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(maxsplit=1)
        name = name.split("{", 1)[0]
        if name in wanted:
            result[name] = result.get(name, 0.0) + float(value)
    assert set(result) == wanted, "etcd byte metrics missing"
    return result


def _summary(values):
    ordered = sorted(values)

    def percentile(fraction):
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "samples": len(ordered),
        "p50": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _etcd_revision(endpoint: str, prefix: str) -> int:
    key = prefix.encode()
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(_prefix_end(key)).decode(),
            "limit": "1",
        },
        timeout=5,
    )
    response.raise_for_status()
    return int(response.json()["header"]["revision"])


def _etcd_keys(endpoint: str, prefix: str):
    key = prefix.encode()
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={
            "key": base64.b64encode(key).decode(),
            "range_end": base64.b64encode(_prefix_end(key)).decode(),
            "keys_only": True,
        },
        timeout=5,
    )
    response.raise_for_status()
    return [base64.b64decode(row["key"]).decode() for row in response.json().get("kvs", [])]


def _owner_status(manager, incarnation):
    response = requests.get(
        f"http://127.0.0.1:{manager.http_port}/cache/metadata/owners",
        params={"limit": 128},
        timeout=5,
    )
    response.raise_for_status()
    return next((row for row in response.json() if row["owner"] == incarnation), None)


def run(tmp_path: Path, profile: str):
    import torch

    assert torch.cuda.is_available()
    import orbitkv.orbitkv as native
    from orbitkv import BlockHashes, CacheManagerClient, QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

    artifact_paths = {
        "manager": os.environ["ORBITKV_CACHE_MANAGER_BINARY"],
        "etcd": os.environ["ETCD_BIN"],
        "extension": native.__file__,
    }
    if library_dir := os.environ.get("ORBITKV_MOONCAKE_LIB_DIR"):
        artifact_paths["tent"] = str(Path(library_dir) / "libtent_shared.so")
    manifest = {}
    for name, value in artifact_paths.items():
        path = Path(value).resolve()
        manifest[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ["MC_FORCE_TCP"] = "1"
    pages, block_bytes, cycles, low_traffic_samples = 8, 4096, 200, 10
    payload_bytes = pages * block_bytes
    cluster = f"s28-{profile}-{uuid.uuid4().hex[:12]}"
    namespace = f"s2.8:{profile}:{uuid.uuid4().hex}"
    hashes = [hashlib.sha256(f"s2.8:{block}".encode()).digest() for block in range(pages)]
    hash_batch = BlockHashes(hashes)
    payload = _payload(torch, pages, block_bytes, 28)
    result = {
        "inference_latency": None,
        "inference_scope": "client metadata experiment; no model-serving workload",
        "profile": profile,
        "pages": pages,
        "block_bytes": block_bytes,
        "burst_cycles": cycles,
        "low_traffic_samples": low_traffic_samples,
        "expected_input_mutations": (cycles + low_traffic_samples) * pages * 2,
    }

    raw = (tmp_path / "samples.jsonl").open("w", buffering=1)
    with ExitStack() as stack:
        stack.callback(raw.close)
        endpoint, _ = stack.enter_context(etcd_server(tmp_path))
        managers = []
        for node in ("source", "observer"):
            port = find_available_port()
            manager = CacheManagerProcess(
                port,
                pool_size="64mb",
                http_port=find_available_port(),
                bootstrap_socket=f"/tmp/orbitkv-s28-{port}.sock",
                log_path=tmp_path / f"{node}-manager.log",
                extra_args=(
                    "--etcd-endpoints",
                    endpoint,
                    "--node-id",
                    node,
                    "--cluster-name",
                    cluster,
                    "--membership-ttl-secs",
                    "120",
                    "--inventory-journal-bytes",
                    str(16 * 1024 * 1024),
                    "--enable-prometheus",
                    "--inventory-stream-coalesce-ms",
                    profile,
                ),
            )
            stack.callback(manager.stop)
            assert manager.start(), manager.read_logs()
            managers.append(manager)

        source_manager, observer_manager = managers
        source_client = CacheManagerClient(source_manager.bootstrap_socket)
        observer_client = CacheManagerClient(observer_manager.bootstrap_socket)
        stack.callback(source_client.close)
        stack.callback(observer_client.close)
        tensor = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
        observer_tensor = torch.empty_like(tensor)
        tensor.copy_(payload)
        torch.cuda.synchronize()
        device = resolve_device_id()
        registrations = {}
        for client, _manager, instance, target in (
            (source_client, source_manager, "source", tensor),
            (observer_client, observer_manager, "observer", observer_tensor),
        ):
            client.start_session_watcher(instance, namespace, 1, 1)
            registrations[instance] = client.register_context_batch(
                instance,
                namespace,
                0,
                0,
                1,
                1,
                device,
                ["kv:0"],
                [serialize_gpu_buffer(target)],
                [pages],
                [block_bytes],
                [0],
                [1],
                "direct",
                False,
                tensors=[target],
            )

        def unregister_on_exit():
            for client, instance in ((observer_client, "observer"), (source_client, "source")):
                if instance in registrations:
                    with contextlib.suppress(Exception):
                        client.unregister_context(instance)

        stack.callback(unregister_on_exit)
        _until(
            lambda: all(_metadata(manager)["index"]["registration_valid"] for manager in managers),
            managers,
        )
        ok, message = source_client.save(
            registrations["source"], [("kv:0", list(range(pages)), hashes)]
        )
        assert ok, message
        initial_fence = _sync(source_manager)
        _await_fence(observer_manager, initial_fence, managers)

        source_before = _metadata(source_manager)
        observer_before = _metadata(observer_manager)
        etcd_before = _etcd_metrics(endpoint)
        revision_before = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        manager_before = _process_sample(manager.process.pid for manager in managers)
        etcd_process_before = _process_sample([_etcd_pid(endpoint)])
        (tmp_path / "launch.json").write_text(
            json.dumps(
                {
                    "commands": [manager.command for manager in managers],
                    "pids": [manager.process.pid for manager in managers],
                    "etcd_pid": _etcd_pid(endpoint),
                    "source_incarnation": initial_fence["source_incarnation"],
                    "profile": profile,
                },
                indent=2,
            )
        )
        immediate_local_hits = []
        save_ms = []
        query_ms = []
        cleanup_ms = []
        workload_started = time.monotonic()
        for cycle in range(cycles):
            started = time.monotonic()
            assert _cleanup_dram(source_manager)["evicted_blocks"] == pages
            cleanup_ms.append((time.monotonic() - started) * 1000)
            started = time.monotonic()
            ok, message = source_client.save(
                registrations["source"], [("kv:0", list(range(pages)), hashes)]
            )
            save_ms.append((time.monotonic() - started) * 1000)
            assert ok, message
            started = time.monotonic()
            query_calls = 1
            ready = source_client.query_prefetch("source", hash_batch, f"s2.8-burst-{cycle}")
            immediate_local_hits.append(
                ready.num_hit_blocks if isinstance(ready, QueryReady) else 0
            )
            deadline = started + 5
            while not isinstance(ready, QueryReady) or ready.num_hit_blocks != pages:
                if isinstance(ready, QueryReady) and ready.lease:
                    source_client.release(ready.lease)
                assert time.monotonic() < deadline, (cycle, ready)
                time.sleep(0.0005)
                query_calls += 1
                ready = source_client.query_prefetch("source", hash_batch, f"s2.8-burst-{cycle}")
            query_ms.append((time.monotonic() - started) * 1000)
            source_client.release(ready.lease)
            raw.write(
                json.dumps(
                    {
                        "phase": "burst",
                        "cycle": cycle,
                        "query_calls": query_calls,
                        "immediate_hits": immediate_local_hits[-1],
                        "save_ms": save_ms[-1],
                        "query_ms": query_ms[-1],
                        "cleanup_ms": cleanup_ms[-1],
                        "hits": ready.num_hit_blocks,
                    }
                )
                + "\n"
            )

        burst_fence = _sync(source_manager)
        burst_sync_ms = (time.monotonic() - workload_started) * 1000
        visibility_started = time.monotonic()
        _await_fence(observer_manager, burst_fence, managers)
        burst_visibility_ms = (time.monotonic() - visibility_started) * 1000

        low_visibility_ms = []
        immediate_remote_hits = []
        for sample in range(low_traffic_samples):
            _cleanup_dram(observer_manager)
            assert _cleanup_dram(source_manager)["evicted_blocks"] == pages
            delete_fence = _sync(source_manager)
            _await_fence(observer_manager, delete_fence, managers)
            missing = _query_ready(
                observer_client,
                "observer",
                hashes,
                f"deleted-{sample}",
                0,
                managers,
            )
            assert not missing.lease
            target_sequence = _metadata(source_manager)["inventory_sequence"] + pages

            started = time.monotonic()
            ok, message = source_client.save(
                registrations["source"], [("kv:0", list(range(pages)), hashes)]
            )
            assert ok, message
            immediate = observer_client.query_prefetch(
                "observer", hash_batch, f"immediate-{sample}"
            )
            deadline = time.monotonic() + 5
            while not isinstance(immediate, QueryReady):
                assert time.monotonic() < deadline, immediate
                time.sleep(0.0005)
                immediate = observer_client.query_prefetch(
                    "observer", hash_batch, f"immediate-{sample}"
                )
            immediate_remote_hits.append(immediate.num_hit_blocks)
            if immediate.lease:
                observer_client.release(immediate.lease)
            while True:
                status = _owner_status(observer_manager, initial_fence["source_incarnation"])
                if status and status["fresh"] and status["applied_sequence"] >= target_sequence:
                    break
                assert time.monotonic() < deadline, [manager.read_logs() for manager in managers]
                time.sleep(0.0005)
            low_visibility_ms.append((time.monotonic() - started) * 1000)
            ready = _query_ready(
                observer_client,
                "observer",
                hashes,
                f"visible-{sample}",
                pages,
                managers,
            )
            observer_client.release(ready.lease)
            _restore(
                observer_client,
                "observer",
                observer_tensor,
                hashes,
                f"observed-{sample}",
                payload,
                managers,
            )
            raw.write(
                json.dumps(
                    {
                        "phase": "low",
                        "sample": sample,
                        "visibility_ms": low_visibility_ms[-1],
                        "target_sequence": target_sequence,
                        "immediate_remote_hits": immediate_remote_hits[-1],
                    }
                )
                + "\n"
            )
            time.sleep(0.02)

        final_fence = _sync(source_manager)
        _await_fence(observer_manager, final_fence, managers)
        source_after = _metadata(source_manager)
        observer_after = _metadata(observer_manager)
        assert observer_after["index"]["coverage"] == "complete_at_watermarks"
        etcd_after = _etcd_metrics(endpoint)
        revision_after = _etcd_revision(endpoint, f"/orbitkv/v2/{cluster}/")
        manager_after = _process_sample(manager.process.pid for manager in managers)
        etcd_process_after = _process_sample([_etcd_pid(endpoint)])
        keys = _etcd_keys(endpoint, f"/orbitkv/v2/{cluster}/")
        assert not any("/blocks/" in key or "/publishers/" in key for key in keys)
        elapsed = time.monotonic() - workload_started

        counters = {}
        for name in (
            "inventory_delta_input_records",
            "inventory_delta_input_bytes",
            "inventory_delta_output_records",
            "inventory_delta_frames",
            "inventory_delta_encoded_bytes",
            "inventory_coalescing_windows",
            "inventory_coalescing_wait_micros",
        ):
            counters[name] = source_after[name] - source_before[name]
        result.update(
            {
                "cluster": cluster,
                "source_incarnation": final_fence["source_incarnation"],
                "manager_commands": [manager.command for manager in managers],
                "manager_pids": [manager.process.pid for manager in managers],
                "etcd_pid": _etcd_pid(endpoint),
                "workload_seconds": elapsed,
                "burst_sync_ms": burst_sync_ms,
                "burst_visibility_ms": burst_visibility_ms,
                "save_ms": _summary(save_ms),
                "local_query_ms": _summary(query_ms),
                "cleanup_ms": _summary(cleanup_ms),
                "low_traffic_visibility_ms": _summary(low_visibility_ms),
                "effective_local_hit_rate": 1.0,
                "immediate_local_hit_rate": sum(immediate_local_hits) / (pages * cycles),
                "immediate_remote_hit_rate": sum(immediate_remote_hits)
                / (pages * low_traffic_samples),
                "after_visibility_remote_hit_rate": 1.0,
                "final_records": pages,
                "final_fence": final_fence,
                "observer_coverage": observer_after["index"]["coverage"],
                "publisher_counters": counters,
                "stream_encoded_bytes_sent": source_after["stream"]["encoded_bytes_sent"]
                - source_before["stream"]["encoded_bytes_sent"],
                "stream_encoded_bytes_received": observer_after["stream"]["encoded_bytes_received"]
                - observer_before["stream"]["encoded_bytes_received"],
                "etcd_keys": keys,
                "etcd_metrics_delta": {
                    name: etcd_after[name] - etcd_before[name] for name in etcd_before
                },
                "manager_process_before": manager_before,
                "manager_process_after": manager_after,
                "etcd_process_before": etcd_process_before,
                "etcd_process_after": etcd_process_after,
                "etcd_revision_before": revision_before,
                "etcd_revision_after": revision_after,
                "etcd_revision_delta": revision_after - revision_before,
            }
        )
        assert counters["inventory_delta_input_records"] == result["expected_input_mutations"]
        assert (
            counters["inventory_delta_output_records"] <= counters["inventory_delta_input_records"]
        )
        assert (
            source_after["inventory_journal_bytes"]
            <= source_after["inventory_journal_capacity_bytes"]
        )
        assert result["stream_encoded_bytes_sent"] > 0
        assert result["stream_encoded_bytes_received"] > 0
        _restore(source_client, "source", tensor, hashes, "final-local-bytes", payload, managers)
        for label, before, after in (
            ("managers", manager_before, manager_after),
            ("etcd", etcd_process_before, etcd_process_after),
        ):
            assert before["processes"] == after["processes"] and before["processes"] > 0
            assert after["hwm_kib"] - before["hwm_kib"] <= 256 * 1024, label
        (tmp_path / "stream-result.json").write_text(json.dumps(result, indent=2) + "\n")
        for client, instance in ((observer_client, "observer"), (source_client, "source")):
            ok, message = client.unregister_context(instance)
            assert ok, message
            client.close()
            registrations.pop(instance)
        tensor = observer_tensor = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("0", "2", "5"), required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    for variable in ("ETCD_BIN", "ORBITKV_CACHE_MANAGER_BINARY"):
        if not os.environ.get(variable):
            parser.error(f"set {variable} to a frozen artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output, args.profile)
    except BaseException as error:
        (args.output / "failure.txt").write_text(repr(error) + "\n")
        raise


if __name__ == "__main__":
    main()
