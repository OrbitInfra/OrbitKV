"""Native common-prefix ownership across two installed Manager processes.

Trigger: shard coordinator, control protocol or official handshake changes.
Requires the installed wheel and actual CUDA. This is a same-GPU byte/lifetime
boundary gate, not physical multi-GPU TP or cross-host qualification.
"""

import contextlib
import hashlib
import json
import os
import signal
import sys
import time

import pytest
import requests

from tests.support.cache_manager import ClientContext, find_available_port
from tests.support.installed_serving import manager_command, service, wait_for_drain
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("tier", ["dram", "ssd"])
def test_two_manager_native_common_prefix_restores_exact_node_payload(tier, tmp_path):
    import torch

    from orbitkv import (
        BlockHashes,
        CacheManagerClient,
        OrbitKVError,
        QueryLoading,
        ShardedQueryReady,
    )

    ports = [find_available_port() for _ in range(4)]
    grpc, http = ports[:2], ports[2:]
    endpoints = [f"http://127.0.0.1:{port}" for port in grpc]
    namespaces = [f"native-shards-{tier}-{node}" for node in range(2)]
    hashes = [hashlib.sha256(f"{tier}:block:{block}".encode()).digest() for block in range(4)]
    clients, contexts, processes, evidence = [], [], [], {}
    with contextlib.ExitStack() as stack:
        for node in range(2):
            directory = tmp_path / f"node-{node}"
            directory.mkdir()
            command = manager_command(sys.executable, grpc[node], http[node], tier, directory)
            command.append("--enable-query-control")
            processes.append(
                stack.enter_context(
                    service(
                        command,
                        f"http://127.0.0.1:{http[node]}",
                        dict(os.environ),
                        directory,
                        f"manager-{node}",
                    )
                )
            )
            client = CacheManagerClient(f"/tmp/orbitkv-{grpc[node]}.sock")
            clients.append(client)
            stack.callback(client.close)
            client.start_session_watcher("dense", namespaces[node], 1, 1)
            context = ClientContext(
                client, "dense", namespaces[node], num_blocks=4, num_heads=1, head_size=64
            )
            contexts.append(context)
            context.register_kv_caches()
        assert processes[0].pid != processes[1].pid
        targets = [
            client.export_query_target("dense", namespace, 1, 1)
            for client, namespace in zip(clients, namespaces, strict=True)
        ]
        local = clients[0]
        for _ in range(2):
            local.configure_shard_queries(
                "dense", namespaces[0], 1, 1, list(zip(endpoints, namespaces, targets, strict=True))
            )
        with pytest.raises(ValueError):
            local.configure_shard_queries(
                "dense",
                namespaces[0],
                1,
                1,
                list(zip(endpoints, namespaces, reversed(targets), strict=True)),
            )
        assert local.health()[0]

        def query(batch, request):
            batch = BlockHashes(batch)
            deadline = time.monotonic() + 35
            while True:
                result = local.query_shards("dense", batch, request)
                if not isinstance(result, QueryLoading):
                    assert isinstance(result, ShardedQueryReady)
                    return result
                assert result.admitted
                assert time.monotonic() < deadline, "native shard query timed out"
                time.sleep(0.002)

        cold = query(hashes, "cold")
        assert cold.num_hit_blocks == 0 and cold.leases == (b"", b"") and not cold.control_id
        expected = []
        for node, context in enumerate(contexts):
            tensor = context.get_kv_cache()
            pattern = (
                torch.arange(tensor.numel(), device=tensor.device).reshape(tensor.shape) % 127
                + 128 * node
                + 1
            ).to(tensor.dtype)
            tensor.copy_(pattern)
            expected.append(pattern.cpu())
            torch.cuda.synchronize()
            prefix = 3 - node
            ok, message = clients[node].save(
                context.registration,
                [("layer_0", list(range(prefix)), hashes[:prefix])],
            )
            assert ok, message
            response = requests.post(f"http://127.0.0.1:{http[node]}/cache/sync", timeout=30)
            response.raise_for_status()
            if tier == "ssd":
                response = requests.post(
                    f"http://127.0.0.1:{http[node]}/cache/memory/cleanup", timeout=10
                )
                response.raise_for_status()
                cleanup = response.json()
                assert cleanup["still_referenced_blocks"] == 0
                assert cleanup["evicted_blocks"] == prefix
            evidence[f"before-{node}"] = fetch_orbitkv_metrics(http[node])
        ready = query(hashes, "unequal-prefix")
        assert ready.num_hit_blocks == 2 and len(ready.leases) == 2
        assert all(len(lease) == 16 and any(lease) for lease in ready.leases)
        assert len(ready.control_id) == 16 and any(ready.control_id)
        evidence["coordinator-ready"] = fetch_orbitkv_metrics(http[0])
        assert evidence["coordinator-ready"]["orbitkv_shard_query_submissions_total"] >= 2
        assert evidence["coordinator-ready"]["orbitkv_shard_query_active"] == 1
        assert evidence["coordinator-ready"]["orbitkv_shard_query_holds"] == 1
        for context in contexts:
            context.get_kv_cache().zero_()
        torch.cuda.synchronize()
        for node, context in enumerate(contexts):
            operation = clients[node].start_restore(
                "dense",
                0,
                context.device_id,
                [["layer_0"]],
                [(ready.leases[node], [[0, 1]])],
                ready_stream=torch.cuda.current_stream(context.device).cuda_stream,
            )
            status = clients[node].wait_restore(operation, timeout=20)
            assert status.success, status.message
            torch.cuda.synchronize()
            actual = context.get_kv_cache().cpu()
            oracle = torch.zeros_like(expected[node])
            oracle[:, :2] = expected[node][:, :2]
            assert torch.equal(actual.view(torch.uint8), oracle.view(torch.uint8))
            evidence[f"restore-{node}"] = fetch_orbitkv_metrics(http[node])
            if tier == "ssd":
                assert evidence[f"restore-{node}"]["orbitkv_ssd_prefetch_bytes_total"] > evidence[
                    f"before-{node}"
                ].get("orbitkv_ssd_prefetch_bytes_total", 0)
        assert not torch.equal(expected[0].view(torch.uint8), expected[1].view(torch.uint8))
        local.release_shard_query(ready.control_id)
        local.release_shard_query(ready.control_id)
        for port in http:
            drained = wait_for_drain(port)
            for name in (
                "orbitkv_query_control_interests",
                "orbitkv_shard_query_active",
                "orbitkv_shard_query_holds",
            ):
                assert drained[name] == 0

        # Revision/cancel must retire known source interests before the next read.
        waiting = local.query_shards("dense", BlockHashes(hashes), "revision")
        if isinstance(waiting, ShardedQueryReady) and waiting.control_id:
            local.release_shard_query(waiting.control_id)
        revised = query(hashes[:1], "revision")
        assert revised.num_hit_blocks == 1
        local.release_shard_query(revised.control_id)
        waiting = local.query_shards("dense", BlockHashes(hashes), "canceled")
        if isinstance(waiting, ShardedQueryReady) and waiting.control_id:
            local.release_shard_query(waiting.control_id)
        local.cancel_query("dense", "canceled")
        for port in http:
            wait_for_drain(port)

        # Replacing one source registration invalidates the frozen handshake.
        contexts[1].unregister_context()
        contexts[1].register_kv_caches()
        assert clients[1].export_query_target("dense", namespaces[1], 1, 1) != targets[1]
        with pytest.raises(OrbitKVError):
            query(hashes, "stale-generation")
        for port in http:
            wait_for_drain(port)
        for node, context in enumerate(contexts):
            ok, message = clients[node].unregister_context("dense")
            assert ok, message
            context._registered = False
        for client in clients:
            client.close()
        for node, port in enumerate(http):
            evidence[f"final-{node}"] = wait_for_drain(port)
        # Both services must leave normally; the outer service owner asserts exit 0.
        processes[1].send_signal(signal.SIGTERM)
        processes[1].wait(timeout=30)
        assert processes[1].returncode == 0
        evidence.update(
            {
                "manager_pids": [process.pid for process in processes],
                "tier": tier,
                "stored_prefixes": [3, 2],
                "common_blocks": 2,
                "restored_bytes_per_node": 8192,
                "scope": "two actual processes on one GPU; not physical TP",
            }
        )
        (tmp_path / "RESULT.json").write_text(json.dumps(evidence, indent=2) + "\n")
