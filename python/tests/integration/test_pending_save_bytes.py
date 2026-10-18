"""Installed SaveHandle/registration/Restore byte gate on one owned CUDA GPU.

Run separately from CPU feedback after exact runtime and launch review. This
normal-path gate does not establish failed CUDA/RDMA/SSD physical termination.
"""

import contextlib
import hashlib
import json
import os
import time
import uuid

import pytest
import requests

from tests.support.ipc_credit import hold_counters, release_receipt
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.registration_campaign import identity, write

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def register(client, identity, namespace, tensor):
    return client.register_context_batch(
        identity,
        namespace,
        0,
        0,
        1,
        1,
        tensor.device.index,
        ["bytes"],
        [8],
        [4096],
        [0],
        [1],
        "direct",
        False,
        tensors=[tensor],
    )


def wait_idle(http_port, names):
    deadline = time.monotonic() + 30
    while True:
        metrics = fetch_orbitkv_metrics(http_port)
        missing = set(names) - metrics.keys()
        if missing:
            raise RuntimeError(f"Missing ownership metrics: {sorted(missing)}")
        if all(metrics[name] == 0 for name in names):
            return metrics
        assert time.monotonic() < deadline, metrics
        time.sleep(0.01)


@pytest.mark.parametrize(
    "channel_server",
    [{"tier": "dram", "installed": True}, {"tier": "ssd", "installed": True}],
    ids=["dram", "ssd"],
    indirect=True,
)
def test_installed_pending_save_has_exact_payload_and_normal_release(
    channel_server, tmp_path, request
):
    import torch

    from orbitkv import BlockHashes, CacheManagerClient, QueryReady, SaveNotSubmittedError

    if not torch.cuda.is_available():
        pytest.fail("Installed byte qualification requires a working CUDA GPU")
    if channel_server.ssd_cache_path is not None:
        assert request.config.getoption("--ssd-backend") == "uring", (
            "SSD byte gate requires explicit io_uring backend"
        )
        assert request.config.getoption("--ssd-read-path") == "uring", (
            "SSD byte gate requires explicit io_uring restore"
        )
    namespace = f"pending-bytes-{uuid.uuid4().hex}"
    source = CacheManagerClient(channel_server.bootstrap_socket)
    consumer = CacheManagerClient(channel_server.bootstrap_socket)
    source_id, consumer_id = f"source-{uuid.uuid4().hex}", f"consumer-{uuid.uuid4().hex}"
    source_registered = consumer_registered = False
    source_tensor = torch.empty((8, 4096), dtype=torch.uint8, device="cuda")
    target = torch.empty_like(source_tensor)
    source_indices, target_indices = [0, 2, 4, 6], [1, 3, 5, 7]
    records = []
    counter_descriptors = contextlib.ExitStack()
    try:
        for round_id in range(3):
            payload = (
                (
                    torch.arange(8 * 4096, dtype=torch.int64).reshape(8, 4096)
                    + torch.arange(8, dtype=torch.int64)[:, None] * 37
                    + round_id * 71
                )
                .remainder(251)
                .to(torch.uint8)
            )
            source_tensor.copy_(payload)
            torch.cuda.synchronize()
            source_registration = register(source, source_id, namespace, source_tensor)
            source_registered = True
            write(
                tmp_path / f"round-{round_id}-source.credit-owner.json",
                {
                    "owner": identity(os.getpid()),
                    "instance_id": source_id,
                    "evidence": json.loads(source_registration.ipc_credit_evidence),
                },
                time.monotonic() + 30,
            )
            source_counters = hold_counters(
                counter_descriptors, tmp_path / f"round-{round_id}-source.credit-owner.json"
            )
            if round_id == 0:
                channel_server.record_runtime()
            writes_before = fetch_orbitkv_metrics(channel_server.http_port).get(
                "orbitkv_ssd_write_bytes_total", 0
            )
            hashes = [
                hashlib.sha256(f"{namespace}/{round_id}/{index}".encode()).digest()
                for index in source_indices
            ]
            handle = source.start_save(source_registration, [("bytes", source_indices, hashes)])
            status = source.wait_save(handle, timeout=30)
            assert status.done and status.success, status.message
            success, message = source.unregister_context(source_id)
            assert success, message
            source_registered = False
            write(
                tmp_path / f"round-{round_id}-source.credit-release.json",
                release_receipt(source_counters),
                time.monotonic() + 30,
            )
            with pytest.raises(SaveNotSubmittedError, match="no longer active") as rejected:
                source.start_save(source_registration, [("bytes", source_indices, hashes)])
            assert isinstance(rejected.value.__cause__, ValueError)
            del handle, source_registration
            idle_names = [
                "orbitkv_save_active_operations",
                "orbitkv_save_active_metadata_bytes",
                "orbitkv_save_active_logical_bytes",
                "orbitkv_save_retained_operations",
                "orbitkv_inflight_bytes",
            ]
            before = wait_idle(channel_server.http_port, idle_names)
            if channel_server.ssd_cache_path is not None:
                write_deadline = time.monotonic() + 30
                while True:
                    written = fetch_orbitkv_metrics(channel_server.http_port)
                    if written["orbitkv_ssd_write_bytes_total"] > writes_before:
                        break
                    assert time.monotonic() < write_deadline, written
                    time.sleep(0.01)
                before = wait_idle(
                    channel_server.http_port,
                    idle_names + ["orbitkv_ssd_write_inflight", "orbitkv_ssd_write_queue_pending"],
                )
                assert before["orbitkv_ssd_write_bytes_total"] > 0
                reply = requests.post(
                    f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=30
                )
                reply.raise_for_status()
                assert reply.json()["evicted_blocks"] > 0
                assert reply.json()["still_referenced_blocks"] == 0
            target.fill_(167)
            torch.cuda.synchronize()
            consumer_registration = register(consumer, consumer_id, namespace, target)
            consumer_registered = True
            write(
                tmp_path / f"round-{round_id}-consumer.credit-owner.json",
                {
                    "owner": identity(os.getpid()),
                    "instance_id": consumer_id,
                    "evidence": json.loads(consumer_registration.ipc_credit_evidence),
                },
                time.monotonic() + 30,
            )
            consumer_counters = hold_counters(
                counter_descriptors, tmp_path / f"round-{round_id}-consumer.credit-owner.json"
            )
            io_before = fetch_orbitkv_metrics(channel_server.http_port)
            read_before = {
                name: io_before.get(name)
                for name in (
                    "orbitkv_ssd_prefetch_bytes_total",
                    "orbitkv_ssd_uring_read_operations_total",
                )
            }
            deadline = time.monotonic() + 30
            while True:
                query = consumer.query_prefetch(
                    consumer_id, BlockHashes(hashes), f"round-{round_id}"
                )
                if isinstance(query, QueryReady) and query.num_hit_blocks == len(hashes):
                    break
                assert time.monotonic() < deadline, query
                time.sleep(0.01)
            restore = consumer.start_restore(
                consumer_id,
                0,
                0,
                [["bytes"]],
                [(query.lease, [target_indices])],
                ready_stream=torch.cuda.current_stream().cuda_stream,
            )
            status = consumer.wait_restore(restore, timeout=30)
            assert status.done and status.success, status.message
            torch.cuda.synchronize()
            expected = torch.full((8, 4096), 167, dtype=torch.uint8)
            expected[target_indices] = payload[source_indices]
            actual = target.cpu()
            assert torch.equal(actual, expected), "payload or untouched sentinel differs"
            # Deliberate mutation checks the same full-byte oracle.
            actual[1, 0] ^= 1
            assert not torch.equal(actual, expected), "byte oracle ignored a corruption"
            actual[1, 0] ^= 1
            after = fetch_orbitkv_metrics(channel_server.http_port)
            if channel_server.ssd_cache_path is not None:
                for name in (
                    "orbitkv_ssd_prefetch_bytes_total",
                    "orbitkv_ssd_uring_read_operations_total",
                ):
                    assert name in after, f"Missing actual io_uring read metric: {name}"
                    assert (
                        after[name] > 0
                        if read_before[name] is None
                        else after[name] > read_before[name]
                    )
            success, message = consumer.unregister_context(consumer_id)
            assert success, message
            consumer_registered = False
            write(
                tmp_path / f"round-{round_id}-consumer.credit-release.json",
                release_receipt(consumer_counters),
                time.monotonic() + 30,
            )
            del consumer_registration, restore
            wait_idle(channel_server.http_port, idle_names + ["orbitkv_query_reserved_bytes"])
            raw = bytes(actual.flatten().tolist())
            (tmp_path / f"round-{round_id}.payload.bin").write_bytes(raw)
            records.append(
                {
                    "round": round_id,
                    "copied_bytes": 4 * 4096,
                    "observed_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "full_oracle": True,
                    "mutation_rejected": True,
                    "tier": "ssd-uring" if channel_server.ssd_cache_path else "dram",
                    "source_identity": source_id,
                    "consumer_identity": consumer_id,
                    "normal_unregister": True,
                    "read_before": read_before,
                    "read_after": {name: after.get(name) for name in read_before},
                }
            )
            write(
                tmp_path / f"round-{round_id}.native-release.json",
                {
                    "owner": identity(os.getpid()),
                    "source": source_id,
                    "consumer": consumer_id,
                    "unregistered": True,
                    "completed_mono_ns": time.monotonic_ns(),
                    "physical_fault_termination_claimed": False,
                },
                time.monotonic() + 30,
            )
        (tmp_path / "PENDING-SAVE-BYTES.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        # Manager must stay alive through native release. Unknown work keeps the
        # ordinary owners waiting; the external controller retains its locks.
        if consumer_registered:
            success, message = consumer.unregister_context(consumer_id)
            assert success, message
        if source_registered:
            success, message = source.unregister_context(source_id)
            assert success, message
        consumer.close()
        source.close()
        counter_descriptors.close()
        write(
            tmp_path / "clients.native-close.json",
            {
                "owner": identity(os.getpid()),
                "source": source_id,
                "consumer": consumer_id,
                "closed": True,
                "normal_unregister": not source_registered and not consumer_registered,
            },
            time.monotonic() + 30,
        )
