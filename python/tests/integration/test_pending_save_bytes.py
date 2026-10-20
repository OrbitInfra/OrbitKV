"""Installed SaveHandle/registration/Restore byte gate on one owned CUDA GPU.

Run separately from CPU feedback after exact runtime and launch review. This
normal-path gate does not establish failed CUDA/RDMA/SSD physical termination.
"""

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
import requests

from tests.support.ipc_credit import hold_counters, release_receipt
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.registration_campaign import identity, write

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def save_bytes_snapshot(http_port: int) -> int:
    deadline = time.monotonic() + 10
    helper = Path(__file__).parents[1] / "support/save_metrics_http.py"
    try:
        response = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(helper), str(http_port)],
            capture_output=True,
            timeout=max(0.001, deadline - time.monotonic()),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Save byte counter response exceeded its time bound") from error
    raw = response.stdout
    if response.returncode or len(raw) > 1024**2:
        raise RuntimeError("Save byte counter response failed or exceeded its byte bound")
    if time.monotonic() >= deadline:
        raise RuntimeError("Save byte counter response exceeded its time bound")
    samples = []
    label_sets = set()
    for line in raw.decode("ascii", errors="strict").splitlines():
        if len(line) > 4096:
            raise RuntimeError("Save byte counter response exceeded its line bound")
        if not line.startswith("orbitkv_save_bytes_total"):
            continue
        match = re.fullmatch(
            r"orbitkv_save_bytes_total(?:\{([^}]*)\})?\s+([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)",
            line,
        )
        if match is None or len(match[2]) > 128 or len(samples) >= 4096:
            raise RuntimeError("Save byte counter is missing or malformed")
        labels = match[1] or ""
        label_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\[\\"n])*"'
        if len(labels) > 1024 or (
            labels and re.fullmatch(label_pattern + r"(?:," + label_pattern + r")*", labels) is None
        ):
            raise RuntimeError("Save byte counter labels are malformed or oversized")
        pairs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:[^"\\]|\\[\\"n])*")', labels)
        label_set = tuple(sorted(pairs))
        if len({key for key, _ in pairs}) != len(pairs) or label_set in label_sets:
            raise RuntimeError("Save byte counter labels are duplicated")
        label_sets.add(label_set)
        try:
            value = Decimal(match[2])
        except InvalidOperation as error:
            raise RuntimeError("Save byte counter is missing or malformed") from error
        if not value.is_finite() or not 0 <= value <= 2**53 or value != value.to_integral_value():
            raise RuntimeError("Save byte counter is missing, nonintegral or outside exact range")
        samples.append(int(value))
        if time.monotonic() >= deadline:
            raise RuntimeError("Save byte counter response exceeded its time bound")
    if not samples or sum(samples) > 2**53:
        raise RuntimeError("Save byte counter is missing or outside exact range")
    return sum(samples)


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
            admission_started = time.monotonic_ns()
            admission_before = save_bytes_snapshot(channel_server.http_port)
            canceled = source.prepare_save(
                source_registration,
                [("bytes", [source_indices[0]], [hashlib.sha256(b"never-published").digest()])],
            )
            pending = source.wait_save(canceled, timeout=0)
            if pending.done or pending.success:
                raise RuntimeError("Prepared Save reported completion before submission")
            if not source.cancel_save(canceled):
                raise RuntimeError("Never-submitted Save cancellation lacked a receipt")
            canceled_status = source.wait_save(canceled, timeout=0)
            if not canceled_status.done or canceled_status.success:
                raise RuntimeError("Prepared cancellation reported a successful Publish")
            handles = [
                source.prepare_save(
                    source_registration,
                    [("bytes", source_indices[start : start + 2], hashes[start : start + 2])],
                )
                for start in (0, 2)
            ]
            operation_ids = [canceled.operation_id, *(handle.operation_id for handle in handles)]
            if len(set(operation_ids)) != 3:
                raise RuntimeError("Prepared Save operation identities were reused")
            for handle in handles:
                pending = source.wait_save(handle, timeout=0)
                if pending.done or pending.success:
                    raise RuntimeError("Prepared batch completed before Rust submission")
            admission_prepared = save_bytes_snapshot(channel_server.http_port)
            if admission_prepared != admission_before:
                raise RuntimeError("Preparation/cancellation copied source bytes")
            source.submit_saves(handles)
            for handle in handles:
                if source.cancel_save(handle):
                    raise RuntimeError("Submitted Save was canceled as never-submitted")
                status = source.wait_save(handle, timeout=30)
                if not status.done or not status.success:
                    raise RuntimeError(f"Prepared Save did not complete: {status.message}")
            admission_completed = time.monotonic_ns()
            success, message = source.unregister_context(source_id)
            assert success, message
            source_registered = False
            write(
                tmp_path / f"round-{round_id}-source.credit-release.json",
                release_receipt(source_counters),
                time.monotonic() + 30,
            )
            synced = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/sync", timeout=35
            )
            synced.raise_for_status()
            assert synced.json()["local_flush_complete"] is True
            idle_names = [
                "orbitkv_save_active_operations",
                "orbitkv_save_active_metadata_bytes",
                "orbitkv_save_active_logical_bytes",
                "orbitkv_save_retained_operations",
                "orbitkv_inflight_bytes",
            ]
            before = wait_idle(channel_server.http_port, idle_names)
            submitted_bytes = save_bytes_snapshot(channel_server.http_port) - admission_before
            if submitted_bytes != 16384:
                raise RuntimeError("Rust submitted batch copied an unexpected number of bytes")
            admission = {
                "api": "prepare_save/submit_saves/cancel_save",
                "operation_ids": operation_ids,
                "pending_before_submit": 2,
                "canceled_before_submit": True,
                "canceled_done": canceled_status.done,
                "canceled_success": canceled_status.success,
                "submitted_cancel_refused": 2,
                "completed_successfully": 2,
                "prepare_copy_bytes": admission_prepared - admission_before,
                "submitted_copy_bytes": submitted_bytes,
                "started_mono_ns": admission_started,
                "completed_mono_ns": admission_completed,
                "physical_fault_termination_claimed": False,
            }
            if channel_server.ssd_cache_path is not None:
                write_deadline = time.monotonic() + 30
                while True:
                    written = fetch_orbitkv_metrics(channel_server.http_port)
                    if written.get("orbitkv_ssd_write_bytes_total", 0) >= writes_before + 16384:
                        break
                    assert time.monotonic() < write_deadline, written
                    time.sleep(0.01)
                before = wait_idle(
                    channel_server.http_port,
                    idle_names + ["orbitkv_ssd_write_inflight", "orbitkv_ssd_write_queue_pending"],
                )
                assert before["orbitkv_ssd_write_bytes_total"] > 0
            io_names = (
                "orbitkv_save_bytes_total",
                "orbitkv_load_bytes_total",
                "orbitkv_ssd_write_bytes_total",
                "orbitkv_ssd_prefetch_bytes_total",
                "orbitkv_ssd_uring_read_operations_total",
            )
            io_before_rejection = {name: before.get(name) for name in io_names}
            assert io_before_rejection["orbitkv_save_bytes_total"] >= 16384
            assert io_before_rejection["orbitkv_load_bytes_total"] is not None
            rejected_started = time.monotonic_ns()
            with pytest.raises(SaveNotSubmittedError, match="no longer active") as rejected:
                source.prepare_save(source_registration, [("bytes", source_indices, hashes)])
            rejected_ended = time.monotonic_ns()
            assert isinstance(rejected.value.__cause__, ValueError)
            observed = fetch_orbitkv_metrics(channel_server.http_port)
            io_after_rejection = {name: observed.get(name) for name in io_names}
            assert io_after_rejection == io_before_rejection, "I/O advanced across rejection"
            rejection = {
                "instance_id": source_id,
                "exception": type(rejected.value).__name__,
                "cause": type(rejected.value.__cause__).__name__,
                "started_mono_ns": rejected_started,
                "ended_mono_ns": rejected_ended,
                "io_before": io_before_rejection,
                "io_after": io_after_rejection,
                "publish_count_claimed": False,
                "physical_termination_claimed": False,
            }
            del handle, handles, canceled, source_registration
            if channel_server.ssd_cache_path is not None:
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
                    "rejected_save": rejection,
                    "producer_admission": admission,
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
