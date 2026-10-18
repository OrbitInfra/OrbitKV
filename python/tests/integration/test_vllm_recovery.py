"""Native recovery validation and the vLLM scheduler/worker CUDA handoff."""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from tests.support.ipc_credit import hold_counters, release_receipt
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.registration_campaign import identity as process_identity
from tests.support.registration_campaign import write

pytestmark = pytest.mark.integration


def config():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    return SimpleNamespace(
        kv_cache_groups=(
            SimpleNamespace(
                layer_names=("model.layers.1.mamba",),
                kv_cache_spec=MambaSpec(
                    block_size=16,
                    shapes=((64,), (64,)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_cache_mode="align",
                ),
            ),
            SimpleNamespace(
                layer_names=("model.layers.0.attn",),
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=32,
                    dtype=torch.float32,
                ),
            ),
        )
    )


def context(client, *, instance_id="instance", namespace="model/layout"):
    from orbitkv.vllm.config import ConnectorContext

    return ConnectorContext(
        instance_id=instance_id,
        namespace=namespace,
        block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        client=client,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("channel_server", ["dram", "ssd"], indirect=True)
@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_window_and_checkpoint_restore_only_required_destinations(channel_server, with_checkpoint):
    torch = pytest.importorskip("torch")
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from orbitkv import CacheManagerClient
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata
    from orbitkv.vllm.scheduler import SchedulerAdapter
    from orbitkv.vllm.worker import WorkerAdapter
    from tests.support.metrics import fetch_orbitkv_metrics

    groups = list(config().kv_cache_groups)
    window = SimpleNamespace(
        layer_names=("model.layers.2.window",),
        kv_cache_spec=SlidingWindowSpec(
            block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float32, sliding_window=33
        ),
    )
    groups = [window, groups[1], *([groups[0]] if with_checkpoint else [])]
    identity = f"window-checkpoint-{uuid.uuid4().hex}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    ctx = context(client, instance_id=identity, namespace=identity)
    cache_config = SimpleNamespace(kv_cache_groups=groups)
    client.start_session_watcher(identity, identity, 1, 1)
    scheduler = SchedulerAdapter(ctx, kv_cache_config=cache_config)
    worker = WorkerAdapter(ctx, kv_cache_config=cache_config)
    kv = torch.arange(16 * 2 * 16 * 32, device="cuda", dtype=torch.float32).reshape(
        16, 2, 16, 1, 32
    )
    swa = kv.clone() + 50000
    state = torch.arange(16 * 128, device="cuda", dtype=torch.float32).reshape(16, 128) + 100000
    expected = [tensor.clone() for tensor in (kv, swa, state)]
    hashes = [index.to_bytes(32, "little") for index in range(4)]
    buffers = {"model.layers.0.attn": kv, "model.layers.2.window": swa}
    if with_checkpoint:
        buffers["model.layers.1.mamba"] = (state[:, :64], state[:, 64:])

    def lookup(rid):
        req = SimpleNamespace(
            request_id=rid, num_tokens=65, block_hashes=hashes, shared_prefix_boundary=0
        )
        deadline = time.monotonic() + 15
        while (result := scheduler.get_num_new_matched_tokens(req, 0))[0] is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        return req, result

    try:
        worker.register_kv_caches(buffers)
        torch.cuda.synchronize()
        saved, message = client.save(
            worker._registration, [("model.layers.0.attn", [1, 2, 3, 4], hashes)]
        )
        assert saved, message
        assert lookup("missing-window")[1] == (0, False)
        saves = [("model.layers.2.window", [3, 4], hashes[2:])]
        saved, message = client.save(worker._registration, saves)
        assert saved, message
        if with_checkpoint:
            assert lookup("missing-checkpoint")[1] == (0, False)
            saved, message = client.save(
                worker._registration, [("model.layers.1.mamba", [4], hashes[3:])]
            )
            assert saved, message
        if channel_server.ssd_cache_path is not None:
            deadline = time.monotonic() + 15
            while True:
                metrics = fetch_orbitkv_metrics(channel_server.http_port)
                if metrics.get("orbitkv_ssd_write_bytes_total", 0) and not any(
                    metrics.get(name, 0)
                    for name in (
                        "orbitkv_ssd_write_inflight",
                        "orbitkv_ssd_write_queue_pending",
                        "orbitkv_inflight_bytes",
                    )
                ):
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=10
            )
            response.raise_for_status()
            assert response.json()["evicted_blocks"] == 6 + with_checkpoint
        req, result = lookup("restore-window")
        assert result == (64, False)
        destinations = [
            [0, 0, 12, 13],
            [8, 9, 10, 11],
            *([[0, 0, 0, 14]] if with_checkpoint else []),
        ]
        blocks = SimpleNamespace(
            get_block_ids=lambda: destinations,
            blocks=[[SimpleNamespace(block_hash=None) for _ in range(4)] for _ in groups],
        )
        scheduler.update_state_after_alloc(req, blocks, 64)
        kv[8:12].fill_(-1)
        swa[8:14].fill_(-1)
        state[8:15].fill_(-1)
        torch.cuda.synchronize()
        worker.start_load_kv(
            OrbitKVConnectorMetadata(load_intents=scheduler._pending_load_intents),
        )
        worker.wait_for_all_layers()
        worker.wait_for_save()
        assert not worker.get_transfer_results(set()).finished_recving
        torch.cuda.synchronize()
        assert torch.equal(kv[8:12], expected[0][1:5])
        assert torch.equal(swa[12:14], expected[1][3:5])
        assert (swa[8:12] == -1).all()
        if with_checkpoint:
            assert torch.equal(state[14], expected[2][4])
        assert (state[8:14] == -1).all()
        metrics = fetch_orbitkv_metrics(channel_server.http_port)
        payload = (
            6 * kv[0].numel() * kv.element_size()
            + with_checkpoint * state[0].numel() * state.element_size()
        )
        assert metrics["orbitkv_load_bytes_total"] == payload
        if channel_server.ssd_cache_path is not None:
            assert metrics["orbitkv_ssd_prefetch_bytes_total"] == payload
        assert metrics.get("orbitkv_query_reserved_bytes", 0) == 0
    finally:
        scheduler.shutdown()
        worker.shutdown()
        client.close()


@pytest.mark.gpu
@pytest.mark.parametrize("fence", [True, False], ids=["fenced", "no-fence-control"])
@pytest.mark.parametrize(
    "channel_server",
    [{"tier": "dram", "installed": True}, {"tier": "ssd", "installed": True}],
    ids=["dram", "ssd"],
    indirect=True,
)
def test_save_fences_its_producer_without_waiting_for_unrelated_gpu_work(
    channel_server, tmp_path, fence
):
    torch = pytest.importorskip("torch")
    from orbitkv import BlockHashes, CacheManagerClient, QueryReady
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata, SaveIntent
    from orbitkv.vllm.worker import WorkerAdapter

    if importlib.metadata.version("vllm") != "0.31.0":
        pytest.fail("Prepared Save adapter gate requires official vLLM 0.31.0")
    client = CacheManagerClient(channel_server.bootstrap_socket)
    identity = f"producer-{uuid.uuid4().hex}"
    client.start_session_watcher(identity, identity, 1, 1)
    worker = WorkerAdapter(context(client, instance_id=identity, namespace=identity))
    kv = torch.zeros((2, 8, 16, 1, 32), device="cuda", dtype=torch.float32)
    payload = torch.arange(2 * 16 * 32, dtype=torch.float32).reshape(2, 16, 1, 32)
    producer_payload = payload.to("cuda")
    expected = torch.full_like(kv, -167, device="cpu")
    expected[:, 1] = payload
    expected[:, 2] = payload
    producer, unrelated = torch.cuda.Stream(), torch.cuda.Stream()
    unrelated_done = torch.cuda.Event()
    producer_done = torch.cuda.Event()
    submitted = threading.Event()
    overlapped = []
    prepared_ids = []
    producer_ready_at_submit = []
    hashes = BlockHashes([b"h" * 32])
    counters = contextlib.ExitStack()

    def prepare(*args):
        handle = client.prepare_save(*args)
        prepared_ids.append(handle.operation_id)
        return handle

    def submit(handles):
        overlapped.append(not unrelated_done.query())
        producer_ready_at_submit.append(producer_done.query())
        submitted.set()
        return client.submit_saves(handles)

    if not fence:
        make_task = worker._make_save_task

        class WithoutFence:
            def __init__(self, event):
                self.event = event

            def synchronize(self):
                pass

        def omit_fence(*args, **kwargs):
            task = make_task(*args, **kwargs)
            task.ready = WithoutFence(task.ready)
            return task

        worker._make_save_task = omit_fence

    try:
        worker.register_kv_caches({"layer": kv})
        original_registration = worker._registration
        write(
            tmp_path / "original.credit-owner.json",
            {
                "owner": process_identity(os.getpid()),
                "instance_id": identity,
                "evidence": json.loads(original_registration.ipc_credit_evidence),
            },
        )
        original_counters = hold_counters(counters, tmp_path / "original.credit-owner.json")
        channel_server.record_runtime()
        worker._client = MagicMock(wraps=client)
        worker._client.prepare_save.side_effect = prepare
        worker._client.submit_saves.side_effect = submit
        kv.fill_(-167)
        torch.cuda.synchronize()
        # A delayed producer verifies that the background thread cannot read
        # the initial zeros. Independent work must not extend its save fence.
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_500_000_000)
            kv[:, 1].copy_(producer_payload)
            producer_done.record()
        with torch.cuda.stream(unrelated):
            torch.cuda._sleep(3_000_000_000)
            unrelated_done.record()
        with torch.cuda.stream(producer):
            assert not producer_done.query(), "producer completed before handoff fence test"
            worker._current_metadata = OrbitKVConnectorMetadata(
                save_intents={
                    "save": SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"h" * 32,))
                }
            )
            worker.wait_for_save()

        assert len(prepared_ids) == 1, "producer did not reserve native ownership before return"
        assert submitted.wait(10), "the save worker never submitted Publish"
        worker._save_queue.join()
        assert overlapped == [True], "Publish waited for unrelated GPU work"
        assert producer_ready_at_submit == [fence], "native submission did not prove its fence"
        assert worker.get_transfer_results({"save"}).finished_sending == {"save"}
        worker._client.prepare_save.assert_called_once()
        worker._client.submit_saves.assert_called_once()
        worker._client.wait_save.assert_called_once()
        worker._client.start_save.assert_not_called()
        if channel_server.ssd_cache_path:
            synced = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/sync", timeout=35
            )
            synced.raise_for_status()
            assert synced.json()["local_flush_complete"] is True
            from tests.support.cache_manager import evict_dram_after_ssd_writes

            evict_dram_after_ssd_writes(channel_server.http_port)
        before_restore = fetch_orbitkv_metrics(channel_server.http_port)
        deadline = time.monotonic() + 10
        while True:
            ready = client.query_prefetch(identity, hashes, "restore")
            if isinstance(ready, QueryReady) and ready.num_hit_blocks == 1:
                break
            assert time.monotonic() < deadline, ready
            time.sleep(0.01)
        restored = client.wait_restore(
            client.start_restore(
                identity,
                0,
                0,
                [["layer"]],
                [(ready.lease, [[2]])],
                ready_stream=torch.cuda.current_stream(0).cuda_stream,
            ),
            timeout=10,
        )
        assert restored.success, restored.message
        producer.synchronize()
        torch.cuda.synchronize()
        actual = kv.cpu()
        oracle_matches = torch.equal(actual, expected)
        if fence:
            assert oracle_matches, "complete payload/untouched sentinel differs"
            actual[0, 2, 0, 0, 0] += 1
            assert not torch.equal(actual, expected), "mutation was invisible to full-byte oracle"
            actual[0, 2, 0, 0, 0] -= 1
        else:
            assert not oracle_matches, "no-fence control unexpectedly passed the positive oracle"
            untouched = expected.clone()
            untouched[:, 2].fill_(-167)
            assert torch.equal(actual, untouched), "negative control corrupted unowned bytes"
        (tmp_path / "vllm-prepared-save.payload.bin").write_bytes(actual.numpy().tobytes())
        after_restore = fetch_orbitkv_metrics(channel_server.http_port)
        if channel_server.ssd_cache_path:
            for name in (
                "orbitkv_ssd_prefetch_bytes_total",
                "orbitkv_ssd_uring_read_operations_total",
            ):
                assert name in after_restore and after_restore[name] > before_restore.get(name, 0)
        worker.unregister_context()
        write(tmp_path / "original.credit-release.json", release_receipt(original_counters))
        assert worker._registration is None
        worker.register_kv_caches({"layer": kv})
        replacement = worker._registration
        write(
            tmp_path / "replacement.credit-owner.json",
            {
                "owner": process_identity(os.getpid()),
                "instance_id": identity,
                "evidence": json.loads(replacement.ipc_credit_evidence),
            },
        )
        replacement_counters = hold_counters(counters, tmp_path / "replacement.credit-owner.json")
        assert replacement.generation != original_registration.generation
        from orbitkv import SaveNotSubmittedError

        with pytest.raises(SaveNotSubmittedError, match="no longer active"):
            client.prepare_save(original_registration, [("layer", [1], [b"h" * 32])])
        receipt = {
            "pid": os.getpid(),
            "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
            "release": "0.31.0",
            "instance_id": identity,
            "prepared_ids": prepared_ids,
            "producer_reserved": True,
            "producer_pending_at_handoff": True,
            "prepared_submission_consumed": True,
            "full_destination_bytes": kv[:, 2].numel() * kv.element_size(),
            "full_buffer_bytes": kv.numel() * kv.element_size(),
            "payload_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
            "full_byte_oracle": True,
            "mutation_rejected": True,
            "fence_enabled": fence,
            "positive_byte_oracle_passed": oracle_matches,
            "producer_ready_at_submit": producer_ready_at_submit,
            "qualification_passed": fence,
            "tier": "ssd-uring" if channel_server.ssd_cache_path else "dram",
            "read_before": before_restore,
            "read_after": after_restore,
            "replacement_generation": replacement.generation.hex(),
            "registration_generation": original_registration.generation.hex(),
            "stale_registration_rejected": True,
            "physical_fault_termination_claimed": False,
        }
    finally:
        try:
            producer.synchronize()
            unrelated.synchronize()
            worker.shutdown()
            try:
                if "replacement_counters" in locals():
                    write(
                        tmp_path / "replacement.credit-release.json",
                        release_receipt(replacement_counters),
                    )
            finally:
                client.close()
        finally:
            counters.close()
    if worker._save_thread.is_alive() or worker._registration is not None:
        raise RuntimeError("vLLM worker or GPU registration survived normal shutdown")
    receipt.update(
        workers_joined=True,
        registration_retired=True,
        client_closed=True,
        completed_mono_ns=time.monotonic_ns(),
    )
    (tmp_path / "vllm-prepared-save.native-release.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


@pytest.mark.parametrize(
    ("attention", "positions", "tokens", "expected"),
    [
        (4, [1, 3], 128, 32),
        (4, [1, 3], 129, 64),
        (4, [], 129, 0),
        (4, [3], 128, 0),
        (2, [1], 129, 32),
    ],
    ids=[
        "clamp-to-earlier-checkpoint",
        "last-checkpoint",
        "missing-checkpoint",
        "clamp-has-no-checkpoint",
        "partial-attention",
    ],
)
def test_native_contract_gates_vllm_hits(attention, positions, tokens, expected):
    cache_config = config()
    from orbitkv import QueryCandidates, QueryReady
    from orbitkv.vllm.scheduler import SchedulerAdapter

    client = MagicMock()
    client.query_candidates.side_effect = [
        QueryCandidates(list(range(attention))),
        QueryCandidates(positions),
    ]
    client.read_recovery.side_effect = lambda *args: (
        QueryReady((args[6] - args[5]) // 16, b"attention")
        if args[-1] == 0
        else QueryReady(1, b"state", [(args[6] - args[5]) // 16 - 1])
    )
    scheduler = SchedulerAdapter(context(client), kv_cache_config=cache_config)
    try:
        req = SimpleNamespace(
            request_id="r",
            num_tokens=tokens,
            shared_prefix_boundary=0,
            block_hashes=[bytes([index]) for index in range(tokens // 16)],
        )
        assert scheduler.get_num_new_matched_tokens(req, 64) == (expected, False)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("positions", [[1, 1], [3, 1]])
def test_native_contract_rejects_duplicate_or_unordered_evidence(positions):
    cache_config = config()
    from orbitkv import QueryCandidates
    from orbitkv.vllm.scheduler import SchedulerAdapter

    client = MagicMock()
    client.query_candidates.side_effect = [
        QueryCandidates([0, 1, 2, 3]),
        QueryCandidates(positions),
    ]
    scheduler = SchedulerAdapter(context(client), kv_cache_config=cache_config)
    req = SimpleNamespace(
        request_id="r",
        num_tokens=129,
        shared_prefix_boundary=0,
        block_hashes=[bytes([index]) for index in range(8)],
    )
    try:
        with pytest.raises(ValueError, match="page ends"):
            scheduler.get_num_new_matched_tokens(req, 64)
        client.read_recovery.assert_not_called()
        client.release.assert_not_called()
        assert not scheduler._pending_query_probes
    finally:
        scheduler.shutdown()


@pytest.mark.gpu
@pytest.mark.parametrize("channel_server", ["dram", "ssd"], indirect=True)
def test_complete_checkpoint_restores_through_vllm_worker(channel_server):
    torch = pytest.importorskip("torch")
    cache_config = config()
    from orbitkv import CacheManagerClient
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata
    from orbitkv.vllm.scheduler import SchedulerAdapter
    from orbitkv.vllm.worker import WorkerAdapter
    from tests.support.metrics import fetch_orbitkv_metrics

    client = CacheManagerClient(channel_server.bootstrap_socket)
    identity = f"recovery-{uuid.uuid4().hex}"
    ctx = context(client, instance_id=identity, namespace=identity)
    client.start_session_watcher(identity, identity, 1, 1)
    scheduler = SchedulerAdapter(ctx, kv_cache_config=cache_config)
    worker = WorkerAdapter(ctx, kv_cache_config=cache_config)
    kv = torch.arange(16 * 2 * 16 * 32, device="cuda", dtype=torch.float32).reshape(
        16, 2, 16, 1, 32
    )
    state = torch.arange(16 * 128, device="cuda", dtype=torch.float32).reshape(16, 128) + 10000
    expected_kv, expected_state = kv.clone(), state.clone()
    hashes = [index.to_bytes(32, "little") for index in range(8)]
    try:
        worker.register_kv_caches(
            {
                "model.layers.0.attn": kv,
                "model.layers.1.mamba": (state[:, :64], state[:, 64:]),
            }
        )
        req = SimpleNamespace(
            request_id="cold",
            num_tokens=128,
            block_hashes=hashes,
            shared_prefix_boundary=0,
        )
        deadline = time.monotonic() + 15
        while (result := scheduler.get_num_new_matched_tokens(req, 64))[0] is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == (0, False)
        cold_metrics = fetch_orbitkv_metrics(channel_server.http_port)
        assert cold_metrics["orbitkv_cache_candidate_misses_total"] == 8
        assert cold_metrics["orbitkv_hll_total_requests"] > 0
        assert cold_metrics.get("orbitkv_ssd_prefetch_bytes_total", 0) == 0
        assert cold_metrics.get("orbitkv_query_reserved_bytes", 0) == 0
        torch.cuda.synchronize()
        ok, message = client.save(
            worker._registration,
            [
                ("model.layers.0.attn", [1, 2, 3, 4], hashes[4:]),
                ("model.layers.1.mamba", [1, 3], [hashes[5], hashes[7]]),
            ],
        )
        assert ok, message
        if channel_server.ssd_cache_path is not None:
            deadline = time.monotonic() + 15
            while True:
                metrics = fetch_orbitkv_metrics(channel_server.http_port)
                if metrics.get("orbitkv_ssd_write_bytes_total", 0) and not any(
                    metrics.get(name, 0)
                    for name in (
                        "orbitkv_ssd_write_inflight",
                        "orbitkv_ssd_write_queue_pending",
                        "orbitkv_inflight_bytes",
                    )
                ):
                    break
                assert time.monotonic() < deadline, metrics
                time.sleep(0.01)
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=10
            )
            response.raise_for_status()
            assert response.json()["evicted_blocks"] == 6

        req = SimpleNamespace(
            request_id="restore",
            num_tokens=128,
            block_hashes=hashes,
            shared_prefix_boundary=0,
        )
        deadline = time.monotonic() + 15
        while (result := scheduler.get_num_new_matched_tokens(req, 64))[0] is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == (32, False)
        probe = scheduler._pending_query_probes["restore"]
        assert probe.selected_boundary == 96
        assert probe.leased_blocks == 2
        assert probe.recovery_hold.last_position == 1

        blocks = SimpleNamespace(
            get_block_ids=lambda: ([0, 1, 2, 3, 8, 9, 10, 11],) * 2,
            blocks=[
                [SimpleNamespace(block_hash=b"hbm" if index < 4 else None) for index in range(8)]
                for _ in range(2)
            ],
        )
        scheduler.update_state_after_alloc(req, blocks, 32)
        kv[8:12].fill_(-1)
        state[8:12].fill_(-1)
        torch.cuda.synchronize()
        worker.start_load_kv(
            OrbitKVConnectorMetadata(load_intents=scheduler._pending_load_intents),
        )
        worker.wait_for_all_layers()
        worker.wait_for_save()
        assert not worker.get_transfer_results(set()).finished_recving
        torch.cuda.synchronize()
        assert torch.equal(kv[8:10], expected_kv[1:3])
        assert torch.equal(state[9], expected_state[1])  # Both conv and temporal tensors.
        assert (kv[10:12] == -1).all()  # Extra leased pages have no destination.
        assert (state[[8, 10, 11]] == -1).all()
        assert worker._restore is None
        metrics = fetch_orbitkv_metrics(channel_server.http_port)
        assert metrics["orbitkv_cache_candidate_hits_total"] == 6
        assert metrics["orbitkv_cache_candidate_misses_total"] == 10
        # One cold and one warm discovery, regardless of HLL window count.
        assert (
            metrics["orbitkv_hll_total_requests"] == 2 * cold_metrics["orbitkv_hll_total_requests"]
        )
        if channel_server.ssd_cache_path is not None:
            page_bytes = kv[0].numel() * kv.element_size()
            state_bytes = state[0].numel() * state.element_size()
            assert (
                metrics.get("orbitkv_ssd_prefetch_bytes_total", 0) == 2 * page_bytes + state_bytes
            )
            assert metrics.get("orbitkv_load_bytes_total", 0) == 2 * page_bytes + state_bytes
            assert metrics.get("orbitkv_query_reserved_bytes", 0) == 0
    finally:
        scheduler.shutdown()
        worker.shutdown()
        client.close()
