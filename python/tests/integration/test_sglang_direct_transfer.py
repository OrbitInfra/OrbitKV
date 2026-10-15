"""Prove the SGLang-style GPU buffer layout survives a real D2H/H2D round trip."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture
def load_linker(monkeypatch):
    pytest.importorskip("sglang")
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.linker import OrbitKVLinker

    linker = object.__new__(OrbitKVLinker)
    linker.instance_id = "layered-restore"
    linker.device_id = 0
    linker._torch_device = "cuda:0"
    monkeypatch.setattr("orbitkv.sglang.linker.torch.cuda.Event", MagicMock())
    monkeypatch.setattr(
        "orbitkv.sglang.linker.torch.cuda.current_stream", lambda: SimpleNamespace(cuda_stream=17)
    )
    linker.layout = SimpleNamespace(
        num_layers=1,
        pools={
            "kv": SimpleNamespace(layer_names=["kv:0"], entry=SimpleNamespace(layer_mapping={0: 0}))
        },
    )
    linker._load_error = None
    linker._load_queue = queue.Queue()
    linker._completed_loads = queue.Queue()
    linker.layer_done_counter = _LayerDoneCounter(linker.layout)
    linker.client = MagicMock()
    return linker


@pytest.mark.parametrize("failure", ["submit", "enqueue", "drain", "timeout"])
def test_linker_failure_never_acknowledges_gpu_destinations(load_linker, failure):
    from orbitkv.sglang.linker import _Load

    linker = load_linker
    method = {
        "submit": "start_restore",
        "enqueue": "wait_restore_enqueued",
        "drain": "wait_restore",
        "timeout": "wait_restore",
    }[failure]
    error = (
        TimeoutError("restore deadline")
        if failure == "timeout"
        else ConnectionError("lost completion")
    )
    getattr(linker.client, method).side_effect = error
    index = linker.layer_done_counter.update_producer()
    linker._load_queue.put((index, [_Load(str(i), (("kv", bytes([i]), (i,)),)) for i in range(12)]))
    linker._load_queue.put(None)
    thread = threading.Thread(target=linker._load_worker, daemon=True)
    thread.start()
    if failure in ("submit", "enqueue"):
        with pytest.raises(ConnectionError):
            linker.layer_done_counter.set_consumer(index)
    else:
        linker.layer_done_counter.set_consumer(index)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert linker._completed_loads.empty()
    linker.client.release.assert_not_called()
    for observe in (linker.num_completed_loads, linker.pop_completed_load):
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            observe()


def test_layer_publication_does_not_acknowledge_a_partially_drained_batch(load_linker):
    from orbitkv.sglang.linker import _Load

    linker = load_linker
    index = linker.layer_done_counter.update_producer()
    draining, finish = threading.Event(), threading.Event()

    def complete(handle, **kwargs):
        draining.set()
        assert finish.wait(timeout=5)
        return SimpleNamespace(success=True)

    linker.client.wait_restore.side_effect = complete
    pending = [_Load(str(i), (("kv", bytes([i]), (i,)),)) for i in range(12)]
    linker._load_queue.put((index, pending))
    linker._load_queue.put(None)
    thread = threading.Thread(target=linker._load_worker, daemon=True)
    thread.start()
    try:
        linker.layer_done_counter.set_consumer(index)
        assert draining.wait(timeout=1)
        assert linker._completed_loads.empty()
        linker.client.start_restore.assert_called_once()
        assert len(linker.client.start_restore.call_args.args[4]) == len(pending)
    finally:
        finish.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert linker.pop_completed_load() == [load.rid for load in pending]
    linker.client.release.assert_not_called()


def test_cancel_before_forward_releases_only_unsubmitted_leases(load_linker):
    from orbitkv.sglang.linker import _Load

    linker = load_linker
    index = linker.layer_done_counter.update_producer()
    linker._load_queue.put((index, [_Load("cancelled", (("kv", b"lease", (1,)),))]))
    linker._load_queue.put(None)
    linker.layer_done_counter.cancel_pending()
    linker._load_worker()
    linker.client.start_restore.assert_not_called()
    linker.client.release.assert_called_once_with(b"lease")
    assert linker._completed_loads.empty()


@pytest.mark.parametrize(
    ("layer_count", "page_count", "page_first", "channel_server"),
    [
        (2, 2, False, "dram"),
        (36, 64, False, "dram"),
        (36, 64, True, "dram"),
        (36, 64, False, "ssd"),
        (36, 64, True, "ssd"),
    ],
    indirect=["channel_server"],
)
def test_direct_page_transfer_overwrites_poisoned_gpu_slots(
    channel_server, layer_count, page_count, page_first, request
):
    from orbitkv import BlockHashes

    torch = pytest.importorskip("torch")
    from orbitkv import CacheManagerClient, QueryLoading, QueryReady
    from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.linker import OrbitKVLinker, _Load

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    page_size = 64
    num_blocks = page_count + 4
    shape = (page_size * num_blocks, 2, 128)
    tensors = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for _ in range(layer_count)]
    expected = []
    for layer, tensor in enumerate(tensors):
        values = torch.arange(page_size * page_count * 2 * 128, device="cuda").reshape(
            page_size * page_count, 2, 128
        )
        tensor[page_size : page_size * (page_count + 1)] = (
            (values + layer * 13) % 4096 / 256 - 8
        ).to(torch.bfloat16)
        original = tensor[page_size : page_size * (page_count + 1)].clone()
        if channel_server.ssd_cache_path and request.config.getoption("--storage-codec") == "fp8":
            original = original.to(torch.float8_e4m3fn).to(original.dtype)
        expected.append(original)
    torch.cuda.synchronize()

    names = [f"kv:{layer}" for layer in range(layer_count)]
    hashes = [hashlib.sha256(f"page-{i}".encode()).digest() for i in range(page_count)]
    instance = f"sglang-layout-{uuid.uuid4().hex}"
    namespace = f"sglang-layout:{instance}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    try:
        client.start_session_watcher(instance, namespace, 1, 1)
        wrappers = [serialize_gpu_buffer(tensor) for tensor in tensors]
        block_bytes = page_size * tensors[0].stride(0) * tensors[0].element_size()
        registration = client.register_context_batch(
            instance,
            namespace,
            0,
            0,
            1,
            1,
            resolve_device_id(),
            names,
            wrappers,
            [num_blocks] * layer_count,
            [block_bytes] * layer_count,
            [0] * layer_count,
            [1] * layer_count,
            "direct",
            page_first,
            tensors=tensors,
            layer_formats=["bf16"] * layer_count,
        )

        success, message = client.save(
            registration,
            [(name, list(range(1, page_count + 1)), hashes) for name in names],
        )
        assert success, message
        if channel_server.ssd_cache_path is not None:
            saved_bytes = sum(tensor.numel() * tensor.element_size() for tensor in expected)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                observed = fetch_orbitkv_metrics(channel_server.http_port)
                if 0 < observed.get("orbitkv_ssd_write_bytes_total", 0) <= saved_bytes and not any(
                    observed.get(name, 0)
                    for name in ("orbitkv_ssd_write_inflight", "orbitkv_ssd_write_queue_pending")
                ):
                    break
                time.sleep(0.01)
            else:
                pytest.fail(f"SSD writes did not complete: {observed}")
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=10
            )
            response.raise_for_status()
            cleanup = response.json()
            assert cleanup["evicted_blocks"] == page_count
            assert cleanup["still_referenced_blocks"] == 0

            # Lose interest while SSD work owns real source buffers. A late
            # result must release its lease without a readiness poll or GC.
            abandoned = CacheManagerClient(channel_server.bootstrap_socket)
            survivor = CacheManagerClient(channel_server.bootstrap_socket)
            try:
                outcome = abandoned.query_prefetch(instance, BlockHashes(hashes), "abandoned")
                assert isinstance(outcome, QueryLoading)
                surviving_result = survivor.query_prefetch(
                    instance, BlockHashes(hashes), "abandoned"
                )
                if page_first:
                    abandoned.close()
                else:
                    abandoned.cancel_query(instance, "abandoned")
                    abandoned.cancel_query(instance, "abandoned")
                deadline = time.monotonic() + 5
                while isinstance(surviving_result, QueryLoading):
                    assert time.monotonic() < deadline
                    surviving_result = survivor.query_prefetch(
                        instance, BlockHashes(hashes), "abandoned"
                    )
                    time.sleep(0.001)
                assert surviving_result.num_hit_blocks == page_count
                survivor.release(surviving_result.lease)
                quiet = 0
                while quiet < 3:
                    observed = fetch_orbitkv_metrics(channel_server.http_port)
                    complete = (
                        observed.get("orbitkv_ssd_prefetch_bytes_total", 0) >= saved_bytes
                        and observed.get("orbitkv_ssd_prefetch_inflight", 0) == 0
                        and observed.get("orbitkv_query_reserved_bytes", 0) == 0
                    )
                    quiet = quiet + 1 if complete else 0
                    assert time.monotonic() < deadline, observed
                    time.sleep(0.02)
                assert observed["orbitkv_ssd_prefetch_bytes_total"] == saved_bytes
                response = requests.post(
                    f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup",
                    timeout=10,
                )
                response.raise_for_status()
                assert response.json()["evicted_blocks"] > 0
                assert response.json()["still_referenced_blocks"] == 0
            finally:
                abandoned.close()
                survivor.close()

            # An unpolled enqueue warmup leaves evictable DRAM pages, with no
            # ready lease/budget. The later demand must revalidate and lease them.
            before = fetch_orbitkv_metrics(channel_server.http_port)
            warm_hashes = hashes[:4]
            assert client.warm_prefix(instance, BlockHashes(warm_hashes), "queued-warmup")
            deadline = time.monotonic() + 5
            while True:
                observed = fetch_orbitkv_metrics(channel_server.http_port)
                if (
                    observed.get("orbitkv_ssd_prefetch_bytes_total", 0)
                    > before.get("orbitkv_ssd_prefetch_bytes_total", 0)
                    and observed.get("orbitkv_ssd_prefetch_inflight", 0) == 0
                    and observed.get("orbitkv_query_reserved_bytes", 0) == 0
                ):
                    break
                assert time.monotonic() < deadline, observed
                time.sleep(0.01)
            demanded = client.query_prefetch(instance, BlockHashes(warm_hashes), "queued-warmup")
            assert isinstance(demanded, QueryReady)
            assert demanded.num_hit_blocks == len(warm_hashes)
            assert demanded.lease
            client.release(demanded.lease)
            after = fetch_orbitkv_metrics(channel_server.http_port)
            warm_bytes = saved_bytes * len(warm_hashes) // page_count
            assert after["orbitkv_warmup_prepared_bytes_total"] == warm_bytes
            assert after["orbitkv_warmup_pending_bytes"] == warm_bytes
            assert after.get("orbitkv_warmup_restored_bytes_total", 0) == 0
            assert (
                after["orbitkv_ssd_prefetch_bytes_total"]
                == observed["orbitkv_ssd_prefetch_bytes_total"]
            )
            # Releasing a lookup lease is not use. Only the last page owner
            # records unused bytes, and rereading the same key is a new cohort.
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup",
                timeout=10,
            )
            response.raise_for_status()
            assert response.json()["still_referenced_blocks"] == 0
            discarded = fetch_orbitkv_metrics(channel_server.http_port)
            assert discarded["orbitkv_warmup_unused_bytes_total"] == warm_bytes
            assert discarded["orbitkv_warmup_pending_bytes"] == 0
            assert discarded["orbitkv_warmup_wait_byte_seconds_total"] > 0
            assert client.warm_prefix(instance, BlockHashes(warm_hashes), "next-warmup")
            deadline = time.monotonic() + 5
            while True:
                observed = fetch_orbitkv_metrics(channel_server.http_port)
                if (
                    observed["orbitkv_warmup_prepared_bytes_total"] == 2 * warm_bytes
                    and observed.get("orbitkv_query_reserved_bytes", 0) == 0
                ):
                    break
                assert time.monotonic() < deadline, observed
                time.sleep(0.01)
            client.cancel_query(instance, "next-warmup")
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.synchronize()

        linker = object.__new__(OrbitKVLinker)
        linker.instance_id = instance
        linker.device_id = resolve_device_id()
        linker._torch_device = tensors[0].device
        linker.layout = SimpleNamespace(
            num_layers=layer_count,
            pools={
                "kv": SimpleNamespace(
                    layer_names=names,
                    entry=SimpleNamespace(layer_mapping={i: i for i in range(layer_count)}),
                )
            },
        )
        linker.client = client
        linker._load_error = None
        linker._load_queue = queue.Queue()
        linker._completed_loads = queue.Queue()
        linker.layer_done_counter = _LayerDoneCounter(linker.layout)
        pending = []
        # One consuming forward combines independently pinned requests.
        for start in range(0, page_count, 4):
            end = min(start + 4, page_count)
            rid = f"sglang-poison-{start}"
            deadline = time.monotonic() + 30
            while True:
                lookup = client.query_prefetch(instance, BlockHashes(hashes[start:end]), rid)
                if isinstance(lookup, QueryReady) or time.monotonic() >= deadline:
                    break
                time.sleep(0.001)
            assert isinstance(lookup, QueryReady)
            assert lookup.num_hit_blocks == end - start
            pending.append(_Load(rid, (("kv", lookup.lease, tuple(range(start + 3, end + 3))),)))
        index = linker.layer_done_counter.update_producer()
        linker._load_queue.put((index, pending))
        linker._load_queue.put(None)
        thread = threading.Thread(target=linker._load_worker, daemon=True)
        thread.start()
        linker.layer_done_counter.set_consumer(index)
        thread.join(timeout=30)
        assert not thread.is_alive()
        linker.layer_done_counter.wait_until(layer_count - 1)
        assert linker.pop_completed_load() == [load.rid for load in pending]
        torch.cuda.synchronize()
        for tensor, original in zip(tensors, expected, strict=True):
            assert torch.equal(tensor[page_size * 3 : page_size * (page_count + 3)], original)
        if channel_server.ssd_cache_path is not None:
            observed = fetch_orbitkv_metrics(channel_server.http_port)
            # One drained read after cancellation, then the consumed restore.
            assert observed["orbitkv_ssd_prefetch_bytes_total"] == 2 * saved_bytes + warm_bytes
            assert observed["orbitkv_load_bytes_total"] == saved_bytes
            assert observed["orbitkv_warmup_restored_bytes_total"] == warm_bytes
            assert observed["orbitkv_warmup_unused_bytes_total"] == warm_bytes
            assert observed["orbitkv_warmup_pending_bytes"] == 0
            assert observed["orbitkv_warmup_prepared_bytes_total"] == 2 * warm_bytes
    finally:
        client.unregister_context(instance)
        client.close()
