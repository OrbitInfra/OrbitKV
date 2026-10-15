"""Process-boundary integration for the node-local iceoryx2 client."""

import importlib
import os
import select
import sys
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_python_channel_lifecycle_reaches_real_cache_manager(channel_server):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    connection = orbitkv_native.CacheManagerClient(channel_server.bootstrap_socket)
    service_name, session_epoch = connection.service_name, connection.session_epoch
    connection.close()

    client = orbitkv_native.ChannelProbeClient(service_name, session_epoch)
    assert client.service_name == service_name
    assert client.session_epoch == session_epoch
    assert client.ping(value=41, request_id=101) == 42

    stale_client = orbitkv_native.ChannelProbeClient(service_name, session_epoch - 1)
    with pytest.raises(orbitkv_native.OrbitKVError, match="StaleSession"):
        stale_client.ping(request_id=102)

    client.shutdown(request_id=103)
    assert channel_server.process is not None
    assert channel_server.process.wait(timeout=5) == 0, channel_server.read_logs()


def test_channel_lifecycle_runs_with_no_grpc_listener(channel_server):
    import socket

    native = importlib.import_module("orbitkv.orbitkv")
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", channel_server.port)) != 0
    client = native.CacheManagerClient(channel_server.bootstrap_socket)
    assert client.health()[0]
    client.start_session_watcher("local-session", "test", 1, 1)
    assert client.unregister_context("local-session")[0]
    client.close()
    with pytest.raises(native.OrbitKVError, match="reconnect"):
        client.health()


@pytest.mark.parametrize("diagnostic", [False, True], ids=["save", "save-diagnostic"])
def test_query_bundle_uses_bootstrapped_arena_and_core(
    channel_server, channel_client_context, request, diagnostic
):
    orbitkv_native = importlib.import_module("orbitkv.orbitkv")
    torch = pytest.importorskip("torch")
    bootstrap_socket = channel_server.bootstrap_socket
    assert bootstrap_socket is not None
    query_client = orbitkv_native.CacheManagerClient(bootstrap_socket)
    request.addfinalizer(query_client.close)
    assert query_client.notification_fd >= 0

    def query(**kwargs):
        kwargs["block_hashes"] = orbitkv_native.BlockHashes(kwargs["block_hashes"])
        result = query_client.query_prefetch(**kwargs)
        deadline = time.monotonic() + 5
        while isinstance(result, orbitkv_native.QueryLoading):
            assert result.admitted
            assert time.monotonic() < deadline
            result = query_client.query_prefetch(**kwargs)
            time.sleep(0.001)
        return result

    with pytest.raises(orbitkv_native.OrbitKVError, match="Invalid"):
        query(
            instance_id="missing-instance",
            block_hashes=[],
            req_id="missing-query",
        )

    result = query(
        instance_id=channel_client_context.instance_id,
        block_hashes=[],
        req_id="registered-cold-query",
    )
    assert isinstance(result, orbitkv_native.QueryReady)
    assert result.num_hit_blocks == 0
    assert result.lease == b""

    block_hashes = [bytes([1]) * 32, bytes([2]) * 32]
    expected = channel_client_context.get_kv_cache()[:, 0:2].cpu().clone()
    registration = channel_client_context.registration
    assert isinstance(registration, orbitkv_native.GpuRegistration)
    saves = [(channel_client_context._layer_names[0], [0, 1], block_hashes)]
    if diagnostic:
        ok, message, observation = channel_client_context.client.save_diagnostic(
            registration, saves
        )
        assert observation.session_epoch == channel_client_context.client.session_epoch
        assert observation.request_id > 0 and observation.session_token > 0
        assert (
            0 < observation.submitted_mono_ns <= observation.returned_mono_ns <= time.monotonic_ns()
        )
    else:
        ok, message = channel_client_context.client.save(registration, saves)
    assert ok, message
    channel_client_context.get_kv_cache().zero_()
    torch.cuda.synchronize()

    deadline = time.monotonic() + 5
    while True:
        result = query(
            instance_id=channel_client_context.instance_id,
            block_hashes=block_hashes,
            req_id="registered-warm-query",
        )
        if isinstance(result, orbitkv_native.QueryReady) and result.num_hit_blocks == 2:
            break
        assert time.monotonic() < deadline, f"cache query never became ready: {result!r}"
        time.sleep(0.05)

    lease = result.lease
    assert isinstance(lease, bytes) and lease
    assert (
        orbitkv_native.QueryReady(result.num_hit_blocks, lease, result.hit_positions).lease == lease
    )
    restore_client = channel_client_context.client
    operation_id = restore_client.start_restore(
        instance_id=channel_client_context.instance_id,
        tp_rank=0,
        device_id=0,
        layer_groups=[channel_client_context._layer_names],
        loads=[(lease, [[2, 3]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    readable, _, _ = select.select([restore_client.notification_fd], [], [], 5)
    assert readable == [restore_client.notification_fd]
    assert int.from_bytes(os.read(restore_client.notification_fd, 8), byteorder=sys.byteorder) >= 1
    status = restore_client.poll_restore(operation_id)
    assert status.success, status.message
    restored = channel_client_context.get_kv_cache()[:, 2:4].cpu()
    assert restored.equal(expected)
    rejected = restore_client.start_restore(
        instance_id=channel_client_context.instance_id,
        tp_rank=0,
        device_id=0,
        layer_groups=[channel_client_context._layer_names],
        loads=[(result.lease, [[0, 1]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    status = restore_client.wait_restore(rejected, timeout=5)
    assert status.done and not status.success
    assert "lease" in status.message.lower()

    second = query(
        instance_id=channel_client_context.instance_id,
        block_hashes=block_hashes,
        req_id="registered-sync-restore-query",
    )
    assert isinstance(second, orbitkv_native.QueryReady)
    channel_client_context.get_kv_cache()[:, 0:2].zero_()
    torch.cuda.synchronize()
    second_restore = restore_client.start_restore(
        instance_id=channel_client_context.instance_id,
        tp_rank=0,
        device_id=0,
        layer_groups=[channel_client_context._layer_names],
        loads=[(second.lease, [[0, 1]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert restore_client.wait_restore(second_restore, timeout=5).success
    assert channel_client_context.get_kv_cache()[:, 0:2].cpu().equal(expected)

    third = query(
        instance_id=channel_client_context.instance_id,
        block_hashes=block_hashes,
        req_id="registered-release-query",
    )
    assert isinstance(third, orbitkv_native.QueryReady)
    query_client.release(third.lease)
    with pytest.raises(orbitkv_native.OrbitKVError, match="Invalid"):
        query_client.release(third.lease)


@pytest.mark.parametrize("channel_server", ["budget"], indirect=True)
def test_query_bytes_remain_owned_after_delivery_and_release_on_disconnect(
    channel_server, channel_client_context
):
    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady
    from tests.support.metrics import fetch_orbitkv_metrics

    instance = channel_client_context.instance_id
    client = CacheManagerClient(channel_server.bootstrap_socket)
    other = CacheManagerClient(channel_server.bootstrap_socket)
    hashes = [b"budget-a" * 4, b"budget-b" * 4]
    try:
        channel_client_context.client.save(
            channel_client_context.registration,
            [(channel_client_context._layer_names[0], [0, 1], hashes)],
        )
        deadline = time.monotonic() + 5
        while True:
            first = client.query_prefetch(instance, BlockHashes(hashes[:1]), "first")
            if isinstance(first, QueryReady) and first.num_hit_blocks == 1:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (
            fetch_orbitkv_metrics(channel_server.http_port)["orbitkv_query_reserved_bytes"] == 65536
        )
        waiting = other.query_prefetch(instance, BlockHashes(hashes[:1]), "waiting")
        assert isinstance(waiting, QueryLoading) and waiting.admitted
        oversized = other.query_prefetch(instance, BlockHashes(hashes), "too-large")
        assert isinstance(oversized, QueryReady) and oversized.num_hit_blocks == 0
        assert not oversized.lease
        assert (
            fetch_orbitkv_metrics(channel_server.http_port)["orbitkv_query_budget_bypasses_total"]
            == 1
        )
        # A successful reply still owns host pages. Losing the session must
        # release these bytes before another request can be admitted.
        client.close()
        deadline = time.monotonic() + 5
        while isinstance(waiting, QueryLoading):
            waiting = other.query_prefetch(instance, BlockHashes(hashes[:1]), "waiting")
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert waiting.num_hit_blocks == 1
        other.release(waiting.lease)
        assert fetch_orbitkv_metrics(channel_server.http_port)["orbitkv_query_reserved_bytes"] == 0
    finally:
        client.close()
        other.close()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "cuda-graph"])
@pytest.mark.parametrize("channel_server", [{"tier": "dram", "pool_size": "512mb"}], indirect=True)
def test_layer_events_publish_new_data_before_final_restore_drain(channel_server, graph):
    """The first consumer overlaps later H2D, including repeated graph replay."""
    import uuid

    import torch

    from orbitkv import BlockHashes, CacheManagerClient, OrbitKVError, QueryLoading, QueryReady
    from tests.support.cache_manager import ClientContext

    identity = f"layer-events-{uuid.uuid4().hex}"
    client = CacheManagerClient(channel_server.bootstrap_socket)
    ctx = ClientContext(
        client,
        identity,
        identity,
        num_blocks=2,
        num_layers=4,
        block_size=4096,
        num_heads=16,
        head_size=128,
    )
    ctx.register_kv_caches()
    events = [torch.cuda.Event(external=True, enable_timing=True) for _ in ctx._layer_names]
    for event in events:
        event.record()
    consumer = torch.cuda.Stream()
    snapshots = [torch.empty_like(tensor[:, 1, :1]) for tensor in ctx.gpu_kv_caches]
    early = torch.cuda.Event(enable_timing=True, external=True)

    def consume():
        for index, (tensor, event, snapshot) in enumerate(
            zip(ctx.gpu_kv_caches, events, snapshots, strict=True)
        ):
            consumer.wait_event(event)
            snapshot.copy_(tensor[:, 1, :1])
            if index == 0:
                early.record(consumer)

    replay = None
    if graph:
        consumer.wait_stream(torch.cuda.current_stream())
        replay = torch.cuda.CUDAGraph()
        with torch.cuda.graph(replay, stream=consumer):
            consume()
    try:
        for generation in range(2):
            for layer, tensor in enumerate(ctx.gpu_kv_caches):
                tensor[:, 0].fill_(generation * 10 + layer + 1)
                tensor[:, 1].fill_(-1)
            torch.cuda.synchronize()
            key = bytes([generation + 1]) * 32
            ok, message = client.save(
                ctx.registration, [(name, [0], [key]) for name in ctx._layer_names]
            )
            assert ok, message
            deadline = time.monotonic() + 10
            while True:
                query = client.query_prefetch(identity, BlockHashes([key]), f"load-{generation}")
                if not isinstance(query, QueryLoading):
                    break
                assert time.monotonic() < deadline
                time.sleep(0.001)
            assert isinstance(query, QueryReady) and query.num_hit_blocks == 1
            kwargs = {
                "instance_id": identity,
                "tp_rank": 0,
                "device_id": 0,
                "layer_groups": [ctx._layer_names],
                "loads": [(query.lease, [[1]])],
                "ready_stream": consumer.cuda_stream,
            }
            # Rejected bindings must leave the lease reusable for the valid call.
            for bindings in (
                [("missing", events[0])],
                [(name, events[0]) for name in ctx._layer_names],
            ):
                with pytest.raises(OrbitKVError, match="unique live events"):
                    client.start_restore(**kwargs, layer_events=bindings)
            handle = client.start_restore(
                **kwargs, layer_events=list(zip(ctx._layer_names, events, strict=True))
            )
            client.wait_restore_enqueued(handle, timeout=10)
            with torch.cuda.stream(consumer):
                if replay is None:
                    consume()
                else:
                    replay.replay()
            consumer.synchronize()
            # CUDA timestamps prove overlap, rather than inferring it from API order.
            overlap_ms = early.elapsed_time(events[-1])
            assert overlap_ms > 0, (
                f"first layer consumer did not overlap final H2D: {overlap_ms} ms"
            )
            assert client.wait_restore(handle, timeout=10).success
            for layer, (tensor, snapshot) in enumerate(
                zip(ctx.gpu_kv_caches, snapshots, strict=True)
            ):
                assert torch.equal(tensor[:, 0], tensor[:, 1])
                assert (snapshot == generation * 10 + layer + 1).all()
    finally:
        ctx.unregister_context()
        client.close()
