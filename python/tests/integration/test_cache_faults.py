"""Deterministic faults against the real Manager, CUDA mappings and SSD I/O.

Build orbitkv-server with test-hooks and set ORBITKV_FAULT_TESTS=1 explicitly.
The barriers are absent from normal release binaries.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from tests.support.cache_manager import (
    CacheManagerProcess,
    ClientContext,
    find_available_port,
)
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("ORBITKV_FAULT_TESTS") != "1", reason="requires test-hooks Manager"
    ),
]


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not (result := predicate()):
        assert time.monotonic() < deadline, "fault gate timed out"
        time.sleep(0.01)
    return result


@pytest.fixture
def fault_cache(tmp_path, monkeypatch, request):
    import torch

    from orbitkv import CacheManagerClient

    configuration = request.param if isinstance(getattr(request, "param", None), dict) else {}
    backend = configuration.get("ssd_backend", "uring")
    if backend == "cufile" and request.config.getoption("--ssd-backend") != "cufile":
        pytest.skip("pass --ssd-backend cufile for GPU storage qualification")
    monkeypatch.setenv("ORBITKV_TEST_FAULTS", str(tmp_path))
    server = CacheManagerProcess(
        find_available_port(),
        http_port=find_available_port(),
        ssd_cache_path=tmp_path / "ssd",
        ssd_backend=backend,
        extra_args=request.param
        if isinstance(getattr(request, "param", None), tuple)
        else configuration.get("extra_args", ()),
        channel_service=f"orbitkv/fault/{tmp_path.name}"
        if getattr(request, "param", None)
        else None,
    )
    assert server.start(), server.read_logs()
    client = CacheManagerClient(server.bootstrap_socket, timeout_ms=100)
    ctx = ClientContext(
        client=client,
        instance_id="fault-instance",
        namespace="fault-model",
        device_id=0,
        num_blocks=configuration.get("num_blocks", 4),
        block_size=configuration.get("block_size", 16),
        num_layers=configuration.get("num_layers", 1),
        dtype=getattr(torch, configuration.get("dtype", "bfloat16")),
    )
    ctx.register_kv_caches()
    try:
        yield server, client, ctx, tmp_path
    finally:
        for pause in tmp_path.glob("*.pause"):
            pause.unlink(missing_ok=True)
        client.close()
        server.stop()


def arm(directory: Path, name: str):
    (directory / f"{name}.reached").unlink(missing_ok=True)
    (directory / f"{name}.pause").touch()


def reached(directory, name):
    until(lambda: (directory / f"{name}.reached").exists())


def publish(client, ctx, hashes):
    return ctx.client.save(
        ctx.registration, [(ctx._layer_names[0], list(range(len(hashes))), hashes)]
    )


def query(client, ctx, hashes, rid):
    from orbitkv import BlockHashes, QueryReady

    batch = BlockHashes(hashes)

    def poll():
        result = client.query_prefetch(ctx.instance_id, batch, rid)
        return result if isinstance(result, QueryReady) else None

    return until(poll)


def drain_ssd(server):
    def drained():
        stats = fetch_orbitkv_metrics(server.http_port)
        return stats.get("orbitkv_ssd_write_bytes_total", 0) > 0 and not any(
            stats.get(name, 0)
            for name in (
                "orbitkv_ssd_write_inflight",
                "orbitkv_ssd_write_queue_pending",
                "orbitkv_inflight_bytes",
            )
        )

    until(drained)
    response = requests.post(f"http://127.0.0.1:{server.http_port}/cache/memory/cleanup", timeout=5)
    response.raise_for_status()


@pytest.mark.parametrize(
    "fault_cache",
    [
        {"ssd_backend": backend, "block_size": 64, "extra_args": ("--storage-codec", "fp8")}
        for backend in ("uring", "cufile")
    ],
    indirect=True,
)
def test_encoded_ssd_mixed_prefix_cancellation_and_corruption(fault_cache):
    import torch

    from orbitkv import BlockHashes, QueryLoading

    server, client, ctx, directory = fault_cache
    tensor = ctx.get_kv_cache()
    tensor[:, 0:1].fill_(1.25)
    tensor[:, 1:2].view(torch.uint8).random_(0, 256)
    expected = tensor[:, 0:2].view(torch.uint8).cpu().clone()
    torch.cuda.synchronize()
    hashes = [b"compressed", b"raw"]
    assert publish(client, ctx, hashes)[0]
    drain_ssd(server)
    stats = fetch_orbitkv_metrics(server.http_port)
    assert stats["orbitkv_storage_codec_bytes_total"] > 0
    assert stats["orbitkv_ssd_write_bytes_total"] < expected.numel()

    if server.ssd_backend == "cufile":
        # Direct discovery leases the extent without reading a host payload.
        ready = query(client, ctx, hashes, "canceled-codec")
        assert ready.num_hit_blocks == 2
        arm(directory, "cufile_read_completion")
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[2, 3]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        reached(directory, "cufile_read_completion")
        client.cancel_query(ctx.instance_id, "canceled-codec")
        assert not client.poll_restore(handle).done
        held = fetch_orbitkv_metrics(server.http_port)
        assert held["orbitkv_ssd_cufile_inflight_batches"] > 0
        assert held["orbitkv_ssd_read_pinned_bytes"] > 0
        (directory / "cufile_read_completion.pause").unlink()
        assert client.wait_restore(handle, timeout=10).success
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] == 0)
    else:
        arm(directory, "ssd")
        pending = client.query_prefetch(ctx.instance_id, BlockHashes(hashes), "canceled-codec")
        assert isinstance(pending, QueryLoading)
        reached(directory, "ssd")
        assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_prefetch_inflight"] > 0
        client.cancel_query(ctx.instance_id, "canceled-codec")
        # Submitted I/O still owns its pinned encoded pages after cancellation.
        assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_prefetch_inflight"] > 0
        (directory / "ssd.pause").unlink()
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_prefetch_inflight"] == 0)
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_storage_codec_reserved_bytes"] == 0
    ready = query(client, ctx, hashes, "mixed-codec")
    assert ready.num_hit_blocks == 2
    restore = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2, 3]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(restore, timeout=10).success
    assert torch.equal(tensor[:, 2:4].view(torch.uint8).cpu(), expected)
    drain_ssd(server)

    # Damage the ephemeral file after all I/O drained. No corrupted block may
    # enter the cache, including when FP8 can still decode the damaged stream.
    with Path(server.ssd_cache_path).open("r+b", buffering=0) as stream:
        stream.write(bytes(int(stats["orbitkv_ssd_write_bytes_total"])))
    missing = query(client, ctx, [hashes[0]], "corrupt-codec")
    if server.ssd_backend == "cufile":
        assert missing.num_hit_blocks == 1
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(missing.lease, [[2]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert not client.wait_restore(handle, timeout=10).success
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] == 0)
        missing = query(client, ctx, [hashes[0]], "invalidated-codec")
    assert missing.num_hit_blocks == 0
    stats = fetch_orbitkv_metrics(server.http_port)
    assert stats["orbitkv_storage_codec_decode_failures_total"] > 0
    assert stats["orbitkv_storage_codec_reserved_bytes"] == 0
    # Recomputing the same key must replace the damaged disk generation.
    assert publish(client, ctx, [hashes[0]])[0]
    drain_ssd(server)
    repaired = query(client, ctx, [hashes[0]], "repaired-codec")
    assert repaired.num_hit_blocks == 1
    tensor[:, 2:3].zero_()
    torch.cuda.synchronize()
    restore = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(repaired.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(restore, timeout=10).success
    assert torch.equal(tensor[:, 2:3].view(torch.uint8).cpu(), expected[:, 0:1])


@pytest.mark.parametrize(
    "fault_cache",
    [
        {
            "dtype": dtype,
            "block_size": block_size,
            "extra_args": ("--storage-codec", "fp8", "--storage-codec-budget", budget),
        }
        for dtype in ("bfloat16", "float16")
        for budget, block_size in (("64mb", 64), ("4kb", 64), ("12kb", 8))
    ],
    indirect=True,
)
def test_fp8_storage_matches_torch_scalar_cast_and_isolates_exact_registration(fault_cache):
    import torch

    server, client, ctx, _ = fault_cache
    tensor = ctx.get_kv_cache()
    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(tensor.dtype).float()
    values = values[torch.isfinite(values) & (values.abs() <= 448)].to(tensor.dtype).cuda()
    count = tensor[:, 0:1].numel()
    tensor[:, 0:1] = values.repeat((count + values.numel() - 1) // values.numel())[
        :count
    ].reshape_as(tensor[:, 0:1])
    original = tensor[:, 0:1].clone()
    expected = original.to(torch.float8_e4m3fn).to(original.dtype)
    assert torch.any(original != expected)
    torch.cuda.synchronize()
    hashes = [b"every-finite-scalar"]
    assert publish(client, ctx, hashes)[0]
    drain_ssd(server)
    stats = fetch_orbitkv_metrics(server.http_port)
    assert stats["orbitkv_ssd_write_bytes_total"] == original.numel()
    ready = query(client, ctx, hashes, "quantized")
    assert ready.num_hit_blocks == 1
    restore = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(restore, timeout=10).success
    assert torch.equal(tensor[:, 2:3].view(torch.uint8), expected.view(torch.uint8))

    # The same namespace/hash with exact storage policy must not adopt lossy data.
    name = "exact-instance"
    stride = tensor.stride()
    client.register_context_batch(
        name,
        ctx.namespace,
        0,
        0,
        1,
        1,
        0,
        ctx._layer_names,
        [ctx.num_blocks],
        [stride[1] * tensor.element_size()],
        [stride[0] * tensor.element_size()],
        [2],
        "direct",
        False,
        tensors=[tensor],
        layer_formats=["exact"],
    )
    from orbitkv import BlockHashes, QueryReady

    result = client.query_prefetch(name, BlockHashes(hashes), "exact")
    assert isinstance(result, QueryReady) and result.num_hit_blocks == 0
    client.unregister_context(name)


@pytest.mark.parametrize(
    "fault_cache",
    [
        pytest.param(
            {
                "num_layers": 6,
                "block_size": 64,
                "ssd_backend": backend,
                "extra_args": ("--storage-codec", codec),
            },
            id=f"{backend}-{codec}",
        )
        for backend in ("uring", "cufile")
        for codec in ("ans", "fp8", "turboquant-4", "turboquant-3")
    ],
    indirect=True,
)
def test_gpu_codecs_restore_after_registration_and_ssd_eviction(fault_cache, request):
    import torch

    from tests.support.metrics import fetch_orbitkv_codec_bytes

    server, client, ctx, _ = fault_cache
    codec = request.node.callspec.params["fault_cache"]["extra_args"][-1]
    backend = request.node.callspec.params["fault_cache"]["ssd_backend"]
    originals = [tensor[:, :1].clone() for tensor in ctx.gpu_kv_caches]
    hashes = [b"gpu-codec-roundtrip"]
    assert client.save(ctx.registration, [(name, [0], hashes) for name in ctx._layer_names])[0]
    drain_ssd(server)
    encoded = fetch_orbitkv_codec_bytes(server.http_port)
    assert encoded["logical"] > encoded["stored"] > 0
    client.unregister_context(ctx.instance_id)
    ctx._registered = False
    assert (
        fetch_orbitkv_metrics(server.http_port).get("orbitkv_storage_codec_workspace_bytes", 0) == 0
    )
    ctx.register_kv_caches()
    for tensor in ctx.gpu_kv_caches:
        tensor[:, 2:3].fill_(-123)
    torch.cuda.synchronize()
    ready = query(client, ctx, hashes, "restart-codec")
    assert ready.num_hit_blocks == 1
    before_restore = fetch_orbitkv_metrics(server.http_port)
    restore = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(restore, timeout=10).success
    after_restore = fetch_orbitkv_metrics(server.http_port)
    if backend == "cufile":
        assert after_restore["orbitkv_ssd_cufile_write_bytes_total"] > 0
        assert after_restore["orbitkv_ssd_cufile_read_bytes_total"] > 0
        assert after_restore.get("orbitkv_ssd_prefetch_bytes_total", 0) == 0
        assert after_restore.get("orbitkv_ssd_cufile_inflight_batches", 0) == 0
    assert after_restore["orbitkv_load_bytes_total"] - before_restore.get(
        "orbitkv_load_bytes_total", 0
    ) == sum(original.numel() * original.element_size() for original in originals)
    for index, (tensor, original) in enumerate(zip(ctx.gpu_kv_caches, originals, strict=True)):
        actual = tensor[:, 2:3]
        if codec == "fp8":
            reference = original.to(torch.float8_e4m3fn).to(original.dtype)
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
        elif codec == "ans" or index in (0, 1, 4, 5):
            assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
        else:
            relative_error = (actual.float() - original.float()).norm() / original.float().norm()
            assert 0 < relative_error < (0.3 if codec == "turboquant-3" else 0.18)
            norms = actual[0].float().norm(dim=-1) / original[0].float().norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=0.01, rtol=0)
    assert after_restore["orbitkv_storage_codec_reserved_bytes"] == 0
    if backend == "cufile":
        # A cached encoded prefix uses the SSD queue's host-decode arena while
        # its disk suffix uses the independent GDS decoder arena.
        assert after_restore.get("orbitkv_cache_resident_bytes", 0) == 0
        restored = [tensor[:, 2:3].clone() for tensor in ctx.gpu_kv_caches]
        hot_hash = b"gpu-codec-hot-prefix"
        assert client.save(
            ctx.registration, [(name, [0], [hot_hash]) for name in ctx._layer_names]
        )[0]
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_cache_resident_bytes"] > 0)
        for tensor in ctx.gpu_kv_caches:
            tensor[:, 2:4].fill_(-123)
        torch.cuda.synchronize()
        mixed = query(client, ctx, [hot_hash, hashes[0]], "mixed-codec-drain")
        assert mixed.num_hit_blocks == 2
        before_mixed = fetch_orbitkv_metrics(server.http_port)
        assert before_mixed["orbitkv_ssd_read_pinned_bytes"] > 0
        restore = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(mixed.lease, [[2, 3]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert client.wait_restore(restore, timeout=10).success
        after_mixed = fetch_orbitkv_metrics(server.http_port)
        assert (
            after_mixed["orbitkv_ssd_cufile_read_bytes_total"]
            > before_mixed["orbitkv_ssd_cufile_read_bytes_total"]
        )
        assert (
            after_mixed["orbitkv_storage_codec_workspace_bytes"]
            > before_mixed["orbitkv_storage_codec_workspace_bytes"]
        ), "mixed restore must retain the SSD queue's additional host-decode arena"
        assert after_mixed["orbitkv_storage_codec_reserved_bytes"] == 0
        for tensor, expected in zip(ctx.gpu_kv_caches, restored, strict=True):
            for block in (2, 3):
                assert torch.equal(
                    tensor[:, block : block + 1].view(torch.uint8), expected.view(torch.uint8)
                )
    client.unregister_context(ctx.instance_id)
    ctx._registered = False
    assert (
        fetch_orbitkv_metrics(server.http_port).get("orbitkv_storage_codec_workspace_bytes", 0) == 0
    )


@pytest.mark.parametrize(
    "fault_cache",
    [
        pytest.param(
            {
                "ssd_backend": "cufile",
                "num_layers": 6,
                "block_size": 64,
                "extra_args": ("--storage-codec", codec),
            },
            id=codec,
        )
        for codec in ("ans", "fp8", "turboquant-4")
    ],
    indirect=True,
)
def test_encoded_cufile_corruption_rejects_gpu_restore_and_allows_republication(fault_cache):
    import torch

    server, client, ctx, _ = fault_cache
    hashes = [b"repair-gpu-encoded"]

    def save():
        return client.save(ctx.registration, [(name, [0], hashes) for name in ctx._layer_names])[0]

    assert save()
    drain_ssd(server)
    stats = fetch_orbitkv_metrics(server.http_port)
    assert stats["orbitkv_ssd_cufile_write_bytes_total"] > 0
    with Path(server.ssd_cache_path).open("r+b", buffering=0) as stream:
        stream.write(bytes(int(stats["orbitkv_ssd_write_bytes_total"])))

    # Discovery reads metadata only. Device CRC validation must reject the
    # damaged payload before any decompressor consumes that segment.
    ready = query(client, ctx, hashes, "damaged-gpu-object")
    assert ready.num_hit_blocks == 1
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert not client.wait_restore(handle, timeout=10).success
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] == 0)
    assert (
        fetch_orbitkv_metrics(server.http_port)["orbitkv_storage_codec_decode_failures_total"] > 0
    )
    assert query(client, ctx, hashes, "invalidated-gpu-object").num_hit_blocks == 0

    assert save()
    drain_ssd(server)
    repaired = query(client, ctx, hashes, "repaired-gpu-object")
    assert repaired.num_hit_blocks == 1
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(repaired.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(handle, timeout=10).success
    after = fetch_orbitkv_metrics(server.http_port)
    assert after.get("orbitkv_ssd_prefetch_bytes_total", 0) == 0
    assert after.get("orbitkv_ssd_cufile_inflight_batches", 0) == 0


def test_ssd_cancellation_revisions_hold_buffers_until_io_drains(fault_cache):
    from orbitkv import BlockHashes, QueryLoading

    server, client, ctx, directory = fault_cache
    hashes = [bytes([i]) * 32 for i in (1, 2)]
    publish(client, ctx, hashes)
    drain_ssd(server)
    arm(directory, "ssd")
    assert isinstance(
        client.query_prefetch(ctx.instance_id, BlockHashes(hashes), "slow"), QueryLoading
    )
    reached(directory, "ssd")
    reserved = fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"]
    assert reserved > 0
    client.cancel_query(ctx.instance_id, "slow")
    # A late result cannot reappear under a reused request name/revision.
    assert query(client, ctx, [b"different"], "slow").num_hit_blocks == 0
    assert client.health()[0]
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == reserved
    # Start and cancel a second interest while the same submitted read is live.
    assert isinstance(
        client.query_prefetch(ctx.instance_id, BlockHashes(hashes), "second"), QueryLoading
    )
    client.cancel_query(ctx.instance_id, "second")
    (directory / "ssd.pause").unlink()
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )
    ready = query(client, ctx, hashes, "fresh")
    assert ready.num_hit_blocks == 2
    client.release(ready.lease)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )


@pytest.mark.parametrize(
    ("fault_cache", "stop"),
    [
        (("--query-read-batch", "1"), "cancel"),
        (("--query-read-batch", "1", "--query-read-timeout-ms", "100"), "timeout"),
        (("--query-read-batch", "1", "--query-read-max-batches", "1"), "best-effort"),
    ],
    indirect=["fault_cache"],
)
def test_stopping_submits_no_second_ssd_batch_and_drains_the_first(fault_cache, stop):
    from orbitkv import BlockHashes, QueryLoading

    server, client, ctx, directory = fault_cache
    hashes = [bytes([i]) * 32 for i in range(1, 5)]
    publish(client, ctx, hashes)
    drain_ssd(server)
    before = fetch_orbitkv_metrics(server.http_port)
    arm(directory, "ssd")
    batch = BlockHashes(hashes)
    assert isinstance(client.query_prefetch(ctx.instance_id, batch, "bounded"), QueryLoading)
    reached(directory, "ssd")
    if stop == "cancel":
        client.cancel_query(ctx.instance_id, "bounded")
    elif stop == "timeout":
        time.sleep(0.15)
        timed_out = query(client, ctx, hashes, "bounded")
        assert timed_out.num_hit_blocks == 0 and not timed_out.lease
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] > 0
    assert query(client, ctx, [b"independent"], "other").num_hit_blocks == 0
    observed_lookups = fetch_orbitkv_metrics(server.http_port).get("orbitkv_hll_total_requests", 0)
    (directory / "ssd.pause").unlink()
    if stop == "best-effort":
        result = query(client, ctx, hashes, "bounded")
        assert result.num_hit_blocks == 1
        client.release(result.lease)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )
    after = fetch_orbitkv_metrics(server.http_port)
    page_bytes = ctx.get_kv_cache().numel() * ctx.get_kv_cache().element_size() // ctx.num_blocks
    assert (
        after["orbitkv_ssd_prefetch_bytes_total"]
        - before.get("orbitkv_ssd_prefetch_bytes_total", 0)
        == page_bytes
    )
    assert after.get("orbitkv_ssd_prefetch_inflight", 0) == 0
    assert after.get("orbitkv_hll_total_requests", 0) == observed_lookups, (
        "unread pages are not authoritative cache misses"
    )


@pytest.mark.parametrize("fault_cache", [("--query-read-batch", "1")], indirect=True)
def test_cancelled_preparation_does_not_stop_another_shared_read_owner(fault_cache):
    from orbitkv import BlockHashes, QueryLoading, RecoveryContract

    server, client, ctx, directory = fault_cache
    hashes = [bytes([i]) * 32 for i in range(1, 5)]
    publish(client, ctx, hashes)
    drain_ssd(server)
    arm(directory, "ssd")
    batch = BlockHashes(hashes)
    contract = RecoveryContract(ctx.namespace, 16, [(0, "attention", 0)])
    assert client.prepare_recovery(
        ctx.instance_id, batch, "forecast", contract, ctx.namespace, 0, 64, 0
    )
    reached(directory, "ssd")
    assert isinstance(client.query_prefetch(ctx.instance_id, batch, "demand"), QueryLoading)
    client.cancel_query(ctx.instance_id, "forecast")
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] > 0
    (directory / "ssd.pause").unlink()
    result = query(client, ctx, hashes, "demand")
    assert result.num_hit_blocks == 4
    client.release(result.lease)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_coalesced_reads_total"] > 0


def test_prepared_result_expires_without_poll_then_claim_holds_bytes_through_restore(fault_cache):
    import torch

    from orbitkv import BlockHashes, QueryReady, RecoveryContract

    server, client, ctx, _ = fault_cache
    hashes = [b"prepared" * 4]
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    publish(client, ctx, hashes)
    contract = RecoveryContract(ctx.namespace, 16, [(0, "attention", 0)])
    args = (ctx.instance_id, BlockHashes(hashes), "prepared", contract, ctx.namespace, 0, 16, 0)
    assert client.prepare_recovery(*args)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) > 0
    )
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )
    assert client.prepare_recovery(*args), "the expired ticket cannot prevent new preparation"

    def claim():
        result = client.read_recovery(*args)
        return result if isinstance(result, QueryReady) else None

    ready = until(claim)
    assert ready.num_hit_blocks == 1 and ready.lease
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] > 0
    ctx.get_kv_cache()[:, 1:2].zero_()
    torch.cuda.synchronize()
    restore = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[1]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(restore, timeout=10).success
    assert torch.equal(ctx.get_kv_cache()[:, 1:2].cpu(), expected)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )


@pytest.mark.parametrize(
    "fault_cache,encoded",
    [
        pytest.param({"ssd_backend": "cufile"}, False, id="raw"),
        *[
            pytest.param(
                {
                    "ssd_backend": "cufile",
                    "block_size": 64,
                    "extra_args": ("--storage-codec", codec),
                },
                True,
                id=codec,
            )
            for codec in ("ans", "fp8")
        ],
    ],
    indirect=["fault_cache"],
)
@pytest.mark.parametrize("outcome", ["complete", "error", "kill"])
def test_cufile_write_holds_pages_and_publishes_only_completed_objects(
    fault_cache, encoded, outcome
):
    import torch

    server, client, ctx, directory = fault_cache
    if encoded:
        ctx.get_kv_cache().fill_(1.0)
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    assert publish(client, ctx, [b"warm"])[0]
    if encoded:
        assert fetch_orbitkv_metrics(server.http_port)["orbitkv_storage_codec_bytes_total"] > 0
    warm = query(client, ctx, [b"warm"], "warm")
    arm(directory, "cufile_write")
    pool = ThreadPoolExecutor(1)
    future = pool.submit(publish, client, ctx, [b"writing"])
    try:
        reached(directory, "cufile_write")
        time.sleep(0.15)
        assert not future.done(), "GPU sources must remain owned throughout cuFileWrite"
        assert query(client, ctx, [b"writing"], "uncommitted").num_hit_blocks == 0
        assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_write_inflight"] > 0
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(warm.lease, [[3]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert client.wait_restore(handle, timeout=5).success
        assert torch.equal(ctx.get_kv_cache()[:, 3:4].cpu(), expected)
        if outcome == "kill":
            server.stop()
            with pytest.raises(Exception, match="exited|reconnect"):
                future.result(timeout=10)
            return
        if outcome == "error":
            arm(directory, "cufile_write_error")
        (directory / "cufile_write.pause").unlink()
        if outcome == "error":
            with pytest.raises(Exception, match="Internal"):
                future.result(timeout=10)
            assert query(client, ctx, [b"writing"], "failed").num_hit_blocks == 0
            assert (
                fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_cufile_write_failures_total"]
                > 0
            )
            (directory / "cufile_write_error.pause").unlink()
            failed = fetch_orbitkv_metrics(server.http_port)
            assert failed["orbitkv_ssd_write_inflight"] == 0
            assert failed["orbitkv_ssd_cufile_inflight_batches"] == 0
            assert failed.get("orbitkv_storage_codec_reserved_bytes", 0) == 0
            assert publish(client, ctx, [b"writing"])[0], "an aborted extent must be retryable"
        else:
            assert future.result(timeout=10)[0]
        drain_ssd(server)
        ready = query(client, ctx, [b"writing"], "committed")
        assert ready.num_hit_blocks == 1
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[2]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert client.wait_restore(handle, timeout=10).success
        assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
        stats = fetch_orbitkv_metrics(server.http_port)
        assert stats["orbitkv_ssd_cufile_write_bytes_total"] > 0
        assert stats["orbitkv_ssd_cufile_read_bytes_total"] > 0
        assert stats["orbitkv_ssd_write_inflight"] == 0
        assert client.unregister_context(ctx.instance_id)[0]
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_gpu_staging_bytes"] == 0)
    finally:
        (directory / "cufile_write.pause").unlink(missing_ok=True)
        (directory / "cufile_write_error.pause").unlink(missing_ok=True)
        if not future.done():
            server.stop()
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "fault_cache,completion",
    [
        ({"ssd_backend": "cufile"}, "cufile_write_completion"),
        ({"ssd_backend": "cufile"}, "ssd_host_completion"),
        (
            {"ssd_backend": "cufile", "block_size": 64, "extra_args": ("--storage-codec", "ans")},
            "cufile_write_completion",
        ),
    ],
    indirect=["fault_cache"],
    ids=["raw-disk-completion", "raw-host-completion", "encoded-disk-completion"],
)
def test_submitted_cufile_write_allows_ssd_reads_and_delays_unregister(fault_cache, completion):
    import torch

    server, client, ctx, directory = fault_cache
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    assert publish(client, ctx, [b"on-disk"])[0]
    drain_ssd(server)
    ready = query(client, ctx, [b"on-disk"], "read-during-write")
    arm(directory, completion)
    pool = ThreadPoolExecutor(2)
    writing = pool.submit(publish, client, ctx, [b"unconfirmed"])
    try:
        reached(directory, completion)
        before = fetch_orbitkv_metrics(server.http_port)
        if completion == "cufile_write_completion":
            assert before["orbitkv_ssd_cufile_inflight_batches"] == 1
        assert not writing.done()
        if completion == "cufile_write_completion":
            assert query(client, ctx, [b"unconfirmed"], "unpublished").num_hit_blocks == 0
        restore = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[2]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert client.wait_restore(restore, timeout=5).success
        assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
        after = fetch_orbitkv_metrics(server.http_port)
        assert after["orbitkv_ssd_cufile_read_bytes_total"] > before.get(
            "orbitkv_ssd_cufile_read_bytes_total", 0
        )
        if completion == "cufile_write_completion":
            assert after["orbitkv_ssd_cufile_inflight_batches"] == 1
        unregistering = pool.submit(client.unregister_context, ctx.instance_id)
        time.sleep(0.15)
        assert not unregistering.done(), "unregister must retain submitted write ownership"
        if completion == "cufile_write_completion":
            assert after["orbitkv_ssd_write_inflight"] > 0
        (directory / f"{completion}.pause").unlink()
        assert writing.result(timeout=10)[0]
        assert unregistering.result(timeout=10)[0]
        until(
            lambda: all(
                fetch_orbitkv_metrics(server.http_port).get(name, 0) == 0
                for name in (
                    "orbitkv_ssd_cufile_inflight_batches",
                    "orbitkv_ssd_write_inflight",
                    "orbitkv_ssd_read_pinned_bytes",
                    "orbitkv_ssd_gpu_staging_bytes",
                )
            )
        )
    finally:
        (directory / f"{completion}.pause").unlink(missing_ok=True)
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "fault_cache,encoded",
    [
        pytest.param({"ssd_backend": "cufile", "num_blocks": 10}, False, id="raw"),
        *[
            pytest.param(
                {
                    "ssd_backend": "cufile",
                    "num_blocks": 10,
                    "block_size": 64,
                    "extra_args": ("--storage-codec", codec),
                },
                True,
                id=codec,
            )
            for codec in ("ans", "fp8")
        ],
    ],
    indirect=["fault_cache"],
)
def test_full_gpu_write_queue_uses_host_writeback_without_waiting(fault_cache, encoded):
    import torch

    from orbitkv import CacheManagerClient

    server, client, ctx, directory = fault_cache
    tensor = ctx.get_kv_cache()
    if encoded:
        for index in range(9):
            tensor[:, index : index + 1].fill_(index + 1)
    expected = tensor[:, :9].cpu().clone()
    arm(directory, "cufile_write_completion")

    # Each Publish descriptor session has one outstanding call. Independent
    # producers are needed to exercise the shared worker's admission limit.
    producers = []
    producer_contexts = []
    for index in range(9):
        producer = CacheManagerClient(server.bootstrap_socket, timeout_ms=100)
        producers.append(producer)
        context = ClientContext(
            producer,
            f"write-producer-{index}",
            ctx.namespace,
            device_id=ctx.device_id,
            num_blocks=ctx.num_blocks,
            num_layers=ctx.num_layers,
            block_size=ctx.block_size,
            num_heads=ctx.num_heads,
            head_size=ctx.head_size,
            dtype=ctx.dtype,
        )
        context.gpu_kv_caches = ctx.gpu_kv_caches
        context.register_kv_caches()
        producer_contexts.append(context)

    def save_page(producer_context, index):
        return producer_context.client.save(
            producer_context.registration, [(ctx._layer_names[0], [index], [bytes([index]) * 32])]
        )

    pool = ThreadPoolExecutor(9)
    writes = [pool.submit(save_page, context, i) for i, context in enumerate(producer_contexts)]
    try:
        reached(directory, "cufile_write_completion")
        # Encoded extents are reserved only when their worker starts. The one
        # completed publication proves all eight admission permits are occupied.
        completed = until(lambda: [index for index, write in enumerate(writes) if write.done()])
        assert len(completed) == 1, "only the publication beyond the eight permits may finish"
        fallback = completed[0]
        assert writes[fallback].result(timeout=10)[0]
        until(
            lambda: (
                fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_write_inflight"]
                == (1 if encoded else 8)
            )
        )
        assert all(not write.done() for index, write in enumerate(writes) if index != fallback)
        stats = fetch_orbitkv_metrics(server.http_port)
        assert stats["orbitkv_ssd_gpu_write_fallbacks_total"] == 1
        assert stats["orbitkv_ssd_cufile_inflight_batches"] == 1
        assert stats["orbitkv_ssd_gpu_staging_bytes"] == (4 if encoded else 8) << 20
        if encoded:
            assert stats["orbitkv_storage_codec_bytes_total"] > 0
        (directory / "cufile_write_completion.pause").unlink()
        assert all(write.result(timeout=10)[0] for write in writes)
        drain_ssd(server)
        ready = query(client, ctx, [bytes([fallback]) * 32], "fallback-on-disk")
        assert ready.num_hit_blocks == 1
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[9]])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        assert client.wait_restore(handle, timeout=10).success
        assert torch.equal(tensor[:, 9:10].cpu(), expected[:, fallback : fallback + 1])
        assert client.unregister_context(ctx.instance_id)[0]
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_gpu_staging_bytes"] == 0)
    finally:
        (directory / "cufile_write_completion.pause").unlink(missing_ok=True)
        pool.shutdown(wait=True)
        for producer in producers:
            producer.close()


@pytest.mark.parametrize(
    "fault_cache", [{"ssd_backend": "cufile", "block_size": 1024}], indirect=True
)
def test_submitted_cufile_reads_keep_two_slots_and_leases_until_unregister(fault_cache):
    import torch

    server, client, ctx, directory = fault_cache
    tensor = ctx.get_kv_cache()
    expected = tensor.cpu().clone()
    hashes = [bytes([i]) * 32 for i in range(ctx.num_blocks)]
    assert publish(client, ctx, hashes)[0]
    drain_ssd(server)
    ready = query(client, ctx, hashes, "large-read")
    tensor.zero_()
    torch.cuda.synchronize()
    arm(directory, "cufile_read_completion")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [list(range(ctx.num_blocks))])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    pool = ThreadPoolExecutor(1)
    try:
        reached(directory, "cufile_read_completion")
        until(
            lambda: (
                fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_cufile_inflight_batches"] == 2
            )
        )
        client.cancel_query(ctx.instance_id, "large-read")
        unregistering = pool.submit(client.unregister_context, ctx.instance_id)
        time.sleep(0.15)
        assert not unregistering.done()
        assert not client.poll_restore(handle).done
        held = fetch_orbitkv_metrics(server.http_port)
        assert held["orbitkv_ssd_gpu_staging_bytes"] == 8 << 20
        assert held["orbitkv_ssd_read_pinned_bytes"] == tensor.numel() * tensor.element_size()
        (directory / "cufile_read_completion.pause").unlink()
        assert client.wait_restore(handle, timeout=10).success
        assert unregistering.result(timeout=10)[0]
        assert torch.equal(tensor.cpu(), expected)
        after = fetch_orbitkv_metrics(server.http_port)
        assert after["orbitkv_ssd_cufile_read_seconds_count"] == 4
        assert after["orbitkv_ssd_cufile_inflight_batches"] == 0
        assert after["orbitkv_ssd_read_pinned_bytes"] == 0
        assert after["orbitkv_ssd_gpu_staging_bytes"] == 0
    finally:
        (directory / "cufile_read_completion.pause").unlink(missing_ok=True)
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "fault_cache,encoded",
    [
        pytest.param(
            {
                "ssd_backend": "cufile",
                "block_size": 64,
                "extra_args": ("--ssd-read-path", "uring"),
            },
            False,
            id="raw",
        ),
        pytest.param(
            {
                "ssd_backend": "cufile",
                "block_size": 64,
                "extra_args": ("--ssd-read-path", "uring", "--storage-codec", "ans"),
            },
            True,
            id="ans",
        ),
    ],
    indirect=["fault_cache"],
)
def test_submitted_host_reads_keep_leases_and_destinations_until_unregister(fault_cache, encoded):
    import torch

    server, client, ctx, directory = fault_cache
    tensor = ctx.get_kv_cache()
    if encoded:
        for block in range(ctx.num_blocks):
            tensor[:, block : block + 1].fill_(block + 0.25)
    expected = tensor.view(torch.uint8).cpu().clone()
    torch.cuda.synchronize()
    hashes = [bytes([i]) * 32 for i in range(ctx.num_blocks)]
    assert publish(client, ctx, hashes)[0]
    drain_ssd(server)
    before = fetch_orbitkv_metrics(server.http_port)
    assert before["orbitkv_ssd_cufile_write_bytes_total"] > 0
    if encoded:
        assert before["orbitkv_storage_codec_bytes_total"] > 0
        assert before["orbitkv_ssd_write_bytes_total"] < expected.numel()

    ready = query(client, ctx, hashes, "host-read")
    assert ready.num_hit_blocks == ctx.num_blocks and ready.lease
    leased = fetch_orbitkv_metrics(server.http_port)
    assert leased["orbitkv_ssd_read_pinned_bytes"] > 0
    for counter in ("orbitkv_ssd_prefetch_bytes_total", "orbitkv_ssd_cufile_read_bytes_total"):
        assert leased.get(counter, 0) == before.get(counter, 0), "discovery must only pin metadata"
    tensor.zero_()
    torch.cuda.synchronize()
    arm(directory, "ssd")
    pool = ThreadPoolExecutor(1)
    unregistering = None
    try:
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [list(range(ctx.num_blocks))])],
            ready_stream=torch.cuda.current_stream(0).cuda_stream,
        )
        reached(directory, "ssd")
        client.cancel_query(ctx.instance_id, "host-read")
        unregistering = pool.submit(client.unregister_context, ctx.instance_id)
        time.sleep(0.15)
        assert not unregistering.done(), "unregister released a submitted host-staged restore"
        assert not client.poll_restore(handle).done
        held = fetch_orbitkv_metrics(server.http_port)
        assert held["orbitkv_ssd_prefetch_inflight"] > 0
        assert held["orbitkv_ssd_read_pinned_bytes"] > 0
        assert held["orbitkv_query_reserved_bytes"] > 0
        assert held.get("orbitkv_ssd_cufile_read_bytes_total", 0) == before.get(
            "orbitkv_ssd_cufile_read_bytes_total", 0
        )
        (directory / "ssd.pause").unlink()
        assert client.wait_restore(handle, timeout=10).success
        assert unregistering.result(timeout=10)[0]
        assert torch.equal(tensor.view(torch.uint8).cpu(), expected)

        def released():
            stats = fetch_orbitkv_metrics(server.http_port)
            holds = (
                "orbitkv_query_reserved_bytes",
                "orbitkv_ssd_read_pinned_bytes",
                "orbitkv_ssd_prefetch_inflight",
                "orbitkv_ssd_prefetch_queue_pending",
                "orbitkv_ssd_cufile_inflight_batches",
                "orbitkv_ssd_gpu_staging_bytes",
                "orbitkv_storage_codec_reserved_bytes",
                "orbitkv_inflight_bytes",
            )
            return stats if all(stats.get(name, 0) == 0 for name in holds) else None

        after = until(released)
        assert after["orbitkv_ssd_prefetch_bytes_total"] > before.get(
            "orbitkv_ssd_prefetch_bytes_total", 0
        )
        assert after.get("orbitkv_ssd_cufile_read_bytes_total", 0) == before.get(
            "orbitkv_ssd_cufile_read_bytes_total", 0
        )
        assert after["orbitkv_load_bytes_total"] > before.get("orbitkv_load_bytes_total", 0)
        assert after.get("orbitkv_load_failures_total", 0) == 0
        assert after.get("orbitkv_storage_codec_decode_failures_total", 0) == 0
        assert after.get("orbitkv_ssd_backend_fallbacks_total", 0) == 0
    finally:
        (directory / "ssd.pause").unlink(missing_ok=True)
        if unregistering is not None and not unregistering.done():
            server.stop()
        pool.shutdown(wait=True)


@pytest.mark.parametrize(
    "fault_cache,selected,read_calls",
    [
        ({"ssd_backend": "cufile"}, (0, 1, 2, 3), 1),
        (
            {"ssd_backend": "cufile", "extra_args": ("--ssd-cache-shards", "2")},
            (0, 1, 2, 3),
            2,
        ),
        ({"ssd_backend": "cufile"}, (0, 2), 2),
    ],
    indirect=["fault_cache"],
    ids=["adjacent-leases", "separate-files", "unrequested-gap"],
)
def test_cufile_batches_keep_all_leases_and_only_read_selected_pages(
    fault_cache, selected, read_calls
):
    import torch

    server, client, ctx, directory = fault_cache
    cache_path = directory / "ssd"
    files = [cache_path] if cache_path.is_file() else list(cache_path.glob("shard-*.dat"))
    assert files
    for file in files:
        stat = file.stat()
        assert stat.st_size > 0 and stat.st_blocks * 512 >= stat.st_size
    tensor = ctx.get_kv_cache()
    expected = tensor.cpu().clone()
    hashes = [bytes([i]) * 32 for i in range(ctx.num_blocks)]
    # Establish physical file order explicitly; a multi-page Publish may group
    # hashes in any order, so logical pages 0/2 need not have a disk gap.
    for page, block_hash in enumerate(hashes):
        assert client.save(ctx.registration, [(ctx._layer_names[0], [page], [block_hash])])[0]
    drain_ssd(server)
    ready = query(client, ctx, hashes, "batched")
    assert ready.num_hit_blocks == len(hashes)
    before = fetch_orbitkv_metrics(server.http_port)
    assert before["orbitkv_ssd_read_pinned_bytes"] == tensor.numel() * tensor.element_size()
    assert before["orbitkv_ssd_cufile_write_seconds_sum"] > 0
    page_bytes = expected[:, 0].numel() * expected.element_size()
    tensor.zero_()
    torch.cuda.synchronize()

    arm(directory, "cufile")
    destinations = [i if i in selected else None for i in range(ctx.num_blocks)]
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [destinations])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "cufile")
    client.cancel_query(ctx.instance_id, "batched")
    assert not client.poll_restore(handle).done
    # Cancellation releases unselected query sources; every submitted source
    # must remain pinned, including all extents merged into a single read.
    assert fetch_orbitkv_metrics(server.http_port)[
        "orbitkv_ssd_read_pinned_bytes"
    ] == page_bytes * len(selected)
    (directory / "cufile.pause").unlink()
    assert client.wait_restore(handle, timeout=10).success
    actual = tensor.cpu()
    for page in range(ctx.num_blocks):
        if page in selected:
            assert torch.equal(actual[:, page], expected[:, page])
        else:
            assert torch.count_nonzero(actual[:, page]) == 0
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] == 0)
    after = fetch_orbitkv_metrics(server.http_port)
    count = "orbitkv_ssd_cufile_read_seconds_count"
    assert after[count] - before.get(count, 0) == read_calls
    assert after["orbitkv_ssd_cufile_read_seconds_sum"] > 0
    size = "orbitkv_ssd_cufile_read_bytes_total"
    assert after[size] - before.get(size, 0) == page_bytes * len(selected)


@pytest.mark.parametrize("fault_cache", [{"ssd_backend": "cufile"}], indirect=True)
def test_cufile_delay_keeps_sources_and_allows_dram_restores(fault_cache):
    import torch

    server, client, ctx, directory = fault_cache
    hashes = [b"disk-restore"]
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    publish(client, ctx, hashes)
    drain_ssd(server)
    warm_hashes = [b"dram-restore"]
    publish(client, ctx, warm_hashes)
    ready = query(client, ctx, hashes, "disk")
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] > 0
    arm(directory, "cufile")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "cufile")
    client.cancel_query(ctx.instance_id, "disk")
    with pytest.raises(TimeoutError):
        client.wait_restore(handle, timeout=0.02)
    assert not client.poll_restore(handle).done
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_read_pinned_bytes"] > 0

    # A real DRAM restore, not just a miss, must complete while storage is paused.
    warm = query(client, ctx, warm_hashes, "dram")
    other = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(warm.lease, [[3]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    assert client.wait_restore(other, timeout=5).success
    assert torch.equal(ctx.get_kv_cache()[:, 3:4].cpu(), expected)
    arm(directory, "notification")
    (directory / "cufile.pause").unlink()
    assert client.wait_restore(handle, timeout=10).success
    reached(directory, "notification")
    assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
    until(
        lambda: all(
            fetch_orbitkv_metrics(server.http_port).get(name, 0) == 0
            for name in ("orbitkv_query_reserved_bytes", "orbitkv_ssd_read_pinned_bytes")
        )
    )
    assert fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_gpu_staging_bytes"] == 8 << 20
    ok, message = client.unregister_context(ctx.instance_id)
    assert ok, message
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_ssd_gpu_staging_bytes"] == 0)


@pytest.mark.parametrize(
    "reject", [False, True], ids=["cancelled-preparation", "preparation-error"]
)
def test_restore_corrupt_ack_keeps_claimed_handle_and_shared_result(fault_cache, reject):
    import torch

    from orbitkv import OrbitKVError

    server, client, ctx, directory = fault_cache
    hashes = [b"restore-ack"]
    unchanged = ctx.get_kv_cache()[:, 2:3].cpu().clone()
    assert publish(client, ctx, hashes)[0]
    ready = query(client, ctx, hashes, "claimed-before-ack")
    before = fetch_orbitkv_metrics(server.http_port)
    arm(directory, "restore_ack")
    if not reject:
        arm(directory, "restore")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [["missing-layer"]] if reject else [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "restore_ack")
    # The malformed ACK closes descriptor admission. It cannot erase a claimed
    # operation or turn its socket disconnect into a completed GPU transfer.
    with pytest.raises(OrbitKVError, match="reconnect"):
        client.health()
    if not reject:
        reached(directory, "restore")
        with pytest.raises(TimeoutError):
            client.wait_restore(handle, timeout=0.02)
        assert not client.poll_restore(handle).done
        (directory / "restore.pause").unlink()
    result = client.wait_restore(handle, timeout=5)
    assert result.done and not result.success
    if reject:
        assert "missing-layer" in result.message
    else:
        # The Manager admitted preparation, but the engine had not claimed the
        # grant. Lost ACK cancellation may therefore finish with no GPU writes.
        assert "cancelled" in result.message
        assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), unchanged)
    after = fetch_orbitkv_metrics(server.http_port)
    assert after.get("orbitkv_load_bytes_total", 0) == before.get("orbitkv_load_bytes_total", 0)
    with pytest.raises(OrbitKVError, match="consumed"):
        client.poll_restore(handle)


def test_restore_timeout_and_lost_notification_preserve_destinations(fault_cache):
    import torch

    server, client, ctx, directory = fault_cache
    hashes = [b"restore" * 4]
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    publish(client, ctx, hashes)
    ready = query(client, ctx, hashes, "load")
    arm(directory, "restore")
    arm(directory, "notification")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "restore")
    with pytest.raises(TimeoutError):
        client.wait_restore(handle, timeout=0.02)
    assert not client.poll_restore(handle).done
    assert query(client, ctx, [b"unrelated"], "independent").num_hit_blocks == 0
    (directory / "restore.pause").unlink()
    assert client.wait_restore(handle, timeout=5).success
    reached(directory, "notification")
    torch.cuda.synchronize()
    assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
    until(
        lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_query_reserved_bytes", 0) == 0
    )


@pytest.mark.parametrize("end", ["complete", "kill", "corrupt-ack"])
def test_publish_stall_keeps_source_owned_without_stalling_queries(fault_cache, end, capfd):
    server, client, ctx, directory = fault_cache
    name = "publish_ack" if end == "corrupt-ack" else "publish"
    arm(directory, name)
    # Always terminate/release the peer before joining a potentially fenced publisher.
    pool = ThreadPoolExecutor(1)
    future = pool.submit(publish, client, ctx, [b"publish"])
    try:
        reached(directory, name)
        time.sleep(0.15)  # Past the client's 100ms ordinary call deadline.
        assert not future.done(), "publisher released GPU sources on an ambiguous timeout/ack"
        assert query(client, ctx, [b"unrelated"], "independent").num_hit_blocks == 0
        expected_log = "holding source pages" if end == "corrupt-ack" else "retaining source pages"
        assert expected_log in capfd.readouterr().err
        if end == "complete":
            (directory / "publish.pause").unlink()
            assert future.result(timeout=5)[0]
        else:
            server.stop()
            with pytest.raises(Exception, match="exited|reconnect"):
                future.result(timeout=5)
    finally:
        (directory / f"{name}.pause").unlink(missing_ok=True)
        if not future.done():
            server.stop()
        pool.shutdown(wait=True)


@pytest.mark.parametrize("fault_cache", [None, "configured-prefix"], indirect=True)
def test_manager_restart_rejects_old_leases_and_restore_handles(fault_cache):
    import torch

    from orbitkv import CacheManagerClient, OrbitKVError

    server, client, ctx, directory = fault_cache
    publish(client, ctx, [b"restart"])
    ready = query(client, ctx, [b"restart"], "old-lease")
    another = query(client, ctx, [b"restart"], "old-restore")
    arm(directory, "restore")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(another.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "restore")
    server.stop()
    (directory / "restore.pause").unlink()
    assert server.start(), server.read_logs()
    fresh = CacheManagerClient(server.bootstrap_socket)
    try:
        failed = client.wait_restore(handle, timeout=5)
        assert failed.done and not failed.success
        assert "reconnect" in failed.message
        with pytest.raises(ValueError, match="another client"):
            fresh.poll_restore(handle)
        with pytest.raises(OrbitKVError):
            fresh.release(ready.lease)
        replacement = ClientContext(
            client=fresh,
            instance_id=ctx.instance_id,
            namespace=ctx.namespace,
            device_id=0,
            num_blocks=4,
            num_layers=1,
        )
        replacement.register_kv_caches()
        publish(fresh, replacement, [b"new-engine"])
        new = query(fresh, replacement, [b"new-engine"], "fresh")
        assert new.num_hit_blocks == 1
        fresh.release(new.lease)
        replacement.unregister_context()
    finally:
        fresh.close()


@pytest.mark.parametrize("barrier", ["local_restore_claim", "local_restore_dma"])
def test_local_restore_survives_manager_death_after_claim(fault_cache, barrier):
    import torch

    server, client, ctx, directory = fault_cache
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    assert publish(client, ctx, [b"manager-crash-local"])[0]
    ready = query(client, ctx, [b"manager-crash-local"], "local-after-manager-death")
    arm(directory, barrier)
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, barrier)
    with pytest.raises(TimeoutError):
        client.wait_restore(handle, timeout=0.02)
    server.process.kill()
    server.process.wait(timeout=10)
    # Manager death cannot complete a local engine's DMA fence.
    assert not client.poll_restore(handle).done
    (directory / f"{barrier}.pause").unlink()
    status = client.wait_restore(handle, timeout=10)
    assert status.success, status.message
    assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)


def test_local_partial_enqueue_failure_drains_before_page_reuse(fault_cache):
    import torch

    server, client, ctx, directory = fault_cache
    assert publish(client, ctx, [b"partial-local"])[0]
    ready = query(client, ctx, [b"partial-local"], "partial-local")
    arm(directory, "local_restore_error")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    result = client.wait_restore(handle, timeout=10)
    assert result.done and not result.success
    assert "after first local Restore enqueue" in result.message
    ctx.get_kv_cache()[:, 2:3].fill_(37)
    torch.cuda.synchronize()
    assert torch.all(ctx.get_kv_cache()[:, 2:3] == 37)
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == 0)


def test_engine_death_after_claim_quarantines_source_reservation(fault_cache):
    import subprocess
    import sys

    server, client, ctx, directory = fault_cache
    assert publish(client, ctx, [b"engine-crash-source"])[0]
    ready = query(client, ctx, [b"engine-crash-source"], "seal-crash-source")
    client.release(ready.lease)
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == 0)
    arm(directory, "local_restore_claim")
    helper = Path(__file__).parents[1] / "support" / "local_restore_process.py"
    log = directory / "crashing-engine.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-P", str(helper), server.bootstrap_socket, ctx.namespace],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            until(
                lambda: (
                    (directory / "local_restore_claim.reached").exists()
                    or process.poll() is not None
                )
            )
            assert process.poll() is None, log.read_text()
            before = fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"]
            assert before > 0
            process.kill()
            process.wait(timeout=10)
            time.sleep(1)
            assert client.health()[0]
            assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] >= before
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_local_restore_retains_tensor_when_caller_drops_handle_and_tensor(fault_cache):
    import gc
    import weakref

    import torch

    server, client, ctx, directory = fault_cache
    tensor = ctx.get_kv_cache()
    expected = tensor[:, 0:1].cpu().clone()
    assert publish(client, ctx, [b"tensor-owner"])[0]
    ready = query(client, ctx, [b"tensor-owner"], "tensor-owner")
    arm(directory, "local_restore_claim")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    reached(directory, "local_restore_claim")
    reference = weakref.ref(tensor)
    ctx.gpu_kv_caches.clear()
    del tensor, handle
    gc.collect()
    assert reference() is not None
    (directory / "local_restore_claim.pause").unlink()
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == 0)
    assert torch.equal(reference()[:, 2:3].cpu(), expected)
    assert client.unregister_context(ctx.instance_id)[0]
    gc.collect()
    assert reference() is None


def test_local_restore_fences_previous_use_on_nondefault_stream(fault_cache):
    import torch

    _server, client, ctx, _directory = fault_cache
    tensor = ctx.get_kv_cache()
    expected = tensor[:, 0:1].cpu().clone()
    assert publish(client, ctx, [b"stream-order"])[0]
    for attempt in range(2):
        ready = query(client, ctx, [b"stream-order"], f"stream-order-{attempt}")
        previous = torch.cuda.Stream()
        with torch.cuda.stream(previous):
            torch.cuda._sleep(20_000_000)
            tensor[:, 2:3].fill_(37 + attempt)
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[2]])],
            ready_stream=previous.cuda_stream,
        )
        assert client.wait_restore(handle, timeout=10).success
        assert torch.equal(tensor[:, 2:3].cpu(), expected)


@pytest.mark.skipif(
    os.environ.get("ORBITKV_COST_OBSERVATIONS") != "1"
    or os.environ.get("ORBITKV_TRACE_TRANSFERS") != "1",
    reason="requires cost observations and transfer tracing in both processes",
)
def test_local_completion_evidence_excludes_retirement_and_trains_once(fault_cache):
    import json

    import torch

    from orbitkv import OrbitKVError

    server, client, ctx, directory = fault_cache
    hashes = [b"completion-evidence"]
    expected = ctx.get_kv_cache()[:, 0:1].cpu().clone()
    assert publish(client, ctx, hashes)[0]
    ready = query(client, ctx, hashes, "completion-evidence")
    arm(directory, "local_restore_reap")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [[2]])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
    )
    try:
        assert client.wait_restore(handle, timeout=5).success
        reached(directory, "local_restore_reap")
        assert torch.equal(ctx.get_kv_cache()[:, 2:3].cpu(), expected)
        assert fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] > 0
        # GPU readiness and native result consumption precede Manager retirement.
        time.sleep(0.2)
        assert "local_restore_complete" not in server.read_logs()
    finally:
        (directory / "local_restore_reap.pause").unlink(missing_ok=True)
    until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == 0)
    until(lambda: "local_restore_complete" in server.read_logs())
    reports = [
        json.JSONDecoder().raw_decode(line.partition("cache_timeline ")[2])[0]
        for line in server.read_logs().splitlines()
        if "cache_timeline " in line
    ]
    reports = [event for event in reports if event["stage"] == "local_restore_complete"]
    assert len(reports) == 1
    report = reports[0]
    assert report["restore_key"] == handle.key
    assert report["success"]
    assert 0 < report["readiness_ns"] <= report["dispatched_ns"] <= report["dequeued_ns"]
    assert (
        report["dequeued_ns"]
        <= report["claimed_ns"]
        <= report["submitted_ns"]
        <= report["drained_ns"]
    )
    response = requests.get(f"http://127.0.0.1:{server.http_port}/metrics", timeout=5)
    response.raise_for_status()
    samples = {
        line.split("{", 1)[0]: float(line.rsplit(" ", 1)[1])
        for line in response.text.splitlines()
        if 'path="engine_local_restore"' in line
        and 'stage="total"' in line
        and (
            line.startswith("orbitkv_cost_stage_seconds_count{")
            or line.startswith("orbitkv_cost_stage_seconds_sum{")
        )
    }
    assert samples["orbitkv_cost_stage_seconds_count"] == 1
    assert samples["orbitkv_cost_stage_seconds_sum"] == pytest.approx(report["drained_ns"] / 1e9)
    with pytest.raises(OrbitKVError, match="consumed"):
        client.poll_restore(handle)


@pytest.mark.parametrize(
    "fault_cache", [{"num_blocks": 4096, "block_size": 1, "num_layers": 5}], indirect=True
)
@pytest.mark.parametrize("outcome", ["success", "failure", "manager_exit"])
def test_partitioned_restore_keeps_owners_and_completes_only_after_all_parts(fault_cache, outcome):
    import torch

    server, client, ctx, directory = fault_cache
    tensors = [ctx.get_kv_cache(layer) for layer in range(ctx.num_layers)]
    count = ctx.num_blocks // 2
    expected = []
    for layer, tensor in enumerate(tensors):
        values = torch.arange(count, device=tensor.device, dtype=torch.float32).remainder_(251) / 16
        tensor[0, :count] = values.to(tensor.dtype).reshape(count, 1, 1, 1) + layer * 4
        tensor[1, :count] = tensor[0, :count] + 2
        expected.append(tensor[:, :count].flip(1).cpu().clone())
        tensor[:, count:].fill_(-7)
    torch.cuda.synchronize()
    hashes = [index.to_bytes(8, "little") for index in range(count)]
    assert client.save(
        ctx.registration, [(name, list(range(count)), hashes) for name in ctx._layer_names]
    )[0]
    ready = query(client, ctx, hashes, "partitioned")
    assert ready.num_hit_blocks == count
    events = [torch.cuda.Event(external=True) for _ in tensors]
    for event in events:
        event.record()
    arm(directory, "local_restore_part_drained")
    handle = client.start_restore(
        ctx.instance_id,
        0,
        0,
        [ctx._layer_names],
        [(ready.lease, [list(range(2 * count - 1, count - 1, -1))])],
        ready_stream=torch.cuda.current_stream(0).cuda_stream,
        layer_events=list(zip(ctx._layer_names, events, strict=True)),
    )
    reached(directory, "local_restore_part_drained")
    with pytest.raises(TimeoutError):
        client.wait_restore_enqueued(handle, timeout=0.02)
    with pytest.raises(TimeoutError):
        client.wait_restore(handle, timeout=0.02)
    assert not client.poll_restore(handle).done
    held = fetch_orbitkv_metrics(server.http_port)
    assert held["orbitkv_query_reserved_bytes"] > 0
    assert held.get("orbitkv_load_bytes_total", 0) == 0
    if outcome == "failure":
        arm(directory, "local_restore_error")
    if outcome == "manager_exit":
        server.process.kill()
        server.process.wait(timeout=10)
    (directory / "local_restore_part_drained.pause").unlink()
    if outcome == "success":
        client.wait_restore_enqueued(handle, timeout=10)
        for event in events:
            torch.cuda.current_stream().wait_event(event)
    else:
        from orbitkv import OrbitKVError

        with pytest.raises(OrbitKVError):
            client.wait_restore_enqueued(handle, timeout=10)
    status = client.wait_restore(handle, timeout=10)
    if outcome == "success":
        assert status.success, status.message
        for tensor, reference in zip(tensors, expected, strict=True):
            assert torch.equal(tensor[:, count:].cpu(), reference)
    else:
        assert not status.success
        assert (
            "after first local Restore enqueue" if outcome == "failure" else "reconnect"
        ) in status.message
        for tensor in tensors:
            tensor[:, count:].fill_(37)
        torch.cuda.synchronize()
        assert all(torch.all(tensor[:, count:] == 37) for tensor in tensors)
    if outcome != "manager_exit":
        until(lambda: fetch_orbitkv_metrics(server.http_port)["orbitkv_query_reserved_bytes"] == 0)
        stats = fetch_orbitkv_metrics(server.http_port)
        assert stats.get("orbitkv_load_bytes_total", 0) == (
            sum(tensor.numel() * tensor.element_size() for tensor in expected)
            if outcome == "success"
            else 0
        )
