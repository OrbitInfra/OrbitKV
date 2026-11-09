"""Full/window/checkpoint registration and exact CUDA recovery together."""

import time
import uuid
from types import SimpleNamespace

import pytest
import requests

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("channel_server", ["dram", "ssd"], indirect=True)
@pytest.mark.parametrize("temporal_state", [False, True])
def test_three_component_recovery(channel_server, monkeypatch, temporal_state):
    torch = pytest.importorskip("torch")
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.layout import GpuLayout
    from orbitkv.sglang.linker import OrbitKVLinker
    from tests.support.metrics import fetch_orbitkv_metrics

    # Real contiguous GPU buffers in the ordinary engine pool representation.
    # Construction avoids allocating the engine's entire serving budget.
    def attention_pool(offset):
        pool = object.__new__(MHATokenToKVPool)
        pool.k_buffer = [
            torch.arange(64 * 128, device="cuda", dtype=torch.float32).reshape(64, 1, 128) + offset
        ]
        pool.v_buffer = [pool.k_buffer[0].clone() + 10000]
        return pool

    cache = object.__new__(SWAKVPool)
    cache.full_kv_pool = attention_pool(0)
    cache.swa_kv_pool = attention_pool(20000)
    cache.layers_mapping = {0: (0, False), 1: (0, True)}
    cache.start_layer, cache.end_layer = 0, 3 if temporal_state else 2
    conv = torch.arange(64 * 128, device="cuda", dtype=torch.float32).reshape(1, 64, 128) + 40000
    temporal = (
        torch.arange(64 * 256, device="cuda", dtype=torch.float32).reshape(1, 64, 256) + 50000
    )
    state = SimpleNamespace(
        conv=[conv], temporal=temporal if temporal_state else temporal[:, :, :0]
    )
    params = SimpleNamespace(
        page_size=4,
        sliding_window_size=8,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: cache),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=state),
            mamba_map={2 if temporal_state else 0: 0},
            translate_mamba_indices=lambda indices: indices,
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_lora", lambda: SimpleNamespace(enable_lora=False)
    )
    monkeypatch.setenv("ORBITKV_SGLANG_ENDPOINT", f"unix://{channel_server.bootstrap_socket}")
    monkeypatch.setattr(
        "orbitkv.sglang.linker.derive_namespace",
        lambda *args, **kwargs: f"combined-{uuid.uuid4().hex}",
    )
    cache.layer_transfer_counter = _LayerDoneCounter(
        GpuLayout.from_pool(
            params.page_size,
            cache,
            params.req_to_token_pool,
            {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA},
            params.sliding_window_size,
        )
    )
    linker = OrbitKVLinker(None, params, forward_stream=None)
    tensors = [
        *cache.full_kv_pool.k_buffer,
        *cache.full_kv_pool.v_buffer,
        *cache.swa_kv_pool.k_buffer,
        *cache.swa_kv_pool.v_buffer,
        conv[0],
    ]
    if temporal_state:
        tensors.append(temporal[0])
    expected = [tensor.clone() for tensor in tensors]
    keys = [f"page-{index}" for index in range(4)]
    transfers = [PoolTransfer(name=name, keys=keys) for name in linker.layout.pools]
    deadline = time.monotonic() + 20

    def offload(transfers):
        assert linker.offload(transfers)
        while not linker.num_completed_offloads():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert linker.pop_completed_offload()

    try:
        assert [pool.group_id for pool in linker.layout.pools.values()] == [0, 1, 2]
        assert linker.layout.num_layers == cache.end_layer
        offload(
            [
                PoolTransfer(
                    name=PoolName.KV, keys=keys, device_indices=torch.arange(4, 20, device="cuda")
                ),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=keys[3:],
                    device_indices=torch.arange(16, 20, device="cuda"),
                ),
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=keys[3:],
                    device_indices=torch.tensor([1], device="cuda"),
                ),
            ]
        )
        linker._origins["incomplete"] = 64
        while True:
            result = linker.lookup("incomplete", transfers)
            if "incomplete" not in linker._pending_queries:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == []
        linker.cancel_query("incomplete")
        offload(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=keys[2:3],
                    device_indices=torch.arange(12, 16, device="cuda"),
                ),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=keys[2:3],
                    device_indices=torch.arange(12, 16, device="cuda"),
                ),
            ]
        )
        if channel_server.ssd_cache_path is not None:
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
            assert response.json()["evicted_blocks"] == 7
        linker._origins["restore"] = 64
        while not (result := linker.lookup("restore", transfers)):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert result == [4]
        assert (
            fetch_orbitkv_metrics(channel_server.http_port).get(
                "orbitkv_ssd_prefetch_bytes_total", 0
            )
            == 0
        )
        while linker.prepare_recovery("restore", 80) == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        linker._load_boundaries["restore"] = (80, {})
        for tensor in tensors[:4]:
            tensor[32:48].fill_(-1)
        for tensor in tensors[4:]:
            tensor[5].fill_(-1)
        assert linker.load(
            "restore",
            [
                PoolTransfer(
                    name=PoolName.KV, keys=keys, device_indices=torch.arange(32, 48, device="cuda")
                ),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=keys[2:],
                    device_indices=torch.arange(40, 48, device="cuda"),
                ),
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=keys[3:],
                    device_indices=torch.tensor([5], device="cuda"),
                ),
            ],
        )
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(linker.layout.num_layers - 1)
        torch.cuda.synchronize()
        for tensor, saved in zip(tensors[:2], expected[:2], strict=True):
            assert torch.equal(tensor[32:48], saved[4:20])
        for tensor, saved in zip(tensors[2:4], expected[2:4], strict=True):
            assert torch.equal(tensor[40:48], saved[12:20])
            assert (tensor[32:40] == -1).all()
        for tensor, saved in zip(tensors[4:], expected[4:], strict=True):
            assert torch.equal(tensor[5], saved[1])
        metrics = fetch_orbitkv_metrics(channel_server.http_port)
        counts = {PoolName.KV: 4, PoolName.SWA: 2, PoolName.MAMBA: 1}
        payload = sum(
            counts[name] * sum(pool.block_bytes) for name, pool in linker.layout.pools.items()
        )
        assert metrics["orbitkv_load_bytes_total"] == payload
        if channel_server.ssd_cache_path is not None:
            assert metrics["orbitkv_ssd_prefetch_bytes_total"] == payload
        assert metrics.get("orbitkv_query_reserved_bytes", 0) == 0
    finally:
        linker.close()
