"""Exercise compiled hybrid recovery against real CUDA buffers and cache leases."""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("channel_server", ["dram", "ssd"], indirect=True)
@pytest.mark.parametrize("kind", ["recurrent", "window"])
def test_hybrid_recovery_requires_and_restores_complete_state(channel_server, monkeypatch, kind):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang")
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.layout import GpuLayout, GpuPool
    from orbitkv.sglang.linker import OrbitKVLinker
    from orbitkv.sglang.recovery import RecoveryLinkerWrapper
    from tests.support.metrics import fetch_orbitkv_metrics

    page_size = 4
    kv = torch.arange(64 * 128, dtype=torch.float32, device="cuda").reshape(64, 128)
    conv = torch.arange(64 * 128, dtype=torch.float32, device="cuda").reshape(64, 128) + 1000
    temporal = torch.arange(64 * 256, dtype=torch.float32, device="cuda").reshape(64, 256) + 2000
    expected = [tensor.clone() for tensor in (kv, conv, temporal)]

    def pool(name, kind, buffers, tokens, layer):
        entry = DevicePoolEntry(
            name=name,
            indices_from_pool=name,
            device_pool=None,
            components=[[tensor] for tensor in buffers],
            layer_mapping={layer: 0},
            page_size=tokens,
            rows_are_pages=tokens == 1,
        )
        return GpuPool.compile(layer, kind, 8 if kind == "window" else 0, entry)

    auxiliary = PoolName.MAMBA if kind == "recurrent" else PoolName.SWA
    source = (
        torch.tensor([1], device="cuda")
        if kind == "recurrent"
        else torch.arange(12, 20, device="cuda")
    )
    target = (
        torch.tensor([5], device="cuda")
        if kind == "recurrent"
        else torch.arange(40, 48, device="cuda")
    )

    layout = GpuLayout(
        page_size,
        2,
        {
            PoolName.KV: pool(PoolName.KV, "attention", [kv], page_size, 0),
            auxiliary: pool(
                auxiliary, kind, [conv, temporal], 1 if kind == "recurrent" else page_size, 1
            ),
        },
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    namespace = f"hybrid-proof-{uuid.uuid4().hex}"
    cache = SimpleNamespace(layer_transfer_counter=_LayerDoneCounter(layout))
    params = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: cache))
    monkeypatch.setattr("orbitkv.sglang.linker.derive_namespace", lambda *args, **kwargs: namespace)
    monkeypatch.setenv("ORBITKV_SGLANG_ENDPOINT", f"unix://{channel_server.bootstrap_socket}")
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_lora", lambda: SimpleNamespace(enable_lora=False)
    )
    linker = OrbitKVLinker(None, params)
    keys = [f"prefix-{i}" for i in range(4)]
    state_keys = keys[-1:] if kind == "recurrent" else keys[-2:]
    try:
        assert linker.offload(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=keys[:2],
                    device_indices=torch.arange(4, 12, device="cuda"),
                )
            ]
        )
        assert linker.offload(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=keys[2:],
                    device_indices=torch.arange(12, 20, device="cuda"),
                ),
                PoolTransfer(
                    name=auxiliary,
                    keys=state_keys,
                    device_indices=source,
                ),
            ]
        )
        deadline = time.monotonic() + 10
        for _ in range(2):
            while not linker.num_completed_offloads():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert linker.pop_completed_offload()
        # Seed a group left without attention, as independent tier eviction can
        # produce. The SGLang offload callback itself publishes complete groups.
        orphan = "prefix-without-attention"
        orphan_pool = layout.pools[auxiliary]
        orphan_blocks = orphan_pool.block_ids(
            source if kind == "recurrent" else source[-page_size:], 1
        )
        torch.cuda.synchronize()
        saved, message = linker.client.save(
            linker.registration,
            [
                (layer, orphan_blocks * 2, linker._hashes([orphan, keys[0]]))
                for layer in orphan_pool.layer_names
            ],
        )
        assert saved, message
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
                assert time.monotonic() < deadline, (metrics, channel_server.read_logs())
                time.sleep(0.01)
            response = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/memory/cleanup", timeout=10
            )
            response.raise_for_status()
            assert response.json()["evicted_blocks"] > 0

        transfers = [PoolTransfer(name=name, keys=[*keys, orphan]) for name in layout.pools]
        linker._origins["restore"] = 64
        while not (boundaries := linker.lookup("restore", transfers)):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # Attention pages 1..4 exist; boundaries 1 and 4 have complete auxiliary state; only 4 is selected.
        assert boundaries == [1, 4]
        if channel_server.ssd_cache_path is not None:
            assert (
                fetch_orbitkv_metrics(channel_server.http_port).get(
                    "orbitkv_ssd_prefetch_bytes_total", 0
                )
                == 0
            )
        while linker.prepare_recovery("restore", 80) == 0:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        held = linker._lookups["restore"].groups[auxiliary]
        assert held.hit_positions == ([3] if kind == "recurrent" else [2, 3])
        linker._load_boundaries["restore"] = (80, {PoolName.KV: {keys[0]}})
        kv[32:48].fill_(-1)
        # The engine may retain a page populated after lookup and before load.
        kv[32:36].copy_(expected[0][4:8])
        conv[target] = -1
        temporal[target] = -1
        assert linker.load(
            "restore",
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=keys[1:],
                    device_indices=torch.arange(36, 48, device="cuda"),
                ),
                PoolTransfer(
                    name=auxiliary,
                    keys=state_keys,
                    device_indices=target,
                ),
            ],
        )
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(1)
        torch.cuda.synchronize()
        assert torch.equal(kv[32:48], expected[0][4:20])
        assert torch.equal(conv[target], expected[1][source])
        assert torch.equal(temporal[target], expected[2][source])
        metrics = fetch_orbitkv_metrics(channel_server.http_port)
        payload = 3 * sum(layout.pools[PoolName.KV].block_bytes) + len(state_keys) * sum(
            layout.pools[auxiliary].block_bytes
        )
        assert metrics.get("orbitkv_load_bytes_total", 0) == payload
        if channel_server.ssd_cache_path is not None:
            # Stored blocks pad each registered tensor to the 512-byte slot alignment.
            stored = 4 * sum(
                (size + 511) // 512 * 512 for size in layout.pools[PoolName.KV].block_bytes
            )
            stored += len(state_keys) * sum(
                (size + 511) // 512 * 512 for size in layout.pools[auxiliary].block_bytes
            )
            assert metrics.get("orbitkv_ssd_prefetch_bytes_total", 0) == stored

        for rid, boundary, omit_state in [
            ("wrong-boundary", 76, False),
            ("missing-state", 80, True),
            ("overlapping-state", 80, False),
            ("unneeded-state", 80, False),
        ]:
            linker._origins[rid] = 64
            assert linker.lookup(rid, transfers) == [1, 4]
            assert linker.prepare_recovery(rid, 80) == 1
            linker._load_boundaries[rid] = (
                boundary,
                {PoolName.KV: {keys[0]}} if rid == "overlapping-state" else {},
            )
            selected = [
                PoolTransfer(
                    name=PoolName.KV, keys=keys, device_indices=torch.arange(32, 48, device="cuda")
                )
            ]
            if not omit_state:
                selected.append(
                    PoolTransfer(
                        name=auxiliary,
                        keys=[keys[0], *state_keys[1:]] if rid == "unneeded-state" else state_keys,
                        device_indices=target,
                    )
                )
            with pytest.raises(
                ValueError,
                match="unproved recovery boundary|do not cover the recovery plan|outside its required",
            ):
                linker.load(rid, selected)
            assert rid not in linker._queued_loads

        # An aborted request must finish already-published tree destinations,
        # otherwise a later HBM match can read their uninitialized bytes.
        linker.pop_completed_load()
        rid = "cancel-published"
        linker._origins[rid] = 64
        assert linker.lookup(rid, transfers) == [1, 4]
        assert linker.prepare_recovery(rid, 80) == 1
        linker._load_boundaries[rid] = (80, {})
        conv[target] = -1
        temporal[target] = -1
        linker.load(
            rid,
            [
                PoolTransfer(
                    name=PoolName.KV, keys=keys, device_indices=torch.arange(32, 48, device="cuda")
                ),
                PoolTransfer(name=auxiliary, keys=state_keys, device_indices=target),
            ],
        )
        wrapper = object.__new__(RecoveryLinkerWrapper)
        wrapper.restore_from_store = True
        wrapper.cache_linker = linker
        wrapper.cache = SimpleNamespace(dec_lock_ref=MagicMock())
        wrapper.hit_markers = {}
        wrapper.pending_loads = {rid: (7, "pin")}
        wrapper.release_request(rid)
        wrapper.cache.dec_lock_ref.assert_called_once_with(7, "pin")
        assert not wrapper.pending_loads
        assert rid not in linker._origins
        torch.cuda.synchronize()
        assert torch.equal(conv[target], expected[1][source])
        assert torch.equal(temporal[target], expected[2][source])

        # A cache hit can become entirely resident before destination allocation.
        # No load() call occurs, but all lookup evidence still needs retirement.
        rid = "became-resident"
        req = SimpleNamespace(rid=rid)
        wrapper.cache.page_size = page_size
        monkeypatch.setattr(
            UnifiedCacheLinkerWrapper,
            "match",
            lambda self, key, req, result: self.cache_linker.lookup(req.rid, transfers),
        )
        assert wrapper.match(None, req, SimpleNamespace(device_indices=torch.arange(64))) == [1, 4]
        assert rid not in linker._origins
        wrapper.hit_markers[rid] = SimpleNamespace(device_hit_len=64, tail_hashes=keys)
        monkeypatch.setattr(UnifiedCacheLinkerWrapper, "load_back", lambda self, req: None)
        wrapper.load_back(req)
        assert rid not in linker._lookups
        assert rid not in linker._load_boundaries
    finally:
        linker.close()
