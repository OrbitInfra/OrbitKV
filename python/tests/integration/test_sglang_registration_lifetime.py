"""Exercise registration retirement through the released SGLang pool and linker."""

from __future__ import annotations

import gc
import json
import os
import time
import weakref
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("independent_hold", [False, True], ids=["sole-owner", "independent-hold"])
def test_sglang_close_retires_only_its_registration(
    channel_server, model, monkeypatch, tmp_path, independent_hold
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang")
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.runtime_context import (
        SpawnRanks,
        publish,
        restore_context,
        snapshot_context,
    )
    from sglang.srt.server_args import ServerArgs

    from orbitkv import OrbitKVError
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.layout import GpuLayout
    from orbitkv.sglang.linker import OrbitKVLinker

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("ORBITKV_SGLANG_ENDPOINT", f"unix://{channel_server.bootstrap_socket}")
    monkeypatch.setenv("ORBITKV_TRANSFER_BACKEND", "direct")
    previous = snapshot_context()
    linker = None
    close_attempted = False
    try:
        args = ServerArgs(model_path=model, dtype="bfloat16", page_size=64, tp_size=1, pp_size=1)
        publish(args, role="scheduler", ranks=SpawnRanks(world_rank=0, dp_rank=0, gpu_id=0))
        pool = MHATokenToKVPool(
            size=256,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=8,
            head_dim=128,
            layer_num=2,
            device="cuda:0",
            enable_memory_saver=False,
            start_layer=0,
            end_layer=2,
            enable_alt_stream=False,
            kv_cache_layout="nhd",
        )
        requests = ReqToTokenPool(2, 256, "cuda:0", False)
        allocator = PagedTokenToKVPoolAllocator(256, 64, torch.int64, "cuda:0", pool, False)
        params = CacheInitParams(False, requests, allocator, 64)
        layout = GpuLayout.from_pool(64, pool, requests, {ComponentType.FULL})
        counter = _LayerDoneCounter(layout)
        pool.register_layer_transfer_counter(counter)
        linker = OrbitKVLinker(args, params)
        exporters = [weakref.ref(tensor) for tensor in layout.pools[PoolName.KV].entry.kv_buffer]
        assert len(exporters) == 4
        generation = linker.registration.generation.hex()
        geometry = [
            {"object_id": id(tensor), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for tensor in layout.pools[PoolName.KV].entry.kv_buffer
        ]
        for tensor in pool.k_buffer + pool.v_buffer:
            tensor.fill_(1.25)
        del tensor
        transfer = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(64, 128, device="cuda:0"),
            keys=["registration-retirement"],
        )
        assert linker.offload([transfer])
        deadline = time.monotonic() + 30
        while not linker.num_completed_offloads():
            assert time.monotonic() < deadline, "Real SGLang offload did not complete"
            time.sleep(0.01)
        assert linker.pop_completed_offload() is True
        retained = linker.registration if independent_hold else None
        instance = linker.instance_id
        close_attempted = True
        linker.close()
        assert linker.registration is None
        assert not linker._load_thread.is_alive() and not linker._offload_thread.is_alive()
        assert linker._load_queue.unfinished_tasks == linker._offload_queue.unfinished_tasks == 0
        with pytest.raises(OrbitKVError, match="reconnect"):
            linker.client.health()
        del linker, pool, requests, allocator, params, layout, counter, transfer
        linker = None
        torch.cuda.synchronize()
        gc.collect()
        alive_after_close = sum(reference() is not None for reference in exporters)
        assert alive_after_close == (4 if independent_hold else 0)
        del retained
        gc.collect()
        assert all(reference() is None for reference in exporters)
        (tmp_path / "sglang-registration.native-release.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "start_ticks": int(
                        Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
                    ),
                    "instance_id": instance,
                    "registration_generation": generation,
                    "exporters": geometry,
                    "independent_hold": independent_hold,
                    "exporters_after_close": alive_after_close,
                    "exporters_after_last_hold": 0,
                    "offload_completed": True,
                    "workers_joined": True,
                    "registration_retired": True,
                    "client_closed": True,
                    "completed_mono_ns": time.monotonic_ns(),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        try:
            if linker is not None and not close_attempted:
                linker.close()
        finally:
            restore_context(previous)
