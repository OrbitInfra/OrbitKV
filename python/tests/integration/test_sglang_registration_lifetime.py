"""Exercise registration retirement through the released SGLang pool and linker."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import weakref
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from tests.support.ipc_credit import hold_counters, release_receipt
from tests.support.metrics import fetch_orbitkv_metrics
from tests.support.registration_campaign import identity, write

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("independent_hold", [False, True], ids=["sole-owner", "independent-hold"])
@pytest.mark.parametrize(
    "channel_server",
    [{"tier": "dram", "installed": True}, {"tier": "ssd", "installed": True}],
    ids=["dram", "ssd"],
    indirect=True,
)
def test_sglang_close_retires_only_its_registration(
    channel_server, model, monkeypatch, tmp_path, independent_hold
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sglang")
    if importlib.metadata.version("sglang") != "0.5.21":
        pytest.fail("Prepared Save pool gate requires official SGLang 0.5.21")
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
    counters = contextlib.ExitStack()
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
        request_pool = ReqToTokenPool(2, 256, "cuda:0", False)
        allocator = PagedTokenToKVPoolAllocator(256, 64, torch.int64, "cuda:0", pool, False)
        params = CacheInitParams(False, request_pool, allocator, 64)
        layout = GpuLayout.from_pool(64, pool, request_pool, {ComponentType.FULL})
        counter = _LayerDoneCounter(layout)
        pool.register_layer_transfer_counter(counter)
        linker = OrbitKVLinker(args, params)
        channel_server.record_runtime()
        exporters = [weakref.ref(tensor) for tensor in layout.pools[PoolName.KV].entry.kv_buffer]
        assert len(exporters) == 4
        generation = linker.registration.generation.hex()
        write(
            tmp_path / "original.credit-owner.json",
            {
                "owner": identity(os.getpid()),
                "instance_id": linker.instance_id,
                "evidence": json.loads(linker.registration.ipc_credit_evidence),
            },
        )
        held_counters = hold_counters(counters, tmp_path / "original.credit-owner.json")
        geometry = [
            {"object_id": id(tensor), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for tensor in layout.pools[PoolName.KV].entry.kv_buffer
        ]
        expected = []
        for index, tensor in enumerate(pool.k_buffer + pool.v_buffer):
            tensor.fill_(-167)
            tensor[64:128] = (
                torch.arange(tensor[64:128].numel(), device="cuda").reshape(tensor[64:128].shape)
                % 127
                + index * 16
            ).to(tensor.dtype)
            original = tensor.cpu().clone()
            original[128:192] = original[64:128]
            expected.append(original)
        del tensor
        native_client = linker.client
        observed = MagicMock(wraps=native_client)
        linker.client = observed
        transfer = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(64, 128, device="cuda:0"),
            keys=["registration-retirement"],
        )
        assert linker.offload([transfer])
        observed.prepare_save.assert_called_once()
        deadline = time.monotonic() + 30
        while not linker.num_completed_offloads():
            assert time.monotonic() < deadline, "Real SGLang offload did not complete"
            time.sleep(0.01)
        assert linker.pop_completed_offload() is True
        observed.submit_saves.assert_called_once()
        observed.wait_save.assert_called_once()
        observed.start_save.assert_not_called()
        if channel_server.ssd_cache_path:
            synced = requests.post(
                f"http://127.0.0.1:{channel_server.http_port}/cache/sync", timeout=35
            )
            synced.raise_for_status()
            assert synced.json()["local_flush_complete"] is True
            from tests.support.cache_manager import evict_dram_after_ssd_writes

            evict_dram_after_ssd_writes(channel_server.http_port)
        before_restore = fetch_orbitkv_metrics(channel_server.http_port)
        deadline = time.monotonic() + 30
        rid = "prepared-lifetime"
        linker._origins[rid] = 0
        load = PoolTransfer(
            name=PoolName.KV,
            device_indices=torch.arange(128, 192, device="cuda:0"),
            keys=transfer.keys,
        )
        while True:
            boundaries = linker.lookup(rid, [load])
            if boundaries == [1] and linker.prepare_recovery(rid, 64) == 1:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # The released recovery wrapper supplies this semantic boundary before load.
        linker._load_boundaries[rid] = (64, {})
        assert linker.load(rid, [load])
        index = linker.start_layer_wise_loading()
        assert index >= 0
        linker.layer_done_counter.set_consumer(index)
        while not linker.num_completed_loads():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert linker.pop_completed_load() == [rid]
        observed.start_restore.assert_called_once()
        observed.wait_restore.assert_called_once()
        torch.cuda.synchronize()
        payloads = []
        for tensor, original in zip(pool.k_buffer + pool.v_buffer, expected, strict=True):
            actual = tensor.cpu()
            assert torch.equal(actual, original), "SGLang complete pool/sentinel oracle differs"
            actual.flatten()[0] += 1
            assert not torch.equal(actual, original), "mutation escaped pool byte oracle"
            actual.flatten()[0] -= 1
            payloads.append(actual.view(torch.uint8).numpy().tobytes())
        del tensor, actual, original, expected, load
        (tmp_path / "sglang-prepared-save.payload.bin").write_bytes(b"".join(payloads))
        after_restore = fetch_orbitkv_metrics(channel_server.http_port)
        if channel_server.ssd_cache_path:
            for name in (
                "orbitkv_ssd_prefetch_bytes_total",
                "orbitkv_ssd_uring_read_operations_total",
            ):
                assert name in after_restore and after_restore[name] > before_restore.get(name, 0)
        linker.client = native_client
        del observed, native_client
        retained = linker.registration if independent_hold else None
        instance = linker.instance_id
        close_attempted = True
        linker.close()
        write(tmp_path / "original.credit-release.json", release_receipt(held_counters))
        assert linker.registration is None
        assert not linker._load_thread.is_alive() and not linker._offload_thread.is_alive()
        assert linker._load_queue.unfinished_tasks == linker._offload_queue.unfinished_tasks == 0
        with pytest.raises(OrbitKVError, match="reconnect"):
            linker.client.health()
        del linker, pool, request_pool, allocator, params, layout, counter, transfer
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
                    "release": "0.5.21",
                    "prepared_submission_consumed": True,
                    "linker_load_consumed": True,
                    "full_byte_oracle": True,
                    "mutation_rejected": True,
                    "full_buffer_bytes": sum(len(payload) for payload in payloads),
                    "payload_sha256": hashlib.sha256(b"".join(payloads)).hexdigest(),
                    "physical_fault_termination_claimed": False,
                    "tier": "ssd-uring" if channel_server.ssd_cache_path else "dram",
                    "read_before": before_restore,
                    "read_after": after_restore,
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
            counters.close()
            restore_context(previous)
