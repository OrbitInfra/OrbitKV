"""Released V2 ordering and allocator controls, with no GPU or serving claim.

Only producer events and Manager replies are controlled. Scheduler, request,
allocator and connector dispatch are the installed vLLM classes.
"""

import os
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.test_cache_faults import fault_cache  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def allocation():
    import torch
    from vllm import SamplingParams
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request, RequestStatus

    from orbitkv.vllm.config import ConnectorContext
    from orbitkv.vllm.connector import OrbitKVConnector
    from orbitkv.vllm.scheduler import SchedulerAdapter

    register_all_kvcache_specs(SimpleNamespace())
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
    cache = KVCacheConfig(2, [], [KVCacheGroupSpec(["layer"], spec)])
    manager = KVCacheManager(cache, 64, 16, 16, enable_caching=False)
    request = Request("request", [1] * 16, SamplingParams(max_tokens=32), None)
    request.status = RequestStatus.RUNNING
    assert manager.allocate_slots(request, 16) is not None
    request.num_computed_tokens = 16
    block = manager.get_blocks(request.request_id).blocks[0][0]

    # No constructor replacement in vLLM: allocate the actual scheduler object
    # and supply just the CPU state consumed by its retirement methods.
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_cache_manager = manager
    scheduler.kv_cache_config = cache
    scheduler.aux_output_connector = scheduler.ec_connector = None
    scheduler.encoder_cache_manager = EncoderCacheManager(0)
    scheduler._pending_encoder_input_ids = {}
    scheduler._inflight_prefills = set()
    scheduler._kv_fetch_stages = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.requests = {request.request_id: request}
    scheduler.defer_block_free = False
    scheduler.sched_step_seq = scheduler.processed_step_seq = 0
    scheduler.deferred_frees = deque()
    scheduler.running = [request]
    scheduler.waiting = FCFSRequestQueue()
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.deferred_waiting = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.log_stats = False
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=None)

    adapter = SchedulerAdapter(ConnectorContext("v2", "test", 16, 1, 1, 0, 0, MagicMock()))
    scheduler.connector = OrbitKVConnector.__new__(OrbitKVConnector)
    scheduler.connector._scheduler = adapter
    scheduler.connector._worker = None
    yield SimpleNamespace(
        scheduler=scheduler,
        adapter=adapter,
        manager=manager,
        request=request,
        block=block,
        cache=cache,
    )
    adapter.shutdown()


def attach_worker(h, worker, monkeypatch):
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from orbitkv.vllm.connector import OrbitKVConnector

    connector = OrbitKVConnector.__new__(OrbitKVConnector)
    KVConnectorBase_V1.__init__(
        connector,
        SimpleNamespace(
            kv_transfer_config=KVTransferConfig(kv_connector="OrbitKVConnector", kv_role="kv_both")
        ),
        KVConnectorRole.WORKER,
        h.cache,
    )
    connector._worker = worker
    connector._scheduler = None
    connector._ctx = SimpleNamespace(query_control=False)
    active = ActiveKVConnector.__new__(ActiveKVConnector)
    active.kv_connector = connector
    active._disabled = False
    active._pending_load_kwargs = None
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.kv_connector.is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.kv_connector.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}, cudagraph_runtime_mode=None),
    )
    h.worker, h.active = worker, active
    return h


@pytest.fixture
def lifecycle(allocation, monkeypatch):
    import torch

    from tests.unit.test_connector_save_lifecycle import make_worker

    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    return attach_worker(allocation, make_worker(), monkeypatch)


def test_full_single_group_has_no_pre_hook_kv_writes(lifecycle):
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, prefix_match_unit=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert resolve_kv_cache_block_sizes(lifecycle.cache, config) == (16, 16)
    assert not lifecycle.cache.needs_kv_cache_zeroing
    assert lifecycle.manager.take_kv_cache_block_copies() == ([], [])


def test_preemption_fences_reallocated_destination_before_load(lifecycle, monkeypatch):
    from vllm.v1.request import RequestStatus

    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata
    from tests.unit.test_connector_save_lifecycle import enqueue_save, process_next_save

    h = lifecycle
    completion = enqueue_save(h.worker, h.block.block_id)
    waiting = threading.Event()
    wait = completion.wait

    def observed_wait(*args, **kwargs):
        waiting.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(completion, "wait", observed_wait)
    h.scheduler._preempt_request(h.request, 0.0)
    assert h.request.status == RequestStatus.PREEMPTED
    # The scheduler reassigns the ID before worker dispatch. It is the public
    # preemption barrier, not a generation counter, that must fence new writes.
    assert h.manager.block_pool.get_new_blocks(1)[0] is h.block
    reached = threading.Event()
    h.worker.start_load_kv = lambda _: reached.set()
    output = SimpleNamespace(
        kv_connector_metadata=OrbitKVConnectorMetadata(preempted_req_ids={"request"}),
        has_sync_kv_loads=True,
    )
    thread = threading.Thread(target=h.active.pre_forward, args=(output,), daemon=True)
    thread.start()
    try:
        assert waiting.wait(5), "V2 did not reach the native completion wait"
        assert not reached.is_set()
        assert not completion.is_set()
        process_next_save(h.worker)
        thread.join(5)
        assert not thread.is_alive()
        assert completion.is_set() and reached.is_set()
    finally:
        if not completion.is_set():
            process_next_save(h.worker)
        thread.join(5)


@pytest.mark.parametrize("finish", ["abort", "finish"])
def test_finish_holds_allocator_blocks_until_native_save_receipt(lifecycle, finish):
    from vllm.v1.request import RequestStatus

    from tests.unit.test_connector_save_lifecycle import enqueue_save, process_next_save

    h = lifecycle
    enqueue_save(h.worker, h.block.block_id)
    h.adapter._pending_saves.add("request")
    status = RequestStatus.FINISHED_ABORTED if finish == "abort" else RequestStatus.FINISHED_STOPPED
    h.scheduler.finish_requests(["request"], status)
    assert h.block.ref_cnt == 1
    with pytest.raises(ValueError, match="free blocks"):
        h.manager.block_pool.get_new_blocks(1)
    output = h.active.post_forward({"request"})
    assert not output.finished_sending
    h.scheduler._update_from_kv_xfer_finished(output)
    assert h.block.ref_cnt == 1
    process_next_save(h.worker)
    output = h.active.post_forward(set())
    assert output.finished_sending == {"request"}
    h.scheduler._update_from_kv_xfer_finished(output)
    assert h.manager.block_pool.get_new_blocks(1)[0] is h.block


def test_abort_sync_load_keeps_deferred_blocks_until_step_receipt(lifecycle):
    from vllm.v1.request import RequestStatus

    h = lifecycle
    h.scheduler.defer_block_free = True
    h.scheduler.sched_step_seq = h.request.last_sched_seq = 1
    h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_ABORTED)
    h.scheduler._drain_deferred_frees()
    assert h.block.ref_cnt == 1
    with pytest.raises(ValueError, match="free blocks"):
        h.manager.block_pool.get_new_blocks(1)
    # V2 does not return its step output until post_forward waits for Restore.
    # The GPU event/output path itself remains an installed-GPU gate.
    h.worker._restore = SimpleNamespace(
        handle=object(),
        started=0.0,
        blocks=1,
        request_ids=["request"],
        control_ids=(),
    )
    h.worker._client.wait_restore.side_effect = TimeoutError("no physical receipt")
    with pytest.raises(RuntimeError, match="pages remain held"):
        h.active.post_forward({"request"})
    assert h.worker._restore is not None and h.block.ref_cnt == 1
    h.worker._client.wait_restore.side_effect = None
    h.worker._client.wait_restore.return_value = SimpleNamespace(success=True)
    h.active.post_forward({"request"})
    assert h.worker._restore is None
    h.scheduler.processed_step_seq = 1
    h.scheduler._drain_deferred_frees()
    assert h.manager.block_pool.get_new_blocks(1)[0] is h.block


@pytest.mark.gpu
@pytest.mark.vllm_v2(only=True)
@pytest.mark.skipif(
    os.environ.get("ORBITKV_FAULT_TESTS") != "1", reason="requires test-hooks Manager and wheel"
)
@pytest.mark.parametrize("case", ["preempt-save", "finish-save", "abort-load"])
def test_native_v2_page_reuse_bytes(allocation, fault_cache, monkeypatch, case):  # noqa: F811
    """GPU fixture bytes plus released lifecycle; model outputs have a separate gate."""
    import torch
    from vllm.v1.request import RequestStatus

    from orbitkv.vllm.config import ConnectorContext
    from orbitkv.vllm.metadata import LoadIntent, OrbitKVConnectorMetadata, SaveIntent
    from orbitkv.vllm.worker import WorkerAdapter
    from tests.integration.test_cache_faults import arm, publish, query, reached

    _, client, ctx, directory = fault_cache
    worker = WorkerAdapter(
        ConnectorContext(
            ctx.instance_id,
            ctx.namespace,
            16,
            1,
            1,
            0,
            0,
            client,
        )
    )
    # The existing native fixture owns the real CUDA tensor and registration.
    # Do not replace the client, CUDA events, native completions or engine classes.
    worker._registration = ctx.registration
    worker._registered_layers = ctx._layer_names
    worker._torch_device = ctx.device
    worker._layer_events = {name: torch.cuda.Event(external=True) for name in ctx._layer_names}
    for event in worker._layer_events.values():
        event.record(torch.cuda.current_stream(ctx.device))
    h = attach_worker(allocation, worker, monkeypatch)
    tensor = ctx.get_kv_cache()
    source = 0 if case == "abort-load" else h.block.block_id
    tensor[:, source].fill_(19)
    expected = tensor[:, source : source + 1].cpu().clone()
    output = SimpleNamespace(finished_req_ids=set(), has_sync_kv_loads=False)
    thread = None
    failures = []

    def pre_forward():
        try:
            h.active.pre_forward(output)
            if case == "preempt-save":
                tensor[:, h.block.block_id].fill_(83)
        except BaseException as error:
            failures.append(error)

    try:
        if case == "abort-load":
            assert publish(client, ctx, [b"v2-load"])[0]
            ready = query(client, ctx, [b"v2-load"], "v2-load")
            tensor[:, h.block.block_id].fill_(37)
            arm(directory, "local_restore_dma")
            output.kv_connector_metadata = OrbitKVConnectorMetadata(
                load_intents={
                    "request": LoadIntent(((h.block.block_id,),), (ready.lease,), 16),
                }
            )
            output.has_sync_kv_loads = True
            thread = threading.Thread(target=pre_forward, daemon=True)
            thread.start()
            reached(directory, "local_restore_dma")
            h.scheduler.defer_block_free = True
            h.scheduler.sched_step_seq = h.request.last_sched_seq = 1
            h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_ABORTED)
            h.scheduler._drain_deferred_frees()
            assert h.block.ref_cnt == 1
            assert worker._restore is not None
            assert not client.poll_restore(worker._restore.handle).done
            (directory / "local_restore_dma.pause").unlink()
            thread.join(30)
            assert not thread.is_alive()
            assert not failures, failures
            h.active.post_forward({"request"})
            assert worker._restore is None
            torch.cuda.synchronize()
            assert torch.equal(tensor[:, h.block.block_id : h.block.block_id + 1].cpu(), expected)
            h.scheduler.processed_step_seq = 1
            h.scheduler._drain_deferred_frees()
        else:
            arm(directory, "publish")
            output.kv_connector_metadata = OrbitKVConnectorMetadata(
                save_intents={
                    "request": SaveIntent(((h.block.block_id,),), (b"v2-save",)),
                }
            )
            h.active.no_forward(output)
            reached(directory, "publish")
            completion = worker._save_completion_events["request"]
            assert not completion.is_set()
            if case == "preempt-save":
                h.scheduler._preempt_request(h.request, 0.0)
                assert h.manager.block_pool.get_new_blocks(1)[0] is h.block
                output.kv_connector_metadata = OrbitKVConnectorMetadata(
                    preempted_req_ids={"request"}
                )
                waiting = threading.Event()
                wait = completion.wait

                def observed_wait(*args, **kwargs):
                    waiting.set()
                    return wait(*args, **kwargs)

                monkeypatch.setattr(completion, "wait", observed_wait)
                thread = threading.Thread(target=pre_forward, daemon=True)
                thread.start()
                assert waiting.wait(10), "V2 did not reach the native completion wait"
                assert thread.is_alive()
                assert torch.equal(
                    tensor[:, h.block.block_id : h.block.block_id + 1].cpu(), expected
                )
            else:
                h.adapter._pending_saves.add("request")
                h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_STOPPED)
                assert h.block.ref_cnt == 1
                assert not h.active.post_forward({"request"}).finished_sending
            (directory / "publish.pause").unlink()
            assert completion.wait(30)
            if thread is not None:
                thread.join(30)
                assert not thread.is_alive()
                assert not failures, failures
            else:
                h.scheduler._update_from_kv_xfer_finished(h.active.post_forward(set()))
        if case != "preempt-save":
            assert h.manager.block_pool.get_new_blocks(1)[0] is h.block
            tensor[:, h.block.block_id].fill_(83)
        torch.cuda.synchronize()
        assert torch.all(tensor[:, h.block.block_id] == 83)
        if case != "abort-load":
            ready = query(client, ctx, [b"v2-save"], "saved-after-reuse")
            assert ready.num_hit_blocks == 1
            handle = client.start_restore(
                ctx.instance_id,
                0,
                0,
                [ctx._layer_names],
                [(ready.lease, [[2]])],
                ready_stream=torch.cuda.current_stream(ctx.device).cuda_stream,
            )
            assert client.wait_restore(handle, timeout=30).success
            assert torch.equal(tensor[:, 2:3].cpu(), expected)
    finally:
        for pause in directory.glob("*.pause"):
            pause.unlink(missing_ok=True)
        if thread is not None:
            thread.join(30)
        worker.shutdown()
