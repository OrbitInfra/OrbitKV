"""Full/SWA contracts against installed vLLM; events and Manager replies are seams."""

import os
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.test_cache_faults import fault_cache  # noqa: F401
from tests.integration.test_vllm_connector_contract import released_config  # noqa: F401
from tests.integration.test_vllm_v2_contract import allocation  # noqa: F401

pytestmark = pytest.mark.integration
HMA = {"window": 33, "tokens": 64, "blocks": 16}


@pytest.fixture
def hybrid(allocation, monkeypatch):  # noqa: F811
    import torch

    from orbitkv.vllm.layout import CacheGroupLayout
    from tests.unit.test_connector_save_lifecycle import make_worker

    h = allocation
    h.request.block_hashes = [index.to_bytes(32, "little") for index in range(4)]
    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    h.worker = make_worker()
    h.worker._cache_groups = CacheGroupLayout.from_config(h.cache)
    h.worker._registered_layers = ["layer", "window"]
    h.worker._layer_to_group = {"layer": 0, "window": 1}
    h.worker._layer_events = {name: MagicMock() for name in ("layer", "window")}
    return h


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
@pytest.mark.parametrize(
    "full,window,materialized,expected",
    [
        ([0, 1, 2], [1, 2], True, 48),
        ([], [1, 2], True, 0),
        ([0, 1, 2], [], True, 0),
        ([0, 1, 2], [2, 3], True, 0),
        ([0, 1, 2], [1, 2], False, 0),
    ],
    ids=["atomic", "missing-full", "missing-window", "different-boundaries", "evicted"],
)
def test_atomic_group_hit(hybrid, full, window, materialized, expected):
    from orbitkv import QueryCandidates, QueryReady
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata

    h = hybrid
    client = h.adapter._ctx.client
    client.query_candidates.side_effect = [QueryCandidates(full), QueryCandidates(window)]
    client.read_recovery.side_effect = lambda *args: (
        QueryReady(3, b"full", [0, 1, 2])
        if args[-1] == 0
        else QueryReady(2, b"window", [1, 2])
        if materialized
        else QueryReady(0, b"")
    )
    assert h.adapter.get_num_new_matched_tokens(h.request, 0) == (expected, False)
    if not expected:
        assert not h.adapter._pending_load_intents
        if not materialized:
            client.release.assert_called_once_with(b"full")
        return
    h.adapter.update_state_after_alloc(h.request, h.manager.get_blocks("request"), expected)
    intent = h.adapter._pending_load_intents["request"]
    assert intent.recovery_hold.last_position == 2
    assert intent.block_ids_by_group[1][0] is None
    h.worker.start_load_kv(OrbitKVConnectorMetadata(load_intents={"request": intent}))
    # All groups enter the same native Restore, which owns their leases and grants.
    h.worker._client.start_restore.assert_called_once()
    assert h.worker._client.start_restore.call_args.args[3] == [["layer"], ["window"]]
    assert [lease for lease, _ in h.worker._client.start_restore.call_args.args[4]] == [
        b"full",
        b"window",
    ]
    h.worker._client.wait_restore.side_effect = TimeoutError("no physical receipt")
    with pytest.raises(RuntimeError, match="pages remain held"):
        h.worker.wait_for_save()
    assert h.worker._restore is not None
    client.release.assert_not_called()
    h.worker._client.wait_restore.side_effect = None
    h.worker._client.wait_restore.return_value = SimpleNamespace(success=True)
    h.worker.wait_for_save()
    assert h.worker._restore is None


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
@pytest.mark.parametrize("retire", ["window", "preempt", "abort"])
def test_save_pins_survive_engine_retirement_until_drain(hybrid, retire):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.request import RequestStatus

    from tests.unit.test_connector_save_lifecycle import process_next_save

    h = hybrid
    blocks = h.manager.get_blocks("request")
    ids = blocks.get_block_ids()
    saved = blocks.blocks[1][0]
    h.request.num_computed_tokens = 0
    h.adapter.update_state_after_alloc(h.request, blocks, 0)
    output = SchedulerOutput(
        scheduled_new_reqs=[NewRequestData.from_request(h.request, ids)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"request": 64},
        total_num_scheduled_tokens=64,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    metadata = h.adapter.build_connector_meta(output)
    assert not metadata.save_intents
    (job,) = metadata.boundary_save_intents
    assert metadata.boundary_save_intents[job].block_ids_by_group == tuple(map(tuple, ids))
    h.request.num_computed_tokens = 64
    h.worker.start_load_kv(metadata)
    h.worker.wait_for_save()
    assert saved.ref_cnt == 2
    if retire == "window":
        h.manager.remove_skipped_blocks("request", 64)
        assert h.manager.get_block_ids("request")[1][0] == 0
    elif retire == "preempt":
        h.scheduler._preempt_request(h.request, 0.0)
    else:
        h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_ABORTED)
    assert saved.ref_cnt == 1
    pool = h.manager.block_pool
    other = pool.get_new_blocks(pool.get_num_free_blocks())
    assert saved not in other
    with pytest.raises(ValueError, match="free blocks"):
        pool.get_new_blocks(1)
    assert h.worker.build_connector_worker_meta() is None
    assert job in h.adapter._pinned_boundary_jobs
    process_next_save(h.worker)
    h.adapter.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=h.worker.build_connector_worker_meta(),
            finished_sending=set(),
            finished_recving=set(),
        )
    )
    assert saved.ref_cnt == 0
    assert saved in pool.get_new_blocks(pool.get_num_free_blocks())
    assert not h.adapter._pinned_boundary_jobs


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
@pytest.mark.parametrize("finish", ["abort", "preempt"])
def test_load_retirement_waits_for_the_consuming_step(hybrid, monkeypatch, finish):
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.kv_connector_model_runner_mixin import (
        KVConnectorModelRunnerMixin,
    )

    from orbitkv.vllm.connector import OrbitKVConnector
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata

    h = hybrid
    h.scheduler.defer_block_free = True
    h.scheduler.sched_step_seq = h.request.last_sched_seq = 1
    # The load restores boundary 48; the consuming step optimistically advances
    # to 64. vLLM's window retirement subtracts these in-flight tokens.
    h.request.num_in_flight_tokens = 16
    blocks = h.manager.get_blocks("request").blocks
    destinations = [*blocks[0][:3], *blocks[1][1:3]]
    h.worker._restore = SimpleNamespace(
        handle=object(), started=0.0, blocks=5, request_ids=["request"], control_ids=()
    )
    connector = OrbitKVConnector.__new__(OrbitKVConnector)
    KVConnectorBase_V1.__init__(
        connector,
        SimpleNamespace(kv_transfer_config=KVTransferConfig(kv_role="kv_both")),
        KVConnectorRole.WORKER,
        h.cache,
    )
    connector._worker, connector._scheduler = h.worker, None
    connector._ctx = SimpleNamespace(query_control=False)
    # Restore is already submitted. Only the CUDA/Manager submission is a seam;
    # the official V1 finalizer and allocator retirement are exercised below.
    monkeypatch.setattr(connector, "start_load_kv", lambda _: None)
    module = "vllm.v1.worker.kv_connector_model_runner_mixin"
    monkeypatch.setattr(f"{module}.get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(f"{module}.get_forward_context", lambda: SimpleNamespace())
    output = SimpleNamespace(
        kv_connector_metadata=OrbitKVConnectorMetadata(),
        has_sync_kv_loads=True,
        finished_req_ids={"request"},
    )
    h.worker._client.wait_restore.side_effect = TimeoutError("no physical receipt")
    with (
        pytest.raises(RuntimeError, match="pages remain held"),
        KVConnectorModelRunnerMixin._get_kv_connector_output(output),
    ):
        if finish == "abort":
            h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_ABORTED)
        else:
            # schedule() refuses this victim while its consuming step runs.
            assert not h.scheduler._request_blocks_can_be_freed(h.request)
        h.scheduler._drain_deferred_frees()
    assert h.worker._restore is not None
    assert all(block.ref_cnt == 1 for block in destinations)
    pool = h.manager.block_pool
    assert not {block.block_id for block in destinations}.intersection(
        block.block_id for block in pool.get_new_blocks(pool.get_num_free_blocks())
    )
    h.worker._client.wait_restore.side_effect = None
    h.worker._client.wait_restore.return_value = SimpleNamespace(success=True)
    with KVConnectorModelRunnerMixin._get_kv_connector_output(output):
        pass
    assert h.worker._restore is None
    h.scheduler.processed_step_seq = 1
    h.request.num_in_flight_tokens = 0
    if finish == "preempt":
        assert h.scheduler._request_blocks_can_be_freed(h.request)
        h.scheduler._preempt_request(h.request, 0.0)
    h.scheduler._drain_deferred_frees()
    assert all(block.ref_cnt == 0 for block in destinations)


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
def test_allocated_boundary_must_equal_the_atomic_hit(hybrid):
    from orbitkv import QueryCandidates, QueryReady

    h = hybrid
    client = h.adapter._ctx.client
    client.query_candidates.side_effect = [QueryCandidates([0, 1, 2]), QueryCandidates([1, 2])]
    client.read_recovery.side_effect = [
        QueryReady(3, b"full", [0, 1, 2]),
        QueryReady(2, b"window", [1, 2]),
    ]
    assert h.adapter.get_num_new_matched_tokens(h.request, 0) == (48, False)
    with pytest.raises(RuntimeError, match="allocation changed the recovery boundary"):
        h.adapter.update_state_after_alloc(h.request, h.manager.get_blocks("request"), 32)
    assert {call.args[0] for call in client.release.call_args_list} == {b"full", b"window"}
    assert not h.adapter._pending_load_intents


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
def test_save_job_admission_is_bounded(hybrid):
    from orbitkv.vllm.metadata import SaveIntent

    h = hybrid
    intent = SaveIntent(((h.block.block_id,), ()), (b"hash",))
    capacity = len(h.manager.block_pool.blocks) - 1
    jobs = [h.adapter._pin_save_intent(intent) for _ in range(capacity)]
    with pytest.raises(RuntimeError, match="GPU block budget"):
        h.adapter._pin_save_intent(intent)
    assert h.block.ref_cnt == capacity + 1
    h.adapter._release_boundary_jobs(dict.fromkeys(jobs, 1))
    assert h.block.ref_cnt == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ("unknown", "unknown cache specs"),
        ("mla", "MLA"),
        ("recurrent", "recurrent/Mamba"),
        ("zeroing", "KV zeroing"),
        ("speculative", "speculative decoding"),
        ("draft", "draft state"),
        ("retained-speculation", "draft state"),
        ("v2", "V2 rejects multiple"),
        ("block-size", "identical logical block sizes"),
        ("window", "past token"),
        ("host", "host-resident"),
        ("producer", "consumer role"),
        ("pipeline", "pipeline-parallel group projections"),
    ],
)
def test_unproven_hma_profiles_fail_before_connecting(released_config, monkeypatch, change, reason):  # noqa: F811
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    spec = SlidingWindowSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16, sliding_window=33
    )
    cache.kv_cache_groups.append(KVCacheGroupSpec(["window"], spec))
    if change == "unknown":

        class UnknownSpec(FullAttentionSpec):
            pass

        cache.kv_cache_groups[1].kv_cache_spec = UnknownSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )
    elif change == "mla":
        cache.kv_cache_groups[0].kv_cache_spec = MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16
        )
    elif change == "recurrent":
        cache.kv_cache_groups[1].kv_cache_spec = MambaSpec(
            block_size=16, shapes=((2, 4),), dtypes=(torch.float32,), mamba_cache_mode="align"
        )
    elif change == "zeroing":
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, dtype=torch.float32)
        assert cache.needs_kv_cache_zeroing
    elif change == "speculative":
        config.speculative_config = object()
    elif change == "draft":
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, is_draft=True)
    elif change == "retained-speculation":
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, extra_retained_tokens=16)
    elif change == "v2":
        config.use_v2_model_runner = True
    elif change == "block-size":
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, block_size=32)
    elif change == "window":
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, sliding_window=1)
    elif change == "host":
        cache.kv_cache_groups[1].host_resident = True
    elif change == "pipeline":
        config.parallel_config.pipeline_parallel_size = 2
    else:
        config.kv_transfer_config.kv_role = "kv_producer"
    connect = MagicMock(side_effect=AssertionError("unproven profile opened a connection"))
    monkeypatch.setattr("orbitkv.vllm.connector.connect_cache", connect)
    with pytest.raises(RuntimeError, match=reason):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
    connect.assert_not_called()


def test_full_and_window_pass_startup_admission(released_config, monkeypatch):  # noqa: F811
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    config.cache_config = SimpleNamespace(
        block_size=16, prefix_match_unit=None, enable_prefix_caching=True
    )
    config.parallel_config.decode_context_parallel_size = 1
    cache.kv_cache_groups.append(
        KVCacheGroupSpec(
            ["window"],
            SlidingWindowSpec(
                block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16, sliding_window=33
            ),
        )
    )

    class Admitted(Exception):
        pass

    namespace = MagicMock(side_effect=Admitted)
    monkeypatch.setattr("orbitkv.vllm.connector.derive_namespace", namespace)
    with pytest.raises(Admitted):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
    assert namespace.call_args.kwargs["kv_cache_config"] is cache


@pytest.mark.parametrize("allocation", [HMA], indirect=True)
@pytest.mark.parametrize("change", ["order", "window", "block-size", "head-size", "layers"])
def test_resolved_group_identity_cannot_match_old_layout(hybrid, monkeypatch, change):
    from orbitkv.vllm.config import derive_namespace
    from tests.unit.test_derive_namespace import _make_vllm_config

    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "a" * 64)
    config = _make_vllm_config()
    cache = hybrid.cache
    original = derive_namespace(config, 1, kv_cache_config=cache)
    assert original == derive_namespace(config, 1, kv_cache_config=cache)
    if change == "order":
        cache.kv_cache_groups.reverse()
    elif change == "layers":
        cache.kv_cache_groups[1].layer_names = ["different.layer"]
    else:
        field = {"window": "sliding_window", "block-size": "block_size", "head-size": "head_size"}[
            change
        ]
        spec = cache.kv_cache_groups[1].kv_cache_spec
        cache.kv_cache_groups[1].kv_cache_spec = replace(spec, **{field: getattr(spec, field) * 2})
    assert derive_namespace(config, 1, kv_cache_config=cache) != original


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("ORBITKV_FAULT_TESTS") != "1", reason="requires test-hooks Manager and wheel"
)
@pytest.mark.parametrize("allocation", [HMA], indirect=True)
@pytest.mark.parametrize("case", ["window-save", "preempt-save", "abort-load", "preempt-load"])
def test_native_hma_page_reuse_bytes(allocation, fault_cache, case):  # noqa: F811
    """Real two-group GPU bytes and released allocator; no model-serving claim."""
    import torch
    from vllm.v1.request import RequestStatus

    from orbitkv import BlockHashes, QueryReady
    from orbitkv.vllm.config import ConnectorContext
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata, SaveIntent
    from orbitkv.vllm.scheduler import SchedulerAdapter
    from orbitkv.vllm.worker import WorkerAdapter
    from tests.integration.test_cache_faults import arm, reached, until

    h = allocation
    _, client, _, directory = fault_cache
    identity = f"hma-{directory.name}"
    client.start_session_watcher(identity, identity, 1, 1)
    ctx = ConnectorContext(identity, identity, 16, 1, 1, 0, 0, client)
    h.adapter.shutdown()
    h.adapter = SchedulerAdapter(ctx, kv_cache_config=h.cache)
    h.scheduler.connector._scheduler = h.adapter
    h.scheduler.connector.bind_kv_cache_manager(h.manager)
    worker = WorkerAdapter(ctx, kv_cache_config=h.cache)
    tensors = {
        name: torch.arange(16 * 2 * 16 * 2 * 64, device="cuda", dtype=torch.float32)
        .remainder(97)
        .add(index * 100)
        .to(torch.float16)
        .reshape(16, 2, 16, 2, 64)
        for index, name in enumerate(("layer", "window"))
    }
    worker.register_kv_caches(tensors)
    expected = {name: tensor.cpu().clone() for name, tensor in tensors.items()}
    hashes = [index.to_bytes(32, "little") for index in range(4)]
    h.request.block_hashes = hashes
    ids = h.manager.get_block_ids("request")
    saved = h.manager.block_pool.blocks[ids[1][0]]
    thread, failures = None, []

    def restore(metadata):
        try:
            worker.start_load_kv(metadata)
            worker.wait_for_all_layers()
            worker.wait_for_save()
        except BaseException as error:
            failures.append(error)

    try:
        if case.endswith("save"):
            intent = SaveIntent(tuple(map(tuple, ids)), tuple(hashes))
            job = h.adapter._pin_save_intent(intent)
            arm(directory, "publish")
            worker.start_load_kv(OrbitKVConnectorMetadata(boundary_save_intents={job: intent}))
            worker.wait_for_save()
            reached(directory, "publish")
            if case == "window-save":
                h.manager.remove_skipped_blocks("request", 64)
            else:
                h.scheduler._preempt_request(h.request, 0.0)
            assert saved.ref_cnt == 1
            assert worker.build_connector_worker_meta() is None
            assert torch.equal(
                tensors["window"][saved.block_id].cpu(), expected["window"][saved.block_id]
            )
            (directory / "publish.pause").unlink()
            metadata = until(worker.build_connector_worker_meta, timeout=30)
            h.adapter.update_connector_output(
                SimpleNamespace(
                    kv_connector_worker_meta=metadata,
                    finished_sending=set(),
                    finished_recving=set(),
                )
            )
            assert saved.ref_cnt == 0
        else:
            saves = [
                ("layer", ids[0][:3], hashes[:3]),
                ("window", ids[1][1:3], hashes[1:3]),
            ]
            success, message = client.save(worker._registration, saves)
            assert success, message

            def hit():
                result = h.adapter.get_num_new_matched_tokens(h.request, 0)
                return result if result[0] is not None else None

            assert until(hit) == (48, False)
            h.adapter.update_state_after_alloc(h.request, h.manager.get_blocks("request"), 48)
            metadata = OrbitKVConnectorMetadata(load_intents=h.adapter._pending_load_intents)
            destinations = {"layer": ids[0][:3], "window": ids[1][1:3]}
            for name, targets in destinations.items():
                tensors[name][targets] = -1
            torch.cuda.synchronize()
            arm(directory, "local_restore_dma")
            thread = threading.Thread(target=restore, args=(metadata,), daemon=True)
            thread.start()
            reached(directory, "local_restore_dma")
            h.scheduler.defer_block_free = True
            h.scheduler.sched_step_seq = h.request.last_sched_seq = 1
            h.request.num_in_flight_tokens = 16
            if case == "abort-load":
                h.scheduler.finish_requests(["request"], RequestStatus.FINISHED_ABORTED)
                h.scheduler._drain_deferred_frees()
            else:
                assert not h.scheduler._request_blocks_can_be_freed(h.request)
            assert worker._restore is not None
            assert not client.poll_restore(worker._restore.handle).done
            assert all(
                h.manager.block_pool.blocks[bid].ref_cnt == 1
                for targets in destinations.values()
                for bid in targets
            )
            (directory / "local_restore_dma.pause").unlink()
            thread.join(30)
            assert not thread.is_alive() and not failures, failures
            assert worker._restore is None
            torch.cuda.synchronize()
            for name, targets in destinations.items():
                assert torch.equal(tensors[name][targets].cpu(), expected[name][targets])
            h.scheduler.processed_step_seq = 1
            h.request.num_in_flight_tokens = 0
            if case == "preempt-load":
                h.scheduler._preempt_request(h.request, 0.0)
            h.scheduler._drain_deferred_frees()
        pool = h.manager.block_pool
        assert saved in pool.get_new_blocks(pool.get_num_free_blocks())
        tensors["window"][saved.block_id].fill_(83)
        torch.cuda.synchronize()
        assert torch.all(tensors["window"][saved.block_id] == 83)
        if case.endswith("save"):
            loads = []
            for group in range(len(tensors)):

                def ready_group(group=group):
                    result = client.query_prefetch(
                        identity, BlockHashes(hashes), f"saved-{group}", group_id=group
                    )
                    return result if isinstance(result, QueryReady) else None

                ready = until(ready_group)
                assert ready.num_hit_blocks == 4
                targets = [[None] * 4, [None] * 4]
                targets[group] = [12, 13, 14, 15]
                loads.append((ready.lease, targets))
            handle = client.start_restore(
                identity,
                0,
                0,
                [["layer"], ["window"]],
                loads,
                ready_stream=torch.cuda.current_stream().cuda_stream,
            )
            assert client.wait_restore(handle, timeout=30).success
            for group, name in enumerate(tensors):
                assert torch.equal(tensors[name][12:16].cpu(), expected[name][ids[group]])
    finally:
        for pause in directory.glob("*.pause"):
            pause.unlink(missing_ok=True)
        if thread is not None:
            thread.join(30)
        worker.shutdown()
