"""HMA recurrent states are saved from vLLM's per-step boundary hand-offs.

A hybrid hit resumes only at a recurrent checkpoint. vLLM commits an
align-mode boundary block whenever a scheduler step ends on a block boundary
and hands it to the connector as ``(group_id, block_id, boundary_tokens)``
through ``SchedulerOutput.kv_connector_block_state.boundary_state_offloads``.

The connector must:

* file every hand-off under the request hash ending at that boundary, in the
  step it is offered (not once, at request end);
* pin the block in the GPU block pool until every worker reports the save,
  because align-mode tables free superseded state blocks one step later;
* never save recurrent blocks positionally from its own table mirror;
* tell vLLM where an attention prefix runs past the last usable checkpoint,
  so the junction state gets committed and the next sharer resumes there.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.metadata import (  # noqa: E402
    OrbitKVWorkerMetadata,
    RecoveryLoadHold,
    SaveIntent,
)
from orbitkv.vllm.scheduler import SchedulerAdapter  # noqa: E402
from orbitkv.vllm.tp_shards import ShardedQueryReady  # noqa: E402

VBS = 16


def _hash(i: int) -> bytes:
    return f"hash-{i}".encode()


class _FakePool:
    def __init__(self, num_blocks: int = 64):
        self.blocks = [SimpleNamespace(block_id=i, ref_cnt=0) for i in range(num_blocks)]
        self.touched: list[int] = []
        self.freed: list[int] = []

    def touch(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt += 1
            self.touched.append(block.block_id)

    def free_blocks(self, blocks) -> None:
        for block in blocks:
            block.ref_cnt -= 1
            self.freed.append(block.block_id)


def _make_scheduler(world_size: int = 1) -> tuple[SchedulerAdapter, _FakePool]:
    ctx = ConnectorContext(
        instance_id="i",
        namespace="n",
        block_size=VBS,
        tp_size=1,
        world_size=world_size,
        tp_rank=0,
        device_id=0,
        client=MagicMock(),
    )
    scheduler = SchedulerAdapter(ctx)
    scheduler._cache_groups = SimpleNamespace(
        group_count=2,
        hash_group_index=0,
        has_recurrent_state=True,
        window_group_indices=frozenset(),
        recurrent_group_indices=frozenset({1}),
    )
    pool = _FakePool()
    scheduler._gpu_block_pool = pool
    return scheduler, pool


def _register_request(scheduler: SchedulerAdapter, req_id: str, num_hashes: int) -> None:
    request = SimpleNamespace(
        request_id=req_id,
        block_hashes=[_hash(i) for i in range(num_hashes)],
    )
    scheduler._requests[req_id] = request
    scheduler._block_hashes[req_id] = tuple(request.block_hashes)


def _scheduler_output(offloads, *, with_block_state: bool = True) -> SimpleNamespace:
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={},
        preempted_req_ids=set(),
    )
    if with_block_state:
        output.kv_connector_block_state = SimpleNamespace(
            block_ids={},
            boundary_state_offloads=offloads,
        )
    return output


def test_boundary_offloads_become_pinned_recurrent_saves():
    scheduler, pool = _make_scheduler()
    _register_request(scheduler, "r1", 4)

    metadata = scheduler.build_connector_meta(
        _scheduler_output({"r1": [(1, 21, 1 * VBS), (1, 22, 3 * VBS)]})
    )

    # Recurrent rows carry the handed-off block; attention rows are null so
    # the worker saves only the recurrent layers. Keys are the hashes ending
    # at each boundary.
    assert metadata.boundary_save_intents == {
        0: SaveIntent(
            block_ids_by_group=((0, 0), (21, 22)),
            block_hashes=(_hash(0), _hash(2)),
        )
    }
    assert metadata.save_intents == {}
    assert pool.touched == [21, 22]
    assert scheduler.has_pending_push_work()

    scheduler.update_connector_output(
        SimpleNamespace(
            finished_sending=None,
            kv_connector_worker_meta=OrbitKVWorkerMetadata(completed_boundary_jobs={0: 1}),
        )
    )

    # Released newest-first, and only after the worker reported the job.
    assert pool.freed == [22, 21]
    assert all(block.ref_cnt == 0 for block in pool.blocks)
    assert not scheduler.has_pending_push_work()


def test_boundary_job_is_released_only_after_every_worker_reports():
    scheduler, pool = _make_scheduler(world_size=2)
    _register_request(scheduler, "r1", 2)
    scheduler.build_connector_meta(_scheduler_output({"r1": [(1, 21, VBS)]}))

    scheduler.update_connector_output(
        SimpleNamespace(
            finished_sending=None,
            kv_connector_worker_meta=OrbitKVWorkerMetadata(completed_boundary_jobs={0: 1}),
        )
    )
    assert pool.freed == []
    assert scheduler.has_pending_push_work()

    scheduler.update_connector_output(
        SimpleNamespace(
            finished_sending=None,
            kv_connector_worker_meta=OrbitKVWorkerMetadata(completed_boundary_jobs={0: 1}),
        )
    )
    assert pool.freed == [21]
    assert not scheduler.has_pending_push_work()


def test_window_save_outlives_request_and_waits_for_every_worker():
    scheduler, pool = _make_scheduler(world_size=2)
    scheduler._cache_groups.has_recurrent_state = False
    scheduler._cache_groups.recurrent_group_indices = frozenset()
    scheduler._cache_groups.window_group_indices = frozenset({1})
    _register_request(scheduler, "window", 2)
    scheduler._requests["window"].num_computed_tokens = 0
    scheduler._allocated_blocks["window"] = [[1, 2], [0, 21]]
    scheduler._scheduled_tokens["window"] = 0
    scheduler._block_index_offsets["window"] = 0
    scheduler._next_stored_block_idx["window"] = 0
    output = _scheduler_output({})
    output.scheduled_cached_reqs.req_ids = ["window"]
    output.scheduled_cached_reqs.new_block_ids = [None]
    output.scheduled_cached_reqs.num_computed_tokens = [0]
    output.num_scheduled_tokens = {"window": 2 * VBS}
    metadata = scheduler.build_connector_meta(output)
    assert not metadata.save_intents
    assert metadata.boundary_save_intents[0].block_ids_by_group == ((1, 2), (0, 21))
    assert pool.touched == [1, 2, 21]
    scheduler._cleanup_request("window")
    scheduler._release_boundary_jobs({0: 1})
    assert not pool.freed
    scheduler._release_boundary_jobs({0: 1})
    assert pool.freed == [21, 2, 1]
    assert not scheduler.has_pending_push_work()


def test_worker_meta_aggregates_per_job_counts():
    meta = OrbitKVWorkerMetadata(completed_boundary_jobs={0: 1, 1: 1})
    meta.aggregate(OrbitKVWorkerMetadata(completed_boundary_jobs={1: 1, 2: 1}))

    assert meta.completed_boundary_jobs == {0: 1, 1: 2, 2: 1}


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        pytest.param([(0, 11, VBS)], "attention group", id="attention_group"),
        pytest.param([(1, 0, VBS)], "vLLM null block", id="null_block"),
        pytest.param([(1, 23, VBS + 8)], "sub-block partial tail", id="sub_block"),
        pytest.param([(1, 24, 10 * VBS)], "boundary past the hashed prefix", id="past_hashes"),
    ],
)
def test_unusable_boundary_offloads_are_ignored(entries, reason: str):
    scheduler, pool = _make_scheduler()
    _register_request(scheduler, "r1", 4)

    metadata = scheduler.build_connector_meta(_scheduler_output({"r1": entries}))

    assert metadata.boundary_save_intents == {}, reason
    assert pool.touched == [], reason
    assert not scheduler.has_pending_push_work()


def test_offload_for_unknown_request_is_ignored():
    scheduler, pool = _make_scheduler()

    metadata = scheduler.build_connector_meta(_scheduler_output({"gone": [(1, 21, VBS)]}))

    assert metadata.boundary_save_intents == {}
    assert pool.touched == []


def test_boundary_is_saved_once_per_request():
    scheduler, pool = _make_scheduler()
    _register_request(scheduler, "r1", 4)

    first = scheduler.build_connector_meta(_scheduler_output({"r1": [(1, 21, VBS)]}))
    second = scheduler.build_connector_meta(
        _scheduler_output({"r1": [(1, 21, VBS), (1, 22, 2 * VBS)]})
    )

    assert list(first.boundary_save_intents) == [0]
    assert second.boundary_save_intents == {
        1: SaveIntent(block_ids_by_group=((0,), (22,)), block_hashes=(_hash(1),))
    }
    assert pool.touched == [21, 22]

    # Request cleanup drops the dedup state with the rest of its bookkeeping.
    scheduler._cleanup_request("r1")
    assert "r1" not in scheduler._saved_boundaries


def test_hma_requires_boundary_state_hand_off_api():
    scheduler, _ = _make_scheduler()
    _register_request(scheduler, "r1", 2)

    with pytest.raises(RuntimeError, match="kv_connector_block_state"):
        scheduler.build_connector_meta(_scheduler_output({}, with_block_state=False))


@pytest.mark.parametrize("use_v2", [True, False])
def test_hma_rejects_incompatible_vllm_before_opening_cache_connections(monkeypatch, use_v2):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    import orbitkv.vllm as vllm

    attention = object.__new__(FullAttentionSpec)
    object.__setattr__(attention, "block_size", VBS)
    recurrent = object.__new__(MambaSpec)
    object.__setattr__(recurrent, "block_size", VBS)
    object.__setattr__(recurrent, "mamba_cache_mode", "align")
    groups = (
        SimpleNamespace(layer_names=("attention",), kv_cache_spec=attention),
        SimpleNamespace(layer_names=("recurrent",), kv_cache_spec=recurrent),
    )
    output_module = ModuleType("vllm.v1.core.sched.output")
    output_module.SchedulerOutput = type("SchedulerOutput", (), {})
    for name in ("vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, output_module.__name__, output_module)
    connect_cache = MagicMock()
    monkeypatch.setattr("orbitkv.vllm.connector.connect_cache", connect_cache)
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        kv_transfer_config=SimpleNamespace(engine_id="test"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
    )

    with pytest.raises(RuntimeError, match="recurrent/Mamba"):
        vllm.OrbitKVConnector(
            config,
            vllm.KVConnectorRole.SCHEDULER,
            SimpleNamespace(kv_cache_groups=groups),
        )

    connect_cache.assert_not_called()


def test_mid_request_attention_save_never_touches_recurrent_rows():
    """Attention pages keep the dense cadence; recurrent rows are always null.

    The recurrent mirror is shorter than the attention table here (align mode
    reports only the state block it allocated, not the nulls it extended the
    table with) and must neither stall the attention save nor be read.
    """
    scheduler, _ = _make_scheduler()
    _register_request(scheduler, "r1", 3)
    scheduler._allocated_blocks["r1"] = [[11, 12, 13], [21]]
    scheduler._block_index_offsets["r1"] = 0
    scheduler._next_stored_block_idx["r1"] = 0
    scheduler._scheduled_tokens["r1"] = 3 * VBS

    intent = scheduler._consume_full_block_saves("r1")

    assert intent == SaveIntent(
        block_ids_by_group=((11, 12, 13), (0, 0, 0)),
        block_hashes=(_hash(0), _hash(1), _hash(2)),
    )
    assert scheduler._next_stored_block_idx["r1"] == 3


def _request(num_tokens: int, num_hashes: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r1",
        num_tokens=num_tokens,
        block_hashes=[_hash(i) for i in range(num_hashes)],
        shared_prefix_boundary=0,
    )


def test_attention_prefix_without_checkpoint_hints_the_junction():
    """The first sharer of a prefix recomputes it but commits its end state."""
    scheduler, _ = _make_scheduler()
    scheduler._query_recovery = MagicMock(
        return_value=ShardedQueryReady(0, (b"",), attention_hit_blocks=8)
    )
    request = _request(num_tokens=200, num_hashes=12)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.shared_prefix_boundary == 8 * VBS


def test_checkpoint_short_of_attention_prefix_hints_the_junction():
    scheduler, _ = _make_scheduler()
    scheduler._query_recovery = MagicMock(
        return_value=ShardedQueryReady(
            3,
            (b"lease",),
            recovery_hold=RecoveryLoadHold(
                leases=((b"membership",),),
                hit_positions=(((2,),),),
                last_position=2,
            ),
            boundary=3 * VBS,
            attention_hit_blocks=8,
        )
    )
    request = _request(num_tokens=200, num_hashes=12)

    hit_tokens, load_async = scheduler.get_num_new_matched_tokens(request, 0)

    assert (hit_tokens, load_async) == (3 * VBS, False)
    assert request.shared_prefix_boundary == 8 * VBS


@pytest.mark.parametrize(
    ("attention_hit_blocks", "hit_tokens", "num_tokens", "reason"),
    [
        pytest.param(8, 8 * VBS, 200, "checkpoint already at the prefix end", id="resumable"),
        pytest.param(0, 0, 200, "no attention prefix", id="no_prefix"),
        pytest.param(12, 0, 12 * VBS, "junction is the request end", id="at_request_end"),
    ],
)
def test_junction_hint_is_skipped_when_nothing_to_commit(
    attention_hit_blocks: int, hit_tokens: int, num_tokens: int, reason: str
):
    scheduler, _ = _make_scheduler()
    request = _request(num_tokens=num_tokens, num_hashes=12)

    scheduler._hint_shared_prefix_boundary(request, 0, attention_hit_blocks, hit_tokens)

    assert request.shared_prefix_boundary == 0, reason


def test_junction_hint_is_hma_only():
    scheduler, _ = _make_scheduler()
    scheduler._cache_groups = SimpleNamespace(
        group_count=1,
        hash_group_index=0,
        has_recurrent_state=False,
        window_group_indices=frozenset(),
        recurrent_group_indices=frozenset(),
    )
    request = _request(num_tokens=200, num_hashes=12)

    scheduler._hint_shared_prefix_boundary(request, 0, 8, 0)

    assert request.shared_prefix_boundary == 0


def test_load_targets_cover_only_the_selected_checkpoint_prefix():
    """The scheduler clamps before reading; no surplus attention/state lease exists."""
    scheduler, _ = _make_scheduler()
    scheduler._cache_groups.storage_group_ids = (0, 1)
    scheduler._recovery = MagicMock()
    scheduler._recovery.required_ranges.return_value = [(0, 0, 64), (1, 48, 64)]
    scheduler._query_recovery = MagicMock(
        return_value=ShardedQueryReady(
            4,
            (b"lease",),
            recovery_hold=RecoveryLoadHold(
                leases=((b"membership",),),
                hit_positions=(((3,),),),
                last_position=3,
            ),
            boundary=4 * VBS,
            attention_hit_blocks=5,
        )
    )
    request = _request(num_tokens=5 * VBS, num_hashes=5)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (4 * VBS, False)

    blocks = SimpleNamespace(
        get_block_ids=lambda: ([10, 11, 12, 13, 14], [20, 21, 22, 23, 24]),
        blocks=[[SimpleNamespace(block_hash=None)] * 5] * 2,
    )
    scheduler.update_state_after_alloc(request, blocks, 4 * VBS)

    intent = scheduler._pending_load_intents["r1"]
    assert intent.num_tokens == 4 * VBS
    assert intent.block_ids_by_group == ((10, 11, 12, 13), (None, None, None, 23))
    assert intent.recovery_hold.last_position == 3


def test_boundaries_inside_the_loaded_prefix_are_not_saved():
    """vLLM offers every block of an externally loaded prefix after the load,
    but only the checkpoint block was written; saving the rest would file
    stale state under valid hashes."""
    scheduler, pool = _make_scheduler()
    _register_request(scheduler, "r1", 6)
    scheduler._external_matched_blocks["r1"] = 3  # blocks 0..2 came from the store

    metadata = scheduler.build_connector_meta(
        _scheduler_output(
            {"r1": [(1, 21, 1 * VBS), (1, 22, 2 * VBS), (1, 23, 3 * VBS), (1, 24, 4 * VBS)]}
        )
    )

    assert metadata.boundary_save_intents == {
        0: SaveIntent(block_ids_by_group=((0,), (24,)), block_hashes=(_hash(3),))
    }
    assert pool.touched == [24]
