"""Cache hits share a consuming forward without blocking unrelated requests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from orbitkv import BlockHashes, QueryLoading, QueryReady  # noqa: E402
from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.scheduler import SchedulerAdapter  # noqa: E402


def request(req_id, tokens=32):
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=tokens,
        num_prompt_tokens=tokens,
        block_hashes=[f"{req_id}-{i}".encode() for i in range(tokens // 16)],
    )


@pytest.mark.parametrize("preparation", [False, True], ids=["warming", "owned"])
def test_enqueue_warms_only_legal_missing_prefix_without_creating_a_load(monkeypatch, preparation):
    setting = "ORBITKV_PREPARE_REQUESTS" if preparation else "ORBITKV_QUEUE_WARMUP"
    monkeypatch.setenv(setting, "1")
    client = MagicMock()
    scheduler = SchedulerAdapter(
        ConnectorContext(
            instance_id="warm",
            namespace="warm",
            block_size=16,
            tp_size=1,
            world_size=1,
            tp_rank=0,
            device_id=0,
            client=client,
        )
    )
    pool = MagicMock()
    pool.get_cached_block.side_effect = [[object()], None]
    scheduler.bind_gpu_block_pool(pool)
    req = request("queued", 64)
    scheduler.on_new_request(req)
    batch = BlockHashes([b"queued-1", b"queued-2", b"queued-3"])
    if preparation:
        client.prepare_prefix.assert_called_once_with("warm", batch, "queued")
        assert scheduler._recovery is None
    else:
        client.warm_prefix.assert_called_once_with("warm", batch, "queued")
    submit = client.prepare_prefix if preparation else client.warm_prefix
    client.query_prefetch.assert_not_called()
    assert not scheduler._pending_load_intents
    assert not scheduler._pending_query_probes
    assert scheduler.request_finished(req, ()) == (False, None)
    client.cancel_query.assert_called_once_with("warm", "queued")
    assert not scheduler._queued_at
    pool.reset_mock()
    client.reset_mock()
    monkeypatch.delenv(setting)
    scheduler.on_new_request(req)
    pool.get_cached_block.assert_not_called()
    submit.assert_not_called()

    monkeypatch.setenv(setting, "1")
    pool.get_cached_block.side_effect = None
    pool.get_cached_block.return_value = None
    submit.side_effect = RuntimeError("manager unavailable")
    scheduler.on_new_request(req)
    submit.assert_called_once()
    assert req.request_id in scheduler._queued_at
    assert not scheduler._pending_load_intents
    if preparation:
        client.reset_mock()
        scheduler._queued_at.update({f"ahead-{i}": 0 for i in range(4)})
        scheduler.on_new_request(request("distant", 64))
        submit.assert_not_called()


@pytest.mark.parametrize("hits", [1, 4], ids=["partial-prefix", "full-prefix"])
def test_owned_prefix_preparation_keeps_ordinary_lookup_and_page_lifetime(monkeypatch, hits):
    monkeypatch.setenv("ORBITKV_PREPARE_REQUESTS", "1")
    client = MagicMock()
    client.prepare_prefix.return_value = True
    client.query_prefetch.return_value = QueryReady(hits, b"prepared", [])
    scheduler = SchedulerAdapter(
        ConnectorContext(
            instance_id="prepared",
            namespace="prepared",
            block_size=16,
            tp_size=1,
            world_size=1,
            tp_rank=0,
            device_id=0,
            client=client,
        )
    )
    req = request("queued", 64)
    scheduler.on_new_request(req)
    batch = BlockHashes(req.block_hashes)
    client.prepare_prefix.assert_called_once_with("prepared", batch, "queued")
    assert scheduler._recovery is None
    assert not scheduler._pending_load_intents
    expected = (min(hits * 16, 63), False)
    assert scheduler.get_num_new_matched_tokens(req, 0) == expected
    assert scheduler.get_num_new_matched_tokens(req, 0) == expected
    client.query_prefetch.assert_called_once_with(
        "prepared",
        batch,
        req_id="queued",
        wait_for_full_prefix=False,
    )
    client.prepare_recovery.assert_not_called()
    client.read_recovery.assert_not_called()
    client.release.assert_not_called()
    scheduler.request_finished(req, ())
    client.release.assert_called_once_with(b"prepared")
    scheduler.shutdown()


def step(*requests):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=rid, block_ids=(blocks,), num_computed_tokens=computed)
            for rid, blocks, computed, _ in requests
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        num_scheduled_tokens={rid: tokens for rid, _, _, tokens in requests},
        preempted_req_ids=set(),
    )


@pytest.fixture
def restoring():
    scheduler = SchedulerAdapter(
        ConnectorContext(
            instance_id="test",
            namespace="test",
            block_size=16,
            tp_size=1,
            world_size=1,
            tp_rank=0,
            device_id=0,
            client=MagicMock(),
        )
    )
    scheduler._ctx.client.query_prefetch = MagicMock(return_value=QueryReady(2, b"hold"))
    restoring = request("restore")
    assert scheduler.get_num_new_matched_tokens(restoring, 0) == (31, False)
    blocks = SimpleNamespace(
        blocks=([SimpleNamespace(block_hash=None) for _ in range(2)],),
        get_block_ids=lambda: ([1, 2],),
    )
    scheduler.update_state_after_alloc(restoring, blocks, 31)
    yield scheduler, restoring
    scheduler.shutdown()


@pytest.mark.parametrize("kind", ["hit", "miss", "short", "hbm", "loading"])
def test_restore_does_not_defer_unrelated_requests_in_the_same_step(restoring, kind):
    scheduler, _ = restoring
    waiting = request("waiting", {"short": 7, "hbm": 33}.get(kind, 32))
    computed = 32 if kind == "hbm" else 0
    ready = QueryReady(2, b"next") if kind == "hit" else QueryReady(0, b"")
    scheduler._ctx.client.query_prefetch.return_value = (
        QueryLoading() if kind == "loading" else ready
    )
    hit_tokens = 31 if kind == "hit" else None if kind == "loading" else 0
    assert scheduler.get_num_new_matched_tokens(waiting, computed) == (hit_tokens, False)

    scheduled = [("restore", [1, 2], 31, 1)]
    if hit_tokens is not None:
        blocks = [1, 2, 3] if kind == "hbm" else [3, 4]
        scheduler.update_state_after_alloc(
            waiting, SimpleNamespace(get_block_ids=lambda: (blocks,)), hit_tokens
        )
        scheduled.append(("waiting", blocks, computed + hit_tokens, 1))
    metadata = scheduler.build_connector_meta(step(*scheduled))
    assert set(metadata.load_intents) == ({"restore", "waiting"} if kind == "hit" else {"restore"})
    assert metadata.load_intents["restore"].leases == (b"hold",)
    assert metadata.load_intents["restore"].block_ids_by_group == ((1, 2),)
    if kind == "hit":
        assert metadata.load_intents["waiting"].leases == (b"next",)
        assert metadata.load_intents["waiting"].block_ids_by_group == ((3, 4),)
    assert not scheduler._pending_load_intents


def test_allocation_failure_keeps_only_the_unallocated_query_for_the_next_step(restoring):
    scheduler, _ = restoring
    waiting = request("waiting")
    client = scheduler._ctx.client
    client.query_prefetch.return_value = QueryReady(2, b"next")
    assert scheduler.get_num_new_matched_tokens(waiting, 0) == (31, False)

    # A failed allocation has no update_state_after_alloc callback. The earlier
    # synchronous restore is already scheduled and must still reach the worker.
    metadata = scheduler.build_connector_meta(step(("restore", [1, 2], 31, 1)))
    assert set(metadata.load_intents) == {"restore"}
    assert scheduler._pending_query_probes["waiting"].leases == (b"next",)
    client.release.assert_not_called()

    assert scheduler.get_num_new_matched_tokens(waiting, 0) == (31, False)
    assert client.query_prefetch.call_count == 2
    scheduler.update_state_after_alloc(
        waiting, SimpleNamespace(get_block_ids=lambda: ([3, 4],)), 31
    )
    metadata = scheduler.build_connector_meta(step(("waiting", [3, 4], 31, 1)))
    assert set(metadata.load_intents) == {"waiting"}
    assert not scheduler._pending_query_probes


@pytest.mark.parametrize("pending_save", [False, True])
def test_request_finish_preserves_the_submitted_restore_and_pending_save(restoring, pending_save):
    scheduler, restored = restoring
    metadata = scheduler.build_connector_meta(step(("restore", [1, 2], 31, 1)))
    if pending_save:
        scheduler._pending_saves.add(restored.request_id)
    assert scheduler.request_finished(restored, ([1, 2],)) == (pending_save, None)
    assert scheduler.get_num_new_matched_tokens(request("short", 7), 0) == (0, False)
    assert metadata.load_intents["restore"].leases == (b"hold",)
    scheduler._ctx.client.release.assert_not_called()
    assert (restored.request_id in scheduler._held_requests) == pending_save
