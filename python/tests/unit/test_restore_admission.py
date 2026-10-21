"""Shared-prefix tail restores batch without changing independent-prefix admission."""

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


@pytest.mark.parametrize(
    "tokens,computed,hits,expected",
    [
        pytest.param(32, 0, 2, None, id="independent-hit"),
        pytest.param(32, 0, 0, None, id="independent-miss"),
        pytest.param(48, 16, 2, None, id="multi-block-suffix"),
        pytest.param(32, 16, 1, 15, id="tail-hit"),
        pytest.param(32, 16, 0, 0, id="tail-miss"),
        pytest.param(32, 16, None, None, id="tail-loading"),
        pytest.param(7, 0, 0, 0, id="short"),
        pytest.param(33, 32, 0, 0, id="hbm"),
    ],
)
def test_restore_admission_batches_only_native_prefix_tails(
    restoring, tokens, computed, hits, expected
):
    scheduler, _ = restoring
    waiting = request("waiting", tokens)
    waiting.block_hashes[: computed // 16] = request("restore").block_hashes[: computed // 16]
    scheduler._ctx.client.query_prefetch.return_value = (
        QueryLoading() if hits is None else QueryReady(hits, b"next" if hits else b"")
    )
    assert scheduler.get_num_new_matched_tokens(waiting, computed) == (expected, False)

    scheduled = [("restore", [1, 2], 31, 1)]
    if expected is not None:
        blocks = [1, 2, 3] if computed == 32 else [1, 3] if computed == 16 else [3]
        scheduler.update_state_after_alloc(
            waiting, SimpleNamespace(get_block_ids=lambda: (blocks,)), expected
        )
        scheduled.append(("waiting", blocks, computed + expected, 1))
    metadata = scheduler.build_connector_meta(step(*scheduled))
    assert set(metadata.load_intents) == ({"restore", "waiting"} if expected else {"restore"})
    assert metadata.load_intents["restore"].leases == (b"hold",)
    assert metadata.load_intents["restore"].block_ids_by_group == ((1, 2),)
    if expected:
        assert metadata.load_intents["waiting"].leases == (b"next",)
        assert metadata.load_intents["waiting"].block_ids_by_group == ((3,),)
    assert not scheduler._pending_load_intents
    if expected is None and hits is not None:
        calls = scheduler._ctx.client.query_prefetch.call_count
        assert scheduler.get_num_new_matched_tokens(waiting, computed) == (
            min(hits * 16, tokens - computed - 1),
            False,
        )
        assert scheduler._ctx.client.query_prefetch.call_count == calls


def test_allocation_failure_keeps_only_the_unallocated_query_for_the_next_step(restoring):
    scheduler, _ = restoring
    waiting = request("waiting")
    waiting.block_hashes[0] = request("restore").block_hashes[0]
    client = scheduler._ctx.client
    client.query_prefetch.return_value = QueryReady(1, b"next")
    assert scheduler.get_num_new_matched_tokens(waiting, 16) == (15, False)

    # A failed allocation has no update_state_after_alloc callback. The earlier
    # synchronous restore is already scheduled and must still reach the worker.
    metadata = scheduler.build_connector_meta(step(("restore", [1, 2], 31, 1)))
    assert set(metadata.load_intents) == {"restore"}
    assert scheduler._pending_query_probes["waiting"].leases == (b"next",)
    client.release.assert_not_called()

    assert scheduler.get_num_new_matched_tokens(waiting, 16) == (15, False)
    assert client.query_prefetch.call_count == 2
    scheduler.update_state_after_alloc(
        waiting, SimpleNamespace(get_block_ids=lambda: ([1, 3],)), 15
    )
    metadata = scheduler.build_connector_meta(step(("waiting", [1, 3], 31, 1)))
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
