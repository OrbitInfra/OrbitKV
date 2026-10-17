from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorTransferResults,
)

from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.metadata import OrbitKVConnectorMetadata, SaveIntent  # noqa: E402
from orbitkv.vllm.worker import WorkerAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def producer_events():
    with patch("torch.cuda.Event"), patch("torch.cuda.current_stream"):
        yield


def make_worker() -> WorkerAdapter:
    context = ConnectorContext(
        instance_id="test",
        namespace="test",
        block_size=16,
        tp_size=1,
        world_size=1,
        tp_rank=0,
        device_id=0,
        client=MagicMock(),
    )
    with patch("orbitkv.vllm.worker.threading.Thread.start"):
        worker = WorkerAdapter(
            context,
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(get_head_size=lambda: 128), additional_config={}
            ),
        )

    worker._registration = object()
    return worker


def enqueue_save(
    worker: WorkerAdapter,
    block_id: int = 1,
    block_hash: bytes = b"hash",
) -> threading.Event:
    worker._current_metadata = OrbitKVConnectorMetadata(
        save_intents={
            "request": SaveIntent(
                block_ids_by_group=((block_id,),),
                block_hashes=(block_hash,),
            )
        }
    )
    worker.wait_for_save()
    return worker._save_completion_events["request"]


def complete_next_save(worker: WorkerAdapter) -> None:
    task = worker._save_queue.get_nowait()
    worker._complete_save_requests(task.request_ids)


def process_next_save(worker: WorkerAdapter) -> None:
    task = worker._save_queue.get_nowait()
    worker._process_save_batch([task])


@pytest.mark.parametrize("alive", [True, False, None])
def test_publish_progress_never_completes_held_engine_pages(alive):
    worker = make_worker()
    completion = enqueue_save(worker)
    worker._client.publish_progress.return_value = SimpleNamespace(
        active=1,
        retained=1,
        overdue=True,
        active_interval_seconds=61.0,
        manager_alive=alive,
    )
    assert not worker.get_transfer_results({"request"}).finished_sending
    assert not completion.is_set()
    worker._client.publish_progress.side_effect = RuntimeError("observation contention")
    assert not worker.get_transfer_results({"request"}).finished_sending
    assert not completion.is_set()


def test_save_waits_for_each_producer_before_publishing():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    events = [MagicMock(), MagicMock()]
    stream = object()
    calls = []
    with (
        patch("torch.cuda.Event", side_effect=events),
        patch("torch.cuda.current_stream", return_value=stream),
    ):
        completion = enqueue_save(worker, 1, b"one")
        enqueue_save(worker, 2, b"two")
    for index, event in enumerate(events):
        event.record.assert_called_once_with(stream)
        event.synchronize.side_effect = lambda i=index: calls.append(i)

    def save(*_args):
        assert calls == [0, 1], "Publish must follow both producing steps"
        assert not worker._save_completion_events["request"].is_set()
        calls.append("publish")
        return True, ""

    worker._client.save.side_effect = save
    with patch("torch.cuda.synchronize", side_effect=AssertionError("device-wide wait")):
        worker._process_save_batch([worker._save_queue.get_nowait() for _ in events])
    assert completion.is_set()
    assert calls == [0, 1, "publish"]


def test_blocks_are_not_reused_until_every_save_task_completes():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    gpu_blocks = {1: b"first request block", 2: b"second request block"}
    stored_blocks: dict[bytes, bytes] = {}

    def save(_registration, saves):
        for _layer, block_ids, block_hashes in saves:
            for block_id, block_hash in zip(block_ids, block_hashes, strict=True):
                stored_blocks[block_hash] = gpu_blocks[block_id]
        return True, ""

    worker._ctx.client.save.side_effect = save
    enqueue_save(worker, 1, b"first hash")
    enqueue_save(worker, 2, b"second hash")
    process_next_save(worker)

    finished_sending = worker.get_transfer_results({"request"}).finished_sending
    if finished_sending:
        gpu_blocks[2] = b"reused by another request"

    process_next_save(worker)

    assert stored_blocks[b"second hash"] == b"second request block"


def test_later_save_reopens_completed_request():
    worker = make_worker()
    first_completion = enqueue_save(worker)
    complete_next_save(worker)

    assert first_completion.is_set()

    second_completion = enqueue_save(worker)

    assert not second_completion.is_set()
    finished_sending = worker.get_transfer_results({"request"}).finished_sending
    assert not finished_sending

    complete_next_save(worker)

    assert second_completion.is_set()
    finished_sending = worker.get_transfer_results({"request"}).finished_sending
    assert finished_sending == {"request"}


def test_boundary_state_saves_run_async_and_report_job_completion():
    """HMA boundary-state jobs ride the async save queue.

    Their blocks are pinned by the scheduler, so nothing needs to finish
    before the next step; completion (success or not) is reported back as
    OrbitKVWorkerMetadata so the scheduler can release the pins.
    """
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True, group_count=2)
    worker._registered_layers = ["attention", "recurrent"]
    worker._layer_to_group = {"attention": 0, "recurrent": 1}
    saved: list[tuple[str, list[int], list[bytes]]] = []

    def save(_registration, saves):
        saved.extend(saves)
        return False, "store unavailable"

    worker._ctx.client.save.side_effect = save
    worker._current_metadata = OrbitKVConnectorMetadata(
        boundary_save_intents={
            7: SaveIntent(
                block_ids_by_group=((0, 0), (21, 22)),
                block_hashes=(b"h0", b"h2"),
            )
        }
    )

    worker.wait_for_save()
    worker._ctx.client.save.assert_not_called()
    assert worker.build_connector_worker_meta() is None

    process_next_save(worker)

    # Only the recurrent layer sees the handed-off blocks.
    assert saved == [("recurrent", [21, 22], [b"h0", b"h2"])]
    meta = worker.build_connector_worker_meta()
    assert meta is not None
    assert meta.completed_boundary_jobs == {7: 1}
    assert worker.build_connector_worker_meta() is None
    # Boundary jobs are not requests: no finished_sending entry.
    assert worker.get_transfer_results(set()) == KVConnectorTransferResults()


def test_boundary_state_saves_require_hma():
    worker = make_worker()
    worker._current_metadata = OrbitKVConnectorMetadata(
        boundary_save_intents={0: SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"hash",))}
    )

    with pytest.raises(RuntimeError, match="only valid for HMA"):
        worker.wait_for_save()


def test_hma_request_saves_run_async():
    worker = make_worker()
    worker._cache_groups = SimpleNamespace(has_recurrent_state=True, group_count=1)
    worker._registered_layers = ["layer"]
    worker._ctx.client.save.return_value = (True, "")
    worker._current_metadata = OrbitKVConnectorMetadata(
        save_intents={
            "request": SaveIntent(
                block_ids_by_group=((1,),),
                block_hashes=(b"hash",),
            )
        }
    )

    worker.wait_for_save()
    worker._ctx.client.save.assert_not_called()

    process_next_save(worker)
    worker._ctx.client.save.assert_called_once()
    finished_sending = worker.get_transfer_results({"request"}).finished_sending
    assert finished_sending == {"request"}


def test_save_uses_the_selected_data_plane():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    cache_client = MagicMock(transport="iceoryx2")
    cache_client.save.return_value = (True, "")
    worker._client = cache_client
    enqueue_save(worker)

    process_next_save(worker)

    cache_client.save.assert_called_once_with(worker._registration, [("layer", [1], [b"hash"])])
    worker._ctx.client.save.assert_not_called()


def test_preemption_waits_for_every_save_task():
    worker = make_worker()
    completion = enqueue_save(worker)
    enqueue_save(worker)
    wait_started = threading.Event()
    preemption_returned = threading.Event()
    original_wait = completion.wait

    def wait_for_completion(timeout: float | None = None) -> bool:
        wait_started.set()
        return original_wait(timeout)

    def handle_preemption() -> None:
        worker.handle_preemptions({"request"})
        preemption_returned.set()

    completed_tasks = 0
    with patch.object(completion, "wait", side_effect=wait_for_completion):
        preemption_thread = threading.Thread(target=handle_preemption)
        preemption_thread.start()
        try:
            assert wait_started.wait(timeout=1)

            complete_next_save(worker)
            completed_tasks += 1
            assert not preemption_returned.wait(timeout=0.1)

            complete_next_save(worker)
            completed_tasks += 1
            assert preemption_returned.wait(timeout=1)
        finally:
            for _ in range(2 - completed_tasks):
                complete_next_save(worker)
            preemption_thread.join(timeout=1)

    assert not preemption_thread.is_alive()


def test_malformed_save_intent_is_skipped_and_still_completes():
    """A scheduler bug must not take the save thread down with it: the
    request's blocks are released (unsaved) and other intents in the batch
    still reach the store."""
    worker = make_worker()
    worker._registered_layers = ["layer"]
    worker._ctx.client.save.return_value = (True, "")
    worker._current_metadata = OrbitKVConnectorMetadata(
        save_intents={
            "torn": SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"h0", b"h1")),
            "good": SaveIntent(block_ids_by_group=((2,),), block_hashes=(b"h2",)),
        }
    )
    worker.wait_for_save()

    process_next_save(worker)

    (_, saves), _ = worker._ctx.client.save.call_args
    assert saves == [("layer", [2], [b"h2"])]
    finished_sending = worker.get_transfer_results({"torn", "good"}).finished_sending
    assert finished_sending == {"torn", "good"}


def test_save_worker_survives_a_failing_batch():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    ready = MagicMock()
    ready.synchronize.side_effect = RuntimeError("device lost")
    with patch("torch.cuda.Event", return_value=ready):
        completion = enqueue_save(worker)
    worker._save_queue.put(None)

    with patch.object(worker._client, "save") as save:
        worker._save_worker()
        save.assert_not_called()

    assert completion.is_set()
    finished_sending = worker.get_transfer_results({"request"}).finished_sending
    assert finished_sending == {"request"}


def test_shutdown_publishes_pending_saves_before_unregistering():
    worker = make_worker()
    worker._registered_layers = ["layer"]
    completion = enqueue_save(worker)
    calls = []

    def save(*args):
        calls.append("save")
        return True, ""

    def unregister(*args):
        assert completion.is_set(), "source mappings must outlive pending saves"
        calls.append("unregister")
        return True, ""

    worker._ctx.client.save.side_effect = save
    worker._ctx.client.unregister_context.side_effect = unregister
    # Process the queued batch and shutdown sentinel deterministically at join.
    worker._save_thread = MagicMock()
    worker._save_thread.join.side_effect = worker._save_worker
    worker.shutdown()
    assert calls == ["save", "unregister"]
    assert worker._save_queue.unfinished_tasks == 0


@pytest.mark.parametrize("boundary", [False, True], ids=["request", "boundary"])
def test_queued_save_keeps_the_registration_captured_by_its_producer(boundary):
    worker = make_worker()
    worker._registered_layers = ["layer"]
    original = worker._registration
    worker._client.save.return_value = (True, "")
    if boundary:
        worker._cache_groups = SimpleNamespace(has_recurrent_state=True, group_count=2)
        worker._layer_to_group = {"layer": 0}
        worker._current_metadata = OrbitKVConnectorMetadata(
            boundary_save_intents={
                8: SaveIntent(block_ids_by_group=((1,),), block_hashes=(b"hash",))
            }
        )
        worker.wait_for_save()
    else:
        enqueue_save(worker)
    worker._registration = object()

    process_next_save(worker)

    worker._client.save.assert_called_once_with(original, [("layer", [1], [b"hash"])])


def test_batch_with_two_registration_owners_is_rejected_before_publish(caplog):
    worker = make_worker()
    worker._registered_layers = ["layer"]
    enqueue_save(worker, 1, b"one")
    worker._registration = object()
    completion = enqueue_save(worker, 2, b"two")

    worker._process_save_batch([worker._save_queue.get_nowait() for _ in range(2)])

    worker._client.save.assert_not_called()
    assert "save batch spans different GPU registrations" in caplog.text
    assert completion.is_set()


@pytest.mark.parametrize("accepted", [False, True], ids=["failed-unregister", "normal-unregister"])
def test_unregister_keeps_authority_until_native_cleanup_succeeds(accepted):
    worker = make_worker()
    original = worker._registration
    worker._registered_layers = ["layer"]
    worker._client.unregister_context.return_value = accepted, "controlled cleanup"

    worker.unregister_context()

    assert worker._registration is (None if accepted else original)
    assert worker._registered_layers == ([] if accepted else ["layer"])
