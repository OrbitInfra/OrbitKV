"""GPU destinations stay held until a restore has a confirmed terminal result."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from orbitkv import RestoreStatus  # noqa: E402
from orbitkv.vllm.config import ConnectorContext  # noqa: E402
from orbitkv.vllm.metadata import LoadIntent, OrbitKVConnectorMetadata  # noqa: E402
from orbitkv.vllm.worker import WorkerAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def current_cuda_stream(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.vllm.worker.torch.cuda.Event", lambda *_args, **_kwargs: MagicMock()
    )
    monkeypatch.setattr(
        "orbitkv.vllm.worker.torch.cuda.current_stream",
        lambda _device=None: SimpleNamespace(cuda_stream=17, wait_event=MagicMock()),
    )


class FakeEngineClient:
    """Minimal Cache Manager client for restore and lifecycle tests.

    Only implements what WorkerAdapter touches in the load path. Save path is
    not exercised here since these tests are focused on load fault tolerance.
    """

    transport = "local"

    def __init__(self) -> None:
        self.fail_load_with_exception: Exception | None = None
        self.load_calls: list[tuple] = []
        self.registration = object()
        self.register_exception: Exception | None = None
        self.register_calls: list[tuple] = []
        self.register_kwargs: list[dict] = []
        self.unregister_calls: list[str] = []
        self.release_calls: list[bytes] = []

    def start_restore(
        self,
        instance_id: str,
        tp_rank: int,
        device_id: int,
        layer_groups,
        loads,
        *,
        ready_stream: int,
        layer_events,
    ) -> SimpleNamespace:
        block_ids = [block_id for _, groups in loads for ids in groups for block_id in ids]
        self.load_calls.append(
            (
                instance_id,
                tp_rank,
                device_id,
                None,
                [list(group) for group in layer_groups],
                list(block_ids),
            )
        )
        if self.fail_load_with_exception is not None:
            raise self.fail_load_with_exception
        return SimpleNamespace(key=f"restore-{len(self.load_calls)}")

    def wait_restore_enqueued(self, _handle, *, timeout):
        pass

    def wait_restore(self, _handle, *, timeout):
        return RestoreStatus(done=True, success=True)

    def register_context_batch(self, *args, **kwargs) -> object:
        self.register_calls.append(args)
        self.register_kwargs.append(kwargs)
        if self.register_exception is not None:
            raise self.register_exception
        return self.registration

    def health(self) -> tuple[bool, str]:
        return (True, "ok")

    def unregister_context(self, instance_id: str) -> tuple[bool, str]:
        self.unregister_calls.append(instance_id)
        return (True, "ok")

    def release(self, lease: bytes) -> None:
        self.release_calls.append(lease)


def _make_worker(
    pp_rank: int = 0,
    pp_size: int = 1,
    kv_cache_config=None,
    vllm_config=None,
    **ctx_kwargs,
) -> tuple[WorkerAdapter, FakeEngineClient]:
    client = FakeEngineClient()
    defaults = {
        "instance_id": "test_instance",
        "namespace": "ns",
        "block_size": 16,
        "tp_size": 1,
        "world_size": 1,
        "tp_rank": 0,
        "device_id": 0,
        "client": client,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
    }
    defaults.update(ctx_kwargs)
    ctx = ConnectorContext(**defaults)
    worker = WorkerAdapter(
        ctx,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
    )
    worker._cross_layer_mode = True
    worker._cross_layer_key = "ALL_LAYERS"
    worker._layer_events = {"ALL_LAYERS": object()}
    return worker, client


def _single_attention_cache_group(*layer_names: str) -> MagicMock:
    spec = object.__new__(FullAttentionSpec)
    object.__setattr__(spec, "block_size", 16)
    return MagicMock(layer_names=layer_names, kv_cache_spec=spec)


def _load_metadata(req_id: str, block_ids: tuple[int, ...]) -> OrbitKVConnectorMetadata:
    return OrbitKVConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=(block_ids,),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=len(block_ids) * 16,
            )
        }
    )


def _configure_hma_worker(worker: WorkerAdapter) -> None:
    worker._cross_layer_mode = False
    worker._cache_groups = MagicMock(group_count=2, has_recurrent_state=True)
    worker._registered_layers = ["attention", "recurrent"]
    worker._cache_groups.recurrent_layer_names = ["recurrent"]
    worker._layer_events = {name: object() for name in worker._registered_layers}
    worker._layer_to_group = {"attention": 0, "recurrent": 1}


def _hma_load_metadata(req_id: str) -> OrbitKVConnectorMetadata:
    return OrbitKVConnectorMetadata(
        load_intents={
            req_id: LoadIntent(
                block_ids_by_group=((11,), (21,)),
                leases=(f"lease-{req_id}".encode(),),
                num_tokens=16,
            )
        }
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "error", [ConnectionError("lost acknowledgement"), RuntimeError("rejected")]
)
def test_restore_submission_failure_does_not_release_destinations(hybrid, error):
    worker, client = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("submit")
    else:
        metadata = _load_metadata("submit", (1, 2))
    client.fail_load_with_exception = error

    try:
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.start_load_kv(metadata)
        assert client.release_calls == []
        assert worker.get_transfer_results(set()).finished_recving == set()
    finally:
        worker.shutdown()


def test_hma_load_distinguishes_block_zero_from_absent_recurrent_target():
    worker, client = _make_worker()
    _configure_hma_worker(worker)
    metadata = OrbitKVConnectorMetadata(
        load_intents={
            "hma-sparse": LoadIntent(
                block_ids_by_group=((0, 12), (None, 21)),
                leases=(b"lease-hma-sparse",),
                num_tokens=32,
            )
        }
    )

    worker.start_load_kv(metadata)

    assert client.load_calls[0][5] == [0, 12, None, 21]
    worker.shutdown()


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("stage", ["enqueue", "drain"])
@pytest.mark.parametrize("error", [TimeoutError("deadline"), ConnectionError("lost completion")])
def test_restore_failure_after_admission_keeps_destinations(hybrid, stage, error):
    worker, client = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("pending")
    else:
        metadata = _load_metadata("pending", (5, 6))
    method = "wait_restore_enqueued" if stage == "enqueue" else "wait_restore"
    setattr(client, method, MagicMock(side_effect=error))
    try:
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.start_load_kv(metadata)
            worker.wait_for_save()
        assert worker._restore is not None
        assert worker.get_transfer_results(set()).finished_recving == set()
        assert client.release_calls == []
    finally:
        worker.shutdown()


@pytest.mark.parametrize("hybrid", [False, True])
def test_confirmed_failure_after_forward_cannot_recompute_consumed_pages(hybrid):
    worker, client = _make_worker()
    if hybrid:
        _configure_hma_worker(worker)
        metadata = _hma_load_metadata("failed")
    else:
        metadata = _load_metadata("failed", (5, 6))
    client.wait_restore = MagicMock(
        return_value=RestoreStatus(done=True, success=False, message="drained")
    )
    try:
        worker.start_load_kv(metadata)
        with pytest.raises(RuntimeError, match="GPU pages remain held"):
            worker.wait_for_save()
        assert worker._restore is not None
        assert worker.get_transfer_results(set()).finished_recving == set()
        assert client.release_calls == []
    finally:
        worker.shutdown()


def test_load_uses_registered_layer_names():
    """Load must use the same layer names registered with the server."""
    worker, client = _make_worker()
    worker._cross_layer_mode = False
    worker._registered_layers = ["registered.layer.0", "registered.layer.1"]

    worker.start_load_kv(_load_metadata("req_registered_layers", (1, 2)))

    assert len(client.load_calls) == 1
    assert client.load_calls[0][4] == [["registered.layer.0", "registered.layer.1"]]

    worker.shutdown()


def test_worker_consumes_restore_completion(monkeypatch):
    data_client = MagicMock(transport="iceoryx2")
    restore = SimpleNamespace(key="local:41:9")
    data_client.start_restore.return_value = restore
    data_client.wait_restore.return_value = RestoreStatus(done=True, success=True)
    worker, _unused_client = _make_worker(client=data_client, device_id=3)
    worker._torch_device = "cuda:0"
    current_stream = MagicMock(return_value=SimpleNamespace(cuda_stream=17, wait_event=MagicMock()))
    monkeypatch.setattr("orbitkv.vllm.worker.torch.cuda.current_stream", current_stream)

    worker.start_load_kv(_load_metadata("local-restore", (3, 4)))
    assert worker._restore is not None
    data_client.wait_restore.assert_not_called()
    worker.wait_for_layer_load("model.layer.0")
    current_stream.return_value.wait_event.assert_called_once_with(
        worker._layer_events["ALL_LAYERS"]
    )
    worker.wait_for_save()
    assert worker._restore is None
    assert worker.get_transfer_results(set()).finished_recving == set()
    data_client.start_restore.assert_called_once_with(
        "test_instance",
        0,
        3,
        [["ALL_LAYERS"]],
        [(b"lease-local-restore", [[3, 4]])],
        ready_stream=17,
        layer_events=list(worker._layer_events.items()),
    )
    data_client.wait_restore_enqueued.assert_called_once_with(
        restore, timeout=worker.LOAD_TIMEOUT_SECONDS
    )
    data_client.wait_restore.assert_called_once_with(restore, timeout=worker.LOAD_TIMEOUT_SECONDS)
    worker.shutdown()


class FakeTensor:
    shape = (1, 16)
    dtype = "float16"

    def storage_offset(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cuda:0"

    def stride(self) -> tuple[int, int]:
        return (16, 1)

    def element_size(self) -> int:
        return 2


class FakeCudaIPCWrapper:
    def __init__(self, _tensor) -> None:
        pass


def test_register_version_mismatch_raises_startup_error(monkeypatch):
    worker, client = _make_worker()
    client.register_exception = RuntimeError(
        "OrbitKV version mismatch: client=0.22.4 server=0.22.5"
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="OrbitKV version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert "for layer.0" not in str(exc_info.value)
    assert len(client.register_calls) == 1
    assert client.unregister_calls == []

    worker.shutdown()
    assert client.unregister_calls == []


def test_register_non_version_failure_reports_batch_layers(monkeypatch):
    worker, client = _make_worker()
    client.register_exception = RuntimeError("invalid tensor metadata")

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="invalid tensor metadata") as exc_info:
        worker.register_kv_caches(
            {
                "layer.0": FakeTensor(),
                "layer.1": FakeTensor(),
            }
        )

    message = str(exc_info.value)
    assert "invalid tensor metadata" in message
    assert "for layer.1" not in message
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1"]

    worker.shutdown()


def test_register_kv_caches_ignores_shared_by_without_layer_split_opt_in(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
    ]
    worker, client = _make_worker(
        kv_cache_config=kv_cache_config,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": FakeTensor(),
            "layer.1": FakeTensor(),
            "layer.2": FakeTensor(),
        }
    )

    assert worker._registered_layers == ["layer.0", "layer.1", "layer.2"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.0", "layer.1", "layer.2"]

    worker.shutdown()


def test_register_kv_caches_uses_layer_split_shared_by_plan(monkeypatch):
    class MlaTensor(FakeTensor):
        shape = (1, 1, 16, 1)

        def stride(self) -> tuple[int, int, int, int]:
            return (16, 16, 1, 1)

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [
        _single_attention_cache_group("layer.0", "layer.1", "layer.2")
    ]
    kv_cache_config.kv_cache_tensors = [
        MagicMock(shared_by=("layer.1",)),
        MagicMock(shared_by=()),
        MagicMock(shared_by=("layer.0",)),
    ]
    worker, client = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_kv_caches(
        {
            "layer.0": MlaTensor(),
            "layer.1": MlaTensor(),
            "layer.2": MlaTensor(),
        }
    )

    assert worker._registered_layers == ["layer.1", "layer.0"]
    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["layer.1", "layer.0"]

    worker.shutdown()


def test_register_kv_caches_requires_shared_by_layers(monkeypatch):
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [_single_attention_cache_group("layer.0", "layer.1")]
    kv_cache_config.kv_cache_tensors = [MagicMock(shared_by=("layer.1",))]
    worker, _ = _make_worker(
        kv_cache_config=kv_cache_config,
        vllm_config=MagicMock(additional_config={"mla_layer_split_kv_cache": True}),
        is_mla=True,
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="missing layers"):
        worker.register_kv_caches({"layer.0": FakeTensor()})

    worker.shutdown()


def test_cross_layer_registration_uses_pp_suffixed_name(monkeypatch):
    worker, client = _make_worker(pp_rank=1, pp_size=4)

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    worker.register_cross_layers_kv_cache(FakeTensor(), attn_backend=object())

    assert len(client.register_calls) == 1
    assert client.register_calls[0][7] == ["ALL_LAYERS_pp1"]

    worker.shutdown()


def test_register_version_mismatch_rpc_error_stops_startup(monkeypatch):
    worker, client = _make_worker()
    client.register_exception = RuntimeError(
        "register_context_batch RPC failed: status: FailedPrecondition, "
        'message: "OrbitKV version mismatch: client=0.22.4 server=0.22.5"'
    )

    monkeypatch.setattr("orbitkv.client.gpu.CudaIPCWrapper", FakeCudaIPCWrapper)

    with pytest.raises(RuntimeError, match="OrbitKV version mismatch") as exc_info:
        worker.register_kv_caches({"layer.0": FakeTensor()})

    assert "FailedPrecondition" in str(exc_info.value)
    assert "client=0.22.4" in str(exc_info.value)
    assert "server=0.22.5" in str(exc_info.value)
    assert len(client.register_calls) == 1

    worker.shutdown()


def test_early_restore_is_not_submitted_twice_by_pre_forward():
    worker, client = _make_worker()
    metadata = _load_metadata("early", (3, 4))
    try:
        worker.start_load_kv(metadata)
        worker.start_load_kv(metadata)
        assert len(client.load_calls) == 1
        with pytest.raises(RuntimeError, match="previous forward"):
            worker.start_load_kv(_load_metadata("next", (5, 6)))
        worker.wait_for_save()
        worker.start_load_kv(_load_metadata("next", (5, 6)))
        assert len(client.load_calls) == 2
        worker.wait_for_save()
    finally:
        worker.shutdown()


def test_full_graph_waits_on_gpu_and_retains_the_final_drain(monkeypatch):
    worker, client = _make_worker()
    _configure_hma_worker(worker)
    events = []
    stream = SimpleNamespace(cuda_stream=17, wait_event=events.append)
    monkeypatch.setattr("orbitkv.vllm.worker.torch.cuda.current_stream", lambda _: stream)
    client.wait_restore = MagicMock(return_value=RestoreStatus(done=True, success=True))
    try:
        worker.wait_for_all_layers()
        assert not events
        worker.start_load_kv(_hma_load_metadata("graph"))
        events.clear()
        worker.wait_for_all_layers()
        assert events == list(worker._layer_events.values())
        client.wait_restore.assert_not_called()
        assert worker._restore is not None
        worker.wait_for_save()
        client.wait_restore.assert_called_once()
        events.clear()
        worker.wait_for_all_layers()
        assert not events
    finally:
        worker.shutdown()


@pytest.mark.parametrize(
    "mode,has_attention,wait_all",
    [
        ("NONE", True, False),
        ("PIECEWISE", True, False),
        ("NONE", False, True),
        ("FULL", True, True),
    ],
)
def test_connector_links_restores_at_the_graph_consumption_boundary(mode, has_attention, wait_all):
    from vllm.config import CUDAGraphMode

    from orbitkv.vllm.connector import OrbitKVConnector

    connector = object.__new__(OrbitKVConnector)
    connector._worker = MagicMock()
    metadata = _load_metadata("graph-entry", (1, 2))
    connector._get_connector_metadata = lambda: metadata
    context = SimpleNamespace(
        attn_metadata={} if has_attention else None,
        cudagraph_runtime_mode=CUDAGraphMode[mode],
    )
    connector.start_load_kv(context)
    connector._worker.start_load_kv.assert_called_once_with(metadata)
    assert connector._worker.wait_for_all_layers.call_count == int(wait_all)
