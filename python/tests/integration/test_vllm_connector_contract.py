"""Released vLLM consumers of OrbitKV completion and delivery contracts.

Run when upgrading vLLM or changing connector completion/composition callbacks.
Requires official vLLM 0.33.0. Model composition uses test_vllm_native_pd_e2e.py.
"""

import importlib.metadata
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.test_cache_faults import fault_cache  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("callbacks", ["all", "partial", "none", "capture"])
def test_released_layer_callbacks_and_late_load_dispatch(
    released_config, monkeypatch, runner, callbacks
):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.model_executor.layers.attention import kv_transfer_utils
    from vllm.v1.worker import kv_connector_model_runner_mixin as v1
    from vllm.v1.worker.gpu import kv_connector as v2

    from orbitkv.vllm.connector import OrbitKVConnector
    from tests.unit.test_connector_save_lifecycle import layer_step, make_worker, process_next_save

    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: callbacks == "capture")
    config, cache = released_config
    connector = OrbitKVConnector.__new__(OrbitKVConnector)
    KVConnectorBase_V1.__init__(connector, config, KVConnectorRole.WORKER, cache)
    worker = make_worker()
    connector._worker, connector._scheduler = worker, None
    connector._ctx = SimpleNamespace(query_control=False)
    metadata = layer_step(worker)
    output = SimpleNamespace(
        kv_connector_metadata=metadata, has_sync_kv_loads=False, finished_req_ids=set()
    )
    context = SimpleNamespace(attn_metadata={}, cudagraph_runtime_mode=None)
    for module in (v1, v2):
        monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(v1, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(v2, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer_utils, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attention.get_attention_context",
        lambda name: ({}, None, object(), None),
    )
    written = []

    @kv_transfer_utils.maybe_transfer_kv_layer
    def attention(layer_name):
        assert worker._current_metadata is None
        written.append(layer_name)

    def forward():
        for name in (
            ["first", "last"]
            if callbacks in ("all", "capture")
            else ["first"]
            if callbacks == "partial"
            else []
        ):
            attention(name)
            if callbacks == "capture":
                worker._client.prepare_save.assert_not_called()
            else:
                queued = worker._save_queue.queue[-1]
                assert queued.native_handle.saves[0][0] == written[-1]
        if callbacks in ("all", "partial"):
            process_next_save(worker)
            assert not worker.get_transfer_results({"request"}).finished_sending

    if runner == "v1":
        with v1.KVConnectorModelRunnerMixin._get_kv_connector_output(output):
            forward()
    else:
        active = v2.ActiveKVConnector.__new__(v2.ActiveKVConnector)
        active.kv_connector, active._disabled, active._pending_load_kwargs = connector, False, None
        active.pre_forward(output)
        forward()
        active.post_forward(set())
    while not worker._save_queue.empty():
        process_next_save(worker)
    assert worker.get_transfer_results({"request"}).finished_sending == {"request"}
    assert not connector.has_connector_metadata()
    saved = [
        name for call in worker._client.prepare_save.call_args_list for name, _, _ in call.args[1]
    ]
    assert saved == ["first", "last"]


@pytest.mark.parametrize("kwargs", [{}, {"enable_dbo": True}, {"ubatch_size": 2}])
def test_released_microbatch_property_controls_early_save(kwargs):
    from vllm.config import ParallelConfig

    from tests.unit.test_connector_save_lifecycle import make_worker

    config = ParallelConfig(**kwargs)
    worker = make_worker(parallel_config=config)
    assert worker._layerwise_save is (not config.use_ubatching)
    assert worker._layerwise_save is (not kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("fault_cache", [{"num_layers": 2}], indirect=True)
@pytest.mark.parametrize("last_callback", [False, True], ids=["step-end-remainder", "all-layers"])
def test_native_layer_save_bytes_and_partial_visibility(fault_cache, last_callback):  # noqa: F811
    """Written installed gate: real producer events, native Save, partial slots and Restore."""
    import torch

    from orbitkv.vllm.config import ConnectorContext
    from orbitkv.vllm.metadata import OrbitKVConnectorMetadata, SaveIntent
    from orbitkv.vllm.worker import WorkerAdapter
    from tests.integration.test_cache_faults import query, until
    from tests.support.metrics import fetch_orbitkv_metrics

    server, client, ctx, _ = fault_cache
    worker = WorkerAdapter(ConnectorContext(ctx.instance_id, ctx.namespace, 16, 1, 1, 0, 0, client))
    worker._registration, worker._registered_layers = ctx.registration, ctx._layer_names
    worker._torch_device = ctx.device
    metadata = OrbitKVConnectorMetadata(save_intents={"r": SaveIntent(((1,),), (b"layer-save",))})
    producer = torch.cuda.Stream(device=ctx.device)
    producer.wait_stream(torch.cuda.current_stream(ctx.device))
    try:
        with torch.cuda.stream(producer):
            ctx.get_kv_cache(0)[:, 1].fill_(11)
            worker.save_kv_layer(ctx._layer_names[0], metadata)
        completion = worker._save_completion_events["r"]
        # The first native receipt alone must not publish a complete cache hit.
        until(lambda: worker._req_pending_save_tasks.get("r") == 1)
        assert not completion.is_set()
        assert not worker.get_transfer_results({"r"}).finished_sending
        until(lambda: fetch_orbitkv_metrics(server.http_port).get("orbitkv_inflight_bytes", 0) > 0)
        partial = query(client, ctx, [b"layer-save"], "partial")
        assert partial.num_hit_blocks == 0
        client.cancel_query(ctx.instance_id, "partial")
        with torch.cuda.stream(producer):
            ctx.get_kv_cache(1)[:, 1].fill_(23)
            if last_callback:
                worker.save_kv_layer(ctx._layer_names[1], metadata)
            worker.start_load_kv(metadata)
            worker.wait_for_save()
        assert completion.wait(30)
        assert worker.get_transfer_results(set()).finished_sending == {"r"}
        for tensor in ctx.gpu_kv_caches:
            tensor[:, 1].fill_(87)
            tensor[:, 2].zero_()
        attempt = 0

        def published():
            nonlocal attempt
            attempt += 1
            rid = f"restore-{attempt}"
            result = query(client, ctx, [b"layer-save"], rid)
            if result.num_hit_blocks == 1:
                return result
            client.cancel_query(ctx.instance_id, rid)
            return None

        ready = until(published)
        handle = client.start_restore(
            ctx.instance_id,
            0,
            0,
            [ctx._layer_names],
            [(ready.lease, [[2]])],
            ready_stream=torch.cuda.current_stream(ctx.device).cuda_stream,
        )
        assert client.wait_restore(handle, timeout=30).success
        for index, expected in enumerate((11, 23)):
            assert torch.all(ctx.get_kv_cache(index)[:, 2].cpu() == expected)
    finally:
        worker.shutdown()


def test_native_multi_connector_keeps_cache_best_effort_and_handoff_reliable():
    from tests.support.installed_artifacts import ENGINE_VERSIONS

    assert importlib.metadata.version("vllm") == ENGINE_VERSIONS["vllm"]
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    from orbitkv.vllm.connector import OrbitKVConnector

    cache = OrbitKVConnector.__new__(OrbitKVConnector)
    cache._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    prefill = NixlConnector.__new__(NixlConnector)
    prefill._kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    multi = MultiConnector.__new__(MultiConnector)
    multi._connectors = [cache]
    assert not multi.requires_kv_delivery
    multi._connectors.append(prefill)
    assert multi.requires_kv_delivery
    assert not multi.supports_divergent_local_hybrid_hits


@pytest.fixture
def released_config():
    import torch
    from vllm.config import KVTransferConfig
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OrbitKVConnector",
            kv_connector_module_path="orbitkv.vllm",
            kv_role="kv_both",
            engine_id="contract-control",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(), enable_sleep_mode=False),
        lora_config=None,
        use_v2_model_runner=False,
    )
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16)
    cache = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer.0"], spec)])
    return config, cache


@pytest.mark.parametrize("profile", ["v2", "multi-group", "recurrent", "sleep"])
def test_unsupported_released_profiles_fail_before_connecting(
    released_config, monkeypatch, profile
):
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import MambaSpec

    from orbitkv.vllm.connector import OrbitKVConnector

    def unexpected_connection(**kwargs):
        pytest.fail("unsupported profile opened a Manager connection")

    monkeypatch.setattr("orbitkv.vllm.connector.connect_cache", unexpected_connection)
    config, cache = released_config
    if profile == "sleep":
        config.model_config.enable_sleep_mode = True
        with pytest.raises(ValueError, match="does not support sleep mode"):
            VllmConfig._verify_kv_transfer_compat(config)
        return
    if profile == "v2":
        config.use_v2_model_runner = True
    elif profile == "multi-group":
        cache.kv_cache_groups *= 2
    else:
        cache.kv_cache_groups[0].kv_cache_spec = MambaSpec(
            block_size=16, shapes=((2, 4),), dtypes=(torch.float32,), mamba_cache_mode="align"
        )
    with pytest.raises(RuntimeError, match="VLLM_USE_V2|require window|single cache group"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)


def test_released_single_group_and_hit_source_defaults(released_config):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.metrics.cache_hit_source import CacheHitSource

    from orbitkv.vllm.connector import OrbitKVConnector
    from orbitkv.vllm.layout import CacheGroupLayout

    config, cache = released_config
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config = SimpleNamespace(block_size=16, prefix_match_unit=None)
    assert resolve_kv_cache_block_sizes(cache, config) == (16, 16)
    assert CacheGroupLayout.from_config(cache).recovery_groups == ((0, "attention", 0),)
    connector = object.__new__(OrbitKVConnector)
    KVConnectorBase_V1.__init__(connector, config, KVConnectorRole.SCHEDULER, cache)
    assert connector.get_loaded_kv_cache_group_ids(None) == (0,)
    assert connector.get_external_cache_hit_sources(None, 32) == {
        CacheHitSource.EXTERNAL_UNSPECIFIED: 32
    }
    config.cache_config.prefix_match_unit = 8
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix_match_unit"):
        resolve_kv_cache_block_sizes(cache, config)


@pytest.mark.parametrize("opt_in", [False, "true", True])
def test_v2_admission_requires_boolean_opt_in(released_config, monkeypatch, opt_in):
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    config.use_v2_model_runner = True
    config.kv_transfer_config.kv_connector_extra_config["orbitkv.v2"] = opt_in
    config.cache_config = SimpleNamespace(block_size=16, prefix_match_unit=None)
    config.parallel_config.decode_context_parallel_size = 1

    class Admitted(Exception):
        pass

    # Stop after admission, before identity resolution or any Manager connection.
    def admitted(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr("orbitkv.vllm.connector.derive_namespace", admitted)
    if opt_in is True:
        with pytest.raises(Admitted):
            OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
    else:
        with pytest.raises(RuntimeError, match="opt-in"):
            OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)


@pytest.mark.parametrize("profile", ["missing-group", "mla", "speculation", "producer-only"])
def test_v2_opt_in_does_not_widen_the_audited_profile(released_config, profile):
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from orbitkv.vllm.connector import OrbitKVConnector

    config, cache = released_config
    config.use_v2_model_runner = True
    config.kv_transfer_config.kv_connector_extra_config["orbitkv.v2"] = True
    if profile == "missing-group":
        cache.kv_cache_groups = []
    elif profile == "mla":
        cache.kv_cache_groups[0].kv_cache_spec = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        )
    elif profile == "speculation":
        config.speculative_config = object()
    else:
        config.kv_transfer_config.kv_role = "kv_producer"
    with pytest.raises(RuntimeError, match="one uniform FullAttention"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER, cache)
