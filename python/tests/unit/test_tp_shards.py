"""Contracts for splitting one vLLM TP replica across OrbitKV servers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from tests.support.unit_stubs import install_connector_unit_stubs

install_connector_unit_stubs()

from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)

from orbitkv.orbitkv import BlockHashes, QueryLoading, ShardedQueryReady  # noqa: E402
from orbitkv.vllm import OrbitKVConnector  # noqa: E402
from orbitkv.vllm.config import ConnectorContext, TpShardTopology  # noqa: E402
from orbitkv.vllm.metadata import (  # noqa: E402
    LoadIntent,
    OrbitKVConnectorMetadata,
    OrbitKVQueryHandshake,
    OrbitKVWorkerMetadata,
)
from orbitkv.vllm.scheduler import SchedulerAdapter, _QueryProbe  # noqa: E402
from orbitkv.vllm.worker import WorkerAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _available_local_sockets(monkeypatch):
    monkeypatch.setattr("orbitkv.vllm.connector.derive_namespace", lambda *_a, **_k: "identity")
    monkeypatch.setattr("orbitkv.client.connection._is_unix_socket", lambda _path: True)
    monkeypatch.setattr(
        "orbitkv.vllm.connector.get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )


def _topology() -> TpShardTopology:
    return TpShardTopology.from_config(
        default_endpoint="http://unused:50055",
        configured_endpoints=["http://node-a:50055", "http://node-b:50055"],
        global_tp_size=8,
        global_world_size=8,
    )


def _context(**kwargs) -> ConnectorContext:
    defaults = {
        "instance_id": "instance",
        "namespace": "namespace:tp-shard-0-of-2",
        "block_size": 16,
        "tp_size": 8,
        "world_size": 8,
        "tp_rank": 0,
        "device_id": 0,
        "client": MagicMock(),
        "tp_shards": _topology(),
        "query_control": True,
    }
    defaults.update(kwargs)
    return ConnectorContext(**defaults)  # type: ignore[arg-type]


def _vllm_config(*, extra_overrides=None, **parallel_overrides):
    extra_config = {
        "orbitkv.tp_shard_endpoints": [
            "http://127.0.0.1:50055",
            "http://127.0.0.1:50056",
        ]
    }
    extra_config.update(extra_overrides or {})
    kv_transfer_config = SimpleNamespace(
        engine_id="instance",
        get_from_extra_config=lambda key, default: extra_config.get(key, default),
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="model",
            dtype="bfloat16",
            hf_text_config=SimpleNamespace(kv_lora_rank=None),
            get_total_num_kv_heads=lambda: 8,
            get_head_size=lambda: 128,
            get_total_num_hidden_layers=lambda: 32,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            **{
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "world_size": 8,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                **parallel_overrides,
            }
        ),
        kv_transfer_config=kv_transfer_config,
        lora_config=None,
        additional_config={},
        use_v2_model_runner=False,
    )


def test_topology_maps_contiguous_global_tp_ranks_to_local_servers():
    topology = _topology()

    assert topology.local_tp_size == 4
    assert topology.local_world_size == 4
    assert topology.shard_index(3) == 0
    assert topology.shard_index(4) == 1
    assert topology.local_tp_rank(4) == 0
    assert topology.local_tp_rank(7) == 3
    assert topology.namespace("base", 1) == "base:tp-shard-1-of-2"


@pytest.mark.parametrize(
    ("endpoints", "tp_size", "world_size", "message"),
    [
        ([], 8, 8, "non-empty strings"),
        (["http://a", "http://a"], 8, 8, "duplicates"),
        (["http://a", "http://b", "http://c"], 8, 8, "tensor_parallel_size"),
        (["http://a", "http://b"], 8, 9, "world_size"),
    ],
)
def test_topology_rejects_ambiguous_shard_layouts(endpoints, tp_size, world_size, message):
    with pytest.raises(ValueError, match=message):
        TpShardTopology.from_config("http://unused", endpoints, tp_size, world_size)


def test_context_exposes_node_local_server_topology_for_hma():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=False)

    assert context.tp_shard_index == 1
    assert context.effective_tp_rank == 1
    assert context.effective_tp_size == 4
    assert context.effective_world_size == 4


def test_worker_connector_routes_global_tp_rank_to_its_local_manager(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr("orbitkv.vllm.connector.get_tensor_model_parallel_rank", lambda: 5)
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", client_factory)

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.WORKER)
    try:
        assert connector._ctx.namespace.endswith(":tp-shard-1-of-2")
        assert connector._ctx.effective_tp_rank == 1
        assert connector._ctx.effective_tp_size == 4
        assert connector._ctx.effective_world_size == 4
    finally:
        connector.shutdown()

    client_factory.assert_called_once_with(
        "/tmp/orbitkv-50056.sock", timeout_ms=5_000, spin_iterations=64
    )


def test_scheduler_opens_only_its_local_topology_session(monkeypatch):
    first = MagicMock(transport="iceoryx2")
    second = MagicMock(transport="iceoryx2")
    monkeypatch.setattr(
        "orbitkv.client.connection.CacheManagerClient", MagicMock(side_effect=[first, second])
    )

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    try:
        first.start_session_watcher.assert_called_once()
        assert first.start_session_watcher.call_args.args[1].endswith(":tp-shard-0-of-2")
        assert first.start_session_watcher.call_args.args[2:] == (4, 4)
        second.start_session_watcher.assert_not_called()
    finally:
        connector.shutdown()


def test_scheduler_maps_each_tp_shard_to_its_local_socket(monkeypatch):
    clients = [MagicMock(transport="iceoryx2"), MagicMock(transport="iceoryx2")]
    factory = MagicMock(side_effect=clients)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", factory)
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_bootstrap_sockets": [
                "/run/orbitkv/a.sock",
                "/run/orbitkv/b.sock",
            ],
        }
    )

    connector = OrbitKVConnector(config, KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.client is clients[0]
        assert connector._scheduler._clients == (clients[0],)
        assert connector._connections.clients == (clients[0],)
        clients[0].start_session_watcher.assert_called_once()
        clients[1].start_session_watcher.assert_not_called()
    finally:
        connector.shutdown()

    assert factory.call_args_list == [
        call("/run/orbitkv/a.sock", timeout_ms=5_000, spin_iterations=64),
    ]


def test_scheduler_derives_distinct_local_sockets(monkeypatch):
    clients = [MagicMock(transport="iceoryx2"), MagicMock(transport="iceoryx2")]
    factory = MagicMock(side_effect=clients)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", factory)

    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.client is clients[0]
    finally:
        connector.shutdown()
    assert factory.call_args_list == [
        call("/tmp/orbitkv-50055.sock", timeout_ms=5_000, spin_iterations=64),
    ]


def test_scheduler_rejects_remote_inference_shard(monkeypatch):
    monkeypatch.setattr("orbitkv.client.connection._endpoint_is_local", lambda _endpoint: False)
    local_factory = MagicMock()
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", local_factory)

    with pytest.raises(ValueError, match="node-local Cache Manager"):
        OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    local_factory.assert_not_called()


def test_scheduler_fails_if_bootstrap_fails(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.client.connection.CacheManagerClient",
        MagicMock(side_effect=RuntimeError("stale socket")),
    )
    with pytest.raises(RuntimeError, match="stale socket"):
        OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)


def test_scheduler_closes_every_connection_when_session_start_fails(monkeypatch):
    clients = [MagicMock(transport="iceoryx2"), MagicMock(transport="iceoryx2")]
    clients[0].start_session_watcher.side_effect = RuntimeError("session rejected")
    monkeypatch.setattr(
        "orbitkv.client.connection.CacheManagerClient", MagicMock(side_effect=clients)
    )

    with pytest.raises(RuntimeError, match="session rejected"):
        OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)

    clients[0].close.assert_called_once()
    clients[1].close.assert_not_called()


def test_worker_uses_only_its_tp_shard_socket(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr("orbitkv.vllm.connector.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", factory)
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_bootstrap_sockets": ["/run/orbitkv/a.sock", "/run/orbitkv/b.sock"],
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.client is client
        assert connector._worker._client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/run/orbitkv/b.sock", timeout_ms=5_000, spin_iterations=64)


def test_worker_uses_its_local_shard_when_other_shards_are_remote(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr("orbitkv.vllm.connector.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", factory)
    monkeypatch.setattr(
        "orbitkv.client.connection._endpoint_is_local",
        lambda endpoint: endpoint == "http://node-b:50055",
    )
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_endpoints": ["http://node-a:50055", "http://node-b:50055"]
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/tmp/orbitkv-50055.sock", timeout_ms=5_000, spin_iterations=64)


def test_worker_uses_selected_socket_when_other_shards_are_remote(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr("orbitkv.vllm.connector.get_tensor_model_parallel_rank", lambda: 5)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("orbitkv.client.connection.CacheManagerClient", factory)
    monkeypatch.setattr(
        "orbitkv.client.connection._endpoint_is_local",
        lambda endpoint: endpoint == "http://node-b:50055",
    )
    config = _vllm_config(
        extra_overrides={
            "orbitkv.tp_shard_endpoints": ["http://node-a:50055", "http://node-b:50055"],
            "orbitkv.tp_shard_bootstrap_sockets": ["/run/a.sock", "/run/b.sock"],
        }
    )
    connector = OrbitKVConnector(config, KVConnectorRole.WORKER)
    try:
        assert connector._ctx.client is client
    finally:
        connector.shutdown()
    factory.assert_called_once_with("/run/b.sock", timeout_ms=5_000, spin_iterations=64)


def test_full_prefix_prefetch_uses_local_client(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr(
        "orbitkv.client.connection.CacheManagerClient", MagicMock(return_value=client)
    )
    config = _vllm_config(
        extra_overrides={"orbitkv.wait_for_full_prefix": True, "orbitkv.tp_shard_endpoints": None}
    )
    connector = OrbitKVConnector(config, KVConnectorRole.SCHEDULER)
    try:
        assert connector._ctx.wait_for_full_prefix
        assert connector._ctx.client is client
    finally:
        connector.shutdown()
    assert client.close.called


@pytest.mark.parametrize(
    "parallel_overrides",
    [
        {"pipeline_parallel_size": 2, "world_size": 16},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
    ],
)
def test_connector_rejects_non_tp_parallelism_across_server_shards(monkeypatch, parallel_overrides):
    with pytest.raises(ValueError, match="TP-only parallelism"):
        OrbitKVConnector(_vllm_config(**parallel_overrides), KVConnectorRole.SCHEDULER)


def test_pure_mla_collapses_storage_tp_but_stripes_saves_within_each_node():
    context = _context(tp_rank=5, is_mla=True, collapse_mla_tp=True)

    assert context.effective_tp_rank == 0
    assert context.effective_tp_size == 1
    assert context.effective_world_size == 4
    assert context.local_physical_tp_rank == 1
    assert context.local_physical_tp_size == 4


def test_scheduler_delegates_dense_shards_to_one_native_query_owner():
    client = MagicMock()
    response = MagicMock(spec=ShardedQueryReady)
    response.num_hit_blocks = 2
    response.leases = (b"a" * 16, b"b" * 16)
    response.control_id = b"c" * 16
    client.query_shards.return_value = response
    scheduler = SchedulerAdapter(_context(client=client))
    scheduler.query_handshake_ready = True
    probe = _QueryProbe(0, (b"h0", b"h1", b"h2"))
    ready = scheduler._query_recovery("request", probe, 10000)
    assert ready.num_hit_blocks == 2
    assert ready.leases == (b"a" * 16, b"b" * 16)
    assert probe.control_id == b"c" * 16
    client.query_shards.assert_called_once_with(
        "instance", BlockHashes(probe.query_hashes), "request"
    )
    client.query_prefetch.assert_not_called()
    assert scheduler._release_query_probe("request", probe)
    client.release_shard_query.assert_called_once_with(b"c" * 16)
    client.release.assert_not_called()


def test_scheduler_cancels_drifted_native_query_before_replacing_hashes():
    client = MagicMock()
    client.query_shards.return_value = QueryLoading()
    scheduler = SchedulerAdapter(_context(client=client))
    scheduler.query_handshake_ready = True
    request = SimpleNamespace(
        request_id="request", block_hashes=[b"h0", b"h1", b"h2", b"h3"], num_tokens=64
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert scheduler.get_num_new_matched_tokens(request, 32) == (None, False)
    client.cancel_query.assert_called_once_with("instance", "request")
    assert client.query_shards.call_args_list == [
        call("instance", BlockHashes(request.block_hashes), "request"),
        call("instance", BlockHashes(request.block_hashes[2:]), "request"),
    ]
    scheduler.shutdown()
    assert client.cancel_query.call_count == 2


def test_official_handshake_requires_every_rank_and_identical_node_registration(monkeypatch):
    client = MagicMock(transport="iceoryx2")
    monkeypatch.setattr(
        "orbitkv.client.connection.CacheManagerClient", MagicMock(return_value=client)
    )
    connector = OrbitKVConnector(_vllm_config(), KVConnectorRole.SCHEDULER)
    records = {
        rank: OrbitKVQueryHandshake(b"node-a" if rank < 4 else b"node-b") for rank in range(8)
    }
    try:
        with pytest.raises(ValueError, match="every declared"):
            connector.set_xfer_handshake_metadata({rank: records[rank] for rank in range(7)})
        mismatch = dict(records)
        mismatch[3] = OrbitKVQueryHandshake(b"other")
        with pytest.raises(ValueError, match="different query"):
            connector.set_xfer_handshake_metadata(mismatch)
        client.configure_shard_queries.assert_not_called()
        connector.set_xfer_handshake_metadata(records)
        client.configure_shard_queries.assert_called_once_with(
            "instance",
            "identity:tp-shard-0-of-2",
            4,
            4,
            [
                ("http://127.0.0.1:50055", "identity:tp-shard-0-of-2", b"node-a"),
                ("http://127.0.0.1:50056", "identity:tp-shard-1-of-2", b"node-b"),
            ],
        )
        assert connector._scheduler.query_handshake_ready
    finally:
        connector.shutdown()


def test_query_control_release_requires_unique_completion_from_all_global_ranks():
    client = MagicMock()
    scheduler = SchedulerAdapter(_context(client=client))
    control = b"c" * 16
    scheduler._query_controls[control] = set()

    def report(ranks):
        scheduler.update_connector_output(
            SimpleNamespace(
                finished_sending=set(),
                kv_connector_worker_meta=OrbitKVWorkerMetadata({}, {control: ranks}),
            )
        )

    report({0, 1, 2, 3})
    report({0, 1, 2, 3})
    client.release_shard_query.assert_not_called()
    with pytest.raises(ValueError, match="undeclared"):
        report({8})
    report({4, 5, 6, 7})
    client.release_shard_query.assert_called_once_with(control)
    assert control not in scheduler._query_controls
    report({4, 5, 6, 7})
    client.release_shard_query.assert_called_once_with(control)


def test_worker_selects_the_lease_for_its_local_server(monkeypatch):
    monkeypatch.setattr(
        "orbitkv.vllm.worker.torch.cuda.current_stream",
        lambda _device=None: SimpleNamespace(cuda_stream=17),
    )
    engine_client = MagicMock()
    engine_client.start_restore.return_value = SimpleNamespace(key="restore-1")
    context = _context(
        tp_rank=5,
        device_id=1,
        namespace="namespace:tp-shard-1-of-2",
        client=engine_client,
    )
    worker = WorkerAdapter(context)
    worker._registered_layers = ["layer"]
    metadata = OrbitKVConnectorMetadata(
        load_intents={
            "request": LoadIntent(
                block_ids_by_group=((7,),),
                leases=(b"node-a", b"node-b"),
                num_tokens=16,
            )
        }
    )

    try:
        worker.start_load_kv(metadata)
    finally:
        worker._registered_layers = []
        worker.shutdown()

    loads = engine_client.start_restore.call_args.args[4]
    assert loads == [(b"node-b", [[7]])]


def test_each_tp_shard_has_a_local_unregister_leader():
    for tp_rank in range(8):
        engine_client = MagicMock()
        engine_client.unregister_context.return_value = (True, "")
        worker = WorkerAdapter(_context(tp_rank=tp_rank, client=engine_client))
        worker._registered_layers = ["layer"]

        worker.unregister_context()
        worker.shutdown()

        if tp_rank in (0, 4):
            engine_client.unregister_context.assert_called_once_with("instance")
        else:
            engine_client.unregister_context.assert_not_called()


def test_v2_runner_is_rejected_before_connecting(monkeypatch):
    connect = MagicMock()
    monkeypatch.setattr("orbitkv.vllm.connector.connect_cache", connect)
    config = _vllm_config()
    config.use_v2_model_runner = True
    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        OrbitKVConnector(config, KVConnectorRole.SCHEDULER)
    connect.assert_not_called()
