"""Exercise the pinned SGLang admission hook with controlled backing completion."""

import threading
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def linker(monkeypatch):
    pytest.importorskip("sglang")
    from orbitkv.sglang.linker import OrbitKVLinker

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(radix_cache_backend="orbitkv"),
    )
    result = object.__new__(OrbitKVLinker)
    from orbitkv import RecoveryContract

    result.instance_id = "admission"
    result._static_lora_ids = frozenset()
    result.namespace = "test-model-state"
    result.layout = SimpleNamespace(
        pools={"kv": SimpleNamespace(group_id=0, kind="attention", window=0)}
    )
    result.recovery = RecoveryContract(result.namespace, 64, [(0, "attention", 0)])
    result._origins = dict.fromkeys(["slow", "req", "shared", "queued", "other"], 0)
    result._load_boundaries = {}
    result.page_size = 64
    result.client = MagicMock()
    result._lookups = {}
    result._pending_queries = {}
    result._expired_queries = set()
    result._queued_loads = {}
    return result


def request(rid, token_count=256):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(rid, "", array("q", [1] * token_count), SamplingParams())
    req._refresh_fill_ids()
    return req


@pytest.mark.parametrize("alive", [True, False, None])
def test_publish_activity_does_not_complete_offloads(linker, alive):
    import queue

    linker._last_publish_warning = 0.0
    linker._completed_offloads = queue.Queue()
    linker.client.publish_progress.return_value = SimpleNamespace(
        active=1,
        retained=1,
        overdue=True,
        active_interval_seconds=61.0,
        manager_alive=alive,
    )
    assert linker.num_completed_offloads() == 0
    linker.client.publish_progress.side_effect = RuntimeError("observation unavailable")
    assert linker.num_completed_offloads() == 0
    linker._completed_offloads.put(True)
    assert linker.num_completed_offloads() == 1


def transfer(keys):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    return [SimpleNamespace(name=PoolName.KV, keys=keys)]


@pytest.mark.parametrize("preparation", [False, True], ids=["warming", "owned"])
def test_enqueue_uses_the_same_salted_storage_keys_without_triggering_a_load(
    linker, monkeypatch, preparation
):
    from orbitkv import BlockHashes

    setting = "ORBITKV_PREPARE_REQUESTS" if preparation else "ORBITKV_QUEUE_WARMUP"
    monkeypatch.setenv(setting, "1")
    import torch
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )
    from sglang.srt.mem_cache.utils import get_storage_hash_str

    from orbitkv.sglang.admission import enqueue_request

    req = request("queued", 257)
    req.extra_key, req.cache_salt = "tenant", "salt"
    req.return_logprob, req.logprob_start_len = True, 192
    key = RadixKey(req.origin_input_ids, extra_key="tenant", cache_salt="salt", limit=192)
    hashes = get_storage_hash_str(key, page_size=64)
    cache = SimpleNamespace(page_size=64, get_last_hash_value=lambda _: hashes[0])
    wrapper = object.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache, wrapper.cache_linker = cache, linker
    wrapper.restore_from_store = True
    cache.linker = wrapper
    cache.match_prefix = MagicMock(
        return_value=SimpleNamespace(
            device_indices=torch.arange(64),
            last_device_node=object(),
        )
    )
    scheduler = SimpleNamespace(tree_cache=cache, waiting_queue=[])

    def accepted(scheduler, req):
        scheduler.waiting_queue.append(req)

    enqueue_request(accepted, scheduler, req)
    batch = BlockHashes(linker._hashes(hashes[1:]))
    if preparation:
        linker.client.prepare_recovery.assert_called_once_with(
            "admission", batch, req.rid, linker.recovery, linker.namespace, 64, 192, 0
        )
    else:
        linker.client.warm_prefix.assert_called_once_with("admission", batch, req.rid)
    submit = linker.client.prepare_recovery if preparation else linker.client.warm_prefix
    assert cache.match_prefix.call_args.args[0].req is None
    assert not linker._lookups and not linker._queued_loads
    linker.client.reset_mock()
    scheduler.waiting_queue.clear()
    enqueue_request(MagicMock(), scheduler, req)
    submit.assert_not_called()

    submit.side_effect = RuntimeError("manager unavailable")
    enqueue_request(accepted, scheduler, req)
    assert scheduler.waiting_queue[-1] is req
    submit.assert_called_once()
    linker.client.reset_mock()
    cache.match_prefix.reset_mock()
    monkeypatch.delenv(setting)
    enqueue_request(accepted, scheduler, req)
    cache.match_prefix.assert_not_called()
    submit.assert_not_called()
    if preparation:
        monkeypatch.setenv(setting, "1")
        scheduler.waiting_queue[:] = [object()] * 4
        enqueue_request(accepted, scheduler, req)
        submit.assert_not_called()


@pytest.mark.parametrize("outcome", ["SUCCESS", "ABORT"])
def test_native_cache_finish_cancels_aborted_query_without_scheduler_hook(linker, outcome):
    from sglang.srt.mem_cache.base_prefix_cache import (
        CacheRequestHandle,
        CacheRequestOutcome,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    cache = object.__new__(UnifiedRadixCache)
    wrapper = object.__new__(RecoveryLinkerWrapper)
    wrapper.cache, wrapper.cache_linker = cache, linker
    wrapper.hit_markers = {"queued": object()}
    cache.linker = wrapper
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.storage_prefetch_retries = MagicMock()
    cache.buffer_pipeline = None
    cache.discard_storage_prefetch_accounting = MagicMock()
    cache.ongoing_prefetch = {}
    cache.finish(CacheRequestHandle("queued", 0), CacheRequestOutcome[outcome])
    if outcome == "ABORT":
        linker.client.cancel_query.assert_called_once_with("admission", "queued", group_id=0)
        assert "queued" not in linker._origins
        assert not wrapper.hit_markers
    else:
        linker.client.cancel_query.assert_not_called()
        assert "queued" in linker._origins


@pytest.fixture
def layer_counter(monkeypatch):
    from orbitkv.sglang.events import _LayerDoneCounter

    monkeypatch.setattr("torch.cuda.Event", MagicMock())
    stream = SimpleNamespace(cuda_stream=17, wait_event=MagicMock())
    monkeypatch.setattr("torch.cuda.current_stream", lambda: stream)
    layout = SimpleNamespace(
        num_layers=2,
        pools={
            "kv": SimpleNamespace(
                layer_names=["k:0", "k:1", "v:0", "v:1"],
                entry=SimpleNamespace(layer_mapping={0: 0, 1: 1}),
            )
        },
    )
    return _LayerDoneCounter(layout), stream


def test_first_use_is_observed_once_even_when_the_first_layer_wait_repeats(
    layer_counter, monkeypatch
):
    counter, stream = layer_counter
    trace = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.events.trace_transfer", trace)
    index = counter.update_producer()
    counter.request_ids[index] = ["restored"]
    counter.publish_events(index)
    counter.set_consumer(index)
    counter.wait_until(0)
    counter.wait_until(0)
    counter.wait_until(1)
    assert stream.wait_event.call_count == 6
    trace.assert_called_once_with("first_use", "restored", engine="sglang")


def test_graph_capture_keeps_dependencies_and_forward_waits_for_new_records(layer_counter):
    counter, stream = layer_counter
    # Registration stays component-major; Restore follows the consumer's layer order.
    assert counter.layer_groups == [["k:0", "v:0", "k:1", "v:1"]]
    assert counter.layout.pools["kv"].layer_names == ["k:0", "k:1", "v:0", "v:1"]
    # Capture precedes the first Restore; both components must still emit a wait.
    counter.wait_until(0)
    assert stream.wait_event.call_count == 2
    for _ in range(2):
        index = counter.update_producer()
        activation = counter._activations[index]
        entered, executing = threading.Event(), threading.Event()

        def replay(index=index, entered=entered, executing=executing):
            entered.set()
            counter.set_consumer(index)
            executing.set()

        thread = threading.Thread(target=replay, daemon=True)
        thread.start()
        assert entered.wait(timeout=1)
        try:
            assert activation.result(timeout=1) == 17
            assert not executing.wait(timeout=0.05)
        finally:
            counter.publish_events(index)
            thread.join(timeout=1)
        assert executing.is_set()
        assert not counter._activations and not counter._staged
        assert not counter.request_ids


def test_pending_query_defers_only_its_request_and_preserves_ready_lease(linker, monkeypatch):
    from sglang.srt.managers.schedule_policy import AddReqResult

    from orbitkv import QueryLoading, QueryReady
    from orbitkv.sglang.admission import admit_request

    linker.client.query_prefetch.side_effect = [QueryLoading(), QueryReady(2, b"lease")]
    original = MagicMock(return_value=AddReqResult.CONTINUE)
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker, needs_rank_consensus=False),
        _all_reduce_attn_groups=MagicMock(),
    )
    adder = SimpleNamespace(tree_cache=cache)
    req = request("slow")
    keys = transfer(["a", "b"])
    assert linker.lookup(req.rid, keys) == []
    import torch

    def reject_tensor(*args, **kwargs):
        raise AssertionError("Single-rank admission must use the scalar query decision")

    monkeypatch.setattr(torch, "tensor", reject_tensor)
    assert admit_request(original, adder, req) == AddReqResult.CONTINUE
    original.assert_not_called()
    unrelated = request("other")
    admit_request(original, adder, unrelated)
    original.assert_called_once_with(adder, unrelated)

    assert linker.lookup(req.rid, keys) == [1, 2]
    assert linker.lookup(req.rid, keys) == [1, 2]
    assert linker.client.query_prefetch.call_count == 2
    linker.client.release.assert_not_called()
    admit_request(original, adder, req)
    assert original.call_count == 2
    linker.cancel_queued_load(req.rid)
    linker.client.release.assert_called_once_with(b"lease")
    assert not linker._lookups


@pytest.mark.parametrize("kind", ["recurrent", "window"])
def test_hybrid_discovery_preserves_earlier_boundaries_before_reading(linker, kind):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    from orbitkv import (
        BlockHashes,
        QueryCandidates,
        QueryLoading,
        QueryReady,
        RecoveryContract,
    )

    auxiliary = PoolName.MAMBA if kind == "recurrent" else PoolName.SWA
    linker.layout.pools[auxiliary] = SimpleNamespace(
        group_id=1, kind=kind, window=128 if kind == "window" else 0
    )
    linker.recovery = RecoveryContract(
        linker.namespace, 64, [(0, "attention", 0), (1, kind, 128 if kind == "window" else 0)]
    )
    linker._origins["req"] = 64
    keys = ["a", "b", "c", "d"]
    transfers = [SimpleNamespace(name=name, keys=keys) for name in linker.layout.pools]
    linker.client.query_candidates.side_effect = [
        QueryLoading(),
        QueryCandidates([0, 1]),
        QueryCandidates([0, 1, 2]),
    ]
    assert linker.lookup("req", transfers) == []
    assert linker.lookup("req", transfers) == [1, 2]
    assert linker._lookups["req"].boundaries == (128, 192)
    linker.client.read_recovery.assert_not_called()
    linker.client.query_prefetch.assert_not_called()
    assert all(
        entry.args[1] == BlockHashes(linker._hashes(keys))
        for entry in linker.client.query_candidates.call_args_list
    )
    positions = [1] if kind == "recurrent" else [0, 1]
    linker.client.read_recovery.side_effect = [
        QueryReady(2, b"attention", [0, 1]),
        QueryLoading(),
        QueryReady(len(positions), b"state", positions),
    ]
    assert linker.prepare_recovery("req", 192) == 0
    linker.client.release.assert_not_called()
    assert linker.prepare_recovery("req", 192) == 1
    assert [entry.args[-1] for entry in linker.client.read_recovery.call_args_list] == [0, 1, 1]
    linker.cancel_query("req")
    assert sorted(entry.args[0] for entry in linker.client.release.call_args_list) == [
        b"attention",
        b"state",
    ]


def test_invalid_hybrid_candidates_never_materialize_state(linker):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    from orbitkv import QueryCandidates, RecoveryContract

    linker.layout.pools[PoolName.MAMBA] = SimpleNamespace(group_id=1, kind="recurrent", window=0)
    linker.recovery = RecoveryContract(
        linker.namespace, 64, [(0, "attention", 0), (1, "recurrent", 0)]
    )
    linker.client.query_candidates.side_effect = [QueryCandidates([0]), QueryCandidates([2])]
    with pytest.raises(ValueError, match="page ends"):
        linker.lookup(
            "req", [SimpleNamespace(name=name, keys=["a", "b"]) for name in linker.layout.pools]
        )
    assert not linker._lookups and not linker._pending_queries
    linker.client.read_recovery.assert_not_called()
    linker.client.release.assert_not_called()


def test_changed_keys_cancel_old_query_and_deadline_stops_restarting_io(linker):
    from orbitkv import QueryLoading

    linker.client.query_prefetch.return_value = QueryLoading()
    assert linker.lookup("req", transfer(["old"])) == []
    assert linker.lookup("req", transfer(["new"])) == []
    linker.client.cancel_query.assert_called_once_with("admission", "req", group_id=0)
    linker._QUERY_WAIT_SECONDS = 0
    assert linker.lookup("req", transfer(["new"])) == []
    assert linker.query_state("req") == 2
    assert linker.lookup("req", transfer(["new"])) == []
    assert linker.client.query_prefetch.call_count == 2
    assert linker.client.cancel_query.call_count == 2
    linker.cancel_queued_load("req")
    assert linker.query_state("req") == 0


@pytest.mark.parametrize("observation", [False, OSError("invalid peer descriptor")])
def test_pending_peer_failure_preserves_query_ownership(linker, observation):
    from orbitkv import QueryLoading

    linker.client.query_prefetch.return_value = QueryLoading()
    assert linker.lookup("req", transfer(["pending"])) == []
    lookup = linker._lookups["req"]
    linker._QUERY_WAIT_SECONDS = 0
    if isinstance(observation, Exception):
        linker.client.peer_alive.side_effect = observation
    else:
        linker.client.peer_alive.return_value = observation
    with pytest.raises((RuntimeError, OSError), match="exited|descriptor"):
        linker.query_state("req")
    assert linker._lookups["req"] is lookup
    assert "req" in linker._pending_queries
    assert "req" not in linker._expired_queries
    linker.client.release.assert_not_called()
    linker.client.cancel_query.assert_not_called()
    linker.client.health.assert_not_called()


def test_ready_lookup_reuses_boundaries_until_keys_or_origin_change(linker):
    from orbitkv import QueryReady

    linker.client.query_prefetch.return_value = QueryReady(2, b"lease")
    linker.recovery = MagicMock(wraps=linker.recovery)
    for _ in range(3):
        assert linker.lookup("req", transfer(["a", "b"])) == [1, 2]
        assert linker.query_state("req") == 0
    assert linker.recovery.restorable_boundaries.call_count == 1
    assert linker.client.query_prefetch.call_count == 1
    linker.client.peer_alive.assert_not_called()

    linker._origins["req"] = 64
    assert linker.lookup("req", transfer(["a", "b"])) == [1, 2]
    assert linker.recovery.restorable_boundaries.call_count == 2
    assert linker.lookup("req", transfer(["c", "d"])) == [1, 2]
    assert linker.recovery.restorable_boundaries.call_count == 3
    assert linker.client.cancel_query.call_count == 2
    assert linker.client.release.call_count == 2


@pytest.mark.parametrize(
    "tokens,prefix,logprob_start,pending,wait",
    [
        (256, 128, -1, True, True),
        (256, 192, -1, True, False),
        (257, 192, -1, True, True),
        (256, 128, 128, True, False),
        (256, 192, -1, False, False),
    ],
)
def test_resident_prefix_retires_unused_external_query(
    linker, tokens, prefix, logprob_start, pending, wait
):
    from orbitkv import QueryLoading, QueryReady
    from orbitkv.sglang.admission import admit_request

    req = request("shared", tokens)
    req.prefix_indices = list(range(prefix))
    req.return_logprob = logprob_start >= 0
    req.logprob_start_len = logprob_start
    linker.client.query_prefetch.return_value = (
        QueryLoading() if pending else QueryReady(3, b"lease")
    )
    linker.lookup(req.rid, transfer(["a", "b", "c"]))
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker, needs_rank_consensus=False),
        _all_reduce_attn_groups=MagicMock(),
    )
    original = MagicMock()
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    assert original.call_count == int(not wait)
    assert linker.query_state(req.rid) == int(wait)
    if not wait:
        assert req.rid not in linker._lookups
        if pending:
            linker.client.cancel_query.assert_called_once_with("admission", req.rid, group_id=0)
        else:
            linker.client.release.assert_called_once_with(b"lease")


def test_admission_expires_pending_work_without_another_lookup(linker):
    from orbitkv import QueryLoading
    from orbitkv.sglang.admission import admit_request

    req = request("unpolled")
    linker._origins[req.rid] = 0
    linker.client.query_prefetch.return_value = QueryLoading()
    linker.lookup(req.rid, transfer(["a", "b", "c"]))
    linker._QUERY_WAIT_SECONDS = 0
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker, needs_rank_consensus=False),
        _all_reduce_attn_groups=MagicMock(),
    )
    original = MagicMock()
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    original.assert_called_once()
    linker.client.cancel_query.assert_called_once_with("admission", req.rid, group_id=0)
    assert linker.query_state(req.rid) == 2
    assert linker.lookup(req.rid, transfer(["a", "b", "c"])) == []
    linker.client.query_prefetch.assert_called_once()


@pytest.mark.parametrize("peer_state", [1, 2])
def test_attention_ranks_share_wait_and_expiration_decisions(linker, peer_state):
    from orbitkv import QueryReady
    from orbitkv.sglang.admission import admit_request

    linker.client.query_prefetch.return_value = QueryReady(1, b"lease")
    linker.lookup("req", transfer(["key"]))
    cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=linker, needs_rank_consensus=True),
        _all_reduce_attn_groups=lambda state, op: state.fill_(peer_state),
    )
    original = MagicMock()
    req = request("req")
    admit_request(original, SimpleNamespace(tree_cache=cache), req)
    if peer_state == 1:
        original.assert_not_called()
        linker.client.release.assert_not_called()
    else:
        original.assert_called_once()
        linker.client.release.assert_called_once_with(b"lease")
        assert linker.query_state("req") == 2


@pytest.mark.parametrize("resident_tokens", [0, 64])
def test_decode_only_promises_resident_pages_and_never_prepares_external_loads(
    linker, monkeypatch, resident_tokens
):
    import torch
    from sglang.srt.disaggregation.decode_hicache_mixin import (
        DecodeHiCachePreallocMixin,
    )
    from sglang.srt.mem_cache.base_prefix_cache import MatchResult
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from orbitkv.sglang.admission import enqueue_request
    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    wrapper = object.__new__(RecoveryLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.restore_from_store = False
    wrapper.hit_markers = {}
    req = request("req")
    resident = MatchResult(torch.arange(resident_tokens), 7, 7, 7)
    matched = wrapper.match(RadixKey(req.origin_input_ids), req, resident)
    assert matched is resident
    prealloc = SimpleNamespace(scheduler=SimpleNamespace(enable_decode_hicache=False))
    promised = DecodeHiCachePreallocMixin._build_decode_prefix_match(prealloc, req, matched)
    assert promised.decode_prefix_len == resident_tokens
    assert promised.last_device_node == 7
    assert not promised.needs_local_restore
    assert not wrapper.hit_markers and not linker._lookups

    monkeypatch.setenv("ORBITKV_PREPARE_REQUESTS", "1")
    scheduler = SimpleNamespace(tree_cache=SimpleNamespace(linker=wrapper), waiting_queue=[])

    def accepted(scheduler, req):
        scheduler.waiting_queue.append(req)

    enqueue_request(accepted, scheduler, req)
    assert scheduler.waiting_queue == [req]
    assert linker.client.mock_calls == []


@pytest.mark.parametrize("mode", ["null", "prefill", "decode"])
@pytest.mark.parametrize(
    "tp_size,cp_size,attn_size,consensus",
    [
        (1, None, None, False),
        (1, 1, 1, False),
        (1, 2, 1, True),
        (1, 1, 2, True),
        (2, None, None, True),
    ],
    ids=["local", "singleton-groups", "attention-cp", "attention-tp", "fallback-tp"],
)
def test_factory_selects_restore_owner_from_pinned_runtime_role(
    monkeypatch, mode, tp_size, cp_size, attn_size, consensus
):
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    from orbitkv.sglang.plugin import create_cache

    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode=mode,
            disaggregation_transfer_backend="mooncake",
            disaggregation_decode_retraction_backup="none",
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(enable_unified_cache_external_linker=True),
    )
    cache = MagicMock()
    cache.tree_components = (ComponentType.FULL,)
    cache.components = {ComponentType.FULL: MagicMock()}
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache", lambda _: cache
    )
    linker = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.linker.OrbitKVLinker", lambda *args, **kwargs: linker)
    params = SimpleNamespace(
        is_eagle=False,
        mtp_draft_device_pools=None,
        component_registry_override=None,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=MagicMock(),
    )
    ctx = SimpleNamespace(
        disable_radix_cache=False,
        enable_hierarchical_cache=False,
        is_dsa=False,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        params=params,
        server_args=object(),
        tp_worker=MagicMock(),
    )
    import torch

    cache.tp_world_size = tp_size
    cache.attn_cp_group = cp_size
    cache.attn_tp_group = attn_size
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *, group: group)
    assert create_cache(ctx) is cache
    assert cache.linker.needs_rank_consensus == consensus
    assert cache.linker.restore_from_store == (mode != "decode")
    assert cache.linker.cache_linker is linker
    assert cache.write_through_threshold == 1


@pytest.mark.parametrize(
    "consensus,local,peer,ready,expired",
    [
        (False, 1, None, True, False),
        (False, 0, None, False, False),
        (False, -1, None, False, True),
        (True, 1, 0, False, False),
        (True, 1, -1, False, True),
        (True, -1, 1, False, True),
    ],
    ids=[
        "local-ready",
        "local-pending",
        "local-stale",
        "peer-pending",
        "peer-stale",
        "local-stale-peer-ready",
    ],
)
def test_hybrid_match_requires_rank_common_materialized_state(
    linker, monkeypatch, consensus, local, peer, ready, expired
):
    import torch
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinkerWrapper

    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    req = request("req")
    resident = SimpleNamespace(device_indices=torch.arange(64))
    matched = object()
    wrapper = object.__new__(RecoveryLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.restore_from_store = True
    wrapper.needs_rank_consensus = consensus
    wrapper.hit_markers = {req.rid: SimpleNamespace(device_hit_len=64, tail_hashes=["one", "two"])}
    linker.layout.pools["auxiliary"] = object()
    linker.prepare_recovery = MagicMock(return_value=local)
    linker.expire_query = MagicMock()
    reduced = MagicMock(side_effect=lambda state, op: state.fill_(min(int(state.item()), peer)))
    wrapper.cache = SimpleNamespace(page_size=64, _all_reduce_attn_groups=reduced)
    monkeypatch.setattr(UnifiedCacheLinkerWrapper, "match", lambda *args: matched)
    if not consensus:
        monkeypatch.setattr(
            torch, "tensor", lambda *args, **kwargs: pytest.fail("Unexpected rank tensor")
        )
    assert wrapper.match(object(), req, resident) is (matched if ready else resident)
    linker.prepare_recovery.assert_called_once_with(req.rid, 192)
    assert reduced.call_count == int(consensus)
    assert linker.expire_query.call_count == int(expired)
    assert (req.rid in wrapper.hit_markers) == ready
    assert req.rid not in linker._origins


@pytest.mark.parametrize("rejected", [False, True], ids=["success", "stale-registration"])
def test_offload_keeps_producer_registration_and_reports_rejection(linker, monkeypatch, rejected):
    import queue

    from sglang.srt.mem_cache.hicache_storage import PoolName

    original = object()
    linker.registration = original
    linker._offload_queue = queue.Queue()
    linker._completed_offloads = queue.Queue()
    linker.layout.pools = {
        PoolName.KV: SimpleNamespace(
            kind="attention", layer_names=["kv:0"], block_ids=lambda indices, count: [2]
        )
    }
    ready = MagicMock()
    monkeypatch.setattr("orbitkv.sglang.linker.torch.cuda.Event", lambda: ready)
    transfers = [SimpleNamespace(name=PoolName.KV, keys=["key"], device_indices=[2])]
    expected = [("kv:0", [2], linker._hashes(["key"]))]
    if rejected:
        linker.client.save.side_effect = ValueError("GPU registration is no longer active")
    else:
        linker.client.save.return_value = True, ""

    assert linker.offload(transfers)
    linker.registration = object()
    linker._offload_queue.put(None)
    linker._offload_worker()

    linker.client.save.assert_called_once_with(original, expected)
    ready.synchronize.assert_called_once()
    assert linker.pop_completed_offload() is not rejected
    assert linker._offload_queue.unfinished_tasks == 0


@pytest.mark.parametrize("failure", [None, "unregister", "close"])
def test_close_retires_only_the_linkers_own_registration_after_normal_cleanup(linker, failure):
    import queue

    original = object()
    queued = original
    linker.registration = original
    linker.reset = MagicMock()
    linker._load_queue = queue.Queue()
    linker._offload_queue = queue.Queue()
    linker._load_thread = MagicMock()
    linker._offload_thread = MagicMock()
    linker.client.unregister_context.return_value = failure != "unregister", "controlled cleanup"
    if failure == "close":
        linker.client.close.side_effect = RuntimeError("controlled close failure")
        with pytest.raises(RuntimeError, match="controlled close failure"):
            linker.close()
    else:
        linker.close()

    assert linker.registration is (None if failure is None else original)
    assert queued is original
