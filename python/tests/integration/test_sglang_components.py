"""Released component ownership on CPU; only CUDA and Manager receipts are seams."""

from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.test_sglang_admission import linker as base_linker  # noqa: F401
from tests.integration.test_sglang_admission import request

pytestmark = pytest.mark.integration


@pytest.fixture(params=["mamba", "mla"])
def component_cache(request, monkeypatch):
    import torch
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        HybridReqToTokenPool,
        MLATokenToKVPool,
        ReqToTokenPool,
    )
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    from orbitkv.sglang.layout import GpuLayout, GpuPool

    device = "cuda" if request.node.get_closest_marker("gpu") else "cpu"
    args = ServerArgs(model_path="dummy", page_size=1)
    args._mamba_cache_chunk_size = 1
    set_global_server_args_for_scheduler(args)
    components = (ComponentType.FULL,)
    if request.param == "mamba":
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=32,
            n_groups=1,
            num_heads=2,
            head_dim=16,
            state_size=8,
            conv_kernel=4,
        )
        req_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=16,
            mamba_spec_state_size=4,
            max_context_len=64,
            device=device,
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[1]),
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=False,
        )
        kv_pool = HybridLinearKVPool(
            size=64,
            dtype=torch.bfloat16,
            page_size=1,
            head_num=2,
            head_dim=16,
            full_attention_layer_ids=[0],
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_pool.mamba_pool,
        )
        components += (ComponentType.MAMBA,)
    else:
        req_pool = ReqToTokenPool(
            size=4, max_context_len=64, device=device, enable_memory_saver=False
        )
        kv_pool = MLATokenToKVPool(
            size=64,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            layer_num=2,
            device=device,
            enable_memory_saver=False,
            start_layer=0,
            end_layer=2,
        )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        disable=False,
        tree_components=components,
    )
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        cache = UnifiedRadixCache(params)

    # CPU buffers stand in for registered CUDA allocations. Keep the released
    # pool/index implementation; device registration and bytes have a GPU gate.
    def compile_cpu(cls, group_id, kind, window, entry):
        return cls(
            group_id,
            kind,
            window,
            entry,
            [f"{entry.name}:{i}" for i in range(len(entry.kv_buffer))],
            [t.shape[0] // entry._row_span for t in entry.kv_buffer],
            [t.stride(0) * t.element_size() * entry._row_span for t in entry.kv_buffer],
        )

    if device == "cpu":
        monkeypatch.setattr(GpuPool, "compile", classmethod(compile_cpu))
    layout = GpuLayout.from_pool(1, kv_pool, req_pool, set(components))
    return cache, params, layout


@pytest.fixture
def component_linker(component_cache, base_linker, monkeypatch):  # noqa: F811
    import queue

    import torch

    from orbitkv import RecoveryContract
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.recovery import RecoveryLinkerWrapper

    cache, params, layout = component_cache
    linker = base_linker
    linker.layout = layout
    linker.pools = layout.pools
    linker.page_size = 1
    linker.registration = object()
    linker.device_id = 0
    linker._torch_device = "cpu"
    linker.recovery = RecoveryContract(
        linker.namespace,
        1,
        [(p.group_id, p.kind, p.window) for p in layout.pools.values()],
    )
    linker.forward_stream = None
    linker._load_queue, linker._offload_queue = queue.Queue(), queue.Queue()
    linker._completed_offloads = queue.Queue()
    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=17))
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.device_module.Event", MagicMock
    )
    linker.layer_done_counter = _LayerDoneCounter(layout)
    wrapper = RecoveryLinkerWrapper(cache, linker, restore_from_store=True)
    cache.linker = wrapper
    return linker, cache, params, wrapper


def test_component_layout_identity(component_cache, monkeypatch):
    from orbitkv.sglang.config import derive_namespace

    cache, params, layout = component_cache
    monkeypatch.setenv("ORBITKV_MODEL_FINGERPRINT", "a" * 64)
    parallel = SimpleNamespace(tp_rank=0, tp_size=2)
    monkeypatch.setattr("sglang.srt.runtime_context.get_parallel", lambda: parallel)
    args = SimpleNamespace(
        dtype="bfloat16",
        attention_backend="triton",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        model_path="fixture",
        revision=None,
        tokenizer_path=None,
    )

    def namespace():
        return derive_namespace(
            args, params, layout, lora_config=SimpleNamespace(enable_lora=False), static_loras=()
        )

    original = namespace()
    assert namespace() == original
    parallel.tp_rank = 1
    assert namespace() != original, "MLA remains per rank; no shared-entry writers"
    parallel.tp_rank = 0
    state_pool = list(layout.pools.values())[-1].entry.device_pool
    if len(layout.pools) > 1:
        state_pool.conv_window_axis = 0
    else:
        # Same total physical width, different latent/RoPE partition.
        state_pool.kv_lora_rank += 1
        state_pool.qk_rope_head_dim -= 1
    assert namespace() != original
    if len(layout.pools) > 1:
        state_pool.conv_window_axis = -1
    else:
        state_pool.kv_lora_rank -= 1
        state_pool.qk_rope_head_dim += 1
    assert namespace() == original
    tensor = list(layout.pools.values())[-1].entry.kv_buffer[0]
    list(layout.pools.values())[-1].entry.kv_buffer[0] = tensor.to(
        dtype=__import__("torch").float32
    )
    assert namespace() != original


@pytest.mark.parametrize("component_cache", ["mamba"], indirect=True)
@pytest.mark.parametrize(
    "bad",
    [
        "moving",
        "int8",
        "short-conv",
        "ngram",
        "replay",
        "spec-replay",
        "draft",
        "side-state",
        "sibling",
    ],
)
def test_recurrent_rejections(component_cache, bad):
    import torch
    from sglang.srt.mem_cache.memory_pool import MambaPool

    from orbitkv.sglang.layout import validate_recurrent_pool

    _, params, _ = component_cache
    pool = params.req_to_token_pool
    validate_recurrent_pool(pool)
    if bad == "moving":

        class MovingPool(MambaPool):
            pass

        pool.mamba_pool = object.__new__(MovingPool)
    elif bad == "int8":
        pool.mamba_ckpt_pool = object()
    elif bad in {"short-conv", "ngram"}:
        setattr(
            pool,
            "short_conv_pool" if bad == "short-conv" else "ngram_pool",
            SimpleNamespace(enabled=True),
        )
    elif bad == "replay":
        pool.mamba_pool.enable_linear_replayssm = True
    elif bad == "spec-replay":
        pool.mamba_pool.enable_linear_replayssm_spec = True
    elif bad == "draft":
        state = pool.mamba_pool.mamba_cache
        pool.mamba_pool.mamba_cache = MambaPool.SpeculativeState(
            conv=state.conv,
            temporal=state.temporal,
            intermediate_ssm=None,
            intermediate_conv_window=[],
        )
    elif bad == "side-state":
        pool.mamba_pool.mamba_cache = replace(
            pool.mamba_pool.mamba_cache, replayssm_k=torch.empty(1)
        )
    else:
        pool.mamba_pool._slot_siblings = (object(),)
    with pytest.raises(ValueError, match="OrbitKV cannot restore"):
        validate_recurrent_pool(pool)


def match_boundary(linker, cache, outcome="complete"):
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from orbitkv import QueryCandidates, QueryReady

    req = request("req", 5)
    key = RadixKey(array("q", [1, 2, 3, 4]))
    if len(linker.pools) > 1:
        positions = {
            "complete": ([0, 1, 2, 3], [3]),
            "missing": ([0, 1, 2, 3], []),
            "mismatch": ([0, 1, 2], [3]),
        }[outcome]
        linker.client.query_candidates.side_effect = [QueryCandidates(list(p)) for p in positions]
        linker.client.read_recovery.side_effect = [
            QueryReady(4, b"kv", [0, 1, 2, 3]),
            QueryReady(1, b"mamba", [3]),
        ]
    else:
        linker.client.query_prefetch.return_value = QueryReady(
            4 if outcome == "complete" else 0, b"kv"
        )
    match = cache.match_prefix(MatchPrefixParams(key=key, req=req))
    req.last_node = match.best_match_node
    return req, match


@pytest.mark.parametrize("outcome", ["complete", "missing", "mismatch"])
def test_atomic_component_boundary(component_linker, outcome):
    linker, cache, params, wrapper = component_linker
    mla_boundary_change = outcome == "mismatch" and len(linker.pools) == 1
    req, matched = match_boundary(linker, cache, "complete" if mla_boundary_change else outcome)
    if mla_boundary_change:
        free = params.token_to_kv_pool_allocator.available_size()
        wrapper.hit_markers[req.rid].tail_hashes.append("outside-leased-boundary")
        with pytest.raises(ValueError, match="unproved recovery boundary"):
            wrapper.load_back(req)
        assert params.token_to_kv_pool_allocator.available_size() == free
        assert not linker._queued_loads
        return
    if outcome != "complete":
        assert matched.host_hit_length == 0
        assert not wrapper.hit_markers
        assert not linker._queued_loads
        return
    assert matched.host_hit_length == 4
    free = params.token_to_kv_pool_allocator.available_size()
    if len(linker.pools) > 1:
        old_slot = params.req_to_token_pool.mamba_allocator.alloc(1)[0]
        req.kv.mamba_pool_idx = old_slot
        req.kv.mamba_cow_src_index = old_slot
        req.kv.mamba_needs_clear = True
        slots = params.req_to_token_pool.mamba_allocator.available_size()
    loaded, _ = wrapper.load_back(req)
    assert loaded == 4
    assert params.token_to_kv_pool_allocator.available_size() == free - 4
    assert len(linker._queued_loads[req.rid].groups) == len(linker.pools)
    if len(linker.pools) > 1:
        assert not req.kv.mamba_needs_clear and req.kv.mamba_cow_src_index is None
        assert req.kv.holds_mamba and req.kv.mamba_pool_idx != old_slot
        assert params.req_to_token_pool.mamba_allocator.available_size() == slots
    # Cancellation is still a non-submission receipt, for every component lease.
    wrapper.release_request(req.rid)
    assert not linker._queued_loads
    assert {call.args[0] for call in linker.client.release.call_args_list} == (
        {b"kv", b"mamba"} if len(linker.pools) > 1 else {b"kv"}
    )


@pytest.mark.parametrize("terminal", ["abort", "retract"])
def test_started_component_load_holds_slots_through_drain(component_linker, terminal, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from sglang.srt.mem_cache.base_prefix_cache import (
        CacheRequestHandle,
        CacheRequestOutcome,
    )

    linker, cache, params, wrapper = component_linker
    req, _ = match_boundary(linker, cache)
    wrapper.load_back(req)
    row = params.req_to_token_pool.alloc([req])[0]
    params.req_to_token_pool.req_to_token[row, :4] = wrapper.private_load_indices[req.rid]
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 4
    free = params.token_to_kv_pool_allocator.available_size()
    mamba_free = (
        params.req_to_token_pool.mamba_allocator.available_size() if len(linker.pools) > 1 else None
    )
    submitted, drain = threading.Event(), threading.Event()

    def wait(_, **kwargs):
        submitted.set()
        assert drain.wait(5)
        return SimpleNamespace(success=True)

    linker.client.wait_restore.side_effect = wait
    index = wrapper.start_layer_wise_loading()
    linker._load_queue.put(None)
    worker = threading.Thread(target=linker._load_worker, daemon=True)
    worker.start()
    try:
        linker.layer_done_counter.set_consumer(index)
        assert submitted.wait(2)
        if terminal == "abort":
            cache.finish(CacheRequestHandle(req.rid, 0), CacheRequestOutcome.ABORT)
        else:
            # Decode retraction is reached only after Scheduler.run_batch's
            # synchronous finish_external_linker_loads barrier, tested below.
            assert not linker.cancel_queued_load(req.rid)
        assert params.token_to_kv_pool_allocator.available_size() == free
        if mamba_free is not None:
            assert params.req_to_token_pool.mamba_allocator.available_size() == mamba_free
        linker.client.release.assert_not_called()
        with ThreadPoolExecutor(max_workers=1) as executor:

            def complete():
                if terminal == "retract":
                    return run_then_retract(cache, params, req, monkeypatch)
                return cache.finish_external_linker_loads([req])

            completed = executor.submit(complete)
            try:
                with pytest.raises(TimeoutError):
                    completed.result(timeout=0.05)
            finally:
                drain.set()
            assert completed.result(timeout=5) == []
        assert not wrapper.private_load_indices
        assert not linker._load_completions
        if terminal == "retract":
            assert params.token_to_kv_pool_allocator.available_size() == free + 4
            if mamba_free is not None:
                assert params.req_to_token_pool.mamba_allocator.available_size() == mamba_free + 1
    finally:
        drain.set()
        worker.join(5)
    assert not worker.is_alive()


def test_component_save_abort_retains_tree_lock(component_linker):
    import threading

    from sglang.srt.mem_cache.base_prefix_cache import (
        CacheRequestHandle,
        CacheRequestOutcome,
        InsertParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    linker, cache, params, wrapper = component_linker
    submitted, drain = threading.Event(), threading.Event()

    def wait(_, **kwargs):
        submitted.set()
        assert drain.wait(5)
        return SimpleNamespace(done=True, success=True, message="")

    linker.client.wait_save.side_effect = wait
    mamba = params.req_to_token_pool.mamba_allocator.alloc(1) if len(linker.pools) > 1 else None
    inserted = cache.insert(
        InsertParams(
            key=RadixKey(array("q", [1, 2, 3, 4])),
            value=params.token_to_kv_pool_allocator.alloc(4),
            mamba_value=mamba,
        )
    )
    wrapper.offload_nodes([inserted.last_device_node])
    assert wrapper.pending_offloads
    locked = list(wrapper.pending_offloads)
    linker._offload_queue.put(None)
    worker = threading.Thread(target=linker._offload_worker, daemon=True)
    worker.start()
    try:
        assert submitted.wait(2)
        cache.finish(CacheRequestHandle("abort", 0), CacheRequestOutcome.ABORT)
        cache.check_hicache_events()
        assert wrapper.pending_offloads == locked
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        cache.evict(EvictParams(num_tokens=64, mamba_num=16))
        assert params.token_to_kv_pool_allocator.available_size() == 60
        if mamba is not None:
            assert params.req_to_token_pool.mamba_allocator.available_size() == 15
        drain.set()
        worker.join(5)
        cache.check_hicache_events()
        assert not wrapper.pending_offloads
        cache.evict(EvictParams(num_tokens=64, mamba_num=16))
        assert params.token_to_kv_pool_allocator.available_size() == 64
        if mamba is not None:
            assert params.req_to_token_pool.mamba_allocator.available_size() == 16
    finally:
        drain.set()
        worker.join(5)
    assert not worker.is_alive()


def test_released_mla_writer_selection_and_broadcast_rejection(component_cache):
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext

    from orbitkv.sglang.plugin import create_cache

    _, params, _ = component_cache
    for rank in range(4):
        storage = HiCacheStorageConfig(
            tp_rank=rank,
            tp_size=4,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="fixture",
        )
        assert storage.is_storage_writer == (rank == 0)
    ctx = TreeCacheBuildContext(
        server_args=SimpleNamespace(enable_linker_mla_dedup=True),
        params=params,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        tp_worker=None,
        model_config=None,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
    )
    with pytest.raises(ValueError, match="per-rank storage"):
        create_cache(ctx)


def run_then_retract(cache, params, req, monkeypatch):
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.schedule_batch import ScheduleBatch, release_req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    scheduler = object.__new__(Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.metrics_reporter = MagicMock()
    scheduler.forward_ct = 0
    scheduler._sched_idled = False
    scheduler.scripted_scheduler_hook = None
    scheduler.profiler_manager = MagicMock()
    scheduler.forward_sleep_time = None
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.is_generation = True
    scheduler.enable_overlap = scheduler.enable_pdmux = False
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.future_map = MagicMock()
    scheduler.model_worker = MagicMock()
    scheduler.model_worker.forward_batch_generation.return_value = SimpleNamespace(
        has_sampled_token_ids=False
    )
    scheduler.update_cache_from_scheduler = MagicMock()
    scheduler._copy_auxiliary_output_to_cpu = MagicMock()
    scheduler.enable_unified_cache_external_linker = True
    scheduler.tree_cache = cache
    scheduler.chunked_req = None
    scheduler.attn_dp_enabled = False
    # Model compute/input resolution are the hardware seam. The public
    # Scheduler.run_batch completion barrier and release_req remain real.
    monkeypatch.setattr("sglang.srt.managers.scheduler.resolve_forward_inputs", MagicMock())
    batch = object.__new__(ScheduleBatch)
    batch.forward_mode = ForwardMode.EXTEND
    batch.extend_num_tokens = 0
    batch.spec_algorithm = scheduler.spec_algorithm
    batch.return_logprob = batch.return_hidden_states = False
    batch.reqs = [req]
    scheduler.run_batch(batch)
    assert release_req(
        req=req,
        remaing_req_count=0,
        req_to_token_pool=params.req_to_token_pool,
        token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
        tree_cache=cache,
        hisparse_coordinator=None,
        offload_kv=False,
    )
    return []


def test_component_factory_and_layer_barriers(component_linker, monkeypatch):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    from orbitkv.sglang.events import initialize_layer_counter
    from orbitkv.sglang.plugin import create_cache

    linker, _, params, _ = component_linker
    worker = SimpleNamespace(
        get_memory_pool=lambda: (params.req_to_token_pool, params.token_to_kv_pool_allocator),
        sliding_window_size=0,
        register_hicache_layer_transfer_counter=MagicMock(),
        model_runner=SimpleNamespace(forward_stream=None),
    )
    initialize_layer_counter(worker)
    counter = worker.register_hicache_layer_transfer_counter.call_args.args[0]
    assert counter.layout.pools.keys() == linker.pools.keys()
    if PoolName.MAMBA in linker.pools:
        assert params.req_to_token_pool.layer_transfer_counter is counter
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_memory",
        lambda: SimpleNamespace(
            enable_unified_cache_external_linker=True,
            radix_cache_backend="orbitkv",
        ),
    )
    monkeypatch.setattr("orbitkv.sglang.linker.OrbitKVLinker", lambda *a, **kw: linker)
    ctx = TreeCacheBuildContext(
        server_args=SimpleNamespace(enable_linker_mla_dedup=False),
        params=params,
        is_hybrid_swa=False,
        is_hybrid_ssm=PoolName.MAMBA in linker.pools,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        tp_worker=worker,
        model_config=None,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
    )
    cache = create_cache(ctx)
    assert cache.linker.cache_linker is linker
    assert (ComponentType.MAMBA in params.tree_components) == ctx.is_hybrid_ssm
    if ctx.is_hybrid_ssm:
        ctx.is_hybrid_swa = True
        with pytest.raises(ValueError, match="combined lifecycle gate"):
            create_cache(ctx)


@pytest.fixture
def component_fault_server(monkeypatch, tmp_path, request):
    import os

    assert os.environ.get("ORBITKV_FAULT_TESTS") == "1", "requires the test-hooks Manager"
    monkeypatch.setenv("ORBITKV_TEST_FAULTS", str(tmp_path))
    from tests.support.cache_manager import CacheManagerProcess, find_available_port

    server = CacheManagerProcess(
        find_available_port(),
        http_port=find_available_port(),
        ssd_cache_path=tmp_path / "ssd" if request.param == "ssd" else None,
        ssd_backend="uring",
    )
    assert server.start(), server.read_logs()
    try:
        yield server, tmp_path
    finally:
        for pause in tmp_path.glob("*.pause"):
            pause.unlink(missing_ok=True)
        server.stop()


@pytest.mark.gpu
@pytest.mark.parametrize("component_fault_server", ["dram", "ssd"], indirect=True)
def test_native_component_bytes_and_faults(component_cache, component_fault_server, monkeypatch):
    import time
    import uuid
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    import torch
    from sglang.srt.mem_cache.base_prefix_cache import (
        CacheRequestHandle,
        CacheRequestOutcome,
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.linker import OrbitKVLinker
    from orbitkv.sglang.recovery import RecoveryLinkerWrapper
    from tests.support.cache_manager import evict_dram_after_ssd_writes

    cache, params, layout = component_cache
    server, directory = component_fault_server
    params.token_to_kv_pool_allocator.get_kvcache().layer_transfer_counter = _LayerDoneCounter(
        layout
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_lora", lambda: SimpleNamespace(enable_lora=False)
    )
    monkeypatch.setattr(
        "orbitkv.sglang.linker.derive_namespace", lambda *a, **kw: f"component-{uuid.uuid4().hex}"
    )
    monkeypatch.setenv("ORBITKV_SGLANG_ENDPOINT", f"unix://{server.bootstrap_socket}")
    linker = OrbitKVLinker(SimpleNamespace(), params, forward_stream=torch.cuda.current_stream())
    wrapper = cache.linker = RecoveryLinkerWrapper(cache, linker, restore_from_store=True)
    deadline = time.monotonic() + 60

    def until(predicate):
        while not (value := predicate()):
            assert time.monotonic() < deadline, "component fault gate timed out"
            time.sleep(0.01)
        return value

    def arm(name):
        (directory / f"{name}.reached").unlink(missing_ok=True)
        (directory / f"{name}.pause").touch()

    def release(name):
        (directory / f"{name}.pause").unlink(missing_ok=True)

    tensors = [tensor for pool in layout.pools.values() for tensor in pool.entry.kv_buffer]
    for i, tensor in enumerate(tensors):
        tensor.copy_(torch.arange(tensor.numel(), device="cuda").reshape(tensor.shape) % 127 + i)
    expected = [tensor.clone() for tensor in tensors]
    source = params.token_to_kv_pool_allocator.alloc(4)
    mamba = params.req_to_token_pool.mamba_allocator.alloc(1) if len(layout.pools) > 1 else None
    key = RadixKey(array("q", [1, 2, 3, 4]))
    try:
        node = cache.insert(InsertParams(key=key, value=source, mamba_value=mamba)).last_device_node
        arm("publish")
        wrapper.offload_nodes([node])
        until(lambda: (directory / "publish.reached").exists())
        cache.finish(CacheRequestHandle("aborted-producer", 0), CacheRequestOutcome.ABORT)
        cache.check_hicache_events()
        cache.evict(EvictParams(num_tokens=64, mamba_num=16))
        assert wrapper.pending_offloads
        assert params.token_to_kv_pool_allocator.available_size() == 60
        release("publish")
        until(lambda: linker.num_completed_offloads())
        cache.check_hicache_events()
        assert not wrapper.pending_offloads
        cache.evict(EvictParams(num_tokens=64, mamba_num=16))
        if server.ssd_cache_path is not None:
            evict_dram_after_ssd_writes(server.http_port)
        for tensor in tensors:
            tensor.fill_(-1)
        req = request("req", 5)

        def lookup():
            matched = cache.match_prefix(MatchPrefixParams(key=key, req=req))
            req.last_node = matched.best_match_node
            return matched.host_hit_length == 4

        until(lookup)
        wrapper.load_back(req)
        targets = [list(ids) for _, _, ids in linker._queued_loads[req.rid].groups]
        row = params.req_to_token_pool.alloc([req])[0]
        params.req_to_token_pool.req_to_token[row, :4] = wrapper.private_load_indices[req.rid]
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 4
        free = params.token_to_kv_pool_allocator.available_size()
        arm("restore")
        index = wrapper.start_layer_wise_loading()
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                consumer = executor.submit(linker.layer_done_counter.set_consumer, index)
                until(lambda: (directory / "restore.reached").exists())
                cache.finish(CacheRequestHandle(req.rid, 0), CacheRequestOutcome.ABORT)
                completion = executor.submit(cache.finish_external_linker_loads, [req])
                with pytest.raises(TimeoutError):
                    completion.result(0.05)
                assert params.token_to_kv_pool_allocator.available_size() == free
            finally:
                release("restore")
            consumer.result(10)
            assert completion.result(10) == []
        torch.cuda.synchronize()
        index = 0
        for pool, target in zip(layout.pools.values(), targets, strict=True):
            saved = mamba if pool.kind in {"recurrent", "convolution"} else source
            for tensor in pool.entry.kv_buffer:
                assert torch.equal(tensor[target], expected[index][saved])
                mask = torch.ones(tensor.shape[0], dtype=torch.bool, device="cuda")
                mask[target] = False
                assert (tensor[mask] == -1).all(), "Restore changed an unrelated slot"
                index += 1
        # Actual retraction after the native receipt may now reuse every slot.
        run_then_retract(cache, params, req, monkeypatch)
        assert params.token_to_kv_pool_allocator.available_size() == 64
    finally:
        release("publish")
        release("restore")
        linker.close()


@pytest.mark.parametrize("component_cache", ["mamba"], indirect=True)
@pytest.mark.parametrize(
    "last,previous,length,expected",
    [
        (8, 4, 8, (8, 1)),
        (8, 4, 6, (4, 0)),
        (8, 4, 3, None),
    ],
)
def test_finished_checkpoint_never_labels_state_at_a_different_boundary(
    component_cache, last, previous, length, expected
):
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    cache, _, _ = component_cache
    component = next(c for c in cache._components_tuple if c.component_type == ComponentType.MAMBA)
    req = request("checkpoint")
    req.kv.mamba_last_track_seqlen = last
    req.kv.mamba_last_track_idx = 1
    req.kv.mamba_prev_track_seqlen = previous
    assert component._select_finished_checkpoint(req, length) == expected
