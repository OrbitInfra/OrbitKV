"""Installed release boundaries with controlled Manager replies, no CUDA work.

These controls prove adapter decisions/retention, not physical DMA or allocator
generation safety. Process restart and page-reuse evidence requires the GPU gate.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.integration.test_sglang_direct_transfer import load_linker  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "changed", ["revision", "layout", "computation", "representation", "scope"]
)
def test_same_token_hash_cannot_cross_engine_identity(engine, changed, monkeypatch):
    monkeypatch.delenv("ORBITKV_MODEL_FINGERPRINT", raising=False)
    monkeypatch.delenv("ORBITKV_CACHE_SCOPE", raising=False)
    token_hash = b"same-token-hash".ljust(32, b"\0")
    if engine == "vllm":
        from orbitkv.vllm.config import derive_namespace
        from tests.unit.test_derive_namespace import _make_vllm_config

        config = _make_vllm_config()
        config.model_config.model = "fixture/model"

        def namespace():
            return derive_namespace(config, tp_size=1)

        changes = {
            "revision": (config.model_config, "revision", "b" * 40),
            "layout": (config.cache_config, "kv_cache_layout", "LBHNC"),
            "computation": (config.model_config, "dtype", "float32"),
            "representation": (config.cache_config, "cache_dtype", "auto"),
        }
    else:
        from sglang.srt.server_args import ServerArgs

        from orbitkv.sglang.config import derive_namespace

        # Only configuration inputs are controlled; the runtime module and the
        # released ServerArgs class remain real, without model/GPU initialization.
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_rank=0, tp_size=1),
        )
        config = ServerArgs(
            model_path="fixture/model", revision="a" * 40, dtype="bfloat16", kv_cache_dtype="auto"
        )
        layout = SimpleNamespace(page_size=64, pools={})

        def namespace():
            return derive_namespace(
                config,
                SimpleNamespace(pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1),
                layout,
                lora_config=SimpleNamespace(enable_lora=False),
                static_loras=(),
            )

        changes = {
            "revision": (config, "revision", "b" * 40),
            "layout": (layout, "page_size", 32),
            "computation": (config, "dtype", "float32"),
            "representation": (config, "kv_cache_dtype", "fp8_e4m3"),
        }
    original = namespace(), token_hash
    assert (namespace(), token_hash) == original
    if changed == "scope":
        monkeypatch.setenv("ORBITKV_CACHE_SCOPE", "other-tenant")
    else:
        obj, attribute, value = changes[changed]
        setattr(obj, attribute, value)
    assert (namespace(), token_hash) != original


@pytest.mark.parametrize("engine", ["vllm"])
@pytest.mark.parametrize("outcome", ["abort", "queue-cancel", "deadline-miss", "manager-lost"])
def test_released_vllm_pending_query_never_becomes_a_stale_hit(engine, outcome):
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus

    from orbitkv import QueryLoading, QueryReady
    from orbitkv.vllm.config import ConnectorContext
    from orbitkv.vllm.scheduler import SchedulerAdapter

    client = MagicMock()
    client.query_prefetch.return_value = QueryLoading()
    scheduler = SchedulerAdapter(ConnectorContext("pending", "namespace", 16, 1, 1, 0, 0, client))
    request = Request("pending", [1] * 32, SamplingParams(max_tokens=1), None)
    request.block_hashes = [b"a" * 32, b"b" * 32]
    scheduler.on_new_request(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert not scheduler._pending_load_intents
    try:
        if outcome in ("abort", "queue-cancel"):
            request.status = RequestStatus.FINISHED_ABORTED
            assert scheduler.request_finished(request, ()) == (False, None)
            client.cancel_query.assert_called()
            assert not scheduler._pending_query_probes
        elif outcome == "deadline-miss":
            # Native query deadlines may return a miss, never a fabricated hit.
            client.query_prefetch.return_value = QueryReady(0, b"")
            assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        else:
            client.query_prefetch.side_effect = ConnectionError("Manager incarnation ended")
            with pytest.raises(ConnectionError, match="incarnation"):
                scheduler.get_num_new_matched_tokens(request, 0)
        client.start_restore.assert_not_called()
        client.release.assert_not_called()  # No lease was handed to this requester.
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("engine", ["sglang"])
def test_partial_full_swa_load_does_not_publish_a_complete_boundary(engine, load_linker):  # noqa: F811
    from orbitkv.sglang.events import _LayerDoneCounter
    from orbitkv.sglang.linker import _Load

    linker = load_linker
    linker.layout.num_layers = 2
    linker.layout.pools["swa"] = SimpleNamespace(
        layer_names=["swa:1"], entry=SimpleNamespace(layer_mapping={1: 0})
    )
    linker.layer_done_counter = _LayerDoneCounter(linker.layout)
    linker._queued_loads = {
        "r": _Load("r", (("kv", b"full-lease", (1,)), ("swa", b"window-lease", (2,))))
    }
    linker.client.wait_restore.return_value = SimpleNamespace(
        done=True, success=False, message="Full landed, SWA failed"
    )
    index = linker.start_layer_wise_loading()
    linker._load_queue.put(None)
    worker = threading.Thread(target=linker._load_worker, daemon=True)
    worker.start()
    try:
        linker.layer_done_counter.set_consumer(index)
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(linker.client.start_restore.call_args.args[4]) == 2
        linker.client.release.assert_not_called()
        with pytest.raises(RuntimeError, match="pages remain held"):
            linker.finish_layer_wise_loading(index)
        assert index in linker._load_completions
    finally:
        # Do not synthesize a drain receipt. Only stop this test's CPU worker.
        worker.join(timeout=5)
