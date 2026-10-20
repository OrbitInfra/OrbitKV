"""Register and construct the OrbitKV SGLang RadixCache backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def register() -> None:
    from sglang.srt.mem_cache.registry import register_radix_cache_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .admission import admit_request, enqueue_request
    from .events import initialize_layer_counter

    HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.init_cuda_graphs",
        initialize_layer_counter,
        HookType.BEFORE,
    )
    register_radix_cache_backend("orbitkv", create_cache)
    HookRegistry.register(
        "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
        admit_request,
        HookType.AROUND,
    )
    if any(os.getenv(name) == "1" for name in ("ORBITKV_PREPARE_REQUESTS", "ORBITKV_QUEUE_WARMUP")):
        HookRegistry.register(
            "sglang.srt.managers.scheduler.Scheduler._add_request_to_queue",
            enqueue_request,
            HookType.AROUND,
        )


def create_cache(ctx: Any) -> UnifiedRadixCache:
    """Factory selected by SGLang's ``--radix-cache-backend orbitkv``."""
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_disagg, get_memory

    if ctx.disable_radix_cache:
        raise ValueError("OrbitKV direct GPU linker requires RadixCache")
    if ctx.enable_hierarchical_cache:
        raise ValueError("OrbitKV direct GPU linker does not support hierarchical cache")
    if ctx.is_hybrid_ssm:
        raise ValueError(
            "OrbitKV requires a complete released external-linker contract for recurrent state"
        )
    if (
        ctx.is_dsa
        or ctx.params.is_eagle
        or ctx.params.mtp_draft_device_pools
        or ctx.params.component_registry_override
        or hasattr(ctx.params.req_to_token_pool, "req_to_c128_sidecar")
    ):
        raise ValueError(
            "OrbitKV direct GPU linker cannot restore DSA, draft, or auxiliary GPU state; "
            "select a backend with a complete recovery contract for that model"
        )
    disaggregation = get_disagg()
    if not get_memory().enable_unified_cache_external_linker:
        raise ValueError(
            "OrbitKV direct GPU linker requires --enable-unified-cache-external-linker "
            "so SGLang schedules GPU KV restores"
        )
    if get_disagg().disaggregation_decode_retraction_backup == "host_pool":
        raise ValueError("OrbitKV direct GPU linker does not support host-pool retraction")

    from .linker import OrbitKVLinker
    from .recovery import RecoveryLinkerWrapper

    # SGLang's built-in unified-cache factory hardcodes Mooncake/Mori when the
    # external-linker flag is set. Construct its public RadixCache component
    # directly and attach OrbitKV through the public linker interface.
    components = [ComponentType.FULL]
    if ctx.is_hybrid_swa:
        components.append(ComponentType.SWA)
    ctx.params.tree_components = tuple(components)
    cache = UnifiedRadixCache(ctx.params)
    linker = OrbitKVLinker(ctx.server_args, ctx.params)
    try:
        cache.linker = RecoveryLinkerWrapper(
            cache,
            linker,
            restore_from_store=disaggregation.disaggregation_mode != "decode",
        )
    except Exception:
        linker.close()
        raise
    return cache
