"""Nonblocking request admission through SGLang's general plugin hooks."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from orbitkv.logging_utils import get_connector_logger, trace_transfer

from .config import validate_lora_request


def enqueue_request(original: Callable, scheduler: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
    result = original(scheduler, req, *args, **kwargs)
    # This hook only prepares requests accepted by the ordinary serving queue.
    if not scheduler.waiting_queue or scheduler.waiting_queue[-1] is not req:
        return result
    from sglang.srt.runtime_context import get_memory

    if get_memory().radix_cache_backend != "orbitkv":
        return result
    from .linker import OrbitKVLinker

    wrapper = getattr(scheduler.tree_cache, "linker", None)
    linker = getattr(wrapper, "cache_linker", None)
    if (
        not isinstance(linker, OrbitKVLinker)
        or not wrapper.restore_from_store
        or req.positional_embed_overrides is not None
    ):
        return result
    trace_transfer("queued", req.rid, engine="sglang")
    prepare = os.environ.get("ORBITKV_PREPARE_REQUESTS") == "1"
    if not prepare and os.environ.get("ORBITKV_QUEUE_WARMUP") != "1":
        return result
    if prepare and len(scheduler.waiting_queue) > 4:
        return result
    # Hybrid demand must own every required component; the optional warming
    # operation currently prepares only attention group zero.
    if len(linker.layout.pools) != 1:
        return result
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    from orbitkv import BlockHashes

    tokens = req.origin_input_ids + req.output_ids
    key = RadixKey(
        token_ids=tokens,
        extra_key=req.extra_key,
        cache_salt=req.cache_salt,
        limit=req._compute_max_prefix_len(len(tokens)),
    )
    # req=None keeps this HBM-only: no external lookup, allocation or load marker.
    resident = scheduler.tree_cache.match_prefix(MatchPrefixParams(key=key, cow_mamba=False))
    keys = wrapper._tail_hashes(key, resident, resident.device_prefix_len)
    try:
        hashes = BlockHashes(linker._hashes(keys))
        if prepare:
            origin = resident.device_prefix_len // linker.page_size * linker.page_size
            linker.client.prepare_recovery(
                linker.instance_id,
                hashes,
                req.rid,
                linker.recovery,
                linker.namespace,
                origin,
                origin + len(keys) * linker.page_size,
                0,
            )
        else:
            linker.client.warm_prefix(linker.instance_id, hashes, req.rid)
    except (RuntimeError, OSError):
        get_connector_logger().warning(
            "Queued preparation failed for request %s", req.rid, exc_info=True
        )
    return result


def admit_request(original: Callable, adder: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
    from sglang.srt.runtime_context import get_memory

    if get_memory().radix_cache_backend != "orbitkv":
        return original(adder, req, *args, **kwargs)
    from .linker import OrbitKVLinker

    wrapper = getattr(adder.tree_cache, "linker", None)
    linker = getattr(wrapper, "cache_linker", None)
    if isinstance(linker, OrbitKVLinker):
        validate_lora_request(req, linker._static_lora_ids)
        from sglang.srt.managers.schedule_policy import AddReqResult

        match_limit = req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))
        restorable_tokens = match_limit // linker.page_size * linker.page_size
        if req.prefix_len >= restorable_tokens:
            linker.cancel_query(req.rid)
        # Every attention rank takes the same admission decision even when its
        # SSD read finishes in a different scheduler iteration.
        state = linker.query_state(req.rid)
        if wrapper.needs_rank_consensus:
            import torch

            decision = torch.tensor([state], dtype=torch.int)
            adder.tree_cache._all_reduce_attn_groups(decision, torch.distributed.ReduceOp.MAX)
            state = int(decision.item())
        if state == 1:
            return AddReqResult.CONTINUE
        if state == 2:
            linker.expire_query(req.rid)
    return original(adder, req, *args, **kwargs)
