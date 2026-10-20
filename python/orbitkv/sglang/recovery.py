"""Translate SGLang prefix boundaries into OrbitKV recovery requirements."""

from __future__ import annotations

import torch
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)

from .config import validate_lora_request


class RecoveryLinkerWrapper(UnifiedCacheLinkerWrapper):
    """Use native private load destinations and their post-forward completion."""

    def __init__(self, cache, cache_linker, *, restore_from_store: bool):
        if is_swa_req_ring(cache.token_to_kv_pool_allocator):
            raise ValueError("SWA request rings require a separate recovery contract")
        super().__init__(cache, cache_linker)
        self.needs_rank_consensus = cache.tp_world_size > 1 or any(
            group is not None and torch.distributed.get_world_size(group=group) > 1
            for group in (cache.attn_cp_group, cache.attn_tp_group)
        )
        self.restore_from_store = restore_from_store

    def match(self, key, req, result):
        validate_lora_request(req, self.cache_linker._static_lora_ids)
        if not self.restore_from_store:
            return result
        self.cache_linker._origins[req.rid] = result.device_prefix_len
        try:
            matched = super().match(key, req, result)
            hit = self.hit_markers.get(req.rid)
            if hit is None or len(self.cache_linker.layout.pools) == 1:
                return matched
            boundary = hit.device_hit_len + len(hit.tail_hashes) * self.cache.page_size
            state = self.cache_linker.prepare_recovery(req.rid, boundary)
            if self.needs_rank_consensus:
                decision = torch.tensor([state], dtype=torch.int)
                self.cache._all_reduce_attn_groups(decision, torch.distributed.ReduceOp.MIN)
                state = int(decision.item())
            if state == 1:
                return matched
            self.hit_markers.pop(req.rid, None)
            if state < 0:
                self.cache_linker.expire_query(req.rid)
            return result
        finally:
            self.cache_linker._origins.pop(req.rid, None)

    def load_back(self, req):
        hit = self.hit_markers.get(req.rid)
        if hit is not None:
            self.cache_linker._load_boundaries[req.rid] = (
                hit.device_hit_len + len(hit.tail_hashes) * self.cache.page_size,
                {},
            )
        try:
            return super().load_back(req)
        finally:
            self.cache_linker._load_boundaries.pop(req.rid, None)
            self.cache_linker.cancel_query(req.rid)

    def release_request(self, rid):
        loading = rid in self.inflight_load_rids
        super().release_request(rid)
        if not loading:
            self.cache_linker.cancel_queued_load(rid)
