"""Compile SGLang state pools into registered groups and recovery requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType


@dataclass(frozen=True)
class GpuPool:
    group_id: int
    kind: str
    window: int
    entry: DevicePoolEntry
    layer_names: list[str]
    num_blocks: list[int]
    block_bytes: list[int]

    @classmethod
    def compile(cls, group_id: int, kind: str, window: int, entry: DevicePoolEntry) -> GpuPool:
        names, counts, sizes = [], [], []
        for index, tensor in enumerate(entry.kv_buffer):
            if tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError("OrbitKV requires contiguous CUDA state buffers")
            if tensor.shape[0] % entry._row_span:
                raise ValueError("SGLang GPU state buffer is not page aligned")
            names.append(f"{entry.name}:{index}")
            counts.append(tensor.shape[0] // entry._row_span)
            sizes.append(tensor.stride(0) * tensor.element_size() * entry._row_span)
        if not names or len(set(counts)) != 1:
            raise ValueError("SGLang state group has no buffers or inconsistent page counts")
        return cls(group_id, kind, window, entry, names, counts, sizes)

    def attention_layouts(
        self, layer_count: int, start_layer: int
    ) -> list[tuple[int, str, int, int]]:
        if self.kind not in {"attention", "window"}:
            return [(0, "", 0, 0)] * len(self.layer_names)
        mapping = {local: global_id for global_id, local in self.entry.layer_mapping.items()}
        layers = len(mapping)
        result = []
        for index, tensor in enumerate(self.entry.kv_buffer):
            if (
                tensor.dtype not in {torch.bfloat16, torch.float16}
                or tensor.stride(-1) != 1
                or any(stride % tensor.shape[-1] for stride in tensor.stride()[:-1])
            ):
                result.append((0, "", 0, 0))
            else:
                result.append(
                    (
                        tensor.shape[-1],
                        "k" if index < layers else "v",
                        mapping[index % layers] - start_layer,
                        layer_count,
                    )
                )
        return result

    def block_ids(self, indices: torch.Tensor, expected_pages: int) -> list[int]:
        if indices.numel() != expected_pages * self.entry.page_size:
            raise ValueError("SGLang GPU indices do not cover exactly the requested state pages")
        indices = self.entry.translate_indices(indices)
        rows = self.entry.prepare_locations(indices)
        return [row // self.entry._row_span for row in rows]


@dataclass(frozen=True)
class GpuLayout:
    page_size: int
    num_layers: int
    pools: dict[PoolName, GpuPool]

    @classmethod
    def from_pool(
        cls,
        page: int,
        kvcache: Any,
        request_pool: Any,
        components: set[ComponentType],
        sliding_window_size: int = 0,
    ) -> GpuLayout:
        pools = {}

        def attention(name, pool, mapping, kind, window=0):
            if type(pool) not in (MHATokenToKVPool, MLATokenToKVPool):
                raise ValueError(f"Unsupported SGLang attention pool: {type(pool).__name__}")
            if type(pool) is MHATokenToKVPool and (
                pool.k_scale_buffer is not None or pool.native_k_scale_buffer is not None
            ):
                raise ValueError("OrbitKV cannot restore separate SGLang KV scale buffers")
            if type(pool) is MLATokenToKVPool and pool.kv_scale_buffer is not None:
                raise ValueError("OrbitKV cannot restore separate SGLang MLA scale buffers")
            buffers = (
                [list(pool.k_buffer), list(pool.v_buffer)]
                if type(pool) is MHATokenToKVPool
                else [list(pool.kv_buffer)]
            )
            entry = DevicePoolEntry(
                name=name,
                indices_from_pool=name,
                device_pool=pool,
                components=buffers,
                layer_mapping=mapping,
                page_size=page,
                rows_are_pages=bool(getattr(pool, "use_hnd", False)),
            )
            pools[name] = GpuPool.compile(len(pools), kind, window, entry)

        supported = {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA}
        if ComponentType.FULL not in components or not components <= supported:
            raise ValueError(f"Unsupported SGLang state components: {components}")

        if ComponentType.SWA in components:
            from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

            if type(kvcache) is not SWAKVPool:
                raise ValueError("Window recovery requires an ordinary SWAKVPool")
            full = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if not is_swa
            }
            swa = {
                global_id: local_id
                for global_id, (local_id, is_swa) in kvcache.layers_mapping.items()
                if is_swa
            }
            attention(
                PoolName.KV,
                kvcache.full_kv_pool,
                full,
                "mla" if type(kvcache.full_kv_pool) is MLATokenToKVPool else "attention",
            )
            attention(PoolName.SWA, kvcache.swa_kv_pool, swa, "window", sliding_window_size)
        elif ComponentType.MAMBA not in components:
            kind = "mla" if type(kvcache) is MLATokenToKVPool else "attention"
            attention(PoolName.KV, kvcache, {i: i for i in range(kvcache.layer_num)}, kind)
        else:
            if not isinstance(kvcache, HybridLinearKVPool):
                raise ValueError("Recurrent recovery requires HybridLinearKVPool")
            attention(
                PoolName.KV,
                kvcache.full_kv_pool,
                dict(kvcache.full_attention_layer_id_mapping),
                "mla" if kvcache.use_mla else "attention",
            )

        if ComponentType.MAMBA in components:
            mamba = request_pool.mamba_pool
            state = mamba.mamba_cache
            if (
                getattr(mamba, "enable_linear_replayssm", False)
                or getattr(mamba, "enable_linear_replayssm_spec", False)
                or getattr(mamba, "_slot_siblings", ())
            ):
                raise ValueError(
                    "Recurrent replay, draft, and sibling state need separate recovery rules"
                )
            mapping = dict(request_pool.mamba_map)
            layers = len(mapping)
            if not state.conv or state.temporal.shape[0] != layers:
                raise ValueError("Recurrent pool does not cover all declared layers")
            buffers = [list(tensor.unbind(0)) for tensor in state.conv]
            if state.temporal.numel():
                buffers.append(list(state.temporal.unbind(0)))
            if any(len(component) != layers for component in buffers):
                raise ValueError("Convolution and recurrent layer coverage disagree")
            entry = DevicePoolEntry(
                name=PoolName.MAMBA,
                indices_from_pool=PoolName.MAMBA,
                device_pool=mamba,
                components=buffers,
                layer_mapping=mapping,
                page_size=1,
                rows_are_pages=True,
                index_mapper=request_pool.translate_mamba_indices,
            )
            pools[PoolName.MAMBA] = GpuPool.compile(
                len(pools), "recurrent" if state.temporal.numel() else "convolution", 0, entry
            )

        # Convolution/checkpoint state may accompany attention in the same layer.
        # Full and sliding-window KV must still describe disjoint attention layers.
        attention_layers = [
            layer
            for name, pool in pools.items()
            if name != PoolName.MAMBA
            for layer in pool.entry.layer_mapping
        ]
        if len(set(attention_layers)) != len(attention_layers):
            raise ValueError("Model layers belong to more than one attention group")
        layers = {layer for pool in pools.values() for layer in pool.entry.layer_mapping}
        start = getattr(kvcache, "start_layer", 0)
        end = getattr(kvcache, "end_layer", start + len(layers))
        if layers != set(range(start, end)):
            raise ValueError("Registered state does not cover every model layer")
        return cls(page, len(layers), pools)
