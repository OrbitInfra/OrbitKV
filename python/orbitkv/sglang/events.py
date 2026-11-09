"""CUDA event ownership for SGLang graph capture and external restores."""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from orbitkv.logging_utils import trace_transfer

if TYPE_CHECKING:
    import torch

    from .layout import GpuLayout


class _LayerDoneCounter:
    """Own stable per-layer events before capture and admit each consuming forward."""

    def __init__(self, layout: GpuLayout):
        import torch

        self.layout = layout
        self.num_layers = layout.num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self._activations: dict[int, Future[int]] = {}
        self._staged: dict[int, Future[None]] = {}
        self.request_ids: dict[int, list[str]] = {}
        self.layer_events: dict[str, torch.cuda.Event] = {}
        self.layer_groups: list[list[str]] = []
        self._events_by_layer: list[list[torch.cuda.Event]] = [[] for _ in range(self.num_layers)]
        start = min(layer for pool in layout.pools.values() for layer in pool.entry.layer_mapping)
        for pool in layout.pools.values():
            layers = {local: global_id for global_id, local in pool.entry.layer_mapping.items()}
            names_by_layer: dict[int, list[str]] = {}
            for index, name in enumerate(pool.layer_names):
                event = torch.cuda.Event(external=True)
                event.record()
                self.layer_events[name] = event
                layer = layers[index % len(layers)]
                self._events_by_layer[layer - start].append(event)
                names_by_layer.setdefault(layer, []).append(name)
            self.layer_groups.append(
                [name for layer in sorted(names_by_layer) for name in names_by_layer[layer]]
            )

    def update_producer(self) -> int:
        self.producer_index += 1
        self._activations[self.producer_index] = Future()
        self._staged[self.producer_index] = Future()
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        import torch

        self.consumer_index = index
        activation = self._activations.get(index)
        if activation is None:
            return
        # Serialize event reuse against the previous users on the actual forward stream.
        activation.set_result(torch.cuda.current_stream().cuda_stream)
        try:
            self._staged[index].result()
        finally:
            self._activations.pop(index)
            self._staged.pop(index)
        for rid in self.request_ids.pop(index, ()):
            trace_transfer("first_use", rid, engine="sglang")

    def wait_until(self, threshold: int) -> None:
        import torch

        # Emit these waits during capture even when no Restore is pending. External
        # event nodes use the newly recorded generation on every graph replay.
        stream = torch.cuda.current_stream()
        for event in self._events_by_layer[threshold]:
            stream.wait_event(event)

    def publish_events(self, index: int, error: Exception | None = None) -> None:
        future = self._staged.get(index)
        if future is None or future.done():
            return
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)

    def cancel_pending(self) -> None:
        for index, activation in tuple(self._activations.items()):
            if activation.cancel():
                self._staged[index].cancel()

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self._activations.clear()
        self._staged.clear()
        self.request_ids.clear()


def initialize_layer_counter(worker: Any, capture_decode_cuda_graph: bool = True) -> None:
    """Install replay dependencies before SGLang's first graph capture."""
    from sglang.srt.runtime_context import get_memory

    if get_memory().radix_cache_backend != "orbitkv":
        return

    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    from .layout import GpuLayout

    request_pool, allocator = worker.get_memory_pool()
    kvcache = allocator.get_kvcache()
    if isinstance(kvcache.layer_transfer_counter, _LayerDoneCounter):
        return
    components = {ComponentType.FULL}
    if isinstance(request_pool, HybridReqToTokenPool):
        components.add(ComponentType.MAMBA)
    if isinstance(kvcache, SWAKVPool):
        components.add(ComponentType.SWA)
    layout = GpuLayout.from_pool(
        kvcache.page_size,
        kvcache,
        request_pool,
        components,
        worker.sliding_window_size or 0,
    )
    counter = _LayerDoneCounter(layout)
    kvcache.register_layer_transfer_counter(counter)
    if ComponentType.MAMBA in components:
        request_pool.register_layer_transfer_counter(counter)
    worker.register_hicache_layer_transfer_counter(counter)
