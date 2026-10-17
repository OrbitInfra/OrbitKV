"""SGLang GPU-page integration with the node-local Cache Manager."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

from orbitkv import (
    BlockHashes,
    CacheManagerClient,
    QueryCandidates,
    QueryReady,
    RecoveryContract,
)
from orbitkv.client.gpu import resolve_device_id
from orbitkv.logging_utils import TRANSFER_TRACING, trace_transfer

from .config import derive_namespace, resolve_static_loras, resolve_transfer_backend
from .events import _LayerDoneCounter

if TYPE_CHECKING:
    from orbitkv import GpuRegistration

logger = logging.getLogger(__name__)


@dataclass
class _Lookup:
    keys: tuple[str, ...]
    origin: int
    hashes: BlockHashes
    groups: dict[PoolName, Any] = field(default_factory=dict)
    boundaries: tuple[int, ...] = ()
    candidates: dict[PoolName, Any] = field(default_factory=dict)
    selected: int = 0


@dataclass(frozen=True)
class _Load:
    rid: str
    groups: tuple[tuple[PoolName, bytes, tuple[int | None, ...]], ...]


class OrbitKVLinker(UnifiedCacheLinker):
    """Store aligned SGLang GPU pages using OrbitKV's CUDA IPC data path.

    The SGLang radix tree owns GPU slots. Its offload/load callbacks hold those
    slots until the completion queues acknowledge the Cache Manager transfer.
    Each scheduler rank has one short-lived GPU registration and one stable,
    model-scoped namespace that other replicas of the same rank can reuse.
    """

    _QUERY_WAIT_SECONDS = 5.0

    def __init__(self, server_args: Any, params: Any):
        transfer_backend = resolve_transfer_backend()
        counter = params.token_to_kv_pool_allocator.get_kvcache().layer_transfer_counter
        if not isinstance(counter, _LayerDoneCounter):
            raise RuntimeError("OrbitKV layer events must be installed before graph capture")
        self.layer_done_counter = counter
        self.layout = counter.layout
        self.page_size = self.layout.page_size
        from sglang.srt.runtime_context import get_lora

        lora_config = get_lora()
        static_loras = resolve_static_loras(server_args, lora_config)
        self._static_lora_ids = frozenset(ref.lora_id for ref in static_loras)
        self.namespace = derive_namespace(
            server_args, params, self.layout, lora_config=lora_config, static_loras=static_loras
        )
        self.recovery = RecoveryContract(
            self.namespace,
            self.page_size,
            [(pool.group_id, pool.kind, pool.window) for pool in self.layout.pools.values()],
        )
        self.instance_id = f"sglang-{uuid.uuid4().hex}"
        self.device_id = resolve_device_id()
        endpoint = os.environ.get("ORBITKV_SGLANG_ENDPOINT", "unix:///run/orbitkv/orbitkv.sock")
        if not endpoint.startswith("unix://"):
            raise ValueError("ORBITKV_SGLANG_ENDPOINT must be a unix:// socket")
        self.client = CacheManagerClient(endpoint.removeprefix("unix://"))
        try:
            self.client.start_session_watcher(self.instance_id, self.namespace, 1, 1)
            pools = list(self.layout.pools.values())
            self._torch_device = pools[0].entry.kv_buffer[0].device
            tensor_count = sum(len(pool.entry.kv_buffer) for pool in pools)
            self.registration = self.client.register_context_batch(
                self.instance_id,
                self.namespace,
                0,
                0,
                1,
                1,
                self.device_id,
                [name for pool in pools for name in pool.layer_names],
                [count for pool in pools for count in pool.num_blocks],
                [size for pool in pools for size in pool.block_bytes],
                [0] * tensor_count,
                [1] * tensor_count,
                transfer_backend,
                False,
                tensors=[tensor for pool in pools for tensor in pool.entry.kv_buffer],
                layer_group_ids=[pool.group_id for pool in pools for _ in pool.layer_names],
                layer_attention=[
                    item
                    for pool in pools
                    for item in pool.attention_layouts(
                        self.layout.num_layers,
                        min(layer for p in pools for layer in p.entry.layer_mapping),
                    )
                ],
                layer_formats=[
                    {
                        "torch.bfloat16": "bf16",
                        "torch.float16": "fp16",
                        "torch.float8_e4m3fn": "fp8_e4m3",
                    }.get(str(tensor.dtype), "exact")
                    if pool.kind in {"attention", "window", "mla"}
                    else "exact"
                    for pool in pools
                    for tensor in pool.entry.kv_buffer
                ],
            )
        except Exception:
            self.client.close()
            raise

        self._lookups: dict[str, _Lookup] = {}
        self._origins: dict[str, int] = {}
        self._load_boundaries: dict[str, tuple[int, dict[PoolName, set[str]]]] = {}
        self._pending_queries: dict[str, tuple[tuple[str, ...], float]] = {}
        self._expired_queries: set[str] = set()
        self._queued_loads: dict[str, _Load] = {}
        self._load_queue: queue.Queue[tuple[int, list[_Load]] | None] = queue.Queue()
        self._offload_queue: queue.Queue[
            tuple[GpuRegistration, list[tuple[str, list[int], list[bytes]]], torch.cuda.Event]
            | None
        ] = queue.Queue()
        self._completed_loads: queue.Queue[list[str]] = queue.Queue()
        self._load_error: Exception | None = None
        self._completed_offloads: queue.Queue[bool] = queue.Queue()
        self._last_publish_warning = 0.0
        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name="orbitkv-sglang-load"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker, daemon=True, name="orbitkv-sglang-offload"
        )
        self._load_thread.start()
        self._offload_thread.start()
        logger.info(
            "OrbitKV GPU linker registered %s buffers, %s pages on device %s, transfer backend %s",
            sum(len(pool.layer_names) for pool in self.layout.pools.values()),
            self.layout.pools[PoolName.KV].num_blocks[0],
            self.device_id,
            transfer_backend,
        )

    @staticmethod
    def _hashes(keys: list[str] | tuple[str, ...]) -> list[bytes]:
        return [hashlib.sha256(key.encode()).digest() for key in keys]

    def _release_lookup(self, rid: str) -> None:
        lookup = self._lookups.pop(rid, None)
        if lookup is not None:
            for result in lookup.groups.values():
                if result.lease:
                    self.client.release(result.lease)

    def cancel_query(self, rid: str) -> None:
        self._pending_queries.pop(rid, None)
        for pool in self.layout.pools.values():
            self.client.cancel_query(self.instance_id, rid, group_id=pool.group_id)
        self._release_lookup(rid)

    def expire_query(self, rid: str) -> None:
        self.cancel_query(rid)
        self._expired_queries.add(rid)

    def query_state(self, rid: str) -> int:
        pending = self._pending_queries.get(rid)
        if pending is not None and not self.client.peer_alive():
            raise RuntimeError("OrbitKV Cache Manager exited during a pending cache query")
        if rid in self._expired_queries or (
            pending is not None and time.monotonic() - pending[1] >= self._QUERY_WAIT_SECONDS
        ):
            return 2
        return int(rid in self._pending_queries)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        by_name = {transfer.name: transfer for transfer in transfers}
        if len(by_name) != len(transfers) or set(by_name) != set(self.layout.pools):
            raise ValueError("SGLang lookup must declare every registered state pool")
        keys = tuple(by_name[PoolName.KV].keys or ())
        if not keys or rid in self._expired_queries:
            return []
        if rid not in self._origins:
            raise ValueError("SGLang lookup requires an engine-supplied absolute token origin")
        origin = self._origins[rid]
        lookup = self._lookups.get(rid)
        if lookup is not None and (lookup.keys != keys or lookup.origin != origin):
            self.cancel_query(rid)
            lookup = None
        pending = self._pending_queries.get(rid)
        started = pending[1] if pending is not None else time.monotonic()
        if time.monotonic() - started >= self._QUERY_WAIT_SECONDS:
            logger.warning("Cache query wait expired; recomputing request %s", rid)
            self.expire_query(rid)
            return []
        if lookup is None:
            if len(set(keys)) != len(keys):
                raise ValueError("SGLang lookup contains duplicate page hashes")
            lookup = _Lookup(keys, origin, BlockHashes(self._hashes(keys)))
            self._lookups[rid] = lookup
        if len(self.layout.pools) > 1:
            return self._discover_recovery(rid, lookup, started)
        try:
            if PoolName.KV not in lookup.groups:
                result = self.client.query_prefetch(
                    self.instance_id, lookup.hashes, rid, wait_for_full_prefix=False, group_id=0
                )
                if not isinstance(result, QueryReady):
                    self._pending_queries[rid] = (keys, started)
                    return []
                lookup.groups[PoolName.KV] = result
                if not 0 <= result.num_hit_blocks <= len(lookup.hashes):
                    raise ValueError("OrbitKV returned an invalid state hit count")
                if result.num_hit_blocks and not result.lease:
                    raise RuntimeError("OrbitKV reported state hits without a restore lease")
            self._pending_queries.pop(rid, None)
            if not lookup.boundaries:
                count = lookup.groups[PoolName.KV].num_hit_blocks
                lookup.boundaries = tuple(
                    self.recovery.restorable_boundaries(
                        self.namespace,
                        origin,
                        origin + count * self.page_size,
                        [
                            (
                                0,
                                [
                                    origin + (position + 1) * self.page_size
                                    for position in range(count)
                                ],
                            )
                        ],
                    )
                )
        except Exception:
            self.cancel_query(rid)
            raise
        if not lookup.boundaries:
            self._release_lookup(rid)
        return [(boundary - origin) // self.page_size for boundary in lookup.boundaries]

    def _discover_recovery(self, rid: str, lookup: _Lookup, started: float) -> list[int]:
        if lookup.boundaries:
            return [(end - lookup.origin) // self.page_size for end in lookup.boundaries]
        try:
            loading = False
            for name, pool in self.layout.pools.items():
                if name in lookup.candidates:
                    continue
                result = self.client.query_candidates(
                    self.instance_id, lookup.hashes, rid, group_id=pool.group_id
                )
                if not isinstance(result, QueryCandidates):
                    loading = True
                    continue
                lookup.candidates[name] = result
            if loading:
                self._pending_queries[rid] = (lookup.keys, started)
                return []
            lookup.boundaries = tuple(
                self.recovery.common_boundaries(
                    self.namespace,
                    lookup.origin,
                    lookup.origin + len(lookup.keys) * self.page_size,
                    [
                        [
                            (pool.group_id, lookup.candidates[name].hit_positions)
                            for name, pool in self.layout.pools.items()
                        ]
                    ],
                )
            )
            if not lookup.boundaries:
                self._pending_queries.pop(rid, None)
                self._release_lookup(rid)
            return [(end - lookup.origin) // self.page_size for end in lookup.boundaries]
        except Exception:
            self.cancel_query(rid)
            raise

    def prepare_recovery(self, rid: str, boundary: int) -> int:
        """Acquire the selected rank-common boundary before SGLang allocates HBM.

        -1 is stale/missing, 0 is pending, 1 has every required leased page.
        """
        lookup = self._lookups[rid]
        if boundary not in lookup.boundaries or (lookup.selected and lookup.selected != boundary):
            self.expire_query(rid)
            return -1
        lookup.selected = boundary
        try:
            loading = False
            for name, pool in self.layout.pools.items():
                if name in lookup.groups:
                    continue
                result = self.client.read_recovery(
                    self.instance_id,
                    lookup.hashes,
                    rid,
                    self.recovery,
                    self.namespace,
                    lookup.origin,
                    boundary,
                    pool.group_id,
                )
                if not isinstance(result, QueryReady):
                    loading = True
                    continue
                if not result.lease:
                    self.expire_query(rid)
                    return -1
                lookup.groups[name] = result
            if loading:
                self._pending_queries.setdefault(rid, (lookup.keys, time.monotonic()))
                return 0
            self._pending_queries.pop(rid, None)
            return 1
        except Exception:
            self.cancel_query(rid)
            raise

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._check_load_failure()
        if rid in self._queued_loads:
            raise ValueError(f"duplicate OrbitKV load for {rid}")
        lookup = self._lookups.pop(rid, None)
        if lookup is None:
            raise RuntimeError(f"SGLang load for {rid} has no OrbitKV lookup leases")
        groups = []
        try:
            if set(lookup.groups) != set(self.layout.pools):
                raise RuntimeError("SGLang recovery leases are not ready")
            by_name = {transfer.name: transfer for transfer in transfers}
            if len(by_name) != len(transfers) or not set(by_name) <= set(self.layout.pools):
                raise ValueError("SGLang load contains unknown or duplicate state pools")
            boundary, resident = self._load_boundaries.pop(rid)
            if boundary not in lookup.boundaries or (
                len(self.layout.pools) > 1 and lookup.selected != boundary
            ):
                raise ValueError("SGLang selected an unproved recovery boundary")
            required = {
                group: lookup.keys[
                    (start - lookup.origin) // self.page_size : (end - lookup.origin)
                    // self.page_size
                ]
                for group, start, end in self.recovery.required_ranges(
                    self.namespace, lookup.origin, boundary
                )
            }
            for name, result in lookup.groups.items():
                pool = self.layout.pools[name]
                expected = set(required[pool.group_id])
                positions = (
                    list(range(result.num_hit_blocks))
                    if pool.group_id == 0
                    else result.hit_positions
                )
                held = {lookup.keys[position]: index for index, position in enumerate(positions)}
                targets: list[int | None] = [None] * len(positions)
                transfer = by_name.get(name)
                keys = ()
                if transfer is not None:
                    keys = tuple(transfer.keys or ())
                    if transfer.device_indices is None or len(set(keys)) != len(keys):
                        raise ValueError("SGLang load has missing destinations or duplicate keys")
                    block_ids = pool.block_ids(transfer.device_indices, len(keys))
                    for key, block_id in zip(keys, block_ids, strict=True):
                        if key not in expected:
                            raise ValueError("SGLang load is outside its required state range")
                        position = held[key]
                        targets[position] = block_id
                retained = resident.get(name, set())
                if set(keys) & retained or set(keys) | retained != expected:
                    raise ValueError(
                        "GPU destinations and retained state do not cover the recovery plan"
                    )
                if result.lease:
                    groups.append((name, result.lease, tuple(targets)))
            self._queued_loads[rid] = _Load(rid, tuple(groups))
        except Exception:
            for result in lookup.groups.values():
                if result.lease:
                    self.client.release(result.lease)
            raise
        return True

    def start_layer_wise_loading(self) -> int:
        self._check_load_failure()
        if not self._queued_loads:
            return -1
        pending = list(self._queued_loads.values())
        self._queued_loads.clear()
        index = self.layer_done_counter.update_producer()
        self.layer_done_counter.request_ids[index] = [load.rid for load in pending]
        self._load_queue.put((index, pending))
        return index

    def _load_worker(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                index, pending = task
                submitted = False
                try:
                    ready_stream = self.layer_done_counter._activations[index].result()
                    self._check_load_failure()
                    for load in pending:
                        trace_transfer("restore_submit", load.rid, engine="sglang")
                    submitted = True
                    restore = self.client.start_restore(
                        self.instance_id,
                        0,
                        self.device_id,
                        self.layer_done_counter.layer_groups,
                        [
                            (
                                lease,
                                [
                                    list(targets) if name == pool_name else [None] * len(targets)
                                    for name in self.layout.pools
                                ],
                            )
                            for load in pending
                            for pool_name, lease, targets in load.groups
                        ],
                        ready_stream=ready_stream,
                        layer_events=list(self.layer_done_counter.layer_events.items()),
                    )
                    if TRANSFER_TRACING:
                        for load in pending:
                            trace_transfer(
                                "restore_link", load.rid, engine="sglang", restore_key=restore.key
                            )
                    self.client.wait_restore_enqueued(restore, timeout=120)
                    self.layer_done_counter.publish_events(index)
                    status = self.client.wait_restore(restore, timeout=120)
                    if not status.success:
                        raise RuntimeError(status.message)
                    for load in pending:
                        trace_transfer("gpu_ready", load.rid, engine="sglang", success=True)
                    self._completed_loads.put([load.rid for load in pending])
                except CancelledError:
                    # A batch never admitted to forward has not consumed its source leases.
                    for load in pending:
                        for _, lease, _ in load.groups:
                            self.client.release(lease)
                except Exception as error:
                    self._load_error = error
                    logger.exception("OrbitKV SGLang GPU restore failed")
                    if not submitted:
                        for load in pending:
                            for _, lease, _ in load.groups:
                                self.client.release(lease)
                    self.layer_done_counter.publish_events(index, error)
            finally:
                self._load_queue.task_done()

    def cancel_queued_load(self, rid: str) -> bool:
        self.cancel_query(rid)
        self._expired_queries.discard(rid)
        self._origins.pop(rid, None)
        self._load_boundaries.pop(rid, None)
        load = self._queued_loads.pop(rid, None)
        if load is None:
            return False
        for _, lease, _ in load.groups:
            self.client.release(lease)
        return True

    def num_completed_loads(self) -> int:
        self._check_load_failure()
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        self._check_load_failure()
        return self._completed_loads.get_nowait()

    def _check_load_failure(self) -> None:
        if self._load_error is not None:
            raise RuntimeError(
                "OrbitKV restore failed; GPU pages remain held until transfer teardown"
            ) from self._load_error

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        by_name = {transfer.name: transfer for transfer in transfers}
        if (
            len(by_name) != len(transfers)
            or PoolName.KV not in by_name
            or not set(by_name) <= set(self.layout.pools)
        ):
            return False
        saves = []
        # Earlier tree nodes may have lost their auxiliary state. Preserve
        # their attention prefix; the recovery contract decides completeness
        # when it joins this prefix with a later window or checkpoint.
        for name, transfer in by_name.items():
            pool = self.layout.pools[name]
            keys = list(transfer.keys or ())
            if not keys or transfer.device_indices is None:
                return False
            if pool.kind in ("recurrent", "convolution") and len(keys) != 1:
                raise ValueError("Recurrent offload requires exactly one boundary checkpoint")
            full_keys = by_name[PoolName.KV].keys or ()
            if keys != list(full_keys[-len(keys) :]):
                raise ValueError("Offload state groups must end at the same prefix boundary")
            block_ids = pool.block_ids(transfer.device_indices, len(keys))
            hashes = self._hashes(keys)
            saves.extend((layer, block_ids, hashes) for layer in pool.layer_names)
        ready = torch.cuda.Event()
        ready.record()
        self._offload_queue.put((self.registration, saves, ready))
        return True

    def _offload_worker(self) -> None:
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                registration, saves, ready = task
                success = False
                try:
                    ready.synchronize()
                    success, message = self.client.save(registration, saves)
                    if not success:
                        logger.error("OrbitKV SGLang GPU offload failed: %s", message)
                except Exception:
                    logger.exception("OrbitKV SGLang GPU offload failed")
                self._completed_offloads.put(success)
            finally:
                self._offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        try:
            progress = self.client.publish_progress()
            if progress.active and (progress.overdue or progress.retained):
                now = time.monotonic()
                if now - self._last_publish_warning >= 60:
                    self._last_publish_warning = now
                    logger.warning(
                        "OrbitKV Publish still holds pages: active=%s retained=%s interval_s=%s Manager_alive=%s",
                        progress.active,
                        progress.retained,
                        progress.active_interval_seconds,
                        progress.manager_alive,
                    )
        except Exception:
            logger.debug("OrbitKV Publish activity unavailable", exc_info=True)
        return self._completed_offloads.qsize()

    def pop_completed_offload(self) -> bool:
        return self._completed_offloads.get_nowait()

    def reset(self) -> None:
        self.layer_done_counter.cancel_pending()
        self._load_queue.join()
        self._offload_queue.join()
        for rid in list(self._pending_queries):
            self.cancel_query(rid)
        self._expired_queries.clear()
        self._origins.clear()
        self._load_boundaries.clear()
        for rid in list(self._lookups):
            self._release_lookup(rid)
        for rid in list(self._queued_loads):
            self.cancel_queued_load(rid)
        while not self._completed_loads.empty():
            self._completed_loads.get_nowait()
        while not self._completed_offloads.empty():
            self._completed_offloads.get_nowait()
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self._load_queue.put(None)
        self._offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        unregistered = False
        try:
            unregistered, message = self.client.unregister_context(self.instance_id)
            if not unregistered:
                logger.warning("Could not unregister SGLang GPU context: %s", message)
        except Exception:
            logger.warning("Could not unregister SGLang GPU context", exc_info=True)
        self.client.close()
        if unregistered:
            self.registration = None
