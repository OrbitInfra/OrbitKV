"""
Scheduler-side connector logic.
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady
from orbitkv import ShardedQueryReady as NativeShardedQueryReady
from orbitkv.logging_utils import get_connector_logger, trace_transfer
from orbitkv.vllm.config import ConnectorContext
from orbitkv.vllm.layout import CacheGroupLayout
from orbitkv.vllm.metadata import (
    LoadIntent,
    OrbitKVConnectorMetadata,
    OrbitKVWorkerMetadata,
    RecoveryLoadHold,
    SaveIntent,
)
from orbitkv.vllm.metrics import (
    OrbitKVConnectorStats,
    PrefetchTracker,
)
from orbitkv.vllm.tp_shards import ShardedQueryReady, TpShardQueryClient

logger = get_connector_logger()

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


def block_hashes_per_block(block_hashes, hash_scale: int) -> tuple[bytes, ...]:
    """One chained hash per scheduler block: the last hash inside each block.

    This is vLLM's own `BlockHashListWithBlockSize` rule. Trailing hashes of an
    incomplete block are dropped; OrbitKV stores whole blocks only.
    """
    return tuple(block_hashes[hash_scale - 1 :: hash_scale])


@dataclass(slots=True)
class _QueryProbe:
    """One recovery-query snapshot and its owned state groups.

    A probe snapshots:

    * ``computed_blocks`` — blocks already computed locally when the query was
      issued
    * ``query_hashes`` — remaining complete engine block hashes

    If the request keeps making local progress while the backend is loading,
    the current query key may drift.  A *Ready* result is accepted only if the
    current key still matches this snapshot.
    """

    computed_blocks: int
    query_hashes: tuple[bytes, ...]
    native_hashes: BlockHashes = field(init=False)
    started_at: float = field(default_factory=time.monotonic)

    # ``None`` means the backend is still loading.
    hit_blocks: int | None = None
    leases: tuple[bytes, ...] = ()
    control_id: bytes = b""
    # Hybrid (HMA): pinned windows/checkpoints from the membership queries,
    # set when the shared validator found a complete recovery boundary.
    recovery_hold: RecoveryLoadHold | None = None
    # Completed groups stay leased while other groups read from backing tiers.
    # None records a submitted group that must be cancelled on query retirement.
    groups: dict[int, list[tuple[tuple[int, ...], bytes]] | None] = field(default_factory=dict)
    # Attention-only prefix before validation; the junction hint needs this
    # even when no complete hybrid boundary exists.
    attention_hit_blocks: int = 0
    candidates: dict[int, list[tuple[int, ...]] | None] = field(default_factory=dict)
    selected_boundary: int = 0
    # Actual attention pages pinned by the selected group lease.
    leased_blocks: int = 0

    def __post_init__(self) -> None:
        self.native_hashes = BlockHashes(self.query_hashes)

    @property
    def is_ready(self) -> bool:
        return self.hit_blocks is not None

    def matches(
        self,
        computed_blocks: int,
        query_hashes: tuple[bytes, ...],
    ) -> bool:
        return self.computed_blocks == computed_blocks and self.query_hashes == query_hashes

    def mark_ready(self, ready: ShardedQueryReady) -> None:
        hit_blocks = ready.num_hit_blocks
        if hit_blocks > len(self.query_hashes):
            raise RuntimeError(
                f"invariant violated: server returned {hit_blocks} hits for "
                f"{len(self.query_hashes)} hashes"
            )
        self.hit_blocks = hit_blocks
        self.leased_blocks = hit_blocks
        self.leases = ready.leases
        self.control_id = ready.control_id
        self.recovery_hold = ready.recovery_hold
        self.selected_boundary = ready.boundary
        self.attention_hit_blocks = ready.attention_hit_blocks
        if ready.recovery_hold is not None:
            self.groups.clear()  # Ownership moved into the scheduler/worker handoff.

    def require_hit_blocks(self) -> int:
        if self.hit_blocks is None:
            raise RuntimeError("query probe is still loading")
        return self.hit_blocks


class SchedulerAdapter:
    """Holds scheduler-only state and behaviors."""

    _HYBRID_QUERY_WAIT_SECONDS = 5.0

    def __init__(
        self,
        context: ConnectorContext,
        clients: tuple[CacheManagerClient, ...] | None = None,
        kv_cache_config=None,
    ):
        self._ctx = context
        clients = tuple(clients or (context.client,))
        groups = CacheGroupLayout.from_config(kv_cache_config)
        expected_clients = context.tp_shard_count if groups.group_count > 1 else 1
        if len(clients) != expected_clients:
            raise ValueError(
                f"scheduler requires {expected_clients} local cache clients, got {len(clients)}"
            )
        self.query_handshake_ready = not context.query_control
        self._query_controls: dict[bytes, set[int]] = {}
        self._tp_shard_client = TpShardQueryClient(clients)
        self._clients = clients
        self._lora_bindings: dict[tuple[str, str], int] = {}
        self._lora_ids: dict[int, tuple[str, str]] = {}
        self._cache_generation: bytes | None = None
        self._generation_hashes: dict[str, tuple[tuple[bytes, ...], tuple[bytes, ...]]] = {}
        self._queued_at: dict[str, float] = {}
        self._cache_groups = CacheGroupLayout.from_config(kv_cache_config)
        self._recovery = None
        if self._cache_groups.group_count > 1:
            from orbitkv import RecoveryContract

            self._recovery = RecoveryContract(
                context.namespace, context.virtual_block_size, self._cache_groups.recovery_groups
            )
        self._gpu_block_pool = None

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._prefetch_start_times: dict[str, float] = {}
        self._pending_query_probes: dict[str, _QueryProbe] = {}

        # Prefetch tracking (for metrics)
        self._prefetch_tracker = PrefetchTracker()

        # Save state (per-request)
        self._block_hashes: dict[str, tuple[bytes, ...]] = {}
        self._external_matched_blocks: dict[str, int] = {}
        self._block_index_offsets: dict[str, int] = {}
        self._allocated_blocks: dict[str, list[list[int]]] = {}
        self._scheduled_tokens: dict[str, int] = {}
        self._next_stored_block_idx: dict[str, int] = {}

        # Live Request references – used to refresh block_hashes during decode
        # so that newly completed blocks can be saved, not just prefill blocks.
        self._requests: dict[str, Request] = {}

        # Completion tracking
        self._pending_saves: set[str] = set()
        self._held_requests: set[str] = set()

        # Hybrid (HMA) recurrent boundary states. vLLM hands off every
        # committed align-mode boundary block through
        # `SchedulerOutput.kv_connector_block_state.boundary_state_offloads`
        # in the step that writes it. Each hand-off becomes a boundary save
        # job: the block is pinned in the GPU block pool until every worker
        # reports the job done, so the save is independent of the request's
        # own block lifetime (align-mode tables free superseded state blocks
        # one step later, and a request may finish or be preempted first).
        self._next_boundary_job_id = 0
        # job id -> (pinned GPU block ids, workers yet to report)
        self._pinned_boundary_jobs: dict[int, tuple[list[int], int]] = {}
        # req id -> (group index, hash index) already handed off
        self._saved_boundaries: dict[str, set[tuple[int, int]]] = {}

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def _native_block_hashes(self, request: "Request") -> tuple[bytes, ...]:
        """Per-block keys from `Request.block_hashes` (see `ConnectorContext.hash_scale`).

        vLLM hashes full spans only, so the hashes never reach past the
        request's tokens — except when a connector pops the last token after
        vLLM hashed it (NIXL / Mooncake truncate hybrid-model prompts by one
        token on the prefiller so it computes ``h(N-1)``) without trimming
        ``block_hashes``. A hash spanning a token the request no longer has
        fingerprints a prefix vLLM will not compute; drop it. Anything beyond
        that single tail hash means the hashes are finer than the scale says,
        and keying blocks by them would file block ``i`` under the hash of a
        shorter prefix — any request sharing that prefix would then load this
        request's blocks. Fail loudly instead.
        """
        block_hashes = request.block_hashes
        num_tokens = getattr(request, "num_tokens", 0)
        if num_tokens:
            hash_block_size = self._ctx.hash_block_size or self._ctx.virtual_block_size
            hashed = num_tokens // hash_block_size
            stale = len(block_hashes) - hashed
            if stale > 1:
                raise RuntimeError(
                    f"req {request.request_id}: {len(block_hashes)} block hashes span "
                    f"{len(block_hashes) * hash_block_size} tokens but the request has "
                    f"{num_tokens}; vLLM's hash granularity is finer than the connector's "
                    f"{hash_block_size}"
                )
            if stale == 1:
                block_hashes = block_hashes[:hashed]
        return block_hashes_per_block(block_hashes, self._ctx.hash_scale)

    def _request_block_hashes(self, request: "Request") -> tuple[bytes, ...]:
        hashes = self._native_block_hashes(request)
        if self._cache_generation is None:
            return hashes
        cached = self._generation_hashes.get(request.request_id)
        if cached is not None and cached[0] == hashes:
            return cached[1]
        from orbitkv import rekey_hashes

        scoped = rekey_hashes(hashes, self._cache_generation)
        self._generation_hashes[request.request_id] = (hashes, scoped)
        return scoped

    def _validate_lora(self, request: "Request") -> None:
        if self._ctx.static_loras and getattr(request, "resumable", False):
            raise ValueError("OrbitKV static LoRA does not support resumable requests")
        lora = getattr(request, "lora_request", None)
        if lora is None:
            return
        selection = (lora.lora_name, lora.lora_path)
        if selection not in self._ctx.static_loras:
            raise ValueError("OrbitKV request selects an undeclared static LoRA adapter")
        if lora.load_inplace or lora.tensorizer_config_dict is not None or lora.is_3d_lora_weight:
            raise ValueError(
                "OrbitKV static LoRA rejects reload, tensorizer and 3D adapter requests"
            )
        adapter_id = lora.lora_int_id
        if type(adapter_id) is not int or adapter_id <= 0:
            raise ValueError("OrbitKV static LoRA requires a positive integer adapter ID")
        if self._lora_ids.get(adapter_id, selection) != selection or (
            self._lora_bindings.get(selection, adapter_id) != adapter_id
        ):
            raise ValueError("OrbitKV static LoRA adapter ID bindings cannot change")
        self._lora_ids[adapter_id] = selection
        self._lora_bindings[selection] = adapter_id

    def on_new_request(self, request: "Request") -> None:
        self._validate_lora(request)
        self._queued_at[request.request_id] = time.monotonic()
        trace_transfer("queued", request.request_id, engine="vllm")
        prepare = os.environ.get("ORBITKV_PREPARE_REQUESTS") == "1"
        if (
            not self._ctx.read_enabled
            or self._cache_groups.group_count > 1
            or (not prepare and os.environ.get("ORBITKV_QUEUE_WARMUP") != "1")
            or (prepare and len(self._queued_at) > 4)
            or (prepare and self._ctx.wait_for_full_prefix)
        ):
            return
        # Warm the same whole pages admission will query, including a page
        # whose last token will be recomputed. This keeps shared reads identical.
        # Enqueue neither allocates GPU blocks nor pins the resident prefix.
        hashes = self._native_block_hashes(request)
        resident = 0
        if self._gpu_block_pool is not None:
            for block_hash in hashes:
                if self._gpu_block_pool.get_cached_block(block_hash, [0]) is None:
                    break
                resident += 1
        if resident < len(hashes):
            demand = BlockHashes(self._request_block_hashes(request)[resident:])
            for client in self._clients:
                try:
                    if prepare:
                        client.prepare_prefix(self._ctx.instance_id, demand, request.request_id)
                    else:
                        client.warm_prefix(self._ctx.instance_id, demand, request.request_id)
                except (RuntimeError, OSError):
                    logger.warning(
                        "Queued preparation failed for request %s",
                        request.request_id,
                        exc_info=True,
                    )

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        req_id = request.request_id

        if not self._ctx.read_enabled:
            logger.debug(
                "[OrbitKVConnector] req=%s cache_lookup_skipped: mode=%s",
                req_id,
                self._ctx.mode.value,
            )
            return (0, False)

        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size
        query_hashes = self._request_block_hashes(request)[computed_blocks:]

        # Nothing remains to query remotely.
        if not query_hashes:
            for client in self._clients:
                client.cancel_query(self._ctx.instance_id, req_id)
            self._release_pending_query_probe(req_id)
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        probe = self._pending_query_probes.get(req_id)

        if probe is not None and not probe.matches(computed_blocks, query_hashes):
            self._release_pending_query_probe(req_id)
            probe = None
        if probe is not None and probe.is_ready:
            return self._complete_cache_lookup(
                request=request, probe=probe, lookup_us=None, reused=True
            )

        if probe is None:
            probe = _QueryProbe(
                computed_blocks=computed_blocks,
                query_hashes=query_hashes,
            )
            self._pending_query_probes[req_id] = probe

        # Keep completed state groups pinned while remaining queries advance.
        lookup_start = time.perf_counter()
        ready = self._query_recovery(req_id, probe, request.num_tokens - 1)
        lookup_us = (time.perf_counter() - lookup_start) * 1e6

        # Backend is still loading.  Keep the original snapshot.
        if ready is None:
            return (None, False)

        probe.mark_ready(ready)
        return self._complete_cache_lookup(
            request=request,
            probe=probe,
            lookup_us=lookup_us,
            reused=False,
        )

    def _complete_cache_lookup(
        self,
        *,
        request: "Request",
        probe: _QueryProbe,
        lookup_us: float | None,
        reused: bool,
    ) -> tuple[int | None, bool]:
        # Pace independent prefixes while batching the final block of native hits.
        if self._pending_load_intents and (
            probe.computed_blocks == 0 or len(probe.query_hashes) != 1
        ):
            return (None, False)

        # `_finish_cache_lookup` may release the probe, so snapshot what the
        # junction hint needs first.
        computed_blocks = probe.computed_blocks
        attention_hit_blocks = probe.attention_hit_blocks
        hit_tokens, load_async = self._finish_cache_lookup(
            req_id=request.request_id,
            num_tokens=request.num_tokens,
            probe=probe,
            lookup_us=lookup_us,
            reused=reused,
        )
        self._hint_shared_prefix_boundary(
            request, computed_blocks, attention_hit_blocks, hit_tokens
        )
        return hit_tokens, load_async

    def _hint_shared_prefix_boundary(
        self,
        request: "Request",
        computed_blocks: int,
        attention_hit_blocks: int,
        hit_tokens: int,
    ) -> None:
        """Ask vLLM to commit a recurrent state where a shared prefix ends.

        A hybrid hit resumes only at a recurrent checkpoint, so an attention
        prefix that runs past the last usable checkpoint (a new session
        sharing a system prompt, a sibling branching mid-history) is
        recomputed. vLLM's own Marconi-style junction detection needs a local
        prefix hit, which the HMA layout does not expose (see
        `bind_gpu_block_pool`), so report the junction from the external
        index instead: `Request.shared_prefix_boundary` makes the prefill
        chunk stop exactly there, which commits the align-mode boundary state
        under that block's hash and hands it to
        `_consume_boundary_state_offloads`. The next request sharing the
        prefix then resumes from its end.
        """
        if not self._cache_groups.has_recurrent_state or attention_hit_blocks <= 0:
            return
        if not hasattr(request, "shared_prefix_boundary"):
            return
        vbs = self._ctx.virtual_block_size
        junction = (computed_blocks + attention_hit_blocks) * vbs
        resumable = computed_blocks * vbs + hit_tokens
        if junction <= resumable or junction >= request.num_tokens:
            return
        if junction <= request.shared_prefix_boundary:
            return
        request.shared_prefix_boundary = junction
        logger.debug(
            "[OrbitKVConnector] req=%s shared_prefix_boundary=%d "
            "(attention prefix %d blocks, resumable %d tokens)",
            request.request_id,
            junction,
            computed_blocks + attention_hit_blocks,
            resumable,
        )

    def _finish_cache_lookup(
        self,
        *,
        req_id: str,
        num_tokens: int,
        probe: _QueryProbe,
        lookup_us: float | None,
        reused: bool,
    ) -> tuple[int, bool]:
        hit_blocks = probe.require_hit_blocks()
        computed_blocks = probe.computed_blocks
        vbs = self._ctx.virtual_block_size
        hit_tokens = hit_blocks * vbs

        # A request still needs one forward token to produce logits. The last
        # loaded page may contain that token's KV, but vLLM must recompute and
        # overwrite it unless a P/D router supplied a separate decode token.
        locally_computed_tokens = computed_blocks * vbs
        hit_tokens = min(hit_tokens, max(0, num_tokens - locally_computed_tokens - 1))

        if probe.recovery_hold is not None and (
            hit_tokens != hit_blocks * vbs
            or locally_computed_tokens + hit_tokens != probe.selected_boundary
        ):
            # The payload belongs to exactly one checkpoint. Engine budget
            # drift cannot turn it into evidence for a different boundary.
            if self._pending_query_probes.get(req_id) is probe:
                self._release_pending_query_probe(req_id)
            return (0, False)

        loaded_blocks = (hit_tokens + vbs - 1) // vbs
        self._external_matched_blocks[req_id] = computed_blocks + loaded_blocks

        if reused:
            logger.debug(
                "[OrbitKVConnector] req=%s cache_lookup_reuse: hit_blocks=%d "
                "computed_blocks=%d hit_tokens=%d num_tokens=%d total_query_hashes=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                len(probe.query_hashes),
            )
        else:
            logger.info(
                "[OrbitKVConnector] req=%s cache_lookup: hit_blocks=%d computed_blocks=%d "
                "hit_tokens=%d num_tokens=%d lookup_us=%.0f total_query_hashes=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                lookup_us or 0.0,
                len(probe.query_hashes),
            )

        if hit_tokens <= 0:
            # No external load will consume this lease.
            self._release_pending_query_probe(req_id)
            return (0, False)

        # Schedule this request in the consuming forward; layer events fence GPU reads.
        return (hit_tokens, False)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id

        if num_external_tokens == 0:
            probe = self._pending_query_probes.get(req_id)
            if probe is not None:
                # MultiConnector assigned the target pages to another loader.
                self._external_matched_blocks[req_id] = probe.computed_blocks
                self._release_pending_query_probe(req_id)

        # Keep a live reference so we can refresh block_hashes during decode
        # (Request.block_hashes grows as new full blocks are completed).
        self._requests[req_id] = request

        # One key per scheduler block, 1-to-1 with block_ids from the scheduler.
        self._block_hashes[req_id] = self._request_block_hashes(request)
        if req_id not in self._allocated_blocks:
            # The first locally allocated block may be after an external-hit
            # prefix. Track that global block index explicitly.
            base_block_idx = self._external_matched_blocks.get(req_id, 0)
            self._block_index_offsets[req_id] = base_block_idx
            self._allocated_blocks[req_id] = [[] for _ in range(self._cache_groups.group_count)]
            self._scheduled_tokens[req_id] = 0
            self._next_stored_block_idx[req_id] = base_block_idx

        if num_external_tokens > 0:
            pending_probe = self._pending_query_probes.get(req_id)
            if pending_probe is None:
                raise RuntimeError(f"req {req_id} missing query lease for external load")
            block_ids_by_group = (
                self._copy_block_ids_by_group(blocks.get_block_ids())
                if blocks
                else tuple(() for _ in range(self._cache_groups.group_count))
            )
            hash_group_index = self._cache_groups.hash_group_index
            # Synchronous allocation hashes external and newly scheduled pages
            # before this callback. Only the query snapshot identifies local hits.
            num_computed_blocks = pending_probe.computed_blocks
            start_block_idx = num_computed_blocks
            vbs = self._ctx.virtual_block_size
            num_load_blocks = (num_external_tokens + vbs - 1) // vbs
            try:
                query_hashes = self._request_block_hashes(request)[num_computed_blocks:]
                if not pending_probe.matches(num_computed_blocks, query_hashes):
                    raise RuntimeError(f"req {req_id} query identity changed before external load")
                if pending_probe.recovery_hold is not None and (
                    num_external_tokens != pending_probe.require_hit_blocks() * vbs
                    or num_computed_blocks * vbs + num_external_tokens
                    != pending_probe.selected_boundary
                ):
                    raise RuntimeError(f"req {req_id} allocation changed the recovery boundary")
                leased_blocks = pending_probe.require_hit_blocks()
                if leased_blocks != num_load_blocks:
                    raise RuntimeError(
                        f"req {req_id} leased block mismatch: "
                        f"leased={leased_blocks} load={num_load_blocks}"
                    )
                if not pending_probe.leases or any(not lease for lease in pending_probe.leases):
                    raise RuntimeError(f"req {req_id} missing query lease for external load")
                load_block_ids_by_group = self._load_block_ids_by_group(
                    block_ids_by_group,
                    start_block_idx,
                    num_load_blocks,
                    leased_blocks=pending_probe.leased_blocks,
                )
            except RuntimeError:
                self._release_pending_query_probe(req_id)
                raise

            load_intent = LoadIntent(
                block_ids_by_group=load_block_ids_by_group,
                leases=pending_probe.leases,
                num_tokens=num_external_tokens,
                recovery_hold=pending_probe.recovery_hold,
                control_id=pending_probe.control_id,
            )
            self._pending_load_intents[req_id] = load_intent
            if load_intent.control_id:
                self._query_controls[load_intent.control_id] = set()
            self._pending_query_probes.pop(req_id, None)
            logger.debug(
                "[OrbitKVConnector] req=%s alloc: total_blocks=%d computed_blocks=%d "
                "load_blocks=%d start_block_idx=%d load_tokens=%d pending_loads=%d",
                req_id,
                len(block_ids_by_group[hash_group_index]),
                num_computed_blocks,
                len(load_intent.block_ids_by_group[hash_group_index]),
                start_block_idx,
                load_intent.num_tokens,
                len(self._pending_load_intents),
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> OrbitKVConnectorMetadata:
        potential_saves: dict[str, SaveIntent] = {}
        for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
            if tokens > 0 and (queued := self._queued_at.pop(req_id, None)) is not None:
                trace_transfer(
                    "first_use",
                    req_id,
                    engine="vllm",
                    elapsed_us=int((time.monotonic() - queued) * 1e6),
                )
        load_intents = self._pending_load_intents
        self._pending_load_intents = {}

        # Process new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)

            # Verify update_state_after_alloc was called for this request
            assert req_id in self._block_hashes, (
                f"req {req_id} not initialized in update_state_after_alloc"
            )

            # Populate block IDs from scheduler_output — single source of
            # truth for the save path (consistent with offloading connector).
            if req.block_ids:
                self._allocated_blocks[req_id] = [
                    list(group) for group in self._copy_block_ids_by_group(req.block_ids)
                ]
            # The V2 model runner reports resumed requests here rather than in
            # `scheduled_cached_reqs.resumed_req_ids`; for a fresh request the
            # rebase is a no-op.
            self._rebase_resumed_request(req_id)

            self._scheduled_tokens[req_id] = max(
                0,
                req.num_computed_tokens
                + num_tokens
                - self._block_index_offsets[req_id] * self._ctx.virtual_block_size,
            )

            if save_intent := self._consume_full_block_saves(req_id):
                potential_saves[req_id] = save_intent

        # Process cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._block_hashes:
                continue

            # Refresh block hashes from the live Request object so that
            # newly completed blocks during decode are also saved.
            req = self._requests.get(req_id)
            if req is not None:
                self._block_hashes[req_id] = self._request_block_hashes(req)

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)

            # Append newly allocated blocks
            new_block_ids = cached_reqs.new_block_ids[idx]
            if req_id in cached_reqs.resumed_req_ids:
                self._allocated_blocks[req_id] = (
                    [list(group) for group in self._copy_block_ids_by_group(new_block_ids)]
                    if new_block_ids
                    else [[] for _ in range(self._cache_groups.group_count)]
                )
                self._rebase_resumed_request(req_id)
            elif new_block_ids:
                for allocated, new_group in zip(
                    self._allocated_blocks[req_id],
                    self._copy_block_ids_by_group(new_block_ids),
                    strict=True,
                ):
                    allocated.extend(new_group)

            self._scheduled_tokens[req_id] = max(
                0,
                cached_reqs.num_computed_tokens[idx]
                + num_tokens
                - self._block_index_offsets[req_id] * self._ctx.virtual_block_size,
            )

            if save_intent := self._consume_full_block_saves(req_id):
                potential_saves[req_id] = save_intent

        save_intents = potential_saves

        boundary_save_intents = self._consume_boundary_state_offloads(scheduler_output)
        if self._cache_groups.window_group_indices:
            # The engine can retire an SWA page while its request keeps running.
            # Job ownership, rather than request lifetime, fences every queued copy.
            for intent in save_intents.values():
                boundary_save_intents[self._pin_save_intent(intent)] = intent
            save_intents = {}

        self._pending_saves.update(save_intents.keys())

        logger.debug(
            "[OrbitKVConnector] build_connector_meta: %d loads, %d saves, %d boundary saves",
            len(load_intents),
            len(save_intents),
            len(boundary_save_intents),
        )

        return OrbitKVConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
            boundary_save_intents=boundary_save_intents,
            preempted_req_ids=scheduler_output.preempted_req_ids or None,
        )

    def _consume_boundary_state_offloads(
        self, scheduler_output: "SchedulerOutput"
    ) -> dict[int, SaveIntent]:
        """Turn vLLM's recurrent boundary hand-offs into pinned save jobs.

        Each entry is ``(group_id, block_id, boundary_tokens)``: the exact
        align-mode block holding the recurrent state after ``boundary_tokens``
        tokens, committed by this step's forward. The state is filed under
        the request hash ending at that boundary. Only whole-block boundaries
        are stored; a sub-block partial tail has no hash of its own here.

        Every committed boundary is handed off, so a request contributes a
        resume point at each prefill chunk end, at its junction with a shared
        prefix, and at each block it crosses while decoding — not just one
        at request end.
        """
        if not self._cache_groups.has_recurrent_state:
            return {}
        try:
            block_state = scheduler_output.kv_connector_block_state
        except AttributeError as exc:
            raise RuntimeError(
                "OrbitKV HMA requires a vLLM scheduler that hands off mamba "
                "boundary states (SchedulerOutput.kv_connector_block_state)"
            ) from exc
        offloads = block_state.boundary_state_offloads if block_state is not None else None
        if not offloads:
            return {}

        pool = self._gpu_block_pool
        if pool is None:
            raise RuntimeError("GPU block pool was not bound before a boundary-state hand-off")

        vbs = self._ctx.virtual_block_size
        recurrent = self._cache_groups.recurrent_group_indices
        group_count = self._cache_groups.group_count
        intents: dict[int, SaveIntent] = {}
        for req_id, entries in offloads.items():
            request = self._requests.get(req_id)
            if request is None:
                # Finished before this step's metadata; its blocks are going away.
                continue
            block_hashes = self._request_block_hashes(request)
            self._block_hashes[req_id] = block_hashes
            saved = self._saved_boundaries.setdefault(req_id, set())
            # vLLM allocates a real recurrent block per externally loaded
            # block but the load fills only the checkpoint; after the load it
            # still caches (and offers) every block of the loaded prefix. Those
            # blocks hold no state this request computed, so only boundaries
            # past the loaded prefix may be saved.
            loaded_blocks = self._external_matched_blocks.get(req_id, 0)
            rows: list[tuple[int, int, bytes]] = []
            for group_index, block_id, boundary_tokens in entries:
                if group_index not in recurrent or block_id <= 0:
                    continue
                if boundary_tokens <= 0 or boundary_tokens % vbs != 0:
                    continue
                hash_index = boundary_tokens // vbs - 1
                if hash_index < loaded_blocks or hash_index >= len(block_hashes):
                    continue
                key = (group_index, hash_index)
                if key in saved:
                    continue
                saved.add(key)
                rows.append((group_index, block_id, block_hashes[hash_index]))
            if not rows:
                continue

            intent = SaveIntent(
                block_ids_by_group=tuple(
                    tuple(
                        block_id if group_index == group else 0 for group_index, block_id, _ in rows
                    )
                    for group in range(group_count)
                ),
                block_hashes=tuple(block_hash for _, _, block_hash in rows),
            )
            job_id = self._pin_save_intent(intent)
            intents[job_id] = intent
            logger.debug(
                "[OrbitKVConnector] req=%s boundary_save job=%d boundaries=%s blocks=%s",
                req_id,
                job_id,
                [(group_index, (hash_index + 1) * vbs) for group_index, hash_index in saved],
                self._pinned_boundary_jobs[job_id][0],
            )
        return intents

    def _pin_save_intent(self, intent: SaveIntent) -> int:
        pool = self._gpu_block_pool
        if pool is None:
            raise RuntimeError("GPU block pool must be bound before a hybrid save")
        pinned = list(
            dict.fromkeys(
                block_id for group in intent.block_ids_by_group for block_id in group if block_id
            )
        )
        pool.touch([pool.blocks[block_id] for block_id in pinned])
        job_id = self._next_boundary_job_id
        self._next_boundary_job_id += 1
        self._pinned_boundary_jobs[job_id] = (pinned, self._ctx.world_size)
        return job_id

    def _release_boundary_jobs(self, completed: dict[int, int]) -> None:
        pool = self._gpu_block_pool
        for job_id, count in completed.items():
            pinned = self._pinned_boundary_jobs.get(job_id)
            if pinned is None:
                logger.warning(
                    "[OrbitKVConnector] boundary_save job=%d reported by a worker but not pinned",
                    job_id,
                )
                continue
            block_ids, remaining = pinned
            remaining -= count
            if remaining > 0:
                self._pinned_boundary_jobs[job_id] = (block_ids, remaining)
                continue
            if remaining < 0:
                raise RuntimeError(f"boundary_save job={job_id} reported by too many workers")
            del self._pinned_boundary_jobs[job_id]
            assert pool is not None
            # Newest boundary first so a shared prefix is evicted last.
            pool.free_blocks(pool.blocks[block_id] for block_id in reversed(block_ids))

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping while boundary saves still pin blocks.

        Completions only reach the scheduler as worker metadata on a step; an
        engine that quiesced with jobs in flight would hold those references
        (and their GPU blocks) until the next request arrives.
        """
        return bool(self._pinned_boundary_jobs)

    def _rebase_resumed_request(self, req_id: str) -> None:
        """Restart connector-side bookkeeping when vLLM resumes a preempted request.

        Preemption resets the engine's computed extent and block table. Rebase
        the cache offset from the fresh lookup before deriving saveable pages.
        Hashes already stored keep their progress across the new allocation.
        """
        base_block_idx = self._external_matched_blocks.get(req_id, 0)
        self._block_index_offsets[req_id] = base_block_idx
        self._scheduled_tokens[req_id] = 0
        self._next_stored_block_idx[req_id] = max(
            self._next_stored_block_idx.get(req_id, 0), base_block_idx
        )

    def _consume_full_block_saves(self, req_id: str) -> SaveIntent | None:
        block_hashes = self._block_hashes.get(req_id)
        if block_hashes is None:
            return None

        allocated = self._allocated_blocks.get(req_id, [])
        scheduled = self._scheduled_tokens.get(req_id, 0)
        base_block_idx = self._block_index_offsets.get(req_id, 0)
        start_block_idx = self._next_stored_block_idx.get(req_id, base_block_idx)

        # Recurrent (align-mode mamba) groups never save positionally: their
        # table is not append-only (superseded state blocks are freed and
        # nulled, speculative blocks relocate) and mostly null, so the mirror
        # here is neither complete nor trustworthy. Their states are saved
        # exactly from vLLM's boundary hand-offs instead
        # (`_consume_boundary_state_offloads`); attention groups keep the
        # dense per-block cadence.
        recurrent = self._cache_groups.recurrent_group_indices
        attention_lengths = [
            len(group)
            for group_index, group in enumerate(allocated)
            if group_index not in recurrent
        ]

        # _allocated_blocks tracks request block IDs in global request order.
        # In external-hit cases, the prefix-loaded block IDs are still present at
        # the front, so save intents must slice by global block index rather than
        # rebasing to a local-only view. The table length is therefore a global
        # bound of its own: `base_block_idx + scheduled` says which positions
        # hold valid KV, the table says which of them we can address, and a
        # full-block hash exists for the whole prompt from the start. Slicing
        # past the table would silently drop block IDs but not hashes.
        saveable_block_idx = min(
            len(block_hashes),
            min(attention_lengths, default=0),
            base_block_idx + scheduled // self._ctx.virtual_block_size,
        )
        new_blocks = saveable_block_idx - start_block_idx
        if new_blocks <= 0:
            return None

        hash_start = start_block_idx
        save_hashes = block_hashes[hash_start : hash_start + new_blocks]
        save_block_ids_by_group = tuple(
            (0,) * new_blocks
            if group_index in recurrent
            else tuple(group[hash_start : hash_start + new_blocks])
            for group_index, group in enumerate(allocated)
        )
        self._next_stored_block_idx[req_id] = saveable_block_idx

        logger.debug(
            "[OrbitKVConnector] req=%s save_intent: start=%d hash_start=%d "
            "base_block_idx=%d saveable_block_idx=%d new_blocks=%d total_hashes=%d",
            req_id,
            hash_start,
            hash_start,
            base_block_idx,
            saveable_block_idx,
            new_blocks,
            len(block_hashes),
        )

        return SaveIntent(
            block_ids_by_group=save_block_ids_by_group,
            block_hashes=save_hashes,
        )

    def _copy_block_ids_by_group(self, block_ids) -> tuple[tuple[int, ...], ...]:
        groups = tuple(tuple(group) for group in block_ids)
        if len(groups) != self._cache_groups.group_count:
            raise RuntimeError(
                "KV cache group count mismatch: "
                f"expected={self._cache_groups.group_count} actual={len(groups)}"
            )
        if any(block_id < 0 for group in groups for block_id in group):
            raise RuntimeError("KV cache block IDs must be non-negative")
        return groups

    def _load_block_ids_by_group(
        self,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        start_block_idx: int,
        num_load_blocks: int,
        leased_blocks: int | None = None,
    ) -> tuple[tuple[int | None, ...], ...]:
        """Destination block IDs per cache group, one entry per leased block.

        The engine walks the lease and the destination vector in lock step,
        so the vector must be exactly as long as the lease. A hit can end up
        shorter than the lease (validation selects an earlier
        checkpoint; the final prompt token is recomputed): the leased blocks
        past `num_load_blocks` get `None` targets and are skipped.
        """
        end_block_idx = start_block_idx + num_load_blocks
        available = [len(group) for group in block_ids_by_group]
        if any(length < end_block_idx for length in available):
            raise RuntimeError(
                f"load block mismatch: start={start_block_idx} count={num_load_blocks} "
                f"available_by_group={available}"
            )

        if leased_blocks is None:
            leased_blocks = num_load_blocks
        if leased_blocks < num_load_blocks:
            raise RuntimeError(
                f"load block mismatch: leased={leased_blocks} count={num_load_blocks}"
            )
        padding = (None,) * (leased_blocks - num_load_blocks)
        required = None
        if self._recovery is not None:
            vbs = self._ctx.virtual_block_size
            required = {
                group: (start // vbs, end // vbs)
                for group, start, end in self._recovery.required_ranges(
                    self._ctx.namespace, start_block_idx * vbs, end_block_idx * vbs
                )
            }

        result: list[tuple[int | None, ...]] = []
        for group_index, block_ids in enumerate(block_ids_by_group):
            destinations: tuple[int | None, ...] = block_ids[start_block_idx:end_block_idx]
            if required is not None:
                begin, end = required[self._cache_groups.storage_group_ids[group_index]]
                destinations = tuple(
                    block_id if begin <= position < end else None
                    for position, block_id in enumerate(destinations, start=start_block_idx)
                )
            result.append(destinations + padding)
        return tuple(result)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in connector_output.finished_sending or []:
            self._pending_saves.discard(req_id)
            logger.debug("[OrbitKVConnector] Request %s save completed", req_id)

            # Clean up if request already finished
            if req_id in self._held_requests:
                self._cleanup_request(req_id)
                self._held_requests.discard(req_id)

        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(worker_meta, OrbitKVWorkerMetadata):
            if worker_meta.completed_boundary_jobs:
                self._release_boundary_jobs(worker_meta.completed_boundary_jobs)
            expected = set(range(self._ctx.tp_size))
            for control_id, ranks in worker_meta.completed_query_controls.items():
                if not ranks or not ranks <= expected:
                    raise ValueError("query completion contains undeclared global TP ranks")
                received = self._query_controls.get(control_id)
                if received is None:
                    continue
                received.update(ranks)
                if received == expected:
                    self._ctx.client.release_shard_query(control_id)
                    del self._query_controls[control_id]

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict | None]:
        req_id = request.request_id

        # Check if there are pending saves for this request
        if req_id in self._pending_saves:
            self._held_requests.add(req_id)
            logger.debug(
                "[OrbitKVConnector] Request %s blocks held for async save",
                req_id,
            )
            return (True, None)

        # No pending saves, clean up immediately
        self._cleanup_request(req_id)
        return (False, None)

    def _cleanup_request(self, req_id: str) -> None:
        """Clean up all state for a completed request."""
        self._release_pending_query_probe(req_id)
        self._queued_at.pop(req_id, None)
        for client in self._clients:
            client.cancel_query(self._ctx.instance_id, req_id)
        self._requests.pop(req_id, None)
        self._generation_hashes.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._saved_boundaries.pop(req_id, None)
        self._pending_saves.discard(req_id)

    def _query_recovery(
        self, req_id: str, probe: _QueryProbe, limit: int
    ) -> ShardedQueryReady | None:
        """Advance one request-owned recovery query without reacquiring ready groups."""
        try:
            if self._cache_groups.group_count > 1:
                ready = self._query_hybrid(req_id, probe, limit)
            elif probe.leases:
                ready = ShardedQueryReady(
                    probe.leased_blocks, probe.leases, control_id=probe.control_id
                )
            else:
                if self._ctx.query_control:
                    if not self.query_handshake_ready:
                        raise RuntimeError("official worker query handshake is incomplete")
                    result = self._ctx.client.query_shards(
                        self._ctx.instance_id, probe.native_hashes, req_id
                    )
                    if isinstance(result, QueryLoading):
                        ready = None
                    elif isinstance(result, NativeShardedQueryReady):
                        if len(result.leases) != self._ctx.tp_shard_count:
                            self._ctx.client.release_shard_query(result.control_id)
                            raise RuntimeError(
                                "native query response does not cover every TP shard"
                            )
                        ready = ShardedQueryReady(
                            result.num_hit_blocks, result.leases, control_id=result.control_id
                        )
                    else:
                        raise TypeError("query_shards returned an unexpected response")
                else:
                    result = self._ctx.client.query_prefetch(
                        self._ctx.instance_id,
                        probe.native_hashes,
                        req_id=req_id,
                        wait_for_full_prefix=self._ctx.wait_for_full_prefix,
                    )
                    if isinstance(result, QueryLoading):
                        ready = None
                    elif isinstance(result, QueryReady):
                        if result.num_hit_blocks > len(probe.native_hashes) or (
                            result.num_hit_blocks and not result.lease
                        ):
                            if result.lease:
                                self._ctx.client.release(result.lease)
                            raise RuntimeError("invalid local prefix query result")
                        ready = ShardedQueryReady(result.num_hit_blocks, (result.lease,))
                    else:
                        raise TypeError("query_prefetch returned unexpected outcome")
                if ready is not None:
                    probe.leases = ready.leases
                    probe.leased_blocks = ready.num_hit_blocks
                    probe.control_id = ready.control_id
        except Exception:
            self._release_query_probe(req_id, probe)
            if self._pending_query_probes.get(req_id) is probe:
                del self._pending_query_probes[req_id]
            raise

        if ready is None:
            if req_id not in self._prefetch_start_times:
                self._prefetch_start_times[req_id] = time.perf_counter()
                self._prefetch_tracker.on_prefetch_start()
            return None

        if req_id in self._prefetch_start_times:
            duration_ms = (time.perf_counter() - self._prefetch_start_times.pop(req_id)) * 1000
            self._prefetch_tracker.on_prefetch_complete(duration_ms, ready.num_hit_blocks)
        return ready

    def _query_hybrid(
        self, req_id: str, probe: _QueryProbe, limit: int
    ) -> ShardedQueryReady | None:
        def miss():
            return ShardedQueryReady(
                0, probe.leases, attention_hit_blocks=probe.attention_hit_blocks
            )

        if time.monotonic() - probe.started_at >= self._HYBRID_QUERY_WAIT_SECONDS:
            logger.warning("Hybrid cache query wait expired; recomputing request %s", req_id)
            return miss()
        group_ids = tuple(group for group, _, _ in self._cache_groups.recovery_groups)
        vbs = self._ctx.virtual_block_size
        origin = probe.computed_blocks * vbs
        assert self._recovery is not None
        if not probe.selected_boundary:
            loading = False
            for group in group_ids:
                if probe.candidates.get(group) is not None:
                    continue
                probe.candidates[group] = None
                result = self._tp_shard_client.candidates(
                    self._ctx.instance_id, probe.native_hashes, req_id, group
                )
                if result is None:
                    loading = True
                else:
                    probe.candidates[group] = result
            if loading:
                return None
            probe.attention_hit_blocks = min(map(len, probe.candidates[0]))
            shards = [
                [(group, probe.candidates[group][rank]) for group in group_ids]
                for rank in range(len(self._clients))
            ]
            # Rust applies the engine's final-token budget before any reads.
            probe.selected_boundary = (
                self._recovery.select_boundary(
                    self._ctx.namespace,
                    origin,
                    origin + len(probe.query_hashes) * vbs,
                    shards,
                    limit,
                )
                or 0
            )
            if not probe.selected_boundary:
                return miss()
        loading = False
        for group in group_ids:
            if probe.groups.get(group) is not None:
                continue
            probe.groups[group] = None
            result = self._tp_shard_client.read_recovery(
                self._ctx.instance_id,
                probe.native_hashes,
                req_id,
                self._recovery,
                self._ctx.namespace,
                origin,
                probe.selected_boundary,
                group,
            )
            if result is None:
                loading = True
                continue
            probe.groups[group] = result
            if not result:
                return miss()
        if loading:
            return None
        attention = probe.groups.pop(0)
        assert attention is not None
        probe.leases = tuple(lease for _, lease in attention)
        hit_blocks = (probe.selected_boundary - origin) // vbs
        probe.leased_blocks = hit_blocks
        groups = [probe.groups[group] for group in group_ids if group]
        return ShardedQueryReady(
            hit_blocks,
            probe.leases,
            attention_hit_blocks=probe.attention_hit_blocks,
            recovery_hold=RecoveryLoadHold(
                leases=tuple(tuple(lease for _, lease in group) for group in groups),
                hit_positions=tuple(tuple(positions for positions, _ in group) for group in groups),
                last_position=hit_blocks - 1,
            ),
            boundary=probe.selected_boundary,
        )

    def _cancel_prefetch_tracking(self, req_id: str) -> None:
        """Withdraw pending query interest and finish its metrics."""
        if req_id not in self._prefetch_start_times:
            return

        self._tp_shard_client.cancel(self._ctx.instance_id, req_id)

        started_at = self._prefetch_start_times.pop(req_id)
        self._prefetch_tracker.on_prefetch_cancel()
        waited_ms = (time.perf_counter() - started_at) * 1000
        logger.warning(
            "[OrbitKVConnector] Prefetch aborted before ready: req=%s waited_ms=%.2f "
            "pending_prefetches=%d",
            req_id,
            waited_ms,
            self._prefetch_tracker.pending_prefetches,
        )

    def get_stats(self) -> OrbitKVConnectorStats | None:
        """Get current connector stats for metrics exposure."""
        # Get stats from prefetch tracker
        prefetch_stats = self._prefetch_tracker.get_stats()

        data: dict = {
            "pending_prefetches": prefetch_stats["pending_prefetches"],
            "prefetch_duration": prefetch_stats["prefetch_duration"],
            "prefetch_blocks": prefetch_stats["prefetch_blocks"],
        }

        stats = OrbitKVConnectorStats(data=data)
        if stats.is_empty():
            return None
        return stats

    def reset_cache(self) -> bool:
        """Invalidate future external reuse only after all request ownership ends."""
        if (
            self._requests
            or self._pending_query_probes
            or self._pending_load_intents
            or self._pending_saves
            or self._held_requests
            or self._pinned_boundary_jobs
            or self._query_controls
            or self._queued_at
        ):
            return False
        self._cache_generation = uuid.uuid4().bytes
        self._generation_hashes.clear()
        return True

    def shutdown(self) -> None:
        for req_id in list(self._pending_query_probes):
            self._release_pending_query_probe(req_id)
        for req_id in self._queued_at:
            for client in self._clients:
                client.cancel_query(self._ctx.instance_id, req_id)
        self._queued_at.clear()

    def _release_pending_query_probe(self, req_id: str) -> bool:
        probe = self._pending_query_probes.pop(req_id, None)
        if probe is None:
            return True

        return self._release_query_probe(req_id, probe)

    def _release_query_probe(self, req_id: str, probe: _QueryProbe) -> bool:
        released = True
        if probe.control_id:
            try:
                self._ctx.client.release_shard_query(probe.control_id)
            except Exception:
                logger.exception("Could not release shard query: req=%s", req_id)
                released = False
        elif probe.leases and any(probe.leases):
            released = self._tp_shard_client.release(probe.leases, req_id)
        self._cancel_prefetch_tracking(req_id)
        for group_id in probe.candidates.keys() - probe.groups.keys():
            self._tp_shard_client.cancel(self._ctx.instance_id, req_id, group_id)
        for group_id, group in probe.groups.items():
            self._tp_shard_client.cancel(self._ctx.instance_id, req_id, group_id)
            if group is not None:
                released = (
                    self._tp_shard_client.release(tuple(lease for _, lease in group), req_id)
                    and released
                )
        hold = probe.recovery_hold
        if hold is not None:
            for group_leases in hold.leases:
                if not self._tp_shard_client.release(group_leases, req_id):
                    released = False
        return released


__all__ = ["SchedulerAdapter"]
