"""vLLM connector entry point with scheduler and worker-owned implementations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_rank

from orbitkv.client.connection import connect_cache
from orbitkv.client.gpu import resolve_device_id as _resolve_device_id
from orbitkv.logging_utils import get_connector_logger
from orbitkv.vllm.config import (
    ConnectorContext,
    OrbitKVConnectorMode,
    TpShardTopology,
    derive_namespace,
    detect_mla,
    resolve_instance_id,
    resolve_static_loras,
    resolve_transfer_backend,
)
from orbitkv.vllm.layout import CacheGroupLayout
from orbitkv.vllm.metadata import OrbitKVConnectorMetadata, OrbitKVQueryHandshake
from orbitkv.vllm.metrics import OrbitKVConnectorStats, OrbitKVPromMetrics

if TYPE_CHECKING:
    from orbitkv.vllm.scheduler import SchedulerAdapter
    from orbitkv.vllm.worker import WorkerAdapter

logger = get_connector_logger()


class OrbitKVConnector(KVConnectorBase_V1, SupportsHMA):
    """v1 KV connector for OrbitKV with separated scheduler/worker logic."""

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config) -> bool:
        # Sleep remaps tensor storage; the current registration cannot follow it.
        return False

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)

        if vllm_config.kv_transfer_config is None:
            raise ValueError("OrbitKV requires kv_transfer_config")
        instance_id = resolve_instance_id(vllm_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        world_size = vllm_config.parallel_config.world_size
        is_mla = detect_mla(vllm_config)
        cache_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if len(cache_groups) > 1:
            from vllm.v1.kv_cache_interface import (
                FullAttentionSpec,
                MambaSpec,
                MLAAttentionSpec,
                SlidingWindowSpec,
            )

            specs = tuple(group.kv_cache_spec for group in cache_groups)
            if any(isinstance(spec, MambaSpec) for spec in specs):
                raise RuntimeError("OrbitKV HMA rejects recurrent/Mamba state: hand-off unproven")
            if vllm_config.use_v2_model_runner:
                raise RuntimeError("OrbitKV V2 rejects multiple cache groups")
            if is_mla or any(isinstance(spec, MLAAttentionSpec) for spec in specs):
                raise RuntimeError("OrbitKV HMA rejects MLA with multiple cache groups")
            if any(type(spec) not in (FullAttentionSpec, SlidingWindowSpec) for spec in specs):
                raise RuntimeError("OrbitKV HMA rejects unknown cache specs; require Full + SWA")
            if kv_cache_config.needs_kv_cache_zeroing:
                raise RuntimeError("OrbitKV HMA rejects groups requiring KV zeroing")
            if getattr(vllm_config, "speculative_config", None) is not None or any(
                spec.is_draft or getattr(spec, "extra_retained_tokens", 0) for spec in specs
            ):
                raise RuntimeError("OrbitKV HMA rejects speculative decoding and draft state")
            if any(group.host_resident for group in cache_groups):
                raise RuntimeError("OrbitKV HMA rejects host-resident engine cache groups")
            if not vllm_config.kv_transfer_config.is_kv_consumer:
                raise RuntimeError("OrbitKV HMA requires a KV consumer role for load retirement")
            if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
                raise RuntimeError("OrbitKV HMA rejects pipeline-parallel group projections")
        if cache_groups:
            CacheGroupLayout.from_config(kv_cache_config)
        if len(cache_groups) > 1:
            logger.info("[OrbitKVConnector] HMA Full+SWA: groups=%d", len(cache_groups))
        get_option = vllm_config.kv_transfer_config.get_from_extra_config
        if vllm_config.use_v2_model_runner:
            if get_option("orbitkv.v2", False) is not True:
                raise RuntimeError(
                    "OrbitKV V2 is unqualified and opt-in: set orbitkv.v2=true for its "
                    "installed gate, or keep VLLM_USE_V2_MODEL_RUNNER=0."
                )
            from vllm.v1.kv_cache_interface import FullAttentionSpec

            if (
                len(cache_groups) != 1
                or type(cache_groups[0].kv_cache_spec) is not FullAttentionSpec
                or kv_cache_config.needs_kv_cache_zeroing
                or getattr(vllm_config, "speculative_config", None) is not None
                or not vllm_config.kv_transfer_config.is_kv_consumer
            ):
                raise RuntimeError(
                    "OrbitKV V2 requires one uniform FullAttention group and a KV "
                    "consumer role, without KV zeroing or speculative decoding; "
                    "pre-forward state writes and deferred save finalization are "
                    "not admitted."
                )
        collapse_mla_tp = is_mla and len(cache_groups) <= 1
        dcp_world_size = (
            getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1) or 1
        )
        pcp_world_size = (
            getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1) or 1
        )
        effective_tp_size = max(1, dcp_world_size) if collapse_mla_tp else tp_size

        if dcp_world_size > 1 and not is_mla:
            logger.warning(
                "[OrbitKVConnector] DCP with non-MLA model detected "
                "(dcp_world_size=%d). effective_tp_rank will use tp_rank only; "
                "KV data may collide across DCP ranks if workers share "
                "the same tp_rank.",
                dcp_world_size,
            )

        block_size = vllm_config.cache_config.block_size
        hash_block_size: int | None = None
        if kv_cache_config is not None:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            if (
                scheduler_block_size != block_size * dcp_world_size
                or scheduler_block_size % hash_block_size != 0
            ):
                raise ValueError(
                    f"vLLM scheduler block {scheduler_block_size} / hash block "
                    f"{hash_block_size} do not fit the connector block {block_size}"
                )

        cross_layer_blocks = os.environ.get("ORBITKV_CROSS_LAYER_BLOCKS", "1") == "1"
        static_loras = resolve_static_loras(vllm_config)
        base_namespace = derive_namespace(
            vllm_config,
            effective_tp_size,
            dcp_world_size,
            pcp_world_size,
            cross_layer_blocks=cross_layer_blocks,
            hash_block_size=hash_block_size,
            static_loras=static_loras,
            kv_cache_config=kv_cache_config,
        )

        tp_rank: int | None = None
        device_id: int | None = None
        dcp_rank: int = 0
        pp_rank: int = 0
        pp_size: int = getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) or 1
        if role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            pp_group = get_pp_group()
            pp_rank = pp_group.rank_in_group
            if dcp_world_size > 1:
                from vllm.distributed.parallel_state import (
                    get_decode_context_model_parallel_rank,
                )

                dcp_rank = get_decode_context_model_parallel_rank()
            if torch.cuda.is_available():
                device_id = _resolve_device_id()

        server_host = os.environ.get("ORBITKV_HOST") or get_option(
            "orbitkv.host", "http://127.0.0.1"
        )
        server_port = os.environ.get("ORBITKV_PORT") or get_option("orbitkv.port", 50055)
        mode = OrbitKVConnectorMode.from_config(
            get_option("orbitkv.mode", OrbitKVConnectorMode.READ_WRITE.value)
        )
        transfer_backend = resolve_transfer_backend(
            is_mla,
            get_option("orbitkv.transfer_backend", None),
        )
        wait_for_full_prefix = bool(get_option("orbitkv.wait_for_full_prefix", False))
        default_endpoint = f"{server_host}:{server_port}"
        tp_shards = TpShardTopology.from_config(
            default_endpoint=default_endpoint,
            configured_endpoints=get_option("orbitkv.tp_shard_endpoints", None),
            global_tp_size=tp_size,
            global_world_size=world_size,
        )
        if tp_shards.shard_count > 1 and (
            world_size != tp_size or dcp_world_size != 1 or pcp_world_size != 1
        ):
            raise ValueError(
                "orbitkv.tp_shard_endpoints currently supports TP-only parallelism; "
                f"got tp_size={tp_size}, world_size={world_size}, "
                f"dcp_world_size={dcp_world_size}, pcp_world_size={pcp_world_size}"
            )
        query_control = get_option("orbitkv.query_control", tp_shards.shard_count > 1)
        if not isinstance(query_control, bool):
            raise ValueError("orbitkv.query_control must be a bool")
        if tp_shards.shard_count > 1 and not query_control:
            raise ValueError("multiple TP shards require native query control")
        if query_control and (
            vllm_config.use_v2_model_runner
            or is_mla
            or pp_size != 1
            or world_size != tp_size
            or dcp_world_size != 1
            or pcp_world_size != 1
            or wait_for_full_prefix
            or (getattr(vllm_config.parallel_config, "data_parallel_size", 1) or 1) != 1
            or bool(getattr(vllm_config.parallel_config, "enable_expert_parallel", False))
            or getattr(vllm_config, "speculative_config", None) is not None
            or os.environ.get("ORBITKV_QUEUE_WARMUP") == "1"
            or os.environ.get("ORBITKV_PREPARE_REQUESTS") == "1"
        ):
            raise ValueError(
                "native query control requires dense TP-only V1 without speculative decoding"
            )
        shard_index = tp_shards.shard_index(tp_rank) if tp_rank is not None else 0
        namespace = tp_shards.namespace(base_namespace, shard_index)
        self._connections = connect_cache(
            endpoints=tp_shards.endpoints,
            shard_index=shard_index,
            all_shards=False,
            get_option=get_option,
        )
        clients = self._connections.clients
        client = clients[self._connections.selected_index]
        logger.debug("[OrbitKVConnector] Connected to Cache Manager at %s", client.bootstrap_socket)
        logger.info(
            "[OrbitKVConnector] Cache Manager channel: transport=%s target=%s",
            client.transport,
            client.bootstrap_socket,
        )

        self._base_namespace = base_namespace
        self._ctx = ConnectorContext(
            instance_id=instance_id,
            namespace=namespace,
            block_size=block_size,
            tp_size=tp_size,
            world_size=world_size,
            tp_rank=tp_rank,
            device_id=device_id,
            client=client,
            is_mla=is_mla,
            collapse_mla_tp=collapse_mla_tp,
            transfer_backend=transfer_backend,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            dcp_rank=dcp_rank,
            pp_rank=pp_rank,
            pp_size=pp_size,
            mode=mode,
            wait_for_full_prefix=wait_for_full_prefix,
            query_control=query_control,
            tp_shards=tp_shards,
            hash_block_size=hash_block_size,
            static_loras=static_loras,
        )

        # MLA attention backends expose no num-layers stride dimension, so vLLM
        # cannot build a cross-layer (uniform) KV cache for MLA. Requesting it
        # is silently ignored upstream and falls back to per-layer — surface
        # that instead of pretending the request was honored.
        self._prefer_cross_layer = cross_layer_blocks and not is_mla and len(cache_groups) <= 1
        if is_mla and cross_layer_blocks:
            logger.warning(
                "[OrbitKVConnector] ORBITKV_CROSS_LAYER_BLOCKS=1 is ignored for MLA "
                "models: cross-layer KV cache is unsupported by MLA attention backends; "
                "using per-layer registration."
            )

        self._scheduler: SchedulerAdapter | None = None
        self._worker: WorkerAdapter | None = None
        try:
            if role == KVConnectorRole.SCHEDULER:
                from orbitkv.vllm.scheduler import SchedulerAdapter

                self._scheduler = SchedulerAdapter(
                    self._ctx,
                    clients=clients,
                    kv_cache_config=kv_cache_config,
                )
                # Open the liveness stream from the scheduler process only. One
                # stream per vllm replica is enough — if any tp worker crashes,
                # the scheduler dies too, closing this stream and triggering
                # server-side cleanup of the instance's CUDA IPC mappings.
                for client in clients:
                    client.start_session_watcher(
                        instance_id,
                        namespace,
                        self._ctx.effective_tp_size,
                        self._ctx.effective_world_size,
                    )
            else:
                from orbitkv.vllm.worker import WorkerAdapter

                self._worker = WorkerAdapter(
                    self._ctx,
                    vllm_config=vllm_config,
                    kv_cache_config=kv_cache_config,
                )
                if query_control and tp_shards.shard_count > 1 and self._ctx.effective_tp_rank == 0:
                    client.start_session_watcher(
                        instance_id,
                        namespace,
                        self._ctx.effective_tp_size,
                        self._ctx.effective_world_size,
                    )

        except BaseException:
            self.shutdown()
            raise

        logger.debug(
            "[OrbitKVConnector] Initialized role=%s instance_id=%s device=%s "
            "tp_rank=%s tp_size=%d pp_rank=%d pp_size=%d world_size=%d namespace=%s "
            "is_mla=%s collapse_mla_tp=%s transfer_backend=%s dcp_world_size=%d "
            "pcp_world_size=%d dcp_rank=%d tp_shard=%d/%d "
            "mode=%s wait_for_full_prefix=%s data_transport=%s",
            role.name,
            instance_id,
            device_id if device_id is not None else "cpu",
            tp_rank if tp_rank is not None else "N/A",
            tp_size,
            pp_rank,
            pp_size,
            world_size,
            namespace,
            is_mla,
            collapse_mla_tp,
            transfer_backend,
            dcp_world_size,
            pcp_world_size,
            dcp_rank,
            shard_index,
            tp_shards.shard_count,
            mode.value,
            wait_for_full_prefix,
            client.transport,
        )

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        if not self._worker:
            return
        metadata = self._get_connector_metadata()
        if metadata is None:
            return
        self._worker.start_load_kv(metadata)
        # Graph replay can call the connector without attention metadata.
        # Its captured graph has no Python layer callbacks, so link the whole
        # Restore on the replay stream. Decode steps without Restore do no work.
        if (
            forward_context.attn_metadata is None
            or forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            self._worker.wait_for_all_layers()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._worker:
            return
        self._worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        # Save is submitted from wait_for_save() using scheduler metadata.
        # Layer callbacks are intentionally ignored so CUDA graph replay
        # cannot suppress save submission.
        pass

    def wait_for_save(self) -> None:
        if not self._worker:
            return
        self._worker.wait_for_save()

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        if not self._worker:
            return KVConnectorTransferResults()
        return self._worker.get_transfer_results(finished_req_ids)

    def build_connector_worker_meta(self):
        if not self._worker:
            return None
        return self._worker.build_connector_worker_meta()

    def get_handshake_metadata(self):
        if not self._worker or not self._ctx.query_control:
            return None
        return OrbitKVQueryHandshake(
            self._ctx.client.export_query_target(
                self._ctx.instance_id,
                self._ctx.namespace,
                self._ctx.effective_tp_size,
                self._ctx.effective_world_size,
            )
        )

    def set_xfer_handshake_metadata(self, metadata: dict[int, Any]) -> None:
        if not self._scheduler or not self._ctx.query_control:
            return
        topology = self._ctx.tp_shards
        if topology is None or set(metadata) != set(range(self._ctx.tp_size)):
            raise ValueError("query handshake must cover every declared global TP rank")
        targets: list[bytes] = []
        for node in range(topology.shard_count):
            ranks = range(node * topology.local_tp_size, (node + 1) * topology.local_tp_size)
            records = [metadata[rank] for rank in ranks]
            if any(
                not isinstance(record, OrbitKVQueryHandshake) or not record.target
                for record in records
            ):
                raise ValueError("missing registered query handshake")
            if any(record.target != records[0].target for record in records):
                raise ValueError("workers in one node declared different query registrations")
            targets.append(records[0].target)
        namespaces = [
            topology.namespace(self._base_namespace, node) for node in range(topology.shard_count)
        ]
        self._ctx.client.configure_shard_queries(
            self._ctx.instance_id,
            self._ctx.namespace,
            topology.local_tp_size,
            topology.local_world_size,
            list(zip(topology.endpoints, namespaces, targets, strict=True)),
        )
        self._scheduler.query_handshake_ready = True
        logger.info(
            "[OrbitKVConnector] registered query handshake configured: nodes=%d ranks=%d",
            topology.shard_count,
            self._ctx.tp_size,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if not self._worker:
            return
        self._worker.register_kv_caches(kv_caches)

    def unregister_context(self) -> None:
        if self._worker:
            self._worker.unregister_context()

    def handle_preemptions(self, metadata: OrbitKVConnectorMetadata) -> None:
        if not self._worker:
            return
        self._worker.handle_preemptions(metadata.preempted_req_ids)

    def on_new_request(self, request) -> None:
        if self._scheduler:
            self._scheduler.on_new_request(request)

    def update_connector_output(self, connector_output) -> None:
        if self._scheduler:
            self._scheduler.update_connector_output(connector_output)

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        if self._scheduler:
            self._scheduler.bind_gpu_block_pool(gpu_block_pool)

    def request_finished(
        self,
        request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler:
            return self._scheduler.request_finished(request, (block_ids,))
        return (False, None)

    def request_finished_all_groups(
        self,
        request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._scheduler:
            return self._scheduler.request_finished(request, block_ids)
        return (False, None)

    def has_pending_push_work(self) -> bool:
        if not self._scheduler:
            return False
        return self._scheduler.has_pending_push_work()

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if not self._scheduler:
            return (0, False)
        return self._scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request,
        blocks,
        num_external_tokens: int,
    ) -> None:
        if self._scheduler:
            self._scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output) -> OrbitKVConnectorMetadata:
        if not self._scheduler:
            return OrbitKVConnectorMetadata()
        return self._scheduler.build_connector_meta(scheduler_output)

    def get_kv_connector_stats(self) -> OrbitKVConnectorStats | None:
        adapter = self._scheduler if self._scheduler is not None else self._worker
        return adapter.get_stats() if adapter is not None else None

    @classmethod
    def build_kv_connector_stats(cls, data: dict | None = None) -> OrbitKVConnectorStats | None:
        if data is None:
            return None
        return OrbitKVConnectorStats(data=data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config,
        metric_types,
        labelnames,
        per_engine_labelvalues,
    ) -> OrbitKVPromMetrics:
        return OrbitKVPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return self._prefer_cross_layer

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        if not self._worker:
            return
        self._worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def reset_cache(self) -> bool:
        if self._scheduler is None:
            return False
        return self._scheduler.reset_cache()

    def shutdown(self):
        try:
            if self._scheduler:
                self._scheduler.shutdown()
            if self._worker:
                self._worker.shutdown()
        finally:
            self._connections.close()


__all__ = ["OrbitKVConnector", "KVConnectorRole"]
