"""vLLM deployment configuration, model identity, and rank topology."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING

from orbitkv.identity import (
    model_config_identity,
    model_identity,
    state_namespace,
    static_adapter_identity,
)
from orbitkv.logging_utils import get_connector_logger

if TYPE_CHECKING:
    from orbitkv import CacheManagerClient

logger = get_connector_logger()
_TRANSFER_BACKENDS = ("direct", "kernel")


class OrbitKVConnectorMode(str, Enum):
    """Read/write behavior for the OrbitKV connector."""

    READ_WRITE = "read_write"
    SAVE_ONLY = "save_only"

    @classmethod
    def from_config(cls, value: object) -> OrbitKVConnectorMode:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for mode in cls:
                if normalized == mode.value:
                    return mode
        allowed = ", ".join(mode.value for mode in cls)
        raise ValueError(f"Unsupported orbitkv.mode {value!r}; expected one of: {allowed}")


@dataclass(frozen=True)
class TpShardTopology:
    """Equal contiguous TP shards served by node-local OrbitKV instances."""

    endpoints: tuple[str, ...]
    global_tp_size: int
    global_world_size: int

    @classmethod
    def from_config(
        cls,
        default_endpoint: str,
        configured_endpoints: object,
        global_tp_size: int,
        global_world_size: int,
    ) -> TpShardTopology:
        if configured_endpoints is None:
            endpoints = (default_endpoint,)
        elif not isinstance(configured_endpoints, (list, tuple)):
            raise ValueError("orbitkv.tp_shard_endpoints must be a list of endpoints")
        else:
            endpoints = tuple(configured_endpoints)

        if not endpoints or any(
            not isinstance(endpoint, str) or not endpoint for endpoint in endpoints
        ):
            raise ValueError("orbitkv.tp_shard_endpoints must contain non-empty strings")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("orbitkv.tp_shard_endpoints must not contain duplicates")
        if global_tp_size <= 0 or global_tp_size % len(endpoints) != 0:
            raise ValueError(
                f"tensor_parallel_size={global_tp_size} must be divisible by "
                f"the {len(endpoints)} OrbitKV TP shards"
            )
        if global_world_size <= 0 or global_world_size % len(endpoints) != 0:
            raise ValueError(
                f"world_size={global_world_size} must be divisible by "
                f"the {len(endpoints)} OrbitKV TP shards"
            )
        return cls(
            endpoints=endpoints,
            global_tp_size=global_tp_size,
            global_world_size=global_world_size,
        )

    @property
    def shard_count(self) -> int:
        return len(self.endpoints)

    @property
    def local_tp_size(self) -> int:
        return self.global_tp_size // self.shard_count

    @property
    def local_world_size(self) -> int:
        return self.global_world_size // self.shard_count

    def shard_index(self, tp_rank: int) -> int:
        if tp_rank < 0 or tp_rank >= self.global_tp_size:
            raise ValueError(
                f"tp_rank={tp_rank} is outside tensor_parallel_size={self.global_tp_size}"
            )
        return tp_rank // self.local_tp_size

    def local_tp_rank(self, tp_rank: int) -> int:
        return tp_rank % self.local_tp_size

    def namespace(self, base_namespace: str, shard_index: int) -> str:
        if self.shard_count == 1:
            return base_namespace
        if shard_index < 0 or shard_index >= self.shard_count:
            raise ValueError(
                f"TP shard index {shard_index} is outside shard_count={self.shard_count}"
            )
        return f"{base_namespace}:tp-shard-{shard_index}-of-{self.shard_count}"


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""

    instance_id: str
    namespace: str
    block_size: int
    tp_size: int
    world_size: int
    tp_rank: int | None
    device_id: int | None
    client: CacheManagerClient
    is_mla: bool = False
    collapse_mla_tp: bool = True
    transfer_backend: str = "direct"
    dcp_world_size: int = 1
    pcp_world_size: int = 1
    dcp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    mode: OrbitKVConnectorMode = OrbitKVConnectorMode.READ_WRITE
    wait_for_full_prefix: bool = False
    query_control: bool = False
    tp_shards: TpShardTopology | None = None
    # Token span of one `Request.block_hashes` entry; `None` means one per
    # scheduler block.
    hash_block_size: int | None = None
    static_loras: tuple[tuple[str, str], ...] = ()

    @property
    def read_enabled(self) -> bool:
        return self.mode is OrbitKVConnectorMode.READ_WRITE

    @property
    def virtual_block_size(self) -> int:
        """Block size as seen by the scheduler.

        vLLM's scheduler block size is ``block_size * dcp``. PCP changes
        which ranks process a request, but does not change the token
        granularity of scheduler block hashes.
        """
        return self.block_size * self.dcp_world_size

    @property
    def hash_scale(self) -> int:
        """`Request.block_hashes` entries per scheduler block.

        vLLM hashes every `hash_block_size` tokens, which is finer than the
        scheduler block for hybrid models (GCD of the group block sizes, or
        `--prefix-match-unit`). Each hash chains over its whole prefix, so the
        last one inside a block is that block's key.
        """
        return self.virtual_block_size // (self.hash_block_size or self.virtual_block_size)

    @property
    def effective_tp_rank(self) -> int:
        """TP rank for OrbitKV server calls.

        - MLA without DCP: 0 (data identical across TP ranks).
        - MLA with DCP: dcp_rank (each DCP rank stores different interleaved tokens).
        - Hybrid MLA: tp_rank (non-MLA cache groups differ across TP ranks).
        - Non-MLA: tp_rank (each TP rank has different KV heads, already unique).
        """
        if self.is_mla and self.collapse_mla_tp:
            return self.dcp_rank
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def effective_tp_size(self) -> int:
        """TP size for OrbitKV server calls.

        - MLA without DCP: 1.
        - MLA with DCP: dcp_world_size.
        - Hybrid MLA: tp_size.
        - Non-MLA: tp_size (unique per TP rank regardless of DCP).
        """
        if self.is_mla and self.collapse_mla_tp:
            return max(1, self.dcp_world_size)
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def effective_world_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_world_size
        return self.world_size

    @property
    def local_physical_tp_rank(self) -> int:
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def local_physical_tp_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def tp_shard_index(self) -> int:
        if self.tp_shards is None or self.tp_rank is None:
            return 0
        return self.tp_shards.shard_index(self.tp_rank)

    @property
    def tp_shard_count(self) -> int:
        return self.tp_shards.shard_count if self.tp_shards is not None else 1


def parse_env_int(name: str, default: int) -> int:
    """Parse an integer from environment variable with fallback to default.

    Note: This function is typically called at module import time for class-level
    configuration. Changing the environment variable after module import will not
    affect values that were already read.

    Args:
        name: Environment variable name.
        default: Default value if env var is not set or invalid.

    Returns:
        Parsed integer value or default.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %d", name, value, default)
        return default


def resolve_instance_id(vllm_config, dp_rank_suffix: bool = True) -> str:
    """Resolve or generate connector instance_id with optional DP rank suffix."""
    instance_id = vllm_config.kv_transfer_config.engine_id
    if instance_id:
        logger.debug("[OrbitKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get("ORBITKV_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.debug(
            "[OrbitKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id,
        )

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.debug(
                    "[OrbitKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                    instance_id,
                    parallel_config.data_parallel_size,
                    local_dp_rank,
                )

    return instance_id


def resolve_static_loras(vllm_config) -> tuple[tuple[str, str], ...]:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    configured = (
        transfer.get_from_extra_config("orbitkv.static_lora_adapters", None)
        if transfer is not None
        else None
    )
    if vllm_config.lora_config is None:
        if configured is not None:
            raise ValueError("orbitkv.static_lora_adapters requires --enable-lora")
        return ()
    if not isinstance(configured, list) or not configured:
        raise ValueError(
            "OrbitKV requires immutable adapter identities; dynamic LoRA is unsupported. "
            "Declare orbitkv.static_lora_adapters for fixed local adapters."
        )
    if os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0") != "0":
        raise ValueError("OrbitKV static LoRA does not support runtime adapter updates")
    adapters = []
    names = set()
    for item in configured:
        if not isinstance(item, dict) or set(item) != {"name", "path"}:
            raise ValueError("orbitkv.static_lora_adapters entries require only name and path")
        name, path = item["name"], item["path"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("static LoRA adapters require unique non-empty names")
        if not isinstance(path, str) or not path:
            raise ValueError("static LoRA adapters require local directory paths")
        names.add(name)
        adapters.append((name, path))
    return tuple(adapters)


def derive_namespace(
    vllm_config,
    tp_size: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    cross_layer_blocks: bool = False,
    hash_block_size: int | None = None,
    static_loras: tuple[tuple[str, str], ...] | None = None,
    kv_cache_config=None,
) -> str:
    """Resolve the model computation and cache representation once per connector."""
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    additional_config = getattr(vllm_config, "additional_config", None) or {}

    if static_loras is None:
        static_loras = resolve_static_loras(vllm_config)
    artifacts = model_identity(
        model_config.model,
        revision=model_config.revision,
        tokenizer=model_config.tokenizer,
        tokenizer_revision=model_config.tokenizer_revision,
    )
    computation = {
        "hf_config": model_config_identity(json.loads(model_config.hf_config.to_json_string())),
        "quantization": model_config.quantization,
        "attention": vllm_config.attention_config.compute_hash(),
        "kernel": vllm_config.kernel_config.compute_hash(),
        "hash_algorithm": cache_config.prefix_caching_hash_algo,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if vllm_config.lora_config is not None:
        computation["static_lora"] = {
            "configuration": vllm_config.lora_config.compute_hash(),
            "adapters": static_adapter_identity(static_loras),
        }
    factors = {
        "dtype": str(model_config.dtype),
        "kv_cache_layout": cache_config.kv_cache_layout,
        "cache_config": cache_config.compute_hash(),
        "tp_size": tp_size,
        "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager,
        "dcp_world_size": dcp_world_size,
        "pcp_world_size": pcp_world_size,
        "cross_layer_blocks": cross_layer_blocks,
        "mla_layer_split_kv_cache": bool(additional_config.get("mla_layer_split_kv_cache", False)),
        "hash_block_size": hash_block_size,
        "block_size": getattr(cache_config, "block_size", None),
        "mamba_cache_mode": getattr(cache_config, "mamba_cache_mode", None),
        "mamba_ssm_cache_dtype": getattr(cache_config, "mamba_ssm_cache_dtype", None),
    }

    groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if len(groups) > 1:
        factors["cache_groups"] = [
            {
                "layers": tuple(group.layer_names),
                "spec_type": type(group.kv_cache_spec).__name__,
                "spec": {key: str(value) for key, value in asdict(group.kv_cache_spec).items()},
            }
            for group in groups
        ]

    return state_namespace(
        engine="vllm",
        engine_version=version("vllm"),
        model=artifacts,
        computation=computation,
        representation=factors,
    )


def detect_mla(vllm_config) -> bool:
    """Detect if the model uses Multi-head Latent Attention (e.g. DeepSeek V2/V3)."""
    hf_config = vllm_config.model_config.hf_text_config
    return getattr(hf_config, "kv_lora_rank", None) is not None


def resolve_transfer_backend(is_mla: bool, override: str | None) -> str:
    """Pick the engine's H2D/D2H backend for this model.

    MLA models save/load many small, highly fragmented slots where the kernel
    backend's single launch beats one cuMemcpyAsync per slot; everything else
    defaults to direct (best bandwidth for few/large transfers). A non-empty
    `override` (from `orbitkv.transfer_backend`) wins, and an unknown value is
    rejected rather than silently falling back.
    """
    if override is None:
        return "kernel" if is_mla else "direct"
    normalized = override.strip().lower()
    if normalized not in _TRANSFER_BACKENDS:
        allowed = ", ".join(_TRANSFER_BACKENDS)
        raise ValueError(
            f"Unsupported orbitkv.transfer_backend {override!r}; expected one of: {allowed}"
        )
    return normalized
