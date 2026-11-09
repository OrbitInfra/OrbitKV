"""SGLang registration configuration and model-scoped cache identity."""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from orbitkv.identity import (
    artifact_identity,
    model_identity,
    state_namespace,
    static_adapter_identity,
)

if TYPE_CHECKING:
    from .layout import GpuLayout, GpuPool


def _component_format(pool: GpuPool) -> dict:
    if pool.kind == "mla":
        native = pool.entry.device_pool
        return {
            "mla": {
                "latent": native.kv_lora_rank,
                "rope": native.qk_rope_head_dim,
                "physical": native.kv_cache_dim,
            }
        }
    if pool.kind in {"recurrent", "convolution"}:
        native = pool.entry.device_pool
        state = native.mamba_cache
        return {
            "checkpoint": {
                "conv_buffers": len(state.conv),
                "conv_window_axis": native.conv_window_axis,
                "temporal": bool(state.temporal.numel()),
            }
        }
    return {}


def resolve_transfer_backend() -> str:
    backend = os.environ.get("ORBITKV_TRANSFER_BACKEND", "direct")
    if backend not in {"direct", "kernel"}:
        raise ValueError("ORBITKV_TRANSFER_BACKEND must be direct or kernel")
    return backend


def resolve_static_loras(server_args: Any, lora_config: Any) -> tuple[Any, ...]:
    if not lora_config.enable_lora:
        return ()
    if os.environ.get("ORBITKV_STATIC_LORA") != "1":
        raise ValueError(
            "OrbitKV requires immutable adapter identities; dynamic LoRA is unsupported. "
            "Set ORBITKV_STATIC_LORA=1 only for fixed startup --lora-paths."
        )
    refs = tuple(lora_config.lora_paths or ())
    if not refs:
        raise ValueError("OrbitKV static LoRA requires startup --lora-paths")
    ids = [getattr(ref, "lora_id", None) for ref in refs]
    if any(not isinstance(uid, str) or not uid for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError("OrbitKV static LoRA requires normalized unique startup adapter IDs")
    if getattr(server_args, "enable_session_radix_cache", False):
        raise ValueError("OrbitKV static LoRA does not support session RadixCache")
    return refs


def validate_lora_request(req: Any, static_ids: frozenset[str]) -> None:
    if static_ids and getattr(req, "session", None) is not None:
        raise ValueError("OrbitKV static LoRA does not support streaming sessions")
    uid = getattr(req, "lora_id", None)
    if uid is not None and uid not in static_ids:
        raise ValueError("OrbitKV request selects an undeclared static LoRA adapter")
    if static_ids and getattr(req, "extra_key", None) != uid:
        # SGLang concatenates user extra_key and adapter UID. A user key equal
        # to a LoRA UID would otherwise alias its native HBM pages.
        raise ValueError("OrbitKV static LoRA does not support caller extra_key")


def derive_namespace(
    server_args: Any,
    params: Any,
    layout: GpuLayout,
    *,
    lora_config: Any,
    static_loras: tuple[Any, ...],
) -> str:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp_rank = parallel.tp_rank
    tp_size = parallel.tp_size
    computation = {
        "weight_version": getattr(server_args, "weight_version", None),
        "quantization": getattr(server_args, "quantization", None),
        "model_overrides": getattr(server_args, "json_model_override_args", None),
        "dtype": server_args.dtype,
        "attention_backend": server_args.attention_backend,
        "prefill_attention_backend": server_args.prefill_attention_backend,
        "decode_attention_backend": server_args.decode_attention_backend,
        "mamba_backend": getattr(server_args, "mamba_backend", None),
        "linear_attention": {
            name: getattr(server_args, name, None)
            for name in (
                "linear_attn_backend",
                "linear_attn_decode_backend",
                "linear_attn_prefill_backend",
                "linear_attn_verify_backend",
                "enable_mamba_cache_stochastic_rounding",
                "mamba_cache_philox_rounds",
            )
        },
    }
    if lora_config.enable_lora:
        computation["static_lora"] = {
            "adapters": static_adapter_identity(
                tuple((ref.lora_name, ref.lora_path) for ref in static_loras)
            ),
            "native_ids": sorted((ref.lora_name, ref.lora_id) for ref in static_loras),
            "configuration": {
                name: getattr(lora_config, name, None)
                for name in (
                    "lora_backend",
                    "max_lora_rank",
                    "max_loras_per_batch",
                    "max_lora_chunk_size",
                    "experts_shared_outer_loras",
                    "lora_use_virtual_experts",
                    "lora_strict_loading",
                )
            },
            "target_modules": sorted(lora_config.lora_target_modules or ()),
        }
    scale_path = getattr(server_args, "quantization_param_path", None)
    representation = {
        "kv_scale_artifact": artifact_identity(scale_path) if scale_path else None,
        "kv_cache_dtype": getattr(server_args, "kv_cache_dtype", None),
        "tp": [tp_rank, tp_size],
        "pp": [params.pp_rank, params.pp_size],
        "cp": [params.attn_cp_rank, params.attn_cp_size],
        "page_size": layout.page_size,
        "groups": [
            {
                "group": pool.group_id,
                "kind": pool.kind,
                "window": pool.window,
                "layers": sorted(pool.entry.layer_mapping.items()),
                **_component_format(pool),
                "buffers": [
                    {
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape[1:]),
                        "stride": list(tensor.stride()[1:]),
                        "block_bytes": block_bytes,
                    }
                    for tensor, block_bytes in zip(
                        pool.entry.kv_buffer, pool.block_bytes, strict=True
                    )
                ],
            }
            for pool in layout.pools.values()
        ],
    }
    return state_namespace(
        engine="sglang",
        engine_version=version("sglang"),
        model=model_identity(
            server_args.model_path,
            revision=server_args.revision,
            tokenizer=server_args.tokenizer_path,
            tokenizer_revision=server_args.revision,
        ),
        computation=computation,
        representation=representation,
    )
