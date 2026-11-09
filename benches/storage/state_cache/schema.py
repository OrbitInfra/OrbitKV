"""Extract declarative facts; keep physical sizes and replay semantics unresolved."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .sources import SOURCES

MAX_CONFIG_BYTES = 1 << 20
MAX_LAYERS = 512


@dataclass(frozen=True)
class Component:
    kind: str
    layers: tuple[int, ...]
    rule: str
    ratio: int = 1
    window: int = 0


@dataclass(frozen=True)
class Schema:
    source: str
    revision: str
    sha256: str
    layers: int
    components: tuple[Component, ...]
    facts: dict
    unresolved: tuple[str, ...]
    engine_schema_version: str = "unavailable-config-derived-ir-v1"

    def record(self):
        return asdict(self)

    def paging_layers(self, indexer_ratios):
        """(layer, index source, KV owner), using declared group starts only.

        Source-ID lists start contiguous groups; indexer_types='shared' reuses
        the preceding full indexer. Absent declarations imply independent layers.
        Native packing remains a size parameter, not an inferred byte layout.
        """
        active = sorted(
            i
            for c in self.components
            if c.kind == "indexer" and c.ratio in indexer_ratios
            for i in c.layers
        )
        maps = []
        for field in ("index_source_layer_ids", "kv_source_layer_ids"):
            starts = self.facts.get(field)
            if starts is None and field.startswith("index_") and "indexer_types" in self.facts:
                types = self.facts["indexer_types"]
                if len(types) != self.layers or set(types) - {"full", "shared"}:
                    raise ValueError("unsupported indexer reuse pattern")
                starts = [i for i in active if types[i] == "full"]
            if starts is None:
                maps.append({i: i for i in active})
                continue
            if not starts or starts != sorted(set(starts)) or set(starts) - set(active):
                raise ValueError("reuse sources must be ordered active sparse layers")
            if active and starts[0] != active[0]:
                raise ValueError("reuse consumer has no preceding source")
            maps.append({i: max(s for s in starts if s <= i) for i in active})
        shapes = {
            i: (c.kind, c.ratio)
            for c in self.components
            if c.kind in {"kv", "latent", "compressed"}
            for i in c.layers
        }
        if any(shapes[i] != shapes[maps[1][i]] for i in active):
            raise ValueError("KV reuse crosses incompatible component shapes")
        return tuple((i, maps[0][i], maps[1][i]) for i in active)


def integer(value, name, minimum=1, maximum=1 << 30):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def describe(config: dict, source="synthetic", revision="fixture", sha256="fixture") -> Schema:
    """Generic config-field lowering, not an engine manifest or model-name table.

    Preserve sharing declarations for optional layer lowering. Logical source
    groups can alias an owner; native packing and additional aliases remain
    unresolved rather than inferred from model names.
    Draft/vision/embedding state is outside the text-backbone simulation.
    """
    c = config.get("text_config", config)
    n = integer(c["num_hidden_layers"], "num_hidden_layers", maximum=MAX_LAYERS)
    kinds = c.get("layer_types")
    linear = c.get("linear_attn_config", {})
    if kinds is None and linear.get("kda_layers") is not None:
        a, b = linear["kda_layers"], linear["full_attn_layers"]
        # Configs declare either a complete zero-based or a complete one-based partition.
        indices = sorted(a + b)
        origin = 0 if indices == list(range(n)) else 1
        if indices != list(range(origin, n + origin)):
            raise ValueError("linear attention layer lists must partition the backbone")
        kinds = ["linear_attention" if i + origin in a else "full_attention" for i in range(n)]
    if kinds is None:
        kinds = ["full_attention"] * n
    if len(kinds) != n or set(kinds) - {
        "linear_attention",
        "full_attention",
        "sliding_attention",
        "deepseek_sparse_attention",
    }:
        raise ValueError("unknown or incomplete layer pattern")
    ratios = c.get("compress_ratios")
    if ratios is not None:
        if len(ratios) < n:
            raise ValueError("compression pattern shorter than backbone")
        ratios = [integer(x, "compression", minimum=0) for x in ratios[:n]]
    sparse = c.get("sparse_attention_config", {})
    sparse_layers = sparse.get("sparse_attention_freq", [False] * n)
    if len(sparse_layers) != n:
        raise ValueError("sparse pattern shorter than backbone")
    window = c.get("sliding_window") or 0
    groups = {}

    def add(kind, i, rule, ratio=1, width=0):
        groups.setdefault((kind, rule, ratio, width), []).append(i)

    for i, kind in enumerate(kinds):
        if kind == "linear_attention":
            add("checkpoint", i, "checkpoint")
            # Conv state size is explicitly included in the caller's checkpoint size.
            continue
        if kind == "sliding_attention" or ratios is not None:
            integer(window, "sliding_window")
            add("window", i, "window", width=window)
        if kind == "sliding_attention" or (ratios is not None and ratios[i] == 0):
            continue
        ratio = ratios[i] if ratios is not None else 1
        main = "compressed" if ratios is not None else "latent" if c.get("kv_lora_rank") else "kv"
        add(main, i, "prefix", ratio)
        if ratio > 1:
            add("tail", i, "checkpoint", ratio)
        indexed = c.get("index_topk") or c.get("indexer_budget") or sparse_layers[i]
        if indexed:
            add("indexer", i, "prefix", c.get("indexer_compress_ratio", ratio))
    facts = {
        k: c[k]
        for k in (
            "max_position_embeddings",
            "head_dim",
            "global_head_dim",
            "num_key_value_heads",
            "num_global_key_value_heads",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "mamba_ssm_dtype",
            "linear_conv_kernel_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "indexer_types",
            "index_topk",
            "indexer_budget",
            "index_head_dim",
            "index_n_heads",
            "indexer_head_dim",
            "indexer_n_heads",
            "index_kpool",
            "index_kpool_always_select_tail",
            "num_kv_shared_layers",
        )
        if k in c
    }
    facts["linear_attn_config"] = linear
    facts["sparse_attention_config"] = sparse
    return Schema(
        source,
        revision,
        sha256,
        n,
        tuple(Component(k, tuple(v), rule, ratio, w) for (k, rule, ratio, w), v in groups.items()),
        facts,
        (
            "native bytes/scales/padding per physical entry",
            "native packing and undeclared physical aliases",
            "page tokens and alignment",
            "recurrent/conv and compression-carry bytes",
            "replay legality and cost",
            "query-dependent sparse selections",
            "indexer applicability for each compression ratio; parameterized candidates only",
            "draft/vision/embedding state excluded",
        ),
    )


def load_config(model: str, path: Path) -> Schema:
    revision, expected = SOURCES[model]
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_CONFIG_BYTES or digest != expected:
        raise ValueError(f"config does not match pinned revision/hash for {model}")
    return describe(json.loads(raw), model, revision, digest)
