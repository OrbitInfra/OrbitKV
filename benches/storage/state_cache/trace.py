"""Seeded synthetic state producers and tool/selection events, never model execution."""

import math
import random
from dataclasses import dataclass, field, fields, is_dataclass, replace

from .model import Boundary, LimitError, Limits, Node, PageLayer, Recipe
from .schema import Schema


@dataclass(frozen=True)
class Sizes:
    # All are explicit assumed physical bytes, including any native scale cost.
    entry_bytes: int
    index_entry_bytes: int
    checkpoint_bytes: int
    page_tokens: int = 128
    alignment: int = 4096
    metadata_per_page: int = 64
    replay_us_per_token: int = 2
    prefill_us_per_token: int = 10
    native_format: str = "unspecified-parameter"
    # Candidate indexer applicability cannot always be proved by config fields.
    indexer_ratios: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self):
        for name, value in vars(self).items():
            if name not in {"native_format", "indexer_ratios"} and (
                type(value) is not int or not 1 <= value <= 1 << 30
            ):
                raise ValueError(f"invalid size/cost parameter: {name}")
        if len(self.native_format) > 128 or len(self.indexer_ratios) > 32:
            raise LimitError("format/indexer parameter budget")


@dataclass(frozen=True)
class Workload:
    seed: int = 1
    family: str = "chat"
    sessions: int = 4
    turns: int = 3
    forks: int = 1
    pages: int = 16
    append_pages: int = 2
    steps: int = 4
    top_k: int = 2
    overlap: float = 0.75
    prefix_fraction: float = 0.5
    tool_mean_us: int = 20000
    tool_sigma: float = 0.5
    prediction_scale: float = 1.0
    checkpoint_spacing: int = 1
    cold: bool = False
    adversarial: bool = True
    layered: bool = False
    lookahead: int = 1
    layer_overlap: float = 0.5
    new_page_locality: float = 0.5

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 63:
            raise ValueError("seed must be an unsigned 63-bit integer")
        if self.family not in {"chat", "tools", "document"}:
            raise ValueError("unknown synthetic family")
        for name in (
            "sessions",
            "turns",
            "pages",
            "append_pages",
            "steps",
            "top_k",
            "checkpoint_spacing",
        ):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 256:
                raise ValueError(f"invalid workload bound: {name}")
        if (
            not 0 <= self.forks <= 32
            or not 0 <= self.overlap <= 1
            or not 0 <= self.prefix_fraction <= 1
        ):
            raise ValueError("invalid fork/overlap parameter")
        if not 1 <= self.tool_mean_us <= 10**9 or not 0 <= self.tool_sigma <= 4:
            raise ValueError("invalid tool distribution")
        if not math.isfinite(self.prediction_scale) or not 0.1 <= self.prediction_scale <= 10:
            raise ValueError("invalid predictor scale")
        if (
            type(self.layered) is not bool
            or type(self.lookahead) is not int
            or not 0 <= self.lookahead <= 2
        ):
            raise ValueError("layer lookahead must be 0, 1 or 2")
        if not 0 <= self.layer_overlap <= 1 or not 0 <= self.new_page_locality <= 1:
            raise ValueError("invalid layer overlap/locality")


@dataclass(frozen=True)
class Input:
    at: int
    kind: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    events: tuple[Input, ...]
    workload: Workload
    sizes: Sizes
    schema: Schema


def input_size(value, budget):
    """Bound nested records, including repeated references; no recursive expansion."""
    stack, count = [iter((value,))], 0
    while stack:
        item = next(stack[-1], StopIteration)
        if item is StopIteration:
            stack.pop()
            continue
        count += 1
        if count > budget or len(stack) > 32:
            raise LimitError("input metadata/reference budget")
        if isinstance(item, str):
            if len(item) > 4096:
                raise LimitError("input string budget")
        elif is_dataclass(item):
            stack.append(iter(tuple(getattr(item, f.name) for f in fields(item))))
        elif isinstance(item, dict):
            stack.append(iter(item.items()))
        elif isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif item is not None and not isinstance(item, (int, float, bool)):
            raise ValueError("unsupported input value")
    return count


def selections(rng, pool, steps, top_k, overlap, adversarial):
    """Only the simulator's query event reveals each selection to the policy."""
    pool = sorted(pool)
    k = min(top_k, len(pool))
    previous = []
    result = []
    for step in range(steps):
        shared = min(len(previous), round(k * overlap)) if step else 0
        if adversarial and step % 4 == 3:
            shared = 0
        keep = rng.sample(previous, shared)
        other = [p for p in pool if p not in previous]
        if len(other) < k - shared:
            other = [p for p in pool if p not in keep]
        previous = sorted(keep + rng.sample(other, k - shared))
        result.append(tuple(previous))
    return tuple(result)


def layer_selections(rng, layers, w):
    """Synthetic overlap targets, never measured model attention.

    Same-layer retention has first claim on k slots, then adjacent-layer reuse.
    Conflicting targets need not be achievable; the simulator reports realized
    intersections. Fresh pages exclude both sets, with a locality bias toward
    the last observed position. Declared consumers copy their source exactly.
    """
    result, previous = [], {}
    for step in range(w.steps):
        current = {}
        adjacent = ()
        for layer in layers:
            if layer.index_source != layer.layer:
                picked = current[layer.index_source]
            else:
                pool = range(len(layer.pages))
                k = min(w.top_k, len(pool))
                past = previous.get(layer.layer, ())
                disjoint = w.adversarial and step % 4 == 3
                keep = (
                    rng.sample(past, min(len(past), round(k * w.overlap))) if not disjoint else []
                )
                candidates = sorted(set(adjacent) - set(keep))
                count = (
                    0
                    if disjoint
                    else min(
                        k - len(keep),
                        max(0, round(k * w.layer_overlap) - len(set(keep) & set(adjacent))),
                    )
                )
                keep += rng.sample(candidates, min(count, len(candidates)))
                fresh = sorted(set(pool) - set(past) - set(adjacent) - set(keep))
                if len(fresh) < k - len(keep):
                    raise ValueError("page pool cannot realize requested overlap")
                anchor = (
                    past[-1] if past else (adjacent[-1] if adjacent else rng.randrange(len(pool)))
                )
                while len(keep) < k:
                    near = sorted(fresh, key=lambda p: (abs(p - anchor), p))[
                        : max(1, len(fresh) // 4)
                    ]
                    page = rng.choice(near if rng.random() < w.new_page_locality else fresh)
                    fresh.remove(page)
                    keep.append(page)
                picked = tuple(sorted(keep))
            if any(p >= len(layer.pages) for p in picked):
                raise ValueError("reused selection outside consumer page range")
            current[layer.layer] = picked
            adjacent = picked
        result.append(tuple(current[layer.layer] for layer in layers))
        previous = current
    return tuple(result)


def generate(schema: Schema, sizes: Sizes, w=None, limits=None) -> Trace:
    """Open-loop producer boundaries; this does not simulate model prefill kernels.

    Producer-ready times are explicit external arrivals. Restore and decode chains
    use simulated receipts, not these arrival times. No dataset or model-name logic.
    """
    limits, w = limits or Limits(), w or Workload()
    if w.sessions * (1 + w.forks * w.turns) > limits.branches:
        raise LimitError("synthetic branch budget")
    rng = random.Random(w.seed)
    paging = schema.paging_layers(sizes.indexer_ratios) if w.layered else ()
    if w.layered and not paging:
        raise ValueError("layer pipeline requires config-declared sparse selection")
    sources = {i: (index, kv) for i, index, kv in paging}
    components = (
        [
            (ci, replace(c, layers=(i,)), i)
            for ci, c in enumerate(schema.components)
            for i in c.layers
        ]
        if w.layered
        else [(ci, c, None) for ci, c in enumerate(schema.components)]
    )
    events, known = [], {}
    counter, references = 0, 0

    def add(at, kind, **data):
        nonlocal references
        if len(events) >= limits.events:
            raise LimitError("input event budget")
        event = Input(at, kind, data)
        references += input_size(event, limits.planner_work * 4 - references)
        events.append(event)

    for session in range(w.sessions):
        previous = None
        previous_turn = 0
        for turn in range(w.turns):
            count = w.pages + (w.append_pages if w.family == "document" else turn * w.append_pages)
            at = turn * w.tool_mean_us * 128 + session * 100
            branch = f"s{session}"
            roots, history, sparse, new = [], [], [], []
            pools = {i: [] for i, _, _ in paging}
            for ci, c, layer in components:
                if c.kind == "indexer" and c.ratio not in sizes.indexer_ratios:
                    continue
                if c.kind == "tail" and count * sizes.page_tokens % c.ratio == 0:
                    continue
                positions = [count - 1] if c.rule == "checkpoint" else range(count)
                for page in positions:
                    shared = page < int(w.pages * w.prefix_fraction) and c.rule != "checkpoint"
                    owner = "prefix" if shared else branch
                    if w.family == "document":
                        # Independent questions share a document, not the previous
                        # question's suffix or recurrent checkpoint.
                        owner = f"doc{session // 2}" if shared else branch
                        if page >= w.pages or c.rule == "checkpoint":
                            owner = f"{branch}:q{turn}"
                    source = sources.get(layer, (layer, layer))[1]
                    alias = source if c.kind in {"kv", "latent", "compressed", "indexer"} else layer
                    tag = f":l{alias}" if w.layered else ""
                    key = f"{owner}:c{ci}{tag}:p{page}"
                    layers = len(c.layers)
                    entry = sizes.index_entry_bytes if c.kind == "indexer" else sizes.entry_bytes
                    payload = ((sizes.page_tokens + c.ratio - 1) // c.ratio) * entry * layers
                    if c.kind == "checkpoint":
                        payload = sizes.checkpoint_bytes * layers
                    elif c.kind == "tail":
                        payload = max(1, (count * sizes.page_tokens) % c.ratio) * entry * layers
                    physical = (
                        (payload + sizes.metadata_per_page + sizes.alignment - 1) // sizes.alignment
                    ) * sizes.alignment
                    if key not in known:
                        if len(known) >= limits.nodes:
                            raise LimitError("synthetic component budget")
                        kind = (
                            "dense"
                            if w.layered
                            and c.kind in {"kv", "latent", "compressed"}
                            and layer not in sources
                            else c.kind
                        )
                        node = Node(key, physical, kind)
                        known[key] = node
                        new.append(node)
                    history.append(key)
                    if c.rule != "window" or page >= count - math.ceil(
                        c.window / sizes.page_tokens
                    ):
                        roots.append(key)
                    if c.kind in {"kv", "latent", "compressed"}:
                        sparse.append(key)
                        if layer in pools:
                            pools[layer].append(key)
            counter += 1
            boundary = Boundary(
                f"b{counter}",
                tuple(dict.fromkeys(roots)),
                count * sizes.page_tokens,
                count * sizes.page_tokens * sizes.prefill_us_per_token,
                (
                    Recipe(
                        previous.roots,
                        (turn - previous_turn)
                        * w.append_pages
                        * sizes.page_tokens
                        * sizes.replay_us_per_token,
                    ),
                )
                if previous and w.family != "document"
                else (),
                tuple(dict.fromkeys(history)),
                tuple(PageLayer(i, index, tuple(pools[i])) for i, index, _ in paging),
            )
            add(
                at,
                "publish",
                branch=branch,
                boundary=boundary,
                nodes=tuple(new),
                cold=w.cold,
                checkpoint_save=turn % w.checkpoint_spacing == 0,
            )
            targets = [branch]
            for fork in range(w.forks):
                child = f"{branch}:t{turn}:f{fork}"
                add(at + 1, "fork", parent=branch, child=child)
                targets.append(child)
            for ti, target in enumerate(targets):
                case = (session + turn + ti) % 6 if w.adversarial else -1
                delay = min(
                    w.tool_mean_us * 4,
                    round(rng.lognormvariate(math.log(w.tool_mean_us), w.tool_sigma)),
                )
                if w.family == "chat":
                    delay //= 2
                elif w.family == "document":
                    delay *= 2
                if case == 0:
                    delay = 0
                elif case == 1:
                    delay *= 8
                start = at + 10
                # Oracle data lives in the input only, never in History or pack().
                add(
                    start,
                    "tool_start",
                    branch=target,
                    oracle_return=start + delay,
                    prediction_scale=w.prediction_scale,
                )
                if case == 2:
                    add(start + w.tool_mean_us * 16, "expire", branch=target)
                    continue
                if case == 3:
                    add(start + delay // 2, "cancel", branch=target)
                    continue
                add(
                    start + delay,
                    "tool_return",
                    branch=target,
                    selections=layer_selections(rng, boundary.layers, w)
                    if w.layered
                    else selections(rng, sparse, w.steps, w.top_k, w.overlap, w.adversarial),
                )
            if turn % w.checkpoint_spacing == 0:
                previous, previous_turn = boundary, turn
    return Trace(tuple(sorted(events, key=lambda e: e.at)), w, sizes, schema)
