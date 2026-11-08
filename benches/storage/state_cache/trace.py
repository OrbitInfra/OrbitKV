"""Seeded synthetic state producers and tool/selection events, never model execution."""

import math
import random
from dataclasses import dataclass, field, fields, is_dataclass

from .model import Boundary, LimitError, Limits, Node, Recipe
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


def generate(schema: Schema, sizes: Sizes, w=None, limits=None) -> Trace:
    """Open-loop producer boundaries; this does not simulate model prefill kernels.

    Producer-ready times are explicit external arrivals. Restore and decode chains
    use simulated receipts, not these arrival times. No dataset or model-name logic.
    """
    limits, w = limits or Limits(), w or Workload()
    if w.sessions * (1 + w.forks * w.turns) > limits.branches:
        raise LimitError("synthetic branch budget")
    rng = random.Random(w.seed)
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
            count = w.pages + turn * w.append_pages
            at = turn * w.tool_mean_us * 128 + session * 100
            branch = f"s{session}"
            roots, history, sparse, new = [], [], [], []
            for ci, c in enumerate(schema.components):
                if c.kind == "indexer" and c.ratio not in sizes.indexer_ratios:
                    continue
                if c.kind == "tail" and count * sizes.page_tokens % c.ratio == 0:
                    continue
                positions = [count - 1] if c.rule == "checkpoint" else range(count)
                for page in positions:
                    shared = page < int(w.pages * w.prefix_fraction) and c.rule != "checkpoint"
                    owner = "prefix" if shared else branch
                    key = f"{owner}:c{ci}:p{page}"
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
                        node = Node(key, physical, c.kind)
                        known[key] = node
                        new.append(node)
                    history.append(key)
                    if c.rule != "window" or page >= count - math.ceil(
                        c.window / sizes.page_tokens
                    ):
                        roots.append(key)
                    if c.kind in {"kv", "latent", "compressed"}:
                        sparse.append(key)
            counter += 1
            boundary = Boundary(
                f"b{counter}",
                tuple(roots),
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
                if previous
                else (),
                tuple(history),
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
                    selections=selections(rng, sparse, w.steps, w.top_k, w.overlap, w.adversarial),
                )
            if turn % w.checkpoint_spacing == 0:
                previous, previous_turn = boundary, turn
    return Trace(tuple(sorted(events, key=lambda e: e.at)), w, sizes, schema)
