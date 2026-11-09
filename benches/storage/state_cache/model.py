"""Physical-owner ledger with separate readiness, leases and drain receipts."""

from dataclasses import dataclass, field


class LimitError(ValueError):
    """A finite resource limit was reached; reject rather than silently truncate."""


class SafetyError(ValueError):
    """The trace or caller attempted an invalid ownership transition."""


@dataclass(frozen=True)
class Limits:
    nodes: int = 16384
    owners: int = 32768
    events: int = 65536
    processed: int = 262144
    leases: int = 4096
    flights: int = 2048
    branches: int = 256
    candidates: int = 64
    planner_work: int = 65536
    references: int = 65536
    prefetches: int = 256

    def __post_init__(self):
        if any(type(v) is not int or not 1 <= v <= 1 << 22 for v in vars(self).values()):
            raise ValueError("limits must be positive bounded integers")


@dataclass(frozen=True)
class Node:
    key: str
    size: int
    kind: str = "kv"
    dependencies: tuple[str, ...] = ()
    generation: int = 1
    epoch: int = 1
    source: str = "local"
    logical_key: str = ""


@dataclass(frozen=True)
class Recipe:
    roots: tuple[str, ...]
    replay_us: int = 0


@dataclass(frozen=True)
class PageLayer:
    layer: int
    index_source: int
    pages: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    key: str
    roots: tuple[str, ...]
    tokens: int
    recompute_us: int
    alternatives: tuple[Recipe, ...] = ()
    history: tuple[str, ...] = ()
    layers: tuple[PageLayer, ...] = ()


@dataclass
class Replica:
    ready: bool = True
    pins: int = 0
    touched: int = 0


@dataclass
class Flight:
    keys: tuple[str, ...]
    source: str
    destination: str
    ready_at: int
    drained_at: int
    cancelled: bool = False
    ready_seen: bool = False
    targets: tuple[str, ...] = ()


@dataclass
class Ledger:
    capacities: dict[str, int]
    limits: Limits = field(default_factory=Limits)
    nodes: dict[str, Node] = field(default_factory=dict)
    replicas: dict[tuple[str, str], Replica] = field(default_factory=dict)
    leases: dict[int, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    flights: dict[int, Flight] = field(default_factory=dict)
    epochs: dict[str, int] = field(default_factory=lambda: {"local": 1})
    serial: int = 0
    peak_unique_bytes: int = 0
    peak_replicated_bytes: int = 0
    peak_hbm_bytes: int = 0
    peak_owners: int = 0

    def define(self, node: Node):
        if node.key in self.nodes or node.size < 1 or node.generation < 1 or node.epoch < 1:
            raise SafetyError("duplicate or invalid component identity")
        if len(self.nodes) >= self.limits.nodes:
            raise LimitError("component metadata budget")
        if any(k not in self.nodes for k in node.dependencies):
            raise SafetyError("dependencies must precede the component (no cycles)")
        if node.logical_key:
            original = self.nodes.get(node.logical_key)
            if original is None or any(
                getattr(node, f) != getattr(original, f)
                for f in ("size", "kind", "generation", "epoch", "source")
            ):
                raise SafetyError("replicated logical identity does not match its original")
        if max(len(node.key), len(node.source), len(node.logical_key)) > 128:
            raise LimitError("identity length budget")
        if (
            sum(len(n.dependencies) for n in self.nodes.values()) + len(node.dependencies)
            > self.limits.references
        ):
            raise LimitError("dependency edge budget")
        self.epochs.setdefault(node.source, node.epoch)
        self.nodes[node.key] = node

    def closure(self, roots):
        seen, todo, work = set(), list(roots), 0
        while todo:
            work += 1
            if work > self.limits.planner_work:
                raise LimitError("dependency walk budget")
            key = todo.pop()
            if key not in self.nodes:
                raise SafetyError(f"missing component {key}")
            if key not in seen:
                seen.add(key)
                todo.extend(self.nodes[key].dependencies)
        return tuple(sorted(seen))

    def valid(self, key):
        n = self.nodes[key]
        return n.epoch == self.epochs[n.source]

    def available(self, key, tier):
        r = self.replicas.get((key, tier))
        return bool(r and r.ready and self.valid(key))

    def complete(self, roots, tier=None):
        return all(
            self.available(k, tier) if tier else any(self.available(k, t) for t in self.capacities)
            for k in self.closure(roots)
        )

    def used(self, tier):
        return sum(self.nodes[k].size for k, t in self.replicas if t == tier)

    def allocate(self, key, tier, now, ready=True):
        if (key, tier) in self.replicas:
            raise SafetyError("overwriting an existing physical allocation")
        if len(self.replicas) >= self.limits.owners:
            raise LimitError("physical owner budget")
        if self.used(tier) + self.nodes[key].size > self.capacities[tier]:
            raise LimitError(f"{tier} byte budget")
        self.replicas[key, tier] = Replica(ready, 0, now)
        logical = {self.nodes[k].logical_key or k: self.nodes[k].size for k, _ in self.replicas}
        self.peak_unique_bytes = max(self.peak_unique_bytes, sum(logical.values()))
        self.peak_replicated_bytes = max(
            self.peak_replicated_bytes, sum(self.nodes[k].size for k, _ in self.replicas)
        )
        self.peak_hbm_bytes = max(self.peak_hbm_bytes, self.used("hbm"))
        self.peak_owners = max(self.peak_owners, len(self.replicas))

    def evict(self, key, tier):
        r = self.replicas[key, tier]
        if r.pins:
            raise SafetyError("release before drain or reader lease retirement")
        del self.replicas[key, tier]

    def lease(self, roots, tier, now, generations=None):
        keys = self.closure(roots)
        if not self.complete(keys, tier):
            raise SafetyError("stale or incomplete boundary")
        if generations and any(self.nodes[k].generation != g for k, g in generations.items()):
            raise SafetyError("stale allocation generation")
        if len(self.leases) >= self.limits.leases:
            raise LimitError("reader lease budget")
        if self.references() + len(keys) > self.limits.references:
            raise LimitError("lease reference budget")
        self.serial += 1
        refs = tuple(
            (
                k,
                tier
                or next(
                    t
                    for t in ("dram", "ssd", "hbm")
                    if t in self.capacities and self.available(k, t)
                ),
            )
            for k in keys
        )
        for ref in refs:
            self.replicas[ref].pins += 1
            self.replicas[ref].touched = now
        self.leases[self.serial] = refs
        return self.serial

    def references(self):
        return sum(len(v) for v in self.leases.values()) + sum(
            2 * len(f.keys) for f in self.flights.values()
        )

    def release(self, lease):
        if lease not in self.leases:
            raise SafetyError("unknown reader receipt")
        for ref in self.leases.pop(lease):
            self.replicas[ref].pins -= 1

    def submit(self, keys, source, destination, now, ready_at, drained_at, targets=None):
        keys = tuple(keys)
        targets = tuple(targets) if targets is not None else keys
        if (
            not keys
            or ready_at < now
            or drained_at < ready_at
            or (source == destination and keys == targets)
        ):
            raise SafetyError("invalid transfer")
        if (
            len(set(keys)) != len(keys)
            or len(set(targets)) != len(targets)
            or len(keys) != len(targets)
        ):
            raise SafetyError("duplicate or mismatched transfer vectors")
        if any(
            self.nodes[a].size != self.nodes[b].size
            or self.nodes[a].epoch != self.nodes[b].epoch
            or self.nodes[a].generation != self.nodes[b].generation
            for a, b in zip(keys, targets, strict=True)
        ):
            raise SafetyError("copy identity or byte extent mismatch")
        if len(self.flights) >= self.limits.flights:
            raise LimitError("in-flight batch budget")
        if self.references() + 2 * len(keys) > self.limits.references:
            raise LimitError("flight reference budget")
        if any(not self.available(k, source) for k in keys):
            raise SafetyError("source grant has stale or unready bytes")
        if any((k, destination) in self.replicas for k in targets):
            raise SafetyError("destination already owned")
        if len(self.replicas) + len(keys) > self.limits.owners:
            raise LimitError("physical owner budget")
        if (
            self.used(destination) + sum(self.nodes[k].size for k in keys)
            > self.capacities[destination]
        ):
            raise LimitError(f"{destination} byte budget")
        self.serial += 1
        for key, target in zip(keys, targets, strict=True):
            self.allocate(target, destination, now, ready=False)
            self.replicas[key, source].pins += 1
            self.replicas[target, destination].pins += 1
        self.flights[self.serial] = Flight(
            keys, source, destination, ready_at, drained_at, targets=targets
        )
        return self.serial

    def ready(self, flight, now):
        f = self.flights[flight]
        if now < f.ready_at:
            raise SafetyError("premature ready receipt")
        for k in f.targets:
            self.replicas[k, f.destination].ready = not f.cancelled and self.valid(k)
        f.ready_seen = True

    def cancel(self, flight):
        self.flights[flight].cancelled = True

    def drain(self, flight, now):
        f = self.flights[flight]
        if now < f.drained_at or not f.ready_seen:
            raise SafetyError("premature drain receipt")
        for key, target in zip(f.keys, f.targets, strict=True):
            self.replicas[key, f.source].pins -= 1
            self.replicas[target, f.destination].pins -= 1
        del self.flights[flight]

    def retire_epoch(self, source, epoch):
        if epoch <= self.epochs.get(source, 0):
            raise SafetyError("incarnation must advance")
        self.epochs[source] = epoch
        # Existing flights and leases keep every charge until their receipts.
