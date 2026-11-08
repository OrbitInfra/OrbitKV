"""Finite discrete-event execution with one contended byte path and one compute lane.

No native library is imported. Times are integer microseconds; bytes include the
explicit page overhead/alignment. Submission reserves owners until physical drain.
"""

import contextlib
import heapq
import math
from dataclasses import asdict, dataclass, field, replace

from .model import Boundary, Ledger, LimitError, Limits, SafetyError
from .policy import History, Policy, pack
from .trace import Trace, input_size


@dataclass(frozen=True)
class Costs:
    bytes_per_us: int
    latency_us: int
    drain_us: int
    compute_us: int
    indexer_us: int
    ttft_slo_us: int
    tpot_slo_us: int
    ssd_bytes_per_us: int | None = None
    ssd_latency_us: int | None = None

    def __post_init__(self):
        if any(
            type(v) is not int or not 1 <= v <= 10**12
            for k, v in vars(self).items()
            if v is not None or not k.startswith("ssd_")
        ):
            raise ValueError("costs and SLOs must be positive bounded integer parameters")

    def service(self, size, source, destination):
        rate, latency = self.bytes_per_us, self.latency_us
        if "ssd" in (source, destination):
            rate, latency = self.ssd_bytes_per_us or rate, self.ssd_latency_us or latency
        return latency + (size + rate - 1) // rate


@dataclass
class Request:
    branch: str
    boundary: Boundary
    selections: tuple
    arrived: int
    needed: tuple = ()
    step: int = 0
    lease: int | None = None
    source_lease: int | None = None
    last_token: int | None = None
    replay: bool = False
    compute: bool = False


@dataclass
class Metrics:
    requests: int = 0
    completed: int = 0
    rejected: int = 0
    boundary_hits: int = 0
    component_hits: int = 0
    components: int = 0
    hit_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    wire_bytes: int = 0
    destination_bytes: int = 0
    speculative_bytes: int = 0
    speculative_ready_bytes: int = 0
    stall_us: int = 0
    recomputes: int = 0
    replays: int = 0
    forks: int = 0
    peak_unique_bytes: int = 0
    peak_replicated_bytes: int = 0
    peak_hbm_bytes: int = 0
    hbm_byte_us: int = 0
    unique_byte_us: int = 0
    replicated_byte_us: int = 0
    peak_queue: int = 0
    peak_owners: int = 0
    peak_flights: int = 0
    planner_work: int = 0
    cold_ttft: list = field(default_factory=list)
    resume_ttft: list = field(default_factory=list)
    tpot: list = field(default_factory=list)


def quantile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


class Simulation:
    def __init__(
        self, trace: Trace, policy: Policy, capacities, costs: Costs, limits=None, sink=None
    ):
        limits = limits or Limits()
        if set(capacities) != {"hbm", "dram", "ssd"} or any(
            type(v) is not int or not 0 <= v <= 1 << 60 for v in capacities.values()
        ):
            raise ValueError("three bounded tier byte capacities required")
        if len(trace.events) > limits.events:
            raise LimitError("input event budget")
        input_size(trace.events, limits.planner_work * 4)
        self.trace, self.policy, self.costs, self.limits = trace, policy, costs, limits
        self.ledger = Ledger(dict(capacities), limits)
        self.metrics, self.history = Metrics(), History(limits.branches)
        self.queue, self.serial, self.now, self.processed = [], 0, 0, 0
        self.link_free, self.compute_free = 0, 0
        self.branches, self.boundaries, self.tools, self.requests = {}, {}, {}, {}
        self.hits, self.pending, self.computing, self.prefetched = {}, {}, {}, {}
        self.fork_maps, self.fork_ready = {}, {}
        self.sink = sink
        for event in trace.events:
            self.schedule(event.at, event.kind, event.data)

    def schedule(self, at, kind, data):
        if type(at) is not int or at < self.now or len(self.queue) >= self.limits.events:
            raise LimitError("event time or queue budget")
        self.serial += 1
        heapq.heappush(self.queue, (at, self.serial, kind, data))
        self.metrics.peak_queue = max(self.metrics.peak_queue, len(self.queue))

    def closure(self, roots):
        return self.ledger.closure(roots)

    def roots(self, boundary):
        return boundary.roots if self.policy.minimal else boundary.history or boundary.roots

    def mandatory(self, roots):
        return tuple(
            k for k in roots if self.ledger.nodes[k].kind not in {"kv", "latent", "compressed"}
        )

    def protected(self):
        return set().union(*(set(r.needed) for r in self.requests.values()))

    def make_space(self, tier, keys, keep=()):
        needed = sum(
            self.ledger.nodes[k].size for k in keys if (k, tier) not in self.ledger.replicas
        )
        if needed > self.ledger.capacities[tier]:
            raise LimitError(f"working set exceeds {tier}")
        protected = self.protected() | set(keys) if tier == "hbm" else set(keys)
        protected.update(keep)
        if self.policy.dag and tier != "hbm":
            candidates = [(b.roots, self.hits.get(b.key, 0) + 1) for b in self.branches.values()]
            # Keep the most recently observed roots when the bounded search window fills.
            candidates = candidates[-self.limits.candidates :]
            selected, _, work = pack(self.ledger, candidates, self.ledger.capacities[tier] - needed)
            self.metrics.planner_work += work
            protected |= selected
        choices = sorted(
            (r.touched, k)
            for (k, t), r in self.ledger.replicas.items()
            if t == tier and not r.pins and k not in protected
        )
        if self.policy.static and tier != "hbm":
            choices = []
        for _, key in choices:
            if self.ledger.used(tier) + needed <= self.ledger.capacities[tier]:
                break
            self.ledger.evict(key, tier)
            self.prefetched.pop(key, None)
        if self.ledger.used(tier) + needed > self.ledger.capacities[tier]:
            raise LimitError(f"{tier} is retained by live owners or selected closures")

    def submit(
        self, keys, source, destination, speculative=False, retire_source=False, targets=None
    ):
        if not keys:
            return self.now
        targets = tuple(targets) if targets is not None else keys
        self.make_space(destination, targets, keep=keys if source == destination else ())
        size = sum(self.ledger.nodes[k].size for k in keys)
        ready = max(self.now, self.link_free) + self.costs.service(size, source, destination)
        drained = ready + self.costs.drain_us
        flight = self.ledger.submit(keys, source, destination, self.now, ready, drained, targets)
        self.link_free = ready
        self.metrics.wire_bytes += size
        self.metrics.destination_bytes += size
        for key in targets:
            self.pending[key, destination] = (ready, flight)
            if speculative:
                self.prefetched[key] = ready
        if speculative:
            self.metrics.speculative_bytes += size
        self.schedule(ready, "ready", {"flight": flight})
        self.schedule(drained, "drain", {"flight": flight, "retire": retire_source})
        return ready

    def ensure(self, keys, *, speculative=False, compute_us=None):
        keys = self.closure(keys)
        if any(not self.ledger.valid(k) for k in keys):
            raise SafetyError("stale epoch requested")
        self.make_space("hbm", keys)
        finish, groups, missing = self.now, {}, []
        for key in keys:
            if self.ledger.available(key, "hbm"):
                if not speculative and self.prefetched.pop(key, None) is not None:
                    self.metrics.speculative_ready_bytes += self.ledger.nodes[key].size
                continue
            if (key, "hbm") in self.pending:
                finish = max(finish, self.pending[key, "hbm"][0])
            elif key in self.computing:
                finish = max(finish, self.computing[key])
            elif compute_us is not None:
                missing.append(key)
            else:
                source = next((t for t in ("dram", "ssd") if self.ledger.available(key, t)), None)
                if source is None:
                    if speculative:
                        continue
                    raise SafetyError("missing component without a recompute recipe")
                groups.setdefault(source, []).append(key)
        for source, group in groups.items():
            finish = max(finish, self.submit(group, source, "hbm", speculative))
        if compute_us is not None and missing:
            if len(self.ledger.replicas) + len(missing) > self.limits.owners:
                raise LimitError("compute destination owner budget")
            finish = max(finish, self.compute_free) + compute_us
            self.compute_free = finish
            for key in missing:
                self.ledger.allocate(key, "hbm", self.now, ready=False)
                self.ledger.replicas[key, "hbm"].pins += 1
                self.computing[key] = finish
            self.schedule(finish, "computed", {"keys": tuple(missing)})
        return finish

    def estimate(self, keys):
        groups, waiting = {}, 0
        for key in self.closure(keys):
            if not self.ledger.complete((key,)):
                return math.inf
            if (key, "hbm") in self.pending:
                waiting = max(waiting, self.pending[key, "hbm"][0] - self.now)
            elif not self.ledger.available(key, "hbm"):
                source = "dram" if self.ledger.available(key, "dram") else "ssd"
                groups[source] = groups.get(source, 0) + self.ledger.nodes[key].size
        duration = sum(self.costs.service(size, source, "hbm") for source, size in groups.items())
        return max(waiting, max(0, self.link_free - self.now) + duration if groups else 0)

    def publish(self, data):
        b, branch = data["boundary"], data["branch"]
        if len(self.boundaries) >= self.limits.nodes or (
            branch not in self.branches and len(self.branches) >= self.limits.branches
        ):
            raise LimitError("boundary/branch metadata budget")
        for node in data["nodes"]:
            self.ledger.define(node)
        self.closure(b.history or b.roots)
        if not b.roots or b.tokens < 1 or b.recompute_us < 1:
            raise SafetyError("invalid boundary")
        self.boundaries[b.key], self.branches[branch] = b, b
        if not data.get("cold", False):
            for key in self.closure(self.roots(b)):
                if (
                    not data.get("checkpoint_save", True)
                    and self.ledger.nodes[key].kind == "checkpoint"
                ):
                    continue
                if self.ledger.complete((key,)):
                    continue
                for tier in ("dram", "ssd"):
                    try:
                        self.make_space(tier, (key,))
                        self.ledger.allocate(key, tier, self.now)
                        break
                    except LimitError:
                        continue
        if self.policy.dag:
            self.collect()

    def collect(self):
        boundaries = list(self.branches.values()) + [b for _, b in self.tools.values()]
        reachable = set().union(*(set(self.closure(b.roots)) for b in boundaries))
        reachable |= self.protected()
        for (key, tier), r in list(self.ledger.replicas.items()):
            if key not in reachable and not r.pins:
                self.ledger.evict(key, tier)
                self.prefetched.pop(key, None)

    def fork(self, data):
        parent, child = data["parent"], data["child"]
        if child in self.branches or len(self.branches) >= self.limits.branches:
            raise LimitError("fork branch budget")
        b = self.branches[parent]
        self.metrics.forks += 1
        if self.policy.sharing:
            self.branches[child] = (
                b  # O(1) immutable root reference; no checkpoint compute implied.
            )
            return
        mapping = {}
        closure = set(self.closure(self.roots(b)))
        for key in tuple(self.ledger.nodes):
            if key not in closure:
                continue
            n = self.ledger.nodes[key]
            clone = f"{child}/{key}"
            mapping[key] = clone
            # Dependencies were defined first; preserve their graph on the copy.
            dependencies = tuple(mapping[d] for d in n.dependencies)
            self.ledger.define(
                replace(n, key=clone, dependencies=dependencies, logical_key=n.logical_key or key)
            )
        clone_b = replace(
            b,
            key=f"{child}/{b.key}",
            roots=tuple(mapping[k] for k in b.roots),
            history=tuple(mapping[k] for k in b.history if k in mapping),
            alternatives=(),
        )
        self.branches[child], self.fork_maps[child] = clone_b, mapping
        # The baseline pays a serialized byte copy through the same owner ledger.
        groups = {}
        for key in mapping:
            source = next(
                (t for t in ("dram", "ssd", "hbm") if self.ledger.available(key, t)), None
            )
            if source is None:
                continue  # A later demand must recompute this missing component.
            groups.setdefault(source, []).append(key)
        for source, group in groups.items():
            try:
                ready = self.submit(group, source, "ssd", targets=tuple(mapping[k] for k in group))
                self.fork_ready[child] = ready
            except LimitError:
                pass

    def request(self, branch, boundary, selection, arrived):
        self.metrics.requests += 1
        if len(self.requests) >= self.limits.branches:
            self.metrics.rejected += 1
            return
        b = boundary
        keys = self.closure(b.roots)
        if any(not self.ledger.valid(k) for k in keys):
            raise SafetyError("stale requested boundary")
        hit = self.ledger.complete(keys)
        self.metrics.boundary_hits += int(hit)
        self.metrics.hit_tokens += b.tokens if hit else 0
        self.metrics.prefill_tokens += b.tokens
        self.metrics.components += len(keys)
        self.metrics.component_hits += sum(self.ledger.complete((k,)) for k in keys)
        self.hits[b.key] = self.hits.get(b.key, 0) + 1
        request = Request(branch, b, selection, arrived)
        rid = self.serial + 1
        self.requests[rid] = request
        roots = self.roots(b)
        if self.policy.sparse:
            roots = self.mandatory(roots)
        request.needed = self.closure(roots)
        try:
            if self.policy.sparse and hit:
                request.source_lease = self.ledger.lease(b.roots, None, self.now)
            cost = self.estimate(roots)
            recipe = None
            if self.policy.recompute:
                for candidate in b.alternatives:
                    score = (
                        max(self.estimate(candidate.roots), self.compute_free - self.now)
                        + candidate.replay_us
                    )
                    if score < cost:
                        cost, recipe = score, candidate
            recompute = math.isinf(cost) or (
                self.policy.recompute
                and max(0, self.compute_free - self.now) + b.recompute_us < cost
            )
            if self.policy.sparse and not hit:
                # A resident indexer is not a complete recovery boundary. Missing
                # later-selected pages must pay explicit replay/recompute work.
                recompute = True
            if recompute:
                request.compute = True
                self.metrics.recomputes += 1
                when = self.ensure(roots, compute_us=b.recompute_us)
            elif recipe is not None:
                request.needed = self.closure(recipe.roots)
                when = self.ensure(recipe.roots)
                self.schedule(
                    when, "replay", {"rid": rid, "cost": recipe.replay_us, "roots": roots}
                )
                self.metrics.replays += 1
                request.replay = True
                return
            else:
                when = self.ensure(roots)
            self.schedule(when, "query", {"rid": rid})
        except LimitError:
            self.reject(rid)

    def reject(self, rid):
        request = self.requests.pop(rid)
        if request.lease is not None:
            self.ledger.release(request.lease)
        if request.source_lease is not None:
            self.ledger.release(request.source_lease)
        self.metrics.rejected += 1

    def query(self, rid):
        r = self.requests[rid]
        if self.policy.sparse:
            selected = r.selections[r.step] if r.selections else ()
            r.needed = self.closure(self.mandatory(r.boundary.roots))
            try:
                r.lease = self.ledger.lease(r.needed, "hbm", self.now)
            except LimitError:
                self.reject(rid)
                return
            # The current query/indexer precedes visibility of the selection.
            start = max(self.now, self.compute_free)
            self.compute_free = start + self.costs.indexer_us
            self.schedule(self.compute_free, "selection", {"rid": rid, "keys": selected})
        else:
            r.needed = self.closure(r.boundary.roots)
            try:
                when = self.ensure(
                    r.needed, compute_us=r.boundary.recompute_us if r.compute else None
                )
                self.schedule(when, "consume", {"rid": rid})
            except LimitError:
                self.reject(rid)

    def consume(self, rid):
        r = self.requests[rid]
        r.lease = self.ledger.lease(r.needed, "hbm", self.now)
        when = max(self.now, self.compute_free) + self.costs.compute_us
        self.compute_free = when
        self.schedule(when, "token", {"rid": rid})
        if self.policy.sparse and self.policy.prefetch:
            prediction = self.history.predict_selection(r.branch)
            if self.policy.oracle and r.step + 1 < len(r.selections):
                prediction = r.selections[r.step + 1]
            # Rejected admission retains any batches already submitted until drain.
            with contextlib.suppress(LimitError):
                self.ensure(prediction, speculative=True)

    def dispatch(self, kind, d):
        if kind == "publish":
            self.publish(d)
        elif kind == "fork":
            self.fork(d)
        elif kind == "epoch":
            self.ledger.retire_epoch(d["source"], d["epoch"])
        elif kind == "evict":
            self.ledger.evict(d["key"], d["tier"])
        elif kind == "lease":
            self.ledger.lease(d["roots"], d["tier"], self.now, d.get("generations"))
        elif kind == "release":
            self.ledger.release(d["lease"])
        elif kind == "tool_start":
            branch = d["branch"]
            if branch not in self.branches:
                return
            self.tools[branch] = (self.now, self.branches[branch])
            if len(self.tools) > self.limits.branches:
                raise LimitError("tool owner budget")
            # Save only HBM-only state. Existing backing copies need no extra write.
            for key in self.closure(self.branches[branch].roots):
                if self.ledger.available(key, "hbm") and not self.ledger.replicas[key, "hbm"].pins:
                    if any(self.ledger.available(key, t) for t in ("dram", "ssd")):
                        self.ledger.evict(key, "hbm")
                    else:
                        with contextlib.suppress(LimitError):
                            self.submit((key,), "hbm", "ssd", retire_source=True)
            if self.policy.jit:
                expected = self.history.expected_tool()
                if self.policy.oracle:
                    expected = d["oracle_return"] - self.now
                if expected is not None:
                    estimate = self.estimate(self.branches[branch].roots)
                    if math.isfinite(estimate):
                        at = self.now + max(0, round(expected * d["prediction_scale"] - estimate))
                        self.schedule(at, "prefetch", {"branch": branch, "started": self.now})
        elif kind == "prefetch":
            tool = self.tools.get(d["branch"])
            if tool and tool[0] == d["started"]:
                roots = tool[1].roots
                if self.policy.sparse:
                    roots = self.mandatory(roots)
                with contextlib.suppress(LimitError):
                    self.ensure(roots, speculative=True)
        elif kind == "tool_return":
            tool = self.tools.pop(d["branch"], None)
            if tool:
                self.history.observe_tool(self.now - tool[0])
                mapping = self.fork_maps.get(d["branch"], {})
                selection = tuple(
                    tuple(mapping.get(k, k) for k in step) for step in d["selections"]
                )
                when = max(self.now, self.fork_ready.get(d["branch"], 0))
                self.schedule(
                    when,
                    "request",
                    {
                        "branch": d["branch"],
                        "boundary": tool[1],
                        "selection": selection,
                        "arrived": self.now,
                    },
                )
        elif kind in {"cancel", "expire"}:
            branch = d["branch"]
            self.tools.pop(branch, None)
            self.branches.pop(branch, None)
            self.history.selections.pop(branch, None)
            if self.policy.dag:
                self.collect()
        elif kind == "request":
            self.request(**d)
        elif kind == "ready":
            self.ledger.ready(d["flight"], self.now)
        elif kind == "drain":
            f = self.ledger.flights[d["flight"]]
            self.ledger.drain(d["flight"], self.now)
            for key, target in zip(f.keys, f.targets, strict=True):
                self.pending.pop((target, f.destination))
                if d["retire"] and not self.ledger.replicas[key, f.source].pins:
                    self.ledger.evict(key, f.source)
        elif kind == "computed":
            for key in d["keys"]:
                r = self.ledger.replicas[key, "hbm"]
                r.ready, r.pins = self.ledger.valid(key), r.pins - 1
                self.computing.pop(key)
        elif kind == "replay":
            r = self.requests[d["rid"]]
            lease = self.ledger.lease(r.needed, "hbm", self.now)
            r.needed = self.closure(d["roots"])
            try:
                when = self.ensure(r.needed, compute_us=d["cost"])
                self.schedule(when, "release", {"lease": lease})
                self.schedule(when, "query", {"rid": d["rid"]})
            except LimitError:
                self.ledger.release(lease)
                self.reject(d["rid"])
        elif kind == "query":
            self.query(d["rid"])
        elif kind == "selection":
            r = self.requests[d["rid"]]
            self.ledger.release(r.lease)
            r.lease = None
            r.needed = self.closure(self.mandatory(r.boundary.roots) + tuple(d["keys"]))
            self.history.observe_selection(r.branch, d["keys"])
            try:
                when = self.ensure(
                    r.needed, compute_us=r.boundary.recompute_us if r.compute else None
                )
                self.metrics.stall_us += when - self.now
                self.schedule(when, "consume", {"rid": d["rid"]})
            except LimitError:
                self.reject(d["rid"])
        elif kind == "consume":
            self.consume(d["rid"])
        elif kind == "token":
            r = self.requests[d["rid"]]
            self.ledger.release(r.lease)
            r.lease = None
            if self.policy.sparse:
                r.needed = ()
            self.metrics.decode_tokens += 1
            if r.last_token is None:
                values = self.metrics.cold_ttft if r.compute else self.metrics.resume_ttft
                values.append(self.now - r.arrived)
            else:
                self.metrics.tpot.append(self.now - r.last_token)
            r.last_token = self.now
            r.step += 1
            if r.step < max(1, len(r.selections)):
                self.schedule(self.now, "query", {"rid": d["rid"]})
            else:
                self.metrics.completed += 1
                if r.source_lease is not None:
                    self.ledger.release(r.source_lease)
                self.requests.pop(d["rid"])
        else:
            raise SafetyError(f"unknown event {kind}")

    def run(self):
        while self.queue:
            if self.processed >= self.limits.processed:
                raise LimitError("processed event budget")
            at, _, kind, data = heapq.heappop(self.queue)
            keys = {k for k, _ in self.ledger.replicas}
            logical = {
                self.ledger.nodes[k].logical_key or k: self.ledger.nodes[k].size for k in keys
            }
            unique = sum(logical.values())
            replicated = sum(self.ledger.nodes[k].size for k, _ in self.ledger.replicas)
            hbm = self.ledger.used("hbm")
            dt = at - self.now
            m = self.metrics
            m.hbm_byte_us += hbm * dt
            m.unique_byte_us += unique * dt
            m.replicated_byte_us += replicated * dt
            m.peak_flights = max(m.peak_flights, len(self.ledger.flights))
            self.now = at
            self.dispatch(kind, data)
            self.processed += 1
            if self.sink:
                self.sink(
                    {
                        "at_us": at,
                        "event": kind,
                        "owners": len(self.ledger.replicas),
                        "flights": len(self.ledger.flights),
                        "hbm_bytes": self.ledger.used("hbm"),
                    }
                )
        if self.requests or self.ledger.flights or self.ledger.leases or self.computing:
            raise SafetyError("trace ended without required receipts")
        return self.result()

    def result(self):
        m = self.metrics
        for name in ("peak_unique_bytes", "peak_replicated_bytes", "peak_hbm_bytes", "peak_owners"):
            setattr(m, name, getattr(self.ledger, name))
        result = asdict(m)
        for name in ("cold_ttft", "resume_ttft", "tpot"):
            values = result.pop(name)
            result.update({f"{name}_p{p}_us": quantile(values, p / 100) for p in (50, 95, 99)})
        result["ttft_p95_us"] = quantile(m.cold_ttft + m.resume_ttft, 0.95)
        for name in (
            "wire_bytes",
            "destination_bytes",
            "peak_unique_bytes",
            "peak_replicated_bytes",
        ):
            for denominator in ("prefill_tokens", "decode_tokens"):
                result[f"{name}_per_{denominator}"] = (
                    getattr(m, name) / getattr(m, denominator) if getattr(m, denominator) else None
                )
        result.update(
            {
                "simulation_only": True,
                "policy": asdict(self.policy),
                "elapsed_us": self.now,
                "hbm_byte_seconds": m.hbm_byte_us / 10**6,
                "boundary_hit_rate": m.boundary_hits / m.requests if m.requests else 0,
                "component_hit_rate": m.component_hits / m.components if m.components else 0,
                "token_hit_rate": m.hit_tokens / m.prefill_tokens if m.prefill_tokens else 0,
                "completed_tokens_per_s": m.decode_tokens * 10**6 / self.now if self.now else 0,
                "overlap_byte_fraction": m.speculative_ready_bytes / m.speculative_bytes
                if m.speculative_bytes
                else 0,
                "processed_events": self.processed,
                "all_receipts_drained": True,
                "slo_pass": m.completed > 0
                and bool(m.tpot)
                and not m.rejected
                and (quantile(m.cold_ttft + m.resume_ttft, 0.95) or 0) <= self.costs.ttft_slo_us
                and (quantile(m.tpot, 0.95) or 0) <= self.costs.tpot_slo_us,
            }
        )
        return result
