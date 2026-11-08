"""Bounded planning over observations; no trace, event queue or future-time input."""

from dataclasses import dataclass

from .model import Ledger, LimitError


@dataclass(frozen=True)
class Policy:
    name: str
    minimal: bool = True
    dag: bool = False
    recompute: bool = False
    sharing: bool = False
    jit: bool = False
    sparse: bool = False
    prefetch: bool = False
    oracle: bool = False
    static: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for k, v in vars(self).items() if k != "name"):
            raise ValueError("policy switches must be booleans")


POLICIES = {
    p.name: p
    for p in (
        Policy("full_lru", minimal=False),
        Policy("static", minimal=False, static=True),
        Policy("minimal_lru"),
        Policy("shared_lru", sharing=True),
        Policy("p1", dag=True, recompute=True),
        Policy("p2", sharing=True, jit=True),
        Policy("p3", sparse=True, prefetch=True),
        Policy("all", dag=True, recompute=True, sharing=True, jit=True, sparse=True, prefetch=True),
        Policy(
            "oracle",
            dag=True,
            recompute=True,
            sharing=True,
            jit=True,
            sparse=True,
            prefetch=True,
            oracle=True,
        ),
    )
}


def pack(ledger: Ledger, candidates, budget: int, *, exhaustive=False):
    """Maximum observed benefit of complete closures, charging physical aliases once.

    Greedy marginal closure first. Exhaustive subset search is an independently
    enumerated upper bound for this *static* objective, not an online oracle.
    """
    if len(candidates) > ledger.limits.candidates or (exhaustive and len(candidates) > 12):
        raise LimitError("planner candidate budget")
    closures = [(frozenset(ledger.closure(roots)), value) for roots, value in candidates]
    if any(value < 0 for _, value in closures) or budget < 0:
        raise ValueError("negative planner value or capacity")
    work = 0

    def charge(keys):
        nonlocal work
        work += len(keys) + 1
        if work > ledger.limits.planner_work:
            raise LimitError("planner work budget")
        return sum(ledger.nodes[k].size for k in keys)

    def benefit(keys):
        return sum(v for c, v in closures if c <= keys)

    chosen = frozenset()
    if exhaustive:
        for mask in range(1 << len(closures)):
            keys = frozenset().union(*(c for i, (c, _) in enumerate(closures) if mask & (1 << i)))
            if charge(keys) <= budget and benefit(keys) > benefit(chosen):
                chosen = keys
    else:
        while True:
            options = []
            for closure, _ in closures:
                merged = chosen | closure
                extra = charge(closure - chosen)
                if extra and charge(merged) <= budget:
                    gain = benefit(merged) - benefit(chosen)
                    options.append((gain / extra, gain, tuple(sorted(merged))))
            if not options:
                break
            _, _, keys = max(options)
            chosen = frozenset(keys)
    return chosen, benefit(chosen), work


class History:
    """Fixed-size tool history and last observed selections, keyed by live branch."""

    def __init__(self, limit):
        self.limit = limit
        self.durations = []
        self.selections = {}

    def observe_tool(self, duration):
        self.durations.append(duration)
        del self.durations[:-32]

    def expected_tool(self):
        return sum(self.durations) // len(self.durations) if self.durations else None

    def observe_selection(self, branch, keys):
        if branch not in self.selections and len(self.selections) >= self.limit:
            raise LimitError("predictor branch budget")
        self.selections[branch] = tuple(keys)

    def predict_selection(self, branch):
        return self.selections.get(branch, ())
