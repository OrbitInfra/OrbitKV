"""Analytic fixtures, not downloaded model configs or recorded benchmark results."""

from benches.storage.state_cache.engine import Costs
from benches.storage.state_cache.model import Boundary, Node
from benches.storage.state_cache.schema import describe
from benches.storage.state_cache.trace import Input, Sizes, Trace, Workload

SCHEMA = describe({"num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]})
SIZES = Sizes(16, 4, 4096)
COSTS = Costs(10, 2, 5, 3, 1, 10000, 1000)
CAPACITIES = {"hbm": 1000, "dram": 1000, "ssd": 1000}


def hand_trace(nodes=None, roots=("a",), *, recompute_us=100, alternatives=()):
    nodes = (Node("a", 100),) if nodes is None else nodes
    b = Boundary("b", roots, 10, recompute_us, alternatives)
    return Trace(
        (
            Input(0, "publish", {"branch": "s", "boundary": b, "nodes": nodes}),
            Input(0, "request", {"branch": "s", "boundary": b, "selection": ((),), "arrived": 0}),
        ),
        Workload(sessions=1),
        SIZES,
        SCHEMA,
    )
