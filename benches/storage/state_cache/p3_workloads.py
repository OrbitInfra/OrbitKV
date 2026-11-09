"""Frozen synthetic sensitivity cases; no measured model statistics or service costs."""

from dataclasses import asdict

from .trace import Sizes, Workload, generate

MECHANISMS = ["paging", "reuse", "previous_step", "previous_layer", "paging_oracle"]
CONTROLS = ["full_lru", "static", "minimal_lru", "shared_lru", "p3", *MECHANISMS]
GRID = {
    "base": {},
    "zero": {"overlap": 0, "layer_overlap": 0},
    "past": {"overlap": 1, "layer_overlap": 0},
    "adjacent": {"overlap": 0, "layer_overlap": 1},
    "both": {"overlap": 1, "layer_overlap": 1},
    "slow": {"bytes_per_us": 128, "layer_compute_us": 25},
    "fast": {"bytes_per_us": 8192, "layer_compute_us": 100},
    "depth2": {"lookahead": 2},
    "serial": {"lookahead": 0},
    "local": {"new_page_locality": 1},
    "uniform": {"new_page_locality": 0},
    "indexer": {"layer_indexer_us": 25},
}


def parameters(schema, mode):
    """Derive byte budgets from static owners, never from selections or wins.

    HBM holds the union of two sessions' mandatory state plus 8 or 10 maximum
    sparse pages. This deliberately pressures previous-step pages. Every policy
    receives the same physical aliases, budgets, indexer and compute work.
    """
    if mode not in {"grid", "holdout", "probe"}:
        raise ValueError("unknown P3 campaign mode")
    sizes = Sizes(64, 1, 1024)
    for family in ("tools",) if mode == "grid" else ("tools", "document"):
        w = Workload(
            seed=19,
            family=family,
            sessions=2,
            turns=1,
            forks=0,
            pages=8,
            steps=4,
            top_k=2,
            overlap=0.5,
            layer_overlap=0.5,
            layered=True,
            adversarial=False,
            tool_mean_us=1000,
            tool_sigma=0,
        )
        trace = generate(schema, sizes, w)
        nodes = {n.key: n for e in trace.events if e.kind == "publish" for n in e.data["nodes"]}
        roots = {k for e in trace.events if e.kind == "publish" for k in e.data["boundary"].roots}
        mandatory = sum(
            nodes[k].size for k in roots if nodes[k].kind not in {"kv", "latent", "compressed"}
        )
        page = max(n.size for n in nodes.values() if n.kind in {"kv", "latent", "compressed"})
        base = {
            "sizes": asdict(sizes),
            "workload": asdict(w),
            "capacities": {"hbm": mandatory + 8 * page, "dram": 64 * 2**20, "ssd": 64 * 2**20},
            "costs": {
                "bytes_per_us": 1024,
                "latency_us": 10,
                "drain_us": 1,
                "compute_us": 50,
                "indexer_us": 5,
                "layer_compute_us": 50,
                "layer_indexer_us": 5,
                "ttft_slo_us": 70000,
                "tpot_slo_us": 20000,
            },
        }
        if mode == "grid":
            for name, axes in GRID.items():
                # Depth zero is a null control, not an enabled predictor case.
                policies = (
                    ["paging", "p3", "reuse", "paging_oracle"] if name == "serial" else MECHANISMS
                )
                yield (
                    name,
                    {**base, "policies": policies, "sweep": {k: [v] for k, v in axes.items()}},
                )
        else:
            sweep = {"seed": [41, 73]}
            if mode == "holdout":
                sweep.update(hbm_bytes=[mandatory + n * page for n in (8, 10)], sessions=[2, 3])
            yield (
                family,
                {**base, "policies": CONTROLS if mode == "holdout" else MECHANISMS, "sweep": sweep},
            )
