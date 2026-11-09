"""Analytic pipeline, causal prediction and physical-owner controls."""

import heapq
import random
from dataclasses import replace

import pytest

from benches.storage.state_cache.engine import Simulation
from benches.storage.state_cache.model import (
    Boundary,
    Limits,
    Node,
    PageLayer,
    SafetyError,
)
from benches.storage.state_cache.policy import POLICIES
from benches.storage.state_cache.schema import describe
from benches.storage.state_cache.trace import (
    Input,
    Trace,
    Workload,
    generate,
    layer_selections,
)
from benches.tests.storage.state_cache.fixtures import (
    CAPACITIES,
    COSTS,
    SCHEMA,
    SIZES,
    hand_trace,
)


def pipeline(*, selections=(((0,), (0,)),), sources=(0, 0), pages=1, depth=1):
    layers = tuple(
        PageLayer(i, source, tuple(f"l{i}p{p}" for p in range(pages)))
        for i, source in enumerate(sources)
    )
    nodes = tuple(Node(key, 100) for layer in layers for key in layer.pages)
    boundary = Boundary("b", tuple(n.key for n in nodes), 10, 100, layers=layers)
    return Trace(
        (
            Input(0, "publish", {"branch": "s", "boundary": boundary, "nodes": nodes}),
            Input(
                0,
                "request",
                {"branch": "s", "boundary": boundary, "selection": selections, "arrived": 0},
            ),
        ),
        Workload(sessions=1, layered=True, lookahead=depth),
        SIZES,
        SCHEMA,
    )


def run(trace, policy, *, compute=20, hbm=1000, limits=None):
    return Simulation(
        trace,
        POLICIES[policy],
        {"hbm": hbm, "dram": 1 << 20, "ssd": 1 << 20},
        replace(COSTS, layer_compute_us=compute, layer_indexer_us=1),
        limits,
    ).run()


@pytest.mark.parametrize("compute", [7, 12, 20])
def test_exact_reuse_hides_only_real_transfer_time(compute):
    trace = pipeline()
    m = run(trace, "reuse", compute=compute)
    # Source indexer = 1, each gather = 2 + 100/10 = 12. One link
    # serializes the gathers: readiness at 13 and 25, never both at 13.
    assert m["pipeline_stall_us"] == max(0, 12 - compute)
    assert m["ttft_p95_us"] == 1 + 12 + 2 * compute + max(0, 12 - compute)
    assert m["wire_bytes"] == 200 and m["declared_reuse_bytes"] == 100
    assert m["speculative_bytes"] == 0 and m["page_maps"] == 2
    serial = run(
        replace(trace, workload=replace(trace.workload, lookahead=0)), "reuse", compute=compute
    )
    demand = run(trace, "paging", compute=compute)
    assert serial["ttft_p95_us"] == demand["ttft_p95_us"] == 1 + 24 + 2 * compute
    assert serial["pipeline_stall_us"] == 12 and serial["declared_reuse_bytes"] == 0


def test_old_resident_last_set_null_control_is_unchanged():
    trace = replace(
        hand_trace(),
        events=(
            hand_trace().events[0],
            replace(
                hand_trace().events[1],
                data={**hand_trace().events[1].data, "selection": (("a",), ("a",))},
            ),
        ),
    )
    null = Simulation(trace, POLICIES["p3"], CAPACITIES, COSTS).run()
    demand = Simulation(trace, POLICIES["paging"], CAPACITIES, COSTS).run()
    assert null["speculative_bytes"] == 0
    assert (null["wire_bytes"], null["ttft_p95_us"], null["tpot_p95_us"]) == (100, 16, 4)
    assert {k: v for k, v in null.items() if k != "policy"} == {
        k: v for k, v in demand.items() if k != "policy"
    }
    layered = run(pipeline(), "p3")
    assert layered["speculative_bytes"] == 0


@pytest.mark.parametrize("policy", ["previous_step", "previous_layer"])
def test_predictors_reach_nonresident_pages_and_zero_overlap_wastes_traffic(policy):
    # Every demand is fresh, disjoint from both possible predictors. Only three
    # pages fit, so previous-step pages have been evicted before prediction.
    sets = tuple(tuple((step * 4 + layer,) for layer in range(4)) for step in range(3))
    trace = pipeline(selections=sets, sources=(0, 1, 2, 3), pages=16)
    m = run(trace, policy, hbm=300)
    demand = run(trace, "paging", hbm=300)
    assert m["completed"] == demand["completed"] == 1
    assert m["speculative_bytes"] > 0
    assert m["wasted_speculative_bytes"] == m["speculative_bytes"]
    assert m["wire_bytes"] > demand["wire_bytes"]
    assert m["stall_us"] >= demand["stall_us"]


@pytest.mark.parametrize("policy", ["previous_step", "previous_layer"])
def test_prefetch_ablation_disables_new_predictors(policy):
    sets = tuple(tuple((step * 4 + layer,) for layer in range(4)) for step in range(3))
    trace = pipeline(selections=sets, sources=(0, 1, 2, 3), pages=16)
    caps = {"hbm": 300, "dram": 1 << 20, "ssd": 1 << 20}
    costs = replace(COSTS, layer_compute_us=20, layer_indexer_us=1)
    disabled = Simulation(trace, replace(POLICIES[policy], prefetch=False), caps, costs).run()
    demand = Simulation(trace, POLICIES["paging"], caps, costs).run()
    assert disabled["speculative_bytes"] == 0
    assert {k: v for k, v in disabled.items() if k != "policy"} == {
        k: v for k, v in demand.items() if k != "policy"
    }


@pytest.mark.parametrize("policy", ["paging", "reuse", "previous_step", "previous_layer"])
def test_online_prefix_cannot_see_later_layer_selection(policy):
    snapshots = []
    for next_selection in (0, 1):
        trace = pipeline(selections=(((0,), (next_selection,)),), sources=(0, 1), pages=2)
        sim = Simulation(
            trace,
            POLICIES[policy],
            CAPACITIES,
            replace(COSTS, layer_compute_us=20, layer_indexer_us=10),
        )
        log = []
        while sim.queue:
            at, _, kind, data = heapq.heappop(sim.queue)
            if kind == "selection" and sim.requests[data["rid"]].layer == 1:
                break  # Second layer's own indexer has not revealed its set.
            sim.now = at
            sim.dispatch(kind, data)
            log.append(
                (
                    at,
                    kind,
                    sorted(sim.pending),
                    sorted(sim.ledger.replicas),
                    sim.metrics.speculative_bytes,
                )
            )
        snapshots.append(log)
    assert snapshots[0] == snapshots[1]


def test_prefetch_caps_reject_whole_attempts_and_keep_undrained_owners():
    trace = pipeline(selections=(((0,), (0,), (0,)),), sources=(0, 0, 0), depth=2)
    m = run(trace, "reuse", limits=replace(Limits(), prefetches=1))
    assert m["peak_prefetches"] == 1 and m["prefetch_rejected"] > 0
    assert m["completed"] == 1 and m["all_receipts_drained"]
    sim = Simulation(pipeline(), POLICIES["reuse"], CAPACITIES, replace(COSTS, drain_us=100))
    while sim.queue and sim.queue[0][0] < 25:
        at, _, kind, data = heapq.heappop(sim.queue)
        sim.now = at
        sim.dispatch(kind, data)
    assert sim.ahead and sim.metrics.page_maps == 1
    with pytest.raises(SafetyError, match="release before drain"):
        sim.ledger.evict("l1p0", "hbm")
    assert sim.run()["all_receipts_drained"]


def test_field_driven_sources_alias_bytes_and_reject_unsupported_patterns():
    schema = describe(
        {
            "num_hidden_layers": 4,
            "index_topk": 2,
            "index_source_layer_ids": [0, 2],
            "kv_source_layer_ids": [0],
        }
    )
    assert schema.paging_layers((1,)) == ((0, 0, 0), (1, 0, 0), (2, 2, 0), (3, 2, 0))
    w = Workload(sessions=1, turns=1, forks=0, pages=8, steps=3, layered=True, adversarial=False)
    trace = generate(schema, SIZES, w)
    b = trace.events[0].data["boundary"]
    assert all(layer.pages == b.layers[0].pages for layer in b.layers)
    assert len(b.roots) == 16  # Eight main pages + eight indexer pages, once per owner.
    route = describe(
        {
            "num_hidden_layers": 4,
            "index_topk": 2,
            "indexer_types": ["full", "shared", "full", "shared"],
        }
    )
    assert route.paging_layers((1,)) == ((0, 0, 0), (1, 0, 1), (2, 2, 2), (3, 2, 3))
    assert describe({"num_hidden_layers": 4}).paging_layers((1,)) == ()
    with pytest.raises(ValueError, match="config-declared"):
        generate(SCHEMA, SIZES, w)
    bad = replace(schema, facts={"index_source_layer_ids": [1]})
    with pytest.raises(ValueError, match="preceding"):
        bad.paging_layers((1,))


def test_layer_generator_is_seeded_and_declared_reuse_is_exact():
    layers = tuple(PageLayer(i, i if i != 2 else 1, tuple(map(str, range(16)))) for i in range(4))
    w = Workload(layered=True, steps=4, top_k=2, overlap=0, layer_overlap=0, adversarial=False)
    a = layer_selections(random.Random(41), layers, w)
    assert a == layer_selections(random.Random(41), layers, w)
    for step, sets in enumerate(a):
        assert sets[1] == sets[2]
        for i in (0, 1, 3):
            if step:
                assert not set(sets[i]) & set(a[step - 1][i])
            if i:
                assert not set(sets[i]) & set(sets[i - 1])
    for locality in (0, 1):
        full = layer_selections(
            random.Random(73),
            layers,
            replace(w, overlap=1, layer_overlap=1, new_page_locality=locality),
        )
        assert all(s == full[0] for s in full)


def test_incorrect_declared_demand_never_publishes_consumer_page_map():
    trace = pipeline(selections=(((0,), (1,)),), pages=2)
    sim = Simulation(trace, POLICIES["reuse"], CAPACITIES, COSTS)
    with pytest.raises(SafetyError, match="differs"):
        sim.run()
    assert sim.metrics.page_maps == 1


@pytest.mark.parametrize("policy", ["previous_step", "previous_layer"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("overlap", [0, 0.5, 1])
@pytest.mark.parametrize("reuse", [False, True])
def test_campaign_predictor_modes_are_non_null_under_pressure(policy, depth, overlap, reuse):
    config = {"num_hidden_layers": 8, "index_topk": 2}
    if reuse:
        config["indexer_types"] = ["full", "shared", "shared", "shared"] * 2
    schema = describe(config)
    w = Workload(
        seed=41,
        sessions=2,
        turns=1,
        forks=0,
        pages=8,
        steps=4,
        layered=True,
        adversarial=False,
        overlap=overlap,
        layer_overlap=overlap,
        lookahead=depth,
        tool_sigma=0,
    )
    trace = generate(schema, SIZES, w)
    nodes = {n.key: n for e in trace.events if e.kind == "publish" for n in e.data["nodes"]}
    mandatory = sum(n.size for n in nodes.values() if n.kind == "indexer")
    page = max(n.size for n in nodes.values() if n.kind == "kv")
    m = run(trace, policy, hbm=mandatory + 8 * page)
    assert m["completed"] == 2 and not m["rejected"]
    assert m["speculative_bytes"] > 0
    assert m["speculative_bytes"] == m["speculative_consumed_bytes"] + m["wasted_speculative_bytes"]
    assert m["peak_prefetches"] <= depth * 2
