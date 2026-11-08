"""Acceptance: conservation, negative controls, exact traces, bounded search and causality."""

import hashlib
import heapq
import json
import random
from dataclasses import replace

import pytest

from benches.storage.state_cache.engine import Simulation
from benches.storage.state_cache.model import (
    Boundary,
    Ledger,
    LimitError,
    Limits,
    Node,
    Recipe,
    SafetyError,
)
from benches.storage.state_cache.policy import POLICIES, History, pack
from benches.storage.state_cache.schema import describe, load_config
from benches.storage.state_cache.sources import SOURCES
from benches.storage.state_cache.trace import Input, Workload, generate, selections
from benches.tests.storage.state_cache.fixtures import (
    CAPACITIES,
    COSTS,
    SCHEMA,
    SIZES,
    hand_trace,
)


def ledger(limits=None):
    limits = limits or Limits()
    state = Ledger(dict(CAPACITIES), limits)
    state.define(Node("checkpoint", 40))
    state.define(Node("a", 60, dependencies=("checkpoint",)))
    for key in state.nodes:
        state.allocate(key, "dram", 0)
    return state


@pytest.mark.parametrize("interruption", ["cancel", "epoch", "none"])
def test_ready_timeout_cancel_and_epoch_never_release_owners(interruption):
    state = ledger()
    batch = state.submit(("a", "checkpoint"), "dram", "hbm", 0, 10, 20)
    if interruption == "cancel":
        state.cancel(batch)
    if interruption == "epoch":
        state.retire_epoch("local", 2)
    state.ready(batch, 10)
    assert state.complete(("a",), "hbm") == (interruption == "none")
    for tier in ("dram", "hbm"):
        with pytest.raises(SafetyError, match="release before drain"):
            state.evict("a", tier)
    with pytest.raises(SafetyError, match="premature drain"):
        state.drain(batch, 19)
    assert len(state.replicas) == 4
    state.drain(batch, 20)
    for key, tier in tuple(state.replicas):
        state.evict(key, tier)
    assert not state.replicas


def test_alias_dependency_and_generation_controls():
    state = ledger()
    assert state.closure(("a", "a", "checkpoint")) == ("a", "checkpoint")
    assert state.used("dram") == 100
    lease = state.lease(("a", "a"), "dram", 0, {"a": 1})
    with pytest.raises(SafetyError, match="release before drain"):
        state.evict("checkpoint", "dram")
    with pytest.raises(SafetyError, match="stale allocation"):
        state.lease(("a",), "dram", 0, {"a": 2})
    state.release(lease)
    state.evict("checkpoint", "dram")
    assert state.available("a", "dram") and not state.complete(("a",), "dram")
    with pytest.raises(SafetyError, match="incomplete"):
        state.lease(("a",), "dram", 0)
    state.retire_epoch("local", 2)
    with pytest.raises(SafetyError, match="stale"):
        state.submit(("a",), "dram", "hbm", 0, 10, 20)
    state.define(Node("new", 50, epoch=2, generation=2))
    state.allocate("new", "dram", 0)
    assert state.complete(("new",))


@pytest.mark.parametrize("policy", ["full_lru", "minimal_lru", "static", "p1"])
def test_hand_computed_byte_time_and_alias_trace(policy):
    result = Simulation(hand_trace(roots=("a", "a")), POLICIES[policy], CAPACITIES, COSTS).run()
    assert result["resume_ttft_p95_us"] == 15  # 2 latency + 100/10 transfer + 3 compute.
    assert result["elapsed_us"] == 17  # Reader done at 15; physical copy drains at 17.
    assert result["hbm_byte_us"] == 1700  # Destination is charged even before ready.
    assert result["wire_bytes"] == result["destination_bytes"] == 100
    assert result["peak_unique_bytes"] == 100
    assert result["peak_replicated_bytes"] == 200
    assert result["boundary_hit_rate"] == 1 and result["completed"] == 1
    assert result["wire_bytes_per_prefill_tokens"] == 10


def test_large_byte_extents_keep_integer_service_accounting():
    size = (1 << 54) + 1
    costs = replace(COSTS, bytes_per_us=1)
    assert costs.service(size, "dram", "hbm") == size + costs.latency_us


def test_peak_includes_final_materialization_and_alias_mismatch_fails():
    trace = hand_trace()
    sim = Simulation(replace(trace, events=(trace.events[0],)), POLICIES["p1"], CAPACITIES, COSTS)
    assert sim.run()["peak_unique_bytes"] == 100
    with pytest.raises(SafetyError, match="logical identity"):
        sim.ledger.define(Node("wrong-copy", 99, logical_key="a"))


def test_recompute_ablation_chooses_cheaper_compute_without_transfer():
    trace = hand_trace(recompute_us=5)
    planner = Simulation(trace, POLICIES["p1"], CAPACITIES, COSTS).run()
    control = Simulation(trace, replace(POLICIES["p1"], recompute=False), CAPACITIES, COSTS).run()
    assert (planner["cold_ttft_p95_us"], planner["wire_bytes"], planner["recomputes"]) == (8, 0, 1)
    assert control["resume_ttft_p95_us"] == 15


def test_replay_recipe_charges_checkpoint_load_and_replay():
    trace = hand_trace((Node("a", 100), Node("cp", 10)), alternatives=(Recipe(("cp",), 2),))
    cp = Input(
        0,
        "publish",
        {"branch": "checkpoint", "boundary": Boundary("c", ("cp",), 1, 1), "nodes": ()},
    )
    trace = replace(trace, events=(trace.events[0], cp, trace.events[1]))
    result = Simulation(trace, POLICIES["p1"], CAPACITIES, COSTS).run()
    assert result["replays"] == 1
    assert result["wire_bytes"] == 10
    assert result["resume_ttft_p95_us"] == 8  # 2+1 load, 2 replay, 3 token.


def test_exact_packing_bounds_greedy_for_shared_recovery_dags():
    rng = random.Random(82)
    for _ in range(30):
        state = Ledger({"hbm": 1000})
        for i in range(7):
            state.define(
                Node(str(i), rng.randint(1, 20), dependencies=(str(i - 1),) if i and i % 2 else ())
            )
        candidates = [((str(i),), rng.randint(1, 15)) for i in range(7)]
        for budget in (15, 35, 70):
            greedy, gain, _ = pack(state, candidates, budget)
            exact, optimum, _ = pack(state, candidates, budget, exhaustive=True)
            assert gain <= optimum
            assert sum(state.nodes[k].size for k in greedy) <= budget
            assert sum(state.nodes[k].size for k in exact) <= budget
    with pytest.raises(LimitError, match="candidate"):
        pack(state, candidates * 2, 100, exhaustive=True)
    state.limits = replace(state.limits, planner_work=2)
    with pytest.raises(LimitError, match="budget"):
        pack(state, candidates, 100)


@pytest.mark.parametrize(
    "field,value", [("nodes", 1), ("events", 1), ("processed", 1), ("branches", 1)]
)
def test_caps_reject_instead_of_truncating(field, value):
    limits = replace(Limits(), **{field: value})
    trace = generate(SCHEMA, SIZES, Workload(sessions=2, turns=1, pages=2))
    with pytest.raises(LimitError):
        Simulation(
            trace,
            POLICIES["minimal_lru"],
            {"hbm": 1 << 20, "dram": 1 << 20, "ssd": 1 << 20},
            COSTS,
            limits,
        ).run()


def test_owner_exhaustion_rejects_demand_without_overcommitting():
    sim = Simulation(
        hand_trace(), POLICIES["minimal_lru"], CAPACITIES, COSTS, replace(Limits(), owners=1)
    )
    result = sim.run()
    assert result["rejected"] == 1 and result["completed"] == 0
    assert len(sim.ledger.replicas) == 1


def test_lease_and_flight_budgets_are_atomic():
    state = ledger(replace(Limits(), leases=1, flights=1))
    state.lease(("a",), "dram", 0)
    with pytest.raises(LimitError, match="lease"):
        state.lease(("a",), "dram", 0)
    state.submit(("a",), "dram", "hbm", 0, 1, 2)
    before = dict(state.replicas)
    with pytest.raises(LimitError, match="batch"):
        state.submit(("checkpoint",), "dram", "hbm", 0, 1, 2)
    assert state.replicas == before


def test_schema_field_lowering_and_physical_unknowns():
    schema = describe(
        {
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": [
                    "sliding_attention",
                    "linear_attention",
                    "full_attention",
                    "full_attention",
                ],
                "sliding_window": 1024,
                "kv_lora_rank": 512,
                "index_topk": 2048,
                "kv_source_layer_ids": [2],
            }
        }
    )
    assert [(c.kind, c.layers) for c in schema.components] == [
        ("window", (0,)),
        ("checkpoint", (1,)),
        ("latent", (2, 3)),
        ("indexer", (2, 3)),
    ]
    assert schema.facts["kv_source_layer_ids"] == [2]
    assert any("alias" in text for text in schema.unresolved)
    compressed = describe(
        {"num_hidden_layers": 3, "compress_ratios": [0, 4, 128, 4], "sliding_window": 128}
    )
    assert compressed.layers == 3  # MTP config entry is not a backbone layer.
    assert {c.ratio for c in compressed.components if c.kind == "compressed"} == {4, 128}
    recurrent = describe(
        {
            "num_hidden_layers": 4,
            "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]},
        }
    )
    assert recurrent.components[0].layers == (0, 1, 2)
    with pytest.raises(ValueError, match="layer pattern"):
        describe({"num_hidden_layers": 1, "layer_types": ["unknown"]})


def test_config_hash_is_required_not_just_revision_label(tmp_path, monkeypatch):
    raw = json.dumps({"num_hidden_layers": 2}).encode()
    path = tmp_path / "config.json"
    path.write_bytes(raw)
    monkeypatch.setitem(SOURCES, "fixture", ("1" * 40, hashlib.sha256(raw).hexdigest()))
    schema = load_config("fixture", path)
    assert schema.revision == "1" * 40 and schema.sha256 == hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash"):
        load_config("fixture", path)


@pytest.mark.parametrize("policy", POLICIES)
def test_repeated_seed_reproduces_with_all_receipts_drained(policy):
    w = Workload(seed=34, sessions=2, turns=2, pages=4, steps=3)
    trace = generate(SCHEMA, SIZES, w)
    caps = {"hbm": 1 << 20, "dram": 1 << 20, "ssd": 1 << 22}
    a = Simulation(trace, POLICIES[policy], caps, COSTS).run()
    b = Simulation(generate(SCHEMA, SIZES, w), POLICIES[policy], caps, COSTS).run()
    assert a == b and a["all_receipts_drained"] and a["completed"] > 0


def test_online_prefix_is_independent_of_future_returns():
    trace = generate(SCHEMA, SIZES, Workload(sessions=1, turns=2, forks=0, adversarial=False))
    first_return = next(e.at for e in trace.events if e.kind == "tool_return")
    modified = replace(
        trace,
        events=tuple(
            replace(e, data={**e.data, "oracle_return": e.data["oracle_return"] + 10**6})
            if e.kind == "tool_start"
            else replace(e, at=e.at + 10**6)
            if e.kind == "tool_return"
            else e
            for e in trace.events
        ),
    )
    logs = []
    for t in (trace, modified):
        log = []
        sim = Simulation(
            t,
            POLICIES["all"],
            {"hbm": 1 << 22, "dram": 1 << 22, "ssd": 1 << 22},
            COSTS,
            sink=log.append,
        )
        # Stop at the common observed prefix; no tool-return data reached History.
        while sim.queue and sim.queue[0][0] < first_return:
            import heapq

            at, _, kind, data = heapq.heappop(sim.queue)
            sim.now = at
            sim.dispatch(kind, data)
            log.append((at, kind, len(sim.ledger.replicas)))
        logs.append(log)
    assert logs[0] == logs[1]
    history = History(2)
    assert history.expected_tool() is None
    for duration in range(100):
        history.observe_tool(duration)
    assert len(history.durations) == 32 and history.expected_tool() == 83


def test_sparse_overlap_and_adversarial_disjoint_generator():
    sequence = selections(random.Random(3), list("abcdefgh"), 5, 2, 1, True)
    assert sequence[0] == sequence[1] == sequence[2]
    assert not set(sequence[2]) & set(sequence[3])
    assert selections(random.Random(3), list("abcdefgh"), 5, 2, 1, True) == sequence


def test_epoch_event_fences_pending_restore_without_a_success_receipt():
    trace = hand_trace()
    trace = replace(
        trace, events=trace.events + (Input(5, "epoch", {"source": "local", "epoch": 2}),)
    )
    sim = Simulation(trace, POLICIES["minimal_lru"], CAPACITIES, COSTS)
    with pytest.raises(SafetyError, match="stale"):
        sim.run()
    assert sim.ledger.flights and len(sim.ledger.replicas) == 2
    assert sim.metrics.completed == 0


def test_sparse_source_lease_survives_backing_pressure_between_query_and_fetch():
    trace = hand_trace((Node("a", 50), Node("cp", 10, kind="checkpoint")), ("a", "cp"))
    trace = replace(
        trace,
        events=(
            trace.events[0],
            replace(trace.events[1], data={**trace.events[1].data, "selection": (("a",), ("a",))}),
            Input(
                5,
                "publish",
                {
                    "branch": "other",
                    "boundary": Boundary("other", ("b",), 1, 10),
                    "nodes": (Node("b", 60),),
                },
            ),
        ),
    )
    result = Simulation(trace, POLICIES["p3"], {"hbm": 60, "dram": 60, "ssd": 0}, COSTS).run()
    assert result["completed"] == 1 and result["decode_tokens"] == 2
    assert result["all_receipts_drained"]


def test_fork_copy_is_charged_but_shared_root_is_constant_metadata():
    trace = hand_trace()
    trace = replace(
        trace, events=(trace.events[0], Input(0, "fork", {"parent": "s", "child": "fork"}))
    )
    copied = Simulation(trace, POLICIES["minimal_lru"], CAPACITIES, COSTS)
    shared = Simulation(trace, POLICIES["shared_lru"], CAPACITIES, COSTS)
    copy_result, shared_result = copied.run(), shared.run()
    assert copy_result["wire_bytes"] == 100 and shared_result["wire_bytes"] == 0
    assert copy_result["peak_unique_bytes"] == shared_result["peak_unique_bytes"] == 100
    assert copy_result["peak_replicated_bytes"] == 200
    assert shared.branches["s"] is shared.branches["fork"]
    assert len(copied.ledger.nodes) == 2 and len(shared.ledger.nodes) == 1


def test_sparse_missing_backing_requires_recompute_not_a_false_complete_hit():
    trace = hand_trace((Node("a", 50), Node("cp", 10, kind="checkpoint")), ("a", "cp"))
    request = replace(trace.events[1], data={**trace.events[1].data, "selection": (("a",),)})
    trace = replace(
        trace, events=(trace.events[0], Input(0, "evict", {"key": "a", "tier": "dram"}), request)
    )
    result = Simulation(trace, POLICIES["p3"], CAPACITIES, COSTS).run()
    assert result["boundary_hits"] == 0 and result["recomputes"] == 1
    assert result["completed"] == 1 and result["cold_ttft_p95_us"] >= 100


def test_jit_uses_completed_tool_history_and_pays_until_drain():
    trace = hand_trace()
    events = list(trace.events)
    for start, end in ((20, 70), (100, 150)):
        events.extend(
            (
                Input(
                    start,
                    "tool_start",
                    {"branch": "s", "oracle_return": end, "prediction_scale": 1},
                ),
                Input(end, "tool_return", {"branch": "s", "selections": ((),)}),
            )
        )
    trace = replace(trace, events=tuple(events))
    sim = Simulation(trace, POLICIES["p2"], CAPACITIES, COSTS)
    result = sim.run()
    assert sim.metrics.resume_ttft == [15, 15, 3]
    assert result["wire_bytes"] == 300 and result["speculative_bytes"] == 100
    assert result["speculative_ready_bytes"] == 100 and result["elapsed_us"] == 155


def test_indexer_result_cannot_affect_eviction_before_it_is_ready():
    trace = hand_trace(
        (Node("a", 50), Node("b", 50), Node("cp", 10, kind="checkpoint")), ("a", "b", "cp")
    )
    snapshots = []
    for selected in ("a", "b"):
        request = replace(
            trace.events[1], data={**trace.events[1].data, "selection": ((selected,),)}
        )
        sim = Simulation(
            replace(trace, events=(trace.events[0], request)),
            POLICIES["p3"],
            CAPACITIES,
            replace(COSTS, indexer_us=10),
        )
        while sim.queue and sim.queue[0][0] < 13:  # cp ready at 3; indexer completes at 13.
            at, _, kind, data = heapq.heappop(sim.queue)
            sim.now = at
            sim.dispatch(kind, data)
        snapshots.append((sim.protected(), set(sim.pending)))
        with pytest.raises(SafetyError, match="release before drain"):
            sim.ledger.evict("cp", "hbm")
    assert snapshots[0] == snapshots[1] == ({"cp"}, set())


def test_document_questions_share_prefixes_but_never_reuse_question_state():
    trace = generate(
        SCHEMA,
        SIZES,
        Workload(
            family="document",
            sessions=1,
            turns=2,
            forks=0,
            pages=4,
            append_pages=1,
            adversarial=False,
        ),
    )
    boundaries = [e.data["boundary"] for e in trace.events if e.kind == "publish"]
    a, b = boundaries
    assert a.tokens == b.tokens == 5 * SIZES.page_tokens
    assert set(a.roots) & set(b.roots)
    assert len(set(a.roots) - set(b.roots)) == 2  # Fresh question KV and recurrent checkpoint.
    assert not b.alternatives  # Question replacement is not prefix extension.
