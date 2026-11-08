# State-cache simulator

**Simulation only, not hardware evidence.** This implements the hardware-free
experiment surface in [design v2 §8](../../../docs/design-v2.md#8-experiments-and-delivery-gates).
It imports neither an inference engine nor OrbiTE. No production policy, adapter,
model support or qualification changes. The three OrbiTE API gaps stay upstream.

## Inputs and reproduction

Python 3.10+ and the standard library suffice; pytest runs the controls. Plotting
is optional and uses Matplotlib. Run from the repository root:

```bash
uv run --isolated --no-project --with pytest \
  python -m pytest benches/tests/storage/state_cache

# Choose a new directory outside every source checkout.
export SIM_ROOT=/var/tmp/orbitkv-state-sim
mkdir -p "$SIM_ROOT"
python - <<'PY'
import json, os, urllib.request
from pathlib import Path
from benches.storage.state_cache.sources import SOURCES
root = Path(os.environ["SIM_ROOT"])
model = "Qwen/Qwen3.8-27B"
revision, digest = SOURCES[model]
urllib.request.urlretrieve(
    f"https://huggingface.co/{model}/resolve/{revision}/config.json",
    root / "config.json",
)
# These are sensitivity inputs, NOT measured hardware or native cache sizes.
parameters = {
    "sizes": {"entry_bytes": 256, "index_entry_bytes": 32,
              "checkpoint_bytes": 131072, "native_format": "assumed-packed-bytes"},
    "costs": {"bytes_per_us": 4096, "latency_us": 10, "drain_us": 5,
              "compute_us": 500, "indexer_us": 20, "ttft_slo_us": 20000,
              "tpot_slo_us": 4000, "ssd_bytes_per_us": 1024, "ssd_latency_us": 100},
    "capacities": {"hbm": 32 * 2**20, "dram": 64 * 2**20, "ssd": 256 * 2**20},
    "workload": {"turns": 3, "pages": 16, "append_pages": 4, "forks": 1},
    "sweep": {"seed": [41, 73], "family": ["tools", "document"],
              "hbm_bytes": [16 * 2**20, 24 * 2**20, 32 * 2**20], "sessions": [1, 2, 4]},
}
(root / "parameters.json").write_text(json.dumps(parameters, indent=2))
PY
uv run --isolated --no-project --with matplotlib \
  python -m benches.storage.state_cache --model Qwen/Qwen3.8-27B \
  --config "$SIM_ROOT/config.json" --parameters "$SIM_ROOT/parameters.json" \
  --output "$SIM_ROOT/run" --plot
```

`sources.py` records the ten official config revisions from design v2 and their
raw-file SHA-256 hashes. The loader rejects mismatches and files over 1 MiB.
It derives backbone groups, known head dimensions, compression ratios, windows,
linear-state groups and indexer candidates from fields, never model-name branches.
Extra draft/MTP, vision and embedding state is excluded. Preserve engine schema
version as **unavailable: config-derived IR v1**, not a released engine manifest.
No config is redistributed and no dataset is imported.

Bytes per native entry, indexer entry and checkpoint (including convolution
carry) are explicit parameters. Entry sizes must include native scales and any
format-specific overhead; page metadata and alignment are charged separately.
Weight quantization does not establish KV precision. Layer counts multiply these
assumed sizes; no additional cross-layer sharing is inferred from source-layer
lists. Those lists remain unresolved facts. Exact physical aliases are represented
by the same owner key in analytic fixtures and shared prefixes. Indexer candidates
are filtered by the explicit `indexer_ratios` parameter (default 1/2/4); this is
an assumption, not proof that every candidate owns an indexer. Replay legality
and per-token cost are also hypothetical inputs, not released engine contracts.

## Generator and event model

`Workload` supplies deterministic seeds, sessions, turns, fork fan-out, prefix
sharing, append lengths and checkpoint spacing. Chat halves tool delays; document
QA doubles them; tool chains keep the configured scale. These are synthetic
shapes, not fitted real-world distributions. Chat/tools append to prior turns;
document QA shares a fixed document prefix within session pairs and creates a
fresh `append_pages` question suffix each turn. It never replays a prior question's
checkpoint. Log-normal delays use the configured
median scale and log-space sigma, capped at four times the scale before family/adversarial
multipliers. Adversarial cases cycle through immediate, eight-times-late,
never-return (expiry), cancel, and ordinary returns. Turns are separated by 128
times the scale; arrivals are open-loop and do not adapt to policy results.

Publishing creates a boundary and externally materialized warm DRAM/SSD state;
it does **not** model a prefill kernel or charge prior producer writes as transfers.
`cold: true` removes these backing allocations and requires explicit recomputation.
Do not mix warm and cold rows. Every request gets its own serialized token chain.
Forks share immutable roots or pay copied bytes through the same transfer ledger.
Checkpoints are retained at configured turn spacing; other boundaries can only
use a complete earlier recipe plus charged replay, or full recomputation.

Sparse sets have configurable cardinality and overlap; every fourth step can be
disjoint. Selections become visible only after that request's query/indexer step.
The online predictor reuses the last observed set; it cannot read the next set.
The oracle may see next selections and tool returns and is always labeled as
such. It still pays byte movement and queueing: there is no zero-cost overlap mode.

One shared non-preemptive link serializes every transfer, including fork copies,
demotion and speculation. DRAM and SSD can have different service parameters.
One compute lane serializes indexer, replay, recomputation and token work. This
is deliberately conservative; it does not invent layer overlap, NVLink bandwidth,
quality preservation or an installed engine's compute scheduling. A sparse cold
miss may pay full recompute again when a later selected page was not materialized.
HBM capacity means cache bytes after weights and runtime workspace, not device
capacity. Sparse pages aggregate layer groups; no layer pipeline is assumed.
Decode steps read the supplied boundary; KV growth appears at the next producer
boundary. Active inference workspace and output quality are outside this model.

Readiness and drain are separate receipts. Source grants are modeled by epoch-
checked, pinned source references; paged requests lease their complete backing
boundary through decode. Reader leases, transfers and compute reservations pin
physical owners. Cancellation/expiry removes logical reachability only; an epoch
change fences old state. None releases undrained owners. Explicit stale or
incomplete reads and early release attempts fail the control. This is a bounded
protocol/accounting model, not proof of native DMA safety or the future TLA+ model.

## Policies and ablations

All controls are **approximations**, not executions of vLLM, LMCache, HiCache or
any engine's native cache. No product performance comparison is implied.

| Name | Mechanism |
| --- | --- |
| `full_lru` | Whole-history snapshots, copied forks, LRU tiers |
| `static` | Whole-history snapshots, copied forks, first-admitted backing tiers |
| `minimal_lru` | Window-aware complete boundaries with LRU tiers |
| `shared_lru` | Window-aware LRU plus shared prefix/fork roots; stronger COW control |
| `p1` | Marginal-benefit/byte greedy closure retention; costed load/replay/recompute |
| `p2` | Shared versions plus JIT from the last 32 completed tool durations |
| `p3` | Serialized sparse demand paging with a null last-set predictor |
| `all` | P1 + P2 + serialized sparse demand paging |
| `oracle` | `all` with future return/selection knowledge; separate lookahead control |

Planner values use past accesses only. Eviction cannot remove a dependency from
a protected closure, lease or transfer. Tiny exhaustive subset enumeration bounds
the greedy *static packing objective*; it is not a global offline timing optimum.
P1 may reject admission when selected closures consume the budget. It does not
search unbounded future eviction/replay schedules.

Set `policies` to names or records such as
`{"base":"all","name":"no_jit","jit":false}`. Independently disable
`minimal`, `dag`, `recompute`, `sharing`, `jit`, or `prefetch`; keep `sparse` true
to ablate prediction while retaining demand paging. In `consume()`, the last-set
prediction is the selection just observed at this step's `selection` event.
Those pages are already resident and leased in HBM, so the speculative `ensure`
cannot move bytes. Agreement with the no-prefetch ablation is guaranteed by
construction, not evidence for or against P3.
Sweep any scalar `Workload`, `Sizes`, `Costs` field, or `hbm_bytes`, `dram_bytes`,
`ssd_bytes`. Vary relative component bytes, checkpoint spacing, fan-out, route
costs, overlap and `prediction_scale` in separate bounded campaigns. Freeze seeds
and holdouts before comparing; never tune them after observing policy wins.

## Bounds, outputs and interpretation

`Limits` caps component/boundary metadata, physical owners, branches, reader
leases, in-flight batches and references, dependency edges, live events, processed
events and per-call planner work. Defaults are 16,384 nodes, 32,768 owners, 256
branches, 4,096 leases, 2,048 flights, 65,536 live events/references/work units and
262,144 processed events. Nested input metadata is capped at four times planner
work. Inputs, logs and latency samples therefore have finite structural bounds;
these are not claims about Python object RSS. Exact packing stops at 12 candidates;
greedy search admits 64. The runner caps campaigns at 512 trials by default (hard
maximum 4,096). Resource exhaustion rejects admission or fails the run, never
silently truncates a plan or frees a pinned owner.

Artifacts include the revision/dirty state, source hashes, config hash, full
parameters, trace, event log, per-run JSON, CSV sweep, optional plot, point gates
and concurrency-capacity table. Failed runs are preserved. Existing output
directories and paths resolving into this checkout are rejected.
Source snapshots without Git metadata record it as unavailable and still hash
every simulator source file. Plots include rejected/SLO-missing trials; use the
matched tables, not a scatter point alone, to compare policies.

Metrics include complete-boundary, component and token hit rates; cold/resume
TTFT and TPOT p50/p95/p99; completed token throughput; stored unique/replicated
peaks, HBM byte-seconds, wire/destination bytes, fetch stall and useful speculative
bytes. Counts include in-flight destination allocations, checkpoint/indexer state,
page metadata/padding and retained owners. Ratios give separate prefill-exposure
and completed-decode token denominators. An input prefix reused by several
requests counts once per exposure, not once per globally unique text token.
Capacity is the greatest *tested* concurrent session count satisfying the modeled
SLO with no rejected requests; a top-of-sweep result is censored, not a maximum.

`point-gates.json` applies the P1 and P3 byte/latency thresholds against **every**
compatible approximation above. It cannot promote a direction: additionally
require the design's capacity gain, two frozen holdout families, repeated seeds
and adjacent budgets, including the stronger shared/window-aware controls. P2
needs capacity and latency evidence, not just fewer fork bytes. The oracle is
excluded from online winners. Apply kill criteria only to mechanisms actually
modeled; neither a pass nor a failure qualifies or disproves hardware behavior.

### Interpretation of the submitted runs

The [PR #333 runs](https://github.com/OrbitInfra/OrbitKV/pull/333) at `8594d5b8`
are **simulation only, not hardware evidence**; the handoff records external
artifacts and uncalibrated assumptions.

**P2:** every policy, including the baselines, saves HBM-only state and frees HBM
at `tool_start`, subject to pins and physical drain. Freeing HBM during tool waits
is therefore common baseline behavior. P2's possible gains are version sharing
and JIT timing; `shared_lru` already includes version sharing, so comparison with
it isolates JIT timing. Equal tested capacity at 12/12 budget/seed/family points
per config means JIT earned no capacity gain over that control. It does not test
the benefit of freeing HBM during tool waits. Some JIT TTFT gains occur, but the
required capacity gain does not; the P2 configuration does not earn promotion.

**P3 is untested in its main mechanisms.** There is no cross-layer pipeline, and
the online predictor cannot name newly selected, nonresident pages. These runs
test serialized per-step demand paging. `p3` moves zero speculative bytes in all
72 Qwen/DeepSeek rows; `all` and `no_prefetch` have identical speculative bytes
throughout the ablation and sensitivity campaigns, all from JIT. The null
last-set predictor cannot support a P3 kill verdict. The Qwen lookahead oracle's
p95 TPOT of 0.25–0.76× the online combined policy is measured headroom using
**unavailable future knowledge**, not an online gain.
