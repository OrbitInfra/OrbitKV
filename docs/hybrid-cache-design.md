# Hybrid-attention cache design and experiment plan

**Proposal; all experiments are unexecuted.** This document changes no runtime
support or qualification claim. Delivery decisions live in the
[completion plan](completion-plan.md#s7--semantic-retention-and-checkpoint-compiler).
OrbiTE is the only transfer engine. GPU, RDMA, GDS and installed profiles need
their own hardware runs; CPU controls cannot qualify them.

## Problem and existing contract

An external hit must recover a state from which the engine can resume at token
boundary `t`. A hybrid model can require several different kinds of state at
that boundary. A long attention hit is useless if its required checkpoint or
window is missing. Model, computation, layout and prefix identity must match;
equal token counts alone do not establish reuse.

For one layer, let `T` be prefix length, `W` the declared past-token window,
`h` KV heads, `d` head dimension and `b` bytes per scalar. Sizes below are logical
payload estimates before page rounding, alignment, replication or codecs; actual
engine tensor layouts are authoritative.

| Component | Size and reuse | Recovery requirement |
| --- | --- | --- |
| Full attention | Approximately `2*h*d*b*T`; identical-prefix K/V can serve shorter aligned prefixes | Contiguous missing prefix, with valid engine state at the starting boundary |
| SWA | Approximately `2*h*d*b*min(T,W)` for one boundary; different retained boundaries may need different windows | Complete trailing window, including the engine's positional/window convention, rounded to pages |
| Mamba / Gated DeltaNet | Recurrent matrix/state and any convolution history; one snapshot has layer-specific size `S`, independent of `T`; retaining `q` snapshots costs `q*S` bytes | Exact state at `t`, with all auxiliary tensors; the final state cannot be truncated back to an earlier boundary |
| MLA | Compressed latent and positional state per token; size follows the engine layout, not the ordinary two-head-array formula | Prefix recovery where `MlaKv` describes the complete required state |
| Sparse attention | Selected K/V plus any indexer, positional or routing state; size depends on the model and selected pages | An engine-declared sufficient page set and its auxiliary state; MLA compression alone does not supply this contract |

The existing [compiled recovery contract](hybrid-recovery.md), implemented in
[`orbitkv-state`](https://github.com/OrbitInfra/OrbitKV/tree/37722fec3dfa81910f35fcf5885ca646af2607a7/crates/orbitkv-state/src),
is the starting point, not a new generic planner:

- `StateComponent` names attention, MLA, SWA, recurrent checkpoints, convolution,
  draft, indexer and opaque state. Naming a component does not enable it.
- `RecoveryRule::Prefix`, `Window { tokens }` and `Checkpoint` declare demand.
  `StateRequirement` groups components that must be stored and leased together.
  `StateBundle` describes supplied coverage; live leases still prove ownership.
- `RecoveryContract` compiles page-aligned required ranges and validates complete
  common boundaries across groups. Ranks intersect boundary sets, not just their
  largest endpoints. A checkpoint's final-page range denotes a snapshot at that
  end, not recurrent state for every token on the page.
- Current compilation requires a prefix group zero and one shared page-token
  granularity. Prefix accepts attention/MLA, Window accepts SWA, and Checkpoint
  accepts recurrent/convolution state. Draft, indexer and opaque requirements
  are rejected. Pure recurrent/window-only models and arbitrary sparse want-sets
  are outside this compiled contract.

This already avoids reading unnecessary window/checkpoint history when a legal
boundary is available. It does **not** derive rules from model mathematics,
capture checkpoints, choose branch points, retain dependency sets, decide joint
eviction or make unsupported engine handoffs safe. The proposals below must
show incremental value over that existing minimal-range path.

## Released engine boundary

The source baselines are vLLM **0.31.0** (`db9527a4`) and SGLang **0.5.22**
(`baa9befa`). See the [released interface audit](engine-release-audit.md) and
[adapter ownership contracts](adapters.md) for current guards and lifetime rules.

| Direction | Released surface and present limit | What is needed before enabling it |
| --- | --- | --- |
| SGLang Full + SWA retention/transfer | `UnifiedCacheLinker.lookup` reports all restorable boundaries; `load`, `start_layer_wise_loading`, `finish_layer_wise_loading`, `offload` and FIFO offload completion carry `PoolTransfer`s. Released external components are **Full/SWA only** | Use those callbacks and existing private-destination/tree locks; verify the actual model's layout. Normal-path experiments need hardware, not a new component API. Unsupported SWA request rings remain rejected |
| vLLM hybrid port | OrbitKV allows **V1, one attention group**, with `VLLM_USE_V2_MODEL_RUNNER=0`; V2 and multi-group/recurrent serving are rejected. Upstream vLLM itself has a hybrid allocator and `SupportsHMA.request_finished_all_groups` | Consume a complete group-aware handoff through allocation, preemption, load/save and completion before lifting OrbitKV's guards; an HMA marker alone is insufficient |
| Recurrent capture/restore on either engine | OrbitKV's generic checkpoint layout is not a released model lifecycle; SGLang's external linker rejects recurrent/SSM components | Engine-owned snapshot-at-boundary capture, complete recurrent/conv tensor descriptors, producer readiness, destination allocation identity, restore-before-zero/COW/compute ordering and terminal completion. **Blocked on a complete released contract** |
| Sparse indexer-driven loading | `IndexerState` has no accepted recovery rule; neither selected OrbitKV adapter exposes a sufficient-page-set handoff | Engine-owned, versioned indexer output and auxiliary-state recovery, page visibility before attention, invalidation and cancellation. **Blocked on upstream and recovery-contract design** |

Pinned sources: [vLLM connector interface](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
and [SGLang external linker](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py).
In vLLM, the relevant callbacks are `get_num_new_matched_tokens`,
`update_state_after_alloc`, `handle_preemptions`, `start_load_kv`,
`wait_for_layer_load`, `save_kv_layer`, `wait_for_save`,
`request_finished[_all_groups]` and `get_transfer_results`.

Existing upstream work, checked on 2026-11-07, remains distinct from a consumed
release:

| Contract | Existing item | Consequence for this plan |
| --- | --- | --- |
| Allocator-issued generations on vLLM block handoffs | [#62810](https://github.com/vllm-project/vllm/issues/62810), candidate [#62819](https://github.com/vllm-project/vllm/pull/62819) | Open; needed to distinguish reuse within one allocation. Does not itself drain DMA or provide checkpoint capture |
| Allocator-issued SGLang page generations carried with `PoolTransfer.device_indices` | [#45493](https://github.com/sgl-project/sglang/issues/45493), candidate [#45548](https://github.com/sgl-project/sglang/pull/45548) | Open; candidate Full/SWA generation/readiness work explicitly does not enable Mamba. Fault-safe reuse qualification remains blocked |
| V2 preemption drain before zero/COW request updates | [vLLM #59410](https://github.com/vllm-project/vllm/pull/59410) | Open; necessary ordering repair for that direction, not a complete recurrent contract. Keep the V1 guard |
| SGLang pre-capture event preparation via `register_radix_cache_backend(..., prepare_worker=...)` | [#44856](https://github.com/sgl-project/sglang/pull/44856) | Open; replaces the `TpModelWorker.init_cuda_graphs` Hook only after release and consumption; recapture still needs a contract |
| SGLang pending lookup/rank agreement before admission | [#44873](https://github.com/sgl-project/sglang/pull/44873) | Open; replaces the `PrefillAdder.add_one_req` Hook only after release and consumption; it is not payload completion |

Full/SWA prototypes retain the existing Hooks and conservative owner retention.
They do not claim public-lifecycle migration or fault reclamation. No complete
released recurrent snapshot or sparse-indexer extension is identified by those
items; re-audit releases before starting those directions, rather than treating
generation IDs or native HiCache support as sufficient.

## Prior art and claim budget

[Marconi (MLSys 2025)](https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf)
already selects recurrent snapshots at useful prefix branches and considers
saved computation per storage cost. It explains why recurrent state cannot be
recovered at arbitrary earlier prefix ends like ordinary K/V.

[SGLang HiCache](https://www.sglang.io/blog/hicache) tiers state across GPU, host
and external storage; native UnifiedRadixCache recovery includes hybrid state
and incremental branch-checkpoint backup. Its native capabilities are broader
than the selected released **external-linker** contract.

[LMCache's hybrid design](https://docs.lmcache.ai/mp/hybrid_models.html) separates
objects by attention-window groups, including separate recurrent state storage.
Its current documentation is prior art, not evidence that those integrations
are available through OrbitKV's pinned adapters.

[vLLM's hybrid allocator](https://docs.vllm.ai/en/v0.31.0/design/hybrid_kv_cache_manager/)
already groups allocations, frees out-of-window SWA blocks and intersects group
prefix hits. [Mooncake](https://arxiv.org/abs/2407.00079v5) places KV across
distributed DRAM/SSD and schedules around reuse and resource constraints; it is
prior art here, not an OrbitKV transfer backend.

**No new caching algorithm is claimed.** Branch checkpoints, window pruning,
joint recovery and transfer priority already have precedents. The possible
contribution is a bounded external-cache implementation of those policies using
OrbitKV's explicit recovery/ownership contract, with measured benefit beyond
native caching and today's compiled range selection. Even that benefit is a
hypothesis until matched experiments establish it.

## Candidate directions

### D1: branch-point recurrent checkpoints

**Hypothesis:** with a fixed checkpoint budget, snapshots at reused conversation
branches recover more useful tokens than uniform periodic snapshots.
**Mechanism:** use observed prefix branching to request engine-owned captures at
legal aligned boundaries; retain all recurrent/conv tensors atomically. A branch
discovered after prefill cannot conjure a past snapshot: wait for another pass
or explicitly charge recomputation/capture cost. Compare branch selection with
periodic checkpoints and native policy over the same byte budgets.

**Metric/expected gain:** a better checkpoint-bytes versus useful-hit-rate curve;
the decision gate is at least **5 percentage points** more useful token hits at
equal checkpoint bytes, or **20% fewer checkpoint bytes** within one point of
the same hit rate. **Failure:** rare/repeatedly shifting branches, expensive
capture, large state, or full-attention retention dominates. No benefit may
remain after replay/capture time and bandwidth are charged. Blocked as above.

### D2: component-aware storage and joint eviction

**Hypothesis:** retaining sufficient state per boundary beats treating all layer
history as equally valuable. **Mechanism:** keep the union of windows required
by retained boundaries and active leases; drop SWA pages outside that union.
Preserve checkpoint/conv groups jointly with their dependent recovery boundaries.
Evicting a checkpoint invalidates dependent deeper attention-only hits or replay
plans; a later independently complete checkpoint remains valid. Do not remove
an old window merely because it is outside the newest request's window.

**Metric/expected gain:** fewer resident bytes per reusable token, or more useful
hits at equal DRAM/SSD capacity. First-model gate: **20% less resident payload**
at no more than **one percentage point** useful-hit loss, with the serving
non-regression gate below. **Failure:** windows are already pruned by the engine,
their retained union approaches full history, full layers dominate, or dependency
bookkeeping/capture costs exceed saved bytes. Full/SWA comes first; checkpoint
retention waits for D1's engine contract.

### D3: component-aware transfer

**Hypothesis:** a restore should move only missing full-attention prefix pages,
required windows and the exact checkpoint, not every group's full history.
**Mechanism:** extend the existing `required_ranges` consumption into the proposed
retention/capture policies; form bounded file/peer/local descriptors and preserve
group completion. This range minimization already exists for declared contracts;
measure it as the baseline before claiming additional coalescing/policy gains.
Full-attention layers still need their missing prefix: “window plus checkpoint”
does not eliminate that part of a hybrid model.

**Metric/expected gain:** at least **20% fewer physical bytes per recovered token**
versus a valid all-history transfer control; a serving speedup additionally needs
**5% lower p95 TTFT**. **Failure:** small windows were already selected, full-layer
traffic dominates, or extra reads/fragmentation/padding erase savings. If only
bytes improve, report a byte reduction, not a serving benefit. Recurrent
extension is blocked; current Full/SWA range selection can be measured first.

### D4: recurrent state first

**Hypothesis:** small prerequisite state waiting behind large/background transfers
can extend the restore critical path. **Mechanism:** order bounded submissions
with checkpoint/conv descriptors first when dependencies permit; retain the
original scheduler deadline across all submissions. Demand reads, including
checkpoint reads, remain `ForegroundGet/High`; warming is
`BackgroundPrefetch/Low`, and checkpoint **writes** remain `Checkpoint/Low`.
High priority alone cannot order two foreground components. Local copies use
stream/descriptor order; OrbiTE's local-copy API has no per-copy QoS field.

**Metric/expected gain:** **5% lower p95 TTFT** under matched background pressure,
with no more than **5% goodput loss** and no new background deadline misses.
**Failure:** the model waits for all groups anyway, large transfers are not
preemptible, or small submissions reduce bandwidth. Do not promise preemption,
intra-batch priority or CPU/TCP priority statistics. No early model resume without
a released dependency contract; recurrent testing remains blocked upstream.

### D5: later indexer-driven sparse loading

**Hypothesis:** an engine's exact indexer-selected page set is much smaller than
the retained attention history. **Mechanism:** consume a versioned sufficient
want-set and required auxiliary state before attention uses it; never infer
indices or approximation tolerance in the cache. This needs a new OrbitKV
recovery rule and engine handoff, not a reinterpretation of `Prefix`.

**Metric/expected gain:** target **20% fewer physical bytes** with identical
selected pages/results; selection and stall cost must be below saved transfer
time. **Failure:** indices arrive too late, change per token, select nearly all
pages, or depend on state not covered by the snapshot. Defer until the upstream
contract exists; no sparse support or quality claim follows from this proposal.

## Experiment plan (unexecuted)

**First target:** SGLang 0.5.22, **gpt-oss-20b**, Full + SWA, TP=1/PP=1.
Its [model configuration](https://huggingface.co/openai/gpt-oss-20b/blob/cbf31f62664d4b1360b3a78427f7b3c3ed8f0fa8/config.json)
declares alternating full/sliding layers and a 128-token window. Freeze the
complete model/tokenizer revision, engine wheel and effective layout before any
run. Start eager, no speculation, raw BF16 KV where the released backend permits
it; inspect actual dtypes/shapes rather than infer them from weight quantization.
If the released layout needs an unsupported request ring or cannot fit the host,
stop and review a Gemma 3 Full/SWA replacement manifest; do not patch the engine.
Start DRAM, then io_uring SSD; peer, Graph, GDS and RDMA are separate later scopes.

**Second target:** Qwen3-Next (or Kimi Linear), after a released complete recurrent
external-cache contract is consumed. Freeze a supported model/release/hardware
combination then; the current recurrent rejection stays in place.

### Matched baselines and ablations

| Arm | Purpose |
| --- | --- |
| Engine-native caching, external cache disabled | Serving/correctness baseline; retain native HBM prefix caching in every arm |
| Native HiCache with equivalent tiers, when the released model profile permits | Strong component-aware comparator at the same HBM/DRAM/SSD byte budgets |
| Homogeneous full-KV policy on the same hybrid model | Retain/transfer all exportable attention history uniformly; complete recurrent snapshots are still required, never fabricated as K/V |
| Current OrbitKV compiled recovery | Separate existing window/checkpoint range savings from proposed retention, capture and ordering changes |
| One direction at a time, then combinations | Attribute gains; compare branch checkpoints with periodic selection at equal checkpoint budgets and priority ordering with unchanged deadlines |

The homogeneous control must preserve model mathematics and valid destinations.
If the engine has already discarded old SWA pages or cannot export them, mark a
full-history arm **infeasible**. Compare uniform retention of available state
instead, and label a calculated full-KV byte upper bound as analytical, not a
measured baseline. Never recreate attention history by changing the model, write
outside its window, or claim improvement against nonexistent exported bytes.

Use multi-turn chat, branching agent/tool histories and long-document QA, plus
controlled shared-prefix depth/fan-out and a zero-sharing negative control.
Sweep context lengths (initially 4K/16K/64K within native/hardware limits), shared
prefix fractions (0/25/75/100%) and working sets (0.5/1/2/4 times the fixed DRAM
budget). Freeze feasible cells before collection; record exclusions. Keep token
IDs, output caps, arrival traces, native HBM allowance and total tier budgets
identical. Charge capture buffers, metadata, alignment and replication; separate
cold, warm, post-HBM-eviction and engine-restart conditions.

Correctness precedes timing: verify each restored component's bytes against a
cold compute oracle and greedy outputs against the native arm, with any native
numerical tolerance declared in advance. Exercise partial windows, missing or
wrong-boundary checkpoints, one-component eviction, stale identities and late
completions. Cancellation, timeout, page reuse and source loss must preserve
owners; an unavailable allocator/termination contract blocks that qualification,
not a reason to convert the failure into a passing control.

Measure p50/p95 TTFT, output-token throughput and SLO goodput, complete-boundary
request/token hit rate, and logical/physical bytes moved and stored per token.
Transfer denominator: tokens actually recovered; storage denominator: distinct
retained prefix tokens. Count checkpoint captures, replay, padding, staging and
all transfer legs, not just selected payload. Report useful complete hits rather
than summing per-layer hits. Use engine statistics where exposed; missing QoS or
internal-staging counters are a reporting limit, not inferred measurements.
An unmeasurable physical-byte gate remains unqualified.

Run at least five independent paired process runs per chosen cell, randomizing
arm order and separating warmup. Publish per-family distributions and paired
95% confidence intervals across runs; requests in one run are not independent
replicates. Gate throughput claims on the same arrival/load regime and budgets.
Record hardware, NUMA/storage settings, revisions, wheels, seeds, traces, failures
and controls in an external artifact directory under the
[benchmark evidence contract](benchmark-evidence.md); no result files enter Git.

## Milestones and kill criteria

These are proposed decision thresholds, not observed gains. Freeze them before
collection. A gain must clear its threshold with a paired 95% interval; retain
negative cells and require two cache-pressure levels in at least two workload
families. Serving non-regression means the interval excludes more than **5%**
worse p95 TTFT or **5%** lower goodput. Insufficient precision is inconclusive,
not a pass; any extra runs need a declared sampling extension, not moved targets.

| Gate | Decision |
| --- | --- |
| M0: released layout and correctness | Inspect real Full/SWA tensors, reproduce native outputs and complete restore bytes; any missing component/unsafe reuse stops the profile before timing |
| M1: first-model D2 retention | Require the stated 20% resident-byte saving within one point of hit rate, plus serving non-regression; if not met, stop that retention direction on this model rather than promote it |
| M2: D3 transfer ablation | Separate current range-selection savings from new changes; keep only additions that clear their byte gate and do not regress serving; no novelty claim for existing behavior |
| M3: recurrent D1/D4 | Wait for released contracts, then measure checkpoint budget/hit curves and contention ordering separately; reject either policy if its own gain gate or serving non-regression fails |
| M4: sparse follow-up | Require a complete engine/auxiliary-state contract first; stop if selection overhead consumes savings or correctness requires unsupported state |

Any stale write, wrong result, early release, unbounded growth or leaked-owner
growth beyond the configured retention budget is a correctness stop regardless
of speed. No milestone expands the first release profile automatically; hardware
and installed-artifact qualification remain separate maintainer-reviewed gates.

## Ownership impact and bounds

Extend existing registration, query, planning, residency and completion owners;
do not add a second scheduler or Python cache coordinator. Inference engines own
HBM allocation and capture semantics; OrbitKV owns retained state and complete
recovery decisions. Candidate metadata and dependency edges need explicit caps:
initial proposal **4,096 retained boundary records / 32 MiB per Manager**, charged
to the index budget, with admission refusal when either cap is hit. Payload and
capture buffers must fit existing configured byte budgets; no unbounded snapshot
stash. Freeze these experimental caps with the run manifest.

| Direction | Owner or bounded metadata needed | Release/visibility receipt |
| --- | --- | --- |
| D1 checkpoints | Capture owner retains every engine-exported tensor and producer dependency; snapshot bytes, pending captures and branch records consume bounded admission | Producer readiness then complete sealed-group publication; source memory stays held through physical transfer completion |
| D2 retention | Existing residency/index owner tracks boundary dependencies and active leases; cap edges as part of the metadata budget | Invalidate dependent discovery before reclaiming; final lease and all accepted transfers drain before bytes/extent reuse |
| D3 transfer | Existing query/source leases and persistent destination registrations cover every selected group; descriptor/byte limits apply to every split batch | Combined restore completion establishes all required bytes; no independent group success exposes a partial boundary |
| D4 ordering | Existing bounded admission holds queued components under one deadline; scheduler chooses order without allocating a parallel queue | Priority/deadline/cancel are not release receipts; local stream completion and file/peer physical drain remain required |
| D5 sparse | Engine-attested want-set, indexer version and referenced pages form a bounded recovery owner; reject unsupported/oversized sets | Correct auxiliary-state version and all selected page visibility before attention; late results cannot target reused allocations |

Allocation generations must originate in the inference engine. Existing grants,
epochs, requester-specific `close_segment` retirement, local completion owners
and [transport limits](transport.md) continue unchanged. Timeout, cancellation
and Unquiesced retain pins, leases and registrations; a retry of waiting is not
permission to replay unknown work. Reclamation still needs physical drain.

**OrbiTE needs no new features for these directions.** It continues to receive
existing local/file/peer descriptors, request intents, priorities and deadlines;
OrbitKV selects ranges and submission order. OrbiTE owns movement and staging,
not model state or boundary validity. The missing contracts above belong to
inference engines and OrbitKV's recovery/retention policy, not the byte engine.
