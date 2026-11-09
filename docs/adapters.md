# vLLM and SGLang integration

Use official **vLLM 0.33.0** and **SGLang 0.5.23** with the
[single-node quickstart](single-node.md). OrbiTE is the sole cache transfer engine;
GPU/RDMA/GDS and installed profiles need fresh hardware qualification. Historical
[ordinary, Graph, shared-Manager and two-host acceptances](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/adapters.md)
retain their original releases/backend and do not carry over. Current limits are
in [compatibility](compatibility.md), status in the [completion plan](completion-plan.md).

## Release upgrade audit

Checked November 9, 2026 against official final tags and PyPI:

| Engine | Previous → current | Source revision |
| --- | --- | --- |
| vLLM | 0.31.0 → **0.33.0** | [`4ba787ec`](https://github.com/vllm-project/vllm/tree/4ba787ec9959e566d3b587a796dff07673324da5) |
| SGLang | 0.5.22 → **0.5.23** | [`297e54fd`](https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90) |

The vLLM release page still points to `v0.33.0rc3`, but the official
[final tag](https://github.com/vllm-project/vllm/tree/v0.33.0) and
[non-yanked 0.33.0 package](https://pypi.org/project/vllm/0.33.0/) exist.
The annotated final tag resolves to the same source commit as rc3; the pin uses
that final tag, not a release candidate. The extras follow published Linux artifacts: Python 3.11–3.14 for vLLM and
3.11–3.13 for SGLang (no 3.14 wheel or sdist). OrbitKV's engine-free package
retains Python 3.10+. Use separate environments.

| Consumed surface | Change and OrbitKV response |
| --- | --- |
| vLLM `KVConnectorBase_V1` | Adds `supports_sleep_mode`, `unregister_kv_caches` and `restore_kv_caches`. OrbitKV returns false from the capability method: sleep remaps storage and current registrations cannot follow it safely. No sleep/remap support is claimed. |
| vLLM loaded groups / hit attribution | Adds `get_loaded_kv_cache_group_ids` and `get_external_cache_hit_sources`. Inherit the one prefix-cacheable group's ID and `EXTERNAL_UNSPECIFIED`; do not guess a DRAM/disk/peer split from a mixed OrbitKV restore. |
| vLLM completion / preemption | `KVConnectorTransferResults`, `handle_preemptions`, request completion, reset and `SupportsHMA` callbacks retain their signatures. V1 is unchanged; V2 is opt-in for uniform Full attention under the ordering below. V1 Full + SWA uses the HMA contract below; recurrent and V2 multi-group remain rejected. MultiConnector forwards the new capabilities; best-effort cache and reliable native P/D remain separate. |
| vLLM private dependencies | `resolve_kv_cache_block_sizes(config, vllm_config)` stays at `v1.core.kv_cache_utils`; it now rejects a mismatched single-group `prefix_match_unit`. Consume that rejection. `bind_kv_cache_manager` still forwards to `bind_gpu_block_pool`; pool pinning, block hashes, `extract_layer_index`, layout specs and metric construction remain consumed unchanged. |
| SGLang `UnifiedCacheLinker.offload` | Adds optional `session_id`; accept it without using session attribution as cache identity. Native wrapper supplies it. |
| SGLang producer readiness | Consume released `record_offload_ready()`, passing the worker's `forward_stream`. Hold and wait both index/current-stream and forward-stream events before submitting a prepared Save. Unknown completion still retains the tree lock and registration. |
| SGLang wrapper pool contract | Supply `pools` explicitly: the wrapper now uses it to decide whether SWA is available. Full/SWA stay complete; request-ring/replay layouts, sidecar/draft state and separate scale buffers are rejected. |
| SGLang linker construction / components | The new dynamic linker factory supports external classes and Full/SWA/Mamba. Keep the public RadixCache factory because OrbitKV still needs `RecoveryLinkerWrapper` for absolute boundaries and rank agreement. Static Mamba checkpoints now use that released component; see the component audit below. |
| SGLang retained private dependencies | `DevicePoolEntry._row_span`, location translation, `hit_markers`, `inflight_load_rids`, `_tail_hashes`, `_all_reduce_attn_groups`, resolved runtime configuration and the three hook targets below retain their consumed signatures. Native private destinations, post-forward insertion and `finish(ABORT)` remain the owners. No hook is removed. |

Sources: [vLLM base contract](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py),
[SGLang linker and wrapper](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py),
[linker factory](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/linker_factory.py).
The released vLLM LMCache entry remains a thin external-package delegate; it does
not remove these OrbitKV ownership obligations.

CPU API checks validate released classes, rejection and callback wiring. Installed
GPU eager/Graph output and byte checks, abort/preemption, queue waits, engine and
Manager restart, identity collisions, rank shapes and native P/D are **unrun and
unqualified for both new pins**, queued in [M1](completion-plan.md#m1--engine-baseline).

## Ownership contract

| Responsibility | Owner |
| --- | --- |
| Request scheduling, HBM allocation and native prefix cache | Inference engine |
| Layout/rank descriptions and producer/consumer dependencies | Python adapter |
| Replicas, planning, byte admission, leases and accepted tasks | OrbitKV Rust owners |
| Raw DRAM Restore | Native client executor with Manager-held sources |
| Publish and SSD/codec Restore | Manager workers |
| Cache byte movement | OrbiTE local/file/peer APIs |
| Live P/D | Official inference-engine transport and lifecycle |

A hit proves a legal boundary across every required group, not merely a stored
key. Model/computation/layout identity must match. Readiness, terminal completion
and source reaping are separate events; sharing a Manager does not make engine
layouts interchangeable.

Before queueing, `prepare_save` captures immutable requests and the original
`GpuRegistration`. Rust budgets and batches `submit_saves`, retaining exporters
until completion. `cancel_save` proves non-submission only before launch.
Unknown handoff keeps engine completion pending; SGLang also retains FIFO locks.
Only cancellation of the pre-forward activation future proves a Restore was
never submitted. Cancellation during submission/wait retains destinations and
leases and propagates failure.

The selected releases lack allocator-issued page generations for their handoffs.
Registration lifetime cannot detect every same-allocation slot reuse. Uncertain
pages remain held; local counters cannot replace an engine allocator contract.
The [hook and upstream contracts](#remaining-internal-hooks)
tracks proposals separately from released consumption. Existing
[normal-path lifetime evidence](https://github.com/OrbitInfra/OrbitKV/blob/1d173c1a27401f6e219c5120f97ba5eab3e2d52b/docs/adapters.md#ownership-contract)
does not establish cancellation, physical termination or eventual fault reclamation.

## Explicit cache reset

Official vLLM 0.33.0 distinguishes an HBM-only prefix reset from
`POST /reset_prefix_cache?reset_external=true` (development API), or the public
engine `reset_prefix_cache(reset_connector=True)` call. OrbitKV refuses an
external reset while it tracks requests, query leases, restores or saves. Once
idle, it changes the prefix-key generation used by both lookup and publication.
New requests cannot reuse the previous generation; newly saved state can be
reused after an HBM-only reset. Each immutable process restart still derives its
initial domain from the model artifacts and computation configuration.

This reset affects one engine process, not all replicas or physical Manager
storage reclamation. Its keys are not shared with other replicas after reset.
It does not prove safe live weight updates that skip reset, active-request reset
or cancellation of native DMA. SGLang's ordinary `flush_cache` continues to clear
HBM and retain same-weight external reuse; released skipped-flush weight updates
remain unsupported. The completion plan records qualification separately.

## Immutable static LoRA

The fixed local PEFT adapter contract has historical vLLM 0.31.0 / SGLang 0.5.21
dense TP=1/PP=1 evidence in [performance](performance.md#historical-evidence).
It does not qualify the current OrbiTE path.
At startup OrbitKV hashes each adapter's configuration and weights and binds the complete named set and LoRA
computation configuration to the external cache namespace. Engine prefix hashes
still distinguish the base model and individual adapters. Replacing weights at
the same name/path between process restarts changes the namespace. Changing any
adapter or adding one invalidates reuse for the whole declared set, including the
base model; this is conservative deployment isolation, not independent adapter
version migration. `ORBITKV_MODEL_FINGERPRINT` never substitutes for adapter bytes.

For vLLM, declare the same adapters in the released loader and the connector:

```bash
vllm serve /path/to/immutable-model --enable-prefix-caching --enable-lora \
  --lora-modules fixed=/path/to/immutable-adapter \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm","kv_connector_extra_config":{"orbitkv.static_lora_adapters":[{"name":"fixed","path":"/path/to/immutable-adapter"}]}}'
```

The public `on_new_request` callback validates name/path and fixed bidirectional
integer ID bindings before native HBM lookup. It rejects undeclared selections,
`load_inplace`, tensorizer, 3D adapters and resumable requests. The runtime HTTP
update flag must remain disabled. A violated request contract stops EngineCore;
it is not a graceful per-request HTTP rejection.

For SGLang, OrbitKV reads normalized startup adapter references and effective
LoRA options from the official `runtime_context.get_lora()` configuration bag.
`ServerArgs` retains raw inputs, so `--lora-paths` can enable LoRA while its raw
`enable_lora` remains unset. Use the released loader as follows:

```bash
ORBITKV_STATIC_LORA=1 ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --lora-paths fixed=/path/to/immutable-adapter \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

The existing wrapper checks declared adapter UIDs before external lookup,
including complete HBM matches; the existing pending-admission Hook also checks
requests before computation. Caller `extra_key` is disallowed in this profile:
SGLang concatenates it with the adapter UID, so a base-model user key could
otherwise alias an adapter. Use `cache_salt` for request isolation. Radix-cache
sessions and streaming sessions are rejected because their retained pages can
bypass matching when adapters change. The pre-capture and admission Hooks remain
required; this does not close the public-lifecycle migration.

Keep adapter files, symlink targets and loaded bindings unchanged for the entire
engine lifetime. Startup loader configuration must match the declaration. Do
not call engine SDK/RPC add, remove, reload, in-memory tensor loading or weight
updates while this profile runs. Those calls can bypass cache callbacks; OrbitKV
does not intercept them or establish safe dynamic invalidation. Stop the engine,
change artifacts/configuration and restart to update the deployed adapter set.
Remote adapters, dynamic LoRA, live base-weight updates, resumable/session
requests and broader model/topology profiles remain unsupported or unqualified.

## LMCache and FlexKV reference

The [pinned integration audit](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/adapters.md#lmcache-and-flexkv-reference)
compares LMCache 0.5.5, official vLLM/SGLang entries and FlexKV `738ddc14`.
OrbitKV follows their small engine-entry and scheduler/worker separation while
keeping cache state in Rust and SGLang's native tree. Upstream compatibility
branches, P/D prerequisite patches and support matrices are not inherited claims.

## vLLM callbacks and current internal dependencies

`OrbitKVConnector` implements `KVConnectorBase_V1` and `SupportsHMA`.
`scheduler.py` discovers legal coverage; `worker.py` receives allocated pages,
registers tensors and consumes Restore/Save completion. Layout and metadata
modules describe groups/intents, not another scheduler.

The ordinary profile keeps **`VLLM_USE_V2_MODEL_RUNNER=0`**. Single-group
attention is unchanged; V1 Full + SWA follows the [HMA contract](#released-hma-contract).
The V2 gate explicitly sets `VLLM_USE_V2_MODEL_RUNNER=1` and
`kv_connector_extra_config: {"orbitkv.v2": true}`. It admits only one uniform
`FullAttentionSpec`, without zeroing, speculative decoding or native query-control
fan-out. It requires `kv_role=kv_both` or `kv_consumer`, so overlapping loads
receive the scheduler's deferred-free protection. Missing/MLA/hybrid layouts reject before connecting. This is implemented,
**unqualified**, and does not change V1 defaults. No runner replacement,
monkeypatch or parallel prefix tree is used in production.

The selected role alone is constructed. Native sessions close on initialization
failure, plugin conflicts stay visible, and `get_transfer_results` uses the
released completion/failure contract, including MultiConnector. Ordinary cache
publication has `requires_kv_delivery=False`; native reliable P/D retains its own
requirement. Saves submit at step end. Restore errors retain GPU ownership rather
than becoming a silent availability fallback.

### Released V2 lifetime contract

The following audit is pinned to vLLM **0.33.0, `4ba787ec`**:

- Registration: [V2 dispatch, lines 62–96](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/kv_connector.py#L62-L96)
  registers tensors once; `pre_forward` calls `handle_preemptions`, binds metadata,
  then starts synchronous loads. Independent async loads start after forward;
  OrbitKV's admitted restores use the synchronous path.
- Forward: [runner, lines 2050–2099](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L2050-L2099)
  invokes pre-forward before eager/piecewise execution or full-graph replay, then
  `finish_forward`. The [attention wrapper, lines 37–57](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/attention/kv_transfer_utils.py#L37-L57)
  calls `wait_for_layer_load`, attention, `save_kv_layer` on the caller's current
  stream. OrbitKV links full-graph loads on the replay stream and submits saves
  at step end, independent of replay executing Python layer callbacks.
- Completion: [post-forward, lines 136–175](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/kv_connector.py#L136-L175)
  waits/submits saves, collects finished sends/receives and failed receives through
  `KVConnectorTransferResults`, errors/stats/events/worker metadata, then clears
  metadata. Empty steps follow this path. OrbitKV waits for its Restore drain
  here; saves may remain asynchronous. Speculation's deferred finalization is rejected.
- Finish during Save: [the scheduler, lines 3031–3067](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L3031-L3067)
  honors `request_finished[_all_groups]` returning true until
  [`finished_sending`, lines 3556–3582](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L3556-L3582).
  OrbitKV reports that completion only after its own save receipt.
- Abort during Load: with overlapping batches and a consumer connector, the
  scheduler [defers block frees](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L189-L201)
  until the owning step is processed ([free/drain, lines 3077–3119](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L3077-L3119)).
  Without overlap, [EngineCore processes aborts after the worker result](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/engine/core.py#L677-L693).
  A failed/unknown OrbitKV Restore wait cannot return a successful step receipt.
- Preemption: [`_preempt_request`, lines 1799–1818](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L1799-L1818)
  can return IDs to the allocator before worker dispatch. The public
  `handle_preemptions` barrier must drain the old save **before new bytes are
  written**, even if the ID is already reassigned. CPU controls exercise that
  ordering with the actual allocator and V2 dispatcher; this is not a generation proof.

The scope restriction matters: V2 [updates requests](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1775-L1796)
before pre-forward; its [zeroing/CoW writes](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L1291-L1304)
would precede the barrier. Uniform Full attention has [no required zeroing](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1651-L1660),
single-group [hash boundaries equal block boundaries](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_utils.py#L840-L858),
and therefore no partial-tail CoW. Recurrent pre-copy remains outside this profile.
The current stream is the runner's [main stream](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu/model_runner.py#L602-L605);
its output-copy event is not an OrbitKV drain receipt.

[#62810](https://github.com/vllm-project/vllm/issues/62810) remains open (checked
November 9, 2026). No allocator generation is invented here. Forced reuse after
uncertain transfer, permanent worker/device loss, multi-group, Graph and P/D
qualification remain separate. See the [V2 fault gates](fault-qualification.md#v2-opt-in-gates).

### Released HMA contract

Audit at vLLM **0.33.0 / `4ba787ec`**, November 9, 2026. Admit **V1 Full + SWA**
only: exact Full/SlidingWindow specs, equal logical block sizes, no speculation,
zeroing or host-resident groups, and `kv_both`/`kv_consumer`. Unknown subclasses,
MLA multi-group, recurrent state and V2 multi-group fail before connecting.
PP is also rejected for this profile: [the scheduler copies one worker's group
names](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_utils.py#L1096-L1117),
while [PP projects layer membership per worker](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_planning.py#L1413-L1455).
A common resolved-group identity across those projections is not established.

| Boundary | Released contract and OrbitKV consumption |
| --- | --- |
| Hit selection | [`get_num_new_matched_tokens`, lines 516–548](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L516-L548) returns **one** prefix count, not one per group. The [scheduler, lines 1129–1187](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L1129-L1187) waits on `None`, otherwise allocates for the scalar hit. OrbitKV reports it only after every required group is leased at the same boundary; a missing group makes the entire external lookup miss. |
| Allocated destinations | [`update_state_after_alloc`, lines 569–592](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L569-L592) receives [`KVCacheBlocks.blocks[group][block]`, lines 35–107](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L35-L107). The scheduler [passes all groups](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L1482-L1485). OrbitKV rejects an allocation that changes its selected recovery boundary. |
| Step metadata | [`NewRequestData`, lines 40–48](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/output.py#L40-L48) and [`CachedRequestData`, lines 137–152](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/output.py#L137-L152) carry ordered per-group IDs. `build_connector_meta` translates these to OrbitKV load/save vectors; [scheduler-local block state](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/output.py#L214-L228) is cleared before worker dispatch. Worker hooks consume connector-owned metadata and registered layer tensors, not separate per-group hit counts. |
| Worker ordering | V1 calls [`handle_preemptions` before state updates](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_model_runner.py#L3934-L3945), then [bind/load → forward → wait_save → transfer results and worker metadata](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/kv_connector_model_runner_mixin.py#L67-L107). [Layer hooks](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/attention/kv_transfer_utils.py#L37-L57) wait before attention and call Save afterward on the forward stream. OrbitKV submits one Restore over all group leases and waits for its physical receipt before returning the step; Save remains asynchronous. |
| Finish and Save ownership | [`SupportsHMA.request_finished_all_groups`, lines 101–128](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L101-L128) receives all current tables and may delay request free. Window retirement can happen earlier, so OrbitKV uses the explicitly permitted [pool refcount binding, lines 490–503](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L490-L503). Every multi-group Save job pins its source blocks until all workers report drain through `OrbitKVWorkerMetadata`, independently of request finish. |

Full attention needs every page of the restored prefix. A window needs every
page intersecting its **W−1 past tokens at that same boundary**; vLLM includes the
next query token in W ([window manager, lines 1148–1182](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L1148-L1182)).
The shared Rust recovery contract intersects these requirements before publishing
a hit. The scheduler's loaded-group default includes its prefix-cacheable groups
([base, lines 505–513](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L505-L513));
OrbitKV does not use it to advertise partial-group success.

Window retirement [nulls table entries and drops their allocator refs](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L655-L720).
The [pool returns only zero-ref blocks to its free queue](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/block_pool.py#L792-L847),
so a Save pin prevents reassignment. During Load, both [allocation](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L555-L569)
and [finish](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L3367-L3375)
retire windows using processed tokens, subtracting in-flight tokens. The
consumer's deferred-free fence retains the remaining destinations on abort;
[scheduling refuses to preempt an in-flight victim](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L924-L942).
Timeout is not a step or all-worker drain receipt.

**Mamba stays rejected.** The release does expose exact checkpoint offers:
[`MambaSpec`, lines 1094–1186](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1094-L1186)
defines position-indexed state/scratch blocks and legal boundaries;
[`take_boundary_state_offloads`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L880-L900)
offers `(group, block, boundary)` states. Recovery needs the state **exactly at t**,
not an earlier prefix. But [Mamba requires zeroing](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/kv_cache_interface.py#L1651-L1660),
V1 [zeros/CoW-copies in state updates](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_model_runner.py#L1203-L1212)
and [preprocesses Mamba state before connector load](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/worker/gpu_model_runner.py#L4049-L4068).
Its [old state copies retire separately](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L1610-L1657).
OrbitKV has not proven a handoff that supplies the exact loaded state before
those copies or retains every such owner. Generic checkpoint fixtures do not
admit recurrent serving; no private hook is added to bridge this gap.

Resolved group order, layer membership and all spec fields now enter multi-group
namespace identity. Single-group identities are unchanged. Installed-CPU
[controls](../python/tests/integration/test_vllm_hma_contract.py) cover atomic
miss/hit, boundary mismatch, identity, named rejections and allocator retirement.
They control CUDA events/Manager replies, not physical DMA. [GPU gates](fault-qualification.md#fullswa-gates)
remain unrun and unqualified. #62810 remains open on November 9, 2026; references
and barriers are not allocator-generation proof.

## SGLang linker and current internal dependencies

Enable `--enable-unified-cache-external-linker --radix-cache-backend orbitkv`.
`plugin.py` constructs UnifiedRadixCache; the linker handles lookup, queued load,
layer progress, offload, reset and close. SGLang owns tree locks and page allocation.
Plugin discovery does not eagerly load native/GPU resources.

SGLang 0.5.23 owns private destination allocation, cancellation and tree insertion.
`RecoveryLinkerWrapper` translates absolute boundaries and rank agreement;
`finish_layer_wise_loading` waits for native Restore before insertion or release.
Unknown completion raises and retains ownership. OrbitKV admits Full/SWA and
static Mamba conv/SSM checkpoints through the released component. Aborts use native
`BasePrefixCache.finish(ABORT)`; lookup-only cancellation retires unsubmitted queries.

Two private Hooks remain: graph initialization and pending-query admission. An
extra enqueue Hook is enabled only for `ORBITKV_PREPARE_REQUESTS=1` or
`ORBITKV_QUEUE_WARMUP=1`. Replace a Hook only after consuming a released replacement.
A pending lookup stays queued while other requests can progress; required ranks
agree before admission, using scalar decisions for one participant. Physical
multi-rank serving still needs qualification.

`events.py` owns pre-capture CUDA events and producer/consumer counters. Native
completion, not task cancellation, releases destinations. The
[original shutdown record](https://github.com/OrbitInfra/OrbitKV/blob/1d173c1a27401f6e219c5120f97ba5eab3e2d52b/docs/adapters.md#sglang-linker-and-current-internal-dependencies)
preserves warnings and 0.5.21 measurements. Unknown auxiliary/draft state and
unsupported request modes remain rejected.

### Released SGLang component contract

Audited at **v0.5.23, `297e54fd`**, November 10, 2026. These are source and
installed-CPU contracts, **not GPU qualification**. No engine tree, runner or
allocator is replaced.

| Component | Released handoff and ownership | OrbitKV decision |
| --- | --- | --- |
| Full / SWA | The wrapper [requires every pool at one boundary](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L165-L198), intersects rank boundaries, then [allocates private destinations and unwinds refused loads](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L392-L497). Full needs the prefix; SWA needs the trailing window at that boundary. | Existing behavior unchanged; no partial component hit. |
| Mamba / SSM | [OFFLOAD names one checkpoint at the node's final hash; LOAD allocates one slot](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/components/mamba.py#L729-L759). Every conv tensor and nonempty temporal tensor belongs to that slot. [PREPARE adopts it and removes obsolete deferred CoW/clear; ABORT frees only unsubmitted allocations](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/components/mamba.py#L761-L790). | Admit static `HybridReqToTokenPool` / exact `MambaPool.State` with Full attention. Combined Full/SWA/Mamba serving remains rejected pending a combined lifecycle gate. Register all conv/temporal buffers; restore the checkpoint at exactly the selected boundary, never a nearby state. |
| Checkpoint production | [Finished requests donate an exactly tracked checkpoint; running requests donate a tracked slot or copy into a distinct slot](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/components/mamba.py#L582-L727). The [tree floors every component to one common boundary](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_radix_cache.py#L1164-L1238). [Mamba locks protect checkpoints from eviction](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/components/mamba.py#L465-L553). | Keep those engine owners. OrbitKV's Save completion is emitted only after native drain; request finish is not the Save receipt. |
| MLA | [The ordinary MLA pool](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/memory_pool.py#L4754-L4866) holds latent KV plus rotary dimensions in each row. The Full component hands off all these rows; the ordinary wrapper owns them exactly as Full KV. | Retain per-rank storage. Identity now records latent/RoPE/physical widths separately, plus tensor shape, dtype, stride, layer map and TP/PP/CP coordinates. Equal total widths with different partitions cannot alias. Separate scale/indexer buffers remain rejected. |
| MLA writer selection | [`HiCacheStorageConfig.is_storage_writer`](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/hicache_storage.py#L28-L74) selects TP rank 0, or the first replica of each DCP shard. The native linker [accepts follower offloads without sending bytes](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py#L376-L423); the [tree reduces completion count and success across ranks](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_radix_cache.py#L3651-L3667). | A released writer rule exists. M3 still owns shared-entry identity and rank-common bounded admission/failure handling. This PR retains rank-isolated entries; two ranks cannot write one entry. `--enable-linker-mla-dedup` (native rank-0 Load/broadcast) is explicitly rejected: it is not an OrbiTE byte path. |
| Other state | The [released component set is Full/SWA/Mamba](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L59-L65). [Mamba state also has optional replay/speculative fields](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/memory_pool.py#L521-L568); [the native assembler rejects moving, int8 and side-state layouts](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py#L488-L531). | Reject moving/unified pools, int8 checkpoints, ReplaySSM, draft/speculative buffers, short-conv/ngram siblings and unknown pool types. Existing DSA/sidecar/draft-window/request-ring/scale-buffer rejections stay. Convolution-only checkpoints use the same exact-boundary rule. |

The linker is a queue before `start_layer_wise_loading`. A queued cancellation
releases only unsubmitted leases; PREPARE's private slots remain engine-owned.
For a started load, [per-layer pool getters wait for the load counter](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/memory_pool.py#L1667-L1726).
[`Scheduler.run_batch` waits for `finish_external_linker_loads` before returning](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler.py#L4926-L4935);
the [wrapper waits for the native receipt before releasing private-load bookkeeping](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L564-L620).
Thus the next decode step's [retraction/free](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_batch.py#L2360-L2395)
cannot overtake an in-flight load. CPU controls execute the released `run_batch`
and `release_req` with controlled model compute and native receipts. They do not
call retraction illegally in the middle of a forward or claim allocator generations.

[Offload locks and FIFO completion](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py#L501-L562)
are independent of request lifetime. An abort or retraction cannot free a locked
checkpoint or attention source; unknown completion retains locks and registrations.
All-group leases are acquired before reporting an external hit and all required
bytes enter one native Restore. A missing component or mismatched boundary is a
miss, not a mixed recovery. [Controls and unrun hardware gates](fault-qualification.md#sglang-component-gates)
separate logical decisions, fixture bytes and model execution.

## Remaining internal Hooks

The public Hook registry does not make string targets stable APIs. Keep each
correctness protection until a released replacement is actually consumed.

| Target | Purpose | Removal gate |
| --- | --- | --- |
| `TpModelWorker.init_cuda_graphs` BEFORE | Install stable restore events before Graph capture; cache factory runs later | Released worker preparation callback plus actual eager/Graph restore and page-reuse gates |
| `PrefillAdder.add_one_req` AROUND | Defer pending SSD/peer lookup and preserve rank agreement | Released asynchronous lookup/admission contract plus miss/hit/cancel/rank gates |
| `Scheduler._add_request_to_queue` AROUND, opt-in | Optional queued preparation/warming | Released enqueue callback or remove optimization with matched workloads |

Plugin discovery stays lazy and opens no native/GPU resources for an unselected backend.

Missing allocator-generation contracts are tracked in
[vLLM #62810](https://github.com/vllm-project/vllm/issues/62810) and
[SGLang #45493](https://github.com/sgl-project/sglang/issues/45493).
Worker preparation [#44856](https://github.com/sgl-project/sglang/pull/44856) and
pending admission [#44873](https://github.com/sgl-project/sglang/pull/44873) are
upstream replacement candidates, not consumed interfaces in our current pins.
The three SGLang items remain open and unmerged on November 10, 2026; none
is in v0.5.23. vLLM #62810 was still open at its November 9 audit. Neither
allocator exposes the requested physical-page generation. Submission alone never removes
a correctness hook. See the [upstream path](completion-plan.md#upstream-path).

## P/D and cache composition

Manager-to-Manager cache sharing uses OrbiTE. Live P/D uses vLLM
NIXL/MultiConnector or SGLang native disaggregation. The composition candidate
reads/writes cache on P and saves only on D, preserving one incoming D writer.
Cache leases drain independently; ordinary cache Save remains best effort.
See [native P/D recipes and limits](pd.md). Removed fork callbacks and passed
fork results do not qualify official-engine fault behavior.

## Native client

```python
from orbitkv import CacheManagerClient

client = CacheManagerClient("/tmp/orbitkv-50055.sock")
ok, message = client.health()
client.close()
```

| API | Contract |
| --- | --- |
| `peer_alive()` | Original authenticated pidfd; no RPC/reconnect and no drain proof |
| `BlockHashes(...)` | Immutable native hash batch; slices share allocation |
| `query_prefetch` | Native submission/revision/polling ownership |
| `register_context_batch(..., tensors=...)` | Retains real exporters and IPC registration |
| `start_restore(..., ready_stream=...)` | Client-bound operation with actual destination readiness |
| `wait_restore_enqueued` | Fresh layer-event publication, not completion |
| `poll_restore` / `wait_restore` | Final local result after physical drain; source reaping is separate |
| `prepare_save` / `submit_saves` | Capture registration at producer handoff; native batching and retained completion |

`SaveNotSubmittedError` proves that invocation launched no worker, or the complete
batch rolled back. Resolve native methods and arguments before catching it:
Python lookup/attachment/wait errors cannot prove native rollback. SGLang retains
later offloads behind an unknown earlier FIFO result. A different client or
retired registration cannot bind queued Save to a replacement; vLLM rejects
batches spanning registrations.

Publish diagnostics can observe active/retained calls without blocking behind
lifecycle exchange. Overdue/missing/dead-peer observations never complete pages.
Native calls release the GIL. Repeated live registration is rejected; close and
unregister fence admission and drain. A timeout/drop cannot release a page while
copies remain possible. See [native signatures](https://github.com/OrbitInfra/OrbitKV/blob/main/python/orbitkv/orbitkv.pyi)
and [layer consumption](engine-local-restore.md#layer-readiness-and-framework-consumption).

## Process channel

Every inference process requires a same-host Manager. Scheduler Query/Release
and worker Publish/Restore use iceoryx2; registration, health and cleanup use
authenticated UDS. The default path is `/tmp/orbitkv-<orbitkv.port>.sock`;
`orbitkv.bootstrap_socket` overrides it. Standalone mode starts no gRPC listener.

`orbitkv.timeout_ms` defaults to 5,000 for hot requests/health; registration and
unregister allow at least 120 seconds for CUDA setup/drain. `spin_iterations`
defaults to 64. Matching bootstrap **7**, ABI **12**, schema **10** and lifecycle
**13** are required; upgrade clients and Manager together.

Client-local CUDA ordinals resolve through full GPU UUIDs to Manager-visible
devices. Native owners retain that route and NUMA placement. Ordinal equality
is unnecessary; broader multi-GPU deployment remains unqualified.

Raw Restore uses at most 1 MiB plan parts, 32 MiB operation metadata and 64 MiB
prepared metadata per session. Pending queries allow 128 per session / 1,024
globally with a 60-second reply lifetime. `wait_for_full_prefix` polls the same
instance/request/group operation; input changes require a revision, superseded
queries are cancelled, and submitted reads drain despite cancellation/expiry.
See [transport ownership](transport.md) for the complete protocol.

## Connector Modes

`read_write` (default) performs external lookup, Restore and Save. `save_only`
does not query/load and is intended for D-side composition where another
connector owns incoming pages. Set `orbitkv.mode` in `kv_connector_extra_config`;
these are the only values. The [P/D guide](pd.md#vllm-nixl-and-multiconnector)
contains the complete MultiConnector configuration.

## TP shards and host boundary

Workers register with their same-host Manager; the scheduler uses its local UDS.
Official vLLM handshake metadata carries opaque sealed query targets. Rust
validates topology, queries bounded targets and finds the common prefix; Python
never queries foreign Managers or chooses it.

Enable `--enable-query-control` on each Manager and `orbitkv.query_control=true`
for one Manager. Multiple `orbitkv.tp_shard_endpoints` enable native control
automatically. Example connector extras:

```json
{
  "orbitkv.query_control": true,
  "orbitkv.tp_shard_endpoints": ["http://127.0.0.1:50055", "http://127.0.0.1:50056"]
}
```

Every worker gets the same ordered endpoint list; custom UDS paths use
`orbitkv.tp_shard_bootstrap_sockets`. Exactly all declared global ranks must join,
and workers on one node export identical registration. Equal contiguous dense
TP-only shards, V1 and one attention group are required. PP/DP/EP/DCP/PCP, MLA
replication, recurrent state, speculative decode, full-prefix waiting and queue
preparation are rejected. Ordinary single-Manager mode remains the default.

Initial prefix queries run in parallel; longer results are cancelled before
reacquiring the exact common prefix. Every selected source lease is claimed and
retained until all declared ranks report successful native completion. Duplicate
ranks count once. Revision/cancel/connection loss stays bounded; TTL is not
process-death or GPU reclamation evidence.

Only trusted same-host literal-loopback endpoints are accepted. Cross-host
activation needs authenticated encryption and physical topology/fault gates.
The [historical `1ab67a58` installed TP=1 evidence](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/adapters.md#tp-shards-and-host-boundary)
does not qualify physical TP8, NCCL, HA or the current OrbiTE path.

## Complete cache blocks and live prompt tails

Cache stores complete engine blocks. Native P/D transfers live prompt tails;
there is no cache tail mode. `wait_for_full_prefix` is a cache query option, not
live P/D coordination or a watch of local saves.

## Package responsibilities and qualification

| Module | Responsibility |
| --- | --- |
| `identity.py` | Artifact/computation fingerprinting |
| `client/` | Connection configuration and registration handoff |
| `vllm/connector.py`, `scheduler.py`, `worker.py` | Released callbacks and scheduler/worker ownership |
| `vllm/layout.py`, `metadata.py` | Groups and intents |
| `sglang/layout.py`, `recovery.py` | Geometry and recovery handoff |
| `sglang/events.py` | Capture events and forward dependencies |
| `sglang/linker.py`, `plugin.py` | External linker/backend registration |
| Rust state/channel/Core | Shared semantics, native client and physical lifetime |

Keep source-only tests separate from native/engine gates. Installed qualification
checks actual bytes and drain as well as output: disabled startup, native HBM,
cold/partial/full hits, DRAM/SSD/peer, cancel/preemption/restart and each advertised
Graph/rank/container profile. See [test gates](https://github.com/OrbitInfra/OrbitKV/blob/main/python/tests/README.md).

### Registered TP query-control transport

The disabled-by-default Manager option uses `--addr` without needing etcd.
Bound/advertised addresses must be literal loopback with nonzero ports, checked
before CUDA. This is not local multi-user authentication or encrypted forwarding.
`export_query_target()` transfers sealed authority through authenticated UDS;
network interest cannot register/unregister GPU mappings.

Claim retains the same result across reply loss. Close/expiry releases unconsumed
shares while accepted native work remains with its completion owner. Each
interest permits 128 lifetime operation identities, including terminal replay
fences. Open one interest per lookup and close it after handoff/completion; Claim
does not reset admission. Cross-host and physical multi-rank qualification stays
in the [completion plan](completion-plan.md#milestones).
