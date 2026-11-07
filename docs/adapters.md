# vLLM and SGLang integration

Use official **vLLM 0.31.0** and **SGLang 0.5.22** with the
[single-node quickstart](single-node.md). OrbiTE is the sole cache transfer engine;
GPU/RDMA/GDS and installed profiles need fresh hardware qualification. Historical
[ordinary, Graph, shared-Manager and two-host acceptances](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/adapters.md)
retain their original releases/backend and do not carry over. Current limits are
in [compatibility](compatibility.md), status in the [completion plan](completion-plan.md).

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
The [upstream interface record](engine-release-audit.md#upstream-work-and-independent-limits)
tracks proposals separately from released consumption. Existing
[normal-path lifetime evidence](https://github.com/OrbitInfra/OrbitKV/blob/1d173c1a27401f6e219c5120f97ba5eab3e2d52b/docs/adapters.md#ownership-contract)
does not establish cancellation, physical termination or eventual fault reclamation.

## Explicit cache reset

Official vLLM 0.31.0 distinguishes an HBM-only prefix reset from
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
dense TP=1/PP=1 evidence in the [completion plan](completion-plan.md#s53--public-lifecycle-and-hybrid-state-contracts).
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

The profile requires **`VLLM_USE_V2_MODEL_RUNNER=0` and one attention group**.
V1's public preemption callback precedes page changes. V2 and recurrent/multi-group
serving reject before connecting. No runner replacement or native-prefix override
is installed. Framework-neutral recovery validation remains in Rust.

The selected role alone is constructed. Native sessions close on initialization
failure, plugin conflicts stay visible, and `get_transfer_results` uses the
released completion/failure contract, including MultiConnector. Ordinary cache
publication has `requires_kv_delivery=False`; native reliable P/D retains its own
requirement. Saves submit at step end. Restore errors retain GPU ownership rather
than becoming a silent availability fallback.

## SGLang linker and current internal dependencies

Enable `--enable-unified-cache-external-linker --radix-cache-backend orbitkv`.
`plugin.py` constructs UnifiedRadixCache; the linker handles lookup, queued load,
layer progress, offload, reset and close. SGLang owns tree locks and page allocation.
Plugin discovery does not eagerly load native/GPU resources.

SGLang 0.5.22 owns private destination allocation, cancellation and tree insertion.
`RecoveryLinkerWrapper` translates absolute boundaries and rank agreement;
`finish_layer_wise_loading` waits for native Restore before insertion or release.
Unknown completion raises and retains ownership. Full/SWA are the released
external-linker components; recurrent/SSM startup is rejected. Aborts use native
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
copies remain possible. See [native signatures](../python/orbitkv/orbitkv.pyi)
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
Graph/rank/container profile. See [test gates](../python/tests/README.md).

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
in the [completion plan](completion-plan.md#s55--deployment-matrix-and-upstream-maintenance).
