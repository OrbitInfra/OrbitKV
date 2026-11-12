# OrbitKV completion plan

**Goal: a standard, complete distributed KV cache for vLLM and SGLang, accepted
into both upstream mainlines.** OrbiTE is the only byte mover. Engines own
scheduling, HBM and native P/D; OrbitKV owns cache identity, storage, discovery,
grants and resource lifetime. This is the only execution queue.

The current delivery is **M2 PR 2: read-time integrity and persistence recovery bounds**.
Each coherent PR stops for review before dependent work starts. Research designs
and simulator campaigns are removed from the maintained tree; Git history retains
them and external artifacts are untouched. DeepInfer/OrbitInfer and an OrbiTE
NIXL plugin are outside this plan.

Current pins are **vLLM 0.33.0 / SGLang 0.5.23**, verified against final tags and
published package metadata on November 9, 2026. The vLLM release page still names
rc3, but the final tag and package exist; exact revisions and the consumed API
audit are in [adapters](adapters.md#release-upgrade-audit). This updates source
contracts only: all installed GPU profiles for these pins remain unqualified.

## Upstream path

Follow the existing thin, in-tree integrations that import an external package:

- **vLLM:** the [V1 connector directory](https://github.com/vllm-project/vllm/tree/main/vllm/distributed/kv_transfer/kv_connector/v1)
  includes FlexKV, LMCache/LMCache MP, Mooncake, HF3FS and UMBP.
  [FlexKV's entry](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py)
  delegates to its installed package. Propose the same small entry for OrbitKV,
  with upstream tests and documentation. The directory name `v1` does not
  establish V2 model-runner compatibility: consume the released
  [connector lifecycle and HMA contracts](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
  and prove each runner/group combination.
- **SGLang:** the [storage integrations](https://github.com/sgl-project/sglang/tree/v0.5.23/python/sglang/srt/mem_cache/storage)
  include FlexKV, LMCache, Mooncake Store, HF3FS and NIXL, but not every integration
  there implements the same interface. [HiCacheStorage](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/hicache_storage.py)
  serves HiCache's host pools. OrbitKV should use the released
  **[UnifiedCacheLinker](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py)**:
  it connects device pools directly to an external cache, matching our existing
  linker and independent DRAM/SSD ownership. This avoids another mandatory host
  tier and keeps OrbitKV movement in OrbiTE. Follow the lazy imports and
  direct-linker registrations in [the factory](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/unified_cache/linker_factory.py),
  including MooncakeDirectLinker and UMBPDirectLinker. Version 0.5.23 declares
  Full/SWA/Mamba components; consuming that API is not proof of
  OrbitKV recurrent recovery or fault safety.

Keep shared state machines in Rust and Python limited to engine callbacks,
layouts and GPU handoff. Replace private hooks only when a released replacement
is consumed. Recheck missing contracts during M1, including allocator generations
([vLLM #62810](https://github.com/vllm-project/vllm/issues/62810),
[SGLang #45493](https://github.com/sgl-project/sglang/issues/45493)). Do not solve an
upstream gap with a private runner or parallel cache tree. Submission, merge,
release and local consumption are separate states; acceptance into both mainlines
is the final integration goal.

## Standard and complete checklist

**Have** means implemented within the stated scope, not hardware-qualified.
**Partial** means a working subset with a missing contract or gate.
**Missing** means the required product behavior is absent. Comparator references
are precedents, not claims that every product supports every feature or that
its semantics and performance transfer to OrbitKV.

References: **L** = [LMCache storage and operations][L], [legacy layerwise transfer][LL]
and [coordination][LC]; **F** = [FlexKV tiers/integration][F] and [distributed reuse][FD];
**V** = [vLLM native offload][V]; **M** = [Mooncake Store replicas/leases/eviction][M].

### Engine integration and models

| Capability | OrbitKV status and evidence | Precedent |
| --- | --- | --- |
| Latest vLLM, V2 runner, HMA multi-group/hybrid | **Partial:** 0.33.0 V1 single-group and Full + SWA; V2 opt-in remains one uniform Full group. Atomic recovery, Save pins and identity have CPU controls; GPU profiles are unqualified. Recurrent, zeroing and V2 multi-group reject. [Released HMA contract](adapters.md#released-hma-contract). | L/F/M thin entries; released V lifecycle |
| Latest SGLang | **Partial:** 0.5.23 Full/SWA and static Mamba checkpoint linker; MLA stays per-rank; pre-capture and pending-admission hooks remain private. [Contract](adapters.md#sglang-linker-and-current-internal-dependencies). | Released linker and HiCache integrations above |
| Async load | **Have:** pending queries, bounded admission and completion-owned destinations; installed faults remain unqualified. [Lifecycle](vllm-request-state-machine.md). | L/F/V asynchronous offload |
| Layerwise load/save | **Partial:** eligible raw Restore has layer readiness; other routes may wait for whole completion. vLLM saves at step end; `save_kv_layer` does not submit. [Restore](engine-local-restore.md#layer-readiness-and-framework-consumption), [connector](https://github.com/OrbitInfra/OrbitKV/blob/main/python/orbitkv/vllm/connector.py). | L layerwise transfer, V layer hooks |
| Official P/D composition | **Partial:** NIXL/MultiConnector and SGLang native paths are candidates; full faults unqualified. [P/D](pd.md). | V MultiConnector, L/F native integration |
| Dense and hybrid models | **Partial:** dense layouts and declared Prefix/Window/Checkpoint validation exist; complete SWA/MLA/recurrent recovery is not qualified. Unsupported layouts are rejected. [Models](models.md), [recovery](hybrid-recovery.md). | V hybrid offload, L recipes, SGLang components |
| MLA TP deduplication | **Missing:** ordinary TP sharding is not replica deduplication. [TP boundary](adapters.md#tp-shards-and-host-boundary). | SGLang `HiCacheStorageConfig.is_storage_writer` selects MLA writers |

### Storage tiers

| Capability | OrbitKV status and evidence | Precedent |
| --- | --- | --- |
| Host DRAM | **Have:** bounded pinned pools, replicas and leased restores. [Policies](cache-policies.md). | L/F/V host tiers, M memory objects |
| SSD surviving Manager restart | **Partial:** opt-in [durable journal and checksum-gated recovery](storage-persistence.md); default still truncates. Tail recovery preserves a verified prefix; capacity-derived metadata admission stays bounded. CPU process-crash controls run on ext4/tmpfs; installed NVMe serving and power-loss behavior remain unqualified. | L/F disk and V filesystem tiers; persistence needs a separate crash contract |
| Peer DRAM/SSD cluster tier | **Partial:** inventory discovery, source-owned SSD preparation and OrbiTE reads exist; durable cluster recovery/node-loss qualification remain open. [Distributed cache](distributed-cache.md). | F distributed reuse, L P2P, M replicas |
| Shared filesystem / 3FS mount | **Missing:** a local SSD path on a shared mount supplies no shared metadata, locking, GC or safe restart. | L filesystem/HF3FS, V secondary filesystem, F remote storage |

### Distributed behavior

| Capability | OrbitKV status and evidence | Precedent |
| --- | --- | --- |
| Membership | **Have:** etcd leases, persistent epochs and Watch fencing. [Membership](distributed-cache.md#local-global-index-and-etcd-metadata). | F/M coordination |
| Bounded-stale placement directory | **Partial:** covered requester-local inventories and authoritative grants; no external directory API or qualified visibility bound under loss/pressure. [Coverage](distributed-cache.md#owner-inventory-streams). | F metadata service, L coordinator, M master |
| Router lookup / KV events | **Missing:** diagnostics are not a router API. Target [llm-d's event index](https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/kv-management/kv-indexer.md) and [Dynamo's selection interface](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/standalone-selection). | L coordination and V KV events; no new inference router |
| Replica selection | **Partial:** compatible candidates and opt-in cost observations; complete failure-aware selection remains open. [Planning owners](architecture.md#planning-and-cost-observations). | L/F/M replica lookup |
| Node loss | **Partial:** retired epochs are fenced and uncertain owners retained; permanent-loss reclamation/cache HA unqualified. [Retirement](transport.md#peer-departure). | F distributed reuse, M leases/replicas |
| LRU / TTL / watermarks | **Partial:** LRU/protected retention and SSD admission; data TTL and coordinated high/low watermarks missing. Membership TTL is not data TTL. [Policies](cache-policies.md). | L/F LRU, V LRU/ARC, M watermarks/soft pinning |
| Exact capacity accounting | **Partial:** local pool/index/query/registration/file and persistence metadata budgets include retained work; shared-tier and fleet reconciliation remain missing. [Budgets](server.md), [persistence](storage-persistence.md#bounds-gc-and-changed-capacity). | L/F/V tier limits, M replica accounting |
| Tenant namespaces | **Partial:** computation scope and inventory filtering exist; neither authenticates tenants nor provides quotas. [Identity](state-identity.md), [scope](distributed-cache.md#scope-and-bounded-resources). | L/F namespace and management controls |

### Correctness and operations

| Capability | OrbitKV status and evidence | Precedent |
| --- | --- | --- |
| Full identity | **Have** for admitted layouts: immutable artifacts, computation, representation and recovery identity; unknown auxiliary state rejected. [Identity](state-identity.md). | L/F/V cache keys; no cross-engine byte-equivalence assumption |
| Read-time checksums | **Partial:** raw/encoded SSD and peer receives have [tier-crossing integrity](storage-formats.md#read-time-integrity); sealed immutable DRAM trusts ECC without hit scans. CPU and CUDA-shim controls do not qualify GPU/RDMA/GDS serving. [CPU validation](https://github.com/OrbitInfra/OrbitKV/blob/main/crates/orbitkv-core/src/codec/mod.rs), [GPU decode](https://github.com/OrbitInfra/OrbitKV/blob/main/crates/orbitkv-core/src/codec/gpu.rs). | Tier integrity is a requirement to verify, not inferred from advertised disk support |
| Abort/preemption/restart safety | **Partial:** conservative ownership and CPU faults exist; current installed faults, allocation reuse and permanent loss unqualified. [Fault gates](fault-qualification.md). | Released engine lifecycle; comparator smoke tests do not prove ours |
| Per-tier metrics / miss explanations | **Partial:** RAM/SSD/remote attribution and diagnostics; stable user-facing miss reasons and all-tier coverage incomplete. [Metrics](metrics.md). | L observability, F/V tier stats, M monitoring |
| Configuration / deployment recipes | **Have** for current pins and scoped topologies; recipes are not qualification. [Quickstart](single-node.md), [deployment](deployment.md), [Manager](server.md). | L/F/V/M operating guides |
| Authenticated endpoints | **Partial:** local bootstrap credentials; grants authorize allocations, but peer gRPC/HTTP lack a complete authenticated deployment contract. Restrict network access. [Channel](transport.md#host-and-lifecycle-boundary), [deployment](deployment.md). | Production security requirement; cache leases are not authentication |
| No pickle | **Have:** bounded typed native/protobuf/JSON boundaries; no pickle in the adapter or Manager path. [Process contract](transport.md#process-channel). | Thin typed integration, not serialized executable objects |

## Milestones

### M1 — Engine baseline

Reviewed in order, one PR at a time:

1. **Pins and consumed APIs (reviewed, #337):** vLLM 0.33.0 / SGLang 0.5.23;
   keep V1/one attention group and Full/SWA, with CPU API-contract controls.
2. **Real-engine conformance/fault harness (reviewed, #338):** explicit source,
   installed-CPU and installed-GPU registry; required runs reject missing
   prerequisites, empty selections, stubs and unreviewed skips. Run both released
   boundaries in the Python 3.12 wheel leg. See the
   [case matrix](fault-qualification.md#engine-conformance-matrix).
3. **vLLM V2 runner (reviewed, #339):** opt-in uniform Full attention, one group;
   [released ordering](adapters.md#released-v2-lifetime-contract) and installed-CPU
   allocator/completion controls. The early zeroing/CoW and speculative-finalize
   paths remain excluded. GPU byte/model/fault gates are written, unrun and
   unqualified; physical page generations remain an upstream gap.
4. **HMA multi-group/hybrid (reviewed, #341):** V1 Full + SWA,
   one atomic recovery boundary, resolved-group identity and pinned Save jobs.
   [Released contract](adapters.md#released-hma-contract) and installed-CPU
   allocator controls cover window retirement, abort and preemption. Recurrent,
   zeroing and V2 multi-group profiles remain rejected; GPU gates are unrun.
5. **SGLang 0.5.23 components (reviewed, #342):** static Mamba
   conv/SSM checkpoints use the released component, an atomic boundary and native
   drain; MLA remains per-rank with explicit latent/RoPE identity. Installed-CPU
   controls cover real pool allocation, tree locks, scheduler completion and
   retraction. [Named rejections and audit](adapters.md#released-sglang-component-contract)
   delimit the scope. Worker-preparation/pending-admission Hooks remain required;
   #44856, #44873 and #45493 are open on November 10, 2026.

M1's code items are complete for the admitted profiles. Only
its installed-GPU queue remains; unsupported layouts, MLA TP deduplication (M3)
and upstream Hook replacement are not silently promoted into this scope.

Cover abort, queue wait, preemption, engine/Manager restart, identity
collisions, partial load/save and generation reuse. Required gates fail on missing
prerequisites or unexpected skips. Until M2, Manager restart may safely miss;
it must never return stale state as a hit.

**CPU now:** source-only adapter tests, state/recovery contracts, protocol faults,
etcd inventory controls, CPU/TCP/file tests and CUDA-shim byte controls. Installed
0.33.0/0.5.23 controls additionally exercise official request, connector, linker
and completion classes with controlled CUDA events and Manager replies. They
prove adapter decisions, not physical drain, real process restart or page reuse.
**Hardware later:** model outputs/restored bytes for every claimed runner/group/rank,
allocator reuse, cancellation/preemption, Graph and native P/D composition.
CPU mocks and shim passes cannot satisfy these gates.

#### M1 installed verification queue

All entries are **unrun and unqualified** on the current OrbiTE path. Exact
commands and the distinction between fixture bytes and serving are in
[fault qualification](fault-qualification.md#required-installed-gpu-runs).

| Gate | Scope / acceptance |
| --- | --- |
| V1 single-group and SGLang Full/SWA recovery | Installed model outputs, restart hits, actual restored bytes and untouched destinations; DRAM and io_uring separately. |
| V1 Full + SWA | [Exact commands](fault-qualification.md#fullswa-gates): immutable Gemma 3 Full/SWA model outputs and restart hits, per-group fixture bytes, window retirement during Save and abort/preemption during Load. All unrun; recurrent hybrids remain rejected. |
| SGLang components | [Exact commands](fault-qualification.md#sglang-component-gates): Qwen3-Next Full+checkpoint and DeepSeek-V2-Lite MLA model output/restart; each conv/SSM/latent buffer, untouched slots, abort during Save/Load and retraction after drain on DRAM/io_uring. Fixture fault controls do not establish end-to-end model fault qualification. |
| V2 opt-in, uniform Full attention | [Exact commands](fault-qualification.md#v2-opt-in-gates): model outputs/restart restore metrics plus native fixture bytes for save preemption, abort during load, finish before save drain and reallocation. The fixture invokes released scheduler/dispatcher classes; it does not substitute for an end-to-end model fault run or allocator-generation proof. |
| Abort, timeout, cancellation, partial load/save | Native fault barriers plus released adapter controls; no early source/destination/registration/grant release. A normal model-output run alone cannot close these cases. |
| Engine-worker / Manager restart | Live-Manager engine restart restores bytes; Default Manager restart starts cold; opt-in SSD recovery is covered by the M2 queue. Both reject old handles/leases and never report stale state as a hit. |
| Preemption and allocator page reuse | Save holds and registration-replacement controls exist; allocator-slot generation proof remains blocked on vLLM #62810 / SGLang #45493 (open November 9, 2026). No private runner or parallel tree substitutes for the missing contract. |
| Identity collision | Same-token model revision, layout, computation, representation and scope controls run on CPU; installed model/layout cases must also be tested on the admitted profile. |

### M2 — A complete node

1. **Persistent SSD (implemented):** opt-in journal, durable
   overwrite fences, bounded recovery/GC and full-extent checksum verification
   before the first recovered hit. Default behavior stays cold. See the
   [protocol and crash controls](storage-persistence.md).
2. **Read-time integrity (current review):** all stored representations, exact-generation
   invalidation, torn-tail prefix recovery and capacity-derived metadata limits.
   Recovery verification remains in addition to every-read checks.
3. **Layerwise load/save.**
4. **TTL/watermark eviction.**

Reuse storage/lease owners and OrbiTE file I/O. Document
commit/recovery ordering, orphan reclamation and corrupt-data misses. Never reuse
bytes still held by a reader or undrained transfer.

**CPU now:** real io_uring files on ext4, crash/reopen boundaries, short reads,
corruption, orphan GC, disk-full behavior and concurrent leases are CI-qualifiable.
The required child-Manager SIGKILL gate covers commit boundaries, metadata/data
corruption, wrap, retained readers, shrink, checkpoint replacement and ENOSPC.
CPU host registration is a shim; OrbiTE and filesystem I/O are real.
**Hardware later:** installed recovery across Manager restart, layer readiness
under compute, raw/encoded GPU bytes and matched DRAM/SSD serving. The
[NVMe Manager-restart gate](fault-qualification.md#persistent-ssd-recovery) for both
engines is **written, unrun and unqualified**; no power-loss claim follows from
SIGKILL or clean restart.

### M3 — A complete cluster

Separate PRs for bounded-stale placement discovery and router lookup/events;
a shared filesystem/3FS tier; node-loss recovery; MLA TP deduplication and hybrid
coverage allowed by engines. Extend existing inventory/source owners. Define
staleness, missed-event resync, replica fallback, tenant isolation, authentication
and exact capacity accounting before exposing APIs. An index is a hint; only a
current grant authorizes bytes.

**CPU now:** multi-process etcd/TCP faults, epochs, bounded snapshots/events,
router contracts, shared-file races/GC, tenant isolation and dedup accounting.
Loss of coverage is not an authoritative absence result.
**Hardware later:** cross-host GPU/RDMA restores, permanent peer loss/physical
drain, MLA TP layouts and complete hybrid output/restart controls. Same-host
process tests do not establish independent failure domains or HA.

### M4 — Release and upstream

Build the wheel and deployment artifacts, then submit thin vLLM and SGLang
integrations with upstream tests/docs. Track both upstreams' review and release
requirements. Publication needs maintainer approval; the sole distribution is
`orbitkv`, and Rust crates remain unpublished. Before release, consume OrbiTE's
crates.io release and record resolved provenance.

The first candidate is `0.1.0-beta.1` (`0.1.0b1` on PyPI). It pins OrbiTE's
exact crates.io beta and lock checksum; release wheels are rebuilt against that
registry artifact. The beta carries the current installed-profile limits; it is
not the hardware-qualified 0.1.0 gate. See
[candidate and publication steps](releases.md#validate-before-publishing).

**CPU now:** wheel contents/loader checks, upstream interface/registration tests,
source-only suites, docs and reproducible manifests. These do not authorize
publication. **Hardware after M3:** one installed-artifact batch comparing native
caching, OrbitKV and compatible LMCache profiles with matched inputs/budgets,
byte/output, fault, restart and drain controls. Record TTFT, tails, throughput
and limits; do not inherit TENT results. Qualify the release candidate on claimed
hosts before the maintainer publishes 0.1.0. No hardware is requested by this plan.

## Invariants and evidence

Preserve identity/generation/coverage checks, ready versus drained receipts,
source grants, requester-specific segment retirement, epochs and bounded
admission. Timeout, cancellation, TTL and Unquiesced never release physical
owners; unknown submitted work is not replayed. Prefix/Window/Checkpoint
[recovery requirements](hybrid-recovery.md) remain implemented and used.

Current OrbiTE GPU, RDMA, GDS and installed profiles remain **unqualified**.
CPU/TCP, io_uring and shim controls are CI evidence. Historical TENT results and
revisions stay linked from [performance](performance.md); they do not qualify
OrbiTE. Metadata isolation **FAIL**, sixteen-owner **UNQUALIFIED**, and
observer-overhead failures are unchanged. Four owners remain the scoped limit,
not a new current-path hardware acceptance.

For each PR, report commit, scope, changed contracts, exact commands/results,
unrun gates, external artifact locations, remaining limits and line delta.
Keep output outside Git. Review precedes merge; publication and expansion of
qualification require explicit maintainer decisions.

[L]: https://docs.lmcache.ai/
[LL]: https://docs.lmcache.ai/legacy/kv_cache_optimizations/layerwise.html
[LC]: https://docs.lmcache.ai/mp/coordinator.html
[F]: https://github.com/taco-project/FlexKV
[FD]: https://github.com/taco-project/FlexKV/blob/main/docs/dist_reuse/README_en.md
[V]: https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
[M]: https://kvcache-ai.github.io/Mooncake/design/store/mooncake-store.html
