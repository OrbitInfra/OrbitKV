# State demand and transfer planning

Status: readiness, query ownership, compiled page demand and the first SSD
source/path separation are implemented. Cost observations and raw-copy/SSD-route
shadow are opt-in; measured execution selection and predictive planning remain proposals. The
existing cache API, engine-owned HBM, and one Cache Manager per host remain
the foundation.
The [SSD experiment](ssd-performance.md) supplies initial
measurements; predictive policies require separate evaluation.

The [completion plan](completion-plan.md) turns the
proposal into reviewable changes. [Manager decisions below the engine](#cache-manager-decisions-below-the-engine)
define the primary planning boundary; [Dynamo reuse](#reuse-dynamo-for-request-routing)
is a later integration for worker selection.
The design covers local tiers and Mooncake TE peer transfers with shared cost
observations, resource accounting and recovery checks. Implement local decisions
first; qualify distributed modes separately. Worker routing remains a later
milestone and is not required for transfer planning.

## Cache Manager decisions below the engine

Target behavior: an engine already assigned a request asks its local Manager
for compatible state. That Manager discovers local and peer replicas, selects
the recovery boundary and movement plan, acquires resource ownership and
executes it. This must work without a KV-aware request router. It does not
move the inference request to another worker or take over engine HBM allocation.

```mermaid
flowchart TB
    E["Inference engine: demand and page lifetimes"] --> M["Local Cache Manager: recovery and cost decisions"]
    M <--> C["Complete local global index: replica evidence"]
    M <-->|"Preparation, authorization and credits"| P["Peer Cache Manager"]
    M --> L["Local CUDA / io_uring / cuFile execution"]
    M --> T["Mooncake TE: peer payload movement"]
    T <--> P
```

The cluster view is queryable across the cache deployment. Each Manager keeps
its exact local inventory and a local index of installed owner views,
asynchronously synchronized through bounded Manager streams. Resource-pressure summaries remain
bounded evidence rather than replicated execution authority. Extend the existing
[global-index design](distributed-cache.md) without another central decision service.

| Planning evidence | Distribution and use | Authority at execution |
| --- | --- | --- |
| Membership, incarnation, capabilities and topology | Cached membership/configuration; etcd stays off per-read paths | Current owner incarnation and operation compatibility |
| Key, residence, representation, bytes and generation | Bounded owner snapshot/delta streams into local indexes with explicit coverage | Source validates and pins the selected residence |
| Source preparation pressure and export/staging capacity | Bounded, timestamped summaries; optionally refresh only a shortlisted owner | Source admits or rejects against its current credits |
| Transfer, decode and consumer-ready costs | Destination observations keyed by route, source/resource, representation and size; source supplies its own preparation evidence | Actual completion updates the estimates |

Resource summaries need sequence, incarnation, age and uncertainty. A queue
summary from another host is not a reservation or a synchronized clock reading.
Expire derived estimates conservatively and bound source probing. Incomplete
index coverage remains unknown; ordinary block discovery performs only local reads. Correctness
comes from source ownership and destination completion, independently of how
fresh the planning hints are.

For each request, first derive legal `required_ranges`. Discover a bounded set
of compatible replicas without reading payloads, then compare executable local
and peer routes to the same engine-ready state. Account for source preparation,
queueing, transfer, decode and GPU completion along their dependency critical
path; retain confidence and first-use deadlines. Use hard resource budgets and
tenant shares before comparing eligible completion times. The engine may resume
recomputation through its existing miss/admission contract; the Manager cannot
independently schedule engine compute.

Only selected work obtains source leases and destination credits. Bound partial
reservations and retries so simultaneous Managers cannot pin each other's pools
while waiting indefinitely. Source admission protects SSD, export bandwidth and
staging from many destinations choosing the same apparently idle peer. Replan
on stale evidence or rejection; use switching margins to avoid oscillation.
Stop unsubmitted work at the request deadline and retain submitted-I/O ownership
until safe completion, including after cancellation.

Retention, SSD writes, warming and useful replication use the same observations
on a slower budgeted path. Copy placement requires destination admission and
confirmed committed state before updating evidence. Replication goals need a
designated per-key decision owner before coordinated replica deletion; stale
global counts cannot enforce a minimum copy count. Keep this separate from
foreground restore decisions and from ephemeral P/D destinations.

This ownership differs from the audited [KVCR design](https://github.com/ai-dynamo/kvcr/blob/317e62f301300c101bf9c911e020fb89b35f1583/docs/design_overview.md),
which places cross-node inventory in the router and gives KVCR request-scoped
source hints. Borrow its claims and completion rules without adopting that
inventory dependency. NIXL estimates, if available, supply transfer costs;
neither NIXL nor Mooncake TE is the authority for OrbitKV state compatibility
or cluster-wide cache placement. Optional Dynamo routing can later consume
Manager summaries, while Managers retain discovery and physical decisions.

Current implementation has embedded discovery, bounded candidate caching,
per-residence advertisements, source authorization and opt-in cost observations.
Peer fetch first maximizes consecutive coverage. Among owners with the same
coverage and source medium, an additional opt-in can use fresh complete
HostReady estimates scoped by peer incarnation. Unknown/incompatible estimates
preserve the stable owner order, and source admission rejection replans within
the bounded batch. Peer resource summaries and the joint local/peer decision
loop above remain implementation work.

## What can be known ahead of time

| Evidence | What OrbitKV can prepare | Limit |
| --- | --- | --- |
| Tokenized request waiting for engine admission | Its exact reusable prefix and component set | Admission order and start time may change |
| Next scheduled batch or prefill chunk | The next restore and its first consumer | Requires an engine callback before pages are consumed |
| Known layer order | Later layer groups while earlier groups execute | CUDA graph replay needs device-visible dependencies |
| Tool call or application workflow graph | The already-computed conversation or shared system prefix likely to resume | Branches and return times are hints, not guarantees |
| Historical session reuse | Retention priority and bounded speculative DRAM warming | Cannot predict arbitrary new prompt text or safely evict live GPU pages |

Start with declared requests and measured queue delay. Add workflow hints only
after cancellation, resource limits, and demand-fetch fallbacks are reliable.
No learned predictor is needed for the first implementation. Unknown future
tokens never become a claimed exact cache hit.

## Engine signals and their present limits

vLLM 0.30.0 has a useful readiness contract:
`get_num_new_matched_tokens` may return `None`, asking the scheduler to retry.
OrbitKV already uses this for `QueryLoading`. `update_state_after_alloc` then
supplies valid destination pages. Publication and preemption callbacks supply
source lifetime evidence. This supports asynchronous lookup without blocking
the scheduler, but does not expose the entire future batch schedule.

SGLang 0.5.20's `UnifiedCacheLinker.lookup` reports restorable boundaries;
it has no pending result. OrbitKV now uses SGLang's general plugin
`HookRegistry` around `PrefillAdder.add_one_req`: an unresolved query leaves
that request in the waiting queue while other requests can be admitted. The
next prefix match polls the same query and retains its ready lease until load
or cancellation. Attention ranks reduce the wait/expiration decision together.
A five-second waiting budget cancels preparation and permits recomputation;
it does not cancel a submitted GPU restore. This budget is a fixed guard,
not a measured cost policy. No installed engine files are changed and HiCache
storage is not enabled. Multi-rank serving still needs separate qualification.

Both adapters currently acknowledge whole restores. SGLang's eight-request
submission window does not establish layer readiness; vLLM's layer callback is
not a per-layer completion fence. These are prerequisites for true overlap.

Shared client ownership now lives in Rust: query revisions, warming cancellation,
the independent publish connection, and eventfd/fallback restore waiting.
PyO3 exposes this owner directly; Python no longer maintains a second query map,
request counter or waiting loop. Each adapter prepares an immutable Rust
`BlockHashes` batch once per lookup. Prefix views share its allocation, so repeated
polls need neither per-page Python conversion nor hash copying/comparison when
reusing that batch and view. Changed inputs still revise the owned operation;
equivalent independently built batches are compared by value. Python still
checks engine request drift and handles GPU allocation. Hybrid discovery now
separates metadata from payload reads; the selected ranges are sliced and
validated in Rust before the adapter reports a hit.

## Minimal demand contract

Evolve `orbitkv-state` and the existing channel instead of introducing a second
cache service or adapter facade. A demand record needs:

- Request/session identity and a revision for changed or cancelled work.
- Computation identity, logical token boundary, and required state components.
- Estimated first-use time, priority, confidence, and a maximum waiting budget.
- A target engine/rank topology; an HBM destination only after engine allocation.
- An operation/lease ID with a terminal completion or drained cancellation.

The result distinguishes a miss, a candidate that can be fetched, a reserved
source, resident state, and state ready for GPU consumption. A directory hint
alone cannot authorize skipping computation. Speculative warming uses a
bounded budget and does not pin engine pages before admission.

Selected-boundary reads now send `RecoveryDemand` through the existing Rust
channel: page size, engine-declared valid-prefix token origin, selected token end and every required
group range. The registered instance/session supplies the physical namespace.
The Manager validates the complete group set and selected group's hash count
before query admission or byte reservation. Changed boundaries or other-group
ranges revise the operation even when the selected group's hashes are unchanged.
An incomplete selected group returns no lease. Groups still execute separately;
the engine coordinates rank agreement and complete model recovery. Priority,
first-use estimates, joint multi-group admission and destination generations
are not part of this implementation.

## Recovery semantics before placement

The existing model fingerprint establishes computation and storage identity.
A planner also needs a complete `StateBundle` at a legal boundary:

| Model state | Evidence needed to resume at token boundary t |
| --- | --- |
| Full attention | All required attention groups cover the needed prefix |
| Sliding-window attention | The model-specific trailing window ending at t |
| Recurrent/SSM hybrid | An exact checkpoint at t plus convolution and attention state required there |
| MLA or native sparse attention | Compatible latent/positional bytes and any required indexer or auxiliary state |
| Speculative decoding | Accepted-token boundary; draft state cannot be published as committed target state |

For a hybrid model, a longer attention match with a shorter recurrent
checkpoint is not a longer restorable prefix. The engine adapter supplies
model-specific rules; the shared contract checks component coverage and
compatibility. Generation-qualified HBM references must enter the actual
transfer path before stale page IDs can be rejected at that boundary.
The [compiled recovery contract](hybrid-recovery.md) implements SGLang's
prefix, window and checkpoint cases in both engines, including their combination.
The same normalized rules emit `required_ranges` for a selected boundary;
SGLang checks transferred plus retained pages and vLLM masks hybrid destinations
against those ranges. Adapters supply absolute coverage backed by leases and
keep engine-specific allocation and handoff logic. This is a prerequisite for
cost selection,
not a latency estimator or an automatic model-graph proof.
Cross-engine byte reuse, dynamic LoRA, and live weight changes remain outside
the present supported contract.

Hybrid demand now separates candidate discovery from materializing bytes.
The pinned
[LMCache prefetch controller](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_controller.py)
is a reference for trimming per-group demand while preserving an engine-owned
fallback. OrbitKV's implementation discovers DRAM/SSD/catalog positions, uses
Rust to compute legal boundary intersections, then slices actual reads to
`required_ranges`. vLLM applies its final-token budget before reading; SGLang
synchronizes selected-boundary readiness before allocating destinations.
Actual leases must cover every selected page. Stale evidence releases partial
state and falls back to recomputation from the valid HBM origin. This preserves
query-relative positions and independent consumers' read ownership. See
[hybrid recovery](hybrid-recovery.md) for exact-byte gates. Additional discovery
rounds and avoided payload bytes are separate effects; no TTFT gain is claimed.

## Decide when to copy, retain, and restore

Three decisions share observations but have different lifetimes:

| Decision | Inputs | Result |
| --- | --- | --- |
| Restore | Legal boundaries, existing replicas, representation, deadline and resource queues | Required ranges, selected sources and an executable transfer plan, or an engine-owned recompute proposal |
| Retain and publish | Observed reuse, avoided computation, capacity and write pressure | Whether to create/retain a replica, its tier and permitted encoding |
| Schedule | Admitted demand, preparation, writeback and completion dependencies | Bounded batches and resource reservations, with demand headroom and bounded write starvation |

The engine supplies request/revision identity, valid HBM origin, page lifetime
and available scheduling evidence. Roles and TP/PP shape belong to registered
engine instances and operations, not a single role on the Manager's membership
record: one Manager can serve several replicas or both prefill and decode.

Materialize a replica while its immutable source is valid and spare transfer
capacity is available. Copying a replica early does not authorize freeing HBM:
the engine decides semantic liveness and the transfer completion proves that
DMA no longer reads that generation.

Keep likely near-term reuse in DRAM. Consider SSD admission for reusable prefixes
whose avoided recomputation justifies writes and their expected storage time.
Measure saved GPU time, copied bytes, and retained byte-seconds separately;
do not add those quantities without explicit cost weights. Large one-off
prefills should not automatically monopolize the SSD write queue. Restore
traffic needs priority with bounded write starvation.

### Unified replicas, routes and execution ownership

Core's `planning/` uses a bounded `ReplicaSet` for local DRAM, local SSD and
peer DRAM/SSD. Each record separates medium from acquisition evidence: weak DRAM
ownership, an SSD index version, or peer owner/incarnation and inventory sequence.
Peer refresh preserves local evidence and the directory's replica bound. Current
owner inventory advertises preferred DRAM or surviving committed SSD evidence,
including representation family and known stored bytes. Unknown values remain
explicit; HBM records do not become executable routes, and SSD records use their
own source preparation plan rather than masquerading as peer DRAM.

One Storage-owned residency stream tracks current DRAM plus committed SSD for
each owner/key. DRAM is the advertised preference while present; its eviction
reveals the surviving SSD evidence without losing the owner. SSD commit is the
publication boundary, and ring overwrite or corruption removes its evidence.
Peer SSD discovery is not treated as directly readable memory. The requester
binds each selected segment to SSD evidence; the source revalidates an exact SSD
generation and materializes it through bounded io_uring host staging before TE.
Two-host correctness and performance qualification remain open.

`ReadPlan` now retains unresolved candidates for an admitted query batch and
declares host-ready preparation or engine restoration. The coordinator keeps
already acquired DRAM prefix holds. SSD and peer route plans borrow the same
records, validate required coverage, and acquire sources through their existing
owners. Host SSD reads also submit exact-generation leases; they no longer
rediscover by key. Both queued and submitted reads retain these leases through
cancellation until completion. Peer rejection updates only the rejected owner,
without truncating the batch or discarding SSD alternatives.

The top-level selection order remains local DRAM first, then eligible deferred
local SSD restoration. Host materialization tries peer DRAM, permitted local SSD
io_uring, then peer SSD. Explicit local SSD paths do not silently switch to host prefetch. Preparation
produces host state; an engine restore completes only through the existing GPU
completion owner. Metadata discovery still returns positions to engines, so it
is not a reusable read grant. Plans are per admitted query batch, not a joint
optimizer across groups, ranks or all request batches. Full route/resource
estimates, consumed endpoint descriptors and measured cross-source selection
remain open.

Host materialization now asks `ReadPlan` for one bounded `HostReadRoute`, which
returns either the existing peer segment plan or the existing io_uring SSD plan.
The query coordinator performs discovery and owns shared-read lifetime, but no
longer duplicates peer-before-SSD route choice. Planning remains metadata-only:
the selected peer still needs authoritative authorization, while SSD pins the
exact indexed generation only when its execution owner acquires the route.
Peer evaluation counts each candidate's consecutive coverage before allocating
authorization records only for the winner. This removes allocation proportional
to every rejected peer without changing source priority or retry semantics.

After query leases meet real engine destinations, `engine::restore` builds a
consumed `RestorePlan` from only referenced source indices. It binds the
EngineRestore target to one CUDA device, deduplicates source geometry, records
one SSD route plus source bytes/fragments and DRAM mixing, and owns no payload.
The GPU worker rejects a plan for another device or payloads that disagree with
its route. Lane dispatch and complete SSD cost shape consume this plan instead
of independently rediscovering intent from every layer. Source leases, query
reservations, destination pages and completion remain owned by `LoadTask` and
the existing workers.

| Concept | Information and responsibility |
| --- | --- |
| Recovery demand | Existing `RecoveryDemand`, request revision, required groups and ranges, and a declared target state. Engine-visible recovery additionally needs engine-owned destination authorization; speculative host preparation has a different completion target. |
| Resource endpoint | Owner/runtime incarnation, node and concrete resource identity: DRAM pool/NUMA placement, SSD store, or GPU/allocation domain. Medium is DRAM, SSD or HBM. Locality is derived relative to the consumer; another Manager on the same host still requires its ownership protocol. |
| Replica candidate | State identity and coverage, endpoint, immutable residency version, layout/encoding, known logical/stored sizes and bounded freshness evidence. Local evidence retains weak/index references; peer evidence retains an opaque source reference. Discovery owns no payload. |
| Restore route | A source replica, a declared destination state, and supported transfer/staging/decode steps with completion dependencies. Multiple routes may read the same replica and version. |
| Admitted plan | Selected routes plus actual source leases, destination authorization, staging and queue reservations, handed to the existing execution/completion owners. A proposed route or estimate alone grants no access. |

Use a concrete internal enum for evidence that must be acquired differently;
do not build invalid combinations from optional file offsets, GPU pointers and
peer addresses. Share only descriptors actually consumed across module/process
boundaries through `orbitkv-state`. Do not add public types for unsupported
routes or a second query-lease registry.

The endpoint model covers the following residences without inventing a backend
for every local/remote combination:

| Medium | Consumer's node | Another node | Current executable scope |
| --- | --- | --- | --- |
| DRAM | Manager-owned pool | Peer Manager-owned pool | Both supported; peer payload currently stages into requester DRAM |
| SSD | Manager-owned store | Peer Manager-owned store | Local supported; peer io_uring preparation/authorization and fixed-priority requester route implemented, two-host qualification open |
| HBM | Engine pages or Manager GPU allocations, with distinct owners | Peer engine pages or peer Manager GPU allocations | Local registered restore destinations/staging supported; general HBM cache sourcing is not implemented; experimental P/D is a separate contract |

Engine HBM becomes a source only with an engine grant binding instance/session,
page generation, range and access rights. Registration or a known device address
is insufficient. Destination authorization must prevent page reuse until all
submitted writes finish. GPU staging is temporary workspace and is not advertised
as a cache replica unless explicitly admitted for retention. Do not unify these
owners behind unrestricted `allocate`/`free` operations.

Preserve the current `StateKey` during the first migration. Its namespace already
binds storage format, layout, TP/PP and geometry. Initially compare replicas
within that compatibility domain. Comparing raw/ANS or different layouts as
alternative representations later requires an explicit logical-state identity,
physical representation identity and conversion/quality contract, including
versioned index/catalog changes. Adding a format field does not authorize
merging existing namespaces or dropping their compatibility checks.

Route construction uses a small set of implemented templates. io_uring and
cuFile are independent routes over one SSD store and one source generation.
Mooncake connects registered memory endpoints; it does not turn an SSD extent
into remotely readable memory. A peer SSD route must ask the source Manager to
pin that extent and admit its read and staging resources before publishing a
transfer-ready memory window. Charge source preparation and both endpoints;
the requester must not allocate or schedule the peer's internal resources.
Peer HBM routes require the engine-grant contract above before enumeration.

GPUDirect RDMA is a Mooncake TE transport capability for registered GPU memory,
not a residence or another remote backend. A route must retain both memory
endpoints and validated transport capability: host-to-host RDMA, host/GPU RDMA
and GPU-to-GPU RDMA have different prerequisites and resource costs. NIC selection,
connections and low-level retries remain with TE. Do not infer GPU reachability
from a successful TCP/DRAM transfer, nor expose another transport switch here.
Ordinary shared-cache recovery still uses peer DRAM → local DRAM → engine HBM;
the experimental GPU P/D handoff does not advertise peer HBM cache replicas.

Compare only routes satisfying the same demand and completion target. A host-ready
prefetch cannot compete as if it were an engine-visible restore. Do not enumerate
every cluster resource or search an unrestricted transfer graph. Use bounded
cached evidence and demand-driven directory queries; capability/health evidence
filters routes before costing, and acquisition revalidates the chosen version.

`CostEstimateKey` now carries that completion target explicitly as an intent plus its
target resource. HostReady has no engine resource, EngineRestore names the
destination GPU, and GPU save completion names the source whose pages become
reusable. Rewriting a candidate observation kind or execution resource
recomputes this field, so an
SSD restore for GPU 1 cannot accidentally reuse evidence collected for GPU 0.
This is already enforced by raw-copy/SSD shadow and host-route selection. Future
direct-to-D and prefill-restore-plus-handoff candidates must both end at the
same DecodeReady device identity before entering the comparison set.

The first authenticated `CompletionObservation` boundary now records vLLM
prefill-to-decode handoff evidence at the decode owner. The local process
channel accepts a bounded frame containing the registered instance/device,
prefill endpoint identity, nonzero transfer generation,
representation, logical/wire bytes, fragments, admission, outcome and elapsed
time. It contains no request ID or state key. Core hashes the source endpoint
into an `ExecutionResource::PrefillToDecodeHandoff` identity and excludes the
generation from the statistical key. Only admitted completed reports update
the estimate; rejected, failed, cancelled and timed-out reports remain
diagnostics. This adds
evidence, not a candidate enumerator or execution selector. Direct cache restore
records the same post-allocation DecodeReady boundary for Manager-executed
restores: `Restore` begins
after the framework has allocated decode pages, Core consumes fresh source
leases, validates the registered destination ranges, and records their device,
bytes and fragment count in `RestoreTargetShape`. This shape remains in the
`RestorePlan` through completion; it owns no page lifetime or admission.
The framework retains destination pages, and a separate `DecodeRestorePermit`
owns device capacity. The worker finishes the observation only after every
GPU operation is terminal.
This supplies comparable completed-route evidence for those routes. Engine-local
raw grants retain the same device admission but do not yet train this estimator;
their actual drain occurs in the engine before Manager source retirement. See
the [next measurement increment](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).

Completion resource evidence is separately bounded and freshness-checked. A
direct restore has a per-device 128-operation admission owner retained through
terminal completion. P/D observations carry exact admitted decode bytes,
handoff queue depth and parallelism, and the TENT rail inflight-byte/bandwidth
snapshot captured at admission. These values never become metric labels or
stable estimate-key dimensions. Fresh evidence expires after two seconds;
decode-route comparison adjusts estimates by queued execution waves and, when
an RDMA rail is present, current TENT inflight bytes divided by observed rail
bandwidth. Missing or shape-incompatible resource evidence preserves the
deterministic default.

### Cost model for complete routes

Keep the shared measurement substrate and make its two boundaries explicit:
operation samples describe individual I/O/copy/codec work; complete-route samples
describe a named start-to-declared-target completion interval. Engine-visible
restore and host-ready preparation use distinct targets and estimate keys. The
`CostObservationKind` includes both operation and composite boundaries. It is
not a set of additive graph-edge costs.

The planner's estimate should report a predicted ready time, measured uncertainty
and freshness, and the resources needed to execute. Resource demand includes
source holds, pinned DRAM/GPU staging peaks and queue/transfer slots. Retained
byte-seconds and actual I/O are separate costs, not quantities to add to latency
without a defined objective. Existing EWMA absolute error is an empirical margin,
not a calibrated confidence interval or p99 guarantee.

Estimate from route shape, layout/encoding, logical and aligned/stored/wire bytes,
fragmentation, resource identities, actual transport mode and current resource
evidence. Keep cuFile compatibility/native GDS and TCP/host-memory RDMA/GPUDirect
RDMA samples distinguishable; unknown mode is not proof of a native path. Use bounded device/store/peer-runtime
keys, never request IDs or per-replica generations in the statistical index.
Generation belongs to correctness validation. Missing format, size, queue or
cost information stays unknown rather than becoming zero.

There are two valid estimation methods, which must not be added together:

- Fresh, matched complete-route observations can directly predict that route's
  finish time. Their embedded historical waiting cannot then be counted again
  as a separate stage cost; changed contention may invalidate their applicability.
- Disjoint stage observations can feed a bounded dependency schedule. A stage
  starts after its predecessors and required resources are available; parallel
  branches complete at their latest finish. Chunked pipelines require their
  actual dependencies and shared-resource constraints. A source read, network
  transfer and decode cannot all be treated as independent if they contend for
  the same workspace or execution resource.

Current queue/usage snapshots guide estimation but do not reserve capacity.
Actual resource owners remain authoritative. Acquire leases/credits in a bounded
order; if admission fails, release unsubmitted reservations and replan within
the request deadline without accumulating holds across peers. Once submitted,
the completion/drain owner retains resources even if the request is cancelled.
An uncertain or partially completed destination write cannot be replayed through
another route until the old work is safely fenced and the destination revalidated.

Selection first enforces semantic compatibility, exact/approved quality and
resource limits, then compares conservative completion estimates. Preserve the
existing admissible route when evidence is absent or the predicted gain does
not exceed measured error and a declared switching margin. Initial integration
remains shadow-only; do not fetch alternatives just to populate samples. Avoid
new per-tier weights or policy configuration: derive resource identity, capability
and current usage from the real owners, using existing capacity/deadline contracts.
Retention/write admission stays a separate decision using the same observations.

### Code ownership and migration

The implemented Core layout separates physical residency, planning, request
ownership and execution. Existing crates are retained; no generic tier trait or
compatibility re-export was added.

GPU worker pools remain instance-owned so registration, transfer backend and
drain lifetimes stay isolated. Their GPU-storage write admission is process-wide
per CUDA device, because several instances on one Manager contend for the same
staging and storage queue capacity. The lookup happens when a pool is created,
not on the request hot path. `SaveTask` owns an acquired permit until terminal
completion; caller cancellation cannot return capacity early. A full shared
budget preserves the existing host-publication fallback. Shared read/staging
admission and fair shares remain separate work.

The first staging-owner boundary distinguishes automatic SSD demand from an
explicit route control. Automatic demand prefers cuFile when capability is
present but may switch the same immutable extent to io_uring before any worker
submission if another instance owns this CUDA device's persistent staging.
Explicit cuFile is a qualification contract and fails closed instead. One pool
per device owns the persistent two-slot cuFile worker and codec/direct-write
staging until all of its lanes drain; this bounds instance-multiplied HBM while
keeping registered resources warm for its serving lifetime. Idle handoff and
fair device shares remain later work and must be measured before relaxing this
owner model.

```text
orbitkv-state/       shared state/recovery and consumed descriptor contracts
orbitkv-core/
  engine/           registration, EngineConfig and engine-facing orchestration
  storage/          residency assembly and allocator-driven reclamation
    dram/           resident images, eviction policy and inventory
    ssd/            files, immutable extents and independent storage I/O routes
    publish.rs      queued sealing, DRAM insertion and SSD admission
  planning/         discovery -> batch replica evidence -> eligible source routes
  query/            admission, shared reads, materialization and query leases
  transfer/         GPU movement, staging, submission and completion/drain
  peer/             local discovery, source exports, remote READ and release recovery
  codec/            representation validation and encoding/decoding
  cost/             observations, bounded estimates and shadow comparisons
orbitkv-server/
  peer.rs           peer RPC parsing, error mapping and response serialization
```

`planning` does not own an allocator, file, CUDA stream or peer payload.
`cost` does not acquire leases or prove source availability. Its `observation`,
`estimates` and `shadow` modules share bounded path/resource/representation keys;
composite route timings must not be added to their constituent operations.
`query/read.rs` owns shared reads and sparse membership materialization, retaining
candidate batches through acquisition. `storage/publish.rs` owns incomplete
blocks and its worker's DRAM/SSD references instead of passing weak dependency
wrappers through storage forwarding methods.

The former `backing/` and `internode/` trees are removed. `peer/export.rs` is the
authoritative source owner: it checks incarnation, membership, namespace and
residency sequence, accounts pinned allocations and drains completion after
fencing. `peer/transport.rs` holds the registered pinned pool until unregister.
Server owns etcd membership Watch, bounded Manager inventory sessions and the
inbound source gRPC adapter; outbound source-requester clients remain with their
Core peer workflows. The Mooncake crate remains the byte-transfer boundary,
including NIC and transport selection.

These ownership changes preserve default source priority and the opt-in cost
policy. Complete-route comparisons, shared-device admission and new peer memory
endpoints still require their own implementation and qualification.

The refactor sequence is:

1. Bounded records now replace the three-field candidate shape for existing
   DRAM/SSD/peer-DRAM/peer-SSD sources, preserving exact namespaces, metadata-only
   discovery and version revalidation. Admitted read batches now retain these
   records through source selection and acquisition. Joint multi-group/request
   planning and richer consumed endpoint descriptors remain open.
2. Host-ready and engine-restore targets now control existing source routes,
   preserving today's default selection and io_uring/cuFile on one extent. Next
   compare complete routes to the same target, including staging and decode.
3. Bind the selected plan to current query/source/destination/completion owners.
   Establish shared device admission across registrations before dynamic choice.
4. Connect complete-route shadow estimates and resource evidence. Qualify
   lifecycle, source invalidation, deadline/cancellation and matched overhead
   before allowing a measured route to change execution.
5. Source-side SSD preparation and fixed-priority requester execution now use
   the same ticket owner and Mooncake TENT path. Add engine-authorized HBM replicas
   only when their protocols and deployment gates are ready. Cross-representation
   discovery requires its own identity migration; none is implied by endpoint normalization.

Upstream mechanisms supporting this organization were checked at pinned sources:
[LMCache MP](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_controller.py)
separates policy, I/O and lock/completion ownership;
[vLLM](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/core/block_pool.py)
separates logical hashes from physical blocks and references;
[SGLang](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/radix_cache.py)
holds distinct device/host residencies and references for one logical node;
[FlexKV](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/transfer/scheduler.py)
uses transfer dependencies and completion-driven scheduling. These are design
inputs, not evidence that either their default policies or OrbitKV already
perform unified measured path selection.

### Transfer paths and cost observations

Represent a candidate by location, representation and valid generation, rather
than a tier name alone. A block may have several replicas. Only discover
metadata while enumerating alternatives; reserve and revalidate the selected
sources before reading. Estimates never prove source availability or readiness.

Core discovery now retains concrete local DRAM, local SSD and known peer DRAM/SSD
evidence together instead of collapsing each key immediately to a Boolean.
DRAM evidence is a weak reference to its stored image; SSD evidence includes
the indexed representation, bytes and immutable generation token. Neither
pins payloads. SSD admission checks eligibility and then pins that exact
generation under the index lock; replacement makes the candidate stale even
when its key and file offset match. Peer evidence contains owner incarnation
and residency sequence; absent byte/format information remains unknown.
Discovery uses bounded batches and one directory deadline, while local hits
can reuse cached peer evidence without another RPC. The current engine API
still receives positions, and reads reacquire source evidence. This preserves
alternatives inside discovery; it does not yet rank them by complete cost.

```mermaid
flowchart LR
    H[Engine HBM] <-->|DMA or copy kernel| D[Pinned DRAM]
    H <-->|Gather, scatter or codec| G[Manager GPU staging]
    G <-->|DMA| D
    D <-->|io_uring| S[Local SSD]
    G <-->|cuFile| S
    P[Peer registered DRAM] -->|Mooncake TE| D
    PS[Peer SSD] -->|Source io_uring staging| P
```

CUDA IPC imports engine allocations; it is not the HBM/DRAM movement mechanism.
Current GDS paths use registered GPU staging, not arbitrary direct engine-page
I/O. Speculative SSD preparation targets DRAM. Peer SSD uses owner-local
io_uring staging before the same TE path; peer-HBM cache sourcing remains future
work. The experimental vLLM GPU-to-GPU P/D connector is a separate handoff path. See [GPU storage](gds.md) and
[distributed source ownership](distributed-cache.md#transfer-lifetime-and-resource-control).

Current selection is not a calibrated policy: DMA/kernel is selected at instance
registration, SSD `auto` selects capability and disables new cuFile work after
failure, and cache discovery uses existing tier/coverage rules. Rust observes
these owners and evaluates raw-copy and SSD-route candidates in shadow. Execution
uses the registered copy backend and the default or explicitly configured SSD route.

Host-staged io_uring and cuFile GPU staging are now independent executable
candidates over the same SSD index, representation and immutable extent lease.
Healthy cuFile does not suppress the io_uring candidate. The host route reads the
leased generation through the existing bounded reader queue on a separate restore
lane, then copies/decodes into engine HBM; it does not look up a replacement by key.
The cuFile route retains its existing registered staging and completion owners.
Neither route makes staging a separate cache tier.

`--ssd-backend` controls cuFile capability initialization and existing write behavior.
`--ssd-read-path uring|cufile` explicitly chooses the demand-read route for ablations;
when omitted, existing source priority and capability-based selection remain.
An unavailable explicit SSD route is not silently replaced by host prefetch;
ordinary remote recovery remains available. Prepare/warmup still fills DRAM.
This is a local source/path boundary, not a general cost-driven tier planner.
Measured selection, shared-device scheduling and native GDS qualification remain
open. Failure never revokes submitted source or destination ownership.

The following evidence and estimation requirements describe the target policy.
The implemented P4.1 subset is specified in the observation contract below.

| Candidate | Cost evidence |
| --- | --- |
| DRAM to HBM, DMA | Logical/payload bytes, coalesced ranges, copy queue and PCIe/NUMA path |
| DRAM to HBM, copy kernel | Fragment count, launch time and interference with inference kernels |
| SSD through DRAM to HBM | SSD read queue/service, host staging, H2D and measured overlap |
| SSD through cuFile to HBM | Actual I/O mode, alignment/padding, GPU staging, scatter/decode and completion |
| Encoded restore | Stored and transferred bytes, codec/CRC service, workspace pressure and inference interference |
| Peer DRAM through TE to HBM | Discovery, source authorization, source/destination pressure, TE service and local restore |

Build bounded Rust estimates from execution events, with the same observations
feeding exported metrics. Do not query the metrics service from the planner.
Key estimates by direction, resource/device/NUMA path, size/fragmentation bucket
and format. Peer estimates also need incarnation and transport context; an
endpoint address alone must not carry old measurements across restarts.
Track short/long rolling estimates, sample counts and prediction error. Separate
queue wait from service time and account for shared SSD, PCIe, NIC and GPU
resources once. Timeouts are censored observations, not exact completion times;
continue tracking submitted work through terminal completion.

Start with a small set of supported executable paths. A generic shortest-path
solver over fixed edge weights cannot account for budgets, contention or
pipelined work. Engine prefill profiles need model, shape and live load evidence;
the Manager cannot infer a universal recompute cost from token count alone.

### Candidate routes and eligibility

Here, local means the requesting engine's host. Enumerate only sources backed by
current candidate evidence and paths with implemented, qualified executors.
The following is the design space, not the present shadow selector's coverage.
Use [residence plus owner/resource location](distributed-cache.md#residency-proximity-and-access-paths)
to discover replicas; access methods and temporary staging are not additional
cache tiers. Several residences may contain the same compatible state.

| Route to the consuming engine | Current boundary / admission requirement |
| --- | --- |
| Local DRAM → local HBM | Implemented DMA/kernel; raw-copy shadow preserves the registered backend |
| Local SSD → io_uring → local DRAM → local HBM | Explicit route over an immutable lease, separate host lane; full-restore shadow includes optional decode and H2D |
| Local SSD → cuFile → local GPU staging → local HBM | Independent route and full-restore shadow over the same lease; native GDS qualification and measured selection remain open |
| Peer DRAM → TE → local DRAM → local HBM | Current shared-cache route; physical two-host/RDMA qualification remains open |
| Peer DRAM → TE → local GPU staging → local HBM | Planned; GPU registration, capacity and decode/scatter ownership required |
| Peer DRAM → TE → local engine HBM | Planned; registered destinations, exact engine-ready bytes/layout and completion visibility required |
| Peer HBM → TE → local DRAM or GPU memory | General cache sourcing planned; source engine must grant a lifetime lease. GPU-to-GPU vLLM P/D is an existing experimental handoff, not general peer-HBM reuse |
| Peer SSD → owner DRAM → TE → local DRAM or GPU memory | Source io_uring staging, two-phase source credits and explicit requester planning implemented; two-host qualification remains open |
| Peer SSD → owner GPU staging via cuFile → TE → local GPU memory | Later candidate only on qualified source GDS and GPU RDMA paths; charge source GPU space and interference |
| Mounted remote storage → cuFile → local GPU staging | Separate future storage deployment; not an implicit Mooncake peer-memory path |

Native GPU RDMA requires registered GPU allocations and a supported device,
driver and network topology. TCP support alone does not make a GPU-direct route
eligible. GPU staging and engine HBM are different destinations: encoded bytes
need validation/decode before the engine consumes them. Do not enumerate every
peer/tier/codec combination or recursively fetch through peers to manufacture
an available source. Score bounded, useful alternatives, including local
recomputation, against the same required state and consumer deadline.

Ordinary reuse uses requester-driven TE READ; P/D uses producer-driven WRITE
into admitted destinations with completion notification. READ/WRITE specify the
initiator and operation, not whether either memory endpoint is DRAM or HBM.
OrbitKV selects sources, destination ownership and staging; TE owns the actual
network transport, rails and endpoints. Include observed transport identity in
cost evidence when available, without duplicating TE's transport scheduler.


### P4.1 observation contract

The Rust cost observer sits in the existing GPU workers, SSD queues and peer
fetch owner. It keeps at most **512 entries** for path, resource incarnation,
representation family and logarithmic byte/descriptor-count buckets. SSD route
keys additionally distinguish whole leased source-image bytes/fragments and the
SSD-derived target bytes/fragments, including mixed DRAM/SSD jobs. These source
sizes describe stored images, not inferred unpadded logical data. An entry
needs four successful samples and expires after five idle minutes; stale evidence
must warm up again. It maintains an EWMA estimate and absolute error: individual
operation paths use completed service time, while `ssd_uring_restore` and
`ssd_cufile_restore` use enqueue-to-GPU-terminal total time for the complete route.
Estimator contention drops an update instead of blocking execution. Resource,
request and state identities never become metric labels.

| Boundary | Meaning of the host-observed sample |
| --- | --- |
| Raw GPU load/save | Enqueue, worker admission, backend submission, existing stream completion; successful DMA reports copied payload bytes |
| Encoded GPU load/save | Inclusive codec and staging operation; overlaps existing codec/transfer timers |
| SSD route restore | `ssd_uring_restore` / `ssd_cufile_restore` span enqueue through GPU terminal completion, including queueing, materialization and copy/decode or staging/scatter; physical bytes belong to underlying I/O owners |
| GPU SSD write | Inclusive worker job; underlying cuFile or io_uring owns the physical byte count |
| io_uring read/write | Queue, SQ preparation and observed CQE; actual returned bytes include short I/O, and alignment is physical rather than logical data |
| SSD prefetch | One shared physical batch, including allocation and reconstruction, excluding subsequent H2D; consumers are counted separately by existing query metrics |
| SSD write batch | Ingest queue, admission and the batch's final ordered completion/commit; child I/O owns physical bytes |
| cuFile slot | Submission to existing completion event; read includes scatter, write includes gather; actual transferred bytes include padding |
| Peer authorization / READ | Authorization is separate from the blocking READ queue and terminal native completion; evidence is scoped to peer incarnation |

`orbitkv_cost_stage_seconds` exports queue, admission, completed service and
inclusive total. These are host clocks, not CUDA event or device-service times.
**Do not sum parent and child path timers.** Failed, cancelled, timed-out or
abandoned operations never train a successful service or route-total estimate. A dropped caller
cannot end an in-flight physical observation: io_uring and TE retain theirs at
the completion owner. Missing terminal evidence is not a zero-duration success.

`orbitkv_cost_logical_bytes` is the known attempted logical data within each path;
raw backing images without unpadded metadata increment `cost_logical_unknown`
instead of treating allocation padding as logical state. GPU layouts and encoded
metadata provide exact logical sizes.
`orbitkv_cost_io_bytes` contains only physical bytes known at that I/O owner.
Unknown counts have their own counter rather than being reported as zero.
Partial failed GPU/TE work may have unknown bytes. Kernel descriptor uploads and
codec internal traffic do not yet have complete byte attribution. Physical I/O
is counted once when a shared read serves several logical consumers.

The initial shadow consumer inspects the **actual raw copy descriptors**. DMA
is feasible for ordinary copies; the kernel alternative requires mapped host
addresses. It reports predictions, age, count and error for those candidates
without launching alternatives or changing readiness, ownership or registration.
Unobserved alternatives remain unknown. Raw-copy keys retain both descriptor
count and DMA-coalesced range count, using the executor's own merge iterator.
The actual descriptors refine the observation and shadow together without
restarting queue timing. These shapes still do not model NUMA placement or
interference with inference kernels.
The [fixed-backend DRAM comparison](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#dmakernel-comparison-final-evidence)
exceeds the throughput and TTFT p50 regression budgets for kernel in both
engines; it does not qualify automatic selection or change registration defaults.

SSD-route shadow uses the same leased sources and requested GPU destinations,
considering io_uring and cuFile only when their route requirements are satisfied.
It compares full-restore estimates with the same completion target. It launches no
alternative reads and does not choose execution; unobserved alternatives remain
unknown. Route totals overlap child operation timers and must not be added to
them. Dynamic route, representation and legal-boundary selection remain later
work. No measured speedup is attributed to an unexecuted path.

The observer explicitly distinguishes submission-to-completion operation samples
from enqueue-to-completion SSD-route samples. Resource keys separate GPU,
SSD store/file and peer-incarnation identities; a complete SSD route retains
its destination GPU, copy backend, source-store set and mixed DRAM presence.
These are internal cost identities, not exported labels, live resource credits
or full replica endpoint descriptors.

Shadow now rejects incompatible comparison families, resource identities,
representations or shape buckets. It accepts only the existing load-copy,
save-copy and SSD-restore pairs derived by their execution owners from the same
actual work. Matching estimator buckets alone does not prove equal demand.
Cross-source local/peer comparisons still require a separate complete-route and
admission contract; they cannot reuse this same-source check blindly.

The declared margin is 5% of the current estimate. A different path is
suggested only if its mean plus empirical error is below the current mean minus
its error by more than that margin. Faster means inside this guard report
`within_margin`, incompatible evidence reports `incomparable`, and missing
estimates remain `unknown`. Shadow decisions never alter execution. The narrow
peer-owner selector applies the same guard only to equal-coverage, same-medium
HostReady routes; it does not qualify cross-medium or local/peer switching. See
[cost metrics](metrics.md#bounded-cost-observations) for the complete labels.

Observations and shadow work are **off by default**. Set
`ORBITKV_COST_OBSERVATIONS=1` before starting the Manager to enable them;
unset or `0` preserves the disabled baseline. The earlier observation-only matrix
passes five of six cells; SGLang ANS SSD exceeds the TTFT p50 budget, so default
enablement remains unqualified. That matrix predates independent SSD-route
execution; [current validation](https://github.com/OrbitInfra/OrbitKV/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/implementation-plan.md#ssd-sourcepath-separation-final-evidence)
is recorded separately.
Set `ORBITKV_COST_SELECTION=1` in addition to observations to allow the guarded
same-medium peer-owner choice. Either variable absent or unequal to `1` keeps
the stable longest-coverage/owner ordering. Selection never expands candidate
coverage, bypasses authorization or treats a prediction as a resource permit.
With observations alone, equal-coverage local SSD, peer DRAM and peer SSD routes
also enter a complete HostReady shadow when each peer alternative has one source
owner. Different coverage, unknown shape and multi-owner prefixes do not compare.
This shadow does not change the fixed host-route order. Experimental execution
requires the two flags above plus `ORBITKV_CROSS_MEDIUM_SELECTION=1`; every
equal-coverage candidate must have fresh compatible evidence. The selected
source still performs authoritative acquisition/admission. Pre-payload rejection
can replan; submitted payload failure cannot switch media.
There are no Python hot-loop callbacks, extra GPU synchronizations or payload
reads for telemetry. See [the paired workload](../benches/serving/README.md#cost-observation-overhead)
for the predeclared overhead budget and final qualification evidence.

### Choose a legal boundary and completion target

For the current whole-restore admission model, estimate times from the same
decision point:

```text
state_ready(boundary, path) = resource wait + restore critical path + completion visibility
first_token(boundary, path) = max(engine_admission, state_ready) + remaining_prefill(boundary)
```

Compare a bounded set of legal boundaries and their exact `required_ranges`,
including recomputation from the engine's valid HBM origin. The longest hit
need not yield the earliest usable state: a shorter DRAM boundary can beat a
longer SSD/peer restore. The engine confirms execution/recompute admission;
the Manager owns the selected transfers. Preserve complete rank/component
coverage, then revalidate actual leases before accepting the boundary.

Respect declared TTFT and inter-token-latency targets before optimizing goodput
and capacity. Include active-decode interference and uncertainty; do not add
milliseconds, byte-seconds and failure probability without explicit units or
constraints. Existing [codec controls](storage-formats.md#dram-control) improve
throughput while increasing median TTFT in some cases, so neither compression
ratio nor cache hit rate is a sufficient objective.

Initially select paths for existing representations. Adaptive encoding requires
explicit logical-state/physical-format identity and qualified readers; it cannot
silently change the current storage namespace. Lossy modes require an allowed
quality policy. CPU SIMD can reduce codec time but does not imply compressed
PCIe traffic when the CPU path restores full-width data before H2D.

Do not equate a predicted deadline with readiness. Switch plans before
submission, or only for unsubmitted batches with their own ownership. Submitted
I/O retains its source, destination and scratch until completion or proven
revocation. Use one remaining wait budget across stages and retries, interpreted
on each host's local clock; do not compare monotonic timestamps across hosts.

### Prepare before use and retain useful replicas

Schedule prefetch from its estimated deadline backwards:

```text
start <= first use - measured remaining critical-path latency - uncertainty margin
```

Stage SSD or remote bytes in DRAM while the request waits. Reserve HBM close to
admission. Prepared pages remain budgeted until consumption, cancellation or
expiry. Bound speculation, GPU staging/codec workspace, outstanding SSD/TE bytes
and per-instance shares, not only DRAM capacity. Current SSD writes use weak
source references and may be dropped under pressure; any future promised replica
needs a reservation. Once correctness supports it, pipeline disk chunks, H2D layer
groups, and computation through a dependency graph:

```mermaid
flowchart LR
    A[Queued request and recovery boundary] --> B[Reserve source and DRAM budget]
    B --> C[SSD or remote read]
    C --> D[H2D first layer group]
    D --> E[Compute first layer group]
    C --> F[H2D next layer group]
    E --> G[Compute next layer group]
    F --> G
```

Use measured overlap rather than adding all stage durations. A node may report
readiness only after its required device dependency is established. Cancellation
revokes scheduling interest, drains submitted transfers, and only then releases
pages or mappings. CUDA graph capture and replay both need qualification.

The transfer scheduler belongs in `orbitkv-core`: bound outstanding bytes,
prioritize demand reads by slack, cap speculative reads, and account for shared
SSD/PCIe/NUMA/NIC resources. The adapters supply demand and lifecycle events;
`orbitkv-channel` carries them. A future router can consume summaries after
the local planner is useful. No new central Catalog is required for this work.

## Policies by deployment mode

These are planning contracts, not additional supported-mode claims. DP, TP, PP
and P/D are composable deployment dimensions: a P/D service can have several
replicas with TP inside each. Attach operation intent, instance/rank topology
and completion requirements to work instead of introducing one exclusive
deployment-mode switch. Current evidence is listed in [deployment](deployment.md).

| Mode or operation | Decision objective | Completion and failure contract |
| --- | --- | --- |
| Single instance | Compare DRAM/SSD restore with legal recomputation; protect active decode | Engine owns admission/HBM; cancel drains submitted work |
| Several instances on one host | Share capacity while bounding each instance's pinned bytes and device queues | One model or speculative queue cannot exhaust demand capacity; layouts remain isolated |
| Independent replicas / DP | Compare local restore, authorized peer sources and recomputation for an already selected worker | Stale candidates are rejected; bounded discovery/retry may become a cache miss, with the engine confirming recomputation |
| P/D handoff plus reuse | Deliver current-request state to the admitted decode consumer, while retaining reusable state separately | Reserve destination capacity and complete required groups before decode; failed handoff needs explicit retry, reroute/re-prefill or request failure, never an ordinary cache-miss acknowledgement |
| TP inside a replica | Select a rank-common legal boundary and minimize time until all required shards are ready | Slowest required shard and collective dependencies affect admission; a local rank hit cannot authorize global readiness |
| PP, optionally with TP/DP/P/D | Prepare each stage's owned state ahead of its consumer and limit pipeline stalls | Stage dependencies and restart boundaries remain explicit; storage support alone does not establish PP integration |

Remote byte movement uses Mooncake TE through `orbitkv-transfer`. OrbitKV owns
state compatibility, replica discovery, source capabilities, budgets and the
restore plan; do not duplicate TE's transport implementation or infer RDMA from
a successful transfer. Current shared-cache recovery stages peer bytes in local
DRAM before GPU restore. Native RDMA, general peer-HBM restore, peer-SSD
two-host qualification and TP/PP integration each need separate evidence.

Extend the same cost observations with discovery, authorization, owner-local
preparation, TE transfer, destination decode/H2D and engine-visible completion.
Count release acknowledgement and retained source byte-seconds as resource costs
without automatically adding them to the request critical path. Cached directory
evidence avoids repeated discovery, not source lifetime checks; etcd remains
membership/configuration coordination and never a per-read dependency.

For remote SSD support, the owner prepares only its own SSD bytes into bounded
registered memory, then TE moves them. Source-side io_uring staging now reserves
ticket/session and allocator-rounded byte credits before allocation. Its detached
batch owns exact extent leases and staging credit through drain, atomically
commits actual allocation footprint, and publishes only after result handoff.
Requester plans use this route only after peer DRAM and eligible local SSD host
routes, with a distinct authorization cost path. Measured source selection and
two-host destination qualification remain open. Do not recursively fetch from
another peer or advertise SSD-only state as immediately RDMA-readable.
Peer-HBM sourcing similarly needs engine-owned lifetime evidence; it is not a
free Manager memory pool. Different TP/PP shapes require explicit compatibility
or a separately implemented resharding step.

P/D must also transfer reused P-side state needed by D; skipping P computation
does not prove D already owns its KV. Layer overlap needs actual producer and
consumer fences, not only average network bandwidth. A future request router
chooses eligible workers and consumes summaries; each selected Manager still
revalidates sources and leases. Keep routing optional.

## Reuse Dynamo for request routing

Source audit: September 24, 2026, Dynamo
[v1.5.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0), commit
`b83b1d9304ebfc624709ac46db32b1b6f1ff1615`. The capabilities below were checked
in that release, not inferred from the newer `dev` documentation. OrbitKV has
not yet built or integrated this dependency.

The routing implementation is already reusable:

- [`dynamo-kv-router`](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/lib/kv-router/README.md)
  exports prefix indexers, a local request scheduler, load accounting, and
  worker selection. Its
  [Cargo features](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/lib/kv-router/Cargo.toml)
  make `dynamo-runtime` optional. A Rust integration need not adopt the whole
  Dynamo runtime, although sibling hashing/token crates and enabled service
  dependencies still need build qualification.
- The release's
  [default selector](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/lib/kv-router/src/scheduling/selector/default.rs)
  combines projected prefill/decode load with device, pinned-host, disk, and
  shared-cache credits. Its score is expressed in weighted block equivalents;
  it is not a prediction of an SSD operation's completion time in milliseconds.
- The
  [standalone selection contract](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md)
  supports HTTP selection and active-load reservation without forwarding
  inference. `SelectionServiceBuilder` embeds that lifecycle in Rust. A
  production integration should use it rather than the intentionally local,
  unsynchronized `SelectionCore` helper.
- The experimental [custom selection API](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx)
  exposes Rust `WorkerFilter`, `WorkerScorer` and `WorkerPicker` policies while
  Dynamo retains eligibility, queues, reservations and accounting. Custom
  policies build against the same checkout as the frontend or EPP. Existing
  [topology constraints](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/router-topology.md)
  also support required or preferred P/D transfer domains; these constraints
  and biases do not measure a particular transfer's duration.

| Decision | Owner | Information required |
| --- | --- | --- |
| Select an inference worker or DP rank | Dynamo request router | Prefix overlap, worker eligibility, active prefill/decode load |
| Decide whether a token boundary is recoverable | OrbitKV recovery contract plus adapter evidence | Model/storage identity, token coverage, complete components |
| Choose and reserve the source replica | Cache Manager | Current residency, replica generation, source lease |
| Schedule SSD reads, H2D, D2H, and later remote reads | Cache Manager transfer planner | Bytes, queue pressure, staging capacity, first-use budget, completion dependencies |
| Allocate or reuse engine HBM and schedule computation | Inference engine | Page lifetime, batch membership, execution dependencies |

A Dynamo load reservation books projected inference work. An OrbitKV lease
holds a source or destination generation alive. They have different owners
and completion conditions. Selecting a worker does not reserve SSD extents,
pinned memory, or CUDA destination pages.

Even with one worker, requests may compete for SSD/PCIe bandwidth and staging
capacity. Worker selection alone cannot order those physical operations or
establish a CUDA completion fence. Dynamo also developed tiered block management,
but [KVBM is deprecated in v1.5.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0)
with removal targeted for v1.6.0. Do not introduce a new KVBM dependency.

The separate [KV Cache Runner (KVCR)](https://github.com/ai-dynamo/kvcr/blob/317e62f301300c101bf9c911e020fb89b35f1583/README.md)
is experimental: its initial release supports cross-node DRAM sharing and
local disk caching for vLLM; SGLang integration remains future work. Its
[design](https://github.com/ai-dynamo/kvcr/blob/317e62f301300c101bf9c911e020fb89b35f1583/docs/design_overview.md)
separates discovery, fetch/delivery, claims and release, with router source
hints and NIXL movement. Borrow those ownership contracts. A submitted hint
does not currently initiate movement or pin a remote source; proactive hint
fetching is planned. KVCR is not a KVBM replacement or a qualified Rust
dependency for OrbitKV's two engine adapters.

Unified transfer interfaces also exist. NIXL v1.4.1's
[backend contract](https://github.com/ai-dynamo/nixl/blob/v1.4.1/docs/BackendGuide.md)
includes optional `estimateXferCost`, returning duration, error margin and
estimation method; [UCX implements it](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/ucx/ucx_backend.cpp).
NIXL also has a [Mooncake TE plugin](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/mooncake/README.md),
marked Preview. Its audited
[backend class](https://github.com/ai-dynamo/nixl/blob/v1.4.1/src/plugins/mooncake/mooncake_backend.h)
does not override the cost estimator. Choosing Mooncake therefore does not
exclude NIXL, but an API declaration is not evidence of estimates on every route.

NIXL can be embedded independently of Dynamo. Its published
[Rust bindings](https://github.com/ai-dynamo/nixl/blob/v1.4.1/README.md#rust-bindings)
provide `nixl-sys`; a native integration need not route operations through
Python. Qualify the real native library and selected plugins at runtime, not
just a successful binding build. Backend availability still depends on the
build, registered memory types and hardware.

The backend guide describes selection by eligible memory registrations and
backend preference when several plugins match. An optional transfer estimator
does not establish automatic end-to-end cache optimization. Keep cache source,
format, required-range and staging decisions with the Manager; leave backend
transport mechanics with NIXL or TE. Compare direct TE with NIXL plus the same
TE backend separately from comparisons that also change the backend to UCX.

Before building a broader transfer interface, evaluate NIXL with Mooncake in
an isolated prototype. Qualify native integration, pinned TE compatibility,
registration lifetime, cancellation/completion, codec staging and overhead.
Keep the current direct TE path until that comparison justifies a change.
Backend estimates can seed individual transfer costs; OrbitKV must still
account for owner preparation, queues, decode, required ranges and the actual
consumer fence. Missing estimates remain unknown, not zero-cost routes.

The later routing milestone will first integrate the upstream default selector,
not copy its formula into a competing implementation. Prefer a pinned Rust
dependency in an optional routing component owned by `orbitkv-server`; keep
the GPU/SSD core independent of it. Deployments that already use Dynamo may
retain its selection service instead. Choose one request-selection owner in
each deployment. Neither mode becomes a prerequisite for a single-node cache.

The future integration must supply real events and load lifecycle, rather
than assuming an existing OrbitKV block hash is a Dynamo routing hash:

- Partition by computation identity, tokenizer/hash scheme, block size, and
  engine/rank compatibility. Derive matching query and event hashes using the
  same scheme; do not reinterpret OrbitKV's 32-byte keys as router integers.
- Publish HBM events from the engine and DRAM/SSD residency changes from the
  Manager, with epochs, ordering, removals, and inventory recovery. Announce an
  SSD replica after successful writing. A pending read is not a resident hit.
- Resolve overlapping tier coverage before scoring. Advertise lower-tier
  credit only for adapter paths that can actually restore that state.
- Book load on assignment; update it on prefill completion, cancellation,
  generation progress, and request completion as required by the selected
  upstream model. Reconcile worker restarts and abandoned bookings.
- Revalidate state and obtain transfer leases at the selected Manager. Router
  events and cost estimates may be stale; they never authorize skipping prefill.

Start with upstream tier weights as a routing baseline. Later, feed calibrated
Manager estimates into the upstream `WorkerScorer` interface only if measurements
justify it. The policy API exists; carrying OrbitKV's estimates into its inputs
still needs an integration contract. Normalize units and avoid counting cache
savings twice: adding a
millisecond transfer estimate directly to the default block score is invalid.
Keep estimates bounded and timestamped; export coarse summaries rather than
asking every Manager to reserve data for each candidate request.

## Research to borrow from

- [KVFlow (2025)](https://arxiv.org/abs/2507.07400) uses an agent execution graph
  and proximity to future steps to guide retention and CPU-to-GPU prefetch.
  Workflow hints are a useful extension when applications can provide them.
- [Marconi (MLSys 2025)](https://arxiv.org/abs/2411.19379) handles recurrent
  state constraints and values reuse by compute savings relative to memory.
  This motivates boundary-aware admission rather than treating every token
  page as an interchangeable recovery point.
- [ECHO (OSDI 2026)](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)
  overlaps lossless recall with indexer computation for native sparse attention
  using graph-compatible GPU mechanisms. Its model-specific opportunity does
  not justify dropping dense Qwen3 attention KV or assuming unchanged accuracy.
- [Dynamo's routing model](https://github.com/ai-dynamo/dynamo/blob/v1.5.0/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/routing-concepts.md)
  combines cache locality with active work. Reuse its implementation at the
  later routing milestone, with the ownership boundary described above.

These mechanisms are prior work, not OrbitKV inventions. The proposed direction
combines explicit demand, legal recovery boundaries, resource scheduling, and
measured restore-versus-recompute decisions. Performance and novelty claims
require comparisons against compatible implementations on the same workloads.

## Delivery and qualification

The [completion plan](completion-plan.md) owns the execution sequence. Its S3
covers page generations and drain; S4 covers storage/copy overlap; S5 covers
released engine lifecycles; S6 covers consumed physical choices and optional
router work; S7 covers semantic retention. The former P0–P6/R1 chronology is
retained in [the historical design](https://github.com/OrbitInfra/OrbitKV/blob/ec3add9bcaf91a0171685ba42b5ae8330076c397/docs/state-planning.md#implementation-sequence).
Current pending-query ownership is documented in [transport](transport.md),
SGLang admission in [adapters](adapters.md), preparation in
[request preparation](request-preparation.md), and layer dependencies in
[engine-local restore](engine-local-restore.md). Completed mechanisms must not
be reimplemented from an old stage number.

## Evaluation contract

Keep programs in `benches/` and results in external artifact storage; keep correctness and failure tests in
the existing Python integration/E2E and Rust test suites. Extend those owners
rather than duplicating launchers or adding forwarding packages.

Preserve the original serial forced-tier test as a regression baseline. Add
natural DRAM pressure, concurrency 1/4/8/16, partial-prefix reuse, cancellation,
and mixed SSD reads/writes. Compare compatible native-engine, LMCache, and
FlexKV SSD configurations under the same model revision, token/page budgets,
storage path and I/O mode. Record unavailable combinations as such.

Measure latency from request arrival, including scheduler waiting. Do not hide
prefetch time by resetting the timer at admission. Record application hint
lead time separately. Predeclare sample sizes, SLOs, and regression budgets;
five samples per cell do not qualify tail latency. Preserve unsuccessful reads
as misses. An oracle with perfect future knowledge is an upper bound, separate
from deployable policies. Run independent ablations for demand timing, overlap,
and admission, then repeat the combined policy on held-out workloads.
