# Engine-local GPU restore

## Implemented scope

Unencoded resident DRAM Restore runs in the inference process through OrbiTE
`local_copy`. The Manager retains source allocations and query reservations;
the native client retains imported arenas, destination tensors and completions.
SSD, encoded and mixed-source routes keep Manager workers. A failed local route
is never retried through a second executor.

Each operation targets one GPU. Per-layer events permit consumption after that
layer's copies; whole-operation drain separately protects all source/destination
owners. Manager SSD/codec routes publish consumer events after their final drain.
All current OrbiTE GPU and installed profiles require re-qualification; the
historical gates below do not carry over. See the [completion plan](completion-plan.md).

## Connected execution path

1. `register_context_batch(..., tensors=...)` retains real exporters and validates
   geometry. The registration reply transfers size-sealed payload arena FDs over
   authenticated UDS. The client maps and registers them once per GPU binding.
2. `start_restore(..., ready_stream=...)` retains the destination dependency.
   An idle stream needs no new event; a busy stream records and waits on a fresh
   executor event before preparation.
3. The client reserves a shared operation identity. The Manager claims it before
   consuming leases, validates all destinations, compacts ranges within the same
   arena/allocation bounds and installs `RawRestoreGrant` before `Granted`.
4. The native worker wins `Granted → Active`, copies and validates the pointer-free
   plan, then submits checked contiguous/strided descriptors to OrbiTE.
5. A layer event follows its last required range. `wait_restore_enqueued` reports
   event publication, not drain. `poll_restore` / `wait_restore` report the final
   local result only after accepted copies drain, including partial failures.
6. The Manager independently reaps source owners. Client acknowledgement then
   permits the shared record to serve a new generation. Source reaping is off the
   successful consumer critical path.

Owners live in [Core preparation](../crates/orbitkv-core/src/engine/restore.rs),
[local execution](../crates/orbitkv-core/src/transfer/local.rs),
[the native worker](../python/src/local_restore.rs) and
[grant retirement](../crates/orbitkv-server/src/endpoint/restore.rs).

## DMA and mapped-host copy kernels

`direct` and `kernel` select OrbiTE's local DMA and mapped-host kernel copies.
Both move payload over the host/device interconnect. A mapped host address is
still backed by DRAM; kernel descriptors are not an HBM payload image. Kernel
copies use SM resources shared with inference, so component speed does not
establish serving benefit. OrbitKV contains no payload copy kernel or fallback.
See [local-copy bounds](transport.md#local-gpudram-copies) and
[historical kernel evidence](kernel-optimization.md).

## Resource ownership

| Resource | Owner | Release or reuse condition |
| --- | --- | --- |
| Payload backing | Each process's independent FD/mapping | Its accesses drain |
| Host registration | Process using the mapping | All accesses through that registration drain |
| Source allocation and query bytes | Manager grant | Revoke before claim, or matching physical drain |
| Destination tensor | Native worker's real exporter owner | All accepted operations drain |
| Logical page assignment | Framework scheduler/connector | Terminal Restore plus the framework's page-use rules |
| Plan bank | Manager | Plan consumed locally or revoked before claim |
| Grant record | Retained session mapping | Sources reaped and retirement acknowledged |
| User result | Native operation | Matched consumption; dropping the handle cannot cancel work |

A mapped memfd survives Manager death but cannot prevent a live Manager from
recycling an offset: the retained allocation does that. A source reference cannot
keep a destination tensor alive. Both sides retain real owners. Tensor lifetime
also does not reserve page IDs; adapters must hold those assignments through drain.

Repeated registration of a live binding is rejected. Unregister/close fences
admission and waits for accepted operations before releasing bindings. New
registration creates fresh owners, never replaces a live binding in place.

## Payload arenas and range validation

Pinned shards use size-sealed `MAP_SHARED` memfds, including reserved huge-page
backing. The Manager applies NUMA policy before first touch and CUDA registration.
Import validates seals, size, arena identity and context, then derives pointers
from the importing process's own mapping. This is a same-UID trusted boundary;
possession of an FD exposes the arena and does not provide tenant isolation.

Each copy binds an arena ID, monotonically increasing allocation ID, allocation
bounds, source subrange, registered layer and destination-relative offset.
Both sides check `source ⊆ allocation ⊆ arena`; the executor validates destination
bounds and rejects overlaps. Session epoch/token and operation generation fence
reuse. Restart creates new backing and a new session; no Manager pointer or
allocation-directory lookup appears on the copy path.

Local grants and Manager workers share 128 admitted restores per GPU. The local
grant keeps its permit through retirement, including quarantine. Unregister cannot
reset it. Geometry is resolved through the shared [layout owner](../crates/orbitkv-core/src/transfer/layout.rs).
Local `engine_local_restore` observations and Manager `cache_restore` have different
start boundaries and cannot select against each other merely because targets match.

## Shared grant protocol

Matching clients and Managers use bootstrap **7**, channel ABI **12**, lifecycle
**13**, cache schema **10** and shared-grant schema **5**. See
[transport versions](transport.md#bootstrap-registration-and-versions) for FDs and
registration. Lifecycle 13 carries no selected-engine field.

| Transition | Writer | Meaning |
| --- | --- | --- |
| Free/Acknowledged → Reserved | Engine | Choose operation identity before sending preparation |
| Reserved → Preparing | Manager | Deduplicate admission before lease consumption |
| Reserved → Acknowledged | Engine | Cancellation proves preparation never started |
| Preparing → CancelRequested | Engine | Request cancellation after ambiguous submission |
| Preparing → Granted | Manager | Source owner installed and immutable plan published |
| Preparing/CancelRequested → Reaped | Manager | Rejected or cancelled preparation released its sources |
| Granted → Active | Engine | Claim before reading plan or submitting CUDA |
| Granted → Revoked | Manager | Competing claim CAS proves engine submission is forbidden |
| Active → Drained | Engine | No-submit proof or all accepted local copies drained |
| Drained/Revoked → Reaped | Manager | Source owners and query credits released |
| Reaped → Acknowledged | Engine | Record can be reused with a newer generation |
| Preparing/CancelRequested → Managed → Reaped | Manager | Selected SSD/codec worker owns execution and publishes its drained outcome |

A lost or malformed ACK retains the already-reserved identity after preparation
was claimed. Requests are not retransmitted. Handles retain their client issuer
and shared mappings after descriptor-channel closure.

Claim/revoke CAS is the authority boundary. Only the executor that won the claim
can prove no submission or physical drain. `Active` observed by another caller,
TTL, UDS closure and process exit cannot do so. `plan_consumed` frees metadata
capacity independently of source lifetime. Generation-tagged records are
authoritative; a bounded dirty bitset and eventfd drive retirement.

### Bounds and current limits

| Resource | Limit |
| --- | --- |
| Grant records per session | 1,024 × 192 bytes; error text at most 88 bytes |
| Shared plan bank | 1 MiB; full banks defer prepared grants |
| Live or retained sessions | 64; roughly 72.25 MiB grant/plan metadata, excluding descriptors/payloads |
| Native pending operations / imported arenas | 1,024 / 64 per client |
| Metadata per operation / prepared metadata per session | 32 MiB / 64 MiB |

Plans larger than the bank are partitioned after allocation-aware compaction.
All destinations validate before any part is published. One operation retains
sources, query credits and tensors through every part; nonfinal `PartDrained`
acknowledgements queue the next part once. Only final drain releases the operation.
Session loss can revoke an unclaimed part, while active work stays quarantined.

## CUDA readiness, failures, and shutdown

`ready_stream` identifies the actual prior user of the destination pages. A
background thread's default stream cannot substitute. Context/query errors reject
preparation. The reusable readiness event is serialized and refreshed for each
busy-stream call. Publish has its own producer fence.

| Failure | Ownership rule |
| --- | --- |
| Preparation or validation rejection | Release only with proof of no submission |
| Lost ACK or wake | Keep known identity; inspect authoritative grant state |
| Partial submission | Retain imported arenas and tensors through accepted-copy drain |
| Wait timeout or dropped handle | Native worker still owns the operation |
| UDS disconnect | Fence new descriptors; retained records can still complete |
| Engine exits before claim | Manager may revoke unclaimed sources |
| Engine exits after claim | Quarantine sources/session credits without a drain receipt |
| Manager exits during local copy | Independent imports survive; local result still requires drain |
| Unregister/close | Fence admission and drain before releasing bindings |

Unknown completion cannot report a reusable page. Failed native cleanup in Drop
follows the [fail-closed policy](transport.md#native-cleanup-at-shutdown).
A pidfd is process evidence, not a CUDA fence; quarantine may exhaust admission.
Retirement-owner cancellation applies the same conservative retention rule.

## Layer readiness and framework consumption

`layer_events=[(registered_name, event), ...]` retains actual distinct CUDA events
for registered layers. Validation precedes lease consumption. Even a layer with
no bytes receives a fresh generation. Event publication and intermediate part
ACKs never release source allocations or logical GPU pages.

The selected vLLM profile uses V1 and one attention group; V2/recurrent serving
is rejected. Attention callbacks consume layer dependencies, with supported
full-graph paths using entry waits. SGLang installs external events before graph
capture, admits load when the consuming forward supplies its stream and uses
per-layer waits during replay. Split K/V copies follow consumer-layer order;
registration/storage identity remains component-major. Exact released hooks and
unsupported profiles are in [adapter contracts](adapters.md).

A multipart plan publishes a layer event only after its last required range.
Earlier parts drain before successors and publication waits until the final part
is submitted, limiting overlap for fragmented plans. Packed cross-layer bindings
share an event. Once a forward consumes a dependency, a final copy failure is an
execution failure; pages already read cannot be transparently recomputed.

## Remaining execution work

Manager workers still own SSD/codec source leases, decode compute and completion;
OrbiTE moves their bytes. Demand peer Restore goes directly to GPU, while
Prepare/Warmup fills DRAM. See [peer Restore](transport.md#peer-restore-into-gpu).
Changes to early consumption, page generations and fault reclamation require the
[completion plan's gates](completion-plan.md); this page adds no delivery queue.

## Layer readiness qualification, 2026-09-29

The H20 source build passes 410 source-only Python tests, 297 Rust core tests
(18 hardware-specific cases ignored), 60 channel tests (one ignored), and strict
workspace Clippy. The native GPU/adapter suite passes 47 tests, including CUDA
event timestamps that demonstrate the first consumer runs before the final H2D
copy completes, repeated external-event graph replay, exact destination bytes,
and retained ownership after submission/enqueue/drain errors. The test-hooks
process-fault suite passes 10 selected cases; serving uses the normal build.

Serving gates use vLLM 0.29.0 and SGLang 0.5.20 with Qwen3-8B and
Qwen3.5-0.8B. vLLM passes forced `FULL` replay for the dense model (six passed,
one hybrid-only skip), native graph defaults for the hybrid model on DRAM and
SSD (seven passed each), and a `MultiConnector` with a no-op child followed by
OrbitKV (seven passed). SGLang passes DRAM and SSD restart recovery for each
model with the final consumer-ordered buffer copies (two passed per model).
The no-op composition tests metadata routing and recurrent initialization; it
is not a P/D transfer qualification.

Raw logs and the source/binary manifest are under
`/root/orbitkv-artifacts/s1-evidence-20260929/legacy-results/runs/layered-restore-20260929/`. Reproduction switches are in
[the Python test guide](../python/tests/README.md). These gates establish
correctness and native copy/compute overlap. The separate
[30-cohort serving matrix](https://github.com/OrbitInfra/OrbitKV/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md#repeated-serving-comparison-after-layer-readiness)
passes exact output controls. It shows a vLLM improvement for the full scheduling
and copy increment; SGLang throughput does not improve over the old implementation.

## Qualification gates

The recorded first-slice gates passed for lost ACK/wake behavior, local
partial-enqueue drain, tensor retention after Python references are dropped,
nondefault-stream readiness, Manager death after claim/first enqueue, and
source quarantine after engine death following claim. CUDA context preservation
also passed. Exact artifact identities and scoped suite counts are in
[fault qualification](fault-qualification.md#engine-local-raw-restore-gates).

The broader gate matrix remains the qualification contract; a passed selected
case does not qualify every failure point or deployment in its row:

| Gate | Required evidence |
| --- | --- |
| Protocol | Claim/revoke race, cancellation, lost ACK/wake, stale/reused records, full plan/record budgets, and reconnect fencing |
| Shared payload | Independent process mappings and CUDA registrations, accurate GPU bytes, registration rollback, exporter death, and checked allocation bounds |
| Ownership | Query release, session cleanup, eviction, and allocator pressure cannot reuse active or quarantined allocations |
| Partial enqueue | Accepted CUDA work retains both source and destination owners through drain despite a later error |
| Process death | Engine death before/after claim and during copies; Manager death during a locally owned copy; only actual drain evidence releases active source credits |
| Layout | Contiguous, split K/V, MLA, fused/page-first placement, storage groups, TP slot selection, and destination overlap rejection |
| Lifecycle | Dropped handles/timeouts, repeated registration rejection, unregister/close drain, and fresh-session isolation |
| Serving | vLLM 0.29.0 and SGLang 0.5.20 correctness, restart reuse, hybrid errors, and every advertised eager/graph mode |
| Extended environments | Multiple GPUs, huge-page imports, NUMA policies, and supported container/deployment configurations |

Use deterministic hooks to identify claim, accepted-copy, and drain states.
The passed after-enqueue Manager-death case does not prove that hardware was
still copying at the precise SIGKILL instant. The engine-death case proves
quarantine after claim, not death during a proven in-flight copy. Serialization
tests and normal exporter teardown cannot fill those gaps. Single-GPU dense
Qwen3-8B DRAM serving and restart reuse now pass in both pinned engines;
the later layer-readiness gate above adds dense/hybrid DRAM/SSD and graph evidence.
Multiple GPUs, huge-page imports and sustained allocator pressure remain outside
those qualifications.
Record further results against exact binaries and configuration before
broadening those claims.

Measure preparation, queue delay, native submission, GPU drain, connector
observation, first engine use, and source-retirement lag separately. Compare
serving TTFT/ITL and goodput at equal CPU, host/HBM memory, model, and quality
budgets. The [communication measurements](https://github.com/OrbitInfra/OrbitKV/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md) record
matched microbenchmarks; a metadata speedup alone is not a serving improvement.

## Engine-local completion evidence

Six cumulative nanosecond offsets from one native `Instant` record readiness,
dispatch, dequeue, grant claim, enqueue return and GPU drain. They are host
observations, including conversion, locks, queueing and descriptor compilation,
not device kernel timestamps. The session/operation-fenced record accompanies
`Active → Drained` and is accepted once before source reaping. Invalid, missing or
over-one-day reports cannot block drain or reuse. Failed/no-submit operations do
not train success estimates.

Enable `ORBITKV_COST_OBSERVATIONS=1` in both processes for `engine_local_restore`.
Its caller-to-drain interval is excluded from comparisons with Manager preparation
or P/D handoff intervals. `ORBITKV_TRACE_TRANSFERS=1` additionally emits
`local_restore_complete` and `local_restore_observed`. The parser counts each
physical batch once and separates native stages, consumer wait and first use.
Disabled observation takes no stage clocks; source reaping never extends the
engine-ready latency. See [metrics](metrics.md#bounded-cost-observations).
