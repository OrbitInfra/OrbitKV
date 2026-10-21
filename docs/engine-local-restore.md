# Engine-local GPU restore

## Implemented scope

Unencoded DRAM Restore now submits CUDA work inside the inference process.
The Manager prepares a pointer-free plan and retains the exact source
allocations and query byte reservations. A native engine worker imports the
shared payload arenas, copies into the engine's retained tensors, and reports
whole-operation completion after its copy stream drains. Both vLLM and SGLang
hand their tensors and destination readiness stream to this worker.

SSD, encoded, and mixed-source restores retain the Manager's existing physical
workers. This choice follows the prepared sources; a failed local raw restore
is never retried through a second executor. Publish and GPU storage encoding
also retain Manager workers and CUDA IPC tensor registration.

Each operation targets one GPU. Raw resident Restore records persistent engine
CUDA events after each registered layer's final copy. The consuming stream may
start that layer while later copies continue. All source, destination, query and
event owners remain retained through a separate whole-operation drain. Large raw
plans execute in bounded parts under that same ownership fence. Manager-owned
SSD/codec execution still publishes consumer events only after its final drain.
The frozen single-GPU native/fault bundle passed **38 tests**, with **30 cuFile cases skipped** because that
configuration was not selected. This includes the new local Restore lifecycle
cases; the caller CUDA-context preservation test also passed. See the
[recorded artifacts and scoped results](fault-qualification.md#engine-local-raw-restore-gates).
The same production build subsequently passed single-H20 Qwen3-8B DRAM serving
correctness in vLLM 0.29.0 and SGLang 0.5.20, including engine-restart reuse.
See the [serving configuration and evidence](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md#engine-local-restore-serving-qualification).
Multiple-GPU, huge-page, sustained allocator-pressure, and exhaustive graph-mode
qualification remain open. Correctness alone does not establish a serving speedup.

## Connected execution path

1. `register_context_batch(..., tensors=...)` supplies actual tensor/exporter
   objects as well as the CUDA IPC registrations still needed by Publish and
   Manager physical routes. The native client validates destination geometry
   and keeps these objects alive.
2. A successful GPU registration reply transfers the Manager's payload arena
   FDs over authenticated UDS with `SCM_RIGHTS`. The engine independently maps
   each size-sealed memfd and registers that mapping with its CUDA primary
   context. Registration is per arena and GPU binding, not per restored block.
3. `start_restore(..., ready_stream=...)` reserves native operation ownership
   and queries the supplied engine stream. An idle stream already proves prior
   destination users completed. A busy stream records and synchronizes the
   executor-owned reusable event before preparation; readiness does not yet
   overlap with the copy stream.
4. The client reserves a shared operation identity before sending the Restore
   descriptor. The Manager authenticates and claims it before decoding or
   consuming leases. Validation compiles a bounded raw plan before lease
   consumption, then moves the selected source owners into `RawRestoreGrant`.
   Within each layer, preparation sorts destination pages, traverses K and V
   separately, and merges consecutive source/destination ranges only within
   the same arena and allocation identity/bounds. Global destination-overlap
   validation still runs before consuming leases. This reduces descriptors
   before encoding and shared-memory transport. Raw plan version 2 marks each
   layer's final range across all parts; cache schema 10 rejects mismatched clients.
5. The Manager installs the grant owner before publishing `Granted`. The native
   worker wins `Granted → Active`, copies the plan locally, validates source
   and destination ranges, and submits through the existing memcpy or mapped
   memory kernel backend. Direct DMA groups contiguous ranges or explicit
   equal-width strided rows within one CUDA host registration and GPU allocation.
   Each source allocation retains its own generation, checked bounds and lease;
   no gaps or unlisted rows become accessible through coalescing.
6. Each layer event is recorded after all its required ranges. The native
   `wait_restore_enqueued` call returns once those event records have been
   submitted, allowing the engine to enqueue waits before each consumer.
   It does not report DMA completion. The worker drains all accepted copy work,
   including partial failures, before publishing `Drained` and the final result
   consumed by `poll_restore`/`wait_restore`. Neither consumer readiness nor the
   local result waits for Manager source reaping.
7. A separate retirement loop holds the operation identity and session mapping
   until the Manager releases sources and publishes `Reaped`. The engine then
   acknowledges the record so its slot can serve a new generation.

The concrete owners are [core preparation](../crates/orbitkv-core/src/engine/restore.rs),
[the local CUDA executor](../crates/orbitkv-core/src/transfer/local.rs),
[the native operation worker](../python/src/local_restore.rs), and
[Manager grant retirement](../crates/orbitkv-server/src/endpoint/restore.rs).
Python supplies framework callbacks, tensor objects, layouts, and stream
handles. Rust owns grant state, submission, waiting, and reclamation; blocking
native calls release the GIL.

## DMA and mapped-host copy kernels

Both raw host-copy backends move bytes from pinned DRAM directly into the final
engine KV allocation. The `direct` backend submits CUDA memcpy operations for
coalesced ranges or constant-pitch rows, normally serviced by GPU copy engines.
The `kernel` backend runs SM threads that read the GPU-visible mapping of host
DRAM and store into those destination pages. A mapped host pointer remains backed
by DRAM; obtaining its device address does not allocate a second payload in HBM.
See [CUDA mapped host memory](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-best-practices-guide/index.html#zero-copy).

The kernel uploads a small descriptor table containing destination, source and
length (24 bytes per fragment), not a full KV staging image. Both backends still
move payload over the host/device interconnect. Kernel submission can reduce
residual fragmented-copy overhead, but consumes SM resources also used by model
execution. Compare against the actual coalescing DMA backend under serving load;
a kernel microbenchmark alone cannot establish an application improvement.

This raw host path is separate from SSD cuFile staging/scatter and storage
encoding, which have their own buffers, transformations and qualification gates.
CUDA copy code remains in the maintained standalone `transfer/kernel.cu` file.

## Resource ownership

| Resource | Owner | Condition for release or reuse |
| --- | --- | --- |
| Payload backing | Manager and each importing engine's independent FD/mapping | Each process has drained its accesses before dropping its mapping |
| CUDA host registration | Process using that mapping | All GPU accesses through that registration have drained |
| Allocated source range | Manager grant holding the selected sealed block/allocation | Revoked before engine claim, or matching engine drain evidence accepted |
| Query byte reservation | Manager grant after lease consumption | Same terminal condition as the source, including quarantine |
| Destination tensor allocation | Native worker retaining the real exporter | Every accepted operation using the binding has drained |
| Logical destination page IDs | Framework scheduler/connector | Restore is terminal and the engine's own page-use rules permit reassignment |
| Shared plan bytes | Manager plan bank | Engine copied them locally, or grant was revoked before claim |
| Grant record | Retained session mapping | Manager reaped source owners and engine acknowledged retirement |
| User-visible result | Native operation | Matching result was consumed; dropping its handle does not cancel the work |

A mapped memfd keeps the backing object alive after Manager death. It does not
prevent a live Manager from recycling an offset. The retained allocation owner
prevents that reuse. Conversely, a Manager source reference does not keep an
engine tensor or CUDA registration alive. Both processes retain real owners.

Tensor ownership also does not reserve logical page IDs inside a tensor. The
connectors must hold those assignments until local DMA completion, including
when a caller times out or drops its Python handle.

Repeated registration of the same instance/rank/device on one native client is
rejected. Unregister and close stop concurrent admission through the native
registration lock and wait for accepted operations to drain before releasing
bindings. They do not replace a live binding in place. A new registration after
unregister creates fresh owners.

## Payload arenas and range validation

Every pinned pool shard uses a size-sealed memfd and `MAP_SHARED`. The Manager
first-touches pages according to NUMA policy before its CUDA registration.
Regular and reserved huge pages use the same backing implementation; the old
private mapping, `cudaHostAlloc` pool branch, and `cpu_readable` selector are gone.

The importer checks the advertised size, required size seals, arena identity,
and CUDA context. It derives its host and device-visible pointers from its own
mapping. Neither pointer is copied from the Manager process. This runs within
the existing same-UID trusted-engine boundary: possession of an arena FD exposes
that arena, so range checks are protocol validation rather than tenant isolation.

Each wire copy contains:

- A process-scoped arena ID and a monotonically increasing allocation ID within
  that arena; allocation IDs are not reused when allocator offsets are reused.
- Allocation offset and length, plus the source's absolute arena offset and
  copy length. Manager preparation checks the exact source segment and actual
  allocation; the importer checks `source ⊆ allocation ⊆ arena` again.
- The registered layer name and destination-relative byte offset. The executor
  resolves the name against its retained local tensor and validates the range.

The session epoch, client token, and operation generation bind the plan to its
session. A Manager restart creates new backing objects and a new session; a
surviving mapping is never reopened as the replacement Manager's allocator.
There is no eventually consistent allocation directory on the DMA path.

Local grants and Manager workers share the per-GPU limit of 128 admitted restores.
A local grant holds its permit through source retirement, including quarantine;
unregistering an instance cannot reset that budget. Manager-executed restores
retain their `CacheRestore` completion observations. Local raw grants report
caller-to-drain durations as `engine_local_restore`; this separate estimate
cannot select against Manager-preparation or P/D intervals. Manager reaping
never substitutes for engine readiness. See the
[completion evidence contract](engine-local-restore.md#engine-local-completion-evidence).


[Affine geometry](../crates/orbitkv-core/src/transfer/layout.rs) remains the
shared implementation for contiguous and split layouts. Preparation resolves
storage groups, TP slots, page-first offsets, source representation, and
registered destinations once. The local executor validates the resulting byte
ranges and sorts copies by destination address, rejecting overlaps before
submission. Raw plans contain no Manager virtual addresses.

## Shared grant protocol

Clients and Managers must be rebuilt together. Bootstrap version **7**,
channel ABI **11**, and lifecycle version **4** reject older peers. Bootstrap
transfers five FDs: descriptor memfd, grant memfd, Manager-to-engine Restore
eventfd, engine-to-Manager retirement eventfd, and Publish reply eventfd.
Payload FDs arrive only with GPU registration replies. The native worker also
owns an engine-local completion eventfd, separate from Manager notifications.

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

A lost or malformed descriptor ACK retains the already-reserved handle once
the Manager has claimed preparation. Requests are not retransmitted. Shared
results remain accessible after descriptor-channel closure. Handles also keep
their native client issuer, so a replacement client cannot consume an old
handle even if an operator reuses a configured epoch.

The claim/revoke CAS is the authority boundary. Once `Active` is visible, TTL,
UDS closure, or process exit cannot prove whether CUDA work was submitted.
Manager cleanup retains active source owners; a valid `Drained` record is the
release proof. After copying a claimed plan, the engine publishes
`plan_consumed`; this frees plan-bank capacity independently of source lifetime.

Notifications are hints. Atomic generation-tagged records are authoritative.
The Manager reads a bounded dirty bitset and is woken by eventfd; maintenance
also services pending retirements. The bitset cannot overflow and does not
require scanning all 1024 records during normal operation. Source retirement
runs independently of the descriptor dispatcher and Python result consumption.

### Bounds and current limits

Each session has 1024 records of 192 bytes and a shared 1 MiB plan bank. Bounded
error text occupies at most 88 bytes inside a record. The Manager allows at most
64 live or retained session mappings; unresolved disconnected sessions count
against that limit. The native pending-operation bound is 1024, and payload
arena imports are capped at 64 per client. The metadata maps total about
72.25 MiB at 64 sessions, excluding descriptor arenas and retained payloads.

A full shared plan bank defers already-prepared grants until plan consumption
returns capacity. Source references and query reservations remain held while
they wait. Quarantined grants likewise retain their source-byte and record
credits; dropping a session does not make those resources reusable.

The 1 MiB limit applies to each encoded part **after allocation-aware
compaction**. Larger fragmented plans are partitioned automatically. A single
operation is limited to 32 MiB of metadata, checked before lease consumption;
each session reserves at most 64 MiB for prepared plans. A session budget
rejection consumes the admitted lease but releases its sources without DMA.
All destinations are validated together before any part is published. One
operation ID retains source allocations, query reservations, metadata credits
and GPU tensor owners until every part drains or the operation fails. The
Manager acknowledges nonfinal `PartDrained` states once and queues the next
part fairly; only the final `Drained` state releases whole-operation owners.
Session loss can revoke an unclaimed part, but active DMA remains quarantined.
The shared-grant schema is version 5; no earlier decoder is retained.

## CUDA readiness, failures, and shutdown

`ready_stream` must identify the actual engine stream whose earlier users of
the destination pages need to finish. Native code validates its CUDA context
and queries completion in the tensor's primary context. An idle stream needs
no new GPU event; a busy stream records and waits on the retained local event.
Query errors reject preparation. Calls serialize access to this event, and each
busy-stream call records a fresh completion point. A background thread's default
stream does not substitute for the engine dependency. Publish has its own
producer-stream fence and keeps the existing Manager-side transfer owner.

The local executor currently uses one copy stream and a whole-operation drain.
After any partial enqueue error it still synchronizes that stream before
returning a reusable-page result. If completion cannot be established, the
engine process aborts; it does not report success or manufacture drain evidence.
The live Manager then retains any unresolved active grants.

| Failure | Implemented ownership rule |
| --- | --- |
| Preparation or local validation rejects | Release only after proving no local copy was submitted |
| Lost submission ACK | Keep known operation identity; read authoritative grant state |
| Partial CUDA enqueue failure | Keep imported arenas and destination tensors until accepted copies drain |
| Wait timeout or dropped handle | Native worker continues to own and complete the operation |
| UDS disconnect with engine alive | Stop descriptor admission; existing local operations can drain through retained shared records |
| Engine exits before claim | Manager may revoke an unclaimed grant and release its sources |
| Engine exits after claim without drain evidence | Retain sources and session credits in bounded quarantine |
| Manager exits during claimed local copy | Engine's independent mapping and registration survive; report only after local drain |
| Orderly unregister or close | Wait for accepted operations before dropping tensor and CUDA owners |

A pidfd or broken socket is process/control evidence, not a CUDA fence. There
is no timeout-based source reuse after engine death. Quarantine can exhaust
admission and requires operational recovery; it deliberately preserves
allocations when completion is unknown. If the retirement task itself is
cancelled or panics, its source-owner destructor applies the same conservative
retention rule.

## Layer readiness and framework consumption

`start_restore(..., layer_events=[(registered_name, event), ...])` retains the
actual event objects alongside the operation. Events must be live and distinct
for registered layers; invalid bindings are rejected before lease consumption.
A layer with no required bytes receives a fresh event generation too.
The existing `wait_restore` remains the only whole-operation completion call.
Event publication and intermediate part acknowledgements never release a source
allocation or acknowledge a framework's destination pages.

| Engine | Admission and first use | Completion and graph behavior |
| --- | --- | --- |
| vLLM 0.30.0 | A synchronous external hit enters the consuming forward. `start_load_kv` submits one batched Restore, waits for event publication, and attention callbacks wait on their registered layer. Recurrent operators have no layer callback, so their events are waited before the runner migrates checkpoint state. | `wait_for_save` consumes the final Restore outcome after forward submission. Piecewise graphs keep per-layer attention waits. Full-graph replay links all required layer events on its entry stream; Decode steps with no Restore keep their original graph path. |
| SGLang 0.5.20 | Persistent external events are installed in its GPU pools before the first graph capture. `set_consumer` supplies the actual forward stream and waits for new event records; pool accessors enqueue per-layer waits. | Captured external wait nodes execute on every replay. The background native operation continues to final drain before the completion queue acknowledges the batch. Cancellation of already-published tree destinations completes their Restore before releasing references. |

The pinned vLLM GPU runner migrates recurrent state before its public
`pre_forward` connector callback. OrbitKV installs one scoped `update_requests`
boundary: drain preempted saves before page zeroing, let the runner initialize
pages and apply copy-on-write, then submit synchronous Restore and recurrent
waits before `preprocess_state`. The normal callback binds the same metadata;
its repeat load does not resubmit or consume the lease twice. The boundary
preserves `MultiConnector` child-to-metadata ordering, so composing cache reuse
with another connector does not bypass recurrent readiness. Other connectors
and disabled profiling runs retain their original runner behavior. This adapter
boundary must be requalified when updating vLLM; it does not modify the engine's
state migration algorithm. Recurrent Restore requires the V2 runner and rejects
an explicitly selected V1 runner at startup. Correctness controls retain the
same native graph configuration as OrbitKV.
The pinned V2 full-graph branch supplies no attention metadata to the connector;
that boundary links all Restore events before replay, whose captured body has no
Python layer callbacks. It does not force ordinary Decode into piecewise graphs.

SGLang admits a batch only when its consuming forward supplies the readiness
stream. This prevents a later batch from re-recording shared events before an
earlier graph replay has used them. Initial events are recorded once so a
cold forward or capture has valid dependencies even with no pending Restore.
Within each registered pool, Restore orders buffers by their consuming layer
(K0, V0, K1, V1 for split attention). Registration and storage slot identities
remain component-major. This lets the first layer consume both components
without waiting for every later K buffer.
The relevant CUDA semantics are [event record and stream wait](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html)
and [PyTorch external graph events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).

A metadata part may end within a layer. Only its last required range can publish
that layer's event. Publication currently waits until the final part is submitted,
and preceding parts drain before the next is acquired; a highly fragmented plan
therefore has less opportunity to overlap than a single part. Packed cross-layer
bindings also share one event. Splitting physical buffers, pipelining parts, and
moving recurrent waits into actual operators require separate ownership and
performance gates. No source is released early when page-first allocations are
shared across groups.

Once a forward has consumed a layer dependency, any final transfer error is an
execution failure. It cannot transparently recompute pages already used by that
forward. A timeout or missing acknowledgement retains destinations until teardown.

Idle-stream queries, reusable destination-readiness events, and plan compaction
reduce the existing handoff overhead; see [matched measurements](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md).
This layer implementation does not remove the destination's previous-user fence
or prove a serving speedup. Compare complete requests under matched graph modes,
HBM budgets and workload pressure before making a performance claim.

## Remaining execution work

SSD host materialization, codec staging/decode, and direct cuFile execution may
later move through the grant contract. Each migration must move its real I/O,
file extent, scratch, destination, and completion owners. Current Manager
workers remain active consumers of those responsibilities. Remote TENT READ
first materializes and validates local residency; an unencoded resident result
then uses the same local grant path. Remote source authorization and completion
retain their existing independent export protocol.

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
[30-cohort serving matrix](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md#repeated-serving-comparison-after-layer-readiness)
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
budgets. The [communication measurements](https://github.com/OrbitInfra/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md) record
matched microbenchmarks; a metadata speedup alone is not a serving improvement.

## Engine-local completion evidence

The native worker carries six cumulative nanosecond offsets from one engine
`Instant`: readiness finished, dispatched, dequeued, grant claimed, CUDA enqueue
returned, and GPU drain observed. The first interval includes native argument
conversion, client/executor locking and destination readiness. Dispatch reaches
native job construction; the queue interval includes enqueue handoff and worker
delay. Grant wait includes worker scheduling and plan consumption.
Submission includes validation and descriptor compilation. These are host
observations, not GPU kernel timestamps.

The existing session/operation-fenced completion record transports the bounded
report with `Active -> Drained`. The Manager accepts it when consuming the source
owner, once, before reaping. Invalid, absent or over-one-day reports are ignored
without blocking drain, reaping or record reuse. Failed and never-submitted
operations do not train success estimates. Late Manager observation cannot
extend engine-ready latency, and no cross-process timestamp subtraction is used.

`ORBITKV_COST_OBSERVATIONS=1` on both processes records the independent
`engine_local_restore` cost key. It is deliberately excluded from route selection:
Manager `cache_restore` starts at preparation, and P/D starts at handoff enqueue.
A shared target GPU does not make these start boundaries comparable.

`ORBITKV_TRACE_TRANSFERS=1` additionally emits `local_restore_complete` from the
Manager and `local_restore_observed` from native result consumption. The common
Rust tracer serves both processes. The benchmark parser joins those records to
existing connector request links, counts each physical batch once, and reports
native stage and consumer-wait quantiles separately from framework first-use
callbacks. Local completions do not require a Manager-to-engine notification.
Tracing and cost collection are opt-in; the disabled path takes no stage clocks.
