# OrbitKV transport architecture

> OrbiTE is the only transfer engine. All current GPU/RDMA/GDS and installed
> profiles require hardware re-qualification. Historical results below retain
> their original backends and revisions; they do not qualify this data path.

Prepared Save requests share the existing
native pending owner with submitted requests. Metadata stays charged until its
last owner drops, including caller-held completed handles; registration tensors
are retained through submission or proven never-submitted cancellation. The
same 1,024-operation/32-MiB process cap includes conservative merge scratch.
This metadata limit is separate from registered allocation and payload budgets.

This document separates the inference-to-Cache-Manager channel, remote data
movement, and replica discovery so none of them becomes an accidental second
source of KV truth. "Same-host" describes where the client connects; the
Cache Manager decides whether a requested block is resident in RAM, needs SSD
prefetch, or can be fetched from a peer through OrbiTE.

## Native cleanup at shutdown

Native cleanup inside `Drop` cannot return a recoverable error. If cleanup fails,
OrbitKV aborts rather than release memory a device might still touch. This covers
CUDA allocation frees, host unregister, and restoring the caller's CUDA context;
failed stream drain likewise cannot authorize reuse. A timeout, cancellation or
`Unquiesced` result remains an unknown drain and retains its owners.

The single cleanup exception is `cudaErrorCudartUnloading` from a pinned pool's
runtime unregister at process exit. CUDA is destroying its contexts, so no device
access can follow. OrbitKV logs once, skips `munmap`, and lets process exit reclaim
the mapping. Other unregister failures still abort.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local Cache Manager | iceoryx2 requests and shared grants | raw DRAM via engine-imported payload arenas; Publish/SSD/codec via Manager CUDA IPC bindings | raw local executor integrated; qualification gates tracked separately |
| local bootstrap and lifecycle | Unix socket with credentials and file-descriptor passing | descriptor/grant memfds, three eventfds, and GPU-registration payload arena FDs | implemented |
| Cache Manager to Cache Manager | OrbiTE P2P discovery and epoch fencing | OrbiTE READ/WRITE over TCP or RDMA | CPU/TCP CI evidence; GPU/RDMA profiles unqualified |
| replica directory | soft-state network API | no KV bytes | embedded fixed shards; replication planned |
| administration | HTTP | no KV bytes | existing |

Both framework adapters require `orbitkv-channel` for hot data operations. Every
inference process must connect to a Cache Manager on its own host; a missing
Unix socket fails fast.
Registration, health, session watching, and unregistration use the
authenticated bootstrap UDS for both adapters.

Registration returns an opaque native `GpuRegistration` containing the existing
InstanceContext generation, fixed client/rank/device identity and retained tensor
exporters. Publish accepts that object instead of a mutable instance/rank/device
lookup. Both adapters capture it when queuing their producing step. The native
client validates the original client and generation after acquiring its worker
lock, so a wait or late callback cannot bind a replacement registration. Each
Publish chunk carries the captured generation; the Core owner rechecks the same
retained instance before topology or GPU pointer binding. Matching lifecycle 13
and cache schema 10 artifacts remain required; retired schemas are rejected.

This is registration lifetime fencing, not allocation/page generation or a
physical termination receipt. Holding the registration also holds its exporters
after unregister; it does not reauthorize them. Workloads and adapters retire
their own references after normal cleanup and retain them if cleanup fails.
A separately queued reference keeps its exporters until that task retires. Restore retains its existing
query-lease checks and API. The earlier `03cbcebd` acceptance covers its snapshot
and Core/lifecycle controls. The producer-bound authority has bounded installed
registration acceptance;
matching real-model regression and physical lifetime remain separate gates in
[the completion plan](completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership).
CPU authority controls use an authenticated process fixture and an idle worker
with no CUDA executor; they do not register, compute or submit device DMA.

The official vLLM 0.31.0 `KVCacheBlock` callback and SGLang 0.5.22 external-linker
page descriptors expose reusable slot IDs without allocator-issued allocation
generations. OrbitKV cannot repair that boundary with an adapter-local map: such
a map misses allocations and frees performed by native prefix, scheduler and
other connector paths. The upstream interface work is tracked in
[vLLM #62810](https://github.com/vllm-project/vllm/issues/62810) and
[SGLang #45493](https://github.com/sgl-project/sglang/issues/45493). Until a
released interface is consumed, adapters retain unknown work and do not claim
generation-safe page reuse.

Draft [vLLM #62819](https://github.com/vllm-project/vllm/pull/62819), exact
`b37aa38c`, has independent source/CPU acceptance for allocation and finish-time
identity handoff. It remains an upstream candidate rather than a selected
release; OrbitKV does not patch the pinned engine or enable a compatibility
fallback around it.

Draft [SGLang #45548](https://github.com/sgl-project/sglang/pull/45548), exact
`3cc5121f`, has independent source/CPU acceptance plus a scoped H20 allocator
primitive pass for warmed steady-state asynchronous generation issuance,
snapshot and cross-stream readiness. Its unwarmed first invocation still waits
for preceding CUDA work. This does not qualify the current installed engine,
its default Rust tree runtime under faults, serving performance, page reuse
after cancellation or physical termination; the selected 0.5.22 package remains
unchanged.

Explicit registration replaces the framework pickle payload with
versioned protobuf storage/view metadata. The native client exports retained
tensors directly; Manager validation precedes CUDA device discovery/import.
Storage suballocation and tensor view offsets are separate; dtype/element size,
shape/stride extent, handles, ref-counter and readiness event are checked.
The selected PyTorch 2.13 ordinary storage export includes a version-3,
`cudaMalloc` type envelope around the 64-byte CUDA allocation handle. Registration
validates the complete 66-byte envelope and passes it unchanged to PyTorch's
importer. Bare legacy handles, unknown versions and expandable-segment exports
are rejected; they do not share this allocation-owner contract.
A native exporter-owner generation is distinct from the Manager registration
and engine page-slot generations. The importer still uses the pinned PyTorch
storage IPC primitive and holds the resulting tensor mapping until its owner
retires. The importer creates a zero-sized typed tensor with `torch.empty`, then
binds the validated storage/view with `set_`; a Rust byte vector becomes Python
bytes and cannot serve as an empty `torch.tensor` input. The old wrapper-byte
field and helper/API are removed.
The client retains the exact storage objects used for initial geometry. It
exports those objects, rather than reacquiring storage after a wait, and
rechecks the tensor pointer, offset and layout before sending and after the
import reply. A changed tensor binding is rejected; possibly imported work
remains retained. Registered engine pools must keep their binding immutable.
Native registration reserves its client/key before creating any export. Each
returned credit is charged to the attempt owner immediately; a never-sent export
can be repaid exactly once. Register replies bind the export generation and
explicit credit disposition. Only NeverImported permits repayment; importer
ownership is a handoff, and Unknown never proves import or device termination.
The selected framework's rollback helper is best effort and can return after
silently skipping a decrement. The native attempt owner therefore holds the
actual exported counter file, checks its identity and exclusive value before
rollback, and removes a credit only after the same slot is observed to change
from one to zero. Missing/replaced files, unchanged counters and call errors
retain the credit without another repayment attempt. This is a reference-credit
proof; it is not a device-operation terminal receipt.
A failed, partial or lost reply retains the attempt's client, tensors and credits,
even after caller/client destruction. Unregister refuses a pending/retained
attempt rather than reporting completed cleanup. The process limits these
attempts to 64, each with at most 4,096 tensors/credits, and conservatively charges
at most 1 TiB of tensor storage bytes. Within one validated single-device batch,
exact `(storage base, storage bytes)` aliases are charged once; different extents
and partially overlapping ranges remain conservatively additive. Each layer
still retains its tensor/storage owners and its own export credit. Attempts
remain independently charged, including failed attempts with uncertain credits. These limits are separate from ordinary
payload admission and existing transfer descriptor limits. Retention counters
and per-generation states are available from the native client. The original
unaccepted implementation and its negative review are preserved.
`GpuRegistration.ipc_credit_evidence` is an immutable snapshot from importer
handoff. `retained_credit_evidence(instance_id)` observes the original held
counter descriptors for failed attempts belonging to that client and instance.
The versioned JSON reports original and current-path identity separately, the
value observed in the held slot, and explicit read/path errors. A missing or
replaced path cannot become a successful observation. These diagnostics add no
repayment/release action, and a zero counter is not a device drain receipt.
Successful handoff drops the extra descriptors after the snapshot; uncertain
attempts retain them under the existing admission limits. Diagnostic output is
bounded to 4,096 credits and 4 MiB per call and is consumed by the strict installed
registration gate before the producer exits.

Independent installed prerequisite acceptance covers partial-export repayment,
conservative throw retention, duplicate registration rejection, native Restore
byte assertions, producer-stream fencing and bounded pool-retirement controls.
Actual model recovery is recorded separately under S3/S5 in the completion plan.
CUDA IPC warnings and an uncertain retained attempt remain explicit limits.
These checks do not supply allocation/page generations or physical DMA termination.

## Process channel

An isolated deployment can set the same absolute `ORBITKV_CHANNEL_CONFIG` file
for client and Manager. The Rust channel owner reads at most 64 KiB once per
process and passes that config explicitly to both Node builders. The file must
be regular and not symlinked, with an existing canonical absolute root, a private
prefix and node/service directories inside the root. Invalid input refuses Node
creation. Precreate every node/service directory component; the owner rejects
symlinks and retains their device/inode identities for a check before each Node.
Keep that task-owned directory chain immutable from admission through cleanup.
The dependency operates on pathnames after this check, so a privileged concurrent
rename is outside this isolation contract and must be excluded by the resource
owner/lock, rather than treated as atomically fenced by the library. Freeze the
TOML bytes and domain resources for qualification; changing
the file after first channel construction cannot rebind that process. This
separates iceoryx2's persistent counter, shared resources and stale-node cleanup
from a retained historical domain; unique service names alone do not. The default
deployment still uses the dependency's standard configuration.

### Physical drain

Cancellation, a deadline and a logical terminal state are not release receipts.
OrbitKV retains grants, leases, registrations and actual backing until physical
drain. Unknown effects are never replayed. The OrbiTE contracts below preserve
these rules for peer, file and local transfers.

The removed TENT implementation's failure controls and original revisions remain
in the [historical ownership record](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#native-tent-failure-retention),
including [Mooncake #4816](https://github.com/kvcache-ai/Mooncake/issues/4816).
That TENT evidence does not qualify OrbiTE.

`orbitkv-channel` uses iceoryx2 `0.10.0` and a fixed 64-byte command carrying
protocol version, opcode, request identity, Manager epoch, descriptor offset,
length/generation, and two opcode-specific scalar fields. Variable-length
hashes, leases, and page arrays use the descriptor arena. KV payload bytes do
not travel through either command or descriptor memory.

Each native cache client has a query/restore endpoint and opens a separate
Publish endpoint on its first save. The Manager owns the thread-safe iceoryx2
service on a dedicated control thread. Clients enqueue before signaling the
required request event; the Manager briefly spins and then waits for an event
or maintenance deadline. Fixed 50 us idle polling is removed. Notifications
are hints, so a failed wake after enqueue never permits early page reuse.

### Bootstrap, registration, and versions

A mode-0600 UDS authenticates the Manager's uid with `SO_PEERCRED` and assigns
an exclusive descriptor slot and session token. Bootstrap version **7** passes
five FDs: descriptor memfd, grant memfd, Manager-to-engine Restore eventfd,
engine-to-Manager retirement eventfd, and Publish reply eventfd. Memfds are
size-sealed. Descriptor request/response generations advance monotonically,
including when a slot is reused by a replacement session.

After bootstrap, the persistent UDS carries epoch-checked lifecycle version
**13** frames, with a 64 MiB metadata limit. Registration reuses protobuf metadata
without HTTP/2 or gRPC. A successful GPU registration reply also attaches
payload arena FDs, identities, and sizes with `SCM_RIGHTS`. The engine validates
the seals and independently maps and CUDA-registers each backing. This setup
happens once per arena and GPU binding, not per restored block.

Channel ABI **12** rejects older clients. Client, native extension, and Manager
must be rebuilt together; there is no old-wire decoder or alternate runtime
protocol. Registration requires actual tensor/exporter objects through
`register_context_batch(..., tensors=...)`, keeping them alive with the local
CUDA binding. Repeated registration of the same instance/rank/device on one
client is rejected. Unregister and close drain accepted operations before
releasing the bindings.

Every Manager incarnation advertises a unique iceoryx2 service name behind the
stable UDS address; `--channel-service` is a prefix. Old handles cannot be
adopted by a new client. Reconnect establishes new mappings and tensor bindings.
See [fault qualification](fault-qualification.md).

Native `peer_alive()` reads the pidfd of the Manager authenticated at bootstrap.
It remains independent of UDS health and descriptor/lifecycle waits. The restore
completion owner uses it for pending peer-exit diagnosis and to wake readiness
checks after process exit, while Active/ActiveMore local DMA stays pending until
the engine reports its drain. It neither reconnects nor changes generations.

`ObserveCompletion` carries bounded P/D completion and decode-resource evidence,
without request IDs or state keys. Cache schema **10** combines these observations
with the engine-local Restore grant protocol; older schemas are rejected.

### Restore and source ownership

`Restore` reserves a shared operation identity before descriptor submission.
The Manager authenticates and claims it before consuming leases. A lost or
malformed ACK retains the known handle once preparation was claimed; the
request is not retransmitted. Preparation failures are terminal results on
that handle. UDS closure stops descriptor admission but does not invalidate
retained completion mappings or prove DMA completion.

For unencoded DRAM sources, the Manager compiles a bounded plan containing
arena/allocation identities, source allocation bounds and subranges, registered
layer names, and destination-relative offsets. It installs the selected source
and query-reservation owners before publishing `Granted`. The engine wins
`Granted → Active` before reading the plan and uses its own pointers and CUDA
context. `start_restore(..., ready_stream=...)` supplies the actual engine
stream whose previous use of the destination pages must finish. The first
slice waits for that dependency and uses a single copy stream with a
whole-operation fence.

After the final local drain, the native worker publishes `Drained` and makes the local
result available through `poll_restore`/`wait_restore`. An engine-local eventfd
wakes framework completion handling. The Manager's separate retirement task
then releases source owners and publishes `Reaped`; engine acknowledgement
allows record reuse. Source reaping is not on the successful page-consumption
critical path. There is no terminal Poll RPC.

The grant mapping contains 1024 records of 192 bytes and a 1 MiB plan bank.
Error text is limited to 88 bytes per shared record. Plan consumption releases
plan-bank capacity independently of DMA completion, and a bounded dirty bitset
plus eventfd drives Manager retirement. Full shared plan-bank capacity defers
prepared grants. Larger raw plans are automatically partitioned under one
operation ID; nonfinal `PartDrained` records keep the result pending and retain
all source/target owners. The Manager acknowledges each part once before
publishing its successor. Operation metadata above 32 MiB is rejected before
lease consumption, and each session reserves at most 64 MiB of prepared
metadata. Shared-grant schema 5 requires matched Manager and native client builds.

At most 64 session mappings can be live or retained. An engine that dies after
claim without drain evidence leaves its sources, byte reservations, and session
credits quarantined. Neither TTL, UDS closure, nor pidfd exit authorizes source
reuse. If the Manager dies during a claimed local copy, the engine retains its
own imported mapping, CUDA registration, and tensors until local drain.
A managed Restore without a terminal publication also remains pending after
Manager process exit. An unreadable/unknown completion keeps the native worker
job and registered tensors owned; it does not produce `RestoreStatus.done`.
`poll_restore` can expose a bounded retention reason while `wait_restore` still
respects its deadline. Pending operation admission remains charged until a
known completion or a proven never-submitted drain. This candidate does not
provide a device/RDMA failure fence or reclaim orphaned pages.
Generic claim errors cannot publish drain merely by observing `Active` or
`ActiveMore`: another executor may own that transition. Only the completion
owner that won the CAS can establish never-submitted work when it rejects a
plan before returning its bytes. CAS loss and unknown failures remain pending.
Independent acceptance at `d9f779c9` covers code/CPU owner controls and the
matching installed wheel's import/ELF checks. It does not qualify live Manager
endpoint/GPU bytes, device/RDMA termination, page reuse or adapter faults.


SSD, encoded, and mixed-source restores use a `Managed` execution state and
retain Manager workers for their materialization, decode, and I/O. This follows
the prepared physical route, not a retry after local failure. CUDA IPC remains
necessary for those routes and Publish. The obsolete Manager raw-descriptor
worker branch is removed. See [engine-local Restore](engine-local-restore.md)
for full ownership, limits, and pending qualification gates.

Both native and Manager executors drain accepted copy work after partial
enqueue failure. The SaveTask owner candidate retains its source, destination
and admission owners after reply loss or an execution unwind. Failed Manager
stream or codec synchronization retains the call instead of using process exit
as completion evidence. Engine-local Restore failure handling remains separate. A wait timeout or dropped Python
handle does not cancel accepted work; the connector must retain logical
GPU-page assignments until a terminal result. A proven drained failure can use
vLLM's single-cache-group recomputation path; hybrid failures retain their
existing fail-closed engine semantics.

A missing GPU save worker reply leaves the admitted Publish future pending.
The caller retains its instance and imported GPU context; no consumed error is
published from reply loss alone. The SaveTask execution boundary borrows its
resources across raw/codec operations and contains an ordinary Rust unwind.
SaveTask also owns the registered instance through caller-future cancellation.
The shared process admission limits accepted Save operations to 1,024 and retained
request metadata to 32 MiB, including vector spare capacity. Worker completion
and proven pre-submit rejection repay the charge; reply loss and loss of logical
admission owners do not. Active operation, metadata and logical-byte gauges
remain charged on parked sync/unwind paths, and reply/owner loss additionally
increments the retained-operation gauge. Logical bytes describe requested GPU
ranges and are distinct from physical pool/SSD/GPU budgets. Arbitrary SSD worker
unwind has not gained a physical termination proof.
This preserves software ownership; it does not prove that a failed device access
ended. Actual GPU fault gates and bounded operational reporting remain separate.

### Publish and query control

`Publish` holds its iceoryx2 reply while the Manager save runs asynchronously.
The dispatcher serves other requests during D2H, but the caller waits until
copies complete and host publication is queued. Queries see blocks after the
write pipeline seals them. After bounded spinning, Publish waits on its own
reply eventfd and Manager pidfd. A 10 ms response-queue recheck recovers lost
wakes. Publish and Restore do not consume each other's notification counter.

Publish requires a Manager pidfd before submission. An ordinary deadline logs
a warning, then at most once per minute; it does not release source pages.
The request service permits exactly one Manager recipient. Clients reject a
service configured for more recipients; restart the Manager and clients together
when upgrading from the earlier service configuration. The native request-send
error remains typed. Only request-loan, pre-delivery connection and active-request
admission failures prove no submission. Other send errors may follow delivery
and cannot release source pages, even with one configured recipient.
Ambiguous/malformed replies and post-submit transport errors retain the source
owner even after confirmed Manager exit. Process liveness cannot prove CUDA
completion. Before parking on these errors, the client closes the session and
releases its descriptor lock. Other descriptor callers on that same client fail
without reusing the slot; the original source owner remains held.
Only a matched valid result following the Manager copy drain, or
a proven never-submitted send failure, permits return. This conservative
retention still blocks the caller; bounded operational responses and adapter
page-generation consumers remain separate pending work. The independently
accepted SaveTask correction keeps a lost worker reply pending and retains
admission instead of publishing a consumed error. Its code/CPU and package-load
acceptance does not establish installed GPU or physical fault completion.
Publish batches split to
the negotiated descriptor slot capacity, retaining matching page ranges across
layers until all chunks finish.
The local Publish activity candidate reports active/retained callers, the elapsed
interval since activity last became nonempty, overdue status and authenticated
Manager liveness. It uses immediate local lock attempts and pidfd polling, without
waiting for the blocked descriptor/lifecycle exchange; contention returns an
explicit unavailable error. The interval can include several overlapping calls
and is not a per-request latency. vLLM/SGLang completion polls use it for rate
limited diagnostics only. Pending, overdue, retained and dead-peer observations
never mark engine pages complete. At most 1,024 callers enter one channel's
Publish activity; the existing Save worker budget separately bounds accepted
Manager work. The original synchronous caller remains held until its established
completion path; an asynchronous operational return requires a later owner API.
Independent `26884461` code/CPU and candidate-package API acceptance covers this
diagnostic scope. Per-poll native/pidfd overhead and debug logging overhead are
unmeasured; neither package loading nor the controlled SGLang method qualifies
Manager/GPU operation, asynchronous completion or physical fault reclamation.

The native Save handle candidate admits a request under a process-wide
1,024-operation/32-MiB request-metadata budget, including retained vector capacity.
It retains the native caller, registration worker and tensor exporters on a Rust
thread. Dropping a Python handle does not cancel accepted work or release pages.
`wait_save(timeout=...)` returns pending when its wait expires; no timeout is a
terminal receipt. Without a timeout, the existing Rust owner waits through late
completion and retains unknown outcomes. Client close/unregister blocks new
admission while waiting for accepted Saves, then follows the existing worker and
Manager registry drain. The adapters retain task/page completion state until a
matched receipt and reject reset/unregister after an unknown wait response.
This candidate has no page-slot or physical-fault reclamation qualification;
request metadata and tensor/allocation bytes have separate owners and bounds.
The accepted `d49f6deb` correction exposes `SaveNotSubmittedError` only for an
explicit rejection before native worker launch. It proves nothing about older
Saves; unknown allocation, attachment and wait outcomes still retain pages.
SGLang also keeps already queued tasks behind an unknown offload from advancing
its completion FIFO. Code/CPU and package review, installed normal DRAM/io_uring
byte gates, both official-release model gates and same-host TCP shared-cache
restart gates pass for the frozen `cb19ce82` wheel. See the
[qualification scope](completion-plan.md#s5--released-engine-integration-and-upstream-contributions).
Prepared Save handles remain charged while held in Python queues, retained
adapter tasks or user-held completed handles, until their last native owner is
dropped. This metadata cap is separate from physical payload budgets.
Unknown-owner reconciliation, GPU/model fault gates, page generations and
physical termination remain separate requirements.
This Publish partitioning does not yet partition oversized raw Restore plans.

Rust `CacheClient`, exposed directly by PyO3 as `CacheManagerClient`, owns query
tickets for both adapters. Python constructs an immutable `BlockHashes` batch
once per lookup; views share its allocation and native pending-query identity
checks avoid repeated per-page conversion. Submission, revision, polling, and
cancellation remain native, and blocking calls release the GIL.

Cache schema 10 distinguishes metadata-only candidates from leased reads.
Discovery returns `Candidates`, never a Restore lease, and does not reserve
payload bytes. `read_recovery` translates compiled demand into exact hash views
and validates complete leased coverage. The warmup flag prepares pages without
a lease, skips warmup-budget pressure, and retires without polling. See
[queued warming](queued-warming.md). Consumer-owned preparation and exact
revision claims retain their contract in [request preparation](request-preparation.md).

Pending query operations are bounded to 128 per session and 1024 globally,
with a 60-second reply lifetime. Revision changes can replace hashes or wait
policy while preserving instance/request/group identity. Old polls and cancels
cannot consume replacements; unknown polls do not submit work. Submitted reads
retain ownership through cancellation or expiry, and undelivered results release
their leases without another poll. Delivered query reservations pass into the
Restore grant or Manager worker and survive until its actual terminal fence.

### Host and lifecycle boundary

The adapter derives `/tmp/orbitkv-<addr-port>.sock` unless
`orbitkv.bootstrap_socket` is set. A scheduler querying several TP shards uses
local sockets in `orbitkv.tp_shard_bootstrap_sockets`. The current centralized
vLLM scheduler requires all configured query shards on its host; cross-host TP
query fan-out remains future work. `orbitkv.wait_for_full_prefix` keeps its
existing local pending-query semantics.

Standalone mode starts no gRPC listener. Distributed placement configuration
enables the peer transfer-control listener. UDS and HTTP cleanup serialize
instance lifecycle, close Manager GPU queues, drain their submitted work, and
then release imported CUDA IPC mappings. Imported contexts use exact instance,
TP/PP rank and GPU UUID keys; punctuation in a legal instance name cannot make
cleanup select another instance that shares its prefix. Independent `f8cb87bb`
acceptance covers the selector/actor CPU controls and matched installed import;
actual CUDA mapping/lifetime gates remain open. The native client separately drains
its engine-local operations. Manager cleanup cannot discard active local grant
owners merely because the instance or UDS session has closed.

Manager shutdown first closes one atomic Publish admission/count state. A
Publish that registers before that close is an accepted operation and its full
continuation is drained; a control iteration that was prechecked but has not yet
registered is rejected after close and cannot enter during resource drain. No
mutex is held while an accepted continuation completes.

Lifecycle close is durable rather than an edge-triggered notification. A full
header and payload establish the accepted-operation boundary. Shutdown may close
a partial header or payload without dispatch, but an accepted operation keeps its
dispatch and bounded response write; the connection then observes the latched
close even when the client stays open and idle. Connections accepted around the
boundary either join the tracked pre-close set or are rejected by the same
closeable connection gate. Session-owner cleanup remains part of connection
drain.

## Measured process-channel baseline

The [current communication measurements](https://github.com/OrbitInfra/OrbitKV/blob/251f463d8997eff783871ab2820da83a13a97196/docs/communication-performance.md) compare
request events, Publish reply notification and encoding changes against the
preceding revision, including CPU cost and raw-ping regressions.

Measurements were collected on one H20 node with two Linux processes and a
64-byte request/response descriptor, before bootstrap version 2. They are
engineering evidence for the transport choice, not measurements of this
revision or end-to-end serving results.

| Path | Mean RTT | p50 | p95 | p99 | Sequential RTT/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| iceoryx2 two-process 64-byte ping | 4.052 us | 3.877 us | 4.585 us | 9.263 us | 246.8k |
| real Python/PyO3 local `QueryBundle` | 106.861 us | 107.404 us | 114.390 us | 120.635 us | 9,358 |
| real Python/PyO3 gRPC `QueryBundle` | 489.575 us | 485.715 us | 547.946 us | 635.653 us | 2,043 |

In that historical run, the real local path was about 4.58x faster than gRPC by
both mean RTT and sequential throughput. Its roughly 107 us RTT was far above
the 4 us iceoryx2 substrate. The current request event removes the fixed 50 us
idle poll, and request encoding removes repeated allocation and speculative
Publish chunk copies. The benchmark is sequential because the scheduler
needs one answer before committing a recovery boundary.

The iceoryx2 result can be reproduced with the two binaries documented in
[`crates/orbitkv-channel/README.md`](../crates/orbitkv-channel/README.md).

## Earlier local validation

The following serving results predate the engine-local raw Restore cutover;
its new process-fault and serving gates are tracked separately in
[engine-local Restore](engine-local-restore.md#qualification-gates).

With the pinned vLLM `0.29.0`, a single-node H20 run passed the applicable
pure-attention E2E recovery gates after an inference-process restart, comparing
ordered generated text to vLLM native prefix caching. SGLang `0.5.20` passed
its direct GPU-page restore gate after a radix-cache flush. These are local
correctness checks, not multi-host reliability or throughput measurements.
The SGLang multi-rank and cross-host TP paths still need qualification. See
[Python test gates](../python/tests/README.md) and the [completion plan](completion-plan.md).

## Transfer engine

OrbiTE is OrbitKV's only byte-transfer engine. The default build enables `orbite`
and CUDA 13. `--transfer-transport auto` uses `Transport::Unspec`; `tcp` and
`rdma` are explicit operator hints. RDMA requires `orbite-rdma` plus an explicit
RNIC configuration. `orbite-gds` enables optional cuFile support, loaded only
when GDS is configured. HP-TCP, SHM, NVLink, MNNVL and automatic failover remain off.

The Rust dependency is pinned to
[`1312e9b180ec9029535f8c2fb751d57c4da93b76`](https://github.com/OrbitInfra/OrbiTE/commit/1312e9b180ec9029535f8c2fb751d57c4da93b76).
Only its Rust API is consumed. OrbitKV and OrbiTE share `cudarc` 0.19.8.
`orbitkv-core::transfer_engine::TransferEngine` is a concrete owner for retained
pinned/CUDA registrations, epoch-bound peer handles, file segments and stream
bindings. The former multi-backend trait, TENT crates, native libraries and local
copy implementations are removed. Source grants remain authoritative; engine
registration permissions do not replace authorization.

Peer, planning, storage and ownership code compile without backend features.
Such a build starts no native engine; networking, SSD and local byte movement
return typed `Unavailable` errors. GPU workers reject missing engines before
CUDA initialization. `scripts/check-backend-features.py` confines optional native
code to adapters and build glue. There is no local-copy fallback.

OrbiTE bounds registrations to 4,096 regions and the configured pool bytes plus
2 GiB for persistent GPU tensor/codec registrations. Peer batches admit 16,384
descriptors and 64 MiB; outstanding work is capped at 32 batches and 64 MiB.
Process-wide registration admission remains 1,024 pools / 1 TiB. Uncertain owners
keep those charges. Transport slicing and pipelining belong to OrbiTE.

Requests carry intent, priority and the scheduler deadline. Owners survive
cancel, timeout and Unquiesced until a separate physical-drain receipt.
Shutdown fences both directions; failure retains engine, backing and admission.
Remote WRITE uncertainty never permits replay. Source epochs come from grants;
each requester opens its own `RemoteSegment`.

Real CPU/TCP controls exercise policy selection, fragmented READ/WRITE, bytes,
cancellation, epochs, retirement and churn. Linux io_uring controls check actual
files; CUDA-shim controls check CPU bytes and ownership only. **All current
OrbiTE GPU, RDMA, GDS and installed profiles remain unqualified until hardware
re-qualification.** Older TENT and OrbiTE measurements below retain their
original revisions and do not qualify this data path.

The [TENT implementation and native provenance](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#mooncake-findings)
are historical. No Mooncake fork or OrbiTE C ABI is loaded by this build.

## Peer restore into GPU

Demand and WaitForFullPrefix peer queries return immutable source descriptors and
retain the authorization ticket; they allocate no requester pinned payloads.
Restore binds the descriptors to the registered layer/page layout, including
split K/V and sparse block destinations, and submits READs to the selected
`TransferEngine`. Raw bytes land in the final GPU pages. Encoded bytes land in
GPU codec input allocations; OrbitKV retains representation checks and GPU
decode compute. Exact siblings go directly to their final pages. Peer ranges
never enter OrbitKV's pinned-pool/H2D restore path.

Prepare and Warmup explicitly populate the DRAM cache tier and retain their host
READ destination. Query coalescing distinguishes this intent from GPU restore.
A GPU restore does not create or advertise a DRAM replica. Authorization retries
remain pre-submit only; a failed or uncertain submitted READ is not replayed.
The source epoch still comes exclusively from its grant. Membership and the
deadline are rechecked before each bounded GPU batch. Expired grants reject
submission without replacing the source or disturbing the tensor registration.

At import, the registry registers each GPU KV storage allocation with the
selected engine once and lends that registration to its layer layouts. Views of
the same imported storage share it. The registration owns the actual Python
tensor and CUDA context; the native provider charges its existing registration
allowance for the entire lifetime, including retained failures. Pointer-only
Rust layouts still support local copies but reject peer restore.
`register_context_layer_batch_strided` accepts persistent `CudaRegistration`
owners for this path. Release fences new loans and unregisters only after active
transfers drain and all importing contexts release the storage. Failed cleanup
retains backing and charges without retrying unregister.

The planner batches all destination ranges sharing a source grant, up to 64 MiB
and 16,384 descriptors in one engine submission. Larger plans use bounded batches
against the same registrations; OrbitKV performs no per-chunk registration or
unregistration. OrbiTE receives those descriptors directly with a 64 MiB request
ceiling; the engine owns transport slicing, multirail and pipelining. The existing
128 admitted restores per device and source deadline remain unchanged.

Encoded restores create one persistent GPU codec input pool per worker on first
use, capped at the smaller of 64 MiB and half its codec budget. Later restores
reuse its registration and decode before reusing its bytes. Remaining codec
scratch accounts for this resident pool; a lower budget is rejected rather than
reallocating under live work. Worker drain closes the pool after all transfers.
Unknown drain or backend unwind retains source grants, tensor/input mappings,
query/device admission and the completion sender. The worker is fenced and
instance drain fails; no client receipt permits page reuse.

OrbiTE owns bounded CUDA staging for TCP and, by default, RDMA.
`--orbite-rdma-gpu-direct` selects strict Direct CUDA registration and requires
`MLX5_SCATTER_TO_CQE=0` before startup; failure never silently stages or replays.


CPU controls drive the production planner, submission and lifetime owners with
host backing: eight restores register once and never unregister before tensor
release; an 18 MiB, two-range restore submits once with full byte and guard checks.
They also cover input-pool reuse, batch limits, layouts/codec bounds, source
fences, concurrent release, failed cleanup, unwinds and unknown drain. CI runs
these with OrbiTE and the no-backend ownership controls. These controls do not qualify
CUDA/RDMA. `peer_gpu_restore` cost samples stay separate from H2D and HostReady.

Installed GPU byte gates are written, compiled and **unrun**. With CUDA 13,
`ETCD_BIN`, isolated test-owned peers and external artifact paths configured:

```bash
cargo test --locked -p orbitkv-server --no-default-features --features cuda-13,orbite --lib cluster::tests::p2p_orbite::p2p_orbite_remote_fetch_roundtrip -- --exact --ignored --nocapture
cargo test --locked -p orbitkv-server --no-default-features --features cuda-13,orbite-rdma --lib cluster::tests::p2p_orbite::p2p_orbite_rdma_remote_fetch_roundtrip -- --exact --ignored --nocapture
cargo test --locked -p orbitkv-server --lib cluster::tests::p2p_orbite::encoded_peer_payloads_restore_the_same_gpu_image -- --exact --ignored --nocapture
cargo test --locked -p orbitkv-server --no-default-features --features cuda-13,orbite --lib cluster::tests::p2p_orbite::orbite_encoded_peer_gpu_restore -- --exact --ignored --nocapture
```

The RDMA gate additionally requires `ORBITKV_TEST_ORBITE_RDMA_DEVICE`,
`ORBITKV_TEST_ORBITE_RDMA_PORT`, and `ORBITKV_TEST_ORBITE_RDMA_GID_INDEX`.
Run it separately with `ORBITKV_TEST_ORBITE_GPU_DIRECT=1` and
`MLX5_SCATTER_TO_CQE=0` for the strict GDR profile. Encoded gates require nvCOMP
5.3. Preserve native-route evidence and qualify physical cross-host operation
separately. The raw gate verifies direct GPU bytes without DRAM cache insertion,
then explicitly populates and restores the DRAM tier. The existing installed
CUDA IPC peer gate remains required; no source/CPU result qualifies that boundary.

### Scheduling metadata

| KV operation | Engine intent | Priority |
| --- | --- | --- |
| Demand / WaitForFullPrefix restore | `foreground_get` | High |
| Prepare / Warmup DRAM-tier fill | `background_prefetch` | Low |
| SSD save / write-back | `checkpoint` | Low |

Checkpoint describes persisting sealed KV state. Placement and eviction remain
OrbitKV decisions; these writes do not perform an engine migration.
The Manager's read-admission deadline reaches source authorization, SSD queues,
peer grants and the final READ descriptors without being reset. Shared reads
require the same deadline and placement; sharing cannot replace one caller's
budget with another's. Source grants may shorten the deadline further. Write-back
has no request scheduler owner, so its existing 30-second budget starts at queue
admission and is shared by every extent. Expiry remains a logical outcome, never
a physical release receipt.

Peer gRPC carries foreground/background intent and the remaining read budget in
milliseconds. A source caps that relative budget at 30 seconds for materializing
its SSD export; the requester still enforces its original absolute deadline.
This does not assume synchronized host clocks. A missing/zero budget is rejected.
Upgrade all peers together; lifecycle 13 requires matching clients and
Managers and rejects lifecycle 12. The registration reply no longer selects an engine.

OrbiTE receives all three fields on peer and file requests. `--ssd-backend auto`
uses Unspec file selection; explicit io_uring/GDS choices remain hints. Only
configured providers are enabled, so auto does not load cuFile. Local stream copies have
no QoS fields in the consumed API and retain caller-stream ordering.

CPU/TCP priorities are advisory. The consumed engine API does not expose per-priority
CPU queue/dispatch statistics, so the CPU tests prove field mapping, policy
consumption and real bytes, not foreground-before-background service. RDMA
priority ordering and serving improvements require installed hardware controls.
The real TCP selector test rejects an unmapped intent or priority instead of
quietly selecting the same route.

OrbiTE RDMA NIC load is read from `nic_load_stats` and exported as
`orbitkv_orbite_nic_inflight_bytes` and
`orbitkv_orbite_nic_bandwidth_bytes_per_second`. These aggregate engine statistics cannot replace the cost module's
per-KV-path latency samples. Codec, SSD and restore observations remain; replacing
pure transfer-leg latency estimates needs a per-transfer engine timing API and
is a follow-up.

### Peer departure

The etcd watch installs each node's epoch, incarnation and endpoint together.
Removal or replacement permanently revokes that installed identity before any
native cleanup. Discovery, outstanding authorizations and deferred restore
plans cannot schedule more bytes using the revoked identity. A temporary snapshot
repair suspends admission; an unchanged authoritative snapshot preserves its
identity. Local lease expiry or fencing retires all cached peers.

Each requester owns its own `RemoteSegment` cache; handles never cross engines.
The cache admits at most 1,024 identities, including uncertain retirement and
query-held identities, and at most 1,024 operations per identity in addition to
existing transfer budgets. One retirement worker per requester scans every
100 ms and waits at most 100 ms per `close_segment(endpoint, epoch, timeout)`
call. Native cleanup itself cannot be preempted; a blocked native call can delay
that worker, but holds no admission lock for other identities and creates no
additional cleanup workers.

OrbiTE closes native admission and drains the identity's admitted operations.
`Ok` releases the cached handle even if a query still holds the revoked grant;
dropping handles after retirement lets the engine reclaim its bounded identity
table. Timeout/Unquiesced keeps grants, leases, pins, descriptors and budget
charges. Repeated calls only wait for the engine's recorded retirement result;
uncertain native destruction is never retried. Runtime teardown is not a drain
receipt. A successful whole-engine shutdown supplies a physical receipt too.
A fresh identity uses the new epoch from the authoritative source grant; stale
grants remain fenced and Describe rejects a mismatched native source epoch.

Ticket release and RPC acknowledgement retries retain their independent completion ownership.

CPU/TCP controls hold an actual reply while membership removes the peer, verify
source pins and destination owners survive the retirement timeout, and finish
with byte checks. They cover two identities on each of two requester engines,
unrelated-peer progress, epoch replacement, stale rejection, and churn beyond a
two-slot native identity table while old query identities remain alive. Run
`cargo test --locked -p orbitkv-core --no-default-features --features cuda-13,orbite --lib peer::orbite::tests::retirement`;
these controls are also included in the existing CI peer gates. GPU/RDMA native
retirement and installed shared-cache failure profiles remain unqualified.

## SSD file transfers

SSD flush waits for logical write outcomes. It does not promise fsync durability
or prove physical drain of failed operations. Only complete successful writes
become index-visible.

Every SSD byte transfer uses OrbiTE file segments, including local-only deployments.
The native OrbitKV io_uring/cuFile owners and `--ssd-io-engine` switch are removed.
Failure after submission never selects a different engine or replays a WRITE.

OrbitKV still creates/truncates and sizes the shards, owns placement, eviction,
logical ring extents, encoding metadata and index publication. The engine sees
only byte ranges. Its file name refers to the retained inode through
`/proc/self/fd`, so replacement of a path cannot redirect an admitted transfer.
Offsets are byte offsets from file base zero. Whole host extents preserve
512-byte O_DIRECT alignment; GPU extents preserve 4 KiB alignment. Payload ranges
may be unaligned. Layout holes and tails are `Padding` elements: WRITE emits
zeros and READ discards file bytes without a local allocation.

`--ssd-backend uring` uses host materialization; `auto` additionally permits GPU
destinations through OrbiTE-owned CUDA staging and io_uring. `cufile` requests
native-only GDS and requires `orbite-gds`; padded vectors explicitly return
Unavailable for GDS. This does not silently select io_uring. The `cufile` read
route selects a GPU destination; OrbiTE owns how bytes reach it.

OrbiTE receives **one `TransferRequest::file_vectored` per extent**, including
its padding. It owns gather/scatter and storage staging; OrbitKV has no GPU
layout slots or zero buffer. Imported tensors reuse their persistent engine
registrations. Direct Rust callers must supply those allocation owners to layer
registration; pointer-only layouts cannot execute GPU file transfers. The bounded
codec input pool registers once per allocation;
encoded writes retain the actual codec arena until drain. The worker holds
caller completion and admission as well as backing on unknown GPU drain, and
fences subsequent work. DRAM representations still initialize their own padding
because peer exports can expose those complete host ranges.

Each store admits at most 32 file operations, including queued and retained work.
Each vector is capped at 64 MiB and **1,024 elements including padding**, with a
30-second deadline. Two file receipts per GPU worker, at most one WRITE, bound
its active work. Repeated file ranges targeting different GPU destinations
require separate vectors: the API does not express fan-out in one vector. Raw reads
are grouped by immutable extent generation, never across source generations.
There are at most 256 shards; SSD queue/inflight settings must be 1–1,024. Unknown drain or failed
unregister retains the exact fd, engine, registration, backing, extent lease and
admission until process exit. A failed WRITE stays unpublished and cannot be
evicted or overwritten. `orbitkv_ssd_engine_retained_operations_total` records
such retention. A failed or short READ never becomes a cache hit.

OrbiTE owns padded/unaligned staging. Only a physical receipt permits owner
destruction. Cancel, deadline, receiver loss and logical terminal status do not.
Cleanup is never retried after uncertainty. The file prefix includes padding;
a short READ is not a cache hit.

`storage/ssd/gpu.rs` is deleted. SSD reader/writer code plans layouts, manages
queues/leases and publishes metadata; file bytes move through the selected
engine. All-padded extents are legal and need no consumer-owned buffer.

CI requires real Linux io_uring and real bytes, with a labelled CPU host-pin
shim. Controls cover raw/encoded shards, leases, padded and odd vectors above
4 MiB, all-padding extents, unconfigured GDS rejection, alignment, truncation,
pre-submit cancellation, detached completion and uncertain receipts. The latter
uses a receipt-only fault seam because normal file I/O cannot produce a lying
or missing physical receipt. Engine creation, backing and files remain real.
Both ext4 and tmpfs runs verify the filesystem type; ext4 additionally verifies
an engine-owned O_DIRECT descriptor. EOF within padding or data is rejected
before touching the destination. GPU/CUDA staging, GDS and installed serving
remain unqualified. Source hardware gates are written but unrun:

```bash
ORBITKV_SSD_TEST_ROOT=/path/to/test-owned/ssd cargo test --locked -p orbitkv-core --test ssd_engine -- --ignored --exact cuda_staged_file_roundtrip --test-threads=1
ORBITKV_SSD_TEST_ROOT=/path/to/test-owned/ssd cargo test --locked -p orbitkv-core --features orbite-gds --test ssd_engine -- --ignored --exact native_gds_file_roundtrip --test-threads=1
```

Native GDS qualification also requires cuFile/nvidia-fs route evidence from that
same installed run; configuration and successful bytes alone do not prove the
native route. OrbiTE sets `EngineConfig::gds_enabled` only for configured
`cufile` storage. Compiling `orbite-gds` no longer loads cuFile for peer-only,
`uring` or `auto` engines. Configured GDS discovers the driver lazily on its first
eligible file submission; a missing runtime SDK fails that request.

## Local GPU↔DRAM copies

OrbiTE owns local payload movement in Manager Publish/Restore, codec payload
uploads/downloads, mixed SSD jobs and the native client's raw DRAM Restore.
`TransferEngine::bind_local` binds `cuda::local_copy::LocalCopyEngine` once per
worker stream with a persistent tensor registration. OrbitKV's MemcpyBackend,
KernelBackend and TentLocalCopy are deleted. `direct`/`kernel` instance choices
map to OrbiTE DMA/kernel selection. Unowned layouts fail binding.

OrbitKV supplies checked block/layer ranges. OrbiTE compiles contiguous and
strided rows and owns descriptor uploads, copy kernels, events and DMA. A worker
admits windows of at most 1,024 descriptors and 128 MiB; larger layouts drain
preceding windows before submitting the next. Each binding has eight native
slots and a 256 MiB payload/descriptor budget. Admission waits for completed slots
and bytes before enqueueing; the client retains at most eight completion handles.
Host registrations and tensor/codec registrations also consume the existing engine limits. No uncertain window is
replayed. A failed submit, timeout or cancelled wait never authorizes byte reuse.
A failed drain retains the source grant, slot, tensor, stream and completion.

Each physical pinned pool has one `PinnedHostRegistration`, retaining its actual
mapping independently of the allocator. Canonical `HostMemory` views additionally
retain the specific allocator slot. Codec arenas keep registrations until they
are replaced or released after drain; a stream binding anchors a persistent
tensor so it cannot retain an obsolete codec arena. CPU decode writes into a
charged pinned output slot. Codec kernels, descriptor/codebook uploads and status
readback remain codec compute; they do not implement payload transport.

Lifecycle **13** removes the selected-engine field from the registration receipt.
Clients and Managers must upgrade together; version 12 is rejected. The client registers its
actual retained tensor exporters and imported arena pins once. It records each
framework layer event after that layer's engine submission and holds native
completions through the whole-operation drain before publishing `Drained`.
Existing readiness waits fence the producer before grant admission; compute on
the worker stream is already ordered, so submission needs no extra ProducerEvent.
OrbiTE still rejects overlapping active local-copy loans, including read aliases;
callers must drain before reusing such ranges. Its general host registration API
also follows the selected RDMA memory profile; it has no separate local-only
native-registration profile. The default staged host profile needs no local-copy MR.

CI builds `tests/native/local_copy_cuda_shim.cpp` as a private `libcuda.so.1` and
requires `ORBITKV_REQUIRE_LOCAL_COPY_SHIM=1` for the ignored
`peer::orbite::local::tests` controls. They exercise production OrbiTE DMA/kernel
owners with CPU bytes: 2D gaps, bounded windows, partial submission, client layer
events/context restoration across thirteen layers, delayed slot retention and
failed completion query.
These controls provide no GPU qualification. The installed gate now covers both
copy modes; with the private runtime and resources described in the installed
profile, run:

```sh
ORBITKV_TEST_INSTALLED_ORBITE=1 ORBITKV_TEST_STRICT_SHUTDOWN=1 \
  python -m pytest -m release_smoke tests/release/test_installed_orbite.py \
  -k tcp
```

This gate remains unrun for this revision. GPU, RDMA, GDS, serving and fault
qualification remain open for the current pin. The git dependency uses the `orbite` crate and imports, with `orbite`,
`orbite-rdma` and `orbite-gds` features. Lifecycle 13 requires matching client and Manager packages. Peer endpoints use `orbite://`, `orbite+tcp://` or `orbite+rdma://`;
run matching builds across the cluster. No old-name aliases are accepted.

## Historical OrbiTE allocation bridge

The opt-in `orbitkv-core/orbite` feature pins
[OrbiTE `2e50af25`](https://github.com/OrbitInfra/OrbiTE/commit/2e50af25007ca9681912ba5175109d09488a8f99).
`PinnedAllocation::orbite_memory` returns a canonical `Arc<HostMemory>` retaining
that exact allocation and its shared physical CUDA pin. A pool owner alone would keep the mapping alive while
allowing a slot to be reused. The allocation holds only a weak cached view, so
concurrent aliases share one access lock without creating a reference cycle.
The unsafe caller must coordinate CPU, GPU and other process access; this lock
cannot synchronize unrelated CUDA streams or another process's mapping.

Register these views through OrbiTE's host registration/batch API and retain
its returned handles. Accepted requests retain the allocation through logical
cancellation, wait timeout, unregister and a detached batch until the backend
proves local drain. This first bridge registers allocation ranges; it does not
add a whole-pool MR cache or switch the Manager peer path. OrbitKV still owns
source grants, allocation generations and completion-mailbox matching.
The peer profiles below consume source grants and generation-scoped gRPC
completion. Transfer-notification mailbox integration and installed serving
qualification remain open.

Source CI checks the optional feature with the normal CUDA 13 build. Run the
hardware gate explicitly on an isolated GPU with a compatible CUDA runtime:

```sh
cargo test --locked -p orbitkv-core --features orbite --lib \
  memory::pool::backend::orbite_tests -- --ignored --test-threads=1
```

The two tests use the production `PinnedAllocator` (memfd, CUDA host registration
and real slot reclamation), actual OrbiTE engines and TCP bytes. They cover
concurrent aliases, duplicate/subrange registrations, READ/WRITE integrity,
blocked transfer cancellation and detach, failed early free, unregister, slot
exhaustion, allocator drop and eventual owner release. A TCP relay delays the
actual data connection without implementing a fake transfer engine. They are
ignored by ordinary tests because a real GPU is required; an unavailable GPU
leaves this gate unqualified.

Historical allocator pilots used a private CUDA 12.2 runtime. Their immutable
source/artifact records are retained, but current OrbitKV builds require CUDA 13;
CUDA 12 is unsupported. Run the current allocator gate on a CUDA 13-capable
machine. Keep exact source/binary/runtime hashes, device identity, resource
locks, bytes and cleanup outside the checkout; do not infer RDMA, GPU DMA or
crash recovery from a TCP ownership gate.

## Experimental OrbiTE peer TCP profile

OrbiTE TCP is compiled by default. Select `--transfer-transport tcp` to pin TCP,
or leave `auto` for engine policy. The `orbite-rdma` feature additionally enables
RDMA. CPU/TCP CI evidence does not qualify installed serving or GPU transfers.
Run matching builds across peers.

Manager still authorizes residency and holds generation-scoped gRPC tickets.
The OrbiTE source registers the actual allocations of each accepted grant and
keeps those explicit references with its window/slot/generation. Concurrent
grants share canonical allocation views and registration references. Expiry
retains the source. A completion for an older generation cannot retire a newer
grant. In-flight OrbiTE handlers retain the backing even after logical unregister.

The requester opens the `orbite+tcp://IP:port` endpoint with the epoch returned
by the source grant and uses it unchanged for READ. A refreshed descriptor cannot
replace that authority. Provider mismatches fail before payload I/O. A blocking
READ owner retains source tickets and actual destination allocations after caller
cancellation. A locally unquiesced outcome quarantines both the ticket and buffers,
keeping completion admission charged instead of sending a release RPC. Source
shutdown fences admission even if requester drain fails; failed cleanup retains
its engine and process pool budget.

This TCP profile requires a concrete numeric advertised IP and has no RDMA NIC
filter. It uses the [runtime registration and batch limits](#transfer-engine).
A source registration or READ exceeding those limits is rejected before payload
I/O; GPU restore plans partition larger requests into bounded batches without
reregistering. Use bounded source grants. Whole-pool MR caching, multirail,
RDMA/device DMA, GDS and broader faults remain outside this TCP profile.

Transfer notifications and their generation mailbox remain a separate transfer
API contract. Manager releases source grants through its existing gRPC completion
owner. This profile does not turn a notification receipt into a physical drain proof.

CPU CI runs real TCP epoch/restart and protocol controls, the existing ticket
loss/retry/cancellation controls, and the unquiesced quarantine control in its
own exact-filter process. Run the GPU gates explicitly with a compatible runtime:

```sh
cargo test --locked -p orbitkv-core --features orbite --lib \
  peer::export::tests::orbite_grant_registrations_share_backing_and_fence_old_completion_generations \
  -- --exact --ignored --test-threads=1
ETCD_BIN=/path/to/etcd ORBITKV_METADATA_ARTIFACT_DIR=/var/tmp/orbitkv-peer-gate \
  cargo test --locked -p orbitkv-server --features orbite --lib \
  cluster::tests::p2p_orbite::p2p_orbite_remote_fetch_roundtrip \
  -- --exact --ignored --test-threads=1
```

The core gate uses the real pinned allocator, duplicate grants, slot generation
reuse, eviction and actual TCP bytes. The server gate reuses the maintained
workload with actual Manager gRPC services, etcd/inventory discovery, source
budget/replay/fencing controls and GPU Save/Restore byte comparison. It awaits
both service tasks and records test-owned etcd wait results outside the checkout.
At implementation `5a50d092c651d28ff3f79f6de661c9da8120b2b3`, both H20
source gates passed and were independently rerun with a private CUDA 12.2
runtime. Source checks, protocol/ticket controls, exact artifact hashes, mapped
libraries, GPU leases and owned native/supervisor/etcd reaping were checked.
The retained failed first attempt and corrected runs are outside the checkout at
`/root/orbitte-artifacts/peer-read-20261021/`; see [PR #304](https://github.com/OrbitInfra/OrbitKV/pull/304).
This is a same-host embedded Manager source pilot. It does not qualify an
installed Manager, serving, CUDA 13 runtime, physical cross-host Manager traffic,
RDMA or device DMA.

## Experimental OrbiTE peer RDMA profile

Build the explicit single-rail host-staging candidate with GNU C/C++ compilers.
CI fixes Ubuntu 24.04 with matching Clang/libclang 18 for `ibverbs-sys` binding
generation; use `CLANG_PATH=/usr/bin/clang-18` and
`LIBCLANG_PATH=/usr/lib/llvm-18/lib` for that parser profile.

```sh
CC=gcc CXX=g++ cargo build --locked -p orbitkv-server --no-default-features \
  --features cuda-13,orbite-rdma
```

Select `--transfer-transport rdma --orbite-rdma-device mlx5_0
--orbite-rdma-port 1 --orbite-rdma-gid-index 3` alongside the ordinary
membership/peer flags. Device/port/GID must match the actual RNIC and route;
these values describe the supplied RoCE test profile, not device discovery.
The device is required. Rust users
set `EngineConfig.orbite_rdma` with the re-exported `OrbiteRdmaConfig`.
The default `orbite` feature does not enable RDMA.

The provider advertises `orbite+rdma://IP:port` and fixes every payload request
to `Transport::Rdma`. Metadata, descriptor bootstrap and DONE/ACK use TCP;
payload bytes use RC verbs and private host staging. An unavailable device,
source or build feature returns an error without silently selecting TCP payloads.
Device availability is checked when the first native endpoint opens, not by
successful metadata lookup. A TCP namespace cannot authorize this provider.
The pinned allocation, grant epoch, registration references, cancellation owner
and generation-scoped completion remain those described above.

This is the existing OrbiTE staging path: a PD/CQ/QP/MR per slice, up to 64
endpoints and 64 MiB staging per engine by default. These charges are separate
from source tickets and the process pool budget. The TCP profile's 64 MiB READ,
16,384-descriptor and 4,096-allocation limits still apply. READ commits after
local CQ completion, successful local QP/MR destruction and the source ACK.
No application/GPU pointer is registered with the RNIC; this is not GDR or
zero-copy host registration. Endpoint reuse, multirail and pipelining remain open.

Local `locally_drained` describes requester resources. A source native cleanup
failure currently arrives as an ordinary remote backend error; it does not
supply a source termination receipt. The source transfer engine independently
retains its actual registered backing, admission, failed staging and budget,
poisons new native admission and prevents a successful source close. A gRPC
ticket release cannot override those owners. Local Unquiesced still retains
the requester ticket and destination allocations. Native calls can exceed a
Rust deadline; neither timeout nor watchdog action is physical drain evidence.

The explicit host component gate builds with CUDA 13 but never initializes CUDA:

```sh
CC=gcc CXX=g++ cargo test --locked -p orbitkv-core --no-default-features \
  --features cuda-13,orbite-rdma --lib --no-run
python3 /path/to/OrbiTE/tests/hardware/run.py \
  --profile /external/orbitkv-host-profile.json \
  --binary /external/core-test-binary --output /external/host-run-001
```

Use the maintained OrbiTE runner with `runner: "orbitkv-peer-host"`, explicit
RNIC/port/GID, identity-pinned SSH nodes, both kinds `cpu` and transport `rdma`.
It invokes the ignored exact test
`peer::orbite::tests::rdma_host::registered_host_rdma_peer`. Native assertions
check fragmented registered READ bytes, all guards, old epochs, protocol mismatch,
actual requester CQ counts, zero endpoint/staging/registration charges, mapped
verbs libraries, absent CUDA/TENT libraries and owned shutdown. The controller
holds RNIC leases until native wait/reap. This exercises the production
registered-read execution with ordinary host backing; it does not test the
PinnedAllocator or Manager GPU path.

Independent source/CPU review and physical host reruns pass at consumer
`660f893d`: A-to-B with a 4 KiB payload region and B-to-A with a 1 MiB payload
region, each with 64-byte guards at both ends and three fragmented READ CQ
completions per requester. Guards and unrequested bytes remain unchanged. Normal close returns all registrations/endpoints/staging to
zero. Both processes map `libibverbs`/`libmlx5` and no CUDA/TENT; native children
and supervisors exit and release their RNIC leases. The initial readiness-only
failure is retained with successful runs. Frozen binary/source provenance,
raw runs and independent review locations are linked from
[OrbitKV #305](https://github.com/OrbitInfra/OrbitKV/pull/305) and
[OrbitTE #12](https://github.com/OrbitInfra/OrbitTE/pull/12).

The embedded Manager GPU gate is separate and requires a working CUDA 13 runtime:

```sh
ETCD_BIN=/path/to/etcd ORBITKV_METADATA_ARTIFACT_DIR=/external/manager-rdma \
ORBITKV_TEST_ORBITE_RDMA_DEVICE=mlx5_0 ORBITKV_TEST_ORBITE_RDMA_PORT=1 \
ORBITKV_TEST_ORBITE_RDMA_GID_INDEX=3 \
  cargo test --locked -p orbitkv-server --no-default-features \
  --features cuda-13,orbite-rdma --lib \
  cluster::tests::p2p_orbite::p2p_orbite_rdma_remote_fetch_roundtrip \
  -- --exact --ignored --test-threads=1
```

The supplied H20 nodes combine kernel driver 535.161.08 with
`cuda-compat-13-1` user-mode driver libraries 590.48.01. Kernel driver R535 alone
does not rule out CUDA 13: NVIDIA's
[forward-compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
allows this combination on supported data-center GPUs. Record actual mapped
libraries and run each required contract; a CUDA compute probe alone does not
qualify CUDA IPC, Manager operation or GDR. Do not substitute CUDA 12.

At consumer `660f893d`, the A-node CUDA 13.1 source profile passes real pinned
source-grant controls and embedded Manager GPU Save/Restore with both TCP and
same-RNIC RDMA staging. Two embedded engines exercise actual etcd, inventory,
gRPC source grants, exact epochs and payload bytes. Maps contain the installed
`libcudart.so.13.1.80`, compatibility `libcuda.so.590.48.01` and, for RDMA,
`libibverbs`/`libmlx5`, with no TENT. Source/service workers drain and owned
native, supervisor and coordinator processes are reaped; the fixture kills
its private etcd coordinator, so this is not graceful-etcd qualification.

Reproduce that compatibility profile by prefixing the gate command with
`LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/targets/x86_64-linux/lib`
and selecting an exclusively leased GPU/RNIC. Library paths are deployment
inputs, not a fallback selected by OrbitKV. Keep the external source/binary,
loader, resource and independent-review evidence linked from
[OrbitKV #306](https://github.com/OrbitInfra/OrbitKV/pull/306).
B-node GPU computation remains untested while its canonical GPU leases are busy.
That embedded profile does not qualify installed CUDA IPC. The separate
installed profile follows below. Installed serving, physical cross-host
At that revision, Manager/GPU, RNIC faults and GDR remained open. The former
TENT-parity qualification requirement is retired.

### Installed client and Manager profile

**Historical evidence at the revisions below. It does not qualify the current OrbiTE path.**

[PR #307](https://github.com/OrbitInfra/OrbitKV/pull/307), source `eb1cc90f`, adds
complete optional wheels and a maintained two-Manager installed gate. Independent
H20 reruns pass both TCP and same-RNIC RDMA using the complete CPython 3.11 wheel
`36b491e1a0daa97d`, built from identical package inputs at `487beb85`. This
profile uses kernel driver 535.161.08, compatibility driver 590.48.01, Torch
2.13.0+cu130 and one explicitly selected CUDA 13.0.96 runtime. The compiler is
CUDA 13.1; that does not make the loaded runtime CUDA 13.1.

Each provider registers two clients through cross-process CUDA IPC, publishes
source GPU pages, awaits the inventory fence and fetches through the selected
OrbiTE payload provider. It restores 16,384 GPU bytes and checks every untouched
page, exact Save/remote/H2D byte counters and source completion. Four original
held IPC slots per case change from one to zero before normal Manager exit.
All client/Manager runtime files match the frozen inputs, RDMA maps real
verbs/mlx5, and neither provider maps TENT. The client, Managers, supervisors
and outer waiters exit normally, canonical leases are released and GPU memory
returns to zero. The private etcd fixture is killed and reaped; graceful-etcd
shutdown is outside this acceptance. Producer-exit warnings reject the
qualification.

Runtime selection is an explicit deployment input. Frozen `cudarc` 0.19.8 tries
`libcudart.so` before `libcudart.so.13`. Torch supplies the latter; a system
unversioned library can therefore load a second runtime even with Torch's
versioned file preloaded. This profile freezes a private directory containing a
standard unversioned alias to the actual Torch runtime and places it first:

```sh
mkdir -m 700 /absolute/private/runtime
ln -s /absolute/torch-site/nvidia/cu13/lib/libcudart.so.13 \
  /absolute/private/runtime/libcudart.so
export LD_LIBRARY_PATH=/absolute/private/runtime:/usr/local/cuda-13.1/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=/absolute/torch-site/nvidia/cu13/lib/libcudart.so.13
```

Freeze that directory, alias identity and target before launching. The gate
requires exactly the selected CUDART file in every client/Manager mapping; its
version observer uses `RTLD_NOLOAD` and checks that observation preserves the
mapping set. Full loader-order CPU controls preserve the mixed-runtime negative
cases. Failed startup, duplicate-evidence and mixed-runtime attempts remain
archived with their verdicts. Raw runs, immutable wheel/source/loader manifests
and independent review are outside the checkout at
`/root/orbitte-artifacts/installed-rdma-20261022/`, indexed from PR #307.

DRAM Restore uses the client's local H2D executor; CUDA IPC is exercised by
registration and Publish. This scoped gate does not qualify Manager IPC Restore,
inference serving, physical cross-host Manager/GPU, RNIC faults, GDR, multirail
or broader hardware profiles. A uniform installed CUDA 13.1 runtime remains open;
the separately accepted CUDA 13.1 embedded/source profile above is unchanged.

## Historical TENT implementation

TENT is removed, including its C wrapper, Python API, separate file engine,
status mapping and built-in copy code. Its [original contracts and source pins](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#what-orbitkv-reuses)
and [registration failure controls](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#native-registration-error-control)
remain immutable historical evidence. They do not qualify OrbiTE.
Confirmed upstream defects should receive an issue promptly and a focused fix
PR when the cause is clear; OrbitKV does not duplicate a competing transport.

## Remote operation choice

- current demand-driven remote cache reuse uses OrbiTE READ into the final GPU destination;
  DRAM cache fills keep a host destination;
- vLLM live P/D uses the official NIXL connector and its native ownership;
- SGLang live P/D uses the official disaggregation backend. OrbitKV's external
  linker is independent and disables historical restores on D;
- proactive replica placement, retry across replicas, and bundle-aware
  publication are planning targets, not current guarantees.

The target recovery contract must reject incomplete or incompatible bundles;
the Manager wire protocol does not yet enforce `StateBundle` completeness.
SGLang validates prefix/window/checkpoint requirements and vLLM validates
attention/recurrent requirements in their adapters before restore, using the
shared Rust contract with absolute token coverage and leased group positions;
see [hybrid recovery](hybrid-recovery.md). Logical framework page generations
remain separate from the local source allocation IDs used by raw grants.


### Historical TCP copy error controls

The [TENT copy-error record](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#tcp-copy-error-propagation-candidate)
retains its original fork revision, CPU scope and unqualified hardware limits.

## Remaining work

The local raw payload arena protocol and native executor are implemented.
Bounded large-plan partitioning and native layer readiness are implemented;
broader graph/topology, page-generation and deployment gates remain separate.
CUDA IPC metadata still serves Publish and Manager SSD/codec routes. Peer
metadata currently uses gRPC. The local global index is synchronized through
bounded Manager inventory snapshots/deltas, with etcd limited to membership,
and has no foreground discovery RPC. Source grants/completions retain gRPC;
OrbiTE owns payload READ/WRITE. The custom native metadata bus is outside the
delivery plan. See the [communication plan](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains) and
[peer-control boundary](peer-control.md).

The removed TENT worker and unregister fault candidates remain in the
[original record](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/transport.md#remaining-work); they are not pending
OrbitKV implementations. Current OrbiTE hardware and installed gates remain open.
