# OrbitKV transport architecture

This document separates the inference-to-Cache-Manager channel, remote data
movement, and replica discovery so none of them becomes an accidental second
source of KV truth. "Same-host" describes where the client connects; the
Cache Manager decides whether a requested block is resident in RAM, needs SSD
prefetch, or can be fetched from a peer with Mooncake.

## Decision

| Boundary | Control | Payload | Status |
| --- | --- | --- | --- |
| inference process to local Cache Manager | iceoryx2 requests and shared grants | raw DRAM via engine-imported payload arenas; Publish/SSD/codec via Manager CUDA IPC bindings | raw local executor integrated; qualification gates tracked separately |
| local bootstrap and lifecycle | Unix socket with credentials and file-descriptor passing | descriptor/grant memfds, three eventfds, and GPU-registration payload arena FDs | implemented |
| Cache Manager to Cache Manager | Mooncake P2P handshake | Mooncake BatchTransfer over RDMA/TCP | stable Mooncake runtime integrated |
| replica directory | soft-state network API | no KV bytes | embedded fixed shards; replication planned |
| administration | HTTP | no KV bytes | existing |

Both framework adapters require `orbitkv-channel` for hot data operations. Every
inference process must connect to a Cache Manager on its own host; a missing
Unix socket fails fast.
Registration, health, session watching, and unregistration use the
authenticated bootstrap UDS for both adapters.

## Process channel

### Native TENT failure retention

The pinned TENT C ABI's `tent_free_batch()` accepts a deferred free request; its
zero return is not an independent physical-termination acknowledgement. Native
`FAILED`, `CANCELED`, `TIMEOUT`, `INVALID` or unknown task states do not authorize
source release or destination reuse. OrbitKV retains the synchronous native
call, batch descriptors and caller-owned buffers after observing one. It does
not send source release from that call or recycle its admission slot. A wrapper
deadline initiates best-effort cancellation and retains ownership while tasks
remain pending; only later successful completion of every task permits return
of the original logical failure. A native failure is not converted into success
by a later poll or replacement transport.

Each TENT engine admits at most 1,024 batches and 32 MiB of C request descriptors
across live and retained calls, rejecting excess before native allocation or
submission. These limits bound descriptor memory, not the retained payload;
physical memory remains owned by the existing pool/query/source budgets. Native
transport failover on polling is disabled until the preceding attempt's
termination can be established. Successful normal TCP/RDMA/CUDA transfers keep
the existing completion path; physical-failure recovery remains unqualified.

Retention is visible through `orbitkv_native_transfer_batches`,
`orbitkv_native_transfer_request_bytes`,
`orbitkv_native_transfer_retained_batches`,
`orbitkv_native_transfer_retained_descriptor_bytes` and admission rejections.
The retained descriptor-byte sum counts requested payload lengths, including
overlapping slices; it is not a physical allocation-size gauge. Uncertain calls
keep the owning process isolated and can prevent graceful shutdown. Retention
does not authorize killing that process or reclaiming remote buffers. See S3 in
the completion plan for the remaining native termination and installed gates.

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
**4** frames, with a 64 MiB metadata limit. Registration reuses protobuf metadata
without HTTP/2 or gRPC. A successful GPU registration reply also attaches
payload arena FDs, identities, and sizes with `SCM_RIGHTS`. The engine validates
the seals and independently maps and CUDA-registers each backing. This setup
happens once per arena and GPU binding, not per restored block.

Channel ABI **11** rejects older clients. Client, native extension, and Manager
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

`ObserveCompletion` carries bounded P/D completion and decode-resource evidence,
without request IDs or state keys. Cache schema **8** combines these observations
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

SSD, encoded, and mixed-source restores use a `Managed` execution state and
retain Manager workers for their materialization, decode, and I/O. This follows
the prepared physical route, not a retry after local failure. CUDA IPC remains
necessary for those routes and Publish. The obsolete Manager raw-descriptor
worker branch is removed. See [engine-local Restore](engine-local-restore.md)
for full ownership, limits, and pending qualification gates.

Both native and Manager executors drain accepted copy work after partial
enqueue failure. If CUDA cannot establish completion, the owning process
terminates without reporting reusable pages. A wait timeout or dropped Python
handle does not cancel accepted work; the connector must retain logical
GPU-page assignments until a terminal result. A proven drained failure can use
vLLM's single-cache-group recomputation path; hybrid failures retain their
existing fail-closed engine semantics.

### Publish and query control

`Publish` holds its iceoryx2 reply while the Manager save runs asynchronously.
The dispatcher serves other requests during D2H, but the caller waits until
copies complete and host publication is queued. Queries see blocks after the
write pipeline seals them. After bounded spinning, Publish waits on its own
reply eventfd and Manager pidfd. A 10 ms response-queue recheck recovers lost
wakes. Publish and Restore do not consume each other's notification counter.

Publish requires a Manager pidfd before submission. An ordinary deadline logs
a warning, then at most once per minute; it does not release source pages.
Ambiguous/malformed replies retain the source fence until a valid outcome or
confirmed Manager exit. A live Manager that never completes requires
operational recovery. Publish batches split to the negotiated descriptor slot
capacity, retaining matching page ranges across layers until all chunks finish.
This Publish partitioning does not yet partition oversized raw Restore plans.

Rust `CacheClient`, exposed directly by PyO3 as `CacheManagerClient`, owns query
tickets for both adapters. Python constructs an immutable `BlockHashes` batch
once per lookup; views share its allocation and native pending-query identity
checks avoid repeated per-page conversion. Submission, revision, polling, and
cancellation remain native, and blocking calls release the GIL.

Cache schema 8 distinguishes metadata-only candidates from leased reads.
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
then release imported CUDA IPC mappings. The native client separately drains
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

The [current communication measurements](communication-performance.md) compare
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

## Mooncake findings

Mooncake separates three responsibilities:

1. `TransferEngine` exposes Segment, Buffer, and BatchTransfer abstractions.
2. Transfer metadata publishes protocol, topology, registered regions, and
   connection endpoints.
3. Mooncake Store Master owns generic object allocation and the
   `PROCESSING -> COMPLETE` replica lifecycle.

The useful transport mechanisms are:

- topology-aware path selection across CPU, GPU, and multiple NICs;
- request slicing and multi-rail bandwidth aggregation;
- on-demand endpoint creation with a bounded SIEVE endpoint pool;
- two-phase endpoint retirement after outstanding work requests drain;
- retry on alternate local or peer rails;
- SHM, TCP, RDMA, EFA, NVMe-oF, and accelerator-specific backends behind one
  BatchTransfer interface.

`P2PHANDSHAKE` does not mean that Mooncake is control-plane-free. It starts a
custom TCP socket daemon, exchanges JSON metadata and QP/MR information, and
then uses the selected data transport. Larger deployments may instead publish
Segment metadata through etcd, Redis, or HTTP.

OrbitKV pins Mooncake `v0.3.13.post1` at
`719735896c86b56fabec6cf3e825fb2ea640597a` and builds its shared Transfer
Engine through the TENT `tent_shared` target in `orbitkv-mooncake-sys`. Native
loading first checks `ORBITKV_MOONCAKE_LIB_DIR`, then the executable or Python
extension directory, then the local `.orbitkv/mooncake/{cuda|cpu}/lib` build cache. Wheels
bundle `libtent_shared.so`, `libmooncake_common.so`, and `libasio.so`; the legacy
`libtransfer_engine.so` is removed from staged runtimes. System
RDMA/CUDA libraries remain deployment prerequisites.

## What OrbitKV reuses

`orbitkv-transfer::TransferEngine` is a narrow wrapper over the pinned upstream
Mooncake TENT C ABI. There is no legacy runtime selector and no OrbitKV-owned verbs
implementation. Authorized plans are lowered directly to Mooncake Segment
addresses, BatchTransfer operations, and notifications.

Memory registration returns an RAII `MemoryRegistration`. The token retains the
TransferEngine and unregisters its region on explicit completion or drop. It
does not own the allocation itself: the current host pool remains beside the
tokens in `MooncakeTransport` and is released only after registrations clear.
A future engine-HBM export must similarly keep its page-generation grant beside
the token, using Mooncake's `cuda:N` location, until native completion drains.
The current C ABI does not report the data transport actually selected for a
batch, so OrbitKV does not infer TCP/RDMA/GPUDirect from `--nics` or success;
qualification records Mooncake logs and NIC/device counters externally.

TENT task cancellation is best effort. OrbitKV requests cancellation after a
deadline or partial submit and continues polling. It calls `tent_free_batch`
only after every task reports `COMPLETED`; a successful free request is not a
completion fence. A native non-completed terminal or unknown status retains the
call and its source/destination buffers, as described in
[native failure retention](#native-tent-failure-retention).
After every task completes, the call returns the first recorded submission,
polling or cancellation error. It returns `Timeout` only when no such error was
recorded and the logical deadline expired.

OrbitKV also exposes TENT's bounded NIC load snapshot: device name, in-flight
bytes and EWMA bandwidth. This is live rail-pressure evidence, not a per-batch
transport receipt. P/D may use the aggregate for diagnostics, while shared-cache
qualification still checks external NIC counters before claiming RDMA or
GPUDirect.

The runtime loader resolves only `tent_*` symbols from `libtent_shared.so`.
CPU and CUDA variants are built separately from the same pinned source; the
CUDA variant includes CUDA/GDS/NVLink-capable TENT components. Missing TENT
artifacts fail startup and never fall back to `libtransfer_engine.so`.

### TENT versus the legacy Transfer Engine

The important change is the execution contract, not just a renamed shared
library. TENT exposes one segment and batch model across CPU, CUDA, file and
network-capable transports; extended registration carries memory location and
permission, while topology and rail selection remain inside Mooncake. Its C ABI
also exposes per-task terminal status, cancellation, notifications and NIC-load
snapshots. OrbitKV retains each allocation while the exact batch lacks completion
evidence and can feed live resource pressure into planning without owning an
RDMA implementation. Native failure statuses do not establish physical drain.

The legacy wrapper primarily exposed submit/poll/free around the older
TransferEngine object. In particular, freeing its batch handle was too easy to
misread as a completion fence, and its Python integrations commonly held only
raw addresses rather than registration owners. OrbitKV's TENT path instead
uses Rust RAII registration tokens and cancellation with completion or retained
ownership. This does not make every transfer automatically faster: HBM-to-HBM speed still
depends on the selected transport, topology, registration cost and chunking.
It does make the ownership and observability needed for safe high-performance
selection explicit. Because the current C ABI has no per-batch transport
receipt, RDMA or GPUDirect must still be proven with runtime logs and hardware
counters rather than inferred from a successful call.

TENT implements a real peer probe internally, but the pinned stable C ABI does
not export it. `tent_available` reports only local-engine health, and segment
metadata can remain thread-locally cached for an hour, so neither is a valid
substitute. OrbitKV therefore rejects SGLang's optional failed-session probe
mode instead of allowing a false-positive unblacklist. A failed native batch
keeps its caller and source-completion owner held; it does not return to
invalidate a cached peer segment while physical termination remains unproven.

OrbitKV does not adopt Mooncake Store Master as its semantic authority. Today
the cache engine handles versioned model/storage keys and leases above Mooncake.
The common recovery plan additionally includes:

- validated `StateDescriptor`, `StateBundle`, and `RecoveryContract` evidence;
- replica selection and restore-versus-recompute planning;
- query leases and generation validation;
- semantic and execution frontiers;
- publication only after every required component is complete.

The current Catalog advertises candidate owners, not durable KV data or
permanent raw addresses/rkeys. A selected source authorizes and pins current
bytes before transfer. Fixed shards and owner replay are implemented; directory replication is future work.

## Remote operation choice

- current demand-driven remote cache reuse uses Mooncake READ into destination
  pinned memory before restoring into framework HBM;
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


### TCP copy error propagation candidate

The pinned native build applies a scoped backport of merged Mooncake #3476 and
#3973 in a private build-source directory; see
[backport provenance](../crates/orbitkv-mooncake-sys/patches/README.md). TCP client
and server copy errors propagate to native FAILED. A READ error response whose
length equals the requested payload is padded, preventing error text from being
consumed as valid bytes. CUDA host-only copies use memcpy, while device copies
use the allocation-owning device and the driver's pointer classification.
Unexpected pointer-query failures do not select host memcpy, and error exits
after device selection restore the caller's device when one was previously
active. Controlled CUDA API checks cover these failure paths; healthy-GPU
execution remains unqualified. After a CUDA memcpy attempt reports an error,
the native owner retains the stream, current frame and RPC staging buffers;
errors are not a DMA termination proof. Its separate live/retained device-copy
admission is 64 calls and 1 GiB logical copy bytes; RPC strings allocated before
that boundary are separate transient allocations. Existing Rust descriptor
and physical-memory owners keep their budgets. Unusable CUDA bundle admission
fails before transport startup. No timeout-based reclamation is introduced.

These changes preserve the existing conservative failed-batch retention. A copy
error or FAILED status does not prove physical quiescence, and the wrapper does
not free retained source or destination owners. Deterministic CPU controls and
installed host-buffer byte tests have a separate trigger from healthy-GPU,
GPUDirect RDMA, Manager SSD and adapter-fault qualification. The fixed pin is not
replaced by an unreviewed upstream-main bundle.

## Remaining work

The local raw payload arena protocol and native executor are implemented.
Bounded large-plan partitioning and native layer readiness are implemented;
broader graph/topology, page-generation and deployment gates remain separate.
CUDA IPC metadata still serves Publish and Manager SSD/codec routes. Peer
metadata currently uses gRPC. The local global index is synchronized through
bounded Manager inventory snapshots/deltas, with etcd limited to membership,
and has no foreground discovery RPC. Source grants/completions retain gRPC;
TENT retains payload READ/WRITE. The custom native metadata bus is outside the
delivery plan. See the [communication plan](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains) and
[peer-control boundary](peer-control.md).
