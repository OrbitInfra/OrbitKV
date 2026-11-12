# OrbitKV transport architecture

OrbiTE moves all shared-cache bytes: local GPU↔DRAM, SSD file vectors and peer
READs. OrbitKV owns state compatibility, placement, grants, extents and GPU codec
compute. Live P/D uses the inference engine's native transport independently.

CPU/TCP, Linux io_uring and CUDA-shim controls are CI evidence. **Every current
GPU, RDMA, GDS and installed profile requires hardware re-qualification.**
Historical results retain their original backend and revision. The
[completion plan](completion-plan.md) is the status and delivery record.

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

| Boundary | Control and ownership | Payload |
| --- | --- | --- |
| Inference process ↔ local Manager | iceoryx2 commands and shared grants | OrbiTE local copies from imported DRAM arenas; Manager workers for SSD/codec routes |
| Bootstrap and lifecycle | Credential-checked Unix socket, versioned registration | Descriptor/grant memfds, eventfds and payload arena FDs |
| Manager ↔ Manager | Inventory discovery, gRPC source grants and release | OrbiTE READ into GPU restore destinations or DRAM cache fills |
| Replica discovery | Bounded owner inventory streams; etcd membership | Metadata only |
| Administration | HTTP health, configuration and metrics | No KV bytes |

Each inference process needs a Manager on its own host. A missing Unix socket
fails startup. Standalone mode needs neither etcd nor a peer gRPC listener.
See [adapter contracts](adapters.md) and [distributed cache](distributed-cache.md).

## Process channel

`orbitkv-channel` uses iceoryx2 0.10.0. Its fixed 64-byte command carries version,
opcode, request/session identity and a descriptor offset/generation. Variable
hashes, leases and page arrays use descriptor memory; KV bytes never do.

A native client has independent query/Restore and Publish endpoints. Eventfds
wake the Manager and waiting clients; notifications are hints, while versioned
records are authoritative. A failed wake after enqueue cannot release pages.

An isolated deployment may set the same absolute `ORBITKV_CHANNEL_CONFIG` on
client and Manager. The owner reads at most 64 KiB once per process, rejects
symlinks and noncanonical roots, and checks precreated node/service directory
identities before Node construction. Keep that directory chain immutable through
cleanup: privileged renames after the check are outside the contract. Unique
service names alone do not isolate iceoryx2's persistent counters and cleanup.

### Physical drain

Cancellation, timeout, lease expiry, process exit and logical terminal status
are not release receipts. Retain grants, leases, registrations, backing and
admission until physical drain. Never replay an operation with unknown effects.
Metadata ACKs and gRPC ticket-release replies cannot replace native completion.

### Bootstrap, registration, and versions

Client, native extension and Manager must upgrade together. No old-wire decoder
is retained. Current versions are bootstrap **7**, channel ABI **12**, lifecycle
**13**, cache schema **10** and shared-grant schema **5**.

The mode-0600 UDS authenticates the Manager uid with `SO_PEERCRED`, assigns a
session token and descriptor slot, and passes five FDs: descriptor/grant memfds
and Restore, retirement and Publish-reply eventfds. Memfds are size-sealed.
A new Manager incarnation advertises a unique service behind the stable socket;
reconnect creates new mappings and bindings, never adopts old handles.

Lifecycle registration has a 64 MiB metadata limit and returns an opaque
`GpuRegistration` plus payload arena FDs. There is no selected-engine field.
The client independently maps and CUDA-registers each arena once per GPU binding.
Tensor UUIDs resolve client-local ordinals to Manager-local devices.

Registration uses explicit protobuf storage/view metadata and retained tensors.
The PyTorch 2.13 version-3 `cudaMalloc` export envelope is validated before import;
bare legacy handles, unknown versions and expandable segments are rejected.
Storage bounds, view offsets, shape/stride, readiness and ref-counter identity
are checked. Tensor bindings must remain immutable across export and import.

Registration reserves the client/key before exporting. Each export credit belongs
to its attempt immediately. Only `NeverImported` permits one repayment, verified
on the original counter descriptor changing from one to zero. Missing/replaced
files, unchanged counters, exceptions or ambiguous replies retain the attempt;
unregister cannot claim cleanup. Limits are 64 attempts, 4,096 tensors/credits per
attempt and 1 TiB of conservatively charged storage. Exact storage aliases in a
single-device batch share the byte charge, while export credits remain distinct.

`ipc_credit_evidence` snapshots handoff; `retained_credit_evidence(instance_id)`
observes held original descriptors for failed attempts. Output is bounded to
4,096 credits / 4 MiB. It cannot repay a credit or prove device drain. See the
[registration contract and original evidence](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/transport.md#decision).

### Restore and source ownership

Restore reserves an operation identity before submission. The Manager claims
it before consuming leases and installs source owners before publishing a grant.
A lost ACK retains that identity and is never retransmitted. Raw DRAM Restore
runs in the native client through OrbiTE; SSD, encoded and mixed routes keep
Manager workers. This is a planned route, never a retry after local failure.

The executor claims before reading the pointer-free plan. Layer events establish
consumer readiness; a separate whole-operation drain permits `Drained`. The
Manager then releases sources and publishes `Reaped`; client acknowledgement
permits record reuse. Unknown completion retains tensors, mappings and budget.
See [engine-local Restore](engine-local-restore.md) for transitions and bounds.

### Publish and query control

Publish captures the original `GpuRegistration` at producer handoff. Native and
Core owners recheck the client, instance and generation; queued work cannot bind
a replacement registration. Holding exporters preserves lifetime, not authority.
Allocator-issued page generations are a separate missing upstream contract.

`prepare_save` retains immutable source requests; `submit_saves` batches them.
Prepared, queued and caller-held completed handles share a process cap of 1,024
operations / 32 MiB of request metadata, including spare capacity and merge
scratch. Physical payload and registration budgets remain separate.
`SaveNotSubmittedError` proves rejection before worker launch, or rollback of the
entire batch. Other attachment, submission and wait errors retain the page hold.
Dropping a handle or timing out `wait_save` cannot cancel submitted bytes.

The Manager serves other requests while Publish waits for copies to drain and
host insertion to be queued. Only sealed blocks become query-visible. Reply loss,
worker unwind or failed stream/codec synchronization keeps the accepted owner
pending. Client send errors release pages only when they prove non-submission;
ambiguous replies or post-submit errors retain them even after Manager exit.
The client closes descriptor admission before parking such an outcome.

Publish requires an authenticated Manager pidfd and one service recipient. It
waits on its own reply eventfd, with a 10 ms queue recheck for lost wakes. The
local activity probe reports active/retained calls without waiting behind the
blocked lifecycle lock. Liveness and overdue warnings are diagnostics only.

Rust `CacheClient` owns queries. Python builds one immutable `BlockHashes` batch;
views share its allocation and blocking native calls release the GIL. Metadata
`Candidates` reserve no payload and grant no Restore lease. `read_recovery`
validates complete leased coverage of compiled demand. Prepare/Warmup targets
DRAM and has separate [preparation ownership](request-preparation.md).

Pending queries are bounded to 128 per session / 1,024 globally, with a 60-second
reply lifetime. Revisions can change hashes or wait policy while preserving
instance/request/group identity. Old polls/cancels cannot consume replacements.
Cancellation stops interest; submitted reads still drain. Undelivered results
release leases, while delivered reservations move to the Restore owner.

### Host and lifecycle boundary

Adapters derive `/tmp/orbitkv-<addr-port>.sock` unless configured otherwise.
Experimental TP query control uses sealed targets and bounded Rust fan-out;
its same-host limits are in [the adapter guide](adapters.md#tp-shards-and-host-boundary).

Shutdown atomically closes Publish admission and drains accepted continuations.
A full lifecycle header and payload mark acceptance; partial frames may close
without dispatch. Accepted operations keep their dispatch and bounded response
write, then observe the latched close. Instance cleanup uses exact instance,
rank and GPU UUID keys, drains queues, then releases imported mappings. The
client separately drains local operations; session closure cannot release an
active grant. `peer_alive()` observes the original pidfd and never reconnects.

## Transfer engine

The default build enables CUDA 13 and the exact
[`orbite` 0.1.0-beta.1 crate](https://crates.io/crates/orbite/0.1.0-beta.1).
Its registry checksum is locked; see the [release gate](releases.md#build-a-candidate).
OrbitKV consumes its Rust API and shares `cudarc` 0.19.8. The concrete
`TransferEngine` owns registrations, epoch-bound peer handles, file segments and
stream bindings; registration permissions do not replace source grants.
OrbitKV builds a bounded typed `EngineConfig` from Manager settings; it does not
forward legacy `MC_*` configuration or read `ORBITE_CONF`. Unknown native batch
states or errors are `Unquiesced`, never success or permission to release owners.

| Setting | Effect |
| --- | --- |
| `--transfer-transport auto` (default) | `Transport::Unspec`; OrbiTE policy selects |
| `--transfer-transport tcp` | Explicit TCP hint |
| `--transfer-transport rdma` | Requires `orbite-rdma` and RNIC/port/GID configuration |
| `orbite-gds` | Compiles cuFile support; loads it only for configured GDS |
| No backend features | Starts no native engine; byte movement returns typed `Unavailable` before worker CUDA setup |

HP-TCP, SHM, NVLink, MNNVL and automatic failover are off. There is no local-copy
or native SSD fallback. `scripts/check-backend-features.py` checks the optional
native-code boundary.

| Resource | Bound |
| --- | --- |
| Engine registrations | 4,096 regions; configured pool bytes plus 2 GiB for persistent tensor/codec registrations |
| Peer batch | 16,384 descriptors / 64 MiB |
| Outstanding peer work | 32 batches / 64 MiB |
| Process pool registration | 1,024 pools / 1 TiB |
| GPU restore admission | 128 per device |

Uncertain owners retain these charges. OrbiTE owns slicing and pipelining.
Shutdown fences both directions; failure retains engine, backing and admission.

## Peer restore into GPU

Demand and WaitForFullPrefix queries retain immutable source descriptors and
authorization tickets without requester pinned payloads. Restore binds those
ranges to registered GPU layers: raw bytes land in final pages; encoded bytes
land in bounded codec inputs for OrbitKV validation/decode. Prepare/Warmup keeps
a host destination and populates DRAM. GPU restore does not advertise a DRAM copy.

Imported storage is registered once for its lifetime; aliases share the owner
retaining the actual tensor and CUDA context. Pointer-only Rust layouts cannot
execute peer restore. Release fences new loans and unregisters only after all
transfers and importing contexts drain. Failed cleanup retains backing and charge
without another unregister attempt.

Each source grant submits one batch of all destination ranges within the peer
limits. Larger plans use bounded batches against the same registration. The
source epoch comes from the grant; membership/deadline are rechecked before each
batch. Authorization retries are pre-submit only; failed/uncertain READs are not
replayed. Codec inputs use a persistent worker pool capped at the smaller of
64 MiB and half its codec budget. Decode finishes before input bytes are reused.

Unknown drain retains tickets, tensor/input mappings, query/device admission and
completion, and fences the worker. OrbiTE stages CUDA bytes for TCP and by default
RDMA. `--orbite-rdma-gpu-direct` requests strict GDR registration and requires
`MLX5_SCATTER_TO_CQE=0`; failure never silently stages or replays.

### Scheduling metadata

| Operation | Intent | Priority |
| --- | --- | --- |
| Demand / WaitForFullPrefix restore | `foreground_get` | High |
| Prepare / Warmup | `background_prefetch` | Low |
| SSD save / write-back | `checkpoint` | Low |

Checkpoint persists sealed KV state; placement/eviction remain OrbitKV decisions.
Read deadlines survive queues, authorization and READ descriptors without reset.
Shared reads require matching deadline and placement. Grants can shorten a budget.
Peer gRPC carries intent and remaining milliseconds, capped at 30 seconds for
source SSD materialization; the requester keeps its original absolute deadline.
Write-back shares one 30-second budget from queue admission across its extents.
Local copies have stream ordering, with no QoS fields in the consumed API.

CPU/TCP priorities are advisory; the API has no per-priority dispatch statistics.
Tests establish mapping, policy consumption and bytes, not priority ordering.
OrbiTE `nic_load_stats` supplies RDMA load metrics. KV-path codec/SSD/Restore
observations remain separate; replacing pure transfer-leg timing needs an engine
API. See [cost observation boundaries](architecture.md#planning-and-cost-observations).

### Peer departure

The etcd watch installs epoch, incarnation and endpoint together. Removal or
replacement revokes admission before cleanup, fencing discovery, outstanding
authorization and deferred Restore plans. Snapshot repair suspends admission;
an unchanged authoritative snapshot preserves identity. Local lease expiry
retires all cached peers.

Each requester opens its own `RemoteSegment`; handles never cross engines. Its
cache admits 1,024 identities, including retiring/query-held identities, and
1,024 operations per identity in addition to transfer budgets. One retirement
worker sweeps every 100 ms and waits at most 100 ms per
`close_segment(endpoint, epoch, timeout)` call. Native cleanup cannot be
preempted, but holds no admission lock and creates no extra cleanup workers.

`Ok` releases the retired cached handle and native identity capacity. Timeout or
`Unquiesced` retains grants, leases, pins, descriptors and charges. Repeated calls
only wait for the recorded retirement result; uncertain native destruction is
never retried. A new authoritative epoch can open; stale grants remain fenced.
Whole-engine shutdown releases only with a physical receipt. Ticket-release RPC
retries retain their independent completion ownership.

## SSD file transfers

OrbitKV owns file creation/sizing, placement, eviction, immutable extents and
index publication. OrbiTE sees byte ranges through retained `/proc/self/fd`
inodes, so pathname replacement cannot redirect admitted I/O. Offsets start at
file base zero. Successful complete writes alone become visible; flush is not an
fsync durability promise. Failed/short READs never become hits.

Each extent becomes one `TransferRequest::file_vectored`. Gaps and tails use
`Padding`: WRITE emits zeros, READ discards. Whole host extents keep 512-byte
alignment and GPU extents 4 KiB alignment; local ranges may be unaligned. OrbiTE
owns gather/scatter and staging. All-padding vectors need no local buffer; fan-out
to different destinations needs separate vectors. Generations never mix.

`--ssd-backend uring` uses host materialization; `auto` also permits GPU vectors
through engine CUDA staging/io_uring. `cufile` requests native-only GDS and
requires `orbite-gds`. Explicit GDS padding returns Unavailable without fallback.
`gds_enabled` is false unless cufile is configured; the driver loads lazily on
eligible submission. Compiling GDS alone does not load cuFile.

Per store, 32 operations include queued and retained work. Each vector admits
64 MiB / 1,024 elements including padding. GPU workers hold at most two receipts
and one active WRITE. There are at most 256 shards; queue/inflight settings must
be 1–1,024. Tensor/codec registrations persist through drain. Unknown drain or
failed cleanup retains fd, engine, backing, extent lease, completion and admission;
failed writes remain unpublished and cannot be overwritten. Cleanup is not retried.
DRAM representations still initialize padding exposed through host peer exports.

Real-file CI checks ext4 O_DIRECT and tmpfs bytes, padding, tails, short reads and
ownership using a labelled CPU host-pin shim. GPU source gates remain unrun:

```bash
ORBITKV_SSD_TEST_ROOT=/path/to/test-owned/ssd cargo test --locked -p orbitkv-core --test ssd_engine -- --ignored --exact cuda_staged_file_roundtrip --test-threads=1
ORBITKV_SSD_TEST_ROOT=/path/to/test-owned/ssd cargo test --locked -p orbitkv-core --features orbite-gds --test ssd_engine -- --ignored --exact native_gds_file_roundtrip --test-threads=1
```

Native GDS also needs same-run cuFile/nvidia-fs route evidence. See [GPU storage](gds.md).

## Local GPU↔DRAM copies

OrbiTE `LocalCopyEngine` binds once per worker stream for Manager transfers,
codec payload movement and native-client raw Restore. OrbitKV supplies checked
block/layer descriptors; `direct`/`kernel` select OrbiTE DMA/kernel behavior.
Unowned layouts fail binding. Codec compute remains in OrbitKV.

A window admits 1,024 descriptors / 128 MiB; larger layouts drain preceding
windows first. Each binding has eight slots / 256 MiB payload-and-descriptor
budget; the client retains at most eight completions. One physical pinned pool
has one `PinnedHostRegistration` retaining its mapping; canonical `HostMemory`
views additionally retain the allocator slot. Tensor/codec owners and admission
stay held through physical drain, including partial submissions.

Layer events follow that layer's submission; whole-operation completions stay
held until publishing `Drained`. Existing producer waits and worker-stream order
supply readiness. Overlapping active local-copy loans, including read aliases,
must drain before reuse. OrbiTE's general host registration follows the RDMA
profile; it has no separate local-only registration profile. The default staged
profile needs no local-copy MR.

CPU CUDA-shim gates check bytes, strides, layer events and uncertain completion.
They do not qualify GPU operation. Installed direct/kernel reproduction uses
`tests/release/test_installed_orbite.py` with `ORBITKV_TEST_INSTALLED_ORBITE=1`
and `ORBITKV_TEST_STRICT_SHUTDOWN=1`; see [release gates](releases.md).

## Remote operation choice

Demand restore uses OrbiTE READ into GPU pages; DRAM-tier fills keep host
destinations. Native vLLM NIXL/MultiConnector and SGLang disaggregation own live
P/D. See [P/D composition](pd.md). Proactive placement, replica retry and
bundle-aware publication are not current guarantees. Adapters validate complete
[recovery requirements](hybrid-recovery.md); wire metadata alone does not prove
model-state completeness.

## Remaining qualification

Current GPU/RDMA/GDS and installed profiles are unqualified. Timeout, partial
submission, permanent peer loss and page reuse require physical fault controls;
CPU/TCP, io_uring and CUDA-shim passes do not supply hardware evidence. Follow
[fault qualification](fault-qualification.md) and the [completion plan](completion-plan.md).
