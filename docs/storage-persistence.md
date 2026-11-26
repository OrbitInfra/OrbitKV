# Persistent SSD cache

Persistence is an opt-in cache contract, not a database or a serving qualification.
The default still discards SSD state when the Manager starts. With
`--ssd-persistent`, a restarted Manager can reuse committed extents with exactly
the same storage namespace and state key. OrbiTE remains the only payload byte
mover. OrbitKV owns the journal, extent allocation and durability ordering.

The implementation is opt-in. Required CPU process-crash controls exercise
ext4 and tmpfs; real NVMe serving and power-loss behavior remain unqualified.

## Commit and overwrite protocol

Each shard remains a bounded, aligned ring. A checksummed, bounded sidecar
journal covers the complete shard set. Data files are exclusively locked before
resize or reuse; a second Manager cannot open the same files. The journal binds
the shard ordering and inode identities to a stable cache UUID. Each record also
names the writing Manager incarnation, independently of peer epochs.

Before parsing or resizing, acquire and immediately release an exclusive Linux
file lease on each shard. A Manager flock alone is insufficient: older kernels
can leave io_uring file users alive after process exit (see the
[kernel exit fix](https://github.com/torvalds/linux/commit/044e5f30b3e0efe96dac1b75e15621bfbdb0a767)).
Linux [write leases](https://man7.org/linux/man-pages/man2/F_SETLEASE.2const.html)
reject other read/write opens, including descriptions still retained by kernel
requests. A busy or unsupported lease fails startup before modifying metadata
or payloads; retry after cleanup. Data files must belong to the Manager user
(or it needs CAP_LEASE). The private-filesystem/cooperating-writer requirement
still applies; a lease is not authorization against arbitrary external writes.

This persistence profile admits `uring` and `auto` (engine io_uring, with optional
engine CUDA staging). Explicit `cufile` persistence is rejected: this PR does not
establish the GDS driver's process-exit/file-reference lifetime contract. It
does not silently switch transports. Nonpersistent GDS remains unchanged.

For one extent:

1. Reserve through the existing ring allocator. A reader, active writer or
   undrained transfer prevents reuse of its range. This reservation is not a hit.
2. Append and sync a reservation record **before any payload write**. It advances
   the durable logical head and invalidates every older extent below the new
   tail. A crash after this fence may lose cache entries, but cannot resurrect
   overwritten ones. Failure prevents submission.
3. Write all stored bytes and alignment padding through OrbiTE. Wait for its
   physical receipt; timeout, cancellation and Unquiesced are not receipts.
4. `fdatasync` the retained shard inode. O_DIRECT alone is not durability.
5. Append the complete commit record, with the **source** CRC references, and
   `fdatasync` the journal.
6. Publish the SSD replica in the in-memory index and inventory.

The immutable source is checksummed before submission: host sealing computes
one CRC pass, or an already sealed source supplies its reference; a GPU source
uses the existing codec CRC owner once. Padding CRCs describe zeros emitted by
OrbiTE. The source, registration and extent stay owned until the write drains,
so the reference binds to the submitted bytes. Neither persistent nor ephemeral
writes read the file back. Data sync precedes commit sync: a committed record
cannot intentionally precede its data. A silent device miswrite is detected by
verification of the subsequent read, including the first recovered hit, and
becomes a counted miss. This removes read-back I/O without treating a source CRC
as proof that the device stored the bytes correctly.

Host Save completion may precede asynchronous SSD write-back, as before. Only
the SSD commit establishes persistence. An explicit flush waits for prior SSD
work; it does not turn a failed write into a durable entry.

Sync or append failure fences further persistent writes for that Manager. A
failed, drained write has no published SSD replica and its reservation is
reclaimable. An uncertain transfer retains its actual extent, buffer and native
admission. It is never converted into a failed-but-free extent.

## Format and trust boundary

Frames have bounded lengths and CRC32C-protected metadata. The versioned header
contains the cache UUID, shard identities/capacities, ring alignment, durable
heads and sequence watermark. Every extent record contains:

- format version, cache UUID, writing Manager UUID and monotonic sequence;
- complete `StateKey`: storage namespace and the full group-framed content hash;
- shard, logical begin, aligned length and physical offset;
- every slot's segment sizes, NUMA hint and encoding metadata;
- optional original publication and expiry times in Unix milliseconds;
- reservation/commit/invalidation kind, stored-byte IEEE CRC32 plus bounded
  layer/segment CRC references for commits, and a
  frame checksum covering the record.

The namespace binds model revision, layout, computation scope and representation;
recovery never derives it from a path or model name. Checksums detect accidental
corruption, not a malicious writer with filesystem access. Use private local
filesystems or the [managed namespace partitions](#shared-root-namespace-partitions) described below.

## Recovery and first hit

Parse the entire bounded journal before exposing any recovered entry. Replay
durable heads and reservations before accepting commits. Newer reservations
invalidate overlapping old generations even when the replacement never commits.
Uncommitted reservations are orphans, not hits. A complete record that survived
a process kill before its sync may be recovered if data and metadata validate;
the caller received no durable acknowledgement for that write.

A damaged append tail retains the valid prefix. Recovery scans the bounded
remainder for a valid frame: finding one means mid-journal corruption and a
**cold start**, never resynchronization. Only one invalid/incomplete frame ending
at EOF or followed by zeros is discarded. Truncate to the prefix and `fdatasync`
before any append or checkpoint. Count both the reason and the discarded bytes.
An unsynced fence never admitted its write; an unsynced commit is an orphan.
Recovered payload verification below additionally catches ambiguous overlap.
A bad header, unknown version, invalid identity, impossible sequence, geometry
or encoding still cold-starts the set. Recovery never trusts later records after
corruption.

Format 3 preserves fixed data TTL. Recovery removes expired records before
exposing candidates and checkpoints those removals; checksum completion checks
the deadline again before publishing inventory. Neither restart nor SSD-to-DRAM
materialization restarts a lifetime. An expired active write remains reserved
until physical completion, then finishes without a visible replica. Logical
expiry of a committed extent keeps existing reader pins. See the
[clock assumptions and cleanup bounds](cache-policies.md#fixed-data-ttl).
Earlier format versions start cold; there is no compatibility migration.

Optional [SSD watermarks](cache-policies.md#ssd-highlow-watermarks) use the same
invalidation record. Each victim is durably removed before inventory retirement
and FIFO reuse; an invalidation error retains the range and fences persistent
writes. Held readers and active writes block their shard. Maintenance does not
truncate files, change alignment or introduce another persistent format.

Valid metadata initially creates **unverified** entries. They are invisible to
queries and peer inventory. A bounded background pass pins each current
generation and verifies the entire stored-byte CRC through OrbiTE. Only a
successful, drained read makes that entry visible. Failure counts a drop;
unknown drain keeps the pin and verification backing. Concurrent eviction may
drop an unverified entry; verification cannot publish a different generation.
[Every subsequent SSD read](storage-formats.md#read-time-integrity) is also verified. The recovery pass controls visibility;
it does not replace read-time integrity.

## Bounds, GC and changed capacity

Let `E = min(total configured shard capacity / alignment, 1,048,576)`;
alignment is the minimum legal extent. Entry admission is capped at `E`, metadata
at `clamp(E * 8192, 64 KiB, 1 GiB)`, the journal at four times that metadata
budget (at most 4 GiB), and a frame at one quarter of the metadata budget,
clamped to 16 KiB–1 MiB. Actual allocations are charged on use, not preallocated.
Large records can exhaust the byte budget before the entry budget: admission
returns `WouldBlock`, never truncates. These are limits on cache availability.

On a capacity shrink, replay uses the larger of the recorded and current
geometry's finite metadata budgets; `replay_limits` reports that startup cost.
Before checkpoint or publication, drop extents outside the new capacity and any
remaining metadata excess (`capacity_metadata`), then use the new admission
bounds. This preserves fitting entries without exceeding the configured live
cache budget. The global maxima above also bound replay of the old geometry.

Parsing gets 30 seconds plus one second per 16 MiB of journal budget. Background
verification gets 180 seconds plus one second per 64 MiB of payload capacity,
capped at 24 hours. Checks occur between operations; a blocked kernel sync cannot
be forcibly timed out. The journal record charge excludes the duplicated ring
index and recovery list; those have the same finite entry/slot bounds. Tail
scanning uses at most one maximum frame plus 64 KiB and separately bounds CRC
work. One 1 MiB pinned scratch allocation is charged to the existing host pool
and serialized across file verification. Unknown drain retains it and its extent;
it cannot be reused. Recovery reports the derived limits, metadata charge,
parsed/verified/discarded bytes, entries, drops and elapsed time.

Before journal growth reaches its limit, serialize the current bounded live
records and durable heads to a temporary checkpoint, sync it, rename it over
the journal and sync its directory. A checkpoint includes outstanding
reservations; compaction cannot undo their overwrite fences. Append operations
are serialized with checkpoint replacement. A failed replacement fences writes.
The old or new complete journal is sufficient after a crash. Temporary files
are not recovery input.

Ring overwrite reclaims orphan data; journal checkpoints reclaim obsolete
metadata. In-memory readers still protect physical ranges during eviction and
verification. No GC operation bypasses the native drain receipt.

When capacity changes, discard extents outside the new physical bounds before
serving. Rebase surviving extents onto their physical offsets, checkpoint the
new ring geometry durably, then resize the shards. Changed shard identities or
ordering start cold. A smaller budget cannot admit more than its configured
capacity. Metadata disk space is separately bounded: one journal and one
temporary checkpoint, in addition to the configured payload budget.

## Verification and remaining gates

The CPU gate uses real child Manager processes and SIGKILL at the data-write,
data-sync, partial-record, unsynced-record and post-publication boundaries.
Restarts require correct bytes or a counted miss. Controls also corrupt metadata
and data, change namespace/version, wrap during overwrite, hold readers and
native completion, shrink capacity, exhaust disk space and force checkpoint
replacement. An independent retained file description also blocks restart until
released, even after the Manager's flock disappears. Required io_uring runs set
`ORBITE_REQUIRE_IO_URING=1` and fail if
the engine cannot use it.

ext4 tests exercise local filesystem ordering and O_DIRECT. tmpfs exercises
process failure, framing and ownership, but not persistent media. SIGKILL is a
process-failure test; it does not remove the kernel page cache, simulate a power
cut, validate device flush firmware or qualify an NVMe power-loss guarantee.

The installed gate waits for a real NVMe and immutable vLLM 0.33.0 and SGLang
0.5.23 model profiles: restart the Manager under serving load, require restored
bytes and model-output checks, and retain the power-loss caveat. No hardware
has been requested or qualified by this PR.

## Shared root namespace partitions

Managers serving different storage namespaces can write concurrently under one
operator-provisioned directory. Configure each Manager with one existing root,
the exact namespace from engine registration, and its own payload budget.

```sh
orbitkv-cache-manager --ssd-cache-path "$ROOT" --ssd-persistent \
  --ssd-shared-namespace "$STORAGE_NAMESPACE" --ssd-cache-capacity 16gb \
  --ssd-shared-root-capacity 32gb
```

The Manager creates an owner-only directory named `orbitkv-ssd-<namespace>`
under the root. Shards start at `shard-000000.dat`. Each partition has one
active Manager and a fixed shard count. Changing that count requires a newly
provisioned partition. Starting with the wrong count, a private SSD path
inside this reserved directory, or existing data with a missing or unreadable
partition header fails before reuse. Keep the data files and journal together.
Other partitions can continue writing, reading and reclaiming their own extents
during recovery.
A replacement using the same partition must pass the file-lease drain check and
verify recovered bytes before publication.

SSD write admission compares the full storage namespace before reserving queue
or ring space. The journal binds the namespace in its header and checks it on
each record. Opening a valid journal with a different scope fails before
checkpoint or resize. A foreign key with an identical content hash misses the
local SSD index. DRAM, peer grants and router credentials have their own
contracts. Operators authenticate tenants and assign their Managers and storage
namespaces. Keep root entries under trusted administrative control throughout
operation and use a filesystem that implements the required locks, file leases
and sync operations.

The example admits two 16 GB partitions under a 32 GB root payload limit.
Each partition reserves the total of its aligned shard lengths, even while its
Manager is stopped. Leave filesystem space for journals, checkpoints and other
files. A root without a configured limit relies on operator provisioning.

To enable the limit on an existing root, first upgrade every Manager that can
open it. Run capped Managers under a service identity that can read every
partition's owner-only directory. The Manager rejects shards with extra hard
links. Keep each managed shard at its original path and avoid exposing it to a
private Manager through a bind mount. The first Manager to specify a limit checks
existing shards and publishes the root's `.orbitkv-ssd-capacity` file using
Linux `O_TMPFILE` and a hard link. This requires support for both operations
on that filesystem. The Manager publishes the limit after checking partition
ownership and recovering its journal, before resizing any shard. Keep the
policy file with the root. Every subsequent Manager reads the limit, including when its command
line omits the flag. A conflicting value, corrupt policy or over-budget request
fails startup. If the existing shards exceed the intended limit, shrink their
partitions before enabling it.

Managers serialize admission and resizing under a directory lock. A restart
charges each requested shard once and keeps any excess old length until the
resize completes. If a Manager dies partway through startup, the next Manager
accounts for the shard lengths left on disk. A successful shrink makes space
available to another namespace. Reads and writes run without the root lock.

The Linux process gate on ext4 and tmpfs exercises concurrent admission, killed
writers, byte-checked reopen and shrink. It also writes different bytes under
the same block hash in two namespaces and holds a read across wrap pressure in
one while the other reuses its space. Installed cross-host and 3FS profiles
still require separate runs.

### Reclaim an offline partition

To release the space reserved by a stopped Manager, run this one-shot command
on the host that owns the shared root. It evicts **all** SSD blocks in the exact
storage namespace you name. Other namespaces can continue serving.

```sh
orbitkv-cache-manager reclaim-ssd-partition --root "$ROOT" --namespace "$STORAGE_NAMESPACE"
```

The command holds the same root lock as Manager startup. It takes every shard's
exclusive lock and checks Linux write leases on the shards and the journal.
A live owner or a file description left open after the owner exits blocks
reclamation before any bytes change. The journal must bind the named namespace
to the device and inode of every shard.

The command keeps the journal header and syncs the removal of its cache records
and the journal directory before truncating the first shard. It syncs each
shard and reports how many payload bytes it released. The shard files retain
their inodes. If the command stops partway, admission still charges the shard
lengths on disk. Rerun it while the partition remains offline to release the
rest. A second completed run on an empty partition reports zero. If a Manager
has since repopulated the namespace, a new invocation evicts that data too.

The Linux process gate races admission against a paused reclaimer and kills
it after journal retirement and after the first of two shards is truncated.
It also checks that a neighboring Manager can still read its data and another
namespace can use the released capacity. Cross-host fencing and physical host
or volume loss require separate qualification.

### Relinquish a live partition

On the Manager's host, add a positive wait in seconds to ask a serving Manager
to relinquish **all** SSD blocks in the named partition. For example:

```sh
orbitkv-cache-manager reclaim-ssd-partition --root "$ROOT" --namespace "$STORAGE_NAMESPACE" --wait-for-owner-secs 30
```

The command holds an exclusive lock on the namespace directory. The Manager
checks that directory during its regular cache maintenance, even when watermarks
are disabled. It closes SSD admission and withdraws its SSD replicas from the
inventory. Existing DRAM copies and peer service remain available. An already
admitted SSD read keeps its backing through completion. `flush_all` still waits
for earlier writes to reach a logical outcome. Startup for the same
namespace waits for root admission and rejects an active release request.

For a remote READ, the source reads the SSD block into pinned memory before
issuing the grant. The transfer ticket keeps those bytes registered and
charged until the requester completes the READ and releases the ticket.
The file can therefore drain and its capacity can be reused by another
namespace while the earlier grant is still live. Once the source has
withdrawn its SSD replica, a fresh grant for that residency is rejected.

The command retries ownership checks until the wait expires. It releases the
root lock between attempts so other namespaces can start within the remaining
budget. Once the old writer, file descriptions and kernel I/O drain, it uses
the same journal and shard sync sequence as offline reclamation. The wait
bounds lock contention and drain, and a sync already in progress runs to
completion. Success reports the payload bytes actually released.

A timeout or process exit releases the request lock. If the Manager already
observed the request, its SSD tier stays detached. Retry the command to finish
reclamation. To attach the partition again, restart its Manager with an
available reservation. An older Manager that cannot observe the request keeps
its shard locks. Stop it before using offline reclamation.

## Exclusive handoff after owner loss

On Linux, a replacement Manager can reopen a surviving private shard and its
journal once the old writer and its file users have drained. It joins under its
own identity and advertises each recovered extent after checksum verification.
A requester must rediscover the extent and open a new transfer window, even if
the replacement listens at the old address. Unverified and corrupt extents
stay out of its inventory.

The private-shard CI case runs three Manager processes across the handoff, an
isolated etcd and the real OrbiTE io_uring and TCP paths. It kills a writer
with an issued grant, keeps the requester running, and reuses the peer address
under a different node ID. A verification pause and a damaged neighboring
extent check visibility. The replacement rejects the old owner and window,
and the requester checks every byte in two slots obtained through a fresh
source grant.

A second case suspends the writer of a shared partition while it holds an
issued grant. The test observes the writer's leased member key disappear from
etcd and its placements disappear from the requester. The stopped process still
owns the shard. A contender on a different port fails to acquire its lock, and
the root budget rejects a reservation for another namespace. When engine
initialization fails after joining the cluster, the contender revokes its own
etcd lease before returning the storage error. If the coordinator is
unreachable, the registration expires at its TTL. After resuming,
the old Manager can read locally and rejects the same peer grant request it
previously served. Even after it exits, an open reader on the shard prevents takeover.
The test checks the data, journal and root capacity file across the rejected
attempts. Once the reader closes, the original node ID and peer address can
return with a higher membership epoch and a new incarnation. The requester
then obtains verified bytes from that owner.

A third case holds the Manager's gRPC port open during startup. It observes
the leased member in etcd, the bind error after SSD initialization and the
removal of that member by process exit. The same node ID then reopens the
partition and listens on a free port before the original TTL elapses.
The live-reclaim case pauses a cold requester after authorization and before
its first TCP READ. Another namespace claims the released capacity and writes
different bytes under the same block hash. The requester checks the original
bytes after resuming, and the transfer counters return to zero.
A fifth case keeps etcd and the TCP data path reachable and disconnects the
source gRPC endpoint after it has granted an SSD READ. The requester checks
the bytes before reconnecting. A failed release must leave the source block,
reserved bytes and requester completion slot outstanding. After reconnecting,
all three gauges return to zero and the successful fetch byte count stays fixed.
A sixth case drops traffic from the source gRPC endpoint after the source has
processed the release. Its block and byte charges have drained, while the
requester still holds one completion record. After reconnection the same ticket
releases zero further blocks, its ACK clears the record, and the verified READ
bytes are counted once.
The `ssd-process-crash` CI artifact contains separate ext4 and tmpfs directories.
Each includes `handoff.json`, `expired-owner-handoff.json`,
`control-bind-failure.json`, `live-reclaim-authorized-read.json`,
`grpc-release-retry.json` and `grpc-release-ack-loss.json`.

To run the cases with the same test-hooks binary and host-registration shim as
the persistent SSD gate, choose an existing scratch directory for `MOUNT` and
a new external results directory for `OUT`:

```sh
PYTHONPATH="$PWD/python" ETCD_BIN=/usr/bin/etcd \
  ORBITKV_REQUIRE_ENGINE_SSD=1 ORBITE_REQUIRE_IO_URING=1 \
  python3 crates/orbitkv-server/tests/process/ssd_owner_handoff.py \
  --binary target/debug/orbitkv-cache-manager --root "$MOUNT" --artifacts "$OUT"
```

This profile covers same-host process loss and exclusive access to surviving
files. Cross-host filesystem fencing, concurrent writes to the same partition,
3FS and physical host or volume loss remain open.

## Precedents

[LMCache's local disk backend](https://docs.lmcache.ai/api_reference/configurations.html),
[Mooncake Store SSD offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
and [FlexKV's SSD tier](https://github.com/taco-project/FlexKV/blob/main/docs/flexkv_config_reference/README_en.md)
are precedents for an independently budgeted local cache tier. Their support and
durability guarantees do not establish OrbitKV's protocol; the ordering,
recovery rejection and fault controls above define this implementation.
