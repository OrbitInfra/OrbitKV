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
5. Read back the complete aligned extent through OrbiTE in bounded host chunks
   and compute CRC32C, including padding. This deliberately pays read amplification
   to cover raw, encoded and GPU-originated bytes. This read-back is retained
   for durability binding; nonpersistent SSD commits now use it as well.
6. Append the complete checksummed commit record and `fdatasync` the journal.
7. Publish the SSD replica in the in-memory index and inventory.

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
- reservation/commit/invalidation kind, stored-byte CRC32C for commits, and a
  frame checksum covering the record.

The namespace binds model revision, layout, computation scope and representation;
recovery never derives it from a path or model name. Checksums detect accidental
corruption, not a malicious writer with filesystem access. Use private local
filesystems; this is not the shared-storage tier or tenant authentication.

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

Valid metadata initially creates **unverified** entries. They are invisible to
queries and peer inventory. A bounded background pass pins each current
generation and verifies the entire stored-byte CRC through OrbiTE. Only a
successful, drained read makes that entry visible. Failure counts a drop;
unknown drain keeps the pin and verification backing. Concurrent eviction may
drop an unverified entry; verification cannot publish a different generation.
[Every subsequent read](storage-formats.md#read-time-integrity) is also verified. The recovery pass controls visibility;
it does not replace read-time integrity.

## Bounds, GC and changed capacity

Let `E = min(total configured shard capacity / alignment, 1,048,576)`;
alignment is the minimum legal extent. Entry admission is capped at `E`, metadata
at `clamp(E * 8192, 64 KiB, 1 GiB)`, the journal at four times that metadata
budget (at most 4 GiB), and a frame at one quarter of the metadata budget,
clamped to 16 KiB–1 MiB. Actual allocations are charged on use, not preallocated.
Large records can exhaust the byte budget before the entry budget: admission
returns `WouldBlock`, never truncates. These are limits on cache availability.

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

## Precedents

[LMCache's local disk backend](https://docs.lmcache.ai/api_reference/configurations.html),
[Mooncake Store SSD offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
and [FlexKV's SSD tier](https://github.com/taco-project/FlexKV/blob/main/docs/flexkv_config_reference/README_en.md)
are precedents for an independently budgeted local cache tier. Their support and
durability guarantees do not establish OrbitKV's protocol; the ordering,
recovery rejection and fault controls above define this implementation.
