# Peer coordination

`catalog/mod.rs` owns the bounded candidate index lookup and 16 shard workers.
`catalog/sync.rs` owns per-shard snapshot/delta recovery. The
[`orbitkv-catalog`](../../../orbitkv-catalog/README.md) library owns cached
membership, fixed placement and the receiving catalog. etcd registration and
Watch live in the Manager's `cluster/` adapter; core performs no etcd I/O.

```text
DramStore insert/evict (one residency lock)
  -> per-shard index + bounded journal + sequence
  -> background snapshot/delta workers
  -> assigned Manager catalog, via cached Node ID -> endpoint/incarnation

Query -> DRAM/SSD -> candidate index -> missing shard lookups
      -> contiguous source spans -> source authorization -> OrbiTE READ
```

No inventory index or journal is allocated in standalone mode. Distributed
`EngineConfig` takes one `MembershipView`; its owner provides both the advertised
endpoint and runtime identity. There is no fixed directory-address option or
unregistered remote mode.

Positive hints have a five-second TTL and a 16 MiB LRU budget. Queries group
missing keys by shard and coalesce concurrent misses. Lookups have a three-second
RPC budget; unavailable shards preserve known evidence without proving absence.
The requester never broadcasts the whole query to every Manager.

Inventory streams have independent contiguous sequences and bounded journal
partitions. A changed target incarnation or receiver epoch triggers snapshot
recovery; ambiguous replies and overflow cannot silently mark a partial view
complete. Source pinning checks exact insertion sequences under the residency
lock, independently of catalog hints. Invalid membership disables remote work
while local cache operations continue.

The blocking OrbiTE READ owns destination memory and its release guard through
caller cancellation. Source allocations remain budgeted after timeout and release
replies have bounded retries. Orphan revocation and cross-host failure qualification
remain unresolved. See [deployment and failure boundaries](../../../../docs/p2p.md).

`export.rs` owns source admission and held allocations. `read.rs` and
`completion.rs` own requester reads and terminal release;
`segment.rs` owns bounded requester-specific handles and membership retirement; `transport.rs` holds
both the OrbiTE registration and pinned pool. The inbound gRPC adapter lives
in `orbitkv-server/src/peer.rs`. No remote SSD or engine HBM export is implied
by this module layout.
