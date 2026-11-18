# OrbitKV local global index

Each distributed Manager holds an in-memory candidate index and membership
leases. The server follows etcd membership and receives residency snapshots
and deltas directly from the other Managers. The catalog performs no networking.

`GlobalIndex::lookup` returns bounded remote candidates per medium. Entries are
qualified by runtime UUID, exact residency sequence, medium, representation and
known stored bytes. Each row reports its coverage at the installed watermarks.
Complete coverage requires matching membership and a receipt less than three
seconds old from each expected peer. Source grants validate and pin actual
storage before exposing addresses.

Index overflow withdraws the affected owner. Gaps or conflicting updates stop
the stream and require contiguous recovery or a new snapshot.
The configured budget is conservative logical accounting, not a process RSS cap.
DRAM and SSD are independent records. A batch source rejection does not mutate
the shared index; only background metadata synchronization does.

See [distributed design](../../docs/distributed-cache.md) for key layout,
publication/retry fencing, snapshot/Watch recovery and remaining production
qualification. See [deployment](../../docs/p2p.md) for commands and tests.

```bash
cargo test -p orbitkv-catalog --lib
```
