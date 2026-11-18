# Distributed cache design

Each Manager owns local DRAM/SSD and a covered index of peer replicas. Etcd holds
protocol identity, persistent node epochs and leased membership. Bounded owner
inventory streams carry snapshots/deltas; ordinary queries use the local index,
then obtain an authoritative source grant before OrbiTE transfers bytes.

Start with [P2P deployment](p2p.md). Current support stays at four owners;
historical two-H20 RDMA evidence does not qualify the current OrbiTE path.
Full cache HA, physical failure domains and sixteen-owner support remain unqualified. See
[compatibility](compatibility.md).

## Local global index and etcd metadata

`ResidencyInventory` is the source of residency truth. `GlobalIndex` is the
requester's local candidate index. Membership/format use etcd; block churn uses
peer inventory streams rather than per-block etcd records or lookup RPCs. No
second Catalog or directory process is required.

```mermaid
flowchart LR
  E[etcd: protocol, epochs, leased members] --> A
  E --> B
  subgraph A[Manager A]
    IA[Local covered index]
    DA[DRAM / SSD]
  end
  subgraph B[Manager B]
    IB[Local covered index]
    DB[DRAM / SSD]
  end
  A <-.->|Inventory snapshots and deltas| B
  IA -->|Candidate discovery and grant request| B
  B -->|Authoritative source grant| A
  DB -->|OrbiTE READ| D[Requester GPU Restore or DRAM fill]
```

The current format is `orbitkv/inventory-stream/v6`. Stop all Managers before
format migration. Use `scripts/migrate-metadata-format.py` to dry-run, archive
and CAS the namespace. Mixed old etcd-block/v3/v4/v5 metadata paths are unsupported.

## Owner inventory streams

Each directed session binds cluster UUID, source epoch/incarnation, requester
incarnation, scope description/digest and a random session ID. Receiver snapshot
staging remains hidden until complete pages and contiguous replay validate and
install atomically. Frame-consumption ACKs return stream byte credit; they do
not assert installed visibility.

Deltas coalesce exact state/medium identities over a contiguous source interval.
Filtered empty intervals preserve sequence coverage. Duplicate replay is
idempotent; gaps, overlap, identity conflicts and incomplete snapshots never
partially install. A reconnect can resume the same incarnation and scope at a
retained contiguous cursor. Journal overflow, retirement or a new incarnation
requires a replacement snapshot.

Stream interruption makes coverage partial; membership removal excludes the old
incarnation immediately. Source grants revalidate authoritative current ownership,
so a stale index hint cannot authorize a stale replica. Normal Manager shutdown
fences membership before inventory/lifecycle drain and stops new inventory frames.
Session counters are released on task cancellation.

### Placement lookup contract

The Manager reads one local index and membership cut for a batch. Each requested
key has a bounded list of candidate replicas and one coverage result.

| Result | What the consumer can conclude |
| --- | --- |
| `CompleteAtWatermarks` | Every expected peer in this membership cut has a synchronized view with a local receipt less than three seconds old. An empty row means no remote replica was visible in this namespace at those applied sequence numbers. A later publication can change the answer. |
| `PartialHints` | At least one admitted view remains. Returned locations may be tried under a new source grant. An empty row leaves the remote placement unknown. |
| `Unavailable` | The namespace is outside scope, membership is invalid or changing, or no usable view exists. Local cache policy can still serve or recompute the request. |

The three seconds bound **requester receipt age**. Network delay and a backlog
can make a recently applied source sequence older than that. The source emits
idle progress every second. A commit, contiguous advance or matching head
confirmation renews a receipt. Duplicate deltas preserve its original age.
Active and staged views retain the membership lease that admitted them. Revocation
invalidates both even when a node rejoins with the same UUID. The receiver
checks its original lease under the catalog lock before applying frames or
withdrawing inventory. Fence waiters use the same lease to read progress. A journal reset
preserves positive hints while clearing the resume cursor until the replacement
snapshot commits. Retiring the view also reclaims its reset state.

### Router lookups and source events

A router can query one Manager for local placements and remote candidate hints.
To track live changes, subscribe to each source whose blocks the router tracks.
Manager membership comes from the deployment. The Manager never selects an
inference worker, reserves a replica or grants access through these routes.

Enable the routes on a distributed Manager with a loopback HTTP bind, such as
`--http-addr 127.0.0.1:9091 --router-credentials /run/orbitkv/router.json`.
The file is a JSON array with 1–64 entries of this form:

```json
[{"namespace":"<exact orbitkv:v2 storage namespace>","token_sha256":"<64 lowercase hex characters>"}]
```

Generate a random 32-byte token and encode it as 64 lowercase hexadecimal
characters. The fingerprint is SHA-256 of those 64 ASCII characters, without
a trailing newline. Send the token as `Authorization: Bearer <token>` with
`Content-Type: application/json`. Each fingerprint selects one exact namespace
within the Manager's configured scope. Restart to rotate credentials. For a
router on another host, terminate TLS at a trusted proxy and forward only these
two paths. Other Manager HTTP endpoints are operational controls.

1. Send `POST /v1/router/lookup` with `{"hashes":[[1,2],[3,4]]}`, using the
   stored, group-encoded hashes. The response contains `source` (peer endpoint
   and incarnation), `local` (resident DRAM/SSD rows), one `placements` row for
   each input hash, and `cursor`. Each placement carries `peers` and
   `peer_coverage` under the lookup contract above. Local rows and the source
   sequence are sampled under one lock. Peer hints have their own applied
   watermarks and freshness. Refresh peer hints by looking up again or following
   those sources directly.
2. Send `POST /v1/router/events` with `{"cursor":<cursor>,"wait_ms":1000}`.
   The response returns `changes` and the next `cursor`. Upsert or remove each
   `(source incarnation, hash, metadata.medium)` according to `present`.
   Events cover the authorized namespace. Bootstrap only the hashes you track
   and retain their events. The journal coalesces repeated changes to a
   residence within one page. An empty page can still advance through changes
   in another namespace. The source-wide sequence reveals aggregate activity.
3. Install the complete page and its returned cursor together. Retrying the
   same cursor can return a longer interval if the source has advanced. Apply
   rows by sequence, and discard responses from a superseded request or stream.
   Poll each source serially. On `409 resync_required`, drop that source's
   tracked rows and relookup the tracked hashes. When bootstrapping in multiple
   batches, replay from the **earliest** returned cursor on the same stream and
   retain each batch's baseline sequence, including for absent rows. Apply only
   changes newer than that baseline. Restart on any stream change or gap.
   A cursor binds cluster, source epoch/incarnation, membership generation,
   exact namespace and the last source sequence. `503 membership_unavailable`
   suspends use of that source until a fresh lookup succeeds.

Each request accepts at most 320 KiB of JSON with a one-second body deadline.
Lookup accepts 1–128 hashes and 64 KiB of namespace plus raw hash bytes.
Each Manager admits eight lookups and eight event polls independently,
with `429` on saturation. A response holds its slot through the lifetime of
its encoded bytes, capped at 3 MiB per response. A poll waits 0–1,000 ms
and reads at most one existing inventory batch (1,024 records or 512 KiB
before JSON encoding). The router adds no persistent subscriber queue or
second event journal. Source authorization and OrbiTE drain still govern
every payload transfer.

## Scope and bounded resources

Omitting scope selects all namespaces. Repeat `--metadata-namespace` with actual
`orbitkv:v2:<64 lowercase hex>` namespaces for an exact allowlist, or use
`--metadata-empty-scope` for no remote namespaces. Scope changes require restart
and full bootstrap; old cursors/late frames cannot establish new coverage.
Filtering does not provide tenant authentication or remove all-to-all sessions.

Frames are limited to 1,024 records and 512 KiB encoded/decoded content. Current
bounds include 1 MiB messages/per-stream credit, 32 MiB aggregate outbound credit,
128 sessions per direction, two served/received snapshots, 64 fence awaiters and
16 MiB/s source pacing with a 1 MiB burst. Reconnect backoff is 100 ms to 3 s.
Each follower holds its receiver slot across reconnects. With more than 128
remote owners, some streams can wait indefinitely while the admitted peers stay
present. A fresh requester cannot promise complete coverage in that topology.
Scopes allow at most 256 namespaces and a 64 KiB encoded Open.

`--index-budget` charges active and hidden staging views together. Incomplete
coverage cannot support a cluster-wide absence claim. Evicted/retired owner
views stay excluded until complete replacement installation. Local cache
operations remain available for namespaces outside remote discovery scope.

## Synchronization and diagnostics

`POST /cache/sync` drains submitted local saves and, in distributed mode,
returns an inventory fence with protocol, cluster UUID, source epoch/incarnation
and sequence. It does not wait for requesters. Standalone mode returns
`local_flush_complete` without fabricating an inventory fence.

A controlled requester uses `POST /cache/metadata/await` with the fence, exact
scope digest and timeout up to 30 seconds. Success requires a committed matching
owner view and valid membership. Ordinary queries never execute this barrier.

`GET /cache/metadata` exposes membership revision/validity, scope and coverage,
active/staging/accounted index bytes, expected/installed owner views, stream
bytes/frames, session counts, journal state and queue high-water marks.
Ordinary publication-to-install visibility is distinct from serial harness
verification time. Inspect membership traffic separately from stream traffic.

## Residency, proximity and access paths

A candidate names an exact state key, representation, stored size, medium,
source endpoint and incarnation. Matching hashes do not make incompatible
engine byte layouts interchangeable. A complete local miss is meaningful only
inside the declared scope and membership cut.

Remote DRAM transfers use a source export held through completion. Remote SSD
is read/staged by its source owner before export. The current Manager route is
remote DRAM/SSD → OrbiTE READ → destination GPU (or DRAM for cache-tier fills). A GPU-pointer
RDMA component test does not imply that this route directly exports live engine
HBM or qualifies GPU-direct operation.

## Application and transport boundary

| Operation | Implemented path |
| --- | --- |
| Block discovery | Complete local global index; no foreground network query |
| Inventory publication | Bounded Manager inventory streams |
| Index synchronization | Covered inventory snapshots and deltas |
| Membership/incarnations | Background etcd registration, lease renewal and Watch |
| Source grants/completion | Bounded batched OrbitKV gRPC |
| Remote payload | OrbiTE READ under source grants |
| P/D payload | Inference engine native transport |

Admission, incarnation/generation fences, idempotent completion and retry
ownership belong to OrbitKV. OrbiTE owns registered-memory transfer and native
completion. SEND completion, receiver ACK and drained payload READ are distinct
events. A metadata lease or dead requester cannot prove old READs have drained.

Keep one active READ and at most one authorization lookahead. Submitted work
owns its source guard and destination buffers until native drain. A bounded
completion owner retries release until acknowledged. Index replacement must
preserve these implemented limits and ownership.

Live P/D uses each inference engine's native transport; OrbiTE handles Manager
sharing. See [P/D composition](pd.md). The custom router is removed.

## Transfer lifetime and resource control

Query admission owns byte credits and destination reservations. Source grants
own the authorized representation and backing bytes. Index hints, membership
leases, network credits and source memory lifetimes are separate contracts.
Cancellation, timeout or requester disappearance cannot prove physical DMA drain.
Unknown work retains its buffers/registrations until a valid termination proof;
bounded retention is observable and cannot be described as completed reclamation.

Keep replica discovery in Catalog, source authority in existing peer/storage
owners, and payload work in the transfer owner. Deterministic replica selection
stays in the existing [planning owners](architecture.md#planning).
Native engine P/D has an independent transport and destination ownership.

## Limits and validation

Current CPU controls cover migration, quota, journal overflow, membership, scope
and shutdown; see [cluster benchmarks](https://github.com/OrbitInfra/OrbitKV/blob/main/benches/cluster/README.md) and
[fault gates](fault-qualification.md#shared-cache-gates). Missing coverage blocks
remote admission without blocking local cache operation. The supported inventory
scope remains four owners. Historical DRAM/SSD isolation is **FAIL** and sixteen
owners remain **UNQUALIFIED**; process-only etcd controls are not independent
failure-domain or cache-HA evidence. See [qualification limits](performance.md#qualification-limits).
