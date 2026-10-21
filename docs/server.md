# Cache Manager Configuration

## OrbitKV Cache Manager

```bash
orbitkv-cache-manager
```

### Options

- `--addr`: Peer control bind address in distributed mode and local socket port seed (default: `127.0.0.1:50055`)
- `--channel-service`: optional iceoryx2 name prefix. Every Manager startup appends a unique incarnation and advertises it through the stable UDS bootstrap socket. Old clients cannot block a restart by retaining the previous service.
- `--devices`: CUDA device IDs to initialize, comma-separated (default: auto-detect all available GPUs, e.g., `--devices 0,1,2,3`)
- `--pool-size`: Pinned memory pool size (default: `30gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--hint-value-size`: Hint for typical value size to tune cache and allocator (optional, supports: `kb`, `mb`, `gb`, `tb`)
- `--use-hugepages`: Use huge pages for pinned memory (default: `false`, requires pre-configured `/proc/sys/vm/nr_hugepages`)
- `--enable-lfu-admission`: Enable TinyLFU cache admission policy (default: plain LRU)
- `--cache-protected-percent`: Maximum percentage of pinned pool bytes in the demand-protected replacement segment, `0`–`100` (default: `0`, disabled). See [retention and admission](cache-policies.md).
- `--disable-numa-affinity`: Disable NUMA-aware memory allocation (default: enabled)
- `--blockwise-alloc`: Allocate each layer/page segment independently in DRAM-only mode (default: `false`). SSD-backed Managers always use this allocation policy so reads and writes share the same reclaimable units; surviving prefix pages do not pin other pages from a batch.
- `--log-level`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)

### HTTP & Metrics

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`, always enabled)
- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
- `--metrics-otel-endpoint`: OTLP metrics export endpoint (optional, leave unset to disable)
- `--metrics-period-secs`: Metrics export period in seconds (default: `10`, only used with OTLP)

### HTTP Endpoints

- `GET /health`: Health check.
- `GET /metrics`: Prometheus metrics, when `--enable-prometheus` is enabled.
- `GET /instances`: List registered instance IDs.
- `POST /instances/cleanup[?id=<instance_id>]`: Remove one instance, or all instances when `id` is omitted.
- `POST /cache/memory/cleanup`: Evict resident in-memory cache blocks while preserving backing-store data. `evicted_bytes` is the cache footprint removed from residency; `reclaimed_bytes` is the pinned-pool memory actually released immediately.
- `GET /cache/metadata`: Report membership validity/revision, explicit index
  coverage, active/staging/accounted index bytes, installed owner-view counts,
  local journal state and bounded inventory-stream session/byte diagnostics.
- `GET /cache/metadata/owners?after=<uuid>&limit=<1..128>`: Paginate installed
  owner view IDs, applied source sequences, freshness, receipt age and record
  counts. This diagnostic does not query a source or authorize payload access.
- `POST /cache/sync`: Wait for already submitted saves and return a source
  `inventory_fence` in distributed mode. Standalone mode returns
  `{"local_flush_complete":true}` after the same local save barrier without
  inventing a distributed fence. Neither result makes SSD payloads restart-durable.
- `POST /cache/metadata/await`: Controlled-test/operation barrier. Supply an
  `inventory_fence`, this requester's exact `scope_digest`, and `timeout_ms` no
  greater than 30,000. Success means the matching owner view is installed through
  the fence while membership remains valid. Ordinary queries never call it.

### SSD Cache

For normal deployment, set the cache path and the intended capacity; leave
backend selection to the Manager. `--ssd-backend` is an override for diagnosis
and controlled tests. cuFile is an optional library inside the Manager, not a
service to deploy. See [deployment examples](deployment.md#configure-capacity-then-connect-engines).

- `--ssd-cache-path`: Enable SSD cache by providing cache file path (optional)
- `--ssd-cache-capacity`: SSD cache capacity (default: `512gb`, supports: `kb`, `mb`, `gb`, `tb`). Selecting cuFile reserves this physical space before serving; allocation errors fail startup. io_uring uses logical sizing without upfront reservation.
- `--ssd-write-queue-depth`: SSD write queue depth, max pending write batches (default: `8`)
- `--ssd-write-policy`: `all` writes newly saved pages; `reuse` admits foreground-returned pages or repeated publications within a bounded history (default: `all`). Selective admission can require recomputation on the first reuse after DRAM eviction.
- `--ssd-backend`: optional override. `auto` (default) tries native cuFile on ext4/XFS and falls back
  to io_uring if capability initialization fails; a cuFile operation failure switches new
  operations to io_uring until Manager restart. `uring` uses pinned DRAM;
  `cufile` writes complete GPU state groups and restores leased SSD sources
  through two asynchronous 4 MiB GPU slots per instance/device and follows NVIDIA's compatibility
  configuration without automatic backend fallback. Native GDS must be [qualified separately](gds.md).
  Fragmented/multi-writer saves and speculative preparation retain io_uring.
  GPU-direct writes hold Publish pages until storage completion.
  At most one GPU write batch is in flight; up to eight write jobs are queued or
  active per instance/device. Saturation uses host publication/io_uring. These
  fixed GPU limits are separate from the io_uring queue/inflight flags below.
  The process channel admits up to 64 client ports; each cache client uses a
  control port and, after its first save, a separate Publish port.
- `--storage-codec`: `none` (default), GPU `ans`, lossy `fp8`, `turboquant-4` or
  `turboquant-3`. Applies to DRAM, SSD and peer payloads; see [storage formats](storage-formats.md).
- `--storage-codec-budget`: GPU workspace per transfer worker, including retained
  idle arenas (default 64 MB), separate from encoded pinned residency. ANS needs
  nvCOMP 5.3; FP8 selects
  AVX-512F, AVX2 or scalar CPU fallback according to runtime support.

- `--ssd-prefetch-queue-depth`: SSD prefetch queue depth, max pending prefetch batches (default: `2`)
- `--ssd-write-inflight`: SSD write inflight, max concurrent block writes (default: `2`)
- `--ssd-prefetch-inflight`: SSD prefetch inflight, max concurrent block reads (default: `16`)
- Full io_uring read queues wait for capacity while retaining the query's byte reservation.
  Concurrent queries for an identical prefix and storage identity can share a read.

### Query ownership budgets

- `--query-budget`: Maximum query-owned bytes across preparation, ready leases,
  and submitted GPU loads. Defaults to 75% of `--pool-size`.
- `--query-instance-budget`: Limit for one registered instance, shared across its
  client sessions. Defaults to the global query budget.

Both accept the same memory units as `--pool-size`, with
`0 < instance budget <= global budget <= pool size`. Reservations use the
registered storage group's padded block bytes, including its physical layout
and shards. Partial results shrink their reservation. Each owner is charged
independently even when pages or reads are shared; the pinned allocator tracks
physical use separately. The resident cache can still use the full pool.

Temporary budget pressure leaves a query pending. A single query larger than
the configured limit returns an empty restore result and increments
`orbitkv_query_budget_bypasses_total`; it does not wait indefinitely or record
an authoritative backing-store miss. Cancellation drains submitted I/O before
returning its bytes. A delivered lease remains charged until release, session
disconnect, expiry, or completion of every GPU consumer.

`orbitkv_query_reserved_bytes` tracks total ownership under the budget lock;
`orbitkv_query_reserved_bytes_by_phase{phase="preparing|ready|restoring"}` diagnoses
the individual stages. Summed phase samples are not an atomic budget snapshot.
Ownership remains charged through these phases. Preparation includes queued
reads and reconstruction;
it is a conservative payload reservation, not a measurement of a particular
device's buffers. `orbitkv_query_budget_waits_total` counts admission attempts
delayed by bytes, and `orbitkv_query_coalesced_reads_total` counts joined reads.
The old block-count prefetch limit has been removed.

Opt-in consumer preparation adds `preloading` and `prepared` phases. Both stay
within the speculative quarter-budget until a foreground claim; expiry never
releases an unfinished I/O's buffers. `--query-read-batch`,
`--query-read-timeout-ms`, and `--query-read-max-batches` control ordinary read
submission and recompute fallback. Defaults preserve ordinary demand behavior.
See [request preparation](request-preparation.md) for limits and control runs.

### Cross-Node (Multi-Node Setup)

- `--nics`: Optional Mooncake RDMA rail allow-list (e.g., `--nics mlx5_0,mlx5_1` or `--nics mlx5_0 mlx5_1`). Omit it to let Mooncake select the available transport, including TCP fallback.
- `--etcd-endpoints`: comma-separated HTTP etcd endpoints; enables distributed cache. Requires `--node-id`. Etcd stores protocol identity, epochs and leased members, not block locations.
- `--peer-advertise-addr`: optional concrete peer address stored in membership when a proxy/fault gate fronts the actual `--addr` listener.
- `--index-budget`: logical accounting for this Manager's complete global index; defaults to 256 MiB. This is not a process RSS cap.
- `--cluster-name`: etcd namespace, default `orbitkv`. `--membership-ttl-secs` defaults to 30 and accepts 12–3600; remote admission uses half the acknowledged TTL. See [deployment](p2p.md#leased-manager-membership).
- `--transfer-lock-timeout-secs`: Mark source transfers overdue after this many seconds (default: `120`). Timeout never releases memory still exposed to a remote READ.
- `--transfer-budget`: Source allocation reservations, defaulting to half the pinned pool. Entire allocations are charged once per session, including overdue sessions. At most 1024 sessions can be retained. New authorizations fail when either limit is exhausted; permanent requester loss still requires safe transport revocation or coordinated teardown.
- `--inventory-journal-bytes`: Retained residency-change bytes (default: `16777216`, 16 MiB). Lag beyond this history triggers a paginated inventory resnapshot.
- `--inventory-stream-coalesce-ms`: Quiet window for grouping a bounded
  contiguous inventory interval before sending a delta (default: `0`, range
  `0..=5`). `0` preserves immediate delivery. An explicit source fence causes
  subscribed requesters to prioritize the captured target without treating frame
  receipt as installation. The frozen S2.7 experiment qualifies `2` as an opt-in
  tradeoff; see
  [the historical coalescing measurements](https://github.com/OrbitInfra/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/distributed-cache.md#bounded-etcd-publication-coalescing).
- `--metadata-namespace`: Exact generated storage namespace to receive from every
  remote Manager; repeat it for an allowlist. The Manager logs the namespace
  generated by each actual registration. Values must use the complete
  `orbitkv:v2:<64 lowercase hex>` identity; model display names and prefixes are
  invalid. At most 256 namespaces and a 64 KiB encoded stream Open are accepted.
- `--metadata-empty-scope`: Explicitly receive no remote namespaces. This is
  distinct from omitting both scope flags, which retains `AllNamespaces`. Scope
  changes take effect only after a Manager restart and complete bootstrap.

## Distributed metadata

The same Cache Manager binary owns local storage and its local global index.
All distributed Managers use the same etcd cluster and namespace, with distinct
Node IDs. The peer gRPC endpoint serves inventory sessions as well as source
grants and completion.

```bash
orbitkv-cache-manager --addr 10.0.0.1:50055 --pool-size 30gb \
  --etcd-endpoints http://10.0.0.101:2379 --cluster-name inference \
  --node-id cache-a --index-budget 256mb
```

Use the other host's address and Node ID there. `GET /cache/metadata` exposes
coverage, installed views, logical bytes, registration validity and stream
progress; standalone mode returns JSON `null`. See [deployment and failure
behavior](p2p.md) and [metadata recovery](distributed-cache.md).

For an exact scoped deployment, first observe the complete storage namespace
logged by a real engine registration, then restart every requester that should
use that scope with the repeated `--metadata-namespace` values. Scope selection
reduces matching inventory bytes but does not authenticate tenants or remove the
per-member inventory sessions.

The stream format is a coordinated cutover. Stop every Manager first, then
dry-run and apply `scripts/migrate-metadata-format.py` with an explicit cluster
UUID and an archive path outside the checkout. The script refuses live member
records, compares the exact old format revision/value, preserves epoch keys and
deletes retired block/publisher keys in the same transaction. Rollback likewise
requires all new Managers stopped and restarts the frozen old binary with fresh
incarnations; it does not restore live leases or old cursors.
