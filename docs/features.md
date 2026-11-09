# Features

OrbitKV extends the KV cache of official vLLM and SGLang beyond GPU memory.
Use it for repeated documents, shared system prompts, long conversation prefixes
and compatible cache reuse between replicas. Begin with the
[quickstart](single-node.md) and [model recipes](models.md). OrbiTE moves all
cache bytes. Current GPU/RDMA/GDS and installed profiles need hardware
re-qualification; earlier acceptances remain historical. See [compatibility](compatibility.md).

## Cache storage and reuse

| Feature | Behavior | Current boundary |
| --- | --- | --- |
| Host DRAM | Save reusable KV outside HBM and restore after eviction | Dense single-rank profiles on both selected engines |
| Local SSD | Extend host cache through OrbiTE file segments | SSD file is recreated on Manager startup |
| Engine restart reuse | Reconnect the engine to the existing cache | Keep the Manager and matching model/computation identity alive |
| Shared Manager | Multiple local engines share DRAM/SSD capacity | Scoped pressure acceptance; fairness/tail isolation remain open |
| Peer cache | Discover compatible replicas locally and fetch through OrbiTE TCP/RDMA | CPU/TCP CI evidence; GPU/RDMA and installed recovery require re-qualification |

Do not expect a benefit when requests have little compatible prefix reuse or
when native HBM already serves the data. Compare equal memory budgets and
working sets; see [performance](performance.md).

## What distinguishes OrbitKV

- **Rust resource ownership.** Native cache/query/transfer owners hold backing
  memory, reservations and registrations through their completion boundaries.
  Python handles engine callbacks, layout and GPU readiness.
- **Independent per-host Manager.** Reuse DRAM/SSD across engine lifetimes and
  share local capacity without moving inference scheduling into the cache.
- **Inventory-fed discovery.** Etcd maintains leased members and protocol
  identity. Bounded peer streams update covered local indexes; ordinary queries
  avoid per-key etcd calls and obtain authoritative source grants.
- **Engine-owned movement.** OrbiTE owns local DMA/kernel copies, file vectors
  and peer TCP/RDMA. OrbitKV supplies layouts, intent and deadlines while
  retaining grants and physical-drain ownership.
- **Explicit state contracts.** Model/computation/layout identity and declared
  recovery requirements prevent treating incompatible or incomplete state as a
  cache hit. Unknown layouts remain rejected.

These are architectural properties, not a claim of universal superiority over
another cache. Physical fault termination and eventual reclamation remain open
even where conservative ownership prevents premature release.

## Engines and models

The selected releases are vLLM 0.33.0 and SGLang 0.5.23. vLLM uses the official
KV Connector; SGLang uses UnifiedRadixCache's external linker with bounded
internal lifecycle hooks still present. Recorded Qwen3-8B/32B dense recovery and SGLang 0.5.22 public load-completion
results retain their original revisions and transfer paths.
Hybrid/auxiliary-state models need complete recovery qualification.
Check [model recipes](models.md) and [compatibility](compatibility.md).

Native P/D uses vLLM NIXL/MultiConnector or SGLang native disaggregation. OrbitKV
supplies an independent cache adapter; Manager-to-Manager sharing uses OrbiTE.
P/D cancellation/peer-loss/page-reuse composition remains unqualified.

## Optional and experimental features

Storage codecs, retention/cost policies, explicit CUDA kernels, native cuFile/GDS
and broader Graph/rank/container profiles have separate support limits. See
[storage formats](storage-formats.md), [cache policies](cache-policies.md),
[kernel copies](transport.md#local-gpudram-copies) and [GDS](gds.md).
Native GDS is implemented but unqualified. Object-store backends, CacheBlend-style
non-prefix reuse and automatic cross-engine byte interoperability are not current
OrbitKV support claims.

Use [deployment recipes](deployment.md) to select a supported topology,
[Manager configuration](server.md) for budgets and [metrics](metrics.md) to
verify actual external hits. Remaining implementation belongs to the
[completion plan](completion-plan.md).
