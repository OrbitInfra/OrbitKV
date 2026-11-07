# Distributed cache deployment comparison

Reviewed against official documentation and pinned sources on 2026-09-24.
LMCache's latest release was v0.5.5 (`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`);
FlexKV references use `738ddc141a198b4e20de6c5d1f0128e387f7fdb2`. These are documented
integration paths, not OrbitKV benchmark results or a claim that every model,
engine release and parallel layout works. OrbitKV's tested engine versions
remain the pinned releases in [single-node setup](single-node.md).

Control-plane ownership was rechecked on 2026-09-25 against these same commits.
Cache-layer cluster visibility already exists in LMCache and FlexKV. Distinguish
three decisions: locating compatible state, selecting and executing a cache
transfer, and assigning an inference request to a worker. A cache controller
can own the first two independently of a request router. Global visibility
alone is not an OrbitKV innovation or proof of adaptive latency optimization.

## LMCache

The pinned MP sources distinguish service placement from distributed integration:

- [Independent service](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/deployment.rst):
  one cache service per node, shared by inference processes, with Docker and
  DaemonSet/Deployment examples. OrbitKV adopts this service topology; its
  shared-instance, image and container-resource gates are still open.

- [P2P sharing](https://docs.lmcache.ai/mp/p2p.html): each node has an LMCache
  server. A coordinator supplies membership; a peer locks and locates the
  requested KV before transfer into the requesting server's L1 memory.
  NIXL is the default transport; Mooncake TE is another documented option.
  Independent inference replicas can therefore reuse matching prefixes without
  a KV-aware request router; the engine talks to its connected cache server.
- [Multi-server TP](https://docs.lmcache.ai/mp/configuration.html#vllm-client-configuration):
  one vLLM deployment assigns contiguous rank groups to multiple servers.
  This mode explicitly rejects PP > 1 and DP > 1. That restriction concerns
  one deployment's multi-server connector topology; it does not prohibit
  P2P sharing between independent inference replicas.
- [P/D plus cache reuse](https://docs.lmcache.ai/mp/disaggregated_prefill.html):
  vLLM `MultiConnector` composes upstream `NixlConnector` for the current
  request's handoff with `LMCacheMPConnector` for offload and later reuse.
  The guide uses separate LMCache servers for P and D and optionally adds
  P2P sharing. It also names a required vLLM MultiConnector fix; a command
  example alone is not a release-compatibility guarantee.

The old in-process PD examples describe a different integration. Do not mix
their constraints with the MP deployment when comparing capabilities.

Membership is the coordinator's role in the basic P2P discovery path, not its
entire feature set. The pinned [MP coordinator](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/coordinator.rst)
also supports an eventually consistent key-placement directory populated by
optional cache-event reporting, fleet quotas/eviction, and commands for warm
prefetch, pin/unpin and deletion. Custom controllers can extend its policy.
These cache-management APIs do not require routing inference requests through it.

The [legacy Controller Manager](https://docs.lmcache.ai/kv_cache_management/index.html)
uses worker admit/evict reports to track chunks centrally; its P2P backend
queries that directory before NIXL transfer. The current documentation marks
this in-process mode deprecated. Keep its chunk-directory design distinct
from MP's peer-discovery path and optional coordinator views.

The pinned [default prefetch policy](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_policy.py)
chooses the first matching adapter by index. Prefetch retention, store targeting
and eviction have separate owners. These are useful boundaries, not evidence
that an adaptive cost optimizer is already available. The
[upstream design mapping](architecture.md#upstream-designs-and-orbitkv-owners)
records those mechanisms, optional lazy offload, isolated registration, and
release-note/configuration discrepancies that require explicit runtime checks.

## FlexKV

- Its [architecture](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/README.md#design-architecture)
  already has a cache control plane below the engine: `GlobalCacheEngine`
  performs prefix matching, manages space/eviction and selects movement and
  source/destination blocks. `StorageEngine` owns storage and `TransferEngine`
  executes movement. Its name does not imply one central request router.
- [Engine integration](https://github.com/taco-project/FlexKV) covers vLLM and
  SGLang, CPU/SSD offload, TP/PP work and Mooncake-based remote reuse. These
  features do not establish every model/topology combination.
- [Distributed reuse](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/docs/dist_reuse/README_en.md)
  uses a local snapshot of the global index, Redis metadata and Mooncake
  transfer. Avoiding a centralized lookup on each request is therefore not
  unique to OrbitKV. Its example still pins an older vLLM; verify compatibility
  before using it as a current-release benchmark.
- The pinned [transfer scheduler](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/transfer/scheduler.py)
  advances dependency graphs when operations complete. The README also records
  adaptive GPU/CPU copy paths. Neither mechanism alone establishes a calibrated
  cluster-wide source/tier/codec optimizer; evaluate each actual policy instead
  of treating all movement scheduling as the same feature.
- [SGLang integration](https://github.com/taco-project/FlexKV/blob/main/flexkv/integration/sglang/README.md)
  distinguishes standard released-engine support from model-specific adaptations
  tied to an upstream PR. Record the exact engine release and FlexKV commit.

For OrbitKV, local snapshots motivate bounded cached discovery, while their
refresh traffic, memory use and stale-source rejection remain comparison costs.
Prefer [Manager-owned planning](state-planning.md#cache-manager-decisions-below-the-engine)
with qualified path estimates and source admission. Keep request routing as an
optional additional consumer of the same evidence. Benefits over upstream
policies must be demonstrated under changing resource contention, including
the cost of maintaining that evidence.

OrbitKV prioritizes independent-replica sharing and P/D reuse, followed by catalog
HA. Compare bytes read, index/synchronization costs, tail latency and useful
throughput at equal capacity. Native code, separate processes and hybrid-model
support alone do not establish a performance advantage.

## Mooncake

- [vLLM shared storage](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html):
  `MooncakeStoreConnector` uses the Store's distributed CPU/SSD pool for
  offload and prefix reuse across instances. Its P/D example composes
  `MooncakeConnector` with `MooncakeStoreConnector` through `MultiConnector`.
- [SGLang HiCache](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html):
  engine HBM and host caches are private tiers; Mooncake Store supplies the
  shared L3. Multi-rank coordination takes the minimum successful prefix
  across ranks before claiming readiness.
- [SGLang P/D](https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/pd-disaggregation.html):
  the framework's handoff uses Mooncake TE for cross-node movement. The
  transfer path is distinct from HiCache's shared storage backend.

Mooncake TENT moves registered memory; it does not define a framework's
legal recovery boundary. TP/PP, heterogeneous rank layouts and hybrid models
must be qualified against the selected engine connector. Shared storage does
not by itself prove that a TP=4 representation can be restored by TP=8.

## OrbitKV execution order

1. Qualify sustained single-node DRAM/SSD serving for both engines, with
   bounded requests and resources, cancellation/restart correctness, and
   native-engine controls. Profile actual copy destinations before attempting
   to merge H2D restores. See [benchmarks](../benches/README.md).
2. Qualify two real hosts running independent matching TP=1 replicas, separately
   for vLLM and SGLang. Count real remote restore bytes, compare outputs and
   bound failure-to-recompute behavior. Then qualify same-host TP per replica.
3. Qualify P/D and cache reuse together: a P-side cache hit must still reach D;
   completed D-side blocks should be available to a later request on another
   P. Use the released native vLLM/SGLang P/D transports independently of
   OrbitKV's OrbiTE shared cache.
4. Replace sharded Catalog discovery with complete local global indexes,
   historically backed by etcd block metadata and revisioned snapshot/Watch;
   the S2.8 candidate replaces block traffic with owner inventory streams. Qualify quorum
   loss, repair and metadata capacity. Cross-host TP/PP, peer-SSD mixed-load
   qualification and KV-aware routing have separate gates.

Transport completion and page-lifetime safety are requirements at every stage.
Directory hints never authorize memory reuse. Keep engine-to-Manager UDS/iceoryx2,
peer control gRPC, and OrbiTE payload transfer. Full recovery semantics
remain tracked in [state identity](state-identity.md).
