# OrbitKV completion plan

Ship a reliable external KV cache for official vLLM and SGLang: reuse prefixes
from DRAM, SSD and peer Managers, then demonstrate the effect on real inference.
Engines own scheduling and GPU pages; Rust owns cache resources; Mooncake TENT
moves shared-cache bytes. Native P/D keeps its engine-owned transport.

This is the only execution queue. [Compatibility](compatibility.md) describes
what users can run; [models](models.md) provides recipes;
[performance](performance.md) reports reviewed measurements. Architecture pages
define contracts, not additional delivery queues.

## Delivery status and dependencies

| Stage | Delivered | Remaining |
| --- | --- | --- |
| S0/S1 | Repository organization and ownership baseline accepted | Preserve established owners and historical evidence |
| S2 | Accepted engineering scope: inventory-fed indexes, scoped discovery, four-owner baseline and three-host etcd process-only controls | Isolation FAIL; sixteen owners UNQUALIFIED; cross-host cache HA and independent failure domains unqualified |
| S3 | Explicit registration, native Save ownership and bounded normal-path controls, including vLLM 0.31.0 and SGLang 0.5.22 prepared-adapter pairs | Current-release adapter faults, page generations, physical termination and eventual fault reclamation |
| S4 | KDA copy work, SSD batching, bounded kernel improvement and RDMA paths | Inference contention, repeatable complete serving benefit and safe promotion |
| S5 | Official dense cache/restart, scoped Graph/shared-Manager/two-host profiles; custom P/D removed | Public lifecycle, complete hybrid recovery, native P/D faults and broader topology |
| S6 | Cost observations and guarded limited route choices | One consumed route/admission planner |
| S7 | Declared recovery requirements validated | Semantic retention and checkpoint compiler |
| S8 | Documentation, wheel builds and release workflow | Final images, installed-artifact qualification and publication |

S3/S4/S5/S6/S8 remain **Partial**; S7 is **Open**. S2 is complete only for its
accepted engineering scope. Existing failed qualifications keep their verdicts.
Missing hardware is unqualified, never passed.

## Current delivery sequence

The immediate product is dense TP=1/PP=1 caching on vLLM 0.31.0 and SGLang
0.5.22, using DRAM, io_uring SSD and already accepted scoped peer profiles.
Finish one useful delivery at a time; do not turn each failed experiment into
another permanent controller or parallel roadmap.

1. **Maintainable product surface — delivered in [PR #289](https://github.com/OrbitInfra/OrbitKV/pull/289) and [PR #295](https://github.com/OrbitInfra/OrbitKV/pull/295).**
   Retire completed campaign-specific code and
   duplicate reports; keep reusable benchmarks and tests with a current consumer.
   Organize README, model recipes and website around installation, features,
   deployment, operations and performance. Preserve support limits and verify
   historical references before removing their source collections.
2. **Reliable consumed cache path.** Finish S3 ownership prerequisites used by
   the selected engine profile. Validate complete bytes, cold/partial/full hits,
   cancellation, bounded resources and normal shutdown. Do not advertise fault
   reclamation when unknown work can only be retained until process exit.
3. **Measured serving improvement.** Use existing serving/transfer benchmarks
   with equal model, inputs and budgets. Compare native caching, OrbitKV and the
   applicable LMCache/HiCache profile. Fix the measured production bottleneck;
   qualify benefit and regressions under contention before selecting a default.
4. **Installable delivery.** Run claimed profiles with final wheels/images
   outside the source checkout. Publish concise recipes and support boundaries.
   Then extend hybrid models, native P/D faults and additional ranks individually.

This ordering preserves every S4/S5 acceptance obligation below. Independent
healthy-path performance and package work may proceed while a distinct physical
fault contract or unreleased upstream callback remains blocked.

## S2 — Metadata reliability and measured capacity

Etcd stores protocol identity, persistent node epochs and leased membership.
Owner inventory streams maintain covered local indexes. Ordinary lookups do not
perform per-key etcd calls or directory RPCs. Keep incarnation fencing, atomic
snapshot installation, contiguous deltas, bounded queues and explicit coverage.
See [distributed cache](distributed-cache.md) and [P2P deployment](p2p.md).

- [x] S2.1–S2.9: metadata faults, capacity, coalescing, owner streams and scoped
  discovery independently accepted in their recorded profiles.
- [x] S2.10: same-host DRAM/io_uring correctness and sustained collection accepted.
- [x] Three-host etcd process-only normal-SIGTERM cohort accepted.
- [ ] DRAM/SSD foreground isolation: existing qualification is **FAIL**.
- [ ] Sixteen-owner support: ordinary visibility passes; overall **UNQUALIFIED**.
  Support remains four owners.
- [ ] Cross-host cache HA and independent physical etcd failure domains.

The frozen five-percent isolation and fifty-millisecond visibility requirements
remain unchanged. A process fault is not an independent host/power/network fault.
Historical S2-only campaign programs are retired from the maintained tree;
their [source and contracts](https://github.com/OrbitInfra/OrbitKV/tree/dc52461078bb22910ba6c783ecf0d9213599b10b/benches/cluster)
remain available for reproduction. Existing integration tests and the reusable
metadata benchmark continue to exercise the current implementation.

## S3 — Transfer lifetime and generation-safe ownership

Owners: native client, registration, query/peer leases, storage and transfer
workers. Extend these owners; do not introduce a Python coordinator or facade.
See [transport](transport.md), [engine-local restore](engine-local-restore.md)
and [adapter contracts](adapters.md).

- [x] Explicit registration identity and failure propagation; retry ownership
  retained after failed unregister.
- [x] Prepared Save handles, Rust batching, bounded producer admission and
  normal-path installed byte/engine controls within recorded profiles.
- [ ] Carry allocation/page generations through source grants, destination
  ownership and actual engine page reuse. Session/registration epochs alone
  do not establish this contract.
- [ ] Prove physical completion after timeout, partial submission and cancellation
  before recycling buffers, mappings, events, pages or registration owners.
- [ ] Reclaim resources after permanent requester/source loss when a valid
  termination proof exists; retain unresolved owners within explicit budgets.
- [ ] Complete current-release installed adapter fault coverage for both engines.

The selected engine releases do not expose an allocator-issued identity for the
physical page allocation carried by their connector/linker handoffs. The generic
upstream contracts are tracked in
[vLLM #62810](https://github.com/vllm-project/vllm/issues/62810) and
[SGLang #45493](https://github.com/sgl-project/sglang/issues/45493). Candidate
code, an open issue or an upstream merge is not released OrbitKV support; pins
and consumers change only after the matching release and installed page-reuse
fault gates pass. Connector-local counters keyed by block/page ID are rejected
because they cannot observe allocations made by other engine paths.

The vLLM source/CPU API candidate in draft
[vLLM #62819](https://github.com/vllm-project/vllm/pull/62819), exact
`b37aa38c`, independently passes allocator identity, finish-time ordinary/HMA/
partial-tail/MultiConnector handoff and compatibility review. It remains
unmerged, unreleased and unconsumed. Its acceptance does not authorize an engine
pin change, page reuse or physical fault reclamation.

The SGLang source/CPU API candidate in draft
[SGLang #45548](https://github.com/sgl-project/sglang/pull/45548), exact
`3cc5121f`, independently passes allocator ownership, Full/SWA descriptor
geometry, default Rust tree-core adapter enrichment, legacy compatibility,
overflow/resize and unsupported-profile controls. Its H20 allocator primitive
also passes warmed steady-state asynchronous issuance/snapshot,
readiness-event handoff and stale generation rejection. An unwarmed first call
still waits for preceding CUDA work, and the measured primitive has no serving
performance threshold. It remains unmerged, unreleased and unconsumed; complete
installed SGLang page-reuse faults and physical termination stay open.

TENT likewise has no per-transfer receipt proving that failed, canceled or
timed-out CUDA/RDMA work can no longer touch caller memory; the required native
contract is tracked in
[Mooncake #4816](https://github.com/kvcache-ai/Mooncake/issues/4816). The accepted
unregister-with-active-READ reproducer is OrbitKV
[PR #237](https://github.com/OrbitInfra/OrbitKV/pull/237), merge `53e69620`.
Separately, the accepted whole-engine CPU/TCP source-close scope does stop new
pointer READ admission, wait for admitted CPU handlers and dispatch owners, then
unregister and release the pool. Unsupported paths, timeouts and unregister
errors retain the complete engine/registration/pool owner under the existing
1,024-pool/1-TiB process budget. That is real CPU/TCP reclamation after its valid
receipt, not a per-ticket, CUDA or RDMA termination proof.

The Python TENT writable-buffer ownership repair in [PR #288](https://github.com/OrbitInfra/OrbitKV/pull/288)
has source/CPU and installed CPU-native fixture acceptance. It retains real
buffer exports through failed cleanup; borrowed GPU backing and canonical
CUDA/RDMA physical lifetime remain unqualified.

CPU-native TCP source-close controls are accepted only in that scope. Canonical
CUDA-native source close can return unsupported status 2 and retain the engine,
registration and pool until process exit. Token Drop/best-effort destruction,
borrowed GPU buffers and physical RDMA faults remain separate open boundaries.

Acceptance requires real requester/source crash, delayed/lost grants and release
ACKs, partial submission, repeated page reuse and restart. No premature recycle;
retention must be bounded and observable, and reclamation must eventually follow
a valid proof. TTL, lease expiry and logical errors are not physical drain.

Official vLLM 0.31.0 prepared-adapter DRAM/io_uring healthy pairs are
accepted in [PR #290](https://github.com/OrbitInfra/OrbitKV/pull/290).
The selected SGLang 0.5.22 pair for each medium is independently accepted in
[PR #295](https://github.com/OrbitInfra/OrbitKV/pull/295): the released GPU-pool/linker
fixture restores all 2,621,440 bytes including sentinels, repays four original
IPC credits and retires registration with normal process waits. Both sole
ownership and an independent exporter hold pass; the SSD pair clears DRAM and
requires reads from the ext4 NVMe file through io_uring. This is a normal
prepared-adapter fixture, separate from model serving and fault reclamation.
[PR #293](https://github.com/OrbitInfra/OrbitKV/pull/293) retains the earlier
0.5.21 results. The
[immutable lifetime record](https://github.com/OrbitInfra/OrbitKV/blob/1d173c1a27401f6e219c5120f97ba5eab3e2d52b/docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership)
preserves exact source/wheel/review/archive hashes and all failed freezes.
Cancellation, page reuse and physical reclamation remain separate obligations.

## S4 — Finish communication execution and demonstrate gains

**Partial.** Owners: existing transfer, channel, SSD, codec and engine-hook code.
New reuse or early consumption depends on S3; independent healthy-path work does
not require finishing every fault/topology combination first.

- [x] Separate production CUDA source, KDA fragmented-copy work, bounded SSD
  host batching, separated SSD read/write submission and GPU/NUMA identity.
- [x] Explicit SGLang kernel profile accepted: Qwen3-32B BF16/raw, TP1/PP1,
  eager serial 4K/8K. The reviewed 8K TTFT reduction is versus our DMA route;
  this does not select a general default.
- [x] Descriptive two-host Manager RDMA and GPU-pointer component measurements;
  they do not establish complete serving superiority.
- [x] Bounded native observation and offline matched-pair validation implemented.
- [x] Diagnose SGLang 0.5.22 concurrent admission and compare its official
  equal-priority option across four independent process pairs. Shared-prefix
  recovery improves, but mixed-prefix regression guards fail; keep the option
  explicit and defaults unchanged. See [performance](performance.md#sglang-concurrent-admission).
- [ ] **Qualification failed:** the implemented native observer must pass its
  matched overhead pilot before diagnosing first peer recovery and warm steady
  state. Its current tail guard fails.
- [ ] **Qualification failed:** reduce the implemented engine-local H2D
  observer overhead until the full matched guard passes. Keep prior failed
  cohorts and clocks distinct.
- [ ] **Qualification open:** demonstrate repeatable complete serving benefit with concurrent inference,
  cold/native-HBM regression controls, throughput, tails and resource budgets.
- [ ] Qualify S3-dependent reuse, fault reclamation and earlier multipart,
  SSD or codec consumption before enabling those behaviors.

Use [retained benchmarks](../benches/README.md), the
[kernel workflow](kernel-optimization.md) and [SSD methodology](ssd-performance.md).
Profile the consumed path before choosing another optimization. Use KDA for a
CUDA bottleneck; fix native submission/admission/I/O in their existing Rust
owners when those are the cause. Add no speculative policy or unused kernel.

Acceptance: exact restored bytes and output controls; equal inputs, budgets and
coverage; independent matched repetitions and uncertainty; original benefit,
regression and observation-overhead guards. Source built, fixture fast and model
fast are different results. Keep the frozen detailed guards in
[the accepted measurement contracts](https://github.com/OrbitInfra/OrbitKV/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).

## S5 — Released-engine integration and upstream contributions

**Partial.** Engines own allocation, scheduling, native prefix trees, request
readiness and P/D. Python translates callbacks/layout/CUDA readiness; Rust owns
shared-cache state machines, waiting, planning, batching and resource holds.
Use official releases; do not restore fork factories, custom P/D connectors,
handshake/proxy implementations or compatibility facades.

### S5.1 — Release and interface audit

- [x] Select official vLLM 0.31.0 and consume SGLang 0.5.22 public load completion;
  runtime pins and source tags agree. Earlier SGLang acceptances use 0.5.21.
- [x] Repeat the 0.5.22 Qwen3-8B dense TP1/PP1 profile with final installed
  DRAM/io_uring byte and model/restart controls. Known-unsubmitted cancellation
  and independent consumer leases pass; physical cancellation/page reuse remain
  in S3/S5.2. Wider 0.5.21 profiles are not inherited.

On an upgrade, check official tags and consume exact released interfaces before
changing all pins together. Upstream main is not the user's runtime.

### S5.2 — Minimal official cache backends

- [x] vLLM KV Connector and SGLang UnifiedRadixCache external linker; ordinary
  dense TP1/PP1 DRAM/io_uring recovery, engine restart and scoped sharing accepted.
- [x] Scoped shared-Manager pressure, fixed batch-1 FULL decode Graph recovery,
  static immutable LoRA and two-H20 RDMA profiles recorded in
  [compatibility](compatibility.md).
- [x] Selected-release normal prepared-adapter retirement in DRAM/io_uring;
  the SGLang 0.5.22 pool/linker fixture covers sole and independent exporter holds.
- [ ] Finish consumed cancellation/preemption/page-reuse lifetime contracts with
  S3 before extending support. Manager restart does not preserve the SSD file.

vLLM currently requires V1 and one attention cache group. SGLang still uses
bounded private pre-capture and pending-query admission hooks. Preserve their
correctness protection until a consumed released replacement exists.

### S5.3 — Public lifecycle and hybrid-state contracts

- [x] Public SGLang preparation/pending-lookup contributions submitted and CPU
  controls accepted; submission is not released adoption.
- [x] Consume SGLang 0.5.22 private load destinations and public post-forward
  completion; delete custom load polling/tree insertion and recurrent component
  override. Official installed CPU and selected Qwen3-8B GPU controls pass;
  complete hybrid and fault qualification remain open.
- [ ] **Waiting on released APIs:** consume pre-capture and pending-admission replacements and remove
  each superseded private hook in the same delivery.
- [ ] Complete recovery for all required attention, window, recurrent,
  checkpoint and auxiliary state of each claimed hybrid model. Qwen3.8 native
  output controls and historical fork results are not external-cache support.
- [ ] Qualify native HBM hits, legal partial boundaries, cancellation,
  preemption, restart and generation-safe page reuse with actual model output.

Unknown state remains unsupported. Keep [model recipes](models.md),
[hybrid contracts](hybrid-recovery.md) and [interface audit](engine-release-audit.md)
aligned with code guards.

### S5.4 — Official native P/D and independent cache

- [x] Native vLLM NIXL/MultiConnector and SGLang disaggregation selected;
  obsolete custom P/D code removed. Manager shared-cache bytes stay on TENT.
- [ ] Qualify installed native P/D plus independent cache for cancellation,
  peer loss, partial submission, delayed ACK, restart and page reuse.

The candidate allows cache read/write on P and save-only on D, preserving one
writer for incoming D pages and native reliable-delivery requirements. Both
cache query leases and P/D ownership must drain through their own owners.
See [P/D composition](pd.md); ordinary linker abort support does not prove
physical P/D cancellation or timeout-drain safety.

### S5.5 — Deployment matrix and upstream maintenance

- [x] Bounded official native/OrbitKV/LMCache comparison and scoped two-host
  dense RDMA serving accepted, with limitations in [performance](performance.md).
- [ ] Qualify each additional model, Graph shape, DP/TP/PP/EP rank or container
  deployment before claiming support; distinguish byte layout from model output.
- [ ] Complete matched serving gains and S3-dependent faults for retained
  deployment claims; no extrapolation from one-GPU process controls.

For a confirmed Mooncake/TENT, SGLang or vLLM defect, isolate the root cause and
search open and closed issues/PRs. Reuse matching work. When absent, create a
reproducible issue and tested fix PR using the authorized identity; keep private
artifacts and credentials out of public contributions. Track upstream merged,
released and locally consumed states separately.

## S6 — One consumed route and admission planner

**Partial; after retained S3/S4/S5 prerequisites.** Extend existing planning,
cost, query and residency owners. Compare complete compatible DRAM/SSD/peer/direct
recovery and engine recomputation costs, including freshness and uncertainty.
Selected plans must acquire actual source leases, destination ownership and byte
credits; bounded replanning must not blindly retry submitted unknown transfers.

- [ ] Consume one route/admission plan with device/SSD/NIC budgets, per-instance
  fairness, deadlines, retention/write admission and deterministic unknown costs.
- [ ] Add receiver byte windows and pacing plus source egress limits; include
  engine-native P/D traffic in shared physical NIC budgets without replacing its
  transport. Credit return is separate from buffer release.
- [ ] Compare fixed, shadow and executed decisions under contention; qualify
  cross-medium/peer-SSD choices and cancellation/RDMA before promotion.

Router integration, replica-risk/eviction research, WRITE replication, alternate
rails, topology slicing and NIXL expansion stay explicit deferred research.
See [planning contracts](state-planning.md). Do not add a second planner.

## S7 — Semantic retention and checkpoint compiler

**Open; depends on S3/S5/S6.** Compile an explicit supported-state read contract
into legal ranges, checkpoints, retention and reclamation, then consume it in
real storage and planning. Validate soundness, generation safety, bounded-case
minimality, retained bytes, amplification and reclamation latency on dense,
window/sink and recurrent/hybrid models. Demonstrate measured state reduction.
Recovery validation alone is not this compiler; unknown auxiliary state remains
unsupported.

## S8 — Product organization, final artifacts and release

**Partial.** Follow [LMCache's documentation organization](https://docs.lmcache.ai/)
for user navigation while retaining OrbitKV's Rust owners and website identity.

- [x] Consolidate the product documentation and retire completed campaign code
  and duplicate reports in [PR #289](https://github.com/OrbitInfra/OrbitKV/pull/289).
  Feature/model recipes, current support limits, links and historical references
  have independent source/CPU review; runtime support is unchanged.
- [ ] Build final Manager/engine images and wheels, preserve licenses/provenance,
  and qualify clean installed artifacts on claimed hosts and profiles.
- [ ] Publish only within release authorization. The 0.1.0 package remains
  unpublished; a wheel CI build does not establish a published release.

Maintain concise README, quickstart, configuration and benchmark guidance with
each future behavior change. Publication and final artifact qualification remain
open independently of this documentation cleanup.

## Review contract for every delivery

Deliver a coherent user capability or necessary boundary repair. Report the
commit, changed behavior, relevant tests, external evidence and remaining limit.
An independent reviewer inspects the diff/callers, repeats the smallest meaningful
checks and verifies claims before dependent work or merge. Exact-head CI and the
required author/committer identity remain merge requirements.

Keep tests with a current developer, CI, release or operational consumer. Merge
duplicate contract cases; retired one-off campaigns do not belong in routine
gates. Raw results, traces, manifests and per-run reports stay outside Git.

## Historical evidence

The [pre-cleanup plan](https://github.com/OrbitInfra/OrbitKV/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/completion-plan.md)
preserves detailed accepted/failed runs, commits, artifact locations and frozen
measurement contracts. It is an evidence reference, not another execution queue.
Its S2 isolation failures, physical-lifetime limits and unqualified scopes remain
unchanged. Earlier S1 consolidation is retained at
[`ec3add9b`](https://github.com/OrbitInfra/OrbitKV/tree/ec3add9bcaf91a0171685ba42b5ae8330076c397).
