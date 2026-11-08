# OrbitKV completion plan

**Current focus: [design v2](design-v2.md), a design-only proposal awaiting review.**
OrbiTE beta.1 source adaptation is merged in
[PR #330](https://github.com/OrbitInfra/OrbitKV/pull/330), `d54da80e`;
internal Rust crates are unpublishable following
[PR #331](https://github.com/OrbitInfra/OrbitKV/pull/331), `87991239`.
The 2026-11-08 maintainer direction is a state compiler, versioned agent state
trees and attention-guided paging. A trace-driven simulator follows in a separate
PR after design review; implementation and hardware experiments have not started.

**Presentation decision, 2026-11-07:** use [LMCache](https://github.com/LMCache/LMCache)
and [its docs](https://docs.lmcache.ai/) as content/navigation references, and
[kern](https://kern-baa.pages.dev/) ([website source](https://github.com/pegainfer-project/kern/tree/master/website))
as a layout-pattern reference. Keep OrbitKV's palette, banner and identity;
copy no text, images, logos or code, and imply no affiliation.

**Maintainer decision, 2026-11-07:** OrbiTE is the sole transfer engine before
hardware re-qualification. The TENT implementation and engine-family/default
comparison gate are removed. Historical acceptances retain their original
backends, revisions and limits. No current OrbiTE GPU, RDMA, GDS or installed
profile inherits them; CPU/TCP, io_uring and CUDA-shim checks remain CI evidence.

Ship a reliable external cache for official vLLM and SGLang, then demonstrate
its effect on real inference. Engines own scheduling/HBM; Rust owns cache
resources; OrbiTE moves bytes. Live P/D retains its engine-native transport.

This is the only execution queue and status record. [Compatibility](compatibility.md)
describes current boundaries, [models](models.md) gives recipes, and
[performance](performance.md) links reviewed measurements. Architecture guides
specify contracts rather than independent completion claims.

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

The first release profile is dense TP=1/PP=1 on vLLM 0.31.0 and SGLang 0.5.22,
with DRAM, io_uring SSD and the already accepted scoped peer profiles, all
re-qualified on the current path before publication.

1. **Current review:** [design v2](design-v2.md), including verified model schemas,
   upstream contracts, lifetime invariants and falsifiable experiment gates.
2. **Next PR, after review:** a bounded trace-driven simulator. No production
   policy or hardware claim; use the design's schema, workload and ablation plan.
3. **Reliable consumed path:** finish required ownership contracts and qualify
   bytes, cold/partial/full hits, cancellation, bounds and normal shutdown.
   Retaining unknown work is not eventual fault reclamation.
4. **Measured serving benefit:** compare native caching, OrbitKV and applicable
   LMCache/HiCache profiles with equal inputs/budgets and regression controls.
5. **Installed release:** follow S8's artifact and publication gates. Additional
   hybrid, P/D-fault and rank profiles need their own scope and evidence.

Independent healthy-path performance/package work may proceed while a distinct
upstream or physical-fault contract is blocked; it cannot waive that contract.

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

**Partial.** Native client, registration, query/source leases and workers own
lifetime. Extend those owners rather than add a Python coordinator. Current
contracts are in [transport](transport.md), [engine-local Restore](engine-local-restore.md)
and [adapters](adapters.md).

| Accepted delivery | Retained result |
| --- | --- |
| Transfer boundary / dependency, #309–#312 | Typed failures, bounded budgets, registrations and file-segment ownership; later superseded by the concrete OrbiTE owner |
| Peer GPU Restore, `eb9e9785` (#313) | Persistent tensor/codec registrations and one batch per bounded source plan; DRAM fills retain host destinations |
| Native SSD removal, `78e31f8a` (#315) | Engine file I/O; no native OrbitKV io_uring/cuFile fallback |
| Padded file vectors, `25ebc535` (#316), `c1cce1bb` (#319) | Layout ranges/padding reach engines; GPU slots and TENT scratch adapter deleted |
| Local copies, `73570259` (#317) | OrbiTE stream-local engine, bounded windows/slots and completions retained through drain |
| Intent/deadlines, `67d9ac4b` (#321) | Foreground/high Restore, background/low preparation, checkpoint/low write-back; Unspec by default |
| Peer retirement, `a683f991` (#322) | Membership fence before requester-specific close; wait-only retries and identity capacity returned only after drain |
| cfg confinement, `70ebeb48` (#323) | Shared logic compiles without backends; typed Unavailable, no native startup and strict Clippy |
| Unsafe hygiene, `7f68a1a6` (#324) | Invariant notes/lint, redundant unsafe impl removal, owner-before-failure and unregister-before-unmap |
| Panic removal, `c7fd6144` (#325) | Source lints, three justified expectations, typed boundaries and unloading exception to Drop-time abort |
| OrbiTE only, `1f30895a` (#326) | Pin `1312e9b1`, auto/tcp/rdma hint, concrete engine and lifecycle 13; TENT/submodule/build/wheels/custom router removed; 12 CI checks green |

These implementation acceptances preserve grants, epochs, bounded admission and
physical-drain receipts. They are not hardware qualification. Source grants remain
authoritative; every requester owns its `RemoteSegment`. Timeout/cancellation/
Unquiesced keeps owners and capacity; unknown effects are never replayed.
CPU/TCP priorities remain advisory without per-priority dispatch statistics.
Replacing pure transfer-leg timings needs an engine timing API; end-to-end
KV-path observations remain. No TENT API gap is a current OrbitKV dependency.

- [x] Explicit registration identity/failure propagation and retained retry ownership.
- [x] Prepared Save, native batching and bounded producer admission.
- [x] Normal-path registration and selected-release prepared-adapter controls at
  the historical revisions below.
- [ ] Carry allocator-issued page generations through source/destination use and reuse.
- [ ] Qualify physical completion after timeout, partial submission and cancellation.
- [ ] Reclaim after permanent loss only with valid termination proof; unresolved
  owners remain bounded and observable.
- [ ] Complete current-release installed adapter-fault and OrbiTE hardware coverage.

Selected releases lack allocator-issued page identity on connector/linker
handoffs. Registration/session epochs and adapter-local counters cannot observe
all native allocation/free paths. Track [vLLM #62810](https://github.com/vllm-project/vllm/issues/62810)
and [SGLang #45493](https://github.com/sgl-project/sglang/issues/45493).

| Historical upstream candidate | Accepted evidence | Limit |
| --- | --- | --- |
| [vLLM #62819](https://github.com/vllm-project/vllm/pull/62819), `b37aa38c` | Source/CPU allocator identity and ordinary/HMA/partial-tail/MultiConnector finish-time handoff | Unmerged, unreleased and unconsumed at review; no page-reuse or physical-fault acceptance |
| [SGLang #45548](https://github.com/sgl-project/sglang/pull/45548), `3cc5121f` | Source/CPU ownership/geometry and scoped H20 warmed asynchronous issuance/snapshot/readiness/stale-generation primitive | Unwarmed first call waits; no serving threshold or complete installed faults; unreleased/unconsumed at review |

An upstream merge alone does not change supported pins. Consume the released API
and pass installed reuse/fault gates before broadening support.

Historical evidence stays scoped:

- [OrbiTE TCP source pilot](transport.md#experimental-orbite-peer-tcp-profile),
  `5a50d092` / #304: same-host H20/CUDA 12.2; not installed serving or CUDA 13.
- [OrbiTE host RDMA and CUDA 13.1 source profiles](transport.md#experimental-orbite-peer-rdma-profile),
  `660f893d` / #305–#306: host bytes and A-node embedded Manager; not installed
  cross-host Manager/GPU, GDR, physical faults or broader topology.
- [Installed OrbiTE TCP/same-RNIC profile](transport.md#installed-client-and-manager-profile),
  `eb1cc90f` / #307: frozen CPython 3.11 wheel, unified Torch CUDA 13.0.96,
  original IPC-credit retirement; not current-path or serving qualification.
- [Prepared vLLM 0.31.0 pairs](https://github.com/OrbitInfra/OrbitKV/pull/290) and
  [SGLang 0.5.22 pairs](https://github.com/OrbitInfra/OrbitKV/pull/295): normal
  DRAM/io_uring retirement; the SGLang pool/linker fixture checks 2,621,440 bytes,
  sentinels and four original credits with sole/independent exporter holds.
  [#293](https://github.com/OrbitInfra/OrbitKV/pull/293) retains earlier 0.5.21 results.
- [Original lifetime ledger](https://github.com/OrbitInfra/OrbitKV/blob/1d173c1a27401f6e219c5120f97ba5eab3e2d52b/docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership)
  and [pre-C4 transfer ledger](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership)
  preserve exact hashes, failed freezes and earlier TENT source-close/buffer
  repairs, including #237/#288 and [Mooncake #4816](https://github.com/kvcache-ai/Mooncake/issues/4816).
  TENT hardware limits and verdicts remain historical, not new OrbiTE work.

Acceptance still requires real crash, delayed/lost grants/ACKs, partial submit,
repeated reuse and restart. TTL, lease expiry and logical errors are not drain.

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
  obsolete custom P/D code removed. Manager shared-cache bytes now use OrbiTE.
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

### Future stage: design v2

[State compiler, versioned state trees and attention-guided paging](design-v2.md)
proposes the next S6/S7 direction: engine-derived component/dependency IR,
cost-based materialization and recomputation, agent forks/JIT restore, and sparse
paging through upstream vLLM contracts. The safety foundation preserves existing
owners, grants, epochs and physical-drain receipts; OrbiTE remains the only byte
path. All experiments are **unexecuted** and current model guards remain in force.
Review the design first, then the hardware-free simulator, before production work.

The [hybrid-cache proposal](hybrid-cache-design.md), reviewed in
[PR #328](https://github.com/OrbitInfra/OrbitKV/pull/328), stays unchanged as
historical design context. Design v2 supersedes its experiment sequence; it does
not upgrade any qualification or claim existing minimal-range recovery as new.

## S8 — Product organization, final artifacts and release

**Partial.** Follow [LMCache's documentation organization](https://docs.lmcache.ai/)
for user navigation while retaining OrbitKV's Rust owners and website identity.

- [x] Consolidate the product documentation and retire completed campaign code
  and duplicate reports in [PR #289](https://github.com/OrbitInfra/OrbitKV/pull/289).
  Feature/model recipes, current support limits, links and historical references
  have independent source/CPU review; runtime support is unchanged.
- [x] Consolidate the release build into one CUDA 13 `orbitkv-llm` distribution
  and one PyPI Trusted Publishing job. The ten-wheel matrix covers CPython 3.10–3.14
  on Linux x86_64/aarch64; final candidate qualification and publication remain open.
- [x] Remove TENT native packaging and the fork submodule in #326; historical
  provenance remains at immutable source links.
- [x] C4 concise user/contract guides accepted in #327 (`37722fec`); evidence
  pages and frozen verdicts are preserved, with no runtime or claim change.
- [ ] Build final Manager/engine images and wheels, preserve licenses/provenance,
  and qualify clean installed artifacts on claimed hosts and profiles.
- [ ] Re-qualify the current data path before any publication. Qualifications
  recorded before engine-routed SSD, direct peer→GPU restore, removal of
  `storage/ssd/gpu.rs`, and OrbiTE local copies do **not** carry over. Preserve
  those records as historical evidence; they cannot qualify the new path.
- [ ] Before release, replace the OrbiTE git revision with the crates.io release
  (`orbite = "0.1"`) and document the resolved engine version in release provenance.
- [ ] Publish only after GPU re-qualification and maintainer authorization.
  **0.1.0 is OrbitKV's first release.** `orbitkv-llm` stays unpublished until
  then; wheel CI and candidate artifacts do not authorize publication.

Once hardware returns, use this order:

1. OrbiTE **0.1.0**.
2. Re-run OrbitKV's claimed profiles with **installed wheels with OrbiTE**:
   vLLM **0.31.0** and SGLang **0.5.22**, dense **TP=1/PP=1**, DRAM plus
   io_uring SSD, and the accepted scoped peer profiles. Record current artifacts,
   bytes, ownership/drain and fault evidence; do not expand a scoped claim.
3. Build **0.1.0rc1**.
4. Perform clean installed-artifact qualification on the claimed hosts.
5. **0.1.0**.

OrbiTE is the only transfer engine. The former comparison that selected TENT or
OrbiTE as the release default is removed by the 2026-11-07 maintainer decision.
Matched benchmarks and fault controls still inform qualification; publication
remains blocked on the installed hardware gates.

Maintain concise README, quickstart, configuration and benchmark guidance with
each future behavior change. Publication remains the maintainer's action;
implementation and CI results do not authorize it.

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

The [pre-C4 plan](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/completion-plan.md)
preserves the integration and cleanup chronology. The [pre-cleanup plan](https://github.com/OrbitInfra/OrbitKV/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/completion-plan.md)
preserves detailed accepted/failed runs, commits, artifact locations and frozen
measurement contracts. It is an evidence reference, not another execution queue.
Its S2 isolation failures, physical-lifetime limits and unqualified scopes remain
unchanged. Earlier S1 consolidation is retained at
[`ec3add9b`](https://github.com/OrbitInfra/OrbitKV/tree/ec3add9bcaf91a0171685ba42b5ae8330076c397).
