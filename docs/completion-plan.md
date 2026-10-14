# Completion and upstream integration plan

This is OrbitKV's only execution queue. Architecture pages describe contracts;
[compatibility](compatibility.md) describes qualified profiles. Implementation,
independent acceptance and release are different states. Deliver one coherent
stage or substage, then independently review its consumed path and evidence.

The chronological records have been consolidated after verifying their complete
archive. [The immutable pre-refresh plan](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/completion-plan.md) retains every frozen cohort,
failed run, reproduction instruction and earlier review limit. The summaries
below do not replace those evidence packages or change any qualification result.

## Release baseline and reference policy

| Project | Selected release / commit | Role |
| --- | --- | --- |
| vLLM | `0.31.0` / `db9527a46873454610df6dbedf79a36d6bf1a7f6` | Official V1, one-attention-group cache baseline; scoped dense acceptance |
| SGLang | `0.5.21` / `e00930c5489053f26d86b179cee0d087f846acbb` | Official UnifiedRadixCache external linker; public lifecycle migration open |
| LMCache | `0.5.5` / `05a013b29da78cf2321b9b46ec5039dde2fb0bb0` | Released integration and matched-comparison reference |

At each upgrade, recheck official releases, freeze the selected tag/commit and
update dependencies, lockfiles, submodules and qualification together. Upstream
main and submitted patches are contribution material, not release APIs. Keep
historical results at their measured versions; remove superseded compatibility
branches at cutover. See [interface audit](engine-release-audit.md).

## Delivery status and dependencies

| Stage | Current state | Remaining delivery |
| --- | --- | --- |
| S0 | Complete | Handoff/skill migration merged in PR #190 |
| S1 | Accepted at `ec3add9b` | Preserve external evidence and one queue |
| S2 | Accepted implementation; full qualification open | Metadata implementation, four-owner baseline and scoped three-host process-only delivery are accepted. DRAM/SSD isolation FAIL; sixteen owners UNQUALIFIED; support stays four owners. Cross-host cache HA and independent etcd failure domains remain unqualified |
| S3 | Implementation partial; qualification open | Conservative native failure retention has limited code/CPU acceptance at `27f1cb1b`; physical termination proof, installed GPU/adapter gates, page generations, explicit registration and crash reclamation remain open |
| S4 | Partial | Measured copy/SSD/control improvements exist; mixed contention, complete serving gains and S3-dependent faults remain open |
| S5 | Partial | Official dense cache/restart and scoped deployment profiles accepted; public lifecycle, complete hybrid recovery, native P/D faults and broader topology remain open |
| S6 | Partial | Observations and guarded limited choices exist; one consumed route/admission planner remains open |
| S7 | Open | Consumed semantic retention/checkpoint compiler beyond recovery validation |
| S8 | Partial | Documentation and wheel workflow exist; final images, artifact qualification and publication remain open |

S5.1 research and S8 documentation/package preparation may proceed independently
of S2/S3. New payload concurrency, reuse, reclamation and early consumption must
satisfy their lifetime dependencies. Missing hardware leaves a gate unqualified;
it does not stop unrelated work. No stage is closed by this documentation change.

## Baseline: preserve these implementations

| Owner | Existing behavior | Remaining boundary |
| --- | --- | --- |
| Process channel/native client | UDS bootstrap, iceoryx2 commands, shared arenas, bounded accepted-operation ownership | Page generations and live-process stall handling |
| Core transfer/storage | Engine-local raw DRAM Restore, Manager Publish/SSD/codecs, bounded multipart plans and final drain | Safe earlier multipart/SSD/codec consumption and physical lifetime |
| Catalog/server cluster | Scoped inventory streams, local complete views, leased etcd membership and source-incarnation fences | Isolation, scale qualification and real host-failure recovery |
| Peer owner/TENT | Authoritative grants, DRAM/SSD READ, release reconciliation and retained source exports | Permanent requester loss and transport-fault reclamation |
| Engine adapters | Official release callbacks/linker, engine-owned allocation and native P/D candidates | Public SGLang hooks, hybrid/rank/fault profiles |
| Cost/planning | Resource-scoped observations, shadow estimates and guarded peer choices | Complete executed route and admission ownership |
| State contracts | Declared recovery rules and compiled page demands | General retention/read-set IR and semantic reclamation |

Do not add a second Catalog, directory RPC, Python coordinator or transport facade
just to close a checkbox. Recovery and component improvements do not establish a
win over an engine's resident-HBM hit; see [performance](performance.md).

## S1 — Evidence separation and one truthful work queue

**Accepted.** Generated evidence is archived outside Git, benchmark output uses
explicit external directories, and the experiment-output guard protects tracked
source. Keep programs, reusable workloads and small fixtures in the repository.
Verify archives and update every reference before retiring tracked evidence.
No history rewrite or removal of a sole evidence copy is allowed.

**Acceptance:** archives readable; no tracked generated collections; benchmark
and publication/link/search gates pass; every open obligation belongs to S2–S8
or explicit deferred research. The earlier 70-obligation migration is retained
in the immutable plan.


## S2 — Metadata reliability and measured capacity

**Depends on:** S1. **Owners:** `orbitkv-server/src/cluster/`, `orbitkv-catalog/`,
core inventory and metadata diagnostics.

**Accepted engineering scope:** the metadata implementation, four-owner baseline
and scoped three-host process-only delivery are handed off. Their acceptance
does not mean every original
qualification criterion below passed. Historical isolation failures, sixteen-owner
expansion, cross-host cache HA and independent failure domains retain their
failed/unqualified state; their evidence, budgets and thresholds are unchanged.
Those gates require separately authorized fresh cohorts and independent review.
S3 consumes the accepted metadata contract and proceeds independently of those
performance/HA qualification limits.

- Extend real-etcd coverage to quota exhaustion/recovery, slow/disconnected
  Watches, concurrent rebuild and eviction, sustained journal overflow, member
  churn, index exhaustion and network partition/rejoin. Preserve incarnation and
  publication-cursor fences; never advertise a partial index as complete.
- Measure publication latency, Watch lag, snapshot rebuild time, update CPU,
  logical index bytes/RSS and etcd growth across increasing node/key/update loads.
  Define a tested operating envelope instead of claiming unlimited replication.
- Fix concrete bottlenecks and recovery gaps revealed by those runs. Keep local
  discovery free of directory/coordinator RPCs. Separate membership renewal from
  bulk synchronization and apply bounded work/backoff throughout recovery.
- Verify leader loss with a quorum and host loss across actual failure domains.
  Three etcd processes on one machine prove process faults only. Include startup
  readiness, operational diagnosis and recovery instructions.

**Original full qualification criteria (still open):**
  no lost final puts/deletes after convergence, no stale incarnation
  authorization, bounded memory/queues, local cache still usable during metadata
  loss, and a repeatable capacity/recovery report. A fenced runtime restarts with
  a new incarnation; it cannot silently resume using the expired one.

### S2.1 — Quota and index-budget recovery

**Independently accepted.** Real etcd quota/recovery and index-budget controls
retain bounded ownership and explicit completeness. A partial or retired view
cannot authorize a source. Preserve negative controls and exact final-key checks.

### S2.2 — Watch delay, partition and compaction recovery

**Independently accepted.** Delayed/disconnected membership Watches and compaction
recover without reviving an expired incarnation. Preserve reconnect/bootstrap,
lease-expiry and final put/delete convergence controls.

### S2.3 — Measured metadata scaling smoke

**Independently accepted within its smoke scope.** Retain measured publication,
index/queue budgets and etcd growth at the frozen scale. A smoke result is not
sustained capacity, a sixteen-owner qualification or independent-domain HA.

### S2.4 — Bounded live-DRAM journal and metadata-loss correctness

**Independently accepted.** A bounded journal supports live DRAM publication and
recovery. Gaps/overflow require a complete replacement view; they cannot become
silent partial success. Metadata loss must leave local cache correctness intact.

### S2.5 — Full-Manager DRAM and io_uring metadata-fault correctness

**Independently accepted at `f3908a93`.** Full Manager DRAM/io_uring processes pass
transient metadata loss, lease expiry, exact payload/identity checks and normal
cleanup. SSD reads do not masquerade as DRAM residency. This is same-host
multi-process evidence; host HA, RDMA and native GDS were not qualified.

### S2.6 — Sustained metadata churn and capacity

**Independently accepted within the frozen workload.** Sustained puts/deletes and
membership churn converge exactly with bounded queues/index ownership. Preserve
the tested operating envelope and failed capacity controls rather than claiming
unlimited replication.

### S2.7 — Bounded coalescing and frozen comparison baseline

**Independently accepted at `93abd676`.** Bounded coalescing trades stream bytes
against visibility delay. Keep matched no-coalescing controls, all samples and
independent-run uncertainty. A byte reduction alone is not a latency win.

### S2.8 — Owner inventory streams and coordinated protocol cutover

**Independently accepted at `ad5bb8e6`; production fix `344ef6c9`.** etcd retains
protocol identity and leased membership; bounded owner inventory streams feed
local indexes. An owner retired by budget revocation stays retired across
membership refresh until a complete replacement snapshot commits. The mutation
control rejects the former unsafe assignment. No directory RPC or second index
implementation is introduced.

### S2.9 — Scoped discovery and explicit coverage

**Independently accepted.** Scope selects inventory coverage without changing
in-scope candidates or payload correctness. Coverage is explicit: absent coverage
is not a proven miss. Preserve complete snapshot/reconnect and scope-change
fences; scoped bytes may shrink while source authority remains identical.

### S2.10 — Sustained live-store and independent-domain qualification

**Partial.** Same-host DRAM/io_uring all/scoped correctness is accepted at
`65c51aaa`, including the four-owner envelope and normal membership/stream drain.
The frozen formal 25-cell campaigns are valid and independently accepted **as
failed isolation qualifications**. Five sixteen-owner ordinary publication→install
p99 measurements pass 50 ms; install→harness is reported separately. DRAM and
SSD fail the predeclared per-pair and 95% CI ≤1.05 isolation contract.

Support stays **four owners**; sixteen owners remain **UNQUALIFIED**. Never delete
slow samples, relax thresholds or relabel a valid failure as a pass. Preserve
both formal cohorts and their actual terminal receipts.

**Remaining gates:** reviewed isolation diagnosis/remediation and a separately
frozen replacement qualification if justified; cross-host cache/HA; real etcd
leader/quorum/process faults; independent physical failure domains; final
serving/rank profiles. Three processes or hosts do not prove independent power
and network failure domains. Native GDS and S3-dependent failures remain separate.

### S2.10a — Same-host isolation root-cause diagnosis

**Implemented / qualification blocked.** The bounded correlated timeline and
lifecycle fixes reached `3161eaf9`. Three instrumentation pilots failed the
predeclared overhead guard; no formal ABBA/BAAB diagnosis or production policy
repair was authorized from those pilots. Preserve the off/on controls and stop
decisions; overhead failure is not a Manager-isolation root cause.

### S2.10b — Lifecycle runtime acceptance and lower-overhead diagnosis

**Scoped diagnosis accepted; production repair blocked.** Fourteen A100 cells
with 50 warm-up and 1,000 measured operations each have valid byte oracles and
terminal receipts. Tails correlate with the Python observer/return boundary;
order and phase effects prevent a repeatable metadata-pressure causal claim.
No Manager, metadata-lock or SSD-owner repair follows from this evidence.

PR #202's observer/helper protocol and interruption-safe cleanup are merged.
Later valid 25-cell qualification still fails DRAM/SSD isolation; it does not
expand owner support. Original invalid runtime/wheel-shadowing smokes remain
archived rather than replaced or hidden.

A fresh-query-owner readiness correction initializes all six phase-budget
metric series before a first query. Its real Prometheus regression preserves
live accounting across reservation transitions and multiple owners; independent
acceptance and installed-Manager readiness remain open. The old missing-metric
readiness/quarantine evidence is preserved. Harness drain gates must reject
missing applicable ownership/SSD metrics rather than defaulting them to zero.

The three-host harness integration requires complete finite ownership/SSD drain
series and explicitly validated frozen configuration, with real subprocess
PID/start-tick and non-killing cleanup controls. This implementation remains
unqualified for launch until exact runtime inputs, current resources and
independent launch review are accepted. Preserve legacy cohort failures.

The full consumer launch also isolates cached Python execution. Its installed
site/projection paths are explicit, Python environment/user imports are disabled,
and a nonexistent task cache prefix is shared with the Manager's embedded Python.
Source-hash checks alone can accept altered timestamp-valid bytecode; a real
consumer control now rejects that unisolated launch and reads the frozen source
under the isolated prefix. Manager admission records its actual process-local
cache environment before GPU work. Code controls do not replace the blocked
installed-Manager/GPU readiness and independent fault launch review.

A separately frozen process-only three-host etcd cohort may consume the current
node/network/lock owners while GPU/cache qualification remains blocked. Its
accepted result can cover etcd election/quorum/process recovery only; it does
not close Manager source authorization, Watch/cache rebootstrap or independent
physical failure domains. Formal launch remains subject to exact independent
review and fail-stop evidence retention.

The single stage021 process-only cohort at `82c07bd9` is independently
**ACCEPTED_RESULT**. Three hosts pass bootstrap, normal leader SIGTERM/rejoin,
quorum stop/recovery and complete final puts/deletes: 65 acknowledged mutations,
nine complete range oracles and identical 17-key final sets at revision 66.
Six etcd runs have actual SIGTERM/reap records; final owned processes/ports are
absent and three locks release within the cumulative cleanup budget. The leader
shutdown includes etcd's graceful leadership transfer, so abrupt crash detection
is unqualified. Raw evidence and independent replay remain under
`/root/orbitkv-artifacts/s2-three-host-runtime-gates-20261013/` in
`process-stage-021-formal-freeze-approved-bootstrap` and its independent result
review directory. This closes only the frozen process-only contract; Manager
Watch/source authorization/cache HA and independent physical domains remain open.

The process-only launcher binds each host's actual Python executable, standard
library, HTTP dependency file inventory, import origins, symlink targets and
mapped native files to the reviewed input manifest. Controller, launcher, node,
network and lock agents validate that closure before owning resources; remote
Python commands use the frozen interpreter with environment/user imports
disabled and a fresh, nonexistent task bytecode prefix. `-B` alone can read
unfrozen cached code; controls prove that the isolated prefix reads frozen source
and rejects a preexisting task cache. A changed digest or shadowed import fails
before child creation.
Automatic site and `.pth` startup are disabled with `-S`; safe-path mode and
explicit frozen dependency roots keep host startup code outside execution.
The verifier requires those interpreter flags before resource admission.
The frozen inventory also covers actual transitive imports, including optional
HTTP JSON backends. Bootstrap verifies the approved input manifest and integrity
helper using only standard-library paths, then installs the import guard before
adding staged or dependency paths. A stdlib-name shadow cannot execute first.
The independently approved digest is passed through every command owner and
checked before loading runtime/helper inputs; rehashing modified inputs cannot
authorize them.
Import admission rejects executable files absent from the
reviewed manifest before they execute; complete input inventories reject extra
staged files. Top-level package origins alone do not establish this closure.
This is admission code and CPU-control evidence; the new freeze and formal
process result still require independent review.

**Remaining qualification:** keep the original failed/unqualified gates visible.
A future qualification campaign must inspect the actual handoff, verify runtime,
proxy and resource ownership, freeze new inputs and obtain independent launch
review. Retain failures and cleanup evidence. Readiness and ordinary RDMA serving
do not close HA. Engineering acceptance does not authorize S6/S7 or release any old
quarantine. Historical task-specific fail-stop contracts remain unchanged.

## S3 — Transfer lifetime and generation-safe ownership

**Depends on:** S2's metadata contracts. **Owners:** core peer/query/storage owners,
transfer/sys C ABI, process channel and the consumed adapter registrations.

S3 may consume the independently accepted metadata correctness contracts while
the S2 isolation and sixteen-owner gates remain failed/unqualified. The first
native-lifetime substage has independent acceptance at `27f1cb1b` for its
conservative code/CPU retention boundary only:
native failed/canceled/timeout/invalid/unknown statuses cannot return caller
buffers or release the source completion owner. A wrapper deadline may return
its original error only after every task subsequently completes. Poll-time
transport failover is disabled because a successful replacement attempt does
not establish termination of the preceding attempt. Retained work keeps its
batch slot and shares a 32 MiB per-engine native-request descriptor budget;
existing pool/query/export budgets continue to own physical allocations.
Post-submit Rust unwinds are caught without dropping the outer request,
admission, caller-buffer or source-completion owners. The prior `9b5825d4`
candidate remains rejected for its unwind hole; its built wheel is archived.

The fixed TENT C ABI accepts deferred batch free and exposes no independently
qualified physical-failure fence. Physical RDMA/CUDA failure reclamation,
permanent requester-loss recovery and final installed-GPU/adapter gates remain
open. Latest upstream inspection and all native controls are archived under
`/root/orbitkv-artifacts/s3-termination-20261013/`. The corrected CUDA-13 wheel
build/isolated import passes, but the old H20 installed host-buffer payload
control fails: native COMPLETED is reported while the sentinel is unchanged
when CUDAPlatform::copy cannot use a device. This is not payload or hardware
acceptance. Preserve that failed gate; promotion needs a working CUDA allocation
and independently accepted installed payload and adapter/fault gates.

The old three-host readiness quarantine is a separate, scoped audit case. Its
query-source unregister succeeded; the client was deliberately closed before
the missing metric exception, and partial cleanup then retried that closed
client. No TENT payload READ or remote grant was submitted in that case. The
reconnect error is not residual-DMA evidence. This finding does not authorize
release of the old process/tensors/locks: a separately accepted owner-specific
procedure and concrete completion evidence are still required. Preserve its
immutable `BLOCKED_READINESS_CONSUMER_CLEANUP_QUARANTINE` result and archive
manifest `c9854b368263cede1cfa7b8732c01cf4426fcb39dee1863e3ac037f8cfe805d1`.

- Establish transport termination evidence before reclaiming exports belonging
  to a permanently lost requester. Inspect the current pinned TENT and latest
  upstream main/issues before changing the ABI; reuse or submit a focused
  transport fix if needed. Keep OrbitKV metadata/control state in OrbitKV.
- Bound and diagnose retained/quarantined source memory while proof is absent.
  Timeout, etcd expiry, cancellation, remote process disappearance and a control
  ACK are not interchangeable with native drain. An unavailable upstream primitive
  is an explicit blocker to safe reclamation, not permission to free memory.
- Carry owner-authored page/allocation generations through actual registration,
  source grants, destination use and adapter boundaries. Keep semantic last-use
  separate from completion of CUDA/RDMA/SSD work. Extend existing owners instead
  of adding an unused generic fence framework.
- Replace the remaining framework CUDA IPC pickle interpretation in the Manager
  with explicit validated registration data; remove the retired wire/API path.
- **Implementation open:** expose a stable native peer-liveness contract before
  enabling SGLang failed-session probing. Liveness is not transfer termination.
- Add a bounded operational response to a live Manager that never completes a
  Publish, preserving held pages until drain or established process death.

**Acceptance:** real tests for requester/source crash, delayed/lost grants and
  release ACKs, partial submission, repeated page reuse and restart. No use-after-
  free or premature recycle; resource retention is bounded and observable; safe
  reclamation eventually happens when its proof exists. Both adapter lifecycles
  consume the new generation checks. No TTL-based reclamation substitute.

## S4 — Finish communication execution and demonstrate gains

**Partial. Depends on S3 for new reuse/early consumption.** Existing transfer,
channel, SSD/codec and engine-hook owners retain responsibility. Contracts are
in [transport](transport.md) and [engine-local Restore](engine-local-restore.md).

KDA copy work, bounded SSD host batching, separated SSD read/write submission,
GPU/NUMA identity and descriptor/control measurements are implemented within
their reviewed scopes. CUDA sources live in separate files. Rejected descriptor
reuse, READV merging and TENT RPC-thread/lane candidates remain rejected;
correct bytes do not override a performance regression.

- Profile actual adapter hashing/conversions, destination checks, fragmentation,
  dispatch, scratch allocation and repeated shared-prefix H2D before selecting
  the next bottleneck. Keep GIL/observer timing separate from native execution.
- Compare device-event DMA/kernel timings on measured shapes, with registration,
  plan/submission and complete restoration cost reported separately. Preserve
  exact-byte, small-payload, contention and model-output controls. Do not switch
  defaults from a component-only result.
- Improve existing SSD preparation/read parallelism and completion scheduling
  only with a consumed workload and matched byte/drain/latency gates. Native
  GDS/cuFile requires physical eligibility and separate qualification.
- Profile first peer recovery, lazy TENT setup and warm steady state separately.
  A connection-warming candidate must consume an explicit reviewed interface,
  account for setup/fabric resources and pass contention controls. Never use
  unowned dummy payload addresses or omit warming cost from serving latency.
- Retain bounded multipart plans and whole-operation ownership. Earlier layer,
  multipart, codec or SSD readiness requires consumer dependencies and final
  physical drain, not only a readiness notification.
- Run matched native/OrbitKV/LMCache serving comparisons after model recovery
  qualifies. Larger-model native output controls do not close this gate.

**Acceptance:** correct GPU bytes and outputs, unchanged resource/lifetime
contracts, repeatable matched gains with uncertainty and frozen cold/small-path
regression guards. Require an independent review before performance promotion.
Detailed accepted/rejected cohorts remain in [kernel work](kernel-optimization.md),
[SSD methodology](ssd-performance.md), [peer qualification](shared-cache-qualification.md)
and the immutable plan; historical evidence is not requalified by this summary.


## S5 — Released-engine integration and upstream contributions

**Partial.** S5.1 may proceed independently; new lifetime behavior depends on
S2/S3 and measured S4 paths. Engine adapters and native client owners implement
the boundary. Use released [LMCache/FlexKV integration contracts](adapters.md#lmcache-and-flexkv-reference)
as references without duplicating their implementation layers.

### S5.1 — Release and interface audit

**Independently accepted at `38f8dbb2`.** Pins, package probes and source submodules
align with official vLLM 0.31.0 / SGLang 0.5.21. Released callbacks, installed-wheel
ordinary cache/restart and bounded model controls have scoped acceptance.
Upstream V2/recurrent restore ordering, SGLang pre-factory graph capture, missing
public pending tickets and native P/D timeout drain remain interface limits.

At every upgrade, record callback ownership, public/private dependencies, selected
release and exact gates. No fork-only callback, factory, default or CI runtime
may be required. A submitted or merged upstream fix is not a released contract.
See [interface audit](engine-release-audit.md) and [compatibility](compatibility.md).

### S5.2 — Minimal official cache backends

**Scoped ordinary-cache acceptance.** vLLM uses scheduler/worker KV Connector
owners; SGLang uses its external linker and shared native client. Python handles
callbacks, layout and GPU handoff; Rust owns state, batching and waits. Retired
custom P/D connectors, proxy, handshake wrappers and forwarding layers stay removed.
Single-rank admission avoids unnecessary consensus tensors while retaining actual
multi-rank decisions and pending/expiry/cancel ownership.

Official dense DRAM/io_uring recovery/restart, 900-second two-engine shared-Manager
pressure and explicit serial batch-1 FULL decode Graph recovery have separate
H20 acceptance. Broader graph shapes, models, ranks and fault profiles remain open.

**Remaining:** qualify each newly claimed profile from the final installed wheel;
keep cold/native-HBM/full/partial/restart/failure controls, exact outputs, actual
restore bytes, installed-file integrity and truthful process cleanup. Remove
obsolete code only after its replacement is consumed and passes the same contract.

### S5.3 — Public lifecycle and hybrid-state contracts

**Partial.** Idle process-scoped vLLM external reset is accepted at `6d874384`;
immutable local static PEFT identity at `47fea821`; the official dense query-control
handshake at `1ab67a58`. These do not qualify live weight changes, dynamic adapter
mutation or physical multi-rank recovery.

- Replace SGLang's private graph-initialization and pending-query admission hooks
  with consumed public lifecycle interfaces. Document their behavior and released
  alternatives; never remove cancellation/page-reuse protection merely for cleanup.
- Preserve model/tokenizer/computation/layout identity and generation fencing.
  Dynamic mutations require a consumed invalidation contract; stop/restart until
  such a profile qualifies. An idle reset is not that contract.
- Establish complete prefix/window/recurrent/conv/auxiliary state at legal recovery
  boundaries. The current vLLM V2/multi-group/recurrent guard stays active until
  released scheduling/restore order protects every destination page.
- Test cancellation, preemption, restart, partial commit and page reuse with real
  native state. Both semantic non-use and physical completion are required.

Qwen3.8-27B BF16/FP8 native output controls on both official engines are accepted
only for frozen text-only eager TP=1/PP=1 profiles with BF16 KV. They do not prove
external hybrid recovery, quality, cross-engine/precision equivalence or speed.
DeepSeek-V4.1-Flash remains a separate native readiness and complete-state target.
See [models](models.md); retain all original input/startup/cleanup limits.

### S5.4 — Official native P/D and independent cache

**Release-only cutover implemented; native fault qualification open.** vLLM uses
NIXL/MultiConnector; SGLang uses native disaggregation. Manager shared-cache bytes
continue through TENT. There is no maintained fork runtime or custom P/D proxy.

The bounded same-A100 dense model gates retain exact monolithic output, independent
cache reuse after engine restart and already-running request cancellation. They
do not prove cancellation during transfer, partial submission, delayed ACK or
page reuse. P reads/writes cache; D is save-only in the current candidate, avoiding
competing writers into native incoming pages. Test other selection orders separately.

**Acceptance:** independently reproduce installed official native P/D output,
preemption/retraction, cancellation, restart, peer loss, partial submit, delayed
ACK and page reuse for each opened profile. Preserve ownership and S3 drain proof.
Fork patches remain immutable upstream contribution material, not user runtimes.
Current commands and limits are in [P/D](pd.md).

### S5.5 — Deployment matrix and upstream maintenance

**Scoped acceptance, broader qualification open.** Current accepted profiles are
listed in [compatibility](compatibility.md); precise peer cohorts remain in
[shared-cache qualification](shared-cache-qualification.md).

**Completed bounded delivery:** the fresh Qwen3-32B BF16 TP=1/PP=1 local
comparison at harness `e28fe829` is independently accepted as a descriptive
collection: 14 cells, 420 measured requests, two opposite-order independent
runs, official vLLM 0.31.0/SGLang 0.5.21 and available LMCache 0.5.5 controls.
All outputs match their same-engine native controls; model, engine and installed
runtime hashes are unchanged. Native bytes remain the accepted `37b48c50` build.
vLLM OrbitKV pressure recovery is close to official CPU offload; SGLang 8K is
about 14% slower than HiCache with the same aligned host-hit coverage. Five
samples per length/phase/run do not qualify p99, throughput or a serving win.
The vLLM resident label includes a 16 MiB tail restore and remains mixed.
Actual signal statuses and native worker diagnostics are retained; normal parent
completion does not establish S3 fault reclamation. See [performance](performance.md)
and the external cohort/reviews linked from [PR #221](https://github.com/feichai0017/orbitkv/pull/221).

**RDMA component delivery, full campaign incomplete:** the independently
launch-reviewed two-H20 `404b6b84` harness used unchanged `37b48c50` runtime bytes.
All 72 GPU-pointer READ/WRITE cells completed across three restarted runs,
both directions, packed/gapped payloads and NIC allowlists. The 13 completed
Manager cells retain remote/H2D bytes, payload/sentinel oracles and source release.
The 108-cell campaign stopped at cell 86 when the target Manager's gRPC bind
failed with `AddrInUse`, before fixture or payload admission; no later cells
were launched or substituted. That frozen campaign's complete Manager comparison
was not obtained.
See [performance](performance.md) and [external evidence](benchmark-evidence.md).

The failed harness also masked the startup error with a dead-endpoint drain
error. Repair `c89ad887` preserves both errors, checks PID-owned HTTP/gRPC
listeners and normally reaps pre-payload failed services. It is CPU-tested;
the repair was not inserted into the frozen GPU campaign. Normal SIGTERM of
the remaining idle Manager/private etcd released all native resources. The
two CPU-only orphaned owners required forced termination, and platform PID 1
did not reap five zombie records. All sixteen GPUs, owned UDS/listeners and
resource FDs are empty; all 32 quarantine markers were retained at that terminal
handoff. This does not satisfy complete actual-wait cleanup or close S2/S3/S5.

**Accepted bounded delivery:** independent review accepts
physical-resource readmission for a new normal-path stage after matching the five
historical zombie identities, empty native ownership and exact archived markers
under the real resource locks. Their old actual wait/reap stays blocked. The
new `00c4a041` Manager-only harness, including repair `c89ad887`, consumes unchanged
`37b48c50` native bytes, fresh namespaces and distinct non-ephemeral service ports.
All 36 cells complete across three restarted runs, both directions, three sizes
and near/four-NIC allowlists: 1,080 measured and 1,224 complete restores, exact
remote/H2D bytes, full tensor/sentinel hashes and source release. New owned
processes have actual normal waits and empty physical postflight, independently
accepted with the descriptive outcome; 72 raw CUDA IPC warnings remain an S3/IPC
lifetime limit. This collection does not repair the old cohort, expand support,
qualify p99 or establish serving gains. See
[performance](performance.md) and [external evidence](benchmark-evidence.md).

**Accepted bounded delivery:** the `5b2089a5` SGLang original-order restore
diagnosis completes 180 requests and 20 public CPU/CUDA traces with unchanged
`37b48c50` native bytes. Independent review reproduces the 8K DMA gap and retains
the existing KernelBackend for further KDA validation: its TTFT is 10.2% lower
at 8K and 0.6% higher at 4K in this single collection. Fragmented DMA activity
and later forward admission support submission-path investigation; profiler
clocks and missing native CPU API events do not prove the unprofiled cause.
Exact outputs, restore-byte accounting, immutable artifacts and physical
postflight pass; full KV byte controls remain a candidate promotion gate.
Six native child `-9` exits, three IPC warnings and historical missing waits remain limits.
See [performance](performance.md) and [external evidence](benchmark-evidence.md).

The separate `6819edb0` cache-first shutdown smoke stops at its second cell:
SGLang context unregistration requires a live Manager and logs `Broken pipe`
after that Manager exits. The source-only ordering proposal is rejected;
the benchmark retains engine-before-cache shutdown and neutral host monotonic
stop boundaries. Native IPC retirement needs its own consumed protocol proof,
not reordered processes or relaxed error guards. The failed freeze and all
unlaunched cells are retained separately; no lifetime/support gate is promoted.

**Accepted bounded kernel delivery:** the `4b6c7eba` harness consumes unchanged
`37b48c50` native bytes in 13 valid H20 cells. Four byte-control cells check every
registered payload/sentinel byte in the actual released SGLang pool across twelve
restore generations, including partial destinations and deliberate corruption.
Nine freshly restarted model cells contain a native output control and four
balanced independent DMA/kernel pairs: 270 measured Qwen3-32B requests, 4K/8K,
BF16/raw, TP1/PP1, eager, with profiling and diagnostic tracing off.
Independent reconstruction accepts all byte/output controls, frozen inputs,
literal waits, unchanged installed files and empty physical postflight.

8K pressure TTFT improves by a pair-geometric 10.45% (95% interval 9.69%–11.08%);
all four pairs improve and the frozen 3% benefit guard passes. 4K pressure and
both lengths' cold/HBM controls pass the 5% interval guards. This qualifies the
existing explicit kernel direction for this serial profile. Nine native child
`-9` exits, eight IPC warnings, native-client wait gaps and historical missing
waits stay lifetime limits.
Computed model KV, p99/throughput/contention, Graph, broader models/ranks and
native IPC/fault reclamation remain separate; no production default is selected.
See [performance](performance.md#independent-sglang-kernel-pairs) and
[external evidence](benchmark-evidence.md#sglang-independent-kernel-pair-archive).

**Next bounded delivery:** select new CUDA/copy or scheduling code only after
its demonstrated consumed cost and correctness contract are accepted through
KDA. Measure packing, upload, submission and forward admission under the actual
fragment geometry and inference load before changing their owners. Preserve
all-enqueued admission, event generations, source ownership and physical drain;
keep CUDA in standalone files. Representative copy wins cannot select a production
default. Native GDS and physical failure-domain qualification remain separate.
S2/S4/S5 stay Partial.

The initial 14-cell exploratory cohort stopped after three valid local vLLM
cells: the released LMCache MP server completed its lifespan shutdown and
re-emitted SIGTERM, while the benchmark required zero. Its 30 recorded requests
remain an invalid cell; ten later cells were not launched. The complete failed
cohort is archived externally. A narrowly scoped benchmark correction preserves
the literal signal status and requires fresh cache/HTTP shutdown evidence.
The fresh complete cohort above remeasures every cell; no invalid samples are
reused or substituted.

- Ordinary installed dense H20 DRAM/io_uring recovery/restart and shared-Manager
  pressure have acceptance for their frozen budgets and execution settings.
- Registered trusted-loopback query-control authority/leases are accepted at
  `b1ea00a7`; common-prefix coordination and official TP=1 handshake at `1ab67a58`.
  This is not physical TP/PP/EP support. Protected cross-host activation remains
  implementation open.
- Physical two-H20 RDMA dense TP=1/PP=1 serving with DRAM/io_uring, consumer
  restart and controlled empty-source restart has scoped acceptance on both
  official engines. Cache HA, independent failure domains and requester-loss
  physical reclamation remain separate gates.
- GPU registration uses full UUID identity across reordered/partial device views;
  payload placement binds the resolved GPU-local NUMA node before first-touch and
  fails explicitly if disallowed. Test cleanup failures/interrupts remain failures
  and retain ownership receipts; they cannot certify every native child drain.

Track single/multiple instances, dense/hybrid, independent replicas, native P/D,
attention DP, TP, PP, MoE EP and heterogeneous combinations separately. Do not
infer KV partitioning from EP size. Qualify shared budgets, source incarnations,
sender/receiver cancellation, Manager restart, containers/UID/IPC/PID visibility,
physical NIC counters and actual GPU destinations for each claimed cell.

Maintain upstream registration/docs/tests and upgrade gates. Record issue/PR,
maintainer response, merge commit and first released version separately. Submit
lifecycle and TENT fixes independently of the minimal cache backend.

**Acceptance:** executable scenarios with output, physical-transfer and drain
receipts for every support claim. A minimal upstream cache contribution need not
wait for every research cell, and its merge does not qualify those cells.


## S6 — One consumed route and admission planner

**Depends on:** S2/S3, measured S4 paths and S5 P/D authority. **Owners:** existing
`planning/`, `cost/`, `query/`, residency/peer owners and actual engine feedback.

- Extend complete-route estimates to compatible local DRAM/SSD, peer DRAM/SSD,
  supported direct transfers and engine-owned recomputation. Compare only equal
  state coverage and completion targets; include queue/admission/authorization,
  transfer, decode and H2D once, with freshness and uncertainty.
- Make the selected plan acquire the actual source/extent lease, byte credits
  and destination ownership. For P/D vs direct recovery, the same consumed plan
  must own handoff authorization; observation-only messages are insufficient.
- Qualify bounded replanning and contending destinations, device/SSD/NIC budgets,
  per-instance shares, preparation deadlines and retention/write admission.
  Train only observed outcomes. An unknown estimate must not masquerade as zero.
- Compare DMA/kernel and io_uring/native cuFile only when both are physically
  eligible and qualified. Introduce guarded switching with measured margins;
  retain a clear deterministic policy for unknown estimates, without a duplicate
  legacy planner or generic forwarding layer.
- Keep Manager route decisions independent of request routing. After that works,
  integrate the pinned upstream router through a narrow consumed boundary if the
  routing milestone is retained; include worker evidence, replica risk and
  revalidation. NIXL/backend expansion is an evaluation decision, not an assumed
  prerequisite to finishing TENT recovery.

- Add receiver-driven admission using a bounded in-flight byte window **and**
  pacing. One node/physical-NIC/direction budget must include Manager READ and
  engine-owned native P/D traffic; Manager-only admission misses the latter. Lease
  bounded credits in batches; each engine retains its native P/D transport and
  Manager peer-cache movement remains direct through TENT.
- Keep network credits, destination buffer lifetime and source export authority
  distinct. Return credits on verified receive/transport progress; retain memory
  until its final user drains. Credit timeout is never a memory-release proof.
- Schedule ready data using deadlines, remaining critical state/rank/layer work,
  per-instance fairness and aging. Pair receiver admission with source egress
  limits; a receiver cannot control other flows or all fabric bottlenecks. Reserve
  headroom for engine communication using measured physical interference.
- **Qualification open:** cross-medium selection and peer SSD under contention,
  cancellation and RDMA. Two-host TCP byte recovery does not qualify policy gain.
- **Deferred research:** router event normalization and pinned upstream router
  service integration; eviction externalities/replica risk; load-only versus
  overlap-only versus joint routing on one trace; active RDMA WRITE cache
  replication; topology slicing, endpoint pooling and alternate-rail retry;
  NIXL Mooncake-backend evaluation. These need a concrete scope decision and
  measured benefit. Current P/D WRITE is not cache replication. Do not introduce
  a new transport abstraction or deprecated KVBM to close the current plan.

**Acceptance:** selected routes explain their coverage, ownership and predicted
  complete cost; stale evidence/resource rejection replans safely; submitted
  payload failure does not blindly retry another medium. Fixed-policy, shadow
  and executed-policy comparisons demonstrate repeatable benefit under contention
  without violating correctness, memory budgets or cold-path regression limits.

## S7 — Semantic retention and checkpoint compiler

**Depends on:** S3 generations/lifetimes, S5 consumed hybrid contracts and S6
physical planning. **Owners:** `orbitkv-state`, core planning/storage and adapter
state declarations; retain framework scheduling ownership.

- Define a finite, explicit `may_read(query, state)` IR for supported state
  families. Compile full attention, sliding windows/sinks and recurrent state
  into required ranges, legal checkpoints, retention and reclamation decisions.
- Implement minimum persistent state for the defined model and supported
  families, with an exact oracle on bounded cases and explicit approximation
  limits elsewhere. Do not claim a general optimum for undeclared semantics.
- Feed checkpoint placement, prefetch/placement and semantic release decisions
  into real storage/planning owners. Safe reclamation requires both semantic
  non-use and physical completion; compile-only output is not delivery.
- Extend DSA, draft-model or auxiliary state only with a matching semantic
  contract and representative model gate. Unknown state stays unsupported.

**Acceptance:** real dense, SWA/sink and recurrent/hybrid model recovery, plus
  boundary/preemption/restart adversarial cases. Check required-state soundness,
  generation safety, bounded-case minimality, retained bytes, retention
  amplification and semantic reclaim latency. Retain a workload with measured
  state reduction; do not describe recovery validation alone as this compiler.

## S8 — Product organization, final artifacts and release

**Depends on:** accepted scope from S1–S7; retain explicit exclusions for unqualified
hardware/research work. **Owners:** docs/README/website, examples, containers and
existing build/release workflows.

- Organize README/docs like LMCache: product purpose, supported deployment modes,
  installation, short quickstart, engine/model recipes, architecture, operation
  and benchmark methodology. Use its separation of library, tests, examples,
  benchmark programs and tools as guidance, preserving OrbitKV's Rust owners.
- Preserve OrbitKV's visual identity. Keep one canonical technical document per
  subject; remove chronological development diaries and duplicate result tables
  from user paths. Show full local/distributed/P/D ownership and data/control
  paths in diagrams, with status and limits next to capability claims.
- Publish curated experiments from external versioned artifacts, linking model,
  engine/source revisions, transport, budgets, controls and reproducible scripts.
  Verify links/search/navigation after moving legacy result content.
- Build independent Manager and engine images from validated artifacts. Separate
  Manager/client packaging only at a clear dependency/version contract; qualify
  shared-process resources, GPU access and SSD mounts before publishing manifests.
- Preserve licenses and bundled upstream provenance. Keep heavy GPU/RDMA gates
  explicitly marked and distinguish engine support from runtime build success.
- Build final wheels/images, install outside the checkout on both hosts and
  repeat the claimed sharing/P/D/restart gates with those exact hashes. Source
  test evidence cannot replace final artifact qualification.
- Use the existing release workflow for publication within the user's release
  authorization. Publishing is a distinct delivery action; manual candidate
  builds and successful CI do not mean a PyPI release has happened.

**Acceptance:** clean install and documented launch work for each supported
  profile; no undocumented source-tree/native-library dependency; CI and website
  checks pass; no generated experiment results are tracked; package/image hashes,
  external evidence, actual publication status and remaining limits are reviewable.

## Review contract for every delivery

The implementing agent provides:

1. Stage/substage, base and final commits, and the changed user-visible behavior.
2. The concrete state/resource owner and invariants affected; removed APIs and
   duplicate paths; any remaining compatibility or forwarding layer justified.
3. Exact checks and exit status. Run only the gates relevant to the change, plus
   required repo checks. Never build native libraries during live runtime tests.
4. External evidence locations and artifact hashes, with failed controls and
   environmental limitations. No private credentials or results in the commit.
5. Updated completion-plan/support status and the next dependency. Passing unit tests does
   not close a hardware, model-serving or performance cell.

The reviewer inspects implementation and call sites, repeats the smallest tests
that establish the changed contract, audits lifecycle failures and benchmark
fairness, and checks public claims against exact artifacts. Acceptance is
**accepted**, **changes required**, or **implemented / qualification blocked**,
with concrete reasons. The last status does not mark a gate complete.

## References and scope

These sources guide organization and workflows, not OrbitKV capability claims:

- [LMCache project layout and README](https://github.com/LMCache/LMCache).
- [LMCache documentation and recipes](https://docs.lmcache.ai/).
- [LMCache benchmarking guide](https://docs.lmcache.ai/getting_started/benchmarking.html).
- [Codex repository skill discovery](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills).

Architecture contracts remain in [distributed cache](distributed-cache.md),
[transport](transport.md), [state planning](state-planning.md),
[deployment](deployment.md), [P/D](pd.md) and [release](releases.md) documentation.
Existing historical reports describe their recorded revisions only.

## Consolidated documentation and retired work

The former root TODO, implementation handoff, communication sequence and roadmap
have been consolidated here. Their historical revision remains available at
[the S1 source snapshot](https://github.com/feichai0017/orbitkv/tree/ec3add9bcaf91a0171685ba42b5ae8330076c397).
All 70 open obligations were assigned to S2–S8: storage/codec execution and
performance to S4; transfer generations/revocation to S3; engine identity,
Radix/P-D/topology/container contracts to S5; cost/network admission and explicit
router/backend research to S6; retention semantics to S7; package/image/provenance
and recurring release qualification to S8. S2 retains metadata capacity/faults.

Retired proposals remain retired: a second shard Catalog, binary OrbitKV metadata
inside Mooncake, unused Python state facades, a generic region RPC without a
consumer, and duplicated backend/cost wrappers. Existing raw 2D DMA, bounded raw
parts, single-part layer events and local global indexes need extension or
qualification, not another implementation. The former direct engine-page SSD
and multi-writer assembly item is **implementation partial**, not just untested.

Historical result tables stay at immutable revisions or external evidence storage.
Current resource contracts remain in their owner documents. No compatibility
stub documents or duplicate execution queues replace the deleted files.
