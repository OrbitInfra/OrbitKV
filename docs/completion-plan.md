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
| S2 | Complete for the accepted engineering scope; full qualification open | Reviewed metadata contracts, the same-host four-owner baseline and one normal-SIGTERM three-host etcd process-only cohort are accepted. DRAM/SSD isolation FAIL; sixteen owners UNQUALIFIED; support stays four owners. Cross-host cache HA and independent etcd failure domains remain unqualified |
| S3 | Implementation partial; qualification open | Explicit CUDA registration and bounded installed normal-path ownership controls are accepted; page generations, physical termination, fault reclamation and broader adapter lifetime remain open |
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

**Complete for the accepted engineering scope (user closure, 2026-10-15):** reviewed metadata contracts, the same-host
four-owner baseline and one normal-SIGTERM three-host etcd process-only cohort
are handed off. Their
acceptance does not mean every qualification criterion below passed.
Historical isolation failures, sixteen-owner
expansion, cross-host cache HA and independent failure domains retain their
failed/unqualified state; their evidence, budgets and thresholds are unchanged.
Those gates require separately authorized fresh cohorts and independent review.
S3 may proceed only through accepted metadata contracts and its own lifetime
gates. Unaccepted contracts or failed prerequisite gates cannot authorize
dependent work.

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

**Full qualification acceptance (required before claiming full qualification):**
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

**Remaining S2 delivery:** keep the original failed/unqualified gates visible.
Continue evidence-based isolation diagnosis; apply harness or production repairs
only when the evidence justifies them, then obtain fresh matched qualification.
The embedded Rust Manager/GPU runtime and installed full-Manager/cache-fault
gates remain blocked or unqualified, despite code/CPU acceptance of the full
consumer. Cross-host cache HA and independent physical failure-domain
qualification remain in this queue.
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

The copy-status candidate consumes merged upstream TENT fixes #3476 and #3973
on the existing pin in a private build-source copy. Independent code/CPU/installed
host-byte acceptance at `6e099269` does not cover healthy GPU qualification.
Follow-up review found that CUDA pointer-query failures could select host memcpy
and error exits could retain the wrong device binding. A separate remediation
uses the driver's multi-attribute query, rejects unexpected errors and unusable
CUDA admission, and restores an existing caller device across copy failures.
Post-attempt CUDA errors retain the native stream/frame and caller RPC scratch
under a separate 64-call/1-GiB logical-copy admission budget; no error is used
as a DMA fence. Its deterministic CUDA API controls are software evidence; actual GPU and
physical termination gates remain open. Preserve the original candidate/wheel,
controls and review under `/root/orbitkv-artifacts/s3-installed-byte-errors-20261013/`.
The 2026-10-14 merge review also reproduced a combined device-selection and
context-restoration failure that returned through an unchecked guard. One checked
guard now restores on every return and retains admission if restoration fails.
Software controls reject the previous allocator and verify that 64 retained
context-error calls reject further admission without any memcpy submission;
these controls add no GPU or physical termination qualification.
On 2026-10-13 Forge job 20551990 is killed and both saved SSH endpoints refuse
connections. Its historical healthy CUDA gates remain distinct from the old
unusable H20. No new formal cells or hardware qualification have started, and
old resource locks/quarantines are not released.

A subsequent TCP worker candidate consumes only merged upstream #4001's
callback ownership and exception-to-FAILED subset. Real CPU worker controls
pause callback execution across batch free and inject a real staging exception;
old-library failures are preserved. Independent acceptance, installed package
and hardware qualification remain separate. Delayed batch/address reuse and
physical termination are not established by this candidate.

PR #237 is merged at `53e69620`: its accepted CPU/TCP counterexample proves
that logical unregister can return while a previously admitted source copy
still uses the address. The next candidate adds an irreversible native source
admission gate and counts actual inbound handlers before pointer resolution.
Only a CPU-only, TCP-only engine whose handlers have returned can report drain;
the Rust pool owner also requires its local requester admission to be empty
before unregistering. TCP requests carry a native engine epoch, and retries
cannot silently switch to a restarted source epoch. This is whole-engine
shutdown, not per-ticket orphan reclamation or allocation/page generation.
The original source-close packet epoch was captured at the first TENT segment
lookup. Its controls did not bind the OrbitKV source authorization reply; the
separately accepted grant-epoch substage below closes that CPU/TCP first-submit
code/control gap. Installed Manager and hardware qualification remain open.
The focused native extension is an unreleased candidate on the fixed pin;
matching peer/runtime packages are required. CUDA builds and other loaded
transports return unsupported, retaining the engine, registration tokens and
pool. Live and retained source pools share a process admission cap of 1,024
pools and 1 TiB of registered backing, distinct from batch descriptors and
per-query/export payload budgets. Independent `ACCEPTED_CODE_CPU_SCOPE` at
`09b77062` covers source/dispatch drain and actual Rust-owner CPU boundaries;
installed package, GPU/RDMA and adapter qualification remain open. Preserve
all controls under `/root/orbitkv-artifacts/s3-source-read-drain-20261014/`.
The first checkpoint `dd5589c5` was independently rejected for an unguarded
Delegate pointer entry and an invalid close receipt on an unavailable engine.
Its freeze and failing controls remain preserved. The correction refuses that
unversioned delegate entry in the CPU/TCP receipt profile and validates native
availability/epoch before touching receipt outputs. Native byte controls and
test-only host-registration/C ABI controls of the actual Rust pool owner are
separate from real CUDA registration, installed-package and GPU/RDMA gates.
Further teardown review found queued global-executor callbacks and staging RPCs
were not covered by server I/O shutdown. The current correction counts dispatch
before offload, closes that admission, waits actual callback completion within
the same source-close budget, and publishes readiness at the final successful
constructor boundary. Pin/Unpin are refused in the CPU receipt profile. Existing
failed checkpoints remain rejected. Independent exact rereview accepted
`09b770623ff69248235d75b69abbedbd77c4ffd2`, tree `e2d96237`, and reran seven
native source, three queued/running/throw dispatch and five actual-owner CPU
controls. Every child exited zero with verified mappings and actual reap, with
no forced cleanup. Review and verdict are under `independent-final09b-review/`
in that artifact root; review SHA256 is
`7753ed43fbeab8322eb5072b942c8fce4416c2932ac6163db367d17f51f23c66`.
This accepts neither CUDA host registration nor device/RDMA termination,
per-ticket reclamation, old-grant first submission, page reuse or adapters.

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
  The grant-epoch implementation captures the registered transport's epoch in
  the source authorization reply and preserves it through Rust admission and
  the C request into the first TCP attempt. A fresh TENT metadata check rejects
  stale source epochs before batch allocation; the TCP worker also rejects a
  changed epoch after that check. Independent `ACCEPTED_CODE_CPU_SCOPE` at
  `02ab7fe317c003c9b94d7f1f0ed3397af2427365`, tree `1c9a9499`, covers this
  code/TCP control boundary. Its five independently rerun native scenes first
  establish healthy replacement bytes, then explicitly reject the old epoch;
  the paired cleared expectation reads all 8,192 replacement bytes. The prior
  `b7d50489` EOF-only race oracle remains rejected and archived. Review SHA256
  `e6d9c45444f08a37ce5080d9833052ba8f95fdaa126432652bc101626adf7535` and
  new freeze manifest
  `1542bc40046b4eb5a952cd5f735a20f2335be35459a0a504c270cc192988b449`
  are under `/root/orbitkv-artifacts/s3-grant-native-epoch-20261015/`.
  This does not implement allocation/page generation, RDMA QP/MR revocation or
  physical termination. Actual installed Manager/server RPC, GPU and both
  adapter lifetime gates remain open. Before adding page reuse, audit managed
  Restore errors so process exit or unreadable completion cannot return pages
  without an actual completion or never-submitted proof.
- The managed-Restore retention candidate preserves a pending `Managed` record
  after authenticated Manager process exit. Unknown claim/poll/completion errors
  in the actual engine-local worker keep the job and tensor owners instead of
  publishing a terminal failure. Pending diagnostics are bounded to 1 KiB per
  operation, with the existing 1,024-operation worker admission still charged;
  deadlines preserve ownership. Already drained local work and actual Manager
  terminal publications retain their normal completion path. The first
  `e635284f` checkpoint is independently rejected: a generic claim error could
  observe another claimant's `Active` state and incorrectly publish drain.
  Its freeze, wheel and review remain preserved. The correction keeps every
  generic claim error pending. Only the completion owner that won its own CAS
  may mark a rejected plan as never submitted, before returning bytes to any
  executor; deterministic competing-claim and old-branch mutation controls
  exercise that boundary. This is not a
  physical-failure drain or reclamation primitive. Independent
  `ACCEPTED_CODE_CPU_AND_INSTALLED_IMPORT_SCOPE` at
  `d9f779c9b844b5265caf09985d319445587cd05b`, tree `39c30ccb`, reruns 66
  channel units, seven process tests and three actual-owner CPU controls.
  Freeze manifest
  `ed03585ec2c30c608856cf09267b2dd0df7d57419fab5d2efb3490d27794049a`,
  wheel `6413f4c7d9748773cacb888a769d9bd46abec191a8828e7984435c56bfe8a655`
  and review SHA256
  `351db332f4644aaa54f4ebb86e099992ba56d039b9cdc696f5b4f2721f74a4ab`
  are under `/root/orbitkv-artifacts/s3-managed-restore-retention-20261015/`.
  Live Manager endpoint bytes, installed GPU/adapter faults and physical
  termination/reclaim remain open. The imported-context owner now uses a
  structured instance/rank/GPU UUID key; cleanup compares the whole instance
  identity, not a prefix shared by another legal instance such as `a:b`.
  Independent `ACCEPTED_CODE_CPU_AND_INSTALLED_IMPORT_SCOPE` covers
  `f8cb87bbb6cc5def38beb7d55f6c8dced5b0be81`, tree `e58480c7`, and integrated
  HEAD `3415b28c`, tree `08b593c8`. Four independently rerun selector/actor
  controls preserve the other context and its duplicate-registration guard;
  the original selector failure remains archived. Freeze manifest
  `347443e7e8186b380c442b92699562cf045fe1924025d16199a52d761392207c`,
  wheel `4c3e609fc23ec0ff76edb16a6cfa3d5b1d25f8bfc9c86e1a7cafad5a65cd4827`
  and review SHA256
  `9bddfbdcbbef5be31419280e3f4063e28282492dc2ad91d88081d9d2be60dbdb`
  are under `/root/orbitkv-artifacts/s3-registration-owner-20261015/`.
  Actual CUDA mapping/lifetime qualification remains open. This does not
  replace the remaining pickle wire or implement allocation/page generation.
- Registration-generation fencing reuses InstanceContext's existing owner
  generation. GPU registration returns it with the Manager-local ordinal;
  the actual native client used by both adapters snapshots it after acquiring
  the local-worker lock, then carries it through every Publish chunk into the
  retained Core owner check before GPU pointer binding. Same-name restart
  rejects captured old Publish input. Pre-snapshot waits and arbitrary late
  engine callbacks remain unqualified. Restore keeps its existing query-lease
  generation checks; lifecycle 8/cache schema 10 require matching artifacts.
  Independent `ACCEPTED_CODE_CPU_AND_INSTALLED_IMPORT_SCOPE` at
  `03cbcebd6c2ba683547730a6e9350e42ee11acae`, tree `d878a3e8`, covers Core1,
  lifecycle1, channel67/process7 and matching installed import controls.
  Core/lifecycle use the host-registration shim and real read-only local CUDA
  driver discovery; no GPU allocation, compute or DMA is tested. The mutation
  disables the whole generation guard. Its old nonzero-owner failure proves
  guard necessity, without isolating equality removal. Original failed scope
  assertions and both corrective addenda remain archived.
  Freeze `203178d2c65647155652a74db7385f19eec2fd982d2e0841b035e0a40cf849b9`,
  wheel `3777a5ee372cd76b277f78815634e34ced81d231c23e69a6655f542757d8d3f6`,
  review `da45664e6b0d47584a5c27b19c307410a2c05b8aa25447954e95e74f49266fc0`
  are under `/root/orbitkv-artifacts/s3-registration-generation-20261015/`.
  Installed GPU/adapter and allocation/page-generation qualification remain open.
  The next implemented substage issues a native `GpuRegistration` at registration
  and requires it for Publish. vLLM request/boundary tasks and SGLang offload
  tasks capture it at producer handoff. Native validation preserves the old
  generation across worker-lock waits and rejects retired/foreign-client input;
  the authority retains exporters after unregister. Matching stubs, reusable
  workloads and test callers consume the new API; the old Publish signature is
  removed. This closes the pre-snapshot/late-callback code boundary pending
  independent CPU/installed-import acceptance. It does not supply engine page
  generations, physical termination or GPU/model qualification. New controls
  and installation evidence are under
  `/root/orbitkv-artifacts/s3-registration-authority-20261015/`.
  The first freeze at `5e34ab34` remains unaccepted: independent review found
  that workload registration dictionaries retained exporters through restart
  and final cleanup. The remediation retires each consumer-owned reference
  after normal unregister/close, retains it on failure, and leaves independently
  queued authority intact. CPU weak-reference controls execute the consumed
  cleanup bodies and reject the original source. Installed default and
  SGLang-only environments are validated separately; the original combined
  environment collection failure is preserved. Independent source/CPU review
  accepts the corrected consumer cleanup at `f338e84b` and the real released-pool
  lifecycle gate at `ff5a9287`. The earlier frozen `c5681616`/`eefcf8f6` runtime
  independently passes two native, six vLLM and two DRAM/SSD SGLang healthy
  installed gates with normal controller/service waits. Those results do not
  test the later consumer-reference delta. Its matching Python-only artifact
  must pass the separately reviewed real SGLang close/independent-hold GPU gate
  before merge; current evidence is linked from
  [PR #246](https://github.com/feichai0017/orbitkv/pull/246).
  Healthy completion does not close physical faults, allocation/page generation
  or S3. Original failures and native IPC/helper-exit limits remain preserved.
- Explicit CUDA registration is implemented and independently accepted for its
  bounded code/CPU and matching healthy installed profile at production
  `6ee6cdca`, test harness `949d0dd5`, wheel SHA256 `0a05a5c9…` (PR
  [#251](https://github.com/feichai0017/orbitkv/pull/251)). Versioned protobuf
  storage/view metadata replaces the framework pickle wrapper; lifecycle 9
  requires matching client and Manager packages. Per-registration metadata
  validation precedes device resolution/import. The native owner reserves the
  client/key before export,
  retains the original storage binding and per-layer owners/credits, and checks
  generation-bound import dispositions. Uncertain or partial attempts retain
  their client/tensors/credits. Repayment requires the original held counter
  identity and an observed one-to-zero transition; an importer handoff or zero
  counter is not physical device completion.
  Retention is bounded to 64 attempts, 4,096 tensors/credits per attempt and
  1 TiB of conservatively accounted storage. Identical `(base, bytes)` aliases
  are charged once inside a validated single-device batch, avoiding repeated
  OrbitKV accounting of vLLM's shared layer backing. Different extents, overlaps,
  attempts and every layer's ownership remain independent.
  Matching H20 installed gates pass four core cells/seven cases; official
  vLLM 0.31.0 Qwen3-8B TP1/PP1 V1/direct DRAM with CUDA Graph NONE passes
  six cases with one inapplicable hybrid-model skip; official SGLang 0.5.21 Qwen3-8B TP1/PP1
  direct DRAM/io_uring SSD passes both recovery cases. These checks cover
  retained-credit controls, native Restore bytes, producer-stream fencing,
  released-pool retirement, model output and restart recovery under their
  respective frozen contracts. Literal service/controller/foreground waits,
  installed input hashes and selected-resource postflight are independently
  checked. They establish healthy scoped correctness, not complete lifecycle,
  physical-fault reclamation, page generations, RDMA/GDS or performance gains.
  Core and SGLang IPC warnings, uncertain throw retention and official
  engine-internal termination limits remain explicit in the accepted reports.
  Full frozen inputs, original failures, retained old owners and independent
  reports are preserved under
  `/root/orbitkv-artifacts/s3-installed-registration-completion-20261017/`.
  The [immutable pre-acceptance record](https://github.com/feichai0017/orbitkv/blob/949d0dd50e149f9a2b74d738838719a589f481ac/docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership)
  retains the earlier failed candidates, source fixes, runtime closures and
  rejected launch preparations. This summary does not revise their verdicts.

- Publish retains the existing synchronous caller, descriptors and source owners
  after unproven transport errors, invalid acknowledgements, Manager process exit
  or a caught Rust unwind. Fallible stderr diagnostics avoid a panicking logger
  on the final retention path. Request-send errors retain their native type:
  only loan, pre-delivery connection and active-request admission failures prove
  no submission. A send error may follow delivery and cannot be treated as a
  blanket pre-submission exception. Request services enforce one Manager
  recipient at creation and check that exact configuration when opening.
  On an unproven reply or transport error, Publish closes its session and releases
  the shared descriptor lock before retaining the original caller. Other callers on that same
  ChannelClient reject the closed session without reusing its descriptor slot.
  Real CPU process controls cover queue-delivery interruption, source ownership,
  pre-admission rejection, valid late completion, peer exit, malformed replies
  and logger unwind. The historical unsafe send-exception mutation must fail
  the same source-retention control. This is transport-boundary hardening,
  tracked in [PR #259](https://github.com/feichai0017/orbitkv/pull/259).
  Independent acceptance at `47e4f18a` covers code/CPU transport only; main
  `3b48b463` has the same reviewed tree. The review and immutable controls are
  under `/root/orbitkv-artifacts/s3-publish-lifetime-20261018/independent-47e4-review-1/`.
  The accepted SaveTask correction below prevents worker/reply loss alone from
  producing a consumed Internal response. Bounded live-stall response,
  finite parked-operation admission, page generations and physical reclamation
  remain open. New installed-package or GPU/fault qualification cannot be
  inherited from the earlier registration wheel or these CPU controls.
  The next SaveTask owner correction keeps an admitted `batch_save` future
  pending after oneshot reply loss instead of publishing a consumed error.
  Borrowed SaveTask resources and admission survive execution unwinds; raw
  copy and codec D2H boundaries retain their outer resources. Stream or codec
  synchronization failures retain those owners instead of terminating the
  Manager and exposing page reuse through process death. Real CPU queue/reply
  loss and isolated borrowed-owner/admission controls establish the software
  boundary; actual GPU partial-submit, delayed completion, failed sync and
  installed adapter byte/lifetime gates remain required. This candidate does
  not establish RDMA/SSD physical termination or provide a finite live-stall
  response, and no historical quarantine is released.
  SaveTask now carries the existing registration instance after caller-future
  cancellation. A process-wide owner admits at most 1,024 Save operations and
  32 MiB of retained request-vector/SaveTask metadata, counting spare capacity.
  Missing receipts remain charged; only pre-submission rejection or the actual
  worker completion boundary repays admission. Active metadata and logical-byte
  gauges distinguish this budget from the existing physical pool/SSD/GPU owners;
  logical bytes are measured, not a new physical payload cap. Reply/owner loss
  has an explicit retained-operation gauge; parked sync/unwind work remains in
  the active gauges. CPU saturation, overflow, cancellation, reply-loss and
  fresh metric controls are archived under
  `/root/orbitkv-artifacts/s3-publish-worker-drain-20261018/`.
  Independent acceptance at `df8886ed` covers code/CPU Save ownership and matching
  package loading: 320 Core tests passed, 26 ignored, shared-Python Clippy passed,
  the historical reply-loss-error mutation failed, and the isolated wheel loaded
  its extension plus bundled TENT. The review is under
  `/root/orbitkv-artifacts/s3-publish-worker-drain-20261018/independent-df8886-review-1/`;
  [PR #260](https://github.com/feichai0017/orbitkv/pull/260) carries the implementation.
  Package loading does not establish installed Manager/device operation.
  Arbitrary SSD-worker unwind and engine-local Restore failures remain
  separate open ownership/qualification items.
- A native `peer_alive()` observation uses the authenticated Manager pidfd,
  without a control RPC or taking a blocked Publish/lifecycle lock. Restore
  readiness and pending-result checks consume that observation. A live stalled
  peer remains alive; replacing the stable UDS does not change the old process
  identity. Unknown/invalid descriptors are errors. Independent code/CPU and
  installed peer-API acceptance at `cdd45f05` covers the authenticated pidfd and
  replacement controls. Rust readiness is a recheck hint; Python readiness keeps
  its local DMA-result eventfd. SGLang probing and actual GPU/fault gates remain
  open; peer exit still does not fence active engine DMA or release pages.
- Add a bounded operational response to a live Manager that never completes a
  Publish, preserving held pages until matched completion or independently
  established physical termination. Process death alone is not a DMA fence.
  The next local-observation substage exposes synchronous Publish activity,
  explicit retention, elapsed active interval and authenticated pidfd liveness
  independently of the descriptor/lifecycle locks. Contention is an unavailable
  observation, not zero active work. Official vLLM/SGLang completion polls consume
  it for bounded diagnostics only; neither overdue nor process exit completes
  an offload. The synchronous call and page holds remain in place. This is not
  the asynchronous pending-handle response or page-generation implementation;
  those require their own consumed ownership and installed engine gates.
  Independent acceptance at `26884461` covers the local observation, CPU controls
  and candidate installed API. Channel tests, overlap/late-reply controls and
  an expected fake-zero mutation passed their declared results. Both adapters
  consume diagnostics; the SGLang method control uses exact isolated 0.5.21.
  Source/wheel origins and the stale local 0.5.20 result remain separately
  recorded under `/root/orbitkv-artifacts/s3-publish-observation-20261018/`.
  [PR #261](https://github.com/feichai0017/orbitkv/pull/261) adds no asynchronous
  completion, GPU/physical gate or serving-overhead qualification.

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

**Scoped H20 functional acceptance (2026-10-14):** engine-local raw H2D
captures carry Linux thread identity and monotonic submission/part-drain windows.
The installed-wheel SGLang 0.5.21 page-pool gate passed three generations of full
payload/sentinel checking, including partial destinations and deliberate
corruption. Independent replay consumed the production descriptor compiler and
matched 384 batches / 22,144 H2D calls; all native owners exited normally and
installed-file hashes were unchanged. This does not infer packing cost,
first-layer readiness, physical fault reclamation or serving gains; matched
observation-overhead and inference-contention gates remain required. The frozen
one-cell evidence and preceding invalid CLI attempt are preserved with
[PR #238](https://github.com/feichai0017/orbitkv/pull/238).

**Observation-overhead pilot (2026-10-15): implemented; qualification open.**
The matched byte oracle and analyzer were independently accepted at `35dd3e2c`;
all 13 exact-head CI checks passed. The single reviewed eight-cell H20 pilot
stopped after five completed cells because each on-cell directory contained an
unexpected child-process CUPTI capture. The final analyzer rejects this exact
inventory; the external controller's per-cell check missed it and allowed later
cells to run before the stop. Preserve both main and child captures, the normal
native waits and the unchanged runtime; do not treat the five-cell timings as an
overhead PASS or FAIL. Fifteen complete restores passed payload/sentinel and
corruption controls, with SDK/Manager exit zero and explicit native release.
No retry or replacement cell ran; GPU1 was released normally and GPU0's prior
failed-run reservation remains untouched. The invalid cohort and reviews are
preserved outside the checkout with [PR #238](https://github.com/feichai0017/orbitkv/pull/238).

**Capture-scope repair (2026-10-15): implemented; matched overhead FAIL.**
The oracle explicitly loads the frozen collector in its own process before
Torch/engine/native imports and retains it through completion. It rejects
inherited preloads and ambiguous trace directories. CPU-only controls using the
actual collector with stub CUPTI reproduce the old exec-child capture and verify
that explicit loading captures only the main process. This does not establish
old helper lineage, fork support or descendant/DMA safety.

The new v2 pilot contract binds the explicit loader, waited PID, library path and
SHA256. A shared census rejects missing/foreign/symlink captures and disabled-mode
directories; both per-cell admission and final analysis consume it before and
after replay. The controller also rejects changed byte-result evidence before
accepting a cell. No production wheel, collector C++, backend or ownership policy
changes; the original invalid cohort stays invalid.

**Completed H20 v2 cohort (2026-10-15):** the fresh `7601af23` runtime uses
matching lifecycle-8/cache-10 artifacts. All eight cells and 24 full-layout
restores pass payload/sentinel and corruption controls, with native release, normal
SDK/Manager/controller waits and unchanged installed files. The four main-only
captures pass exact inventory and production-compiler replay. The cohort is valid
but `PILOT_OVERHEAD_FAIL`: all four pair ratios exceed 1.05; the pair-geometric
ratio is 1.4133, with independent-pair 95% interval [1.4036, 1.4231]. This measures
combined diagnostic overhead; the small pool-only pilot does not qualify p99,
model output or serving gains. The original invalid cohort and two rejected
zero-GPU prelaunch freezes remain preserved. See
[matched overhead methodology](kernel-optimization.md#matched-observation-overhead-pilot).

**Observation-cost delivery (2026-10-15): implemented; matched overhead FAIL.**
Engine-local trace records serialize borrowed typed fields directly, without
intermediate JSON trees for every descriptor. Schema-2 geometry, cumulative caps,
part clocks and synchronous publication/flush before return are unchanged.
A buffered real-child immediate-exit control protects flush, and the actual
Rust output is parsed by the maintained consumer and production DMA compiler.
The component analyzer adds four balanced off/native/driver/on blocks; each mode
occupies every position once, while the four primary off/on pairs keep their
alternating order and unchanged 5% guard. Component costs are reported separately
without assuming additivity.

The one fresh H20 component cohort uses production `cf4de1e3` and measured harness
`ed09686c`: all 16 cells, 48 complete restore oracles and 96 corruption controls
pass. Independent review accepts the evidence and unchanged failed decision.
All 34 owned process identities end after actual normal waits; GPU1's original
lease is released with zero memory/compute, and GPU0's prior failed-run reservation is unchanged. Frozen input and installed-file
hashes match. The combined/off pair-geometric ratio is 1.1281, with independent-pair
95% interval [1.1242, 1.1324]; every pair exceeds 1.05.
The native/off contrast is 1.1242 while driver/off is near one, identifying the
native observation bundle as the next profiling target. This does not isolate
individual native functions or assume additive effects.
The cohort is valid but `PILOT_OVERHEAD_FAIL`. The maintained consumer at
`fd142c59` additionally rejects foreign or linked cell entries before and after
replay; both consumer revisions reproduce the same raw result. Raw evidence and
the complete runtime closure are archived outside the checkout with
[PR #247](https://github.com/feichai0017/orbitkv/pull/247). See the
[component result](kernel-optimization.md#component-cohort-result).

**Native trace CPU profile (2026-10-15): implemented and collected; CPU scope only.**
The actual private trace owner replays the preceding complete H20 geometry in
five independent staged/whole process pairs. All ten processes and 360 CPU
replays pass full output, identity, clock and cap checks. Encoding and synchronous
publication dominate every staged process; process variability remains explicit,
and this CPU profile does not establish H20 latency or a production speedup.
See the [CPU profile and limits](kernel-optimization.md#native-trace-cpu-profile)
and [PR #248](https://github.com/feichai0017/orbitkv/pull/248).

**Compact trace delivery (2026-10-15): implemented; matched overhead FAIL.**
Schema-3 descriptors contain the same fifteen unsigned geometry values in a
fixed order declared by every opening record. The reader restores named geometry
before all existing validation; archived schema-2 captures remain readable and
mixed-schema executors are rejected. Capture, budgets, clocks, physical drain
and synchronous publication/flush are unchanged. CPU controls cover both formats,
unsigned extremes, malformed rows and the real buffered immediate-exit boundary.
This is an observation-format change, not a copy backend or ownership change.

The ten-process CPU replay preserves every scalar value and record boundary.
For identical emitted values, compact JSON uses 60.27% fewer bytes than named
geometry; this is a representation comparison, not a matched timing gain.
The single fresh H20 cohort at `d3624cc2` completes all 16 cells, 48 full-layout
restores and 96 corruption controls with normal native/Manager/process waits,
unchanged runtime hashes and released GPU1 ownership. Its combined/off
pair-geometric ratio is 1.0729, with independent-pair 95% interval
[1.0690, 1.0765]; every pair remains above 1.05: `PILOT_OVERHEAD_FAIL`.
Complete raw evidence and the runtime closure are retained outside Git with
[PR #249](https://github.com/feichai0017/orbitkv/pull/249). This does not promote
inference contention, a copy default, serving performance or S3 lifetime.

**Remaining delivery:** reduce native observation cost inside the existing
owner. The column-RLE candidate below passes CPU retention but still fails the
complete H20 observation guard. Profile the remaining capture, encoding and
synchronous publication cost before selecting another change. Preserve every
scalar, record order, cumulative cap, completion clock and synchronous flush;
retain the complete byte/capture/drain controls and 5% guard. Any new candidate
needs independent code/runtime/launch review and a newly frozen cohort; completed
cohorts are not retried or replaced. Inference contention and further
kernel/default promotion remain blocked; S3/S4/S5 limits are unchanged.

**JSON workspace candidate: rejected by its fixed CPU comparison.**
The candidate at `b16bead2` reuses at most 64 KiB of JSON workspace in the
existing owner without changing canonical bytes, record order, clocks, capture
caps or synchronous flush. All twenty fixed CPU cells preserve 720 replays and
87,091,200 geometry values, with normal profile/compiler/controller waits and
unchanged frozen inputs. Staged encoding ratio is 0.8040, with independent-pair
95% interval [0.7117, 0.9147], but whole-capture ratio is 1.0333
[0.9134, 1.1688]; two whole pairs exceed 1.05. The candidate fails its
predeclared whole-capture guards and is not retained. Production source is
restored to the merged `2d6c77ac` baseline; the rejected source, binaries and
complete failed cohort remain outside Git at
`/root/orbitkv-artifacts/s4-trace-json-workspace-20261016/`.
No H20 cohort or dependent inference-contention run is admitted; prior results
and S3/S4/S5 limits remain unchanged. Inspect capture allocation, memory footprint and whole-operation
variance before selecting another bounded candidate; the component improvement
does not establish a whole-path gain or identify the cause of failure.

**Baseline A/A diagnosis: collected; repeatability remains scoped.**
The accepted native CPU executable is compared with itself in 32 fixed whole-profile
cells, with four complete ABBA/BAAB blocks separately at CPU0 and CPU2. All cells
preserve complete geometry/canonical bytes and normal profile/compiler waits.
CPU0 does not demonstrate the frozen symmetric 5% repeatability band: block
geometric ratio 0.9146, 95% interval [0.7922, 1.0086]. CPU2 demonstrates it only
in this cohort: 0.9981 [0.9945, 1.0017]. These are identical-code repeatability
results, not optimization gains, a CPU ranking or a correction to prior FAILs.
Shared resource counters do not establish the cause of timed-part differences.
See the [A/A methodology](kernel-optimization.md#baseline-aa-repeatability-diagnosis)
and complete evidence at `/root/orbitkv-artifacts/s4-trace-aa-diagnosis-20261016/`.
Declare and validate subsequent measurement conditions before another candidate;
no CPU setting, GPU run or dependent inference-contended experiment is promoted.

**Thread CPU decomposition: implemented; probe overhead unqualified.**
The explicit CPU benchmark adds opt-in calling-thread CPU stamps and enclosing
context-switch/fault counters. Keep the production owner and original stage
clocks unchanged. Independently calibrate off/thread in the same newly built
executable using 32 fixed processes, guarding both the existing total and an
outer envelope that includes the resource probes, separately at CPU0 and CPU2.
Preserve signed, matched-endpoint residuals and complete geometry validation.
The [measurement contract](kernel-optimization.md#thread-cpu-decomposition)
precedes the sole cohort. All 32 cells preserve complete geometry and normal
profile/compiler/controller exits. CPU0 fails both guards: total ratio 1.0852
[1.0012, 1.1762], enclosing ratio 1.0861 [1.0015, 1.1777]. CPU2 passes both
within this cohort: 1.0137 [1.0107, 1.0166] and 1.0140 [1.0111, 1.0170].
The required conjunction fails; the optional diagnostic remains unqualified.
Retain every sample and the historical CPU0 A/A failure; do not interpret this as
an established probe cost, root cause or CPU ranking. No new candidate, H20 run
or dependent inference contention is admitted.

**Measurement-method preparation (2026-10-17): archive description only.**
All 32 sealed baseline A/A cells and 768 measured parts are reproduced without
new native/GPU runs. The post hoc sequence contrast does not qualify a method or
change any prior FAIL. Before another same-binary instrumentation comparison,
predeclare balanced within-process order, independent restarted-process units,
both CPUs, original 5% guards and complete geometry/byte/drain controls. Obtain
method/source and launch review before collection. This calibration cannot
validate earlier between-binary optimization comparisons or promote serving.
The description and reproduction remain outside Git at
`/root/orbitkv-artifacts/s4-paired-measurement-preparation-20261017/`.

**Prospective interleaved calibration (2026-10-17): implemented; qualification FAIL.**
The sole fixed 16-process CPU cohort completes all 576 parts (384 measured),
full geometry/canonical-byte/cap/drain validation and 33 normal child waits.
The [prospective median3 contract](kernel-optimization.md#prospective-interleaved-calibration)
keeps both CPUs and original numeric 5% guards. All process estimates and
bootstrap intervals satisfy their guards, but adjacent contrasts fail:
CPU0 A/A 7/16 and A/B 3/16; CPU2 A/A passes and A/B fails 1/16, for both metrics.
A/A prerequisite and A/B conjunction fail; the method and optional probe remain
unqualified. Sequence medians do not bound individual parts, sequence sums or
former process medians. No old FAIL, hardware/default or serving boundary changes;
no replacement or subsequent campaign. Complete evidence remains outside Git at
`/root/orbitkv-artifacts/s4-interleaved-calibration-20261017/`.

**Bounded synchronous publication: CPU retention accepted; H20 overhead FAIL.**
The candidate groups already-drained diagnostic JSON lines into
synchronous log messages of at most 64 KiB. Keep opening records on the original
pre-submission path, all tagged JSON lines in order, intact oversized records,
and terminal publication plus logger flush before returning. Clock placement,
cumulative capture budgets and physical ownership remain unchanged. Measure
chunk assembly and terminal-tail publication inside the existing whole-operation
and staged publication clocks; retain at most 64 KiB of publication payload
storage per trace, allocated lazily inside the measured path. Allocator bookkeeping
and temporary serialization storage remain separate measured costs. Log event
grouping changes; compatibility is limited to tested line-preserving loggers.
Independent design, source and complete CPU-data review accepts retention at
`6265bc8c` against merged `6a00e221`. All twenty fixed cells and 720 replays
preserve 87,091,200 reconstructed geometry values. The whole-capture geometric
ratio is 0.8256, with independent-pair 95% interval [0.7514, 0.9071]; all five
whole pairs pass the frozen 1.05 guard. Synchronous publication ratio is 0.4876
[0.4327, 0.5472]. These are CPU replay costs, not GPU restore or serving gains.
The matching installed wheel/runtime and fresh H20 launch pass independent
review. The sole sixteen-cell cohort completes 48 full-layout restores and 96
corruption controls, with normal SDK/Manager/controller/foreground SSH waits
and unchanged frozen files. Native-only geometric ratio is 1.0567, with 95%
interval [1.0437, 1.0770]; combined is 1.0538 [1.0466, 1.0612]. Both fail the
unchanged 1.05 guard. Driver-only is 1.0033 [0.9981, 1.0109] and passes. The
terminal status is `PILOT_OVERHEAD_FAIL`; dependent inference contention remains
blocked. CPU retention does not qualify GPU restoration, serving or defaults.
Prior valid failed cohorts and S3/S4/S5 limits remain unchanged. The KDA draft,
fixed guards and complete evidence are outside Git at
`/root/orbitkv-artifacts/s4-trace-publication-20261016/`.

The first schema-4 row-varint candidate at `5fb3d61f` passes independent source
and complete CPU-data review, but fails its predeclared retention criteria.
All twenty processes preserve 720 restores and every recorded value; the whole
ratio's independent-pair 95% upper bound is 1.1427, its worst pair is 1.4275,
and the encoding geometric ratio is 1.2726. The candidate is not retained and
does not authorize H20. Its frozen source, reader, binaries and complete failed
cohort remain at `/root/orbitkv-artifacts/s4-varint-trace-20261016/`.

The distinct follow-up uses canonical column run-length/value pairs with an
explicit row count and `uvarint-rle-base64-v1` tag. It preserves all fifteen
unsigned values and their row order; the bounded reader restores named geometry
for the unchanged checks. Capture, clocks, cumulative caps and synchronous flush
remain unchanged. Independent review accepts CPU retention at `afe30921`:
all twenty cells, 720 replays and 87,091,200 reconstructed values pass. Whole
candidate/baseline geometric ratio is 0.7370, with independent-pair 95% interval
[0.6157, 0.8822]; every whole pair passes the frozen 1.05 guard. Staged encoding
ratio is 0.6430. These are CPU replay results, not GPU restore or serving gains.

The complete CUDA 13 wheel and consumed-import closure pass independent
runtime/READY/launch review in an isolated H20 prefix. The sole fresh sixteen-cell
cohort completes 48 full-layout restores and 96 corruption controls with actual
normal process waits and unchanged frozen inputs. Four independent matched
blocks distinguish native-only, CUDA-driver-only and combined observation from
off. Native-only geometric ratio is 1.0582, with 95% interval [1.0481, 1.0708];
combined is 1.0609 [1.0567, 1.0656]. Both fail the unchanged 1.05 guard.
Driver-only is 0.9966 [0.9915, 1.0018] and passes. The exact terminal status is
`PILOT_OVERHEAD_FAIL`. There is no second cohort or replacement cell. This is
complete-restoration host-wall observation evidence, not p99, model serving or
a backend/default qualification; inference contention remains blocked.

The offline DMA fixture separately buffers JSON input; its identical full
compiler receipt is independently accepted and its validation time stays outside
trace measurement. CPU retention, terminal raw data and the full frozen runtime
remain outside Git at `/root/orbitkv-artifacts/s4-rle-trace-20261016/`.

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

**Scoped installed-wheel acceptance (2026-10-14):** SGLang
pending-query admission consumes authenticated native `peer_alive()`, without a
health RPC or using peer death as a DMA fence. Peer failure preserves the query
owner and fails explicitly. Ready dense lookups reuse leased recovery boundaries
until their keys or absolute origin change. Native HBM/no-pending admission adds
no peer probe. The final wheel from `f6d4bff2` passes the official SGLang 0.5.21
Qwen3-8B TP=1/PP=1 DRAM and io_uring model controls: restore after flush, concurrent
shared prefixes, restart, changed-identity misses and same-engine output checks.
Independent review accepts both cases, unchanged installed/model hashes and eight
normal service receipts. Failed-peer behavior has software coverage; no GPU
peer-failure injection ran. Six official internal helper SIGKILL exits and CUDA
IPC producer warnings remain explicit S3 limits. Public callbacks, physical
faults and end-to-end performance retain their separate gates. Frozen evidence
and preceding launch failures are preserved with
[PR #238](https://github.com/feichai0017/orbitkv/pull/238).

**vLLM final-wheel controls; native shutdown rejected (2026-10-14):** the
official 0.31.0 Qwen3-8B dense DRAM gate consumes actual save/restart recovery,
same-engine outputs and zero RPC/load-failure assertions: six checks pass and
the unsupported hybrid assertion skips. Four services exit zero without test
forcing. The stricter frozen drain profile still fails because the native Noop
baseline logs `AsyncLLM output_handler failed` / `EngineDeadError` during
intentional shutdown. This matches the released teardown ordering described in
[vLLM #48745](https://github.com/vllm-project/vllm/issues/48745); the associated
[fix #49000](https://github.com/vllm-project/vllm/pull/49000) remains unmerged.
The failed contract and quarantine are preserved with PR #238; functional
assertions do not turn that cohort into a pass or qualify S3 physical lifetime.
The earlier missing-Manager-binary attempt also remains failed. No fork patch
or relaxed log rule is installed, and the support matrix is unchanged.

**Explicit-registration wheel cutover (2026-10-17):** the new installed
registration prerequisite and matching official-release model regression pass
from the same frozen wheel: vLLM dense V1/direct DRAM with CUDA Graph NONE, and
SGLang direct DRAM/io_uring SSD, all TP1/PP1 with Qwen3-8B. The current vLLM native baseline/save/
restart logs also pass the unchanged strict validator; the historical Noop
shutdown failure above keeps its failed verdict. Exact profiles and independent
reviews are tracked under [S3](#s3--transfer-lifetime-and-generation-safe-ownership).
Recheck the actual artifact for each newly claimed engine/media profile. Healthy
output and recovery assertions do not replace strict native shutdown, physical
lifetime or fault qualification, and prior failed cohorts keep their original
verdicts.

### S5.3 — Public lifecycle and hybrid-state contracts

**Partial.** Idle process-scoped vLLM external reset is accepted at `6d874384`;
immutable local static PEFT identity at `47fea821`; the official dense query-control
handshake at `1ab67a58`. These do not qualify live weight changes, dynamic adapter
mutation or physical multi-rank recovery.

**Public worker preparation (2026-10-17): upstream contribution; adoption open.**
The isolated SGLang candidate at `ae5f357e` adds an optional registered-backend
preparation callback before every runner in `TpModelWorker.init_cuda_graphs`.
It preserves original factories, selects the published backend and passes the
original worker and decode-capture flag. Invalid registration is rejected before
insertion, and preparation failure stops before capture. The candidate-source
registry/worker suite passes 32 CPU tests.
[SGLang issue #44855](https://github.com/sgl-project/sglang/issues/44855) and
[draft PR #44856](https://github.com/sgl-project/sglang/pull/44856) track the generic
interface proposal; independent review accepts only contribution CPU scope.
No OrbitKV runtime, engine pin or installed engine file is changed. Real GPU
capture/replay, released adoption and eager/graph restore plus page-reuse gates
remain open; keep the private pre-capture Hook and separate pending-admission
protection. Complete contribution and review evidence stays outside Git at
`/root/orbitkv-artifacts/s5-sglang-worker-preparation-20261017/`.

**Public pending lookup (2026-10-17): implemented contribution; adoption open.**
The isolated SGLang candidate `c950c381` on upstream `bce7b27a` allows `None` for
pending lookup and preserves `[]` for a completed miss. Current-match state
reaches the native `PrefillAdder` guard before budgets, locks or destination
loads, allowing ready requests to continue. Readiness shares slot zero of the
existing sparse-boundary MIN reduction; a local miss is not cancelled while
another rank remains pending. Native abort and session/disabled/HBM bypasses
retire lookup-only interest and old hit markers. Lookup cancellation errors keep
the existing queued/started-load cleanup responsibility; they do not prove DMA
drain. Only actually admitted requests occupy the native LoRA batch set. Backends
retain terminal results and the original finite deadline across rematches, and
own input validation plus stale callbacks. Earlier Scheduler capacity/preemption
checks keep their existing order; the guard protects `PrefillAdder` operations.

The eight affected source CPU suites pass 165 tests and 66 subtests, with 16
explicit CUDA-fixture skips. Genuine module paths, uninitialized CUDA and all
3,722 unchanged installed Python files are recorded; borrowed SGLang distribution
metadata is 0.5.20, separate from tested candidate source and selected 0.5.21.
[SGLang issue #44872](https://github.com/sgl-project/sglang/issues/44872) and
[draft PR #44873](https://github.com/sgl-project/sglang/pull/44873) track the
independently accepted contribution CPU scope. Evidence, including
baseline counterexamples and earlier failed fixture runs, stays at
`/root/orbitkv-artifacts/s5-sglang-pending-admission-20261017/`.
No installed engine, OrbitKV runtime Hook, engine pin or backend default changes.
Real GPU/rank/model/cancellation/page-reuse gates and released adoption remain
open. Static LoRA admission validation remains a separate protected behavior.

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
S4/S5 stay Partial; S2 is complete only for its accepted engineering scope.

**Bounded attribution delivery:** source `4b568860` adds benchmark-only CUPTI
driver callbacks and optional monotonic windows to the native layout fixture.
One H20 functional capture completes both directions, DMA/kernel and event modes:
80 submissions, 32 measured operations, full byte/source/guard controls and 5,017
driver records. The strict analyzer verifies process/thread/window identity and
compiled copy counts; native and controller exit normally with empty physical
postflight. Independent review accepts this functional capture only.
Non-driver host time stays unattributed. Matched observation-overhead controls
and consumed engine attribution remain open before selecting further copy or
scheduling changes; production backends and defaults are unchanged.
See [native diagnosis](kernel-optimization.md#native-driver-submission-diagnosis)
and [external evidence](benchmark-evidence.md#native-driver-capture-archive).

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
