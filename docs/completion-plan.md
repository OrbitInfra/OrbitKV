# Completion and upstream integration plan

This is the single execution queue for OrbitKV. Architecture documents describe
contracts; deployment documents describe supported configurations. Neither is a
second checklist. The implementing agent delivers one stage or coherent substage;
the reviewer independently checks its consumed path, tests and evidence.

## Release baseline and reference policy

Official latest non-prerelease versions were checked on **2026-10-06**:

| Project | Reference release / commit | OrbitKV qualification |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) | `v0.31.0` / `db9527a46873454610df6dbedf79a36d6bf1a7f6` | Aligned official release pins; final installed-wheel DRAM/io_uring and model gates pass locally, including 15 released callback/recovery checks. Independent acceptance and broader profiles remain open; historical 0.30.0 graphs are not inherited. |
| [SGLang](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) | `v0.5.21` / `e00930c5489053f26d86b179cee0d087f846acbb` | Aligned official release pins; final installed-wheel DRAM/io_uring, direct GPU restart and 43 released event/recovery checks pass locally. Independent acceptance and public lifecycle/deployment gates remain open. |
| [LMCache](https://github.com/LMCache/LMCache/releases/tag/v0.5.5) | `v0.5.5` / `05a013b29da78cf2321b9b46ec5039dde2fb0bb0` | Integration and matched-comparison reference, not evidence of OrbitKV support. |

At the start of an engine upgrade, recheck official releases, record the selected
tag and commit, then freeze that version for the review. Inspect upstream main
and related issues/PRs for existing fixes; main-only APIs are not release APIs.
Upgrade dependency pins, lockfiles, submodules, examples and qualification together.
If a required fix is unreleased, identify the upstream patch and keep that profile
experimental. Remove superseded version branches at cutover rather than building
a multi-version compatibility layer. A documentation update alone never upgrades
the runtime or establishes output/performance parity.

The [adapter contract](adapters.md) compares the released LMCache integrations
and defines engine/cache ownership. The first upstream contribution is a small,
maintainable cache backend with explicit support limits; hybrid lifecycle and P/D
extensions can land separately. Maintainer acceptance is external to this plan:
record submitted, merged and released as different states.

## Delivery status and dependencies

| Stage | Status / next action |
| --- | --- |
| S0 | Agent handoff and Codex skill migration merged in PR #190. |
| S1 | Evidence separation independently verified at `ec3add9b`; this delivery consolidates plans and release-based integration guidance. Acceptance covers S1 only; native CUDA qualification remains blocked on this host. |
| S2 | Partial: S2.1–S2.9 and S2.10 same-host correctness are independently accepted, the latter at `65c51aaa`. Independent final review accepts the second frozen S2.10 same-host formal handoff: all 25 cells are valid and five 16-owner ordinary visibility runs pass 50 ms, while both DRAM and SSD isolation fail the frozen per-pair/CI contract. S2.10b lifecycle, low-overhead observation and 14-cell diagnosis are complete. Independent review accepts the diagnostic archive but blocks a production repair: the dominant measured tail is after native return at the Python observer boundary, while no repeatable Manager, metadata-lock, completion-notification or SSD-owner shift is established. The supported envelope stays four owners and all prior formal campaigns remain immutable. Physical two-H20 ordinary-cache RDMA serving has separate scoped S5.5 acceptance below; HA, independent etcd failure domains, native GDS and S3-dependent cells remain unqualified. |
| S3 | Open: native termination proof, page generations and explicit registration. |
| S4 | Partial: optimize measured execution gaps; qualify mixed communication. |
| S5 | Partial: the S5.1 release/interface audit is independently accepted at `38f8dbb2`. Official 0.31.0/0.5.21 installed ordinary-cache/restart and bounded native P/D model gates pass locally. The final wheel and intact H20 cohort are independently accepted for ordinary-cache GPU recovery and 900-second direct/kernel two-engine shared-Manager io_uring pressure gates with exact native outputs, bounded reservations and normal drain. Recovered original review records and missing reviewer raw files are explicitly documented below. Idle process-scoped vLLM external reset is independently accepted at `6d874384`. Immutable local PEFT adapter identity is independently accepted at `47fea821` for dense TP=1/PP=1 eager official-engine reuse. Explicit serial batch-1 FULL decode Graph cache recovery is independently accepted at `c3f55060` for both official engines on H20 with direct/kernel io_uring SSD. Registered loopback query-control authority and lease ownership are independently accepted at `b1ea00a7`; native common-prefix coordination and the consumed official worker handshake are independently accepted at `1ab67a58` for trusted-loopback two-Manager byte gates and TP=1 eager model recovery. Physical two-H20 RDMA serving from DRAM/io_uring SSD, consumer restart and controlled empty-source restart now have scoped independent acceptance on both official engines, consuming unchanged production `1ab67a58`. Public lifecycle migration, dynamic adapter/live-weight identity, cross-host TP and broader lifetime/deployment qualification remain open. |
| S6 | Partial: observations/limited choices exist; unified executed decisions remain open. |
| S7 | Open: consumed retention/checkpoint compiler beyond recovery validation. |
| S8 | Partial: existing wheel workflow; final images, artifact gates and publication remain open. |

S5.1 interface research and S8 documentation/package preparation may proceed while
S2/S3 run. New reuse, reclamation and early-consumption behavior must wait for its
listed lifetime dependencies. Hardware gaps leave their cells open and do not stop
independent work. Use **implementation open**, **implementation partial**,
**qualification open**, **deferred research**, and **release gate** precisely.

The [distributed inventory and recovery design](distributed-design.md) specifies
the authorized evolution beyond the former etcd block index. S2.7–S2.9 and
S2.10 same-host correctness and ordinary visibility are accepted; the current
S2.10b delivery has completed its bounded same-host isolation diagnosis from
accepted handoff `c5cf2782`. Retain both formal campaigns and their isolation
failures unchanged. Independent review found no production owner path that
authorizes a repair; the next discriminating harness experiment requires a new
reviewed freeze. Keep any replacement formal qualification as a separately
reviewed future cohort. Do not enter S3, S6 or S7 from this delivery. Existing S3
lifetime gates still control any new payload concurrency, reuse or reclamation.
Cross-host HA qualification remains open until actual failure-domain evidence exists.
The separate S5.5 ordinary-cache RDMA serving gate does not close those fault cells.

## Baseline: preserve these implementations

| Area | Present implementation | Remaining boundary |
| --- | --- | --- |
| Local transport | UDS bootstrap, iceoryx2 descriptors, shared arenas and bounded operation ownership | Page-generation coverage, explicit registration data and live-process stall handling |
| Raw copies | Contiguous/strided DMA, bounded multipart Restore, engine-local execution | Residual overhead and earlier multipart consumption |
| Compute overlap | Qualified raw single-part layer/group readiness; coarse dependencies where required | SSD/codec pipelines and legal multipart overlap |
| Distributed discovery | Local `GlobalIndex`, bounded scoped owner streams, explicit coverage and independently accepted same-host live-store correctness; no directory RPCs | 5% isolation and 16-owner performance remediation, plus separate host-failure domains |
| Remote recovery | Source authorization, TENT READ, release reconciliation, peer DRAM/SSD | Permanent requester loss and transfer/partition fault qualification |
| Engine support | Official vLLM 0.31.0 and SGLang 0.5.21 ordinary-cache installed/model gates pass locally; historical release cohorts remain unchanged | Independent acceptance, public lifecycle contracts, multi-instance, container, P/D and TP/PP cells |
| Cost decisions | Resource-scoped observations, shadow estimates and guarded experimental peer choices | One consumed planner for complete routes, legal boundaries and P/D authority |
| Hybrid semantics | Declared recovery contracts and compiled page demands | General read-set/retention IR, checkpoint placement and semantic reclamation |
| Packaging | Wheel construction and installed-artifact gates already exist | New final artifact requalification, independent service images and publication |

The current global-index path passes same-host and H20/A100 TCP sharing, restart
and source-SSD gates on both engines. Same-A100 SGLang P/D evidence does not close
its heterogeneous-GPU strict-output gap. Raw-copy and discovery improvements do
not imply an end-to-end win over an engine's resident HBM hit.

The baseline above describes implemented paths, not universal model, topology or
performance guarantees. Keep historical measurements tied to their exact builds.

## S1 — Evidence separation and one truthful work queue

**Scope:** `benches/`, `docs/`, agent skills, `.gitignore`, website content and the
smallest relevant CI check. No performance policy changes.

- Consolidate the former TODO and competing plans into S2–S8 below. Preserve
  missing-code versus qualification status and explicit research deferrals. Remove
  obsolete checklists after migrating their obligations. Do not implement unused
  Python state wrappers, a second replica catalog, a new generic region RPC or
  transport abstraction merely because an old checkbox requests one.
- Archive the tracked `benches/results/` collections before removing them from
  the current tree. An immutable Git revision can retain historical tracked
  evidence; untracked run data needs a verified external archive. Preserve hashes,
  commands, controls and failed runs. Do not rewrite history or delete originals
  before archive verification.
- Move generated output defaults out of the checkout. Use an explicit external
  output directory in benchmark/qualification examples and CI artifacts for CI.
  Keep reusable programs, workload definitions and small deterministic fixtures.
- Replace links into removed result directories with maintained methodology and
  stable evidence links. Keep only concise reviewed conclusions in public docs;
  large tables, per-run reports and generated plots remain external artifacts.
- Add a focused guard against tracking generated experimental output. Validate
  actual tracked files without banning legitimate test fixtures or arbitrary
  source files just because their names contain `result`.

**Acceptance:** tracked generated results are gone; archives are readable; normal
benchmark execution writes outside Git; benchmark tests pass; website build,
search and link tests pass. Every migrated obligation has an owning stage or an
explicit deferred research decision. Do not rename Rust crates to mimic Python
package layout.

## S2 — Metadata reliability and measured capacity

**Depends on:** S1. **Owners:** `orbitkv-server/src/cluster/`, `orbitkv-catalog/`,
core inventory and metadata diagnostics.

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

**Acceptance:** no lost final puts/deletes after convergence, no stale incarnation
  authorization, bounded memory/queues, local cache still usable during metadata
  loss, and a repeatable capacity/recovery report. A fenced runtime restarts with
  a new incarnation; it cannot silently resume using the expired one.

### S2.1 — Quota and index-budget recovery

The real-etcd metadata-owner gate now covers backend quota exhaustion, a stalled
publication cursor, compaction/defragmentation/alarm disarm and resumed publication
without resurrecting a deleted replica. A separate small-index case rejects all
partial results, keeps lease renewal alive beyond its initial conservative
validity window and rebuilds only after eviction permits a complete snapshot.
The existing lost-reply, paginated-snapshot, incarnation, leader-loss and
quorum-loss gates remain part of this substage's validation. The leader-loss
rerun exposed startup failure on an unavailable balanced endpoint after a lease
was granted. Metadata bootstrap now retries transport failures with the same
lease/incarnation under a finite startup budget; format/identity rejection is
terminal. A repeated registration reconciles only its exact committed owner and
lease without advancing the node epoch or stealing another registration.

This scope uses etcd 3.5.21 processes on one container/host and the actual Rust
publisher, Watch, membership and index owners. Synthetic residency records enter
the publisher directly; it does not qualify GPU storage publication, journal
churn through live stores, data-plane availability, cross-host HA or capacity.
Those S2 obligations remain open. The [metadata gate recipe](distributed-cache.md#quota-and-index-budget-recovery)
records commands, external evidence and operational recovery boundaries.

### S2.2 — Watch delay, partition and compaction recovery

A test-owned TCP gate now isolates one Manager's actual etcd connection while the
real etcd service remains writable. It delays Watch responses, severs existing
HTTP/2 streams, permits source publication and compaction through an independent
connection, then heals the Manager connection. The reader retains its last
complete index during the transient partition, hides the index while compacted
history is rebuilt and exposes only the final complete snapshot afterward.

The gate exposed a real classification defect: tonic reports an HTTP/2 Watch-body
reset as gRPC `Unknown`. Treat only `Unknown` messages that identify HTTP/2 or
transport failure as resumable disconnections; other unknown statuses still
force a rebuild. Resume starts at the last applied revision, and etcd compaction
or missing previous metadata then triggers the normal reset/snapshot path. The
[Watch fault recipe](distributed-cache.md#watch-delay-partition-and-compaction)
records the observed lag/rebuild measurements and remaining scope.

This substage uses synthetic publication records and one etcd process on one
host. It does not close sustained inventory-journal overflow, concurrent live
DRAM/SSD publication, local-cache availability through Manager metadata loss,
multi-host partitions or the measured operating envelope.

### S2.3 — Measured metadata scaling smoke

The real-etcd capacity gate registers 1, 4 and 16 source owners with 512 keys per
owner, publishes complete owner states, starts a fresh reader to measure snapshot
rebuild and deletes half of every owner's records. It reports transaction
latency, observed Watch lag, rebuild time, logical index bytes, process CPU ticks
and RSS, and etcd backend growth. A fixed 500 ms delay lets etcd backend-size
statistics settle outside the timed publication and Watch samples.

Three repetitions pass on both the current H20 container and the separate A100
host using identical frozen artifacts. At the largest checked point (16 owners,
8,192 records), the two hosts report publication p50 0.64/0.71 ms, observed Watch
p50 about 2.08 ms, snapshot rebuild medians 56.2/60.1 ms and 2,315,392 logical
index bytes. The 1 ms observer makes Watch values quantized diagnostics rather
than transport-only latency. See the [capacity smoke recipe](distributed-cache.md#measured-metadata-scaling-smoke).

The final correction verifies every retained key, owner/incarnation, sequence and
complete metadata on both the original and rebuilt readers. Codex independently
reran that frozen gate and accepted S2.3 at `680eaa1a`; the earlier sandbox bind
failure is retained separately as an environmental control.

This is a tested smoke envelope, not a maximum: one etcd process per run,
synthetic Publisher records, no inference load and at most 16 registered sources.
Larger node/key/update loads, sustained live-store journal churn, concurrent
Managers, three-host etcd and failure-domain capacity remain open.

### S2.4 — Bounded live-DRAM journal and metadata-loss correctness

An A100 gate now sends real GPU blocks through the production Engine DRAM store,
evicts them and saves replacements while a test-owned TCP gate disconnects the
source Cluster from real etcd. A 1 KiB inventory journal advances from sequence
64 to 192 and explicitly reports a history gap. The observer retains its last
complete view during the partition; after reconnection the publisher snapshot
removes every old key and publishes every replacement under the exact source
owner. Zeroed GPU memory restores exactly 65,536 local bytes both during the
transient partition and after the source's real lease disappears from an
available observer. Healing cannot revive the expired incarnation.

Codex independently reran the frozen A100 artifact and accepted the final
observer-availability barrier at `f7d953a8`. This is one bounded live-DRAM
correctness scenario using Engine and Cluster owners in one test process. It is
not sustained churn or a capacity envelope, and does not qualify a complete
Manager process, SSD, cross-host cache, P/D, RDMA, native GDS or three-host etcd.
See the [live metadata-loss recipe](distributed-cache.md#live-dram-journal-overflow-and-metadata-loss).

### S2.5 — Full-Manager DRAM and io_uring metadata-fault correctness

The Python distributed gate now runs an independent client process against two
real Cache Manager processes and one real etcd process. DRAM and explicit
io_uring SSD cases each cover a no-fault remote restore, three rounds of source
publication/eviction/re-save while only the source's etcd transport is severed,
snapshot recovery after healing, and a second partition through actual lease
expiry. Every round and block uses a distinct deterministic payload; every
restore zeroes the GPU destination and compares all bytes.

The test reads the isolated etcd namespace as an independent oracle, validates
the exact source incarnation and location-key digest, and requires the complete
set of retained hashes under the expected DRAM or SSD medium. Deleted hashes
must be absent. For SSD, each save waits for io_uring completion, evicts DRAM and
requires the SSD-read counter to advance before accepting restored bytes. Before
testing the expired source from the second Manager, its own DRAM is evicted so a
local hit cannot mask remote rejection. Healing cannot re-register the old
source incarnation.

The implementation run passes both media on one A100 host with frozen Manager,
wheel, extension, TENT and etcd artifacts. It covers five eight-block rounds per
medium and forces repeated 1 KiB journal overflow through `e4cfc810`. Codex
independently reran both media and accepted the complete S2.5 delivery at
`f3a44ce1`.
This is same-host multi-process correctness, not cross-host HA, native GDS, a
large sustained-capacity envelope or an engine-serving qualification. See the
[Manager process recipe](distributed-cache.md#manager-process-dram-and-io_uring-metadata-faults).

### S2.6 — Sustained metadata churn and capacity

**Independently accepted at `ab306965`.** A 60-second ignored release gate uses
16 registered production Publishers and real etcd to rotate 24,576 active DRAM/
SSD metadata records through 245,760 changes. It starts fresh complete-index
readers during eviction rounds, checks every final key, source incarnation,
sequence and medium, and measures publication latency, Watch lag, rebuild time,
logical index bytes, bounded publication batches, CPU/RSS and etcd growth against
thresholds committed before execution. Manager metadata status also exposes the
live residency count and retained journal current/peak bytes, capacity and
history-gap count; the accepted S2.5 gate consumes these fields on the actual
storage path.

The committed workload contract and command are in the
[sustained-capacity recipe](distributed-cache.md#sustained-metadata-churn-and-capacity).
The frozen candidate at `85f0f656` passes once locally and three times on the
A100 host. All A100 runs sustain 4,032 changes/second; publication p95 is
1.02–1.04 ms, Watch p95 is 3.46–3.47 ms, fresh-reader snapshots finish within
374 ms, and target convergence finishes within 454 ms. The exact final 24,576
records, 7,015,680-byte index peak and every predeclared resource bound pass.
The same frozen Manager/wheel also reruns the S2.5 DRAM/io_uring gate successfully
with the new journal diagnostics. Codex independently repeated the capacity gate
at 4,032.52 changes/second and both Manager media cases in 38.34 seconds, verified
the frozen evidence and accepted the substage with no blocking findings. This
one-host/one-etcd-member gate cannot close three-host or independent-failure-domain
qualification.

### S2.7 — Bounded coalescing and frozen comparison baseline

**Independently accepted at `afa72863`. Depends on:** independently accepted S2.6
implementation and its available frozen gates. Missing cross-host hardware does
not block this local protocol-preserving optimization. Follow the
[publication contract](distributed-design.md#publication-and-coalescing).

- Coalesce a bounded contiguous inventory interval before forming publication
  transactions. Preserve every final key/medium generation, final deletes,
  cursor CAS and lost-reply reconciliation. A flush bypasses intentional wait;
  the input cursor advances only after the entire covered interval commits.
- Measure original/coalesced mutations, transaction and Watch bytes, intentional
  delay, actual visibility, CPU/RSS and local save/query interference. Freeze
  zero-window and coalesced artifacts for later A/B comparisons.
- Reuse the accepted exact Manager DRAM/io_uring gate and sustained capacity
  workload; add deterministic cases only for new interval/barrier contracts.

**Acceptance:** final puts/deletes and generations match the oracle; no early
barrier completion; bounded buffers; demonstrated mutation reduction on repeated
churn without hiding freshness or hit-rate regressions. This does not implement
the inventory-stream protocol or qualify a larger deployment.

The S2.7 candidate reads bounded contiguous intervals in `ResidencyInventory`,
keeps the newest record per key/medium, and holds the fully covered input cursor
until the final etcd transaction. Flush interrupts the wait. The explicit
[coalescing benchmark](distributed-cache.md#bounded-etcd-publication-coalescing)
compares the frozen S2.6 binary with candidate 0/2/5 ms profiles. All 20 rotated
A100 runs and the selected-default S2.5/S2.6 regressions pass the predeclared
limits. The default remains 0; 2 ms is a qualified opt-in tradeoff. Evidence is
frozen outside the checkout. Codex independently checked all 20 runs and 4,200
raw samples, reran the real-etcd cursor and 60-second capacity gates plus both
full-Manager media cases, verified frozen hashes and accepted the same-host
substage without blocking findings. Cross-host and serving cells remain open.

### S2.8 — Owner inventory streams and coordinated protocol cutover

**Independently accepted at `ad5bb8e6`. Depends on:** accepted S2.7. Implements the
[background protocol](distributed-design.md#background-inventory-protocol),
[barrier replacement](distributed-design.md#synchronization-api-cutover) and
[cutover contract](distributed-design.md#cutover-rollback-and-handoff).

- Keep etcd membership/epochs and one shared format guard. Add bounded Manager
  inventory sessions on the existing peer listener, initially for all namespaces.
  Retain source-authoritative inventory and the existing `GlobalIndex` owner.
- Implement per-owner snapshot scan plus complete journal replay, atomic install,
  contiguous watermarks, duplicate/reconnect repair, multi-subscriber wakeups,
  bounded ACK credit and session/snapshot limits. Separate inventory degradation
  from membership fencing. Expose coverage instead of partial-as-complete results.
- Replace `published_revision` with the source inventory fence and bounded
  requester await endpoint; update all actual consumers. Keep regular discovery
  free of directory/coordinator RPCs and preserve source grants and drain.
- Cut over with frozen rollback artifacts and an offline format migration;
  remove block publication/Watch decoding and the old barrier together. Do not
  ship two production metadata authorities or mixed-version fallback.
- Run the exact protocol fault matrix, real-etcd/Manager byte gates and both
  engines' shared-cache gates on available GPU hardware. Use independent expected
  mutations/payloads as the oracle; etcd no longer contains block records.

**Acceptance:** all-domain correctness matches the frozen baseline; no block or
publisher-cursor mutations reach etcd; metadata pressure stays bounded and local
service progresses; old protocols/incarnations cannot join or authorize; the
coordinated rollback works. Same-host acceptance leaves cross-host cells open.

The S2.8 candidate removes the etcd block publisher and block Watch decoder,
keeps the common format/epoch/member namespace, and rejects the frozen format.
The existing peer listener now carries bounded all-namespace snapshot/delta
sessions into per-owner hidden staging in `GlobalIndex`. Page count, transcript,
session/view identity, contiguous intervals, generations and membership gate the
atomic install. Frame credit, sessions, concurrent snapshots, aggregate queued
bytes, pacing and fence waiters are bounded. `POST /cache/sync` returns a source
inventory fence; `/cache/metadata/await` accepts only an installed matching owner
view. The offline migration tool performs a dry-run/archive and exact CAS in
both directions without restoring live leases or old cursors.

The first independent review at `6f3d37c7` found that a mismatched view ID was
checked after installation, replay did not close the snapshot-page phase, a
replacement that fit only after withdrawing its old view could retry forever,
idle reconnect did not restore freshness, standalone sync returned 409, and the
Manager gate stopped before the real lease key expired. Candidate `46f23928`
validates view/page phase before commit, performs bounded owner withdrawal for a
budgeted rebuild, refreshes only equal installed progress, retains a local-only
sync result, waits for exact lease removal, aborts abandoned source input tasks,
and bounds aggregate-credit and response-queue waits. The rejected report,
environment failures and corrected probes remain with the evidence.

The second independent review at `a25169af` found a budget-repair race: an etcd
membership progress revision could clear the retired state while the old owner
view was being removed in bounded batches. The partially cleaned view could then
resume at its old cursor and report complete coverage with missing rows. Commit
`344ef6c9` makes withdrawal monotonic across membership refresh; only a complete
replacement snapshot can make that still-live owner active again. The review's
120,000-record stress case now converges to all 110,000 final rows after 172
membership refreshes, with both reconnects using full bootstrap.

Frozen A100 evidence at native `344ef6c9` and harness `53314d52` passes the
real-etcd stream reset/overflow gate, membership/leader/quorum gates, a 60-second
16-owner all-to-all run, full-Manager DRAM/io_uring faults, live DRAM overflow and
the 260-block TCP P2P restore. The capacity run sustains 245,760 changes at
4,083.12 changes/s with 29.53 ms visibility p99, a 7,929,600-byte index,
697,538-byte queue peak and zero etcd DB growth during churn. Five-run 0/2 ms
Manager comparisons retain exact GPU restores and zero etcd revisions; 2 ms
reduces median stream bytes by 83.57% while visibility p95 rises from 1.68 to
5.17 ms. Default waiting remains 0. Scope filtering, physical cross-host and
serving gates remain open. Codex independently reran the frozen runtime gates and
the 120,000-record withdrawal/membership-refresh race, then verified that the
targeted regression passes on the fix and fails after restoring the historical
unsafe assignment. It accepted the complete S2.8 delivery at `ad5bb8e6`.

### S2.9 — Scoped discovery and explicit coverage

**Independently accepted at `0b97be08`. Depends on:** accepted S2.8. Implements
the [subscription contract](distributed-design.md#subscription-coverage-and-local-query-semantics).

- Add exact namespace allowlists and canonical scope identity; initially use
  explicit configuration, not query-triggered subscription. Preserve model,
  representation and group identity. Scope changes require a new bootstrap.
- Consume complete-at-watermarks, partial-hints and unavailable states in local
  discovery/diagnostics. Empty filtered intervals may advance a cursor; missing
  owner views cannot prove absence. Member changes update coverage incrementally.
- Account active/staging/reverse-index memory; invalidate owners immediately and
  clean their rows in bounded batches. Measure lookup lock time under churn.
- Compare all-domain and scoped streams against the same in-scope key/payload
  oracle. Report remaining peer-session and all-to-all fanout costs explicitly.

**Acceptance:** no lost in-scope candidates or cross-scope matches, no false
negative-completeness claim, lower matching-workload metadata bytes, and bounded
repair without a directory RPC. Filtering is not tenant authentication.

Production candidate `272803cf` and frozen harness `fddf6c14` implement protocol
v4, canonical all/exact/empty scope identity, source-side snapshot/replay/delta
filtering, scope-bound resume/await and incrementally maintained coverage. Exact
allowlists use complete generated storage namespaces, allow at most 256 entries
and a 64 KiB Open; scope changes restart/bootstrap rather than reuse old views.
Real-etcd and full-Manager DRAM/io_uring gates cover empty/filtered views,
reconnect/gap/restart, membership/incarnation transitions, budget withdrawal and
scope-outside local service with exact GPU bytes.

The frozen same-host A100 forced-TCP matrix runs five independent matched pairs
with 10/10 exit zero. Scoped median stream bytes are 61,616 versus 236,683
all-domain (ratio 0.2603), with maximum async/barrier visibility p99 of
26.714/6.815 ms and scoped/all median bootstrap/repair ratios of 0.9921/1.0044.
Stable churn has zero history gaps; the isolated repair phase records a real gap
and reset; block churn adds zero etcd block/cursor revisions. Scope filtering
retains one peer session and makes no serving, TTFT/ITL, tenant-isolation or
cross-host claim. Raw/failed controls and frozen hashes remain outside the
checkout.

The independent reviewer accepted S2.9 without blocking findings after checking
the complete `3f3b71a3..0b97be08` diff, rerunning the state/catalog unit suites,
verifying all 104 matrix-manifest entries and ten zero exit files, matching the
source archive to production commit `272803cf`, and confirming frozen-process,
socket and GPU cleanup. Acceptance remains limited to two full Managers on one
A100 host with forced TCP. Physical cross-host failure domains, three-host etcd
HA, engine serving, P/D and parallelism profiles, RDMA, native GDS and S2.10's
long live-store cells remain open. The signed-off external report is retained
with the frozen S2.9 handoff; no serving or tenant-isolation claim is inferred.

### S2.10 — Sustained live-store and independent-domain qualification

**Same-host correctness independently accepted at `65c51aaa`; qualification
remains partial because the 16-owner ordinary visibility gate now passes but the
frozen 5% DRAM/SSD isolation gates fail. Depends on:** accepted protocol implementation from S2.8 and S2.9 for
scoped claims. This carries forward S2's remaining performance, capacity, serving
and cross-host obligations; it does not replace missing evidence with a new name.

**Current next delivery:** independently review the task-004 installed-runtime
readiness, two valid A100 tool smokes and the still-disabled eight-cell observer
launch inputs described below. No formal cell starts until that exact freeze and
launch contract are accepted. The completed diagnosis does not authorize a
production repair, cannot replace the accepted failed cohort and cannot start
another formal qualification automatically.
Physical cross-host cache/HA, independent etcd failure domains and native GDS stay
qualification blocked on the hardware listed below. The bounded
fixture pacing/invalid-run cleanup repair has independent acceptance. Merged-artifact
regression and the corrected measurement/oracle contracts are complete; the
formal runner and release pressure-source profile have independent Codex launch
acceptance. Keep the old 51.72 ms serial-barrier and all/scoped isolation failures
as historical failures. Supported capacity remains four owners until replacement
profile evidence and its independent review pass.

| Remaining S2 obligation | Current state | Next independently reviewed evidence |
| --- | --- | --- |
| Same-host live-store correctness and scoped stream oracle | PASS at the retained accepted revisions | Inherit unless a production owner affecting these contracts changes |
| 16-owner ordinary visibility | PASS for five 4.281–4.309 ms runs, but capacity support remains UNQUALIFIED | Replacement isolation must also pass before expanding the four-owner boundary |
| DRAM/SSD isolation | FAIL in the immutable 25-cell cohort | Installed-runtime readiness, observer A/B diagnosis, reviewed harness correction if supported, then one new 25-cell qualification cohort |
| Observer diagnostic launcher | TOOL SMOKES PASS, INDEPENDENT REVIEW OPEN | Review the task-004 tests-only projection, real tensor/SSD readiness, A/B protocol and disabled formal launch inputs |
| Three-host direct network | READY: six directed IPv6 TCP edges pass | Exact etcd client/peer listeners, then OrbitKV-consumed leader/quorum/process-fault campaign |
| Independent host/power/network failure domains | QUALIFICATION BLOCKED | Infrastructure ownership/topology evidence and authorized host/network fault controls |
| H20/A100 physical cache data plane | QUALIFICATION BLOCKED | H20 CUDA compute/IPC readiness plus bidirectional native TENT IPv6 byte transfer |
| Final serving profile | QUALIFICATION OPEN | Consume an independently accepted S2 runtime and existing accepted release profiles; do not inherit unreviewed S5 artifacts |
| Permanent requester loss/inflight DMA reclamation | S3 DEPENDENCY OPEN | Native termination and generation-safe lifetime proof; lease/TTL is not drain |
| RDMA/native GDS | QUALIFICATION BLOCKED | Real NIC/transport counters and supported NVMe/filesystem/nvidia-fs/cuFile device access |

- Report historical serial barrier verification separately from bounded
  concurrent verification, ordinary asynchronous publication-to-install latency,
  and save-start-to-discovery latency. HTTP completion is not an observer install
  timestamp, and ordinary asynchronous samples must not force a sync/flush.
- Bind source publication and observer installation observations to owner,
  incarnation, epoch, scope digest and sequence in one clock domain. Preserve
  bounded owner/session state and exact per-owner barrier checks; do not add a
  directory RPC or second metadata owner.
- Keep all-domain versus scoped as the filtering/byte comparison. Measure local
  isolation with a fixed scope and matched quiet versus metadata-pressure runs.
  Freeze warm-up, at least 1,000 measured rounds per run, five balanced matched
  repetitions, cadence, resource settings, expansion budget and stop rules before
  the formal experiment. The 50 ms visibility and 5% isolation targets do not
  change.

The merged-main integration candidate at `ceb6125a` has completed the A100
installed-wheel/TENT byte, real-etcd, strict restart, four DRAM/SSD all/scoped and
both official-engine shared-cache smoke gates. This is the same-host smoke
profile only; a no-GPU build-host TCP byte-test failure remains recorded. The
measurement-v2 implementation adds bounded publication/install timestamps,
exact-sequence observation, separate serial/concurrent barriers, fixed-scope
quiet/metadata-pressure controls and independent-pair statistics. It is
**implemented, formal measurement readiness pending independent review**.
The independent report `orbitkv-s210-measurement-review-20261003.md` accepts the
bounded timestamp contract and nine capacity smoke cells at `6b0e4fb9`, while
rejecting formal readiness on pressure exposure, sample guards and namespace
configuration. The first frozen short experiment
at `6b0e4fb9` passes all nine 1/4/16-owner endpoint cells with real remote bytes,
but has only 20 samples/cell. Its isolation startup rejects an invalid synthetic
namespace; the failure is preserved and the next harness uses actual registered
storage namespaces. Review also requires independent sustained pressure cadence,
explicit exposure checks and guarded warm-up/sample counts before formal freeze; no new
visibility/isolation qualification is claimed. Evidence is under
`/root/orbitkv-artifacts/s2-s51-20260930/s2-10-performance-qualification-20261002/`.
The next dependent action is to review that implementation and its short-run
contract before freezing the formal workload. Historical failure evidence and
the four-owner supported envelope remain unchanged. Follow-up review at
`9c1731cd` accepts the guarded sample counts, actual namespace derivation and
DRAM pressure smoke. It still blocks formal readiness on publication-clock
bounds and an SSD-only duplicated metrics flag; both failed pilots and separate
review reports remain in the external directory. The next short candidate
repairs those harness boundaries before any formal performance run. Independent
review at `0e9dae0f` closes those blockers and accepts all seven short cells,
including both io_uring conditions. The final formal freeze additionally requires
the agreed generation/block payload header and an untimed final restore for
every capacity owner; these strengthen the byte oracle without changing the
50 ms/5% thresholds. The elevated pilot isolation ratios remain unqualified.

The final harness is `d18a4a3b` with production Manager/wheel code at `6b0e4fb9`.
The metadata-only pressure fixture is now built with the release profile; its
SHA256 is `f34c73078b5c6f8d39ba65f3d70f6553a3b49dd9ceefb098df4341ee22e7a81f`.
All four affected release-source isolation pilots pass, preserving the earlier
dev-profile pilots and failures. Independent Codex review accepts the exact final
oracles and formal controller, including immediate cadence refusal, bounded
non-killing helper/cell watchdogs and rejection of optimized Python execution.
The review and its intermediate blocker reports are archived in the external
`measurement-v2-20261003/codex-final-review/` directory.

The formal input manifest is
`49a51f66f1619324ceefe9884907c34e91d5101da3a8f1dfb1a15130e19b1e86`, under
`measurement-v2-20261003/formal-inputs/`. It fixes five 16-owner ordinary runs and
five balanced quiet/pressure pairs per DRAM/io_uring medium: 50 warm-up and 1,000
measured rounds per run, 1 Hz foreground, 17 ms source bursts, 25 ms symmetric
observer sampling, unchanged budgets and 50 ms/5% thresholds, with no expansion.
The nominal foreground duration is 26,250 seconds; the campaign budget is nine
hours. The detached A100 runner writes live state and evidence below
`candidate-d18a4a3b-release-pressure/formal-qualification-1/`. That campaign
stopped invalid after its second cell. The first 16-owner run completed 50 warm-up
and 1,000 measured rounds with publication-to-install p99 4.295433 ms, exact
final owner bytes and zero etcd revision growth; one run does not qualify the
five-run target. The DRAM quiet run then recorded a 0.205849 ms catch-up interval
after 21.851274 ms scheduling lateness, violating the frozen 8.5 ms minimum.
The remaining 23 cells were not started. The invalid run and its CUDA IPC cleanup
warning remain archived; no old result is reclassified or replaced.

The accepted bounded repair at `fbcfd130` paces the existing source fixture against
both its original 17 ms schedule and the prior completed iteration plus 8.5 ms. Every
mutation, budget and threshold remains unchanged. Terminal measurement rejection
must still exit nonzero after explicit drain/unregister and graceful Manager
exit, with invalid result and cleanup evidence saved. All four 38 ms injected
stall controls pass; the 200 ms control remains invalid/exit 1 with explicit
drain and Manager exit 0, without CUDA IPC producer warnings. Independent Codex
review accepts this repair and separately accepts the new launch inputs.

The new input manifest is
`92312221ad5f0177cab3d32fa9508991d112ba97b85196e8991a5a5331808318`, under
`measurement-v2-20261003/pacing-repair-20261003/formal-inputs/`. The release
fixture SHA256 is `7f32c49ff33a11f89119f4205da0b70f7705ff29da91bfde36c59b5b6f6c8c33`.
The complete fresh 25-cell matrix ran at
`pacing-repair-20261003/candidate-fbcfd130/formal-qualification-2/`; its source
stall variables are forbidden and result guards verify they remain disabled.
The old successful capacity cell is not pooled into the new five-run cohort.
This is one bounded restart: no third automatic campaign or threshold/resource
tuning follows its performance failures. All 25 cells are valid, exit zero and
complete normal cleanup. The five 16-owner ordinary publication-to-install runs
have p99 4.281–4.309 ms and each passes 50 ms; install-to-harness p99 remains a
separate 55.910–56.605 ms observation and is not production visibility. Both
DRAM and SSD isolation fail because at least one pair exceeds 1.05; DRAM query
and both SSD primary-metric CI upper bounds also exceed 1.05. The complete raw
archive, independent recomputation and offline association audit are under
`pacing-repair-20261003/`. Independent final review accepts correctness,
collection validity, cleanup and the ordinary visibility result, while confirming
both isolation failures and the unqualified 16-owner envelope. Its report is under
`pacing-repair-20261003/codex-final-qualification-review/`; isolation and expanded
capacity remain unqualified.

### S2.10a — Same-host isolation root-cause diagnosis

**Authorized from accepted handoff `c5cf2782`; diagnosis and any evidence-backed
minimal fix are open.** Preserve the accepted 25-cell conclusions: collection and
same-host correctness pass, 16-owner ordinary visibility passes, both DRAM and SSD
isolation fail, and the supported envelope remains four owners. The old invalid
campaign and the accepted failed cohort are immutable inputs, not replaceable
cells in this substage.

First audit the existing synchronous Publish path and diagnostics. Decompose one
bounded operation identifier across client submit, process-channel queueing,
Manager receipt, runtime start, engine completion, response notification, client
return and deferred storage publication/SSD completion. Use one host monotonic
clock, bounded opt-in sampling and symmetric quiet/pressure observation. Report
queueing, execution and completion wait separately; `CacheManagerClient.save`
currently includes channel and Manager/GPU-copy work but not deferred insert/SSD
durability. Reuse existing timeline, storage metrics and owner state rather than
adding a coordinator, directory RPC or second Catalog.

Predeclare a short overhead/correlation pilot, then freeze an independently
reviewed diagnostic matrix with at least 1,000 measured requests per p99 run,
balanced ABBA/BAAB order and at most two single-factor ablations. Prefer SSD while
retaining a DRAM regression control. Keep foreground payload, namespace, medium,
residency, hit path, budgets and pressure volume fixed. A diagnostic affinity or
sampling condition is not a deployment qualification. Stop at the frozen budget;
retain invalid runs and do not expand samples or tune after viewing results.

The first three-cell instrumentation pilot at `183fb67` is retained as a failed
overhead control. All cells complete exact SSD restores, required stage linkage,
normal Manager/source exits and zero final resource drain metrics; both enabled
cells record 1,560 correlated events without reaching the 16,384-event limit.
However, enabled/off quiet p50 ratios are 1.347 for save and 1.427 for query,
above the frozen 1.05 limit. The p99 absolute and CPU-tick guards pass, but do not
override either p50 failure. Synchronous JSON timeline logging is therefore being
removed from the measured request threads while retaining the same bounded event
contract. No formal diagnostic matrix may start until the repaired observation
path repeats the entire pilot and receives independent acceptance. Evidence is
external under `s2-10-isolation-diagnosis-20261003/pilot/`; it is not pooled with
the accepted isolation cohort.

The first output-only repair at `f88a3954` also retains a complete matched pilot
under `repair-f88a3954/`. Moving log output to a bounded writer reduces the save
p50 ratio to 1.124 but leaves query at 1.340, so that repair also fails the same
frozen guard. Required stage payloads and query-path transfer records are now
captured as fixed-size queue entries; JSON construction moves to the writer as
well. This observation-only change must repeat the three cells under another
fresh artifact freeze. Neither failed pilot authorizes an isolation fix or a
formal diagnostic launch.

The fixed-record candidate at `14310a67` completes that final bounded repeat.
Correctness, linkage, event bounds, source/Manager exits and resource drain pass,
but save p50 remains 1.105x and query p99 increases by 0.494 ms; the mandatory
limits are 1.05x and 0.25 ms. Its complete evidence is under
`fixed-14310a67/`. S2.10a therefore stops before the formal ABBA/BAAB matrix:
observation overhead remains unqualified, the accepted DRAM/SSD isolation
failures remain unchanged, and no isolation production-path fix is made without
causal evidence. A later substage must predeclare and independently review a new
lower-overhead observation contract before collecting another formal cohort.

Independent review of the stopped stage required five observation-correctness
repairs before handoff: explicit insert/SSD drain before writer flush, bounded
failure behavior when the writer thread cannot start, a before-publication
response boundary, dequeue timing before insert assembly, and completion tracking
for intervening non-diagnostic SSD batches. These repairs change neither the
failed overhead decision nor the isolation qualification boundary; they require
CPU/unit review gates and a fresh artifact freeze, but do not authorize another
pilot or formal campaign in this substage.

The focused rereview additionally required the endpoint owner to fence detached
Publish continuations between channel stop and storage flush; GPU drain alone can
wake a save future before that future assembles and enqueues its deferred batch.
The endpoint now counts admitted Publish tasks and awaits the last continuation
before lifecycle/storage/timeline drain. This correction is unit-qualified and
refrozen only; the stopped pilot cohort is not restarted.

### S2.10b — Lifecycle runtime acceptance and lower-overhead diagnosis

**Separately authorized after the S2.10a stop decision.** Keep all three failed
instrumentation pilots immutable. First run the final review-fixed artifact on
the A100 and prove that an admitted real Publish, deferred insert/SSD work,
response publication, client completion and normal Manager shutdown drain in
owner order without a native child or resource leak. This runtime gate does not
reclassify isolation.

The A100 lifecycle gate is accepted for production tree
`6d120c37dc3a4e486ebac706d2ff4426b74ff1b1`, shared by candidate `54b5b161`
and the final conservative `00f0dc9c` revert. A SIGTERM issued after Manager
processing began and before its completion drained a 2 GiB/2,048-page Publish,
deferred insert and io_uring SSD completion in 1.909 seconds; all ten required
stages were present, the Manager exited zero, its PID disappeared and the client
reported the permitted `peer_exit_after_drain` terminal result. This proves
`completion or peer exit` while the client retains its source until actual
Manager exit; it does not claim cancellation. The immutable evidence root is
`/workspace/orbitkv-three-host-20260930/s2-10-isolation-diagnosis-20261003`
on `orbitkv-a100`; the result is
`s2-10b-candidate-54b5b161/run-3/result.json` (SHA-256
`d960a8fe77a5896c0a6cee07b72944b70ddf6fa94c4d491769de871ba72fa2da`).
The normal-return control and every earlier failed shutdown attempt remain beside
that cohort and are not reclassified.

A focused post-handoff correction now makes both shutdown boundaries durable.
Publish close and active-count registration share one atomic state, so work
registered before close drains and a prechecked late request cannot enter after
the boundary. Lifecycle connections use a latched close state rather than a
one-shot notification: partial frames close without dispatch, while a complete
request finishes dispatch and its bounded response before the idle connection
and session owners drain. CPU race regressions, inverse mutation controls and
real A100 endpoint/session regressions cover both boundaries. This correction is
implemented pending independent Codex review and does not reopen or reclassify
the isolation cohort, observation result or four-owner support limit.

Replace per-event channel delivery with a preallocated, process-bounded in-memory
recording path while preserving the same authenticated operation key, monotonic
clock and stage meanings. No request thread may format JSON, block on a writer or
allocate an unbounded buffer. Repeat the three-cell SSD overhead/correlation
pilot as a new cohort with the existing payload, budgets, 10 warm-up, 120 measured,
250 ms foreground cadence, 17 ms source cadence, 25 ms observer cadence and the
same 1.05 p50, 0.25 ms p99 and 1.15 CPU guards. Preserve every failure and stop
again if any mandatory guard fails.

The bounded-ring candidate is frozen at `0dd472d3` (tree `f9f636f5`). Its fresh
A100 pilot passes every guard: save p50 is 0.1963155 ms off and 0.2017400 ms on
(1.027632x), query p50 is 0.1084830/0.0998240 ms (0.920181x), save p99 increases
0.010287 ms, query p99 decreases 0.026292 ms and CPU ticks are 174/173
(0.994253x). Both enabled cells contain 1,560 required events with complete
authenticated linkage and no limit/truncation record; all cells restore exact
bytes from io_uring, exit normally and end with zero SSD/query residency. The
independent evidence review accepts only the observation-overhead gate. Evidence
is under `s2-10b-ring-0dd472d3/` at the external A100 root above; its frozen,
harness and pilot manifest hashes are respectively `9c026d99`, `2efab6d4` and
`496046e5`.

The independently reviewed formal diagnosis used four SSD pairs in
`QPPQPQQP` order, two smaller DRAM pairs in `QPPQ` order and one symmetric SSD
observer-cadence pair changing only 25 ms to 75 ms. Every run has 50 warm-up plus
1,000 measured operations at 1 Hz, a 32,768-event limit, a 1,500-second watchdog
and the unchanged pressure/payload/storage contract. The first controller attempt
failed its runtime-probe syntax preflight before launching any measurement and is
preserved as `formal-diagnosis-1`; corrected `formal-diagnosis-2` is the only
accepted diagnostic cohort. Its input manifest is `06bc29c1`. All 14 serial cells
completed with 1,000 measured operations, exit zero, valid oracles and empty
per-cell and campaign postflight. Runtime dependencies and every per-cell frozen
hash match. The complete pre-review evidence archive/freeze is
`s2-10-isolation-diagnosis-20261003/s2-10b-ring-0dd472d3/`; its 5,525-entry
manifest SHA-256 is `7e97bf4e`. These diagnosis cells cannot replace the failed
qualification cohort or change the four-owner support boundary.

The SSD baseline does not show a stable metadata-pressure effect. Save p99
pressure/quiet ratios are 0.321, 1.179, 1.079 and 1.073 with a four-pair geometric
mean of 0.814 and bootstrap 95% CI `[0.435, 1.152]`. Query ratios are 2.549,
0.971, 1.016 and 2.397 with geometric mean 1.567 and 95% CI `[0.993, 2.472]`:
both quiet-to-pressure pairs regress while both reverse-order pairs are near one.
DRAM query ratios are 1.0015 and 1.0065, and DRAM save ratios are 0.8847 and
0.9060. Metadata pressure increases aggregate CPU, context switches and update
lock hold time, but total update-lock wait is only 1.84--2.17 ms per roughly
1,000-second pressure run; native Manager stages do not reproduce the endpoint
shift. SSD execute p99 varies from about 1 to 65 ms with inconsistent pair
direction, zero sampled queue/inflight at dequeue/completion and negligible
correlation to the synchronous save endpoint.

The dominant endpoint tail is outside the production channel: across all 14
runs, `local_query_ms` correlates 0.981--0.997 with the interval after the native
return timestamp and before the Python outer timer completes. All 140 per-run
top-ten slow queries and all 154 independently selected p99-tail samples overlap
an in-process observer poll; the strongest correlation with any recorded native
query stage is only 0.242. The 75 ms pair reduces observer work and gives
save/query ratios 1.0247/1.0263, but it is one final pressure-to-quiet pair and is
confounded by observer phase and run order. Independent review therefore accepts
the diagnostic archive and **blocks any production repair**. The result is
consistent with diagnostic Python observer/return-boundary interference, not
evidence of a Manager, metadata-lock, completion-notification or SSD-owner defect.

The authorized follow-up tool substage is a fixed-phase, SSD pressure-only ABBA
crossover comparing the current 25 ms
in-process `_owner_status` observer with the identical polling workload in a
separate helper process. Keep 50 warm-up plus 1,000 measured operations, 1 Hz
foreground, 17 ms pressure, payload, budgets, correctness and drain unchanged.
It must show that externalizing the observer collapses
`returned_mono_ns`-to-`query_end_mono_ns` p99 and slow-sample overlap without
changing native channel stages or pressure exposure. A positive result would
authorize a benchmark-harness repair only, followed by a separately frozen
isolation qualification; it would not authorize a production-path change.
The helper is a bounded benchmark child with explicit
ready/start/complete/stop/exit/error records; it imports no GPU/native runtime,
owns no cache client or tensor and shares the same absolute monotonic slots,
HTTP request/parsing and finite JSONL recording path as the in-process condition.
Any helper failure, sample-count/digest mismatch or phase/cadence violation makes
the cell invalid. Only one 10-warm-up/20-measured smoke per condition may run
before independent review; those smokes establish tool validity only.

The tool candidate is `8be53548`. Independent CPU-host validation passes Ruff,
all 221 benchmark tests, Python compilation and the tracked experiment-output
guard. The first A100 A smoke is preserved as `INVALID_SMOKE`: the accepted
task-001 Manager reached CUDA and SSD initialization but could not load TENT
because `libyaml-cpp.so.0.7` was absent from that reused frozen native bundle and
the host loader path. It stopped before observer start and before foreground
round zero; no samples or performance result exist. Postflight shows zero GPU
memory, no owned process/socket and unchanged runtime hashes. Per the frozen
fail-stop rule, B was not launched, the environment was not repaired and no
formal cell started. This substage is therefore handed off incomplete for
independent review; a corrected runtime bundle and a newly authorized smoke are
required before the frozen eight-cell matrix can be considered for launch.

Task-003 preserves that failed run and closes its two review findings in a new
runtime bundle. The actual task-001 wheel SHA is the 64-character
`2aa0bcfc8ffd67586259b470fb2f32bc4e6e2c5c56a73a37cf3e4d9fb680a091`.
The bundle adds the matching `libyaml-cpp.so.0.7` from the existing Mooncake
`71973589` release/CUDA-13 freeze without changing the reviewed Manager,
server-tests, wheel, extension or TENT bytes. Malformed/mismatched hash controls,
the complete ELF closure, real TENT `dlopen`, installed native import plus CUDA
compute, and a Manager metadata startup/mapping/normal-SIGTERM preflight all
pass within their watchdogs.

The newly authorized A smoke then stops with numeric exit 1 before observer
start and foreground round zero: the namespace-probe Manager's test launcher
prepends the checkout `python/orbitkv` package, which has no extension, ahead of
the validated installed wheel. Its first GPU-tensor registration therefore
reports that the Rust extension is unavailable. Cleanup is normal with no
forced signal, GPU allocation, owned process or socket left. Per the task-003
fail-stop rule, B is not launched, no cell is replaced and the formal matrix
remains unstarted. A later task must correct the controlled Manager Python path
and extend readiness through real tensor registration before authorizing another
fresh two-smoke attempt; the metadata-only Manager preflight was insufficient.

Task-004 implements that explicit installed-runtime boundary at `441bdcc3` and
updates the website audit lock at `a6e4efba`. `CacheManagerProcess` retains its
source-first default for normal tests, but an installed-runtime caller supplies
an exact site-packages list that ignores inherited source paths. The isolation
harness applies it to both namespace-probe and foreground Managers and records
client, extension, Manager and helper Python-path evidence. A tests-only
projection contains `benches/`, `tests/` and pytest configuration but no
`python/orbitkv` package. CI run `37568176894` passes all 13 jobs, including both
CUDA cargo checks, both Python 3.14 wheels and Clippy; a bounded cache probe can
fall back to direct rustc without masking compiler failure. The earlier sccache
503 and CUDA repository mirror-sync failures remain archived.

The first task-004 freeze remains `INVALID_READINESS`: real namespace-probe GPU
tensor registration succeeds, then the selected task-001 server-tests executable
runs zero pressure tests because it lacks the named fixture. No observer smoke
starts and cleanup is normal. The second and final authorized freeze changes only
that fixture to the independently reviewed `0dd472d3` binary. Eight real
SSD/io_uring readiness rounds then pass installed tensor registration,
save/query/cleared-destination restore, 262,144 SSD-read bytes, 96 linked native
events, byte oracles, unregister and drain. A and B each pass 10 warm-up plus 20
measured rounds, 1,240 exact polls and 1,824 pressure rounds. Their payload/key
oracles, 466,944 mutations, 1,205 stream frames and 31,813,205 stream bytes match;
actual phases are 12.584346/12.603977 ms and the B helper imports no GPU/native
module. All Manager/source/helper exits are zero, no forced cleanup occurs and
the A100 final state is empty. These cells establish tool validity only, not
performance or causality.

The separately frozen formal inputs remain `formal_launch_authorized=false`.
Their manifest is `806a52df`; they preserve A1 B1 B2 A2 B3 A3 A4 B4, four
independent pairs, 50 warm-up plus 1,000 measured rounds, the original workload,
42,040 polls/cell, 61,824 pressure rounds/cell, 1,500-second cell watchdog and
13,000-second global bound. Independent review of task-004 is required before a
new launch task may execute any of those eight cells.

The pilot review and formal diagnostic matrix are complete. Any follow-up must
again use independent runs, at least 1,000 measured operations per p99 run,
balanced ordering, a single predeclared factor, bounded budgets and explicit
stop rules. Correlation alone cannot authorize a production change. If a future
experiment isolates a specific owner or wait path, apply the smallest fix,
validate matched before/after correctness, throughput, visibility and drain,
then run a separately frozen new isolation cohort before changing the four-owner
support boundary.

Implement a production change only for a demonstrated owner/waiting path. Do not
reduce pressure, input, correctness work or drain. If evidence is insufficient,
hand off the bounded diagnosis and next discriminating experiment without adding
threads, schedulers or configuration layers speculatively. Any production change
requires refrozen Manager/native/wheel artifacts, matched before/after evidence,
affected Rust/Python and real Manager gates, and independent review. Even a valid
fix does not close isolation: a new formal qualification is a later substage.

Hardware-blocked cells remain explicit:

- **Physical cross-host cache/HA:** H20 and A100 are accessible physical GPU
  machines, and direct IPv6 TCP probes pass in both directions. H20 exposes an
  H20 through `nvidia-smi`, but its default user `libcuda` is 590.48.01 while the
  kernel driver is 535.161.08 and `cuInit(0)` returns 100; a bounded probe of the
  existing matching 535 library timed out and was terminated. Cache qualification
  therefore still needs a short-watchdog CUDA computation/IPC preflight plus the
  actual H20/A100 TENT IPv6 listener and transfer path. IPv4 is not the candidate
  topology because H20 has no route to the other two IPv4 addresses.
- **Three-host etcd:** H20, A100 and `orbitkv-cpu` are three distinct machine
  environments/boot IDs, and owned ephemeral listeners verified all six directed
  direct-IPv6 TCP edges with zero exits and no SSH data tunneling. Three-member
  etcd process-fault, leader, quorum-loss and recovery testing is now actionable,
  subject to preflighting the exact frozen etcd client/peer listener ports. This
  network readiness does not prove independent power or network failure domains
  and is not itself an HA pass.
- **Native GDS:** needs an NVIDIA GPU host with supported NVMe/filesystem,
  `nvidia-fs`/cuFile runtime and container access to the real block device/mount.
  The available SSD qualification path is io_uring, not native GDS.

These cells are `qualification blocked`, never passed by same-host diagnostics.
RDMA, final integrated serving and S3 lifetime evidence retain their own open
status and are outside S2.10a.

- Execute the [frozen workload/acceptance matrix](distributed-design.md#performance-acceptance-and-ablations)
  on actual live DRAM and io_uring storage. Increase one load dimension at a time,
  establish the supported envelope, and exercise expected bounded degradation
  outside it. Include the 30-minute live cell and two-hour fault/contention soak.
- Use three etcd voting members across real independent failure domains; test
  leader/host loss, quorum partition, lease expiry, rejoin and cold bootstrap while
  cache traffic continues. Separate physical TCP and RDMA profiles.
- Reproduce shared-cache serving in both pinned engine environments. Record
  freshness, effective hits, TTFT/ITL, transfer bytes, CPU/RSS, rebuild and resource
  drain; metadata-only benchmark wins do not qualify serving gains.

**Acceptance:** exact convergence and no stale authorization, predeclared envelope
and recovery targets met, frozen implementation and independent-review evidence,
and truthful topology/medium exclusions. S3-dependent crash reclamation remains
open until native termination proof is available. Keep blocked hardware cells
explicit and continue only work independent of them.

Production commit `50f77954` adds graceful inventory-stream shutdown and
abort-safe live session accounting. SIGTERM fences membership before gRPC waits,
then normal lifecycle drain and lease revocation finish. The strict full-Manager
gate observes member deletion before same-node restart, increasing epoch, changed
incarnation and refusal of the old source. Pre-fix `272803cf` times out after ten
seconds with an active stream; soak restarts exit in at most 0.42 seconds and the
separate strict two-start process gate takes about 1.32 seconds, both within the
ten-second limit. An intermediate soak exposed follower-task cancellation
leaking the diagnostic active count; the final candidate returns current and peak
session counts to the one-session topology after receiver/source replacement.

The frozen same-host A100 workload uses actual Manager Publish/Query/Restore and
forced TCP. All/scoped DRAM and all/scoped io_uring SSD each run at least 30
minutes, totaling 7,200 exact cycles and 7,200.50 steady seconds. The first five
matched pairs per medium preserve exact namespace/key/payload oracles but run
faster than the predeclared one-round-per-second cadence; scoped stream-byte
ratios are 0.2551 DRAM and 0.2519 SSD, with median bootstrap/repair ratios within
1.10. SSD recovery clears both source and requester DRAM and observes source SSD
reads plus remote bytes. Fixed-membership churn creates no etcd block/cursor
writes. The replacement five-pair matrix runs at the predeclared one round per
second. Correctness, cadence, stream reduction and bootstrap/repair pass, but both
DRAM and SSD have per-run save/query p99 ratios and 95% confidence bounds above
1.05. The isolation performance cell therefore fails; median-only evidence is
not used to override it.

The final candidate runs 7,181 mixed DRAM/SSD cycles over 7,200.10 seconds while
injecting stream partition/heal, a slow subscriber, journal overflow, receiver
restart, source restart, source lease expiry and epoch-3 recovery. It records zero
wrong bytes and stale authorizations, exact final coverage, bounded index/queue,
one current/peak session in each direction and local save/query p99 pressure
regressions of 2.13%/-2.93%. Quiet and fault-inclusive visibility are reported
separately; the intentional slow-subscriber window is not presented as ordinary
freshness.

The real-Manager supported owner envelope still stops at four. The historical
16-owner 51.72 ms result is retained as a serial-barrier endpoint failure, not an
ordinary visibility measurement. The replacement five-run ordinary endpoint is
exact and complete and passes 50 ms, but the same replacement profile fails the
frozen DRAM/SSD isolation contract, so it does not expand the supported envelope.
Five matched 16 MiB/1 MiB index-pressure
runs degrade explicitly to `partial_hints`, keep two of four owner views, remain
under the configured budget and clear staging. The earlier 256 MiB soak, shutdown
timeout, harness failures and the session-counter failure remain archived.

H20, A100 and the CPU host now have verified six-way direct IPv6 TCP
connectivity, so three-machine etcd process-fault/leader/quorum testing is the
next actionable infrastructure stage after independent review. H20 CUDA remains
unready for cache traffic, and the actual TENT IPv6 data path has not run; once
those preflights pass, H20/A100 cross-host cache qualification follows. The final
integrated S2.9/PR #198 wheel and official vLLM 0.30.0/SGLang 0.5.20
shared-cache serving, independent physical failure domains, RDMA, native GDS and
S3 crash reclamation are not qualified by this substage. Do not mark S2 or full
S2.10 complete from this readiness evidence.

## S3 — Transfer lifetime and generation-safe ownership

**Depends on:** S2's metadata contracts. **Owners:** core peer/query/storage owners,
transfer/sys C ABI, process channel and the consumed adapter registrations.

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

**Depends on:** S3 for new reuse/early-consumption paths. **Owners:** existing raw
transfer compiler/workers, channel completion owners, SSD/codec owners and engine
layer hooks. The current contracts live in [transport](transport.md) and
[engine-local restore](engine-local-restore.md).

- Profile adapter hashing/conversions, repeated destination checks, fragmented
  descriptors, dispatch, allocation/scratch work and duplicate shared-prefix H2D.
  Remove measured redundant work one bottleneck at a time.
- **KDA mapped-host copy implemented; local qualification passed (2026-10-07),
  independent acceptance open:** the existing optional `KernelBackend` now uses
  bounded multi-CTA copies for large fragments in small batches. CUDA sources
  are separate `.cu` files. Thirty component byte controls, six matched native
  runs (34 case/direction rows each, actual Rust submission and coalescing DMA),
  and five installed-kernel model gates pass on A100 SM80. The largest paired
  4 KiB median regression is 1.11%, below the frozen 5% guard. A 12-run follow-up
  on representative 128 KiB model-sized fragments remains near old-kernel parity;
  it does not establish an inference speedup. Existing backend selection,
  descriptor bytes and physical-drain ownership are unchanged. Independent reproduction, contention,
  graphs, H20 and S3-dependent faults remain open. Sources, raw samples, native
  comparisons, final wheel and failed builds remain outside the checkout at
  `/root/orbitkv-artifacts/kda-native-copy-20261007/`; see
  [kernel optimization](kernel-optimization.md). Codec/checksum fusion requires
  a separately measured, consumed bottleneck.
- **SSD host restore batching implemented; native gates pass (2026-10-07),
  installed gates pass; independent acceptance open:** production `50c6d58d` submits at
  most 16 unique SSD generations per existing reader batch and storage owner.
  Repeated layer references remain deduplicated; same-key generations are
  separated and every batch drains before an error returns or GPU sources change.
  No physical I/O coalescing, new early readiness or cache policy is implied.
  The frozen 12-cell A100/ext4-NVMe io_uring cohort passes exact bytes, physical
  reads and all 15 paired median guards (largest ratio 1.0154). Twelve 4 KiB
  blocks become one reader batch, with geometric mean median ratio 0.645;
  the 108 MiB interval crosses parity. Python/native call overhead is included;
  p99 is descriptive and one small 34-block pair rises 0.738 → 2.486 ms. Fourteen
  real SSD native tests pass without environment skips. The unchanged complete
  wheel `8acd6a06…` passes all three official-engine SSD/shared-Manager gates
  in a fresh durable cohort, with 20 normal zero exits and unchanged installed
  files. The original passing cohort has incomplete provenance after temporary
  cleanup; its surviving records are preserved separately and no old cell is
  reused. The final native cohort retains all 3,000 samples in `/workspace`,
  with SSD data on the same NVMe mount. Sources, frozen contracts, failed runs and the
  final handoff stay outside Git at
  `/root/orbitkv-artifacts/s5-ssd-host-batching-20261007/`; see
  [SSD performance](ssd-performance.md).
- **Bounded KDA/physical-read follow-up evaluated (2026-10-08): both candidates
  rejected; measurement support implemented, independent review open.** CPU
  descriptor reuse passes all native byte controls but a grid-stride H2D median
  regresses 5.98%, beyond the frozen 5% guard; later shape/pair runs are not
  started. Adjacent same-batch READV merging reduces 12-block requests from 12
  to one and 34-block requests from 34 to three, but the first matched small
  medians regress 10.02% and 24.61%. Its qualification stops immediately.
  Exact-byte/short-read/lease gates and 14 SSD integration tests pass, which
  does not authorize performance promotion. Both candidate implementations
  are removed from production and archived outside Git; CUDA sources/defaults
  and the accepted scalar read path stay unchanged.
  The final Manager counts queued READV SQEs separately from blocks and NVMe
  commands; the communication harness retains before/after stage snapshots
  outside its timer. A distinct four-cell cost-enabled diagnostic completes
  without promotion: the 108 MiB baseline averages about 20.3 ms SSD preparation
  and 4.8 ms direct H2D; instrumentation overhead is unqualified and overlapping
  stage means cannot decompose percentile latency. Next inspect existing SSD
  preparation/read parallelism and completion scheduling before another candidate;
  the S5.5 startup blocker remains unresolved. Contracts, frozen wheels, failed
  gates, rejected sources and byte controls are archived at
  `/root/orbitkv-artifacts/s4-readv-kda-20261008/`, mirrored at
  `/workspace/orbitkv-perf-20261008/`. This closes neither S4 nor S5.
- **SSD read fanout evaluated (2026-10-08): rejected; no production change.**
  Reducing the existing single-file reader from 15 to seven read queues passes
  all 14 real SSD lifecycle tests and exact-byte/physical-read controls, but the
  third matched small-shape pair exceeds the frozen 5% median guard (ratios
  1.753 and 1.594). The cohort stops there; the remaining model cells and installed
  promotion gates are not started. The accepted scalar reader and 16 total
  io_uring threads remain unchanged. Lower whole-cohort CPU ticks do not justify
  a latency regression. Frozen sources, wheels, controls and failed samples are
  retained at `/root/orbitkv-artifacts/s4-ssd-read-fanout-20261008/` and
  `/workspace/orbitkv-ssd-fanout-20261008/`; see [SSD performance](ssd-performance.md).
- Profile actual descriptor shapes and inference contention before further
  mapped-host copy changes. Consider SSD-staging scatter or encoding/checksum
  fusion only after their consumed route exposes a material dispatch/HBM cost.
  Keep device code in the existing `.cu` files and Rust ownership through final
  drain. Native GDS eligibility, lossy codec quality and physical I/O merging
  are separate gates; [kernel requirements](kernel-optimization.md) specifies
  the controls. Next S5.5 qualification is sustained shared-Manager SSD mixed
  save/restore, per-instance progress and bounded cleanup, not a new scheduler.
- Extend readiness to legal multipart and SSD/codec pipelines where it can
  shorten the engine critical path. Retain the parent source/destination fence
  and final drain; a ready layer does not retire the whole operation.
- Add only consumed deadline/priority hints and bounded preparation needed by
  these paths. Qualify concurrent Query/Publish, fairness and long-running stalls.
- Address the measured SGLang native-CPU latency gap and ANS SSD overhead before
  enabling new defaults. Qualify GPU writeback/admission and codec quality over
  actual dense/hybrid workloads; native GDS needs a real supported mount with
  fallback disabled and physical I/O evidence.

- Measure native completion observation and Python waiting/thread handoffs before
  replacing them. Prefer bounded batching and demand-driven progress; completion
  must return network credits without waiting for parsing, H2D or model callbacks.
  Reuse upstream TENT priority/progress capabilities after checking their actual
  release/configuration behavior. Reuse the node-local Manager and existing
  native execution owners; keep short protocol work near I/O progress and heavy
  codec/SSD work on bounded workers. Do not add a second RPC Agent.
  Do not build a custom TCP stack or modify TENT
  to host OrbitKV's metadata protocol.
- Add mixed inference/network controls: cache READ, P/D WRITE and engine NCCL/
  expert traffic on shared versus separate NICs. Measure decode ITL tails, TTFT,
  SLO goodput, CPU cores, PCIe/NUMA pressure and physical NIC counters. TENT's own
  load counters do not measure non-TENT traffic; RDMA is not bandwidth isolation.
  Separate bare TENT cold/warm READ/WRITE, full Manager recovery and each engine's
  serving comparison. Same-host NVLink needs an observed consumed transport;
  link status cannot substitute for route or payload evidence. Before a TENT
  repair, isolate segment discovery, registration, QP/progress and wrapper
  completion, reproduce against the pinned upstream and retain fault/drain
  controls. Follow [the maintained method](benchmark-evidence.md#tent-and-fabric-diagnostics).
- **First RDMA READ diagnosed (2026-10-10), independent review open; no production
  candidate promoted:** bounded native interposition splits metadata RPC,
  endpoint construction, connect and bootstrap on the frozen `71973589` TENT.
  Eight trace-off/on cells pass the declared warm-observation overhead guards
  and 448 exact 8 MiB byte oracles. Metadata calls are about 1 ms; endpoint
  construction is about 23–33 ms, with overlapping bootstrap calls and source
  RPC serialization. These per-host intervals cannot be added into an end-to-end
  breakdown or identify a CUDA kernel bottleneck.
  Two separate 20-cell, five-pair-per-direction configuration cohorts are valid
  but rejected. RPC threads 1→4 lower paired cold READ ratios to 0.769/0.807,
  above the frozen 0.75 target; one warmed p99 increase is 0.273 ms against a
  0.25 ms guard. RDMA lanes 6→2 do not lower cold READ and regress paired warm
  medians by about 9–10%. All 12,688 pilot/candidate payload oracles pass;
  all 96 endpoints exit zero and recorded PIDs disappear. Native libraries and
  production defaults remain unchanged. No full-Manager candidate promotion or
  replacement run follows these failures. Keep the upstream segment-warmup
  proposal distinct from a released consumed interface; evaluate setup cost,
  first recovery, contention and source/drain ownership before activation.
  Frozen contracts, raw samples, traces and rejected configurations are at
  `/root/orbitkv-artifacts/s4-tent-native-diagnosis-20261010/`.
  **Final recorder remediation implemented, observation qualification blocked:**
  review adds release/acquire event publication, sealed span admission and
  explicit zero-exit validation in the offline analyzer. CPU writer controls
  pass 1,024 concurrent events and reject active calls/unpublished slots;
  306 benchmark tests pass. The separately frozen final eight-cell pilot passes
  448 byte checks and 16 normal exits, but direction 1 warm median ratio 1.0585
  exceeds 1.05 and observed p99 increase 0.3468 ms exceeds 0.25 ms. Stop without
  retry or deeper diagnosis. The initial `5b1ca6bb` cohort stays immutable;
  its overhead result does not qualify the revised recorder. Source, failed
  final pilot and controls are separate at
  `/root/orbitkv-artifacts/s4-tent-native-trace-remediation-20261010/`.
- **Upstream notification-MR repair measured (2026-10-10), component target
  failed; independent review open:** the selected `v0.3.13.post1` has 256 receive
  MRs per RDMA endpoint. The already merged upstream
  [PR #3820](https://github.com/kvcache-ai/Mooncake/pull/3820) uses one MR covering
  those same slots. An isolated release-source baseline and three-file repair
  build differ only in `libtent_shared.so`; seven other shared libraries are
  identical. No production pin, library, wheel or engine is changed, and no
  duplicate upstream issue/PR is filed.
  One untraced 20-cell, five-pair-per-direction formal cohort passes 10,120
  complete 8 MiB READ and 10,240 unique notification oracles. Cold paired ratios
  are 0.9025/0.8698, with 95% independent-pair intervals [0.8130, 0.9901] and
  [0.8578, 0.8820]. Warm median and observed p99 regression guards pass, but
  both directions miss the frozen cold ratio <=0.75 target. Keep
  `VALID_COMPONENT_FAIL`; no replacement cohort or full-Manager promotion.
  Separate readiness checks retain 64 READ and 2,048 notification oracles;
  all 48 endpoint exits are zero and recorded PIDs/listeners disappear.
  Both-host native units pass 130 cases without skips; 312 benchmark tests
  pass. Missing unit GoogleTest libraries and a missing host-1 probe/config
  preflight are preserved; the latter starts no endpoint or payload sample.
  Corrected inputs are verified before a separately frozen smoke/formal launch.
  Frozen builds, contracts, physical port counters, raw samples, failed
  preparation and cleanup evidence are at
  `/root/orbitkv-artifacts/s4-tent-notify-mr-20261010/`.
  These are private CPU-buffer controls, not Manager grants, engine GPU pages,
  serving latency or S3 controller-loss reclamation. S2/S4/S5 remain Partial.
- **Implementation partial:** direct registered engine-page SSD I/O and multi-writer
  GPU assembly remain beyond the existing staged GPU path. Include registration,
  source-page hold time, fragmentation and extra HBM in their admission decision.
- **Qualification open:** native GDS, selective DRAM admission/GPU writeback,
  hybrid codec quality, and the SGLang ANS SSD latency gate. Historical TurboQuant
  greedy-output failures remain visible; keep recurrent state exact until its
  quality and correctness gates pass. Current hardware evidence cannot close them.

**Acceptance:** byte-exact eager/graph, fragmented/packed/hybrid and partial-failure
  gates; repeated matched native HBM, native CPU offload, OrbitKV and LMCache
  comparisons for each engine. At least three order-alternated runs, fixed
  workloads/budgets and thresholds declared before execution. Report TTFT/TPOT,
  throughput, tails, CPU/pinned-memory costs and physical bytes; include misses
  and failed preparations. Reject overhead-only optimizations or keep them
  experimental. Do not compare an external miss to a resident HBM hit as equal work.

## S5 — Released-engine integration and upstream contributions

**Depends on:** S2/S3 for new lifetime behavior; consume S4 when ready. S5.1 may
start independently. **Owners:** engine adapters, existing native client owners
and narrowly scoped upstream engine interfaces. Reference released LMCache
integration contracts from [the adapter guide](adapters.md#lmcache-and-flexkv-reference).

The dated records below retain their frozen cohorts. Current scope is summarized
for the same five substages; a later scoped acceptance does not qualify an earlier
or broader profile.

| Substage | Current delivery | Remaining work |
| --- | --- | --- |
| S5.1 | Release/interface inventory independently accepted at `38f8dbb2`; official engine pins remain 0.31.0/0.5.21 | Track upstream interfaces separately from released support |
| S5.2 | Scoped ordinary dense cache/restart, shared-Manager pressure and serial batch-1 FULL decode SSD recovery accepted; see S5.5 evidence | Broader model/topology and lifetime profiles remain unqualified |
| S5.3 | Idle vLLM external reset accepted at `6d874384`; immutable local PEFT adapter identity accepted at `47fea821`; official dense query handshake consumption accepted at `1ab67a58` within the scope below | Public SGLang lifecycle, dynamic adapter/live-weight identity and native hybrid-state contracts remain open |
| S5.4 | Official native P/D cutover implemented with bounded model checks | Independent native fault, transfer cancellation, delayed-ACK and page-reuse qualification depend on S3 |
| S5.5 | Same-host deployment/pressure/common-prefix coordination and physical two-H20 ordinary dense TP=1/PP=1 eager DRAM/io_uring RDMA serving with controlled restarts have scoped acceptance | Protected cross-host query-control activation is missing; physical TP/rank/container, isolation, HA and broader lifetime/deployment gates remain open |

### S5.1 — Release and interface audit

**2026-10-06 upgrade: vLLM 0.31.0 / SGLang 0.5.21 implemented; independent acceptance open.**
Both are official released sources, with aligned dependency, probe and submodule
pins checked by CI. The complete CUDA 13 wheel integrates only the accepted main
Manager baseline `3af8022e`. Its 27 packaged Python source/stub files match the
checkout. On A100, all four Qwen3-8B installed-wheel dense TP=1/PP=1 eager
DRAM/io_uring cells pass cold/native-HBM/full/partial reuse, RECORD integrity and
normal-exit gates. Each full and partial restore loads 113,246,208 bytes; all 16
service processes exit zero without forced cleanup. Both SSD cells prove actual
io_uring reads after DRAM eviction. The same final wheel passes the standard
vLLM model gate (six tests and one dense-model recurrent skip), SGLang direct
GPU restart recovery (two tests), and released lifecycle selections (15 vLLM,
43 SGLang). The model gate uses the release's default graph selection; this is
not a full-graph-specific, pressure or hybrid qualification. A first native
baseline failed in shared-cache Triton compilation with `Unknown key: 'cubin'`;
the same command passes with new per-run compiler caches. The failure remains
preserved and the former shared cache is not certified. Final GPU use is zero
and the owned-process postflight is empty.
Preserve older cohorts; do not transfer their P/D, hybrid or topology claims to
a new release. Independent review and broader S5 qualification remain open.
Evidence: `/root/orbitkv-artifacts/s5-release-upgrade-20261006/`.

The new release audit still finds V2 restore after recurrent preprocessing,
SGLang graph capture before external factory construction, no public pending
lookup admission ticket and device P/D timeout release without a complete drain
ACK. Keep their dependent profiles closed or experimental. Dynamic LoRA is
already rejected by both adapters; live-weight invalidation and cross-host TP
query coordination remain implementation work, not hardware-only gates.

**Historical vLLM 0.30.0 upgrade record; independent review remains open.**
The following preserves the earlier 0.30.0 delivery. Cache and
P/D workers return native `KVConnectorTransferResults`; failed receive and
finished receive share one snapshot, replacing `PdWorkerMetadata` and its
duplicate queue. Best-effort cache publication explicitly returns false for
`requires_kv_delivery`. The native MultiConnector consumer is covered by a
released-engine test. No 0.29/0.30 compatibility alias remains.

The following records the earlier release-upgrade delivery. Its hybrid and
custom P/D evidence is historical; the current restricted, official native P/D
profile and its final installed-wheel gates are tracked in S5.4.

The complete CUDA 13 wheel passes build/repair/isolated import. Its installed
production package passes A100 Qwen3-8B DRAM eager, DRAM graph and forced-io_uring
SSD eager gates (six checks each; the recurrent-only check is inapplicable).
Same-GPU TCP P/D and save-only MultiConnector each pass one gate; the latter
uses DecodeBench as the load owner and does not qualify live P/D/cache overlap.
Fourteen exact native recovery tests and three released connector-contract tests
pass. Fixed Qwen3.5-0.8B passes seven checks each for DRAM eager, DRAM graph and
forced-io_uring SSD eager modes. The full A100 selection totals 58 passes and
three inapplicable dense-model recurrent skips.
These profiles use TP=1/PP=1; multi-GPU, RDMA and historical model/topology cells
remain separately unqualified on the new release.
Source-only Python passes 413 tests, with one skip and 160 deselected. The
generated `python/uv.lock` is ignored by repository policy; its resolved 0.30.0
snapshot is archived with the wheel rather than force-added.
All 44 Python production modules match the tested wheel byte for byte. The first
six gates preserve all 146 installed-artifact hashes and 72 test-file hashes;
the extended hybrid/recovery selection also preserves its installed/test hashes
and returns GPU usage to zero. Local full Rust tests reach 47 CUDA-device failures
in `orbitkv-core` on this GPU-inaccessible host; they are not recorded as passed.
The PR's CI covers both CUDA builds, Clippy, Python and wheel packaging separately.
Evidence: `/root/orbitkv-artifacts/engine-native-lifecycle-20260930/vllm-0.30/`.
Previous 0.29.0 model, topology and performance measurements remain historical.
This delivery does not close S5.3/S5.4 or establish a performance advantage.

- Audit vLLM 0.31.0 and SGLang 0.5.21 by exact release commits above, and LMCache
  0.5.5 against those interfaces. Check recipe prerequisites in release source;
  do not assume a documented upstream PR is included in either engine release.
- Inventory every public callback, internal Hook, monkey patch, local request
  state machine and configuration entry. Name its resource owner and replacement,
  distinguishing already released APIs from main-only or proposed extensions.
- Upgrade vLLM from 0.29.0 as its own tested change, synchronizing package pins,
  lockfile and engine submodule. Retain 0.29.0 evidence as historical; run new
  release correctness, restart, eager/graph and overhead gates before changing
  public support claims. Keep one maintained release implementation per engine.
- Audit reuse of native vLLM P/D: layout/state coverage, rank mapping, source
  retention, cancellation, error reporting and composition with cache restore.
  Native P/D cutover and remaining qualification are tracked in S5.4.

**Acceptance:** release/API inventory with exact callers and an upstream issue/PR
or a bounded local responsibility for each remaining internal dependency.

### S5.2 — Minimal official cache backends

**Local adapter cleanup implemented and checked on A100; independent review open.**
The public vLLM entry constructs only `SchedulerAdapter` or `WorkerAdapter`,
closes native connections on initialization failure, and inherits unchanged
optional callbacks. The unused service availability owner and its health thread
are removed; restore exceptions still retain destinations until native drain.
SGLang event ownership lives in `events.py`, and disabled-backend plugin/admission
paths leave native and GPU modules unloaded. Registration conflicts are explicit.
In that earlier cleanup, Prefill and Decode directly owned their P/D worker callbacks and request state;
the intermediate Handlers, class-callback mixin and generic executor facade are
removed. SGLang cancellation consumes the released cache finish/linker release
chain instead of a duplicate Scheduler abort Hook. Ordinary cache registration
has two internal Hooks; enqueue preparation remains opt-in. The later S5.4
cutover removes those custom P/D owners and fork observation Hooks entirely;
the following evidence records the earlier cleanup, not current native P/D support.

**Single-rank admission cleanup independently accepted on H20 (2026-10-08).** The SGLang wrapper captures whether released
attention/TP groups need consensus once. Single-rank request admission and hybrid
materialization use scalar decisions without allocating Torch tensors; multirank
MAX/MIN decisions and pending/expiry/cancel ownership remain consumed. The final
complete wheel `fe427c42…` passes 42 installed official SGLang 0.5.21 controls,
four synthetic GPU DRAM/SSD recovery controls and installed SGLang SSD
full/partial/restart. Fresh official vLLM 0.31.0/SGLang 0.5.21 environments resolve
all declared dependencies, with no source OrbitKV package shadowing imports.
Both 900-second shared-Manager direct/kernel cells complete and every fresh/reused
request matches native input/text/token-count controls. All 75 native package
members are unchanged from the prior wheel. The two private safety Hooks and
advertised rank/model scope remain unchanged; no measured speedup is claimed.
The original A100 queue was cancelled before launch after H20 readiness passed;
Traex's S2 campaign was untouched. Earlier evidence is immutable at
`/root/orbitkv-artifacts/s5-single-rank-admission-20261008/`; the complete H20
handoff is `/root/orbitkv-artifacts/s5-h20-closeout-20261008/PARENT-HANDOFF.md`.

**H20 direct CUDA runtime independently accepted in this process profile (2026-10-08).**
Default initialization returned `CUDA_ERROR_NO_DEVICE` with the 590 compatibility
library or timed out with matching 535 and isolated official 580 libraries.
With the same 580 library and GPU UUID, changing only `CUDA_MPS_PIPE_DIRECTORY`
to an empty private directory turns a 30-second timeout into CUDA_SUCCESS in
0.164 seconds. The existing 590 compatibility library then passes too. Both
fresh engine environments pass real CUDA arithmetic and spawned CUDA IPC:
65,536 exact bytes and child computation 100,708,352, exit zero and empty GPU
postflight. Shared MPS, host driver and global library links are unchanged.
This is a direct-CUDA process profile, not a shared-daemon diagnosis or a
cross-container/MPS, TENT/HA or S2 qualification. Failed controls remain in
`/root/orbitkv-artifacts/s5-h20-container-runtime-20261008/` and
`/root/orbitkv-artifacts/s5-h20-readiness-20261008/`; see
[runtime setup](deployment.md#h20-direct-cuda-runtime).

The initial cleanup's source-only gate passed 414 tests. With frozen native artifacts, the pinned
SGLang 0.5.20 admission/event gate passes 22 tests and the vLLM 0.29.0 native
recovery-contract gate passes seven. These integration tests use controlled
completion and CUDA-event doubles, not real GPU DMA. The vLLM gate also corrects
a pre-existing async-load expectation reproduced on the unchanged baseline: a
forward-consumed recovery reports a synchronous scheduler hit.
The A100 Qwen3-8B TP=1/PP=1 serving gates pass: vLLM reports six passed and one
hybrid-only assertion skipped; SGLang passes DRAM and io_uring SSD recovery.
The gates compare model outputs with controls, check native cache reuse and
verify external loads after process restart. The dense model leaves hybrid
serving qualification open.
The SGLang event/linker gate passes 19 tests, including real GPU page overwrite,
full/window/checkpoint recovery on DRAM/SSD, and controlled failure/drain cases.

Two initial SGLang SSD starts failed at TCPStore binding before model loading.
A minimal reproduction showed that loopback availability did not imply wildcard
availability. The test port helper now checks wildcard bind/listen; a real-socket
regression protects this boundary, and the fixture retains Manager logs.
Both failures and the successful rerun remain in external evidence. The original
fixture deleted its temporary Manager logs during the first run; its pytest and
engine logs remain available.

The P/D ownership follow-up passes 413 source-only tests, including 139 P/D
contracts; the assertion-only test for the removed Handler layer is deleted.
SGLang's pinned admission/event gate passes 24 tests, now including native
cache-finish cancellation and successful-finish preservation. A reusable vLLM
P/D E2E gate runs two TP=1/PP=1 eager workers on one A100 over forced TCP:
129/257/769-token prompts, including chunked prefill, produce the same text and
output tokens as native execution; 198,180,864 payload bytes are reported and
all sender/waiter gauges drain. This profile does not qualify multi-GPU/RDMA,
hybrid P/D, live handoff/cache composition or the proxy HTTP service itself.
SGLang DRAM/SSD serving recovery also passes again (two cases), as do four
GPU recurrent/window recovery cases including cancellation of published pages.
The seven frozen native artifacts are unchanged; postflight finds no engine or
Manager processes and GPU usage returns to zero.
The obsolete Python P/D launcher is removed: it selected ordinary cache
connectors and invoked the removed `orbitkv-router` binary. The maintained
deployment entry is `scripts/run_pd_local.sh`.
Follow-up evidence: `/root/orbitkv-artifacts/adapter-cleanup-20260930/pd-followup/`.

Overall S5.2 acceptance, engine upgrades, upstream registration, and additional
multi-GPU/P/D qualification remain open.

The installed-wheel ordinary-cache gate now defines four pinned-release cells:
vLLM/SGLang × DRAM/io_uring SSD, dense TP=1/PP=1 eager. It compares native
controls, cold/native-HBM/full/partial reuse, strict Manager SIGTERM drain and
installed engine/OrbitKV RECORD integrity. Source-injected E2E remains a separate
development gate. Implementation and qualification are recorded separately; this
addition alone does not close S5.2, public-Hook or native P/D lifecycle work.
Evidence: `/root/orbitkv-artifacts/adapter-cleanup-20260930/`.

On 2026-10-06, the frozen CUDA 13 wheel passes all four installed-package
cells on A100 with Qwen3-8B. Native HBM hits add no external query or H2D copy;
full and partial restores each load 113,246,208 bytes. SSD cells prove io_uring
reads after DRAM eviction. All 16 service processes exit zero with no forced
cleanup; installed engine and OrbitKV RECORD hashes are unchanged. The broader
vLLM correctness gate passes six tests with one dense-model hybrid skip;
released-engine lifecycle selections pass 15 vLLM and 43 SGLang tests. These
controlled lifecycle tests do not qualify native P/D faults or hybrid serving.
The installed console now executes the Manager directly, preserving its PID,
signals and exit code. Raw RECORD validation also passes the Python 3.13 CI
missing-file and missing-directory controls.

**Local ordinary-cache checks pass; independent acceptance remains open.**
The first SGLang exit-counter diagnosis is an **invalid measurement**: it read
`offset * 8` from the shared-memory file, including the 64-byte MapInfo header,
instead of `64 + offset * 8` from its counter payload. Its reported remaining
value was the file reference count, not a tensor export credit. The failed
interpretation and original trace remain immutable.

The calibrated follow-up on the same installed wheel checks independent tensors
and shared-storage views with 3 and 72 exports: every credit returns to zero while
the consumer remains alive. Official SGLang normal close similarly returns all
72 OrbitKV credits to zero after successful Manager unregister and at Python
exit; both service exits are zero and GPU postflight is empty. The PyTorch exit
warning remains: its producer bookkeeping is not a per-counter leak oracle.
The counter reader has first/last-live and truncated-slot regression controls.
This is local normal-exit evidence, pending independent review; it does not
qualify crash reclamation, cancellation during DMA, pressure, graphs, multi-GPU,
cross-host cache or native P/D composition. S5/S3 broader lifetime work remains
open. Follow-up evidence: `/root/orbitkv-artifacts/s5-ipc-release-20261006/`;
A100 raw root: `/workspace/orbitkv-s5-ipc-release-20261006/`.
Frozen inputs, failures, cleanup and diagnosis:
`/root/orbitkv-artifacts/s5-installed-cache-20261006/candidate-1b3d4c22/`;
A100 raw mirror:
`/workspace/orbitkv-s5-installed-cache-20261006/candidate-1b3d4c22/`.

- vLLM: retain `OrbitKVConnector` and distinct scheduler/worker responsibilities.
  Implement released lookup/allocation, registration, load/save and terminal
  callbacks; consume success and failure through the selected release's API.
  Keep cache transfers and engine page-allocation ownership separate.
- SGLang: retain `UnifiedRadixCache` plus `OrbitKVLinker`, with native tree and
  component ownership. Reuse the released construction and layer-counter flow;
  propose a small external-linker registration extension only where needed.
- Keep runtime/native imports lazy and dependencies optional. Unselected OrbitKV
  must not initialize CUDA, TENT or a Manager connection. A complete native HBM
  hit must not add synchronous external lookup or a duplicate H2D copy.
- Remove duplicate plugin registration at the official-registry cutover; do not
  suppress all registration errors. Remove internal forwarding facades while
  retaining the one adapter that actually implements each engine contract.
- Submit baseline registration/configuration, focused tests and installation
  docs as small upstream PRs. Core cache policy, etcd and storage stay in OrbitKV.

**Acceptance:** clean supported engine checkout plus an installed OrbitKV wheel;
no source editing needed for the claimed profile. Cold/partial/full hits,
pressure/restart reuse and disabled-backend import behavior pass. Upstream PR
status is reported separately from local adapter qualification.

### S5.3 — Public lifecycle and hybrid-state contracts

The 2026-09-30 source check confirms that released 0.30.0 and current vLLM main
invoke preemption drain after page initialization in the **V2 runner**, and
synchronous restore after recurrent preprocessing. The official V1 runner
already calls preemption before page updates. The generic preemption-ordering fix is submitted
as [vLLM #59410](https://github.com/vllm-project/vllm/pull/59410), linked to
[#59409](https://github.com/vllm-project/vllm/issues/59409), at `2f868f14` based on
main `91dab0eb`. It supersedes the earlier unsubmitted `98b917d5` patch. Local
qualification includes 46 A100 worker tests and one native MultiConnector
Qwen3-8B output gate; the upstream contributor-eligibility gate still needs
maintainer validation. It is not merged or released. Evidence:
`/root/orbitkv-artifacts/engine-native-lifecycle-20260930/vllm-upstream/submission-20260930/HANDOFF.md`.
The release-only profile now uses V1 and rejects multi-group/recurrent serving;
its `runtime.py` and native-prefix override are removed. The recurring-state
ordering gap remains an upstream requirement for reopening that profile. SGLang PR #40595 (external-linker construction) and #40896 (load-failure
lifecycle) are open; #40759 (Mamba lifecycle proof of concept) is closed unmerged.
Do not delete consumed safety behavior based on these proposals.

- vLLM: the runner patch is removed for V1 single-group serving. Reopen V2 and
  recurrent profiles only after the released interface
  guarantees preemption/save drain before page reuse and restore after page
  initialization/COW but before recurrent state preprocessing. Prefer a generic
  ordering fix to a new callback when the existing contract suffices.
- The multi-group native-prefix bypass is removed with that serving profile.
  Reopening it requires correct atomic state-group availability/recovery. Audit existing divergent-hit APIs;
  changing a capability flag alone does not prove all states ready.
- SGLang: move generic recurrent checkpoint allocation/commit/abort and external
  linker lifecycle support into engine components where accepted. Shrink
  `RecoveryLinkerWrapper`; preserve its safety checks until equivalents are
  actually consumed. Public Full/SWA support does not imply recurrent support.
- Replace private enqueue/abort/release and decode-ready Hooks with explicit
  lifecycle contracts. Keep telemetry out of ownership transitions and model
  computation in the engine. Incorporate generation checks from S3.
- **Explicit vLLM external reset independently accepted (2026-10-08):**
  the released `reset_cache()` callback refuses every tracked request, source
  query, destination load and publication owner. Once idle, it assigns a fresh
  prefix-key generation; Rust transforms the native hash batch and the scheduler
  reuses it until the native hashes change. Query and save consume the same keys.
  `reset_external=true` invalidates future reuse by this engine process without
  deleting other replicas' Manager data. Ordinary HBM-only reset and immutable
  process-restart reuse stay unchanged. This does not invalidate other engines,
  persist an invalidation across restart or make weight changes that bypass the
  callback safe. SGLang's ordinary flush retains external reuse; its skipped-flush
  weight updates still require a public consumed lifecycle contract.
  The reviewed production commit is `6d874384`; the complete CUDA 13 wheel is
  `605ec4c0…`. Independent source-only tests pass 296 cases with one skip.
  On one H20 with official vLLM 0.31.0, dense Qwen3-8B, TP=1/PP=1 eager and
  `direct`, DRAM and overlay io_uring SSD reset gates both pass. Old-generation
  load is zero; newly saved state restores 113,246,208 bytes after an HBM-only
  reset, including the same physical io_uring read in SSD. All eight installed
  gate services exit zero without forced cleanup. Standard model correctness
  passes six tests with one dense-model hybrid-only skip, and its 12 cache
  outputs match native controls. The standard Manager audit records cached exit
  zero and closed process group/socket, but not every possible force-signal
  attempt; this does not prove S3 physical DMA termination. All 163 independent
  frozen inputs and installed package RECORD files stay unchanged; the parent
  376-file evidence manifest passes. Final scoped review:
  `/root/orbitkv-artifacts/s5-external-reset-20261008/reviewer/FINAL-REVIEW.md`.
  This closes only the idle reset contract, not S5.3 or the whole S5 stage.
- **Immutable static adapter identity independently accepted (2026-10-09):**
  reviewed production commit `47fea821` binds fixed local PEFT bytes and effective
  LoRA configuration to the deployment namespace. Native keys retain base/adapter
  separation; same-name/path replacement between restarts changes the external
  domain. vLLM public admission enforces declared immutable selections. SGLang
  consumes the released resolved LoRA configuration and validates UIDs through
  its existing linker/admission boundaries. Sessions, resumable requests and
  caller extra-key aliases are rejected in this profile.
  The final repaired CUDA 13 wheel `dec39b87…` independently passes both official
  vLLM 0.31.0 and SGLang 0.5.21 on H20, dense Qwen3-8B, TP=1/PP=1 eager,
  direct transfers and overlay io_uring SSD. Base/A/B cold requests miss;
  native HBM hits add no external query or load. Each restart restores
  113,246,208 bytes through completed io_uring SSD reads and exact native
  outputs/selected logprobs.
  Replacing A at the same name/path produces a cold miss with zero old-domain
  load. Installed engine/OrbitKV RECORD files and all frozen inputs stay unchanged;
  every outer service exits zero and owned GPU/process/socket postflight is empty.
  SGLang's native child teardown and CUDA IPC warnings remain recorded; this does
  not prove physical DMA termination or that every child avoids SIGKILL.
  Adapter files, symlink targets and loaded bindings must stay immutable for the
  entire engine lifetime. Dynamic SDK/RPC changes and live-weight invalidation
  still require consumed engine contracts. Preserve salt, computation, multimodal
  and representation boundaries; see [restrictions](adapters.md#immutable-static-lora).
  Parent failures, independent raw runs and review are preserved outside Git at
  `/root/orbitkv-artifacts/s5-static-adapter-identity-20261009/`.
- **Explicit decode-graph recovery independently accepted (2026-10-09):**
  test/qualification commit `c3f55060` consumes unchanged production `47fea821`
  and installed CUDA 13 wheel `dec39b87…`. Both official vLLM 0.31.0 and
  SGLang 0.5.21 pass direct/kernel io_uring SSD cells on one H20 with dense
  Qwen3-8B, TP=1/PP=1, serial batch=1 and eight generated tokens. The four parent
  and four independent cells retain 64 raw responses and 40 paired cache/native
  output controls. Every response adds at least seven native runtime Graph
  observations; vLLM counts FULL single-token executions, including eligible
  single-token prefill, while SGLang uses its decode Graph counter. Capture logs
  alone cannot pass. Full/partial restores and vLLM reset-generation reuse each
  load 113,246,208 bytes through completed io_uring reads. All 32 outer services
  exit zero without helper force; applicable query/SSD drain gauges are zero,
  installed RECORD snapshots and 84 frozen inputs are unchanged, and owned postflight is
  empty. Native shutdown traces/child teardown warnings remain preserved; these
  are not S3 termination proof. Default eager stays unchanged. Preset capture
  sizes 2/4 do not qualify those runtime batches; long decode across save-block
  boundaries, concurrency/preemption/cancellation, adapters with Graph, prefill
  capture, hybrid and physical topology profiles remain separate. Frozen contract,
  raw evidence and scoped review:
  `/root/orbitkv-artifacts/s5-graph-cache-qualification-20261009/reviewer/FINAL-REVIEW.md`.
- **Registered TP query-control transport independently accepted (2026-10-09):**
  production commit `b1ea00a7` adds opt-in Submit/Poll/Claim/Cancel/Close on the
  existing control listener, restricted to trusted same-host loopback endpoints.
  Bind and advertise addresses are validated before CUDA initialization. Local
  authenticated UDS export binds authority to Manager incarnation, sealed
  registration generation, namespace/layout and local consumer count. Interests
  retain at most 128 operation identities; consumers need a fresh interest per
  lookup and explicit close after native handoff/completion. Local and network
  reads share `PendingQueries` admission. Registration generations fence late
  completion and final lease consumption; shutdown fences query admission and
  drains submitted work before lifecycle/storage teardown.
  The independent reviewer reruns real UDS/gRPC/CUDA byte ownership and all seven
  pending-query controls, installed export enabled/disabled, both official engines'
  six ordinary-cache profiles, and standard vLLM correctness (six passed, one
  dense-inapplicable skip). The final rebuilt wheel `6d905396…` separately passes
  native controls, compiled/installed startup rejection, export bindings and one
  vLLM SSD/direct model recovery. The older wheel's six profiles are not relabelled
  as final-wheel measurements. Old 312 and final 144 frozen inputs and installed
  package hashes remain unchanged; structured outer-service cleanup and owned
  process/port/UDS/GPU postflight pass; native child teardown and standard Manager
  signal-trace limits remain in the scoped review. The byte fixture uses two real Rust Engine/UDS/gRPC endpoints
  in one OS process, not two independent Manager processes or physical TP.
  Scheduler common-prefix coordination and consumed official worker handshake
  are outside this frozen transport delivery; their subsequent acceptance is
  recorded below. Cross-host activation requires protected transport and qualification. S3 physical lifetime and full S5 remain open. Scoped review:
  `/root/orbitkv-artifacts/s5-tp-query-control-20261009/reviewer/FINAL-REVIEW.md`.
- **Native dense query coordination independently accepted (2026-10-09):**
  reviewed production commit `1ab67a58` and complete CUDA 13 wheel `092ff789…`
  put bounded parallel source queries, common-prefix acquisition, exact Claim
  retries, revision/cancel fences and completion-held interests in the existing
  Manager. Authenticated local bootstrap owns immutable topology; channel ABI 12
  and lifecycle version 6 require a matching installed client and Manager.
  PendingQueries/QueryBudget retain source admission, and shutdown fences
  coordinator work before bounded Close and query/lifecycle/storage drain.
  Official vLLM 0.31.0 worker handshake carries sealed targets; the scheduler
  opens only its local UDS. Unique global ranks report successful native Restore
  completion before the coordinator releases its source interests. The former
  Python serial dense-shard query is removed.
  The eight parent gates pass. Independent review passes native coordinator,
  source and pending-query controls (13 tests), installed bindings (two tests),
  two real Manager-process DRAM/io_uring SSD byte cells, enabled vLLM SSD/kernel
  handshake serving and default SGLang 0.5.21 DRAM/direct serving. Each byte cell
  restores a distinct exact 8,192-byte common prefix per node on one H20 GPU;
  parent and reviewer together retain four byte cells.
  The seven parent/reviewer installed-model cells contain 59 stored API responses
  and 38 exact token/text controls. Full/partial restores and vLLM reset-generation
  reuse each load 113,246,208 bytes; SSD cells prove matching completed io_uring
  reads after DRAM eviction. Enabled model cells require the actual released
  handshake log and positive native submission counter. Final coordinator,
  source-interest and applicable query/SSD drain gauges are zero.
  The original standard-gate launch failed before collection because the reviewer
  controller omitted its declared plugin environment. Its immutable record stays
  INVALID_ENVIRONMENT. A separately authorized, frozen controller-only follow-up
  passes standard vLLM correctness (six tests, one dense-inapplicable skip), with
  24 stored texts and 12 exact-text controls; these are not raw-token controls.
  Source-only Python passes 338 tests with one skip. Production-head CI passes
  all 13 jobs. The retirement-to-Busy regression has preserved unpatched failures;
  the repaired control also preserves an unrelated earlier live operation.
  All 145 native/harness frozen inputs and installed engine/OrbitKV RECORD files
  stay unchanged. Parent inputs did not freeze checkpoint assets. Independent
  review separately freezes all 51 Qwen3-8B assets at revision
  `b968826d9c46dd6066d109eabc6255188de91218` and verifies them before/after;
  this does not retroactively add a parent checkpoint freeze.
  The 43 structured outer-service cleanup records report exit zero without
  helper escalation; the standard Manager separately records cached exit zero.
  All 44 declared owned process groups, 49 TCP ports, 20 UDS paths and GPU
  postflight are empty. Native SGLang teardown/IPC warnings, the standard native
  vLLM EngineCore force-kill and incomplete standard Manager signal tracing remain
  recorded. These observations do not prove every child exited normally or S3
  physical DMA termination.
  This experimental profile requires trusted same-host literal loopback and
  equal contiguous dense TP-only shards, V1 and one attention group. One-Manager
  query control is opt-in. Unsupported parallel, hybrid/recurrent, speculative,
  full-prefix waiting and queue-preparation paths are rejected. Physical TP8,
  multi-GPU NCCL, containers, cross-host protected transport/HA, native P/D fault
  handling and performance advantage remain unqualified. Earlier static-adapter
  and Graph acceptances do not qualify their composition with this profile.
  Full S5 remains Partial. Frozen parent, failed candidates, raw independent
  runs, controller correction and scoped final review are external at
  `/root/orbitkv-artifacts/s5-native-shard-query-20261009/reviewer/final-1ab67a58/FINAL-REVIEW.md`.
- **Qualification open:** Full + SWA + temporal recurrent native serving, page
  reuse under preemption, cancellation, engine restart and each advertised graph
  mode. Exact synthetic GPU recovery does not establish native model support.

**Acceptance:** no private runtime replacement remains in the claimed upstream
profile; all-state readiness, GPU ownership and resident-prefix behavior pass.
Unreleased required fixes keep only their dependent profiles experimental.

### S5.4 — Official native P/D and independent cache

**Release-only cutover implemented; bounded same-host model gates locally pass;
independent acceptance and native fault qualification remain open.**
Use official vLLM 0.31.0 and SGLang 0.5.21. Live vLLM P/D uses NIXL and
MultiConnector; SGLang uses native disaggregation. Manager shared-cache traffic
continues to use TENT. There is no maintained engine fork runtime.

On 2026-10-07 the frozen installed CUDA 13 wheel `485d0f8c…` passes the
vLLM save-only-D NIXL/MultiConnector gate and the SGLang native Mooncake/TCP
gate on one A100, dense Qwen3-8B, TP=1/PP=1 eager, with independent ordinary
cache on `direct`. Each gate compares exact monolithic output, reuses real
cache bytes after normal engine restart and cancels an already-running decode
request; SGLang additionally observes native retraction and a restored boundary
including completed decode tokens. Installed engine/OrbitKV files are unchanged,
with no GPU compute processes left. The SGLang test explicitly uses BF16 and
starts P through readiness before D: concurrent loading otherwise contaminates
the released engine's device-global memory estimate. Failed setup/startup and
busy-GPU preflight attempts remain immutable beside the passing cells at
`/root/orbitkv-artifacts/s5-native-pd-20261007/`, mirrored under
`/workspace/orbitkv-s5-native-pd-20261007/`. These local gates do not prove
cancellation during transfer, partial-submit, delayed ACK or safe page reuse
without the S3 lifetime contract.

- Preserve retirement of custom connectors, handshake/proxy and partial-tail
  cache extensions, plus unselected query-lease release and valid save ranges.
  Remove the fork-only TENT adapters, observation callbacks and runtime gates.
- Candidate composition: P reads/writes the independent cache; D uses save-only
  cache and native P/D owns incoming writes. Test other cache-selection orders
  separately. Python owns engine callbacks/layout/events, Rust owns cache
  transfers, batching, leases and resource lifetime.
- Integrate the accepted main Manager baseline; keep pending Traex substages
  separate. Freeze the new wheel,
  run official-engine cold/warm/restart/failure DRAM/SSD and shared-cache gates,
  and verify installed engine files before/after.
- Require real native P/D output, cancellation, preemption/retraction, restart,
  partial-submit, delayed-ACK and page-reuse evidence before opening each profile.
  Model output alone does not close native transfer lifetime qualification.
- Preserve fork patches/tests at immutable commit `9aee895e` and prior evidence
  under `/root/orbitkv-artifacts/native-pd-cutover-20260930/HANDOFF.md` as upstream
  material. They do not establish official release support.
- Current reproduction and limits: [P/D setup](pd.md). Historical evidence remains under
  `/root/orbitkv-artifacts/release-native-pd-20261001/`; the 2026-10-07 gates
  use the separate archive above. Cross-host fault domains,
  RDMA, hybrid/rank combinations and native GDS remain unqualified.

**Acceptance:** independently reproduce the claimed released-engine and frozen
wheel gates. Open profiles only with proven ownership and fault handling; no
fork-only factory, callback, default configuration or CI dependency may remain.

### S5.5 — Deployment matrix and upstream maintenance


**Physical two-H20 ordinary serving independently accepted (2026-10-10):**
benchmark `efbe8323` consumes unchanged production `1ab67a58`, its frozen wheel,
official vLLM 0.31.0 and SGLang 0.5.21. All four dense Qwen3-8B TP=1/PP=1 eager
DRAM/io_uring SSD cells pass: 80 profile restores, four consumer-restart restores
and four controlled empty-source recomputations. Complete generated text and
native token counts match fresh source controls; remote and H2D byte totals agree,
source SSD reads are exact, physical NIC counters advance in the required
direction and exported ownership gauges drain. Required service leaders exit
zero with complete reap and no harness forced cleanup; etcd's literal SIGTERM
exit is retained separately. Installed inputs and final GPU/ports/sockets are
independently verified. The network destination is consumer DRAM, followed by a
local GPU copy; this gate does not measure remote GPU-buffer RDMA.

The original cohort remains `INVALID_FAIL_STOP`: its SGLang profile passed, but
an external controller incorrectly required a leader exit of -9. A separately
frozen controller remediation requires leader zero and records native child
termination separately. It changes no engine, adapter, wheel or Manager bytes.
The new four-cell cohort is independent evidence, not a replacement or relabeling
of the original. Eight SGLang CUDA IPC producer warnings remain in the accepted
handoff; normal drain and final GPU zero do not close S3 lifetime proof. No HA,
native GDS, cross-host TP, stable p99 or matched performance advantage is claimed.
Full S2/S5 remain Partial. Inputs, both cohorts, raw evidence and reviews are at
`/root/orbitkv-artifacts/s5-rdma-serving-20261010/`.
See [the serving methodology](shared-cache-qualification.md#official-engine-cross-host-serving-restores)
and [the reproduction driver](../benches/README.md#first-and-repeated-cross-host-serving-restores).

**SSD peer read-batch measurement implemented; qualification blocked
(2026-10-10):** frozen harness `ab23dd35` consumes unchanged production
`1ab67a58`. Only the consumer's existing ordinary `--query-read-batch` changes
from 32 to 128 MiB; source settings, byte budgets, reader threads, lookahead and
official engine releases remain unchanged. The external controller stops at the
first SGLang cell: its 50 restores pass, but it closes the source Manager before
the source engine unregisters its context, producing a `Broken pipe` traceback.
The frozen shutdown guard correctly marks that cell `INVALID_FAIL_STOP`; the
remaining nine cells are not launched. The frozen controller labels the ten
preceding vLLM cells PASS, but independent raw-log review also finds source
unregister tracebacks and one consumer `EngineDeadError` during shutdown. Leader
exit zero does not establish worker cleanup; those lifecycle labels are not
accepted. All 550 retained rows have exact native outputs and remote/H2D/SSD
bytes, with actual authorization/READ counts reduced from 3/6 to 1/2 for the
72/144 MiB payloads. Both hosts finish with zero exported ownership and GPU use.

Accepted offline analysis `6b356f2d` compares native token counts without
requiring process-specific SGLang timing/UUID metadata to match. It rejects this
incomplete campaign; valid vLLM rows are retained without subset qualification.
A vLLM pair also exceeds the frozen 5% TTFT regression guard. A separate
CPU-only controller remedy checks that both engines unregister while their
Managers remain available and rejects worker shutdown errors; it does not repair
or relabel the hardware cohort.
The 128 MiB setting is not promoted; 32 MiB remains the comparison baseline,
while the production CLI default stays `0` (demand baseline). Full S2/S5,
isolation and owner support boundaries remain unchanged. Frozen
inputs, failures, raw logs, postflight and reviews are outside Git at
`/root/orbitkv-artifacts/s5-ssd-peer-batch-20261010/`.
See [the comparison methodology](../benches/README.md#independent-ssd-read-batch-comparisons).

**Serving shutdown validation accepted for CPU software; runtime qualification open
(2026-10-10):** the reusable benchmark now consumes the controller's required
engine-before-Manager stop order and native shutdown error checks. The offline
read-batch comparator validates all four stop records independently of cell PASS
labels. Single-node benchmarks signal the engine leader, wait for its process
group and reject early/nonzero exits, forced cleanup, unreaped workers and missing,
truncated or failed native shutdown protocols before emitting a summary. Launch
requires matching PID/proc namespaces; the owner adopts orphaned workers and
retains cleanup ownership across process-inspection failures. CPU parent/child
controls verify leader-only SIGTERM, actual child reap and inspection-error
cleanup with and without a primary workload failure; these
controls do not repair or relabel the retained hardware cohort.

The consumer vLLM log matches the intentional-shutdown ordering reported in
upstream [issue #48745](https://github.com/vllm-project/vllm/issues/48745).
[PR #49000](https://github.com/vllm-project/vllm/pull/49000) remains open as checked
on 2026-10-10; the same engine-before-handler shutdown order exists in selected
vLLM 0.31.0. This is an inference from the source and log ordering, not proof that
all `EngineDeadError` occurrences are harmless. Keep the official installation
unchanged and the benchmark's failure check active. Evidence for this software
substage is outside Git at
`/root/orbitkv-artifacts/s5-serving-shutdown-20261010/`.
Independent review accepts commit `5a30417a` for this CPU
software substage: 45 targeted and 364 benchmark tests, six real process-fault
controls and optimized-Python rejection of all eleven retained invalid cells.
The initial rejection and final review remain preserved with their raw controls;
no new GPU run or physical lifetime qualification is implied.

**Initial ownership export and standalone Manager readiness independently accepted
(2026-10-10):** `2c9d4b73` initializes ordinary query, Publish, peer-transfer and
SSD ownership instruments, plus the load-byte counter, inside the existing
metrics singleton. Real SDK/Prometheus tests verify initial zero, active values,
repeat singleton access and release; restoring the original source fails the
same missing-series regression. Independent CPU review accepts this producer
fix. A complete repaired CUDA 13 CPython 3.11 wheel passes isolated native import.
The fresh two-cell diagnostic generates two matching native responses, but
official vLLM 0.31.0 logs an `EngineDeadError` matching the upstream shutdown
order and
the guard stops it before starting any Manager or cache request. Independent
review accepts the retained invalid evidence, not engine shutdown or cache
qualification.

The separate standalone smoke using that complete wheel passes: all ten baseline
instruments have their required Prometheus types and exact zero values in the
first raw response and pre-stop snapshot. The literal Manager and executor exits
are zero with actual reaping, no forced cleanup and empty GPU/port/UDS postflight.
All 66,583 official engine files, 127 private installed files and 97 ELF closure
paths remain unchanged. Live Manager identity/maps are retained. Independent
result review accepts first-export readiness and bounded idle shutdown only;
real ownership allocation/drain remains unqualified. It starts no model or cache
requests and cannot complete or retry the failed two-cell
shutdown contract. The first Manager-only run stops before reading metrics at a
TENT-presence check that does not apply to standalone mode: the frozen source
constructs its transport only when a global index exists. Its literal Manager
exit is zero, while the diagnostic exits one and remains invalid. A separately
frozen standalone contract instead requires no native transport mapping; native
input/ELF hashes stay checked without claiming TENT execution. Its frozen
inputs, unlaunched rejection, invalid run, raw controls and
reviews remain outside Git at
`/root/orbitkv-artifacts/s5-initial-ownership-metrics-20261010/`.

Resolve native and consumer vLLM shutdown on an unchanged official release
before a separately frozen full read-batch cohort; initial exporter acceptance
and CPU ordering controls do not establish that repair. No retry or replacement
cell belongs to the stopped campaign. A passing serial
component still needs matched pressure and cancellation/drain controls before
production promotion, followed by matched official engine/backend comparisons.
Existing peer lookahead remains the baseline. Authorization/materialization
timers alone do not justify a CUDA kernel change; require device-event DMA/kernel
evidence before a KDA candidate.

**Matched eager controls and service-exit evidence implemented (2026-10-10);
model comparison qualification open:** the installed benchmark now consumes
the official eager configuration of both selected releases and records the
execution policy. Service cleanup exports the actual reaped leader exit code,
forced-kill status and interruption separately. Benchmark contract tests pass;
the dedicated Forge H20 controls pass for native HBM, native CPU offload/HiCache
and OrbitKV on both selected releases, plus vLLM LMCache 0.5.5. All 21 successful
request outputs match the cold references and matched backends; post-run installed
file hashes remain unchanged. SGLang LMCache startup is blocked by its missing
released `lmcache_mp_metadata` module; the existing upstream MP fix is not in
0.5.5. Each cell's cleanup retains literal service exit codes and reaped workers:
That earlier SGLang readiness cohort records leader exit -9 and native process
tree termination without a harness SIGKILL; it does not establish healthy
shutdown. The later cross-host gate above requires leader zero. These are
readiness controls with one sample
per phase, unqualified sampler overhead and possible shape/JIT warmup; formal
order-alternated serving comparisons remain open. Inputs, failures, raw cells
and postflight evidence are at
`/root/orbitkv-artifacts/s4-serving-cold-read-20261010/`;
independent review remains open. These measurement changes do not change
production cache policy, transfer backend defaults or S2/S5 qualification.

**Serving comparison readiness implemented; runtime qualification open
(2026-10-09):** `benches.single_node --installed-artifact` selects installed
packages and the bundled Manager, rejects a missing/source-shadowed OrbitKV
wheel and consumes the wheel's SGLang plugin registration. Source developer
mode remains explicit by omitting the flag. Benchmark tests cover both engines
and rejected package origins. This changes the measurement launcher only.

The next comparison uses each official selected engine separately: native HBM,
native CPU offload/HiCache, OrbitKV and the LMCache 0.5.5 MP target, with matched
model, GPU/host budgets, requests and independent order-alternated runs.
Fresh official environments and immutable Qwen3-8B artifacts on the two H20 hosts
are frozen and have the scoped startup/output controls above. Formal warmup,
sampler overhead, repeated run order and sustained serving gates remain open.
Local legacy environments at `/root/orbitkv/.venv/{vllm,sglang}-release` resolve to
0.29.0/0.5.20 and must not be relabeled as 0.31.0/0.5.21 evidence. The sustained
decode-time-per-token field is an aggregate proxy; per-token ITL distributions
and SLO goodput instrumentation remain open. Distributed LMCache needs
a consumed released backend profile and equal total cache budgets. TENT cold/warm,
GPUDirect RDMA and same-host NVLink diagnostics are separate S4 controls; no
transport bug or end-to-end advantage is inferred from a microbenchmark.
Use [the serving method](benchmark-evidence.md#serving-comparisons). This readiness
work does not change S2/S5 status or qualify any new engine/topology profile.

**TENT allocation triage (2026-10-10):** the independent CUDA API probe passes on
both H20 hosts and confirms that valid VMM memory requires its dedicated
shareable-handle API; legacy IPC export rejects it. No TENT payload defect is
reproduced by that probe. Existing upstream issue #2511/PR #3538 covers a
different DMA-BUF registration defect. Our official `v0.3.13.post1` pin also
lacks merged NVLink fixes #3678/#3679; peer ordinal and suballocation-handle
controls are required before local NVLink qualification. Retain separate
released/upstream-main evidence and preserve the current frozen runtime. See
[the allocation and upstream limits](benchmark-evidence.md#tent-and-fabric-diagnostics).

**Local gates passed; independent review open (2026-10-07):** test commit
`06d00d53` adds a reproducible two-engine shared-Manager gate for official
vLLM 0.31.0/SGLang 0.5.21. One A100 runs dense Qwen3-8B, TP=1/PP=1 eager,
a 1 GiB Manager pool and optional 2 GiB io_uring SSD. Each framework has its
own native output control and installed environment. Both media pass concurrent
cold/HBM requests and each engine's graceful exit/restart while the survivor
continues serving through the same Manager PID. Each restarted engine loads
113,246,208 bytes; SSD recovery has zero Manager DRAM beforehand and the same
positive physical io_uring byte delta. All 14 services across these two cells
exit zero without forced cleanup. Installed engine/OrbitKV hashes stay unchanged.

The production wheel remains the S5.2 freeze `9d14998d…`; no Manager, engine,
cache strategy or transport code changes. The test helper is shared with the
ordinary full/partial installed-wheel gate. RECORD validation now covers all
package data/code, with only generated Python bytecode exempt; the engine pin
checker parses `project.optional-dependencies` as TOML, including Python 3.10.
The first SSD attempt stopped at a test-only positive-eviction assertion when a
second cleanup was unnecessary; its logs and complete frozen cohort remain
archived. The corrected gate records every metrics boundary and accepts an
already-empty DRAM only while still requiring positive SSD reads. External
source, commands, raw runs and handoff:
`/root/orbitkv-artifacts/s5-shared-manager-20261007/`, mirrored under
`/workspace/orbitkv-s5-shared-manager-20261007/` on A100. Use
[the maintained reproduction](../python/tests/README.md#two-engines-sharing-one-manager).

A follow-up at `9ee74c90` passes overlapping external restores in both media:
226,492,416 bytes in each pair, with exactly matching physical SSD reads after
DRAM eviction. Both engines drain before a normal Manager restart; a fresh
Manager PID performs cold reconstruction and both engines continue to match their
native controls. All 24 services exit zero, with no forced cleanup or owned/GPU
processes left. This does not claim persistent SSD-index recovery, abrupt
Manager death or native DMA termination. The wheel is unchanged (`9d14998d…`).
Evidence: `/root/orbitkv-artifacts/s5-concurrent-recovery-20261007/`;
A100 mirror: `/workspace/orbitkv-s5-concurrent-recovery-20261007/`.

The SSD host-batching follow-up (`50c6d58d`, frozen wheel `8acd6a06…`) passes
three fresh installed gates: each official engine's ordinary SSD full/partial
restore, plus shared-Manager SSD serving. Both engines use `direct`; each full
or partial restart loads 113,246,208 bytes from SSD after DRAM eviction. Their
overlapping external restores load 226,492,416 bytes, with matching io_uring
reads and four reader batches. All 20 services exit zero without forced cleanup;
all installed package/engine files stay unchanged and GPU postflight is empty.
Normal Manager restart remains a cold rebuild. Functional release-gate files
are on the `/workspace` overlay; the separate 12-cell native comparison uses
ext4/NVMe and demonstrates median improvement, with descriptive p99 regressions
retained. The first aggregate cohort has incomplete provenance; the new complete
raw cohort and all surviving controls are archived at
`/root/orbitkv-artifacts/s5-ssd-host-batching-20261007/`, mirrored at
`/workspace/orbitkv-s5-ssd-host-batching-20261007/`. Independent review and
sustained mixed save/restore tail/fairness qualification remain open.

The bounded S5.5 pressure harness is implemented at `7deb9769`, with strict
primary-error preservation at `36433e55`. `benches.shared_manager_pressure`
uses two installed official engines, per-instance request quotas, 512 MiB shared
DRAM, 8 GiB io_uring SSD, 384 MiB global/192 MiB configured per-instance query
budgets, fresh/reused inputs, native controls for every output and an independent
resource sampler. CPU benchmark contracts pass (246 tests); the default
source-only Python gate passes 273 tests with one skip. The initial 15-second
direct smoke passes 44 requests, physical SSD reads/writes, sampled global/pool
budgets, unchanged installed files and eight normal zero exits. A fresh untraced
file-lifetime diagnostic also passes 43 requests; its short tails are descriptive.
Those frozen cells predate the owner peak metrics recorded below; they do not
measure per-instance reservation peaks.

**Sustained qualification blocked:** two untraced launches fail in the SGLang
client's iceoryx2 `DoesNotExist` discovery before observer start and foreground
round zero. The 900-second direct cell is invalid; its kernel cell is not started.
A complete traced kernel profile passes 85 requests and supplies the actual save
fragment histogram, while idle and registered-GPU delayed joins also pass. These
diagnostics do not establish a cause or a production repair. The SGLang-only
trace wrapper requires forced cleanup and is retained as an invalid diagnostic.
Do not inherit a sustained tail/fairness or backend-advantage claim from any of
these short cells. The frozen task, KDA rejected candidate and complete failure
archive are at `/root/orbitkv-artifacts/s5-shared-pressure-kda-20261007/`, mirrored
under the matching `/workspace/` directory on A100. Existing S3/public-lifecycle
dependencies remain unchanged.

**IPC diagnostics implemented; repair and qualification open (2026-10-08):**
failed request-service and `/requests` event discovery now report the actual
service name, messaging pattern, effective static-config path, file metadata and
mount namespace. This executes only after a failed open; normal submission and
service ownership are unchanged. The channel gate passes 62 unit tests (one
private child fixture ignored at top level), six completion/ownership tests and
one real process round-trip; targeted Clippy and format checks pass.

A single predeclared A100 file/lock-lifetime diagnostic reuses the existing
`8acd6a06…` installed wheel, not these new error messages. Both official engines
start and 87 requests pass exact native input/text/prompt/completion-token
oracles during a 30.506-second window. All eight services exit zero without forced
cleanup; the recorded request/event service files are deleted after Manager
endpoint stop. It does not reproduce the discovery failure or establish a repair.
The complete 262-file artifact manifest is verified locally at
`/root/orbitkv-artifacts/s5-ipc-startup-20261008/a100/diagnostic-v7/` and on A100 at
`/workspace/orbitkv-s5-ipc-startup-20261008/diagnostic-v7/`.

The failure-context wheel built from `1ea56c2e` subsequently passes package repair,
isolated native import and both copied official environments' Manager CLI checks.
Its single untraced startup diagnostic reproduces `DoesNotExist` in vLLM 0.31.0:
the worker registers 36 GPU layers, then the same process's scheduler client cannot
open the matching advertised request service. Its actual static-config path is
absent in the caller namespace (`ENOENT`); the Manager stops 294.416 seconds later.
This broadens the observed failure beyond SGLang and rules out prior Manager exit
for this run, but does not identify the deletion actor or a production repair.
No cached SGLang launch or foreground window starts. The cached vLLM exits one,
the Manager and two native controls exit zero, and no forced cleanup is used.
All five recorded PIDs and the UDS are absent; GPU compute is empty. All installed
candidate/engine hashes and the original frozen environments remain unchanged.

The failed cell, static-Python build failure and successful shared-Python build
are retained at `/root/orbitkv-artifacts/s5-ipc-startup-20261008/`. The corresponding
A100 failure-context evidence is
`/workspace/orbitkv-s5-ipc-startup-20261008/failure-context-1ea56c2e/`; its 250-file
cohort manifest is verified locally, excluding installation directories whose
files are verified separately against installed RECORD hashes. The production
commit's CI run `37720474129` passes all 13 jobs. The next S5.5 action is to inspect
node/service ownership and discovery-file lifetimes around GPU registration and
a subsequent client attach, establish the cause and independently review a repair,
then freeze a new sustained cohort. No diagnostic upgrades the support envelope.

The subsequent single syscall-attribution diagnostic also does not reproduce
the failure: 86 requests match exact native input/text/token controls, eight
services and three tracers exit zero, and installed files remain unchanged.
Its selected file/lock traces do not identify a deletion actor; tracing may
affect scheduling. The earlier failure remains unresolved. Complete evidence:
`/root/orbitkv-artifacts/s5-ipc-causal-20261008/`.

A separate three-cell untraced startup diagnostic completes with the previous
`485eec94…` wheel. Each uses readonly process/service/lock observation, native
output controls and a 30-second shared-engine window. All byte/text/token
oracles and normal cleanup pass. The observer never opens process-monitor lock
files; its own overhead is unqualified. None reproduces the missing-service
failure, so no deletion actor or production repair is established and sustained
qualification remains open. Evidence: `/root/orbitkv-artifacts/s5-finish-20261008/`,
mirrored at `/workspace/orbitkv-s5-finish-20261008/`.

**Budget peak observation implemented; local boundary gate passed (2026-10-08):**
the existing QueryBudget admission owner now retains lifetime global and
maximum-single-instance concurrent reservation peaks under its existing lock.
Observable metrics retain peaks through release without instance labels or a
second state owner. The pressure gate requires both peaks after drain, checks
the configured global/per-instance limits and rejects a reset or missing value.
This covers short reservations between sampler ticks; it does not identify an
engine's own maximum, measure physical pinned/HBM allocation or qualify fairness.
The frozen CUDA 13 wheel from `c3cf15a9` passes one new A100 dense eager
shared-engine/io_uring boundary gate: all 87 exact native input/text/token
controls pass, both engines make bounded progress, all seven Manager/engine
services and the sampler exit zero, and installed files stay unchanged. Global
and maximum-single-instance peaks are 217,055,232 and 113,246,208 bytes, below
the configured 384/192 MiB limits. The final query/SSD ownership drains to zero.
The 30.542-second window does not qualify sustained tails or fairness; independent
review, the earlier unresolved startup failure and all broader limits remain open.
Evidence: `/root/orbitkv-artifacts/s5-budget-peaks-20261008/`, mirrored on A100
under `/workspace/orbitkv-s5-budget-peaks-20261008/`.

**H20 sustained ordinary-cache pressure independently accepted (2026-10-08).** The unchanged `fe427c42…` wheel and frozen
`e1e42d17` measurement program complete a 900.210-second direct
window (2288 vLLM/1876 SGLang requests) and a
900.323-second kernel window (2252 vLLM/1914 SGLang).
Both use dense Qwen3-8B, TP=1/PP=1 eager, 512 MiB DRAM, 8 GiB io_uring SSD
and 384/192 MiB global/per-instance query budgets. Every request matches its
exact native input/text/token-count oracle, the actual worker routes are logged,
both engines maintain bounded progress, and owner reservation peaks stay below
both limits. Seven services and the sampler per cell exit zero; installed
distribution hashes stay unchanged and GPU postflight is empty. The SSD file is
on the container's overlay filesystem: this does not establish physical NVMe
performance. Seeds differ by route; no kernel-versus-DMA advantage, isolation
or fairness claim follows.

The test-only remediation `55ad0343` requires all six final io_uring drain
gauges to exist and equal zero. The shared wait helper retains observed lazy
gauges and rejects missing, disappearing, negative or nonfinite values; cuFile
staging is required only if observed, and is not reported as measured zero on
this io_uring route. Both original frozen captures pass the stricter validator;
no cell is replaced or retried. Default Python/benchmark contracts pass
283/264 tests. The earlier intermittent discovery failures remain unresolved
and immutable. Full inputs, outputs, failures and scope are at
`/root/orbitkv-artifacts/s5-h20-closeout-20261008/`.

Independent review accepts production `e1e42d17` and the test-only `55ad0343`
drain remediation. Both official environments pass independent CUDA/IPC checks,
four installed DRAM/SSD model cells and four synthetic GPU byte/lease controls.
Synthetic controls do not export structured Manager exit codes and do not prove
native recurrent serving or S3 termination. The reviewer verifies all 808
primary artifact hashes and independently reconstructs all 8,330 raw request
oracles. After an unexplained reviewer `/tmp` evidence loss, original reports,
controller read-back and results were recovered from recorded tool executions;
pre-existing report hashes match. Some reviewer raw model/log/probe files remain
missing and are not fabricated. Scoped acceptance uses those original execution
records and the intact primary raw cohort, without retry or replacement. The
report is `/root/orbitkv-artifacts/s5-h20-closeout-20261008/reviewer-evidence-recovery-20261008/FINAL-REVIEW.md`;
future campaigns archive
directly outside `/tmp`.

This independently accepted cohort delivers one same-host deployment profile.
Matched tail isolation/fairness, isolated containers,
physical cross-host/rank combinations, native P/D faults and S3-dependent
reclamation remain open. Shared capacity does not establish interchangeable
cross-engine KV bytes or a latency/throughput advantage.

**Two-host Manager path timing (2026-10-09), implemented and measured;
independent review open:** the bounded measurement uses the installed
`1ab67a58` production freeze on two
dedicated Forge H20 hosts. `benches.shared_cache_profile` separates a first
remote DRAM read from warmed repeated restores at 512 KiB/8 MiB/32 MiB, both
directions and single/four-NIC allowlists. Every sample clears consumer DRAM and
requires exact remote/H2D bytes, source release and GPU tensor/sentinel equality.
The source grants, TENT READ and reconstruction remain owned by the Rust
Managers. Client query/restore timing is separate from GPU clearing, validation
and controller HTTP; no isolated DMA or model-serving timing is inferred.
The frozen `68098007` harness completes all twelve size/direction/NIC cells,
with 432 exact whole-tensor byte oracles (12 cold, 60 warm-up, 360 measured).
Warm Manager fetch medians range from 1.087 to 1.977 ms; cold fetches range from
84.151 to 208.610 ms, primarily in the TENT READ stage. The client fixture's
10 ms query polling is reported separately and is not engine latency. Runtime
inputs stay unchanged on both hosts; all 24 Managers exit zero, workers release
their contexts, etcd shuts down normally, and ports/sockets/GPU use drain.
The executing agent reconstructs reference hashes, metric/NIC deltas and summary
statistics; this does not substitute for independent acceptance. Functional
readiness and descriptive timing evidence are at
`/root/orbitkv-artifacts/forge-rdma-gds-20426175-20261009/manager-dram-profile-20261009/REPORT.md`,
with complete raw host mirrors and the frozen plan beside it.
This descriptive timing substage does not launch a formal
isolation/HA campaign, expand support, close S5 or change production policy.
**First native READ stages measured (2026-10-10), independent review open:**
`benches.tent_stage` uses the same frozen TENT libraries with private CPU mmap
buffers. Ten fresh consumers across both directions complete 110 full 8 MiB
payload oracles; twelve endpoints exit zero and their recorded PIDs disappear.
Cold READ is 165.613–211.246 ms, with more than 99% between submit return and
terminal observation. Warm READ medians are 0.181/0.187 ms. The cheap open call
does not exclude lazy metadata discovery, QP bootstrap or worker scheduling
inside the completion interval. CPU buffers, polling and n=5 per direction do
not establish model latency, a transport defect, tail qualification or GPU-direct
destination recovery. The final source/consumer control protocol additionally
passes a six-READ byte/drain smoke; all native libraries remain unchanged.
Raw cohorts and prior preparation failures are at
`/root/orbitkv-artifacts/s4-serving-cold-read-20261010/`.

The bounded native decomposition and rejected RPC-thread/lane controls are
recorded in S4 above. The next first-recovery candidate needs a consumed explicit
connection-warming interface, accounting for its setup cost and fabric resources;
the upstream proposal is still open. Source authorization, buffer registration
and physical drain remain with the existing owners. Require matched first-recovery
and contention controls before promotion; do not use a dummy payload READ to
unowned addresses or omit warming cost from an end-to-end claim.
CUDA optimization requires separate device-event DMA/kernel measurements and a matched KDA
comparison; client restore timing alone does not establish a kernel bottleneck.
See [the maintained methodology](shared-cache-qualification.md#full-manager-remote-dram-path-timing).

- Track each engine separately: single instance; independent replicas; multiple
  instances sharing a Manager; dense/hybrid; P/D; attention DP, TP, PP, MoE EP and
  relevant combinations; homogeneous/heterogeneous P/D parallelism; same-host TCP,
  physical two-host TCP and RDMA. Label implemented, qualified, experimental and
  unsupported independently. Do not infer KV partitioning from EP world size.
- **Implementation open:** protected cross-host activation of the existing
  native Manager query fan-out, accepted on trusted loopback above.
  The engine still owns rank agreement, legal common recovery boundaries, GPU
  allocation and execution collectives. Define stage-specific PP state/layout;
  reject unsupported resharding and state mappings explicitly.
- Qualify source incarnations, cancellation, sender/receiver budgets and transport
  failures on real hosts. RDMA needs visible NICs and physical counters; multiple
  ranks/processes on one GPU do not qualify multi-GPU or host-failure behavior.
- Qualify simultaneous engines, Manager/engine restart, shared DRAM/SSD budgets,
  containers, UDS/iceoryx2/CUDA IPC, PID visibility and resource drain. Shared
  service support does not establish cross-engine interchangeable cache bytes.
- Maintain upstream registration/docs/tests and a release upgrade gate. Submit
  lifecycle fixes and TENT support separately from the baseline cache entry.
  Record review links, maintainers, merge commit and first containing release.

**Acceptance:** an executable scenario with output, physical-transfer and drain
evidence for every claimed cell. The minimal upstream cache PR need not wait for
all topology/research cells; no untested cell becomes supported by that merge.

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
  engine-process P/D WRITE; Manager-only admission misses the latter. Lease
  bounded credits in batches while keeping payload movement direct through TENT.
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
