# Preparing queued requests

For artifact locations and verification limits, see [benchmark evidence](benchmark-evidence.md).
The [pre-migration report](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/queued-warming.md) retains full tables and historical run details.

The vLLM 0.30.0 and SGLang 0.5.20 adapters can announce an exact missing prefix when a
request enters their ordinary serving queue. OrbitKV uses that interval to
prepare DRAM pages from SSD or a peer. This is the first P3 implementation;
it does not predict future prompts or decide engine admission.

## Request lifecycle

```mermaid
sequenceDiagram
    participant E as Engine queue
    participant M as Cache Manager
    participant B as SSD / peer
    E->>E: Inspect existing HBM prefix
    E->>M: Warm missing page hashes
    M->>M: Try bounded warmup admission
    M->>B: Shared backing read
    B-->>M: Pages ready in DRAM
    M->>M: Release warmup reservation; keep pages evictable
    E->>M: Query current demand at scheduling time
    M-->>E: Revalidated prefix and restore lease
    E->>M: Restore into engine-owned GPU pages
    M-->>E: GPU completion
    E->>E: Consume restored KV
```

The vLLM connector uses `on_new_request` and its existing block hashes. It
checks the bound GPU block pool without allocating or pinning pages. Whole-page
warmup includes the final full page even when admission later recomputes its
last token; this lets identical demand queries join the same backing read.
Hybrid cache groups are excluded from this initial enqueue path.

SGLang uses its plugin hook around `_add_request_to_queue`. Only accepted
ordinary-queue requests are warmed. The adapter matches HBM with `req=None`,
then uses the linker's existing tail-hash calculation, preserving `extra_key`,
`cache_salt`, page boundaries and the logprob recovery limit. It does not create
an external hit marker or allocate GPU destinations. Positional embedding
overrides and disaggregated bootstrap queues are excluded.

## Capacity and cancellation

Warmups reserve their registered, padded page footprint against the existing
global and per-instance query budgets. Each warmup class is capped at **one
quarter** of both budgets. They also have limits of 16 active operations per
session and 128 globally, within the existing 128/1024 total limits. Byte or
operation pressure skips the hint immediately; an oversized hint is skipped
as a whole. A new hint also skips while any foreground query owns preparing,
ready or restoring bytes. Already-submitted reads still drain; this guard is
conservative admission, not preemption or a calibrated deadline scheduler.
These caps reserve accounting headroom, not a bandwidth or latency
guarantee for foreground reads.

A warmup never returns a hit promise or restore lease. Its reservation ends
when preparation completes, even if the engine never polls again. Prepared
pages enter the existing reclaimable cache class, which pressure evicts before
retained pages. Warmup hits use a non-mutating peek: they do not refresh existing
pages' recency or frequency. Foreground demand promotes the matching warmed
page generation to the retained class; a query/lease alone does not count as
successful use in the metrics below. They can disappear
before use. Ordinary admission always revalidates the current hashes, obtains
an independent lease and retains the normal GPU ownership rules.

Each client retains at most 16 pending hint tickets. Hints expire after five
seconds at the Manager; the client retires stale tickets when submitting
further hints. Admission or cancellation retires the matching hint before a
fresh demand operation. An already-submitted read drains under its original
reservation and operation permit. It is not aborted when interest disappears.
The ticket timeout does not expire resident pages; normal cache eviction does.
The existing backing-read coalescer can serve independent warmup/demand owners;
only identical read plans coalesce.

Set `ORBITKV_QUEUE_WARMUP=1` in the **engine** environment to enable automatic
enqueue warming. It is experimental and disabled by default: the initial
pressure controls did not improve throughput and increased SSD reads.
Normal query/prefetch and restore remain available with warming disabled.
The explicit `CacheManagerClient.warm_prefix()` API always attempts the supplied
hint; the environment switch controls automatic engine enqueue hooks only.
Transport failures in this optional enqueue hint are logged by the adapters
without raising out of the already-accepted request's queue callback.
Manager and Python extension must both use channel ABI 11; earlier ABIs are not supported.

## Observing the path

Set `ORBITKV_TRACE_TRANSFERS=1` for both Manager and engine. Their logs emit
`cache_timeline` JSON records containing request IDs, process IDs, stage and
timestamps, without tokens or cache keys:

[Full historical measurement table](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/queued-warming.md)

`first_use` is an engine callback observation, not a CUDA kernel timestamp.
SGLang records it for restored requests; cold/HBM-only requests have no external
layer wait. Python monotonic timestamps may be compared within one process.
Manager `elapsed_us` measures its own preparation interval. Wall timestamps
help inspect logs but are not a cross-host deadline or latency clock.

`orbitkv_query_reserved_bytes_by_phase{phase="warming"}` separates warmup reservations
from preparing/ready/restoring demand. A sampled peak is a lower bound. Existing
cache-tier query counters include warmup probes; use actual SSD/TE/H2D byte
counters to establish transfers rather than interpreting probe counts as
request hits.

## Measuring whether preparation was useful

The following counters follow the actual `SealedBlock` allocation, not a hash
or request ID. The owner that starts a shared read determines its origin:
joining a demand-started read or hitting existing DRAM does not create warmup
bytes. Joining a warmup-started read does not count them again. Rereading an
evicted key creates a new cohort.

[Full historical measurement table](https://github.com/feichai0017/orbitkv/blob/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/docs/queued-warming.md)

At quiescence, prepared bytes equal restored + unused + pending bytes. Query
success, lease creation and cancellation do not resolve a pending page. The
last-reference rule prevents an eviction from classifying a still-running
transfer as unused. Multiple layer/rank copies credit a page only once, after
GPU synchronization succeeds.

These are **page-footprint** counters. A partial-layer/rank transfer credits
the contributing page footprint; use `orbitkv_load_bytes_total` for actual H2D
bytes. A successful transfer does not prove the engine later consumed it, or
that warming saved latency. Remote serving alone is not local H2D use. Physical
pool usage can also include allocation padding and shared slab retention.

Byte-seconds settle only when an outcome occurs; live intervals are excluded.
Benchmark windows retain starting/ending pending bytes and a sampled peak, so
live pages and carry-in are visible instead of being called waste. A window's
restored/prepared ratio is not a cohort success rate when it includes carry-in.
The operation's quarter-budget limit ends at read completion; it does **not**
cap all prepared resident pages. Their replacement priority and the pinned
pool's physical limit govern subsequent retention.

Pressure reclaim stops each batch once the allocation's requested footprint
has been selected (still at most 512 pages), releases those references, then
checks the allocator's actual largest contiguous free region. Fragmentation or
shared slabs can require another batch. The previous count-only batch could
discard every eligible page in a small pool: a Qwen3-8B page in these controls
is 9 MiB, so a 4 GiB pool holds fewer than 512 pages. Byte-bounded reclaim applies
to ordinary demand and writes as well as warming; its effect must be separated
from the warming on/off comparison.

## Initial pressure controls

Measured September 21, 2026 at source `2c1b27e3`, using Qwen3-8B on one H20.
Both modes use the same build and tracing settings. Each engine runs a 30-second
admission window at concurrency 8, with 4,096 input tokens, 16 output tokens,
8,192 GPU-cache tokens, 4 GiB of host cache, a 3 GiB query budget and 16 GiB SSD.
Twelve reusable prefixes occupy 6.75 GiB of KV payload; the reuse probability
is 0.75. Admitted requests drain after the window, and throughput includes that
time. The SSD file uses `O_DIRECT`/`io_uring` on an overlay mount; this is not a
physical-NVMe qualification.

| Engine | Warmup | Requests | Requests/s | TTFT P50 / P95 (ms) | SSD read MiB/request |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM | Off | 126 | 3.933 | 2126 / 2750 | 254.8 |
| vLLM | On | 128 | 3.907 | 2050 / 2685 | 304.7 |
| SGLang | Off | 116 | 3.679 | 1997 / 3171 | 265.7 |
| SGLang | On | 116 | 3.606 | 2054 / 3179 | 427.2 |

The experiment demonstrates bounded preparation, not a throughput improvement.
All four runs drained query reservations to zero. Sampled warmup peaks were
576 MiB for vLLM and 567 MiB for SGLang, below the 768 MiB warmup limit.
SSD bytes per request increased about 20% and 61%, respectively. The small
vLLM latency changes are not evidence of a general gain from one trial. Automatic
warming therefore remains experimental and disabled by default. These results
motivated page-use accounting and the conservative admission policy measured below.

vLLM recorded no prepared-reference output differences. Both SGLang modes
recorded nine differences, all for prefix 11. A separate native SGLang probe
reproduced the same split: cold computation matched the prepared reference;
an immediate 4,032-token HBM hit matched all nine differing cached OrbitKV
responses. The one uncached OrbitKV response for that prefix matched native
cold computation. This reproduces the difference without OrbitKV and does not
establish batch-invariant output equality. Exact GPU-byte and engine recovery
gates remain separate correctness evidence.

The summaries, complete window counters, native output control and its
reproduction script remain in the
[historical dataset snapshot](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results).
The earlier raw samples use `historical run label: queued-warming-*`; those
ignored logs were absent at the S1 migration. Initial SGLang logs repeat the first-layer wait callback;
the collector pairs each enqueue with its first callback, and the adapter now
emits `first_use` once per restored batch.

Reproduce an experimental window from the repository root:

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /workspace/models/qwen3-8b \
  --workload sustained --concurrencies 8 --duration-seconds 30 \
  --max-requests 1000 --working-set 12 --reuse-ratio 0.75 \
  --lengths 4096 --gpu-tokens 8192 --output-tokens 16 \
  --host-gib 4 --query-budget-gib 3 --ssd-gib 16 \
  --queue-warmup on --trace-transfers \
  --output /var/tmp/orbitkv-bench/queued-warming-vllm-on
```

Use a fresh output directory for each run; switch `on` to `off` for the control.
Use `.venv/sglang-release/bin/python` and `--engine sglang` for SGLang.

## Page-use and reclamation controls

Measured September 22, 2026 at clean source `d40100a3`, with the same H20,
model revision, capacities, request sequence and tracing settings as above.
Both modes include byte-bounded reclamation and page-use accounting. Runs use
fresh services, in order vLLM off/on and SGLang on/off; all 484 requests complete.
This is one trial per mode on overlay-backed `O_DIRECT` storage. Differences
from the earlier build do not isolate reclamation from the other changes.

| Engine | Warmup | Requests | Requests/s | TTFT P50 / P95 (ms) | SSD read MiB/request |
| --- | --- | ---: | ---: | ---: | ---: |
| vLLM | Off | 126 | 3.939 | 2013 / 3114 | 224.1 |
| vLLM | On | 126 | 3.916 | 2024 / 3116 | 224.1 |
| SGLang | Off | 116 | 3.703 | 2073 / 3278 | 205.3 |
| SGLang | On | 116 | 3.739 | 2071 / 3250 | 288.4 |

| Engine, warming enabled | Prepared GiB | Restored GiB | Unused GiB | Foreground hint skips |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 0.563 | 0.563 | 0 | 119 |
| SGLang | 7.752 | 0.554 | 7.198 | 61 |

Pending warmup bytes start and end at zero in every window; prepared equals
restored plus unused. All query reservations and SSD operations drain to zero.
Sampled warming and pending peaks are 576 MiB for vLLM and 567 MiB for SGLang,
below the 768 MiB active-warmup limit. This experiment observes no pending
carry-in/out; the general accounting caveats above still apply.

vLLM admits very little warming under sustained foreground ownership. Its one
prepared prefix reaches H2D, but SSD bytes per request are unchanged and there
is no observed throughput gain. SGLang transfers only **7.1%** of its prepared
page footprint before release; **92.9%** becomes unused. Warming adds **40.5%**
SSD bytes per request while throughput differs by only about 1%. A single trial
does not establish a throughput or latency improvement for either engine.

The byte-weighted time from readiness to unused release is 0.44 seconds in
SGLang. Engine queue-to-first-use medians are about 1.6–1.7 seconds, while
observed H2D medians are about 20 ms for vLLM and 27 ms for SGLang. These are
different observation populations, not paired per-page critical-path timings.
They motivate using queue position, expected use time and retention pressure:
foreground query ownership alone does not say when the engine will consume KV.
An engine can be computing with no active query lease while later requests
remain queued. Async demand preparation already overlaps some waiting, so
moving reads earlier need not shorten exposed latency. Automatic warming
therefore stays disabled; a calibrated admission/deadline policy remains P3 work.

vLLM reports no reference-output differences. Each SGLang mode again reports
nine differences, all for prefix 11. Identical input tokens and all nine cached
outputs match the previously recorded native HBM control; the uncached output
matches native cold computation. No new native control or batch-invariance
claim is implied. The exact GPU-byte and both engine recovery gates pass on
this implementation.

The [historical accounting dataset](https://github.com/feichai0017/orbitkv/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
retains final summaries, complete window counters and the recorded native
output comparison.
Earlier raw samples use `historical run label: warmup-accounting-*`; those
ignored logs were absent at the S1 migration. Use the
reproduction command above with a fresh output directory and this source revision.

## Reference implementations and policy order

Source review: September 22, 2026. The following implementations inform the
next P3 changes; the proposed OrbitKV policies below are **not implemented**.
Source availability alone does not qualify another project's performance on
OrbitKV's engine releases or workload.

| Reference | Verified behavior | Application to OrbitKV |
| --- | --- | --- |
| [LMCache v0.5.5 prefetch controller](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/storage_controllers/prefetch_controller.py) | Request lookup transfers completed write reservations into reader locks; WARM finishes writes without retaining reader locks. | Keep demand-owned preparation distinct from optional warming. Reuse existing query/lease ownership when preparing an actual consumer's prefix. |
| [SGLang v0.5.20 HiCache](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/hiradix_cache.py) | Minimum prefetch size, rate limiting, and `best_effort`, `timeout`, `wait_complete` termination; completed results are clamped to a usable prefix. | Define when the engine stops waiting, and expose only a completed legal recovery boundary. Validate rank agreement separately. |
| [FlexKV replacement policies](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/docs/eviction_policy/README_en.md) and [transfer scheduler](https://github.com/taco-project/FlexKV/blob/738ddc141a198b4e20de6c5d1f0128e387f7fdb2/flexkv/transfer/scheduler.py) | LRU/LFU/SLRU and per-tier reclamation are separate from dependency-driven transfer graphs. | Compare retention policies independently from read admission; preserve completion dependencies across SSD, DRAM and GPU. |
| [Dynamo v1.4.2 KVBM offload](https://github.com/ai-dynamo/dynamo/blob/2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a/lib/kvbm-engine/docs/offload.md) and [onboarding](https://github.com/ai-dynamo/dynamo/blob/2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a/lib/kvbm-engine/docs/onboarding.md) | Offload filters precede batching; transfer commitment retains ownership. Session holders protect blocks from eviction until released. | Separate write admission from read prefetch and replacement; preserve transfer lifetimes. Keep Mooncake TE for bytes and request routing as a separate integration. |

LMCache's [warm-prefetch entry point](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/multiprocess/warm_prefetch.py)
explicitly leaves prepared pages unpinned. Thus, "pin every warmup" is not the
lesson from its request lookup path. OrbitKV already retains demand query
results through leases and GPU completion. The separate
[consumer-owned preparation experiment](request-preparation.md) now starts
selected required-range reads before admission and retains ready leases within
the query budget until claim, cancellation or expiry. It is opt-in and uses
the same bounded ownership rather than pinning an entire queue.

Two useful references have a different maturity status:

- [FlexKV #291](https://github.com/taco-project/FlexKV/pull/291), checked open
  and unmerged at head `fb7cc97d6723553688bd64ce0639d206ea3fb325`, proposes
  chunked prefetch, stopping new submissions, draining submitted work, and
  protected-result handoff. Treat it as a design proposal, not a shipped
  baseline or an established speedup.
- [Mooncake RFC #3504](https://github.com/kvcache-ai/Mooncake/issues/3504)
  proposes cached membership and peer route authorities outside the metadata
  hot path. It informs the distributed catalog plan, not a ready-made SSD
  prefetch policy. OrbitKV's etcd membership and Mooncake TE direction remains.

Implement and measure in this order:

1. **Bound demand preparation and retention.** Keep ordinary demand as the
   control. Where an engine exposes candidates close to admission, try a small
   byte-bounded lookahead using the existing query operation/revision and lease
   lifecycle. Account for in-flight buffers and prepared-but-unconsumed pages
   until consumption, cancellation or expiry. Preserve foreground headroom;
   unavailable budget falls back to normal demand. A consumer must acquire its
   validated page references before preparation ownership is released. HBM
   hits, request reordering, boundary changes and disconnects must retire stale
   interest. Queue knowledge must come from the engine; enqueue time alone is
   not an execution-time estimate. This is an experiment, not guaranteed gain.
2. **Add explicit stop policies and bounded submission.** Begin with
   best-effort at actual scheduling eligibility and a relative wait budget;
   retain wait-complete as a control with lifecycle expiry. Stop issuing new
   bounded read batches after cancellation/deadline. Submitted SSD/TE work
   drains while its buffers remain owned. Return only completed contiguous
   pages at a valid component/checkpoint boundary. Shared-read consumers have
   independent interest: cancelling one must not revoke another's work.
3. **Tune retention and write admission separately.** Compare the existing
   replacement classes with the optional Rust
   [reuse-based protected segment and selective SSD admission](cache-policies.md);
   do not promote
   speculative peeks as demand hits. Measure SSD write admission independently
   from read prefetch. A retained replica still needs normal pressure eviction;
   a live transfer's references remain protected. Model-specific recovery
   requirements always take precedence over replacement scores.
4. **Calibrate timing after the lifecycle baseline.** Only then add expected
   first-use estimates, bandwidth/queue cost and priority tuning. Use observed
   data to decide whether restoration beats recomputation in P4. Keep these
   decisions in the existing query/prefetch/storage owners, with engine hints
   supplied by the adapters.

Qualification compares demand-only, current optional warming, bounded
consumer-owned preparation, and then chunked stopping as separate steps.
Keep bytes, model, request sequence, tracing and hardware equal; repeat in
reversed order. Include pressure, duplicate prefixes, cancellation, queue
reordering, expiry without polling and an HBM-hit workload for hook overhead.
Require exact restoration and bounded resource cleanup before claiming lower
TTFT or higher throughput. Report read/write amplification, useful/unused
footprints, prepared-page residency and exposed wait, not just hit counts.

## Qualification and remaining P3 work

CPU gates cover foreground budget headroom, operation pressure, unpolled
completion cleanup, client cancellation and fresh demand tickets. The SGLang
pinned-release hook test checks salted keys, logprob boundaries and HBM skipping.
The GPU integration gate warms real SSD pages without polling, waits for query
bytes to return to zero, leases them through an immediate DRAM query and checks
exact restored GPU contents for both supported stored layouts.
For changes to enqueue warming, run both engine E2E commands from the
[test gates](../python/tests/README.md) with `ORBITKV_QUEUE_WARMUP=1`.

Use the [benchmark harness](../benches/README.md) with a working set exceeding
HBM and host cache. Compare `--queue-warmup on` and `off` at equal capacities;
`--trace-transfers` enables timeline capture. A quiet serial server may see no
benefit, and an overloaded backend may perform extra reads.

P3 follows the [reference-based order](#reference-implementations-and-policy-order):
bounded demand preparation/retention, explicit stopping and drain, then
calibrated priority/deadline hints. Per-device/staging reservations,
engine-consumption-level usefulness and fault qualification remain open.
The fixed warmup share is an initial admission policy, not a cost-aware scheduler.
Restore-versus-recompute decisions remain P4. See the
[implementation sequence](completion-plan.md).
