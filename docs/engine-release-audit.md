# Released engine interface audit

The runtime pins are official **vLLM 0.31.0** (`db9527a46873454610df6dbedf79a36d6bf1a7f6`)
and **SGLang 0.5.21** (`e00930c5489053f26d86b179cee0d087f846acbb`). Package extras
and source submodules agree. The [completion plan](completion-plan.md#s5--released-engine-integration-and-upstream-contributions)
is the execution queue; this document records consumed contracts and restrictions.
The earlier reviewed audit is preserved at
[commit 9aee895e](https://github.com/feichai0017/orbitkv/blob/9aee895ebe0ae2fb97279a477f00452b32025480/docs/engine-release-audit.md).

## 2026-10-06 official release recheck

The current upgrade freezes official vLLM
[0.31.0 / db9527a4](https://github.com/vllm-project/vllm/tree/db9527a46873454610df6dbedf79a36d6bf1a7f6)
and SGLang
[0.5.21 / e00930c5](https://github.com/sgl-project/sglang/tree/e00930c5489053f26d86b179cee0d087f846acbb).
The rebuilt complete wheel passes all four A100 dense eager installed-package
DRAM/io_uring cells, 15 vLLM and 43 SGLang released callback/recovery tests,
the standard vLLM model gate and two SGLang GPU restart gates. Independent
acceptance remains open; historical P/D, hybrid and topology claims are not
inherited. See S5.1 for the preserved compiler-cache failure and exact scope.

The remaining interface gaps still exist in these exact release sources:

- vLLM `v1/worker/gpu/model_runner.py` calls recurrent `preprocess_state` before
  `kv_connector.pre_forward`; `gpu/kv_connector.py` performs load and preemption
  handling in `pre_forward`. Keep V2/recurrent profiles rejected. V1 remains the
  single-attention-group candidate; no runner patch is installed.
- SGLang `mem_cache/registry.py` still selects only Mooncake/Mori external
  factories, and constructs the cache after graph capture. Public radix backend
  registration works, but does not replace OrbitKV's pre-capture event Hook.
- SGLang's `UnifiedCacheLinker.lookup` still returns ready boundaries rather than
  pending tickets. Preserve admission and native abort/drain ownership until an
  engine callback actually consumes its replacement.
- SGLang `disaggregation/decode.py:resolve_deferred_releases` retains host-staged
  destinations without a drain ACK, but still releases device destinations after
  timeout. That host fix does not qualify the current device P/D fault profile.
- SGLang weight updates can skip `flush_cache`; a namespace change only in
  `linker.reset` cannot safely invalidate every live update. Both adapters reject
  dynamic LoRA. The later fixed local adapter profile fingerprints startup
  artifacts; it does not intercept SDK/RPC mutations or qualify dynamic reuse.
  Live-weight invalidation needs a consumed engine contract; do not add a silent
  runtime patch.

LMCache 0.5.5 remains the latest released reference checked on this date. The
older interface inventory below records its 0.30.0/0.5.20 research inputs, not
new-release qualification. The complete delivery status lives in S5 of the
completion plan.

## 2026-10-08 upstream recheck

Official vLLM [0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)
and SGLang [0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)
remain the latest non-prerelease releases. The V2 ordering contribution
[vLLM #59410](https://github.com/vllm-project/vllm/pull/59410), external-linker
factory [SGLang #40595](https://github.com/sgl-project/sglang/pull/40595) and
load-failure lifecycle [SGLang #40896](https://github.com/sgl-project/sglang/pull/40896)
are open and unmerged. No dependent Hook removal or support expansion is
authorized by this recheck. Primary API snapshots are preserved with
`/root/orbitkv-artifacts/s5-budget-peaks-20261008/`.

## 2026-10-14 upstream recheck

The latest official non-prerelease versions remain vLLM 0.31.0 and SGLang 0.5.21.
[SGLang #40896](https://github.com/sgl-project/sglang/pull/40896) merged on
2026-10-12 as `f92199b5`; it is not part of the selected 0.5.21 release.
[SGLang #40595](https://github.com/sgl-project/sglang/pull/40595) and
[vLLM #59410](https://github.com/vllm-project/vllm/pull/59410) remain open.
These facts do not authorize private-Hook removal, fork runtime dependencies or
hybrid/P-D support expansion. SGLang pending admission now consumes the native
authenticated peer observation; that observation preserves held ownership on
failure and does not establish physical transfer termination.

The final installed wheel from `f6d4bff2` passes the official SGLang 0.5.21
Qwen3-8B dense TP=1/PP=1 DRAM/io_uring model gate. Independent review accepts
flush/restart recovery, concurrent shared prefixes, identity isolation and exact
text/token-count plus bounded finite-logprob controls against the same-engine
cold profile. Eight service receipts and installed/model hashes are verified.
The six official internal helper SIGKILL exits and CUDA IPC producer warnings
remain S3 limits; software failed-peer controls do not qualify GPU peer-failure
handling. Raw HTTP response/metric bodies were not separately archived, so no
numeric counter or performance claim is derived from this assertion-based gate.
Evidence is preserved outside the checkout with
[PR #238](https://github.com/feichai0017/orbitkv/pull/238).

The same-wheel vLLM model checks pass six assertions with one explicit hybrid
skip, but their separate native shutdown contract is rejected. An official
Noop baseline emits `EngineDeadError` after intentional teardown; the selected
release closes the engine before cancelling its output handler. Existing
[issue #48745](https://github.com/vllm-project/vllm/issues/48745) and unmerged
[PR #49000](https://github.com/vllm-project/vllm/pull/49000) track this race.
The frozen `--shutdown-timeout 30` gate retains its failure and quarantine;
normal API exits do not qualify physical drain. No expected-error filter,
upstream runtime patch or new support claim is applied.

## 2026-10-17 public worker preparation contribution

The selected runtime stays on official SGLang 0.5.21. An isolated contribution
against main `4fc86f73` adds optional `prepare_worker(worker, capture_decode_cuda_graph)`
to registered radix backends and invokes it before all runners in the existing
worker graph initializer. Two-argument registration and factory identity remain
unchanged; dispatch reads the published backend. Invalid callbacks are rejected
before insertion and preparation errors propagate before runner initialization.
[SGLang issue #44855](https://github.com/sgl-project/sglang/issues/44855) and
[draft PR #44856](https://github.com/sgl-project/sglang/pull/44856) track the generic
interface proposal; independent review accepts only contribution CPU scope.

The 32 candidate-source CPU tests use actual registry/runtime-context code and
the real worker method with recording collaborators. They cover selected-only
preparation, raw/published configuration differences, repeated calls, multiple
workers/runners, both capture flags and failure ordering. The existing local
dependency environment has SGLang distribution metadata 0.5.20; the candidate
checkout supplies tested SGLang modules. This is not installed 0.5.21 or GPU
qualification. No installed engine, OrbitKV Hook, package pin or submodule is
changed. Direct runner recapture and `ensure_decode_cuda_graphs` remain outside
the proposed boundary. Consume a supported released callback and pass actual
eager/graph restoration, cancellation and page-reuse gates before Hook removal.
Pending-ticket admission remains a separate interface task. Evidence stays at
`/root/orbitkv-artifacts/s5-sglang-worker-preparation-20261017/`.

## Released integration boundaries

| Engine | Consumed interface | Ownership |
| --- | --- | --- |
| vLLM | KV Connector factory/module path; scheduler lookup/allocation/save metadata; worker registration/load/layer waits/preemption/transfer results | Engine owns GPU pages and scheduling; OrbitKV translates callbacks and retains its native cache operations until drain |
| SGLang | Package plugin, `register_radix_cache_backend`, native `UnifiedRadixCache`, `UnifiedCacheLinker`, component override | Native tree owns slots/locks; OrbitKV linker supplies cache queries, loads/offloads, events and completion |
| vLLM P/D candidate | Official `NixlConnector` + `MultiConnector`, upstream router | Native NIXL owns live handoff; independent cache stores completed state |
| SGLang P/D candidate | Official disaggregation backend and router | Engine owns live handoff; external linker restores on P and only saves on D |

Ordinary cache is best effort (`requires_kv_delivery=False`). MultiConnector
preserves reliable delivery if any child requires it. The 0.30.0 allocation
fix from vLLM #46865 supplies actual blocks to non-loading children with zero
external tokens, enabling cache saves of native P/D-produced state. Unselected
OrbitKV queries release their leases. Terminal cache failures use the released
`KVConnectorTransferResults`; no duplicate failure queue remains.

The official V1 runner calls `handle_preemptions` before `_update_states`:
[released ordering](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py#L4215).
The V2 runner's pre-forward call is later than page initialization and recurrent
preprocessing. OrbitKV therefore requires `VLLM_USE_V2_MODEL_RUNNER=0` and rejects
multi-group/recurrent models before opening a cache connection. Both the runner
monkey patch and native-prefix method replacement are removed. The V2 ordering
contribution [vLLM #59410](https://github.com/vllm-project/vllm/pull/59410) remains
upstream material, not an installation requirement. Reopening rejected profiles
requires released ordering and atomic state visibility plus model/fault gates.

## Remaining Hook contracts

SGLang's public Hook registry does not make its string targets stable APIs.
These string targets are consumed in 0.5.21 and remain subject to GPU gates.
The older 0.5.20 links below preserve the original interface research:

| Internal target or bridge | What it protects | Official replacement / disposition | Removal gate |
| --- | --- | --- | --- |
| `TpModelWorker.init_cuda_graphs` BEFORE | Installs stable external CUDA events before capture so replay waits for restored layers | Factory is too late: Scheduler captures graphs in `init_model_worker` before constructing tree cache. Keep the guarded Hook. | A released pre-capture worker preparation callback, consumed by the adapter; rerun actual eager/graph restore and page-reuse gates |
| `PrefillAdder.add_one_req` AROUND | Defers pending SSD/peer queries without treating pending as miss; preserves rank agreement and cache admission | External-linker lookup returns ready boundaries, without an asynchronous pending ticket. Keep the guarded Hook. | Released pending-lookup/admission contract; cold/partial/full, timeout, cancellation and rank gates |
| `Scheduler._add_request_to_queue` AROUND (opt-in) | Starts optional preparation/warming for waiting requests | No consumed public enqueue callback. Disabled unless `ORBITKV_PREPARE_REQUESTS=1` or `ORBITKV_QUEUE_WARMUP=1`. | Released enqueue callback or removal of the optional optimization with matched workload checks |
| `RecoveryLinkerWrapper` / recurrent component override | Origin/boundary/rank translation, held destination locks and checkpoint release | Public component override is consumed; generic recurrent lifecycle remains incomplete upstream. Keep correctness ownership; broad model support remains unqualified. | Released component semantics and exact checkpoint/window bytes under abort/preemption |
| Former Scheduler abort Hook | Cancels lookups and drains restores before tree release | Removed: official `BasePrefixCache.finish(ABORT)` reaches `UnifiedRadixCache.release_aborted_request` and linker release. | Covered by consumed native finish tests and GPU restore cancellation |
| Former private P/D observers and fork `PDTransferEvent` callback | Observed P/D lifecycle; did not authorize release | Removed with the fork payload adapters. Native engine metrics/behavior are the evidence source. | No runtime replacement needed for optional telemetry; fault qualification remains open |

Unsupported DSA, speculative draft/auxiliary GPU state, hierarchical-cache mode
and host-pool retraction fail explicitly. No second RadixCache, Python cache
facade or alternative P/D lifecycle is added. Unselected plugins remain lazy
and do not import CUDA/native runtime or connect to a Manager.

## Native P/D release limits

SGLang 0.5.20 defaults `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE`
to false. The candidate explicitly enables it on both workers. However, the
[released decode queue](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/disaggregation/decode.py#L2492)
still frees held pages when the timeout expires without a full drain ACK.
This is a released lifecycle gap: a deadline alone cannot prove remote writes
have stopped. Native P/D transfer cancellation, delayed ACK and peer-loss page
reuse remain unqualified; ordinary linker restore cancellation is a separate
cache-owned contract. The historical upstream patches remain contribution
material; no local runtime patch hides this limitation.

## Comparison with external caches

- [LMCache 0.5.5 vLLM MP adapter](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/integration/vllm/lmcache_mp_connector.py)
  implements the connector in its external package with separate scheduler and
  worker ownership. OrbitKV follows that boundary, keeping storage and cache
  state machines in Rust rather than copying LMCache policy or MP transport.
- [LMCache's NIXL/MultiConnector recipe](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/disaggregated_prefill.rst)
  separates live P/D from historical cache and recommends separate cache servers
  for P and D. OrbitKV adopts this composition as a candidate; shared-Manager
  contention and fault behavior need independent evidence.
- [SGLang 0.5.20 LMCache](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py)
  extends the older RadixCache. OrbitKV keeps the released Unified tree/linker.
  Later main-only Unified integration is not a released replacement.
- [vLLM FlexKV entry](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector.py)
  loads its external implementation; [SGLang FlexKV registration](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/mem_cache/storage/flexkv/__init__.py)
  uses the backend factory. These are useful interface references, not evidence
  that all integrations are free of internal coupling or share our qualification.

## Runtime and qualification policy

The wheel contains only cache adapters and the native cache runtime. It rejects
retired custom P/D packages, TENT payload adapters and runner monkey patches
through the packaging gate. CI does not install an engine fork. The official
engine files and installed OrbitKV wheel are frozen for model tests; record
hashes before and after. Raw results remain outside the checkout.

Cold/hot/restart and DRAM/SSD gates qualify ordinary cache separately from
native P/D cancellation, preemption, partial submission, delayed ACK and page
reuse. The [P/D guide](pd.md) records candidate configuration and preserved fork
contribution material. Historical HMA, topology and fork passes do not transfer
automatically to this official release profile.
