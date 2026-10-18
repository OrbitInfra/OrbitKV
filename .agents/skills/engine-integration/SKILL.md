---
name: engine-integration
description: Implement or review OrbitKV vLLM/SGLang connectors, hybrid recovery, P/D composition, release upgrades and upstream contributions; diagnose engine page-lifecycle or async restore failures.
---

# Released-engine integration

Read `AGENTS.md`, `docs/adapters.md` and S5 in `docs/completion-plan.md` from the
Git root. The completion plan is the only work queue. Use its current support
status; do not treat an upgrade target or upstream feature as qualified behavior.

## Select the actual contract

For an upgrade, check the latest official non-prerelease tag and record its commit.
Inspect that source and the corresponding released LMCache integration. Use main
and related issues/PRs to find fixes; distinguish unreleased fixes explicitly.
Freeze the selected release during review and update pins, lockfile and submodule
together after the consumed adapter passes. Do not add old/new API fallbacks.

## Keep one owner per boundary

- Engines own request scheduling, GPU allocation, native prefix trees, rank
  collectives, model execution and final request readiness.
- vLLM transfer-result and SGLang offload-count polls consume native
  `publish_progress()` for rate limited diagnostics only. An active interval
  can span overlapping calls; overdue, retained or dead Manager observations
  do not complete saves or release their engine pages. Missing observations
  cannot be interpreted as zero active work.
- The Save handle candidate is consumed by both adapter save workers. They
  hand off registered sources using `start_save` and await `wait_save` in Rust.
  Unknown wait responses/exceptions retain task/page completion state and
  prevent reset/unregister. This code is not a page-slot generation or physical
  fault receipt; installed engine/fault acceptance is still required.
- Python translates engine layout, state boundaries, callbacks and CUDA readiness.
  Existing Rust owners manage shared cache policy, transfers, admission and drain.
- vLLM uses the KV Connector contract; SGLang uses UnifiedRadixCache and the
  external linker. Preserve valid native HBM hits without external synchronous
  lookup. Keep unselected backend imports free of native initialization.
- vLLM 0.31.0 workers return `KVConnectorTransferResults` directly. P/D failed
  receives must also be finished receives in the same poll; do not restore a
  custom worker-metadata failure queue. Ordinary cache delivery is best effort,
  while a P/D producer retains the native reliable-delivery requirement.
- SGLang 0.5.21 cancellation already reaches the external linker through
  `BasePrefixCache.finish(ABORT)` and `UnifiedRadixCache.release_aborted_request`.
  Keep cancellation/drain in that consumed lifecycle; do not restore a duplicate
  private Scheduler abort Hook. Queue preparation is opt-in; fork P/D observation
  Hooks are removed.
- Dense TP shard queries use Manager-owned native fan-out, not Python foreign
  UDS loops. Official worker handshake exports opaque registered targets; local
  authenticated configuration validates complete rank/node coverage. Require
  `--enable-query-control`; one-node consumption explicitly opts into
  `orbitkv.query_control=true`. Keep trusted literal loopback scope and equal
  TP-only/one-group V1 guards. Speculative/deferred-finalize metadata ordering
  cannot report native completion safely, so it is rejected in this profile.
  Worker metadata reports unique global ranks only after wait_restore; release
  retained source interests only after all declared ranks complete. Run the
  two-real-Manager installed byte/SSD gate separately from the official one-node
  handshake serving gate; neither qualifies physical multi-GPU/cross-host TP.
- P/D uses official vLLM NIXL/MultiConnector and SGLang native disaggregation;
  Manager shared-cache traffic still uses TENT. Do not reintroduce fork factories,
  callbacks, custom connectors, handshake/proxy or partial-tail cache modes.
- The candidate profile reads/writes cache on P and only saves on D. Native P/D
  owns incoming destinations; cache retains completed state until its own drain.
  Preserve reliable P/D delivery and release unselected cache query leases.
- Official vLLM 0.31.0 requires V1 (`VLLM_USE_V2_MODEL_RUNNER=0`) and one attention
  cache group. Reject V2/recurrent profiles before connection; the runner and
  native-prefix monkey patches are removed. Reopen only after consumed released
  ordering and atomic state hand-off pass GPU/model gates.
- SGLang native P/D candidate enables released deferred KV release on both P
  and D, but 0.5.21 still releases on timeout without a full drain ACK. Keep
  transfer cancellation, peer-loss and delayed-ACK page reuse unqualified;
  ordinary linker cancellation is a separate cache-owned contract.
- vLLM's public `reset_cache()` now invalidates future external reuse through
  a fresh prefix-key generation only after all tracked request/query/load/save
  owners end. HBM-only reset preserves the current generation. Test idle
  `reset_external=true` followed by a cold miss and reuse of newly saved state;
  do not advertise active-request reset, persistent cluster-wide invalidation
  or live weight changes that bypass the callback. SGLang ordinary flush keeps
  same-weight external reuse and is not a universal weight-update notification.
- Fixed local PEFT adapters require startup content fingerprints and effective
  LoRA settings in the deployment namespace. Keep native base/adapter hashes
  intact; reject resumable/session paths and caller extra-key aliases that bypass
  safe matching. This immutable profile does not intercept engine SDK/RPC
  add/remove/reload or live-weight updates; change artifacts only after stopping
  the engine.
- SGLang 0.5.21 preserves raw `ServerArgs` and publishes resolved configuration
  separately. Consume `runtime_context.get_lora()` for effective enable, startup
  LoRARef UIDs, backend and target modules, as its native loader/registry do.
  `--lora-paths` does not make the raw `enable_lora` field true.
- Explicit FULL decode Graph cache recovery is independently qualified only for
  the recorded H20 dense TP1/PP1 serial batch-1 eight-token direct/kernel io_uring
  profile on both official releases. Use `--release-cuda-graph` in the installed
  gate and require native runtime observations; capture-only logs do not qualify.
  Preset sizes 2/4, long decode/save boundaries, adapters with Graph, concurrency,
  preemption/cancellation, prefill capture and physical topology remain separate.
- Run native model composition and lifetime gates separately. Output/restart
  does not prove cancellation, partial-submit, delayed-ACK or page-reuse safety.
  Earlier fork passes remain upstream contribution evidence, not release support.

## Replace internal coupling safely

Audit every remaining Hook against `docs/engine-release-audit.md`. SGLang 0.5.21
captures graphs before the tree-cache factory; preserve pre-capture events until
a safe released callback exists. Its linker lookup lacks pending tickets, so
pending-query admission remains a guarded internal Hook; queue preparation is
opt-in. Do not describe these as public lifecycle integration. Keep native abort
finish, cancellation/drain and page ownership until a replacement is consumed.
Do not add a facade merely to hide internal coupling.

For a stuck or incorrect request, read [the boundary tracing guide](references/lifecycle.md).
Enqueue, layer readiness, final native drain and source retirement are distinct.
Timeout, process disappearance and metadata expiry cannot prove physical drain.

## Verify the claimed profile

Run the relevant gates from `AGENTS.md`: source-only tests for local contracts;
actual engine/GPU tests for changed callbacks, preemption, page reuse and graph
ordering. Include cold/partial/full hits, native HBM, restart and failure controls.
Preserve adapter/salt/model identity and effective committed lengths. Record TP,
PP, attention DP, EP, P/D topology and transport separately; a one-GPU process test
does not qualify their distributed combination. Keep evidence outside checkout.

For SGLang copy-backend changes, use `benches.sglang_kv_oracle` to check the
actual released pool, full payload/sentinel coverage, partial destinations,
event generations and deliberate corruption through the installed native path.
Its deterministic fixture bytes do not dump model-computed KV. Evaluate the
consumed model separately with `benches.kernel_pairs`: freeze fresh independent
process pairs, exact input/output and installed artifacts, real shutdown receipts
and the original benefit/regression guards. Keep profiler clocks separate.
The accepted serial profile and its lifetime limits are recorded in S5.5;
neither fixture success nor pair medians select a general backend default.

For upstream delivery, separate baseline registration/configuration/tests/docs,
generic lifecycle fixes and TENT transport integration into reviewable PRs.
Use a clean engine checkout plus installed wheel as the acceptance environment.
Report local qualification, upstream merge and released support separately.
