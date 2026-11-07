# CUDA copy kernels

> Historical qualification below applies only to its original revisions and
> recorded backend, including TENT. OrbitKV now uses OrbiTE exclusively; all
> current GPU/RDMA/GDS and installed profiles need fresh hardware qualification.
> CPU/TCP and io_uring controls remain CI evidence.

OrbitKV keeps end-to-end copy/serving controls; OrbiTE owns local copy kernels.
Use [NVIDIA KDA](https://github.com/NVlabs/kda) for measured CUDA optimization
and the [completion plan](completion-plan.md) for remaining delivery gates.

## Available copy backends

The ordinary dense profile defaults to direct DMA. The explicit `kernel` backend
copies fragmented mapped host memory with GPU threads. It does not require a
second full staging copy into device memory. GPU work still competes with model
execution, so a faster isolated copy does not select the serving default.

Both choices now use [OrbiTE local copies](transport.md#local-gpudram-copies).
OrbiTE owns copy submission and scratch; OrbitKV retains source and destination
owners through physical completion. The old OrbitKV `KernelBackend` and
`MemcpyBackend` have been removed.

Select the explicit route using:

- vLLM: `kv_connector_extra_config["orbitkv.transfer_backend"]="kernel"`.
- SGLang: `ORBITKV_TRANSFER_BACKEND=kernel`.

The existing MLA resolver can select the kernel separately; this does not
qualify a new MLA model or deployment.

## Current serving result

The historical Qwen3-32B BF16/raw H20 profile used TENT at the revisions
recorded in [performance](performance.md#independent-sglang-kernel-pairs):
TP=1/PP=1, eager and serial. Four independent DMA/kernel pairs passed the frozen
benefit/regression guards:
8K pressure TTFT is 10.45% lower than our DMA route; 4K is effectively unchanged.
This does not establish a general win over HiCache/LMCache, tails, throughput,
Graph, concurrency or inference contention. Full context and uncertainty are in
[performance](performance.md#independent-sglang-kernel-pairs).

## Reproduce a component comparison

The former `native_kernel_copy` benchmark consumed OrbitKV's removed
`KernelBackend`/`MemcpyBackend`. Reproduce it with the
[original source and command](https://github.com/OrbitInfra/OrbitKV/blob/c7fd6144592b64c487bf58a27e547a6f44ab8c26/docs/kernel-optimization.md#reproduce-a-component-comparison).
New byte-engine microbenchmarks belong in OrbiTE.

Freeze source, binary and input layout; use an external output directory and
separate Cargo targets when comparing worktrees. Run GPU measurements
sequentially. Do not rebuild a mapped native library in a live Manager checkout.

For actual SGLang layouts use `benches.transfer.sglang_kv_oracle`, then the
matched model comparison `benches.transfer.kernel_pairs`. Their commands and
inputs live in [transfer benchmarks](../benches/transfer/README.md).

## Measure the bottleneck before changing code

Use one consumed end-to-end operation and separate:

1. Client conversion, descriptor packing and host submission.
2. Transfer/kernel execution and actual stream wait.
3. Model readiness and complete request TTFT/throughput.

A component sample includes submission-to-drain. CPU submission and wait clocks
are not isolated GPU execution; profiler timing is separate from profiler-off
serving. Lazy initialization and first-transfer setup must not be pooled into
warm steady state or silently discarded.

Observation itself must pass its original matched overhead guard before
attributing a regression or promoting a candidate. Current Manager and
engine-local H2D observation pilots have failed guards; their results remain
failed. Healthy performance work unrelated to those observers can use the
existing profiler-off serving benchmarks.

KDA inputs must name the current production owner/shader, representative actual
layouts, correctness oracle and frozen benefit/regression guards. Promote only
a candidate consumed by a measured application path. If submission, admission
or storage is the cause, change that existing Rust owner rather than adding an
unrelated kernel. Do not relax source/destination lifetime or early-consumption
contracts to improve a timing result.

## Historical measurement contracts

Detailed component cohorts, A/A repeatability, CPU traces, thread CPU
attribution, interleaved calibration, capture diagnostics, failed candidates and
external artifact locations are preserved in the
[immutable kernel report](https://github.com/OrbitInfra/OrbitKV/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/kernel-optimization.md).
Those frozen guards remain the authority for their corresponding experiments.
They do not form a second execution queue. Generated traces/results stay outside
Git; see [benchmark evidence](benchmark-evidence.md).
