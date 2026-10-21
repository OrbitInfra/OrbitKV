# CUDA copy kernels

OrbitKV uses [NVIDIA KDA](https://github.com/NVlabs/kda) for measured CUDA
optimization. Keep device code in separate `.cu` files and consume it through
the existing Rust transfer/codec owners. [S4](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains)
owns the remaining delivery; historical experiments are linked below.

## Available copy backends

The ordinary dense profile defaults to direct DMA. The explicit `kernel` backend
copies fragmented mapped host memory with GPU threads. It does not require a
second full staging copy into device memory. GPU work still competes with model
execution, so a faster isolated copy does not select the serving default.

`KernelBackend` reads destination, source and byte count from three `u64` values
per descriptor. The production shader is
`crates/orbitkv-core/src/transfer/kernel.cu`, embedded and compiled with NVRTC.
It uses bounded multi-CTA work, 64-bit indices, aligned vector copies and exact
scalar tails. Rust owns descriptor scratch, streams and buffer holds until drain.
DMA retains legal adjacent/strided coalescing.

Select the explicit route using:

- vLLM: `kv_connector_extra_config["orbitkv.transfer_backend"]="kernel"`.
- SGLang: `ORBITKV_TRANSFER_BACKEND=kernel`.

The existing MLA resolver can select the kernel separately; this does not
qualify a new MLA model or deployment.

## Current serving result

The accepted Qwen3-32B BF16/raw H20 profile is TP=1/PP=1, eager and serial.
Four independent DMA/kernel pairs pass the frozen benefit/regression guards:
8K pressure TTFT is 10.45% lower than our DMA route; 4K is effectively unchanged.
This does not establish a general win over HiCache/LMCache, tails, throughput,
Graph, concurrency or inference contention. Full context and uncertainty are in
[performance](performance.md#independent-sglang-kernel-pairs).

## Reproduce a component comparison

The existing native benchmark consumes actual `KernelBackend`/`MemcpyBackend`,
including packing, upload, submission and synchronization:

```bash
cargo bench -p orbitkv-core --bench native_kernel_copy \
  --no-default-features --features cuda-13,mooncake -- \
  --output /var/tmp/orbitkv-kda/native-001.jsonl --samples 30
```

It covers empty batches, misalignment, scalar tails, reordered ranges and large
batches, checking payload and guards in both directions. Freeze source and
binary; use an external output directory and separate Cargo targets when comparing
worktrees. Run GPU measurements sequentially. Do not overwrite mapped native
libraries by building in a live Manager checkout.

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
