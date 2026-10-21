# Transfer benchmarks

Run from the repository root. Keep source/binaries and input layouts frozen;
write results outside the checkout. [Benchmark navigation](../README.md).

| Entry point | Use |
| --- | --- |
| `benches.transfer.communication` | Native DMA/kernel communication measurements |
| `benches.transfer.raw_copy_shapes` | Fragmented raw-copy shapes |
| `benches.transfer.local_restore_shapes` | Actual native local-Restore layouts |
| `benches.transfer.sglang_kv_oracle` | Released SGLang pool bytes, guards and corruption controls |
| `benches.transfer.kernel_pairs` | Matched DMA/kernel model comparisons |
| `benches.transfer.copy_observation` | Matched production H2D observer overhead |
| `benches.transfer.native_trace_calibration` | Native trace calibration and offline analysis |
| `benches.transfer.cuda_submission` | Host submission and device execution diagnosis |

Use `python -m <entry-point> --help` for current input/output options. Native
CUDA/C++ fixtures stay under `native/`; production code stays in its Rust owner
and separate `.cu` file. The native copy benchmark is:

```bash
cargo bench -p orbitkv-core --bench native_kernel_copy \
  --no-default-features --features cuda-13,mooncake -- \
  --output /var/tmp/orbitkv-bench/native-copy-001.jsonl --samples 30
```

For serving comparisons, first verify actual pool layout, complete payload and
sentinels using `sglang_kv_oracle`; then use `kernel_pairs` with independently
started pairs and profiler-off native output controls. Preserve source holds,
events and exact installed artifacts through normal shutdown. Unknown native
work must not release backing memory.

The actual-layout oracle and copy-observer contracts currently freeze SGLang
0.5.21. Reproduce those measurements with their archived matching wheel and
harness. A new 0.5.22 performance comparison must qualify its actual pool and
update the consumed contract; ordinary-cache upgrade checks do not do this.

Profiler-off application measurements decide promotion. Isolated copy bandwidth,
CUDA event duration and client TTFT have different boundaries. A new observer
must pass its frozen matched-overhead guard before its timings support a causal
claim; current failed pilots remain failed. Do not pool request samples as
independent process runs or promote a serial result to concurrency/Graph support.

Run the relevant CPU tests with `python -m pytest benches/tests/transfer`.
Hardware entry points are explicit and are not default pytest jobs.
[Kernel behavior and current result](../../docs/kernel-optimization.md),
[serving measurements](../../docs/performance.md) and
[the frozen detailed recipes](https://github.com/OrbitInfra/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/benches/transfer/README.md)
preserve the commands/contracts for previous campaigns.
