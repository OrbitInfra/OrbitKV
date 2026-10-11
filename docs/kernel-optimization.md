# CUDA kernel optimization

OrbitKV uses the [NVIDIA KDA workflow](https://github.com/NVlabs/kda) to define,
validate and measure a bounded kernel task. The execution queue remains
[S4](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).

## Fragmented mapped-host copy

`KernelBackend` consumes destination, source and byte count as three `u64`
values per descriptor. Large fragments in small batches now receive up to 16
CTAs per descriptor, bounded by four CTAs per SM and one CTA per 64 KiB.
The grid-stride loop uses 64-bit work indices, vectorizes only jointly aligned
16-byte addresses and copies scalar tails exactly. The worker still owns its
stream, descriptor scratch and source/destination holds through physical drain.
The descriptor format and Python API are unchanged.

CUDA device code lives in `crates/orbitkv-core/src/transfer/kernel.cu`, embedded
with Rust's `include_str!` and compiled through NVRTC. Installed binaries need no
source-file lookup. The original one-CTA baseline is retained in
`benches/kernels/batch_copy_baseline.cu`; `benches/kernels/fragmented_copy.cu`
retains the component candidate. The direct DMA backend still coalesces legal
adjacent/strided ranges and remains the ordinary dense-attention default. vLLM's
existing MLA resolver selects the kernel when no override is supplied; the
measurements below do not qualify MLA serving. Select the kernel explicitly with
vLLM's `kv_connector_extra_config["orbitkv.transfer_backend"]="kernel"` or SGLang's
`ORBITKV_TRANSFER_BACKEND=kernel`. This task changes neither resolver.

The component driver checks H2D and D2H, empty/zero-size batches, independent
address misalignment, tails, reordered ranges, source preservation, guard bytes
and batches exceeding the grid limit before collecting timing samples:

```bash
python -m benches.kernel_copy \
  --source-root /path/to/orbitkv \
  --output /var/tmp/orbitkv-kda/validation-001 --validate-only
python -m benches.kernel_copy \
  --source-root /path/to/orbitkv \
  --output /var/tmp/orbitkv-kda/measurement-001
```

This driver requires PyTorch, CUDA and NVRTC; `--nvrtc` selects a toolkit path.
It records hardware, sources, compiler inputs and kernel/upload/submit-to-drain
boundaries. Python timing is not native restore latency.

The native driver consumes the actual Rust `KernelBackend` and `MemcpyBackend`,
including descriptor packing, upload, submission, synchronization and legal DMA
coalescing. Each backend must preserve all source/guard bytes in both directions:

```bash
cargo bench -p orbitkv-core --bench native_kernel_copy \
  --no-default-features --features cuda-13,mooncake -- \
  --output /var/tmp/orbitkv-kda/native-001.jsonl --samples 30
cargo bench -p orbitkv-core --bench native_kernel_copy \
  --no-default-features --features cuda-13,mooncake -- \
  --output /var/tmp/orbitkv-kda/model-shape-001.jsonl --samples 30 \
  --fragment-bytes 131072 --blocks 864
```

Use a new external output for each attempt and freeze both source and executable.
When comparing worktrees, use isolated Cargo target directories and verify the
CUDA source embedded in the resulting binary; build-cache reuse alone does not
prove the intended shader was consumed. Keep GPU measurements sequential.

The maintained native fixture also retains `kernel_cpu_submission_ms` and
`dma_cpu_submission_ms` beside their `*_stream_wait_ms` arrays. CPU submission
includes packing, upload and driver calls; waiting measures the host's stream
synchronization. The existing outer submission-to-drain boundary remains, with
the added clocks' overhead included. These fields do not measure isolated GPU
execution, and historical totals must not be relabelled as stage timings.
Source/destination holds and partial-submission drain are unchanged. Exact H20
native attribution and separately queued production-shader event timing require
their own frozen input/review and byte gates before selecting another candidate.

## Current qualification

**Implemented; A100 SM80 local gates pass; independent acceptance open.** The
component driver passes 30 byte controls. Six order-alternated native runs cover
34 case/direction rows each, with five warm-ups and 30 matched samples per backend.
The largest paired 4 KiB median regression against the original kernel is 1.11%,
below the predeclared 5% guard. Small batches with large fragments improve against
the original kernel; coalesced DMA remains competitive or faster on regular ranges.
These are native batch timings, not an inference speedup.

A descriptive follow-up covers 24 and 864 BF16-sized 128 KiB fragments with three
matched pairs in each direction. These representative shapes are not a captured
runtime descriptor histogram. The new/old kernel median ratios remain within
about 1.6% of parity. This limits the optimization claim to the measured large,
small-batch gap; it does not establish a model-shaped improvement or a new default.
Raw samples, real DMA comparisons, build-cache failure and sources remain at
`/root/orbitkv-artifacts/kda-native-copy-20261007/`, mirrored at
`/workspace/orbitkv-kda-native-copy-20261007/` on A100. Earlier component measurements
remain immutable at `/root/orbitkv-artifacts/kda-fragment-copy-20261006/`.

The complete CUDA 13 wheel (`485d0f8c…`) embeds the verified new CUDA source and
passes five official-engine gates: vLLM 0.31.0 and SGLang 0.5.21 ordinary cache in
DRAM/io_uring SSD, plus both engines sharing an SSD Manager. All gates explicitly
select `kernel` and assert the consumed worker route in Manager logs. Cold/native
HBM/full/partial hits match native model controls, SSD recovery requires physical
reads after DRAM eviction, and overlapping two-engine restore loads exactly
226,492,416 bytes. Normal Manager restart reconstructs a cold cache; it does not
prove persistent index recovery or abrupt-death reclamation. Installed files
remain unchanged and all 28 services exit normally without forced cleanup.

A follow-up debug profile of the unchanged installed wheel `8acd6a06…` records
actual full-save shapes: vLLM has 432 descriptors of 256 KiB; SGLang has 864 of
128 KiB, with partial saves of 72 descriptors. A native KDA candidate caps the
existing grid at the existing SM-times-four budget. Its first matched pair passes
all byte controls across 42 case/direction rows per executable, but two small
H2D cases regress 5.37% and 8.23%, exceeding the frozen 5% median guard. The
observed large SGLang shape improves only about 1%–2%, below the declared 3%
benefit target. The candidate is rejected; later pairs are not run and production
CUDA/Rust code, installed wheel and backend defaults are unchanged. The native
addresses are synthetic fragmented fixtures with measured descriptor sizes and
counts, not a replay of engine pointers or a captured restore histogram.

The profile passes 85 native-output comparisons with normal cleanup, but tracing
invalidates a performance comparison. Two untraced SGLang startups fail discovery
before the pressure window; the fresh sustained cohort stops at its first cell.
Idle/registered delayed joins and a short untraced file-lifetime diagnostic pass,
without proving the intermittent failure repaired. Nsight Compute/System are
unavailable on this A100; timings do not establish a hardware-counter cause.
Contracts, sources, failed cells and the rejected candidate remain at
`/root/orbitkv-artifacts/s5-shared-pressure-kda-20261007/`, mirrored at
`/workspace/orbitkv-s5-shared-pressure-kda-20261007/`.

Independent reproduction, sustained inference contention, CUDA graph execution,
H20 paired native performance, native transfer faults and S3-dependent reclamation remain unqualified.
These results do not close S4 or establish TTFT/ITL, throughput or NIC isolation.

The final installed wheel `fe427c42…` also passes a 900-second H20 SM90
two-engine shared-Manager io_uring gate with `kernel` explicitly consumed in
worker logs. All fresh/reused requests match native text/token-count controls,
reservation budgets and six actual io_uring drain gauges pass, and all services
exit normally. The corresponding direct cell passes too. This is consumed-path
correctness and sustained progress, independently accepted in the recorded profile;
different seeds and no matched independent pairs preclude a performance claim.
See [the scoped S5.5 result](completion-plan.md#s55--deployment-matrix-and-upstream-maintenance)
and `/root/orbitkv-artifacts/s5-h20-closeout-20261008/`.

## Bounded CPU descriptor reuse follow-up (2026-10-08)

**Rejected; no production CUDA or backend change.** A KDA candidate reused the
worker's pageable CPU packing Vec, retaining at most 96 KiB. Device scratch was
already reused; shader, arguments and direct default were identical. The first
A100 pair checks 34 shape/direction rows per executable, with five warm-ups and
100 samples per backend/row, including exact output, source and guard bytes.
Every byte control passes. Grid-stride H2D median rises 0.74407 → 0.78856 ms
(5.98%), beyond the frozen 5% guard. The workflow stops before later model-shape
or independent pairs and installed-model gates. This is a rejection under the
contract, not proof that Vec reuse always causes that regression.

The candidate is removed from production; source and all 13,600 kernel/DMA timing
values remain at `/root/orbitkv-artifacts/s4-readv-kda-20261008/`, mirrored at
`/workspace/orbitkv-perf-20261008/`. The distinct SSD stage diagnostic in
[SSD performance](ssd-performance.md) points the next investigation at existing
SSD preparation and waiting. Rejected attempts do not change kernel defaults
or establish an inference speedup.

## Consumed follow-up requirements

The next changes remain in S4 of the completion plan. Use the engine's actual
fragment histogram and inference load before selecting a candidate. Device code
belongs in the existing transfer/codec `.cu` files, with Rust retaining resource
ownership; Python supplies engine layout and readiness.

| Candidate | Required measurement and correctness boundary |
| --- | --- |
| Fragmented mapped-host copy and reusable descriptor upload | Measure native packing/upload/submit-to-drain under actual fragment sizes, mixed tails and concurrent model compute. Compare coalescing DMA; the current large-fragment gain does not justify a general kernel default. |
| SSD GPU-staging scatter | First establish that per-range D2D dispatch in the consumed staged cuFile route is material. A batched scatter must retain every source extent, staging slot and engine destination until physical completion. Native GDS eligibility is a separate gate. |
| Encoding/checksum fusion | Establish an extra HBM pass worth removing; preserve stored byte format and IEEE CRC32. Corrupt input must be rejected before engine-page writes, so unverified decode/scatter directly into destination pages is ineligible. Measure encoding, checksum, D2H and SSD/peer I/O together. |
| ANS and FP8/TurboQuant codec kernels | Prioritize exact ANS only if the full encode/read/decode route improves. Lossy formats require declared model-quality controls; historical TurboQuant failures stay visible and recurrent state stays exact. A100 and H20 qualification are separate. |

Supply actual block sizes, tensor strides, fragmentation, inference load and
storage/transport budgets. Use Nsight Compute counters when device permissions
permit; timing alone cannot identify a hardware bottleneck. KDA component byte
controls and matched native measurements precede final-wheel engine gates. SSD
host reader batching is a Rust submission optimization, described in
[SSD performance](ssd-performance.md); it adds no CUDA kernel.
