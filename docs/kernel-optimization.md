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
source-file lookup. The maintained benchmarks consume this production shader;
the direct DMA backend still coalesces legal
adjacent/strided ranges and remains the ordinary dense-attention default. vLLM's
existing MLA resolver selects the kernel when no override is supplied; the
measurements below do not qualify MLA serving. Select the kernel explicitly with
vLLM's `kv_connector_extra_config["orbitkv.transfer_backend"]="kernel"` or SGLang's
`ORBITKV_TRANSFER_BACKEND=kernel`. This task changes neither resolver.

The retired PyTorch/NVRTC component experiment remains reproducible from its
[frozen source revision](https://github.com/feichai0017/orbitkv/tree/21b6cfaf04888413d404fba439f508db69303cd6/benches).
Its baseline, candidate, commands and failed controls are preserved in that
revision and `/root/orbitkv-artifacts/kda-fragment-copy-20261006/`.
Both archive manifests were fully verified before removing the duplicate driver
and shader copies from the maintained tree. Its Python timing is not native
restore latency.

The native driver consumes the actual Rust `KernelBackend` and `MemcpyBackend`,
including descriptor packing, upload, submission, synchronization and legal DMA
coalescing. It covers empty/zero-size batches, independent address misalignment,
scalar tails, reordered ranges and batches exceeding the grid limit. Each backend
must preserve all source/guard bytes in both directions:

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
Source/destination holds and partial-submission drain are unchanged. The bounded
H20 CPU-attribution result below is accepted; separately queued production-shader
event timing and actual model layouts require their own frozen input/review and
byte gates before selecting another candidate.

The native fixture also creates a new `*.device.json` sidecar from its actual
CUDA context UUID, ordinal and PID before backend initialization, and repeats
that UUID in each timing row. Launch validation binds the receipt to the exact
native executable, process identity and consumed library maps. NVML numeric PID
visibility is a separate observation; equal PID namespaces are not assumed.
This fixture targets a non-MIG GPU and does not qualify MIG device identity.

## Consumed Manager raw-copy shapes

Manager worker diagnostics record the actual validated raw-copy descriptors
after their existing physical stream drain. Enable both
`ORBITKV_TRACE_TRANSFERS=1` and a positive
`ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT` (at most 65,536). These observations share the
existing bounded buffer with other diagnostic stages and serialize at normal
Manager shutdown. Default execution adds no diagnostic clocks, descriptor scan
or event allocation. Empty batches still drain but emit no raw-copy sample.

Each `manager_raw_copy_complete` record contains the backend/direction/device,
bytes, fragment histogram, legal compiled DMA ranges, CPU backend submission
and host stream waiting. The histogram's upper bounds are 256 B, 4 KiB, 16 KiB,
64 KiB, 256 KiB, 1 MiB and 4 MiB, followed by larger fragments. DMA range counts
use the actual compiler and device pitch limit, even for a kernel batch; they
do not count that kernel's CUDA launches. No addresses or extra source owners
are recorded. Partial enqueue errors still drain before returning failure.

After independently verifying normal Manager exit and owned-process cleanup,
parse its finalized log into a new external artifact:

```bash
python -m benches.raw_copy_shapes \
  --manager-log /var/tmp/orbitkv-kda/model-001/manager.log \
  --output /var/tmp/orbitkv-kda/model-001/raw-copy-shapes.json
```

The parser requires successful flush coverage, exact buffered record counts and
consistent shapes; missing records, overflow, late records and failed submissions
invalidate evidence. CPU submission and host waiting are not device-event time,
bandwidth or request latency. Stream waiting can include earlier queued work.
The summary includes cold batches and does not infer request linkage or a
confidence interval. Engine-local asynchronous raw restores are outside this
Manager-only scope; do not change their layer events or add synchronization to
obtain these records.

Software `c0b4ff3f` and one H20 official-vLLM 0.31.0/Qwen3-32B dense TP1/PP1 DRAM
structural pilot are independently accepted. The model gate passes six checks
with one dense-inapplicable HMA skip. Its complete 34 direct D2H batches contain
296 MiB, 4,736 descriptors of exactly 64 KiB and 3,648 legal DMA ranges. Existing
DMA compilation reduces the observed range count by about 23%; this describes
the consumed layout without establishing a new speedup. All cold batches remain.
The new diagnostic Manager uses the unchanged accepted client wheel; complete
new-wheel qualification remains open. Original logs, complete input hashes,
normal leader exits, known-resource cleanup and independent recomputation are at
`/root/orbitkv-artifacts/s5-kda-real-copy-shapes-outcome-independent-review-20261011/REVIEW.md`.
Official internal child-force and semaphore warnings remain physical IPC/S3
lifetime limits. Matched overhead, engine-local H2D geometry, device-event timing
and application comparisons remain open before performance promotion. A native
comparison must preserve allocation/registration boundaries and compiled range
counts representative of these batches; this scalar log cannot reconstruct
actual addresses or pitch.

## Allocation-aware native comparison

`benches/native_copy_layout.rs` compares the production copy backends with 64
separate retained device allocations and either one mapped host registration or
one per block. Eight layouts match the observed 64 KiB descriptor/range-count
pairs; a ninth tests host-registration boundaries. Explicit page permutations,
allocation sizes and normalized offsets are serialized with each row. Guard
padding and chosen pitches make this a disclosed representative family, not
replay of captured model addresses. H2D is a synthetic direction control.

The private DMA compiler is shared by source with this fixture; its existing
algorithm and production submission contract are unchanged. Validate layout
equations before initializing CUDA, then run against a new external output:

```bash
cargo bench -p orbitkv-core --bench native_copy_layout \
  --no-default-features --features cuda-13,mooncake -- \
  --validate-layout-only --output /var/tmp/orbitkv-kda/layout-001.jsonl
cargo bench -p orbitkv-core --bench native_copy_layout \
  --no-default-features --features cuda-13,mooncake -- \
  --output /var/tmp/orbitkv-kda/layout-native-001.jsonl --samples 100 --seed 2026101111
```

Each backend/direction/event mode passes full output, source and guard checks
before five warmups; a final check follows all measured submissions. Four modes
(kernel/DMA, events off/on) alternate positions within every four samples.
Event-off wall/submission/drain arrays remain separate from event-on arrays and
CUDA stream-event intervals. Event intervals include descriptor upload and may
include host-feed gaps or scheduling; they are not isolated GPU execution or
bandwidth. All memory owners survive physical drain, including enqueue errors.
Unknown physical completion aborts the fixture instead of releasing memory.
Software/launch/outcome acceptance and application performance promotion are
separate gates; no new shader or backend default is selected by this fixture.

Scoped H20 outcome: three separately launched processes at source `ec8fe990`
pass all byte/guard controls and normal exit/known cleanup checks. The complete
cohort contains 21,600 measured submissions and 75,600 raw clock values, with
all 100 samples retained per backend/event mode. Independent outcome review
accepts the descriptive comparison only. Source CI run `38126055714` passes
13/13 jobs.

In event-off D2H, kernel wall medians are lower in all three processes for all
nine representative layouts. The 64-descriptor/64-range case is 96.210 us versus
237.048 us (median of three process medians); CPU submission is 6.302 us versus
75.633 us. Frozen 34-batch shape weights estimate 7.584–7.608 ms for kernel and
15.574–15.635 ms for DMA. These are representative estimated copy costs,
not measured model runtime. Synthetic H2D has DMA-faster counterexamples at
192/64, 448/192 and 512/192 descriptors/ranges, so direction/layout cannot be
collapsed into a global kernel preference.

Event-on/off observations are separate: the smallest D2H kernel layout has a
1.078 median wall ratio. Stream-event intervals include descriptor upload and
host-feed/scheduling gaps. This does not qualify instrumentation overhead or
isolated GPU execution. Physical pinned NUMA residency, physical fault drain,
inference contention and TTFT/TPOT/native-HBM/LMCache comparisons remain open.
The actual production shader and backend defaults are unchanged.

Definition, rejected launch controls, corrected freeze, complete cohort and
per-process statistics are preserved at
`/root/orbitkv-artifacts/s5-kda-layout-comparison-20261011/`;
independent outcome acceptance is at
`/root/orbitkv-artifacts/s5-kda-layout-comparison-outcome-review-20261011/REVIEW.md`.

## Engine-local raw H2D capture

Manager save observations do not cover raw restoration performed inside the
engine process. `LocalRestoreExecutor` captures that actual submission path when
`ORBITKV_TRACE_TRANSFERS=1` and `ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT` is positive.
The existing 65,536-unit capture ceiling applies to each executor. Its conservative
reservation charges two geometry units per descriptor plus one per part; this
bounds retained descriptors/parts rather than claiming an exact memory-byte cap.
It is default off.
The chosen backend, existing device-address sorting, layer-ready events, final
enqueue notification and physical stream drain stay at their existing boundaries.

Each submitted batch records ordered descriptors, the existing CopyDesc device
owner and host-registration tags, mapped host GPU addresses, granted source
allocation bounds and registered destination spans. The device owner tag is the
retained registered tensor/view address; it is not a driver query of the complete
CUDA allocation base. A registered tensor span can be smaller than its physical
CUDA allocation, so neither full allocation size nor distinct physical allocation
count is inferred. Range counts use the same private production DMA compiler and
actual maximum pitch.
CPU submission is recorded per batch; stream waiting is recorded once for the
whole part, including its layer events. Neither isolates GPU execution time.

The executor publishes a schema-2 opening record before each part's submission,
then serializes its bounded scalar geometry after the existing physical stream
drain. `local_raw_copy_complete` checkpoints all records through that drained
part under the same retained executor identity. It does not seal executor
lifetime or whole-operation completion. Publishing frees that part's buffered
geometry without replenishing the cumulative record budget. JSON/logging is
diagnostic overhead outside the backend submission and stream-wait intervals;
it can affect opt-in operation latency.
Normal engine/Manager exit, byte correctness and owned-resource postflight must
be checked independently. Capacity/allocation failure or unfinished/failed parts
invalidate the capture rather than supplying truncated successful evidence.
This diagnostic has no instrumentation-overhead or application qualification yet.

The first installed H20 dense structural pilot is `INVALID_FAIL_STOP`. Native
control requests finish, but the OrbitKV child starts from the source package
directory and shadows the installed wheel before any H2D submission. The test
helper correction uses a neutral parent working directory on both launch paths;
relative PYTHONPATH entries are interpreted against the caller directory after
environment overrides, without changing explicit search order or absolute path
text. Real CPU module-child controls cover installed/source precedence, relative
entries, overrides, empty entries and symlink/parent components. Independent
review accepted the failure evidence and rejected an intermediate correction
that changed relative-path meaning; its controls and rejection remain retained.
The final import helper is independently accepted. A separate lease-precheck
task stops before model admission, and its successor returns 6 passed/1 skipped
on Qwen3-32B with compilation enabled and CUDA Graph disabled. The successor
still fails collection: both engine logs contain zero records because the
Drop-only recorder has no reliably consumed publication boundary before the
released API owner's immediate EngineCore abort. All three whole failed runs,
warnings and final postflight evidence remain immutable. The separate
drained-part publication repair includes a real CPU child that exits without
destructors: the old recorder produces zero records and fails that regression.
Missing/failed checkpoints and cumulative budget/registration controls remain
strict. The separate repair at `37b48c50` now has scoped installed H20 acceptance:
one official-vLLM 0.31.0/Qwen3-32B DRAM TP1/PP1 structural task returns six passes
and one expected hybrid-only skip in 449.48 s. Actual execution uses Inductor
compilation and CUDA Graph `NONE`. Generated text matches the native control;
restart cache loads and four normal outer API/Manager cleanup receipts pass.
Independent outcome review accepts the complete capture after verifying the
full archive and rerunning the public parser with the frozen native compiler.

Five drained parts from one retained executor contain 320 direct H2D batches,
2,048 descriptors, 1,984 compiled DMA ranges and 160 MiB. The five actual shape
families each have 64 batches, with descriptor/range pairs 3/3, 5/5, 7/7, 8/8
and 9/8. The histogram includes 64 KiB, 256 KiB and 384 KiB descriptors. These
small H2D batches differ from the representative D2H layouts above; the earlier
native comparison cannot select a backend for this path.
Recorded CPU submission totals 3.690573 ms; the five shared whole-part host
waits total 6.128049 ms. They are descriptive instrumented intervals, without an
independent-run confidence interval, overhead qualification, isolated GPU time
or inference-performance claim. Registered tensor/view tags remain registration
bounds, not queried complete CUDA allocation bases/counts.

All 335 opening/batch/part/checkpoint records and the cumulative 4,101/65,536
reservation validate. The complete mirror retains all 1,392 regular files plus
original symlink metadata/TAR. Guardian/SSH wait and final known-process/GPU
cleanup pass; official inner abort/force-kill/semaphore warnings remain S3/IPC
lifetime limits. No separate full raw GPU KV tensor-byte oracle is provided.
The production shader/backend defaults and support boundaries remain unchanged.
Definitions, earlier failures, frozen inputs, complete cohort and descriptive
analysis are at
`/root/orbitkv-artifacts/s5-h2d-shapes-latest-models-20261011/`;
independent outcome acceptance is at
`/root/orbitkv-artifacts/s5-h2d-capture-part-publication-outcome-review-20261012/REVIEW.md`.
See the
[S5.5 execution record](completion-plan.md#s55--deployment-matrix-and-upstream-maintenance).

After an independently reviewed installed model gate and normal shutdown, parse
the retained engine log into a new external output:

```bash
python -m benches.local_restore_shapes \
  --engine-log /var/tmp/orbitkv-kda/h2d-001/engine.log \
  --native-fixture /var/tmp/orbitkv-kda/frozen/native-copy-layout \
  --output /var/tmp/orbitkv-kda/h2d-001/local-h2d.json
```

The parser requires every opened part to have a matching complete checkpoint,
contiguous batch/part coverage, source and destination bounds, nonoverlapping
destinations, exact byte histograms and the cumulative bounded reservation.
Part sequences must be contiguous; executor device, maximum pitch, budget and
source/destination registrations must remain stable across checkpoints. Schema-1
Drop-only output is not accepted as an alternative input.
The frozen native fixture then recomputes every batch's DMA ranges using captured
addresses, owners and maximum pitch through the production compiler, before any
CUDA initialization. Scalar-only parsing is explicitly unqualified; the final
output requires a matching compiler receipt and records the fixture/input hashes.
Arena/tensor registration spans stay consistent across the entire executor trace.
Empty managed/codec parts remain visible but do not count as raw H2D geometry. Request
or source-operation linkage, physical failure drain, pinned NUMA residency and
native-HBM/OrbitKV/LMCache model performance remain separate qualifications.

## SGLang model-pair qualification

The existing mapped-host KernelBackend has independently accepted serial
Qwen3-32B recovery evidence on H20 with official SGLang 0.5.21. Harness
`4b6c7eba` adds a released-pool full-byte oracle and frozen four-pair evaluation,
consuming the unchanged `37b48c50` native build. Four byte cells cover both
backends and 4K/8K across twelve restore generations, including partial targets,
all registered payload/sentinel bytes and deliberate corruption controls.
Nine unprofiled model cells contain a fresh native control and four independently
restarted AB/BA/AB/BA DMA/kernel pairs, totaling 270 measured requests.

The frozen 8K benefit and five cold/HBM/4K regression guards pass: 8K pressure
TTFT improves by a pair-geometric 10.45%, with a 95% interval of 9.69%–11.08%;
4K remains near parity. [Performance](performance.md#independent-sglang-kernel-pairs)
defines the statistic and limits. This qualifies a retained explicit backend
direction in the recorded serial profile. It adds no production shader, transfer
policy or default and does not infer a specific submission-path cause.
Native child `-9`/IPC warnings and historical missing waits remain lifetime
limits; computed model KV, contention, p99, Graph and broader ranks remain open.
Complete frozen inputs and independent reviews are in
[the archive](benchmark-evidence.md#sglang-independent-kernel-pair-archive).

## Earlier A100 component qualification

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

## H20 native CPU attribution (2026-10-11)

**Independently accepted within the native fixture; no candidate promoted.**
Exact `d001b4e2` consumes the production Rust backends and shader on H20 SM90.
Three independent processes retain 102 case/direction rows and 20,400 backend
submissions, with full byte/source/guard checks before each backend's warmup.
All native, guardian, controller and SSH exits are zero and reaped without forced
cleanup. Actual context UUID, executable and mapped runtime inputs match the
freeze; final GPU memory/processes are empty and the lease is free. Numeric
NVML PID visibility is recorded separately and does not prove PID namespace
mapping. Earlier invalid attempts and the rejected observation guard are retained.

The reversed synthetic H2D fixture with 4096 4 KiB fragments spends about 75% of
DMA time in CPU submission; its ascending counterpart spends about 6%. These
fixtures call the backends directly with synthetic owner IDs. Production worker
and local descriptors already sort by device address. Capture actual recovery
layouts and consume existing raw-copy cost observations, including compiled DMA
range counts, before selecting a candidate; do not add duplicate sorting.

Statistics use every sample and type7 per-run quantiles. Three-run observed
min/max ranges are descriptive, not confidence intervals. CPU submission includes
packing, upload and driver work; host synchronization is not device-only time or
bandwidth. Byte controls do not run on every timed repetition. Physical pinned
NUMA residency, transfer faults, compute contention, model benefit and S2/S5/S3
completion remain unqualified. The full independent result and verified raw-data
references are external at
`/root/orbitkv-artifacts/s5-kda-native-attribution-r3-outcome-independent-review-20261011/REVIEW.md`.

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

## Native driver submission diagnosis

`benches/cuda_driver_trace.cpp` is a benchmark-only CUPTI DRIVER_API collector.
It retains at most 262,144 fixed records, formats them after the single-process
fixture returns, and never changes CUDA arguments, return values or memory owners.
Incomplete pairs, overflow and output failures invalidate capture. It does not
support inherited fork children or simultaneous profiler subscriptions.

The existing `native_copy_layout` fixture emits optional `CLOCK_MONOTONIC`
submission/drain windows when `ORBITKV_CUDA_DRIVER_TRACE_DIR` is nonempty.
`--case` selects one disclosed layout for a bounded smoke. Both directions,
DMA/kernel, event-off/event-on and the existing full source/payload/guard controls
remain present; the production backends and kernel are unchanged.

Build the collector against the target runtime's CUPTI ABI and freeze its native
dependencies together with the fixture before an independently reviewed launch:

```bash
g++ -O2 -std=c++17 -Wall -Wextra -Werror -fPIC -shared \
  -I/path/to/cuda/include benches/cuda_driver_trace.cpp \
  -L/path/to/cuda/lib -lcupti -o /external/diagnosis/cuda_driver_trace.so
cargo build --release -p orbitkv-core --bench native_copy_layout \
  --no-default-features --features cuda-13
mkdir /external/diagnosis/trace
LD_PRELOAD=/external/diagnosis/cuda_driver_trace.so \
ORBITKV_CUDA_DRIVER_TRACE_DIR=/external/diagnosis/trace \
  /path/to/native_copy_layout --case d192-r64 --samples 4 \
  --output /external/diagnosis/fixture.jsonl
python -m benches.cuda_submission \
  --trace /external/diagnosis/trace/cuda-driver-PID.jsonl \
  --fixture /external/diagnosis/fixture.jsonl
```

Use the actual fixture PID in the trace filename and resolve the target's CUPTI
SONAME through its frozen loader environment. CPU callback/output controls live
in `benches/tests/cuda_driver_trace_writer.cpp`; compile with the same include
directory, without `-lcupti`, and run `valid`, `unpaired`, `overflow`, `duplicate`
and the `orphan-exit`, `natural-duplicate`, `sealed-pair` controls in separate new
trace directories. Only `valid` and the rejected post-seal pair must exit zero;
every incomplete/duplicate/overflow capture must fail. Analyzer units
run with `pytest benches/tests/test_cuda_submission.py`.

The analyzer requires one complete matching process, thread and directional
fixture, checks copy/launch counts against the compiled geometry, and reports
driver interval unions and remaining unattributed host wall time. Remaining time
is not proven descriptor packing; callbacks themselves perturb API duration.
This diagnostic does not isolate GPU execution or forward admission and cannot
promote performance. Consumed engine capture and matched observation-overhead
controls remain separate gates.

The `4b568860` H20 functional smoke uses `d192-r64`: 192 descriptors, 64 compiled
DMA ranges and 12 MiB of payload. Both directions, both backends and both event
modes complete 80 submissions with 32 measured windows; complete byte/source/guard
controls, exact copy/launch counts, normal owned-process waits and empty physical
postflight pass independent review. The capture contains 5,017 driver records
with a complete footer.
Four profiled samples per group do not qualify overhead or performance. In
particular, lower CPU submission time does not establish lower complete restore
latency. Preserve this capture while evaluating matched unprofiled controls and
the consumed engine path. The [private archive](benchmark-evidence.md#native-driver-capture-archive)
retains frozen inputs, rejected launch controls and raw outputs.


## Consumed engine driver attribution

Engine-local raw H2D geometry records additionally carry `CLOCK_MONOTONIC`, Linux
thread IDs and batch submission windows; each physically drained part supplies
one shared drain timestamp. They are emitted only by the existing opt-in bounded
capture. Transfer backends, CUDA kernels, readiness and allocation owners are
unchanged. Instrumentation overhead remains a separate gate.

After an independently reviewed capture, join one process's driver trace to its
actual engine batches through the existing production geometry validator:

```bash
python -m benches.cuda_submission \
  --trace /external/capture/cuda-driver-PID.jsonl \
  --engine-log /external/capture/engine.log \
  --native-fixture /path/to/native_copy_layout
```

The analyzer validates the clock on every part header and requires copy/launch
API families matching the consumed backend and direction. A 2D API name alone
does not attest its arguments; the production geometry and exact-byte controls
remain required. It rejects incomplete records, mismatched clocks/processes/threads,
wrong copy/launch counts, calls crossing batch boundaries and submissions after
their declared part drain. Its driver union includes callback overhead; remaining
host time is unattributed. Device identity, exact GPU bytes/model output, normal
owned-process cleanup and matched overhead controls require their own receipts.
The collector's inherited-fork and simultaneous-profiler restrictions still apply.
A successful CPU compiler replay does not qualify GPU capture or a kernel default.


The 2026-10-14 H20 functional cell uses the final installed wheel built from
`f6d4bff2`, official SGLang 0.5.21 and fixed Qwen3-32B pool geometry without a
model forward. Three sequential parts verify 12,935,233,536 total pool bytes,
including 2,902,458,368 restored bytes, sentinel pages and corruption controls.
Independent reconstruction matches every plane hash; replay validates all 384
batches and 22,144 `cuMemcpyHtoDAsync_v2` calls against the native compiler and
same-thread monotonic windows. The complete collector contains 42,224 calls,
including operations outside those restoration windows. Oracle, Manager and
controller all exit zero without forced cleanup; input/runtime hashes remain
unchanged. This is functional attribution, with no overhead or performance
promotion. Full traces, controls, preceding invalid CLI launch and independent
reviews remain outside the checkout at
`/root/orbitkv-artifacts/s4-s5-runtime-completion-20261014/`, linked from
[PR #238](https://github.com/feichai0017/orbitkv/pull/238).


## Matched observation overhead pilot

The next bounded S4 gate uses the maintained installed-wheel byte oracle with
`--warmup 1 --samples 2`. This optional mode repeats the same full 4K restore
geometry and deterministic payload; the default three-generation byte gate
retains its full/partial coverage. The three full restores consume 48,387
trace-accounted records under the unchanged 65,536 cap; independent prelaunch
review rejected the original five-restore proposal. Each operation checks every payload/sentinel
byte and corruption control, with distinct native session/operation identities.

Symmetric host monotonic observations begin after target initialization and
synchronization, record all-enqueued acknowledgement, and finish after successful
terminal `wait_restore`. Full byte checks and Manager metric requests occur
outside this interval. It includes Python/native return boundaries and does not
measure device-only time or first-layer readiness.

Freeze four fresh off/on process pairs in alternating order, keeping the runtime,
geometry, payload, CPU placement and Manager budgets equal. On cells enable both
raw geometry and the CUPTI collector; off cells enable neither. Validate complete
captures against the actual compiler, PID and operation windows, and require
explicit unregistration/close plus normal component and Manager waits. Keep
every slow sample and failed cell.

```bash
python -m benches.sglang_kv_oracle --model-config /path/to/frozen-config.json \
  --length 4096 --backend direct --warmup 1 --samples 2 \
  --manager-socket /path/to/owned.sock --manager-url http://127.0.0.1:20642 \
  --gpu-uuid GPU-full-physical-uuid --output /external/pilot/cell/bytes
python -m benches.copy_observation --contract /external/pilot/contract.json \
  --directory /external/pilot/cells --native-fixture /path/to/frozen-validator
```

Every pair median ratio and the independent-pair bootstrap 95% upper bound must
be at most 1.05 before a separate inference-contention contract can open. Warm-up
samples remain archived and are explicitly excluded from the two measured
samples. Requests are not bootstrap replicates. A failed overhead guard blocks
further diagnosis; this small pilot cannot qualify p99, serving gains, fault
reclamation or a new backend default. GPU0's failed-run reservation remains
untouched, and a new card needs an independent launch/resource review.


The single reviewed H20 pilot on 2026-10-15 stopped after five of eight cells.
Each completed cell passed its three full payload/sentinel/corruption checks,
explicit native unregistration/close and normal SDK/Manager waits. However, both
on-cell directories also contain an unexpected child-process capture. Their
complete zero-call footers do not establish the children's launch lineage or
satisfy the frozen exact inventory. The external controller validated only the
main file before marking each cell complete, so this admission check also needs
repair. The final maintained analyzer rejects the foreign file.

The cohort is `INVALID_COHORT_FAIL_STOP`, with observation overhead
`NOT_EVALUATED_INVALID_COHORT`; no pair ratio, CI, p99 or performance acceptance is
published. All main/child captures are preserved; cells 05–07 did not launch.
The external handoff remains at
`/root/orbitkv-artifacts/s4-copy-observation-overhead-20261014/`.
GPU1 was released after explicit terminal native work and normal process waits;
no forced native cleanup or retry occurred. The independently observed unreaped
CPU-only supervisor zombie remains explicit in the postflight evidence; cleanup
acceptance covers the successfully drained native owners. A new freeze needs an independently
reviewed process-scope and per-cell inventory fix before another cohort. The
runtime, installed wheel, descriptor compiler and collector were not changed by
this pilot, and it promotes no backend or kernel.


### Main-process capture repair

The controlled released-pool oracle now accepts `--driver-collector` with an
absolute regular library path. It loads that library before Torch, SGLang or
OrbitKV imports and keeps the handle through native completion. Its result
records explicit loading, its own PID and the library path/SHA256. The controlled
oracle rejects `LD_PRELOAD`, environment-only activation, a missing/nonempty or
symlink trace directory, and preinitialized engine/native modules.

```bash
mkdir /external/new-pilot/cell/driver
ORBITKV_CUDA_DRIVER_TRACE_DIR=/external/new-pilot/cell/driver \
ORBITKV_TRACE_TRANSFERS=1 ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT=65536 \
python -m benches.sglang_kv_oracle --model-config /path/to/frozen-config.json \
  --length 4096 --backend direct --warmup 1 --samples 2 \
  --driver-collector /external/frozen/cuda-driver-trace.so \
  --manager-socket /path/to/owned.sock --manager-url http://127.0.0.1:20642 \
  --gpu-uuid GPU-full-physical-uuid --output /external/new-pilot/cell/bytes
```

The `s4.copy-observation-pilot.v2` contract adds `collector` and
`collector_sha256`; new qualification consumes the explicit activation record.
Old v1 evidence remains archived with its frozen analyzer and cannot be relabeled.
The maintained `capture_file` and `validate_activation` checks are shared by the
external per-cell controller and full cohort analysis. Check the exact regular
main-PID file before and after replay, preserve the byte-result hash, then record
`CELL_VALID` or admit another cell. A complete zero-call foreign file is invalid.
Disabled mode requires absence even of a dangling trace-directory symlink.

CPU-only loader tests compile the actual collector with stub CUPTI and launch a
real exec child: explicit loading emits one main capture; the matched inherited-
preload control emits main and child captures. These tests do not initialize
CUDA or qualify hardware/descendant work. Collector C++, native wheel, production
copy paths, ownership and the original capture/5% overhead guards are unchanged.
The fresh H20 v2 cohort at `7601af23` completes all eight cells with exact
main-process captures, production-compiler replay and normal native/Manager
release. The frozen 5% overhead guard fails in every pair: the pair-geometric
ratio is 1.4133, with independent-pair 95% interval [1.4036, 1.4231]. This is the
combined tracing/collector overhead, not an attribution to either component.
Further inference contention and kernel/default promotion remain blocked; the
next bounded delivery is in [S4](completion-plan.md#s4--finish-communication-execution-and-demonstrate-gains).

The complete run is archived outside the checkout at
`/root/orbitkv-artifacts/s4-h20-observation-7601af23-r3-20261015/remote-run/`.
Its immutable run manifest SHA256 is
`a9c8a4c86f93314ba1564b320c8731b375b9fa2e29fa46db2b3a74d015c78746`.
The invalid first cohort, rejected prelaunch freezes and failed controls remain
preserved separately. S3/S4/S5 remain Partial; no serving, p99 or default
qualification follows from this pool-only result.
