# Benchmark evidence and artifact storage

Keep benchmark programs, reusable workloads, reproduction scripts and small
input fixtures in the repository. Raw responses, JSON/CSV measurements, manifests,
logs, traces, generated plots and per-run reports belong in an explicitly chosen
external directory or CI artifacts. Preserve unsuccessful runs and controls.
Public documentation contains reviewed conclusions, their limits and links to
versioned evidence; passing a correctness gate does not establish a speedup.

## Native driver capture archive

`/root/orbitkv-artifacts/s4-native-submission-gpu-20261014/` retains the native
submission cohort. Its [frozen index](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#native-driver-capture-archive)
records inputs, failed controls, hashes and clock limits. This is component
observation evidence, not a serving or default qualification.

## SGLang independent kernel-pair archive

`/root/orbitkv-artifacts/s5-kernel-layout-qualification-20261013/` retains the
reviewed SGLang 0.5.21 byte controls and four serial DMA/kernel process pairs.
The [frozen index](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#sglang-independent-kernel-pair-archive)
contains complete provenance, checksums and lifetime limits;
[performance](performance.md#independent-sglang-kernel-pairs) states the accepted scope.

## SGLang restore diagnosis archive

`/root/orbitkv-artifacts/s5-sglang-restore-profile-20261012/` retains profiler-off
controls, diagnostic traces and unsuccessful preparations. Use the
[frozen index](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#sglang-restore-diagnosis-archive)
for the original source, ordering, hashes and rejected cleanup attempt.

## Current RDMA component archive

`/root/orbitkv-artifacts/s5-manager-rdma-profile-20261012-r2/` retains the complete
Manager-only cohort; `/root/orbitkv-artifacts/s4-s5-tent-rdma-profile-20261012/`
retains the earlier incomplete campaign and GPU-pointer component data. Their
[frozen index](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#current-rdma-component-archive)
preserves individual runs, failed startup, reviews and checksums. These are
private host archives, not public downloads, HA or fault-reclamation qualification.

## SGLang concurrent admission archive

`/root/orbitkv-artifacts/s4-sglang-admission-20261021/` retains the isolated
one-line scheduler counterfactual and its independent review.
`/root/orbitkv-artifacts/s4-sglang-official-admission-20261021/` retains the
eight-cell official-release comparison, frozen contract, all raw requests,
cleanup, complete wheel/model hashes and process-pair analysis. These private
host archives are not public downloads. The [reviewed delivery](https://github.com/OrbitInfra/OrbitKV/pull/294)
and [performance summary](performance.md#sglang-concurrent-admission) state
the benefit and failed regression guards; no default is selected.

## Run and retain an experiment

Choose a new directory outside the source checkout for each run. Python harnesses
validate `--output` before starting services, including resolution of symlinks.
`benches.serving.sharegpt` requires `--output-dir`; `benches/serving/serving.sh` requires
`RESULT_DIR`. The GDS qualification uses a private directory on the explicitly
selected external `--ssd-dir` mount. Criterion measurements require an external
`CARGO_TARGET_DIR`. Build native libraries before starting any runtime gate.

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /path/to/immutable-model \
  --installed-artifact --enforce-eager \
  --output /var/tmp/orbitkv-bench/vllm-dram-001

python -m benches.common.report /var/tmp/orbitkv-bench/vllm-dram-001 \
  --output /var/tmp/orbitkv-bench/vllm-report-001
```

The [benchmark guide](../benches/README.md) defines workloads and matched controls.
The [preparation reproduction script](../benches/serving/reproduce_preparation.sh) accepts
an external output root as its first argument. Preserve source/engine/model
revisions, binary hashes, hardware and transport, budgets, commands, correctness
and drain evidence, uncertainty and all failed preparations with the result.
Use at least three order-alternated matched runs and declare thresholds before
claiming performance gains. Compare equal state coverage and completion targets.

In CI, select a directory such as `$RUNNER_TEMP/orbitkv-evidence`, write logs and
reports there, and upload it with `actions/upload-artifact@v4` using
`if: always()` so failures survive the job. Release qualification must identify
the exact installed artifact. CI retention expiry is not permanent publication;
copy evidence referenced by a release into versioned durable storage before expiry.

`python3 scripts/check-experiment-output.py` inspects the Git index in local
checks and CI, including force-added files. It reserves `benches/results/`,
`benches/runs/`, `results/` and `runs/` for generated output and rejects tracked
entries there. It deliberately allows deterministic fixtures under `tests/` and
source files whose names contain `result`; it does not infer content from such names.

## Serving comparisons

Compare backends inside each engine first. The selected profiles are official
vLLM 0.31.0 with the V1 runner and one attention group, and official SGLang
0.5.22. Historical comparisons retain their measured 0.5.21 release. Use dense
Qwen3 BF16, TP=1/PP=1 for the initial matrix; broader rank,
hybrid, P/D and Graph profiles need their own qualification. Use separate clean
installed environments and freeze the resolved dependencies and all installed
engine/cache files before and after each run. `--installed-artifact` keeps source
adapters out of the child import path; it does not install or qualify dependencies.
Use `--enforce-eager` for eager controls: it selects vLLM's released eager flag
and disables both SGLang Graph phases through their released configuration.
Omitting it retains each engine's default execution policy, recorded separately
in the manifest. Run the benchmark from a Git checkout so the recorded revision
identifies the consumed harness. Each service emits a `*.cleanup.json` containing
the reaped leader's exit code and whether SIGKILL was needed; also check descendants,
ports, sockets and GPU owners before accepting a run. A reaped leader alone does
not establish that its workers or exported allocations have drained.

| Backend | Purpose | Capacity control |
| --- | --- | --- |
| `native` | Engine HBM prefix-cache reference | Same GPU KV bytes; no host cache |
| `cpu` | Released vLLM OffloadingConnector or SGLang HiCache | Same GPU KV bytes and host budget |
| `orbitkv` | Installed wheel and bundled Manager | Same GPU KV bytes and host budget; DRAM first |
| `lmcache` | Released LMCache MP connector and server | Same GPU KV bytes and host budget; DRAM first |

The LMCache target is
[0.5.5](https://github.com/LMCache/LMCache/releases/tag/v0.5.5), commit
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0`; validate the selected engine/PyTorch
combination before measurement. Its
[released SGLang MP example](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/examples/sgl_integration/README.md)
documents the server/configuration boundary. A launcher option or successful
import does not prove model serving or compatible native kernels.

The historical SGLang 0.5.21 MP integration imports
`lmcache.integration.sglang.lmcache_mp_metadata`, absent from the released
LMCache 0.5.5 wheel. That official startup control was blocked. The
[current upstream quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)
requires the unified MP changes in
[SGLang #38652](https://github.com/sgl-project/sglang/pull/38652) and
[LMCache #4828](https://github.com/LMCache/LMCache/pull/4828); the latter merged
after the 0.5.5 release. A new comparison needs its own consumed official
engine/cache release and startup control. Do not patch installed engines
or substitute an upstream main checkout for a release comparison.

Freeze model/tokenizer content hashes, tokenized prompts, output length, seed,
arrival/concurrency pattern, page/chunk size, GPU KV bytes, host/SSD capacity,
NUMA placement and eager/Graph mode. Record actual effective cache capacity,
staging, scratch, pinned ownership and all cache-server processes in addition to
configured budgets. Preserve native HBM hits; an external miss and a resident
HBM hit do not perform equal work. Start with cold, HBM-hot, external full/partial
hit, cache miss and post-pressure cases. Add independent engine restart and
DRAM-evicted io_uring SSD restores only after their byte/output controls pass.
Native GDS remains a separate deferred qualification.

Use concurrency 1/4/8 and at least three, preferably five, order-alternated
independent runs per matched profile. Declare warmup, sample count, stop rules,
SLOs and acceptance thresholds before launch. Bootstrap independent run pairs,
not correlated requests inside one run. Retain slow samples and report absolute
latency differences alongside ratios and uncertainty. Report TTFT, end-to-end
latency, completed requests/tokens per second, SLO goodput, CPU/pinned-memory
cost, cache-source bytes and physical I/O. The current single-node streaming
driver times first returned text and completion. Sustained reports include a
request-level mean decode-time-per-token proxy; per-token ITL distributions and
SLO goodput instrumentation remain open and cannot be inferred from that proxy.
Output length and text are retained; require the engine's native deterministic
output control and the cache byte/lifetime gates before any speedup claim.

For two-host shared-cache serving, publish on one host and restore in a fresh
consumer on the other. Require zero consumer DRAM beforehand, positive remote
and GPU-copy bytes, matching output, exact source identity and final reservation
drain. The native consumer recomputation control has the same model and request;
a same-host HBM hit is a separate best-case reference. Match global cache
coverage, replica count and total memory, including storage-server segments and
staging, for a distributed LMCache comparison. The documented
[LMCache Mooncake Store backend](https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html)
uses a storage service as well as transport. Verify that the released MP
integration consumes the chosen backend before comparing it with OrbitKV;
that documentation does not establish distributed MP support on these pins.

## TENT and fabric diagnostics

Keep bare TENT measurements separate from full Manager recovery and engine
serving. Freeze the Mooncake commit, wrapper, libraries and configuration. Measure
READ/WRITE for host-to-host, host-to-GPU and GPU-to-GPU payloads; record exact
device/host pointer registration, bytes, placement, source ownership and drain.
Distinguish first segment discovery/connection and memory registration from
preheated transfer. Test packed and fragmented layouts, message-size crossover,
one/two/four NIC selections, NUMA placement and concurrent clients/GPU pairs.
Registration or a transport's internal counter alone does not prove physical
GPUDirect RDMA; retain NIC counters, payload checks and negative route controls.

Current first-READ diagnosis uses the owned Manager cells in
`benches.distributed.rdma_profile` and the installed Rust client. The retired
[CPU-mmap probe](https://github.com/OrbitInfra/OrbitKV/blob/800a755fe88e119cd10b19c28c27e02d78d40d09/benches/distributed/tent_stage.py)
uses an upstream 48-byte ABI, while the current production request is 88 bytes.
Reproduce historical evidence with its frozen source and libraries; do not use
that probe with a production runtime. Its original private-buffer timing does
not establish Manager/model latency, GPUDirect RDMA or crash reclamation.

For the pinned native build, the benchmark-only `tent_native_trace.cpp` records
metadata RPC, endpoint construction, connect and bootstrap RPC without changing
their arguments, results or transport policy. Its exact C++ ABI, bounded event
array and trace-off/on overhead pilot are part of the frozen input contract.
`benches.distributed.tent_trace` validates process identity, event completeness and consumer
READ boundaries offline. Overlapping calls do not form an additive latency
breakdown; compare timestamps only within one host. Missing events, failed output
or failed native drain invalidate the diagnosis. See the build and protocol in
[the benchmark guide](../benches/distributed/README.md#native-tent-first-read-diagnostic).

Separate experiments with existing RPC-thread and RDMA-lane configuration need
fresh endpoints, one changed knob, matched ordering, full bytes and explicit
warm-read regression guards. Account connection warming as setup work and record
when it is consumed; moving it before a timer does not prove an end-to-end gain.
Upstream already has an open [segment warmup proposal](https://github.com/kvcache-ai/Mooncake/pull/3685)
and a merged [duplicate-bootstrap correctness fix](https://github.com/kvcache-ai/Mooncake/pull/1705).
These are reference material, not newly qualified OrbitKV capabilities.

The selected `v0.3.13.post1` source still has 256 notification receive MRs per
RDMA endpoint. Upstream merged [PR #3820](https://github.com/kvcache-ai/Mooncake/pull/3820)
after that release to use one receive MR covering the same 256 slots. An
isolated comparison should rebuild both the release baseline and only that
three-file repair with identical compiler/configuration/dependencies. Verify
all other shared libraries are identical, test slot adjacency and wraparound,
and retain release cleanup ownership when adapting source context. The retired
[comparison controller](https://github.com/OrbitInfra/OrbitKV/blob/800a755fe88e119cd10b19c28c27e02d78d40d09/benches/distributed/tent_compare.py)
is retained only at that immutable revision for historical reproduction. Source
deltas, hashes, input readiness failures, valid smoke and formal results belong
outside the checkout. Neither upstream's own
measurements nor a private-buffer result qualifies OrbitKV's consumed path.

NVLink is a same-host GPU fabric. Hardware link status is readiness evidence,
not measured payload bandwidth or proof that TENT selects a local path. Verify
the available pinned TENT transport first, then record the actual route and
fabric counters. Treat a local CUDA peer-copy ceiling separately from TENT's
consumed path. Cross-host traffic remains on the network. For mixed inference,
compare isolated cache READ with native P/D WRITE and NCCL traffic on shared and
separate NICs; retain decode ITL, TTFT, goodput and GPU/PCIe/CPU/fabric pressure.
This requires the serving instrumentation above, not an extrapolation from a
GPU-to-GPU microbenchmark.

The pinned
[TENT NVLink implementation](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-transfer-engine/tent/src/transport/nvlink/nvlink_transport.cpp)
has CUDA IPC export and asynchronous completion; OrbitKV enables its native
transports unless TCP is forced. Test cudaMalloc-backed tensors, suballocations,
VMM allocations, mixed-device batches and non-default producer streams
separately. The pinned path skips IPC export for VMM memory, so an enabled
transport can still be ineligible for a particular allocation. Preserve the
producer-readiness edge and device/source lifetime when investigating its
caller synchronization cost; a faster copy that races a producer is invalid.

The 2026-10-10 CUDA API triage on both H20 hosts verifies 2 MiB of valid VMM
bytes, successful cudaMalloc IPC export and successful VMM POSIX-handle export;
legacy IPC export of VMM returns `invalid argument`. This tests the allocation
API boundary, not TENT payload routing. Skipping VMM in the legacy IPC path is
an explicit eligibility restriction, not a newly reproduced transport defect.
The existing expandable-segments registration
[issue #2511](https://github.com/kvcache-ai/Mooncake/issues/2511) is closed by
[PR #3538](https://github.com/kvcache-ai/Mooncake/pull/3538), which fixes the
DMA-BUF export range and addresses a different RDMA registration path.

The currently selected official `v0.3.13.post1` source does not contain the
merged NVLink fixes for
[peer device ordinals (#3678)](https://github.com/kvcache-ai/Mooncake/pull/3678)
and [shared suballocation IPC handles (#3679)](https://github.com/kvcache-ai/Mooncake/pull/3679).
Validate those shapes before qualifying the local NVLink profile. A separately
frozen upstream-main control can assess the existing repairs; it does not
replace released-runtime evidence or authorize a default dependency change.
Keep the original RDMA cohort and runtime untouched. Source snapshots, upstream
records, probe source, results, library hashes and empty GPU postflight are at
`/root/orbitkv-artifacts/forge-rdma-gds-20426175-20261009/tent-vmm-triage-20261010/`.

Investigate cold READ segment discovery, metadata fetch, QP setup, memory
registration and progress/completion independently before labeling a TENT bug.
Reproduce a suspected upstream issue outside the Manager wrapper on the pinned
baseline, check existing upstream issues/PRs, and retain a matched candidate
with exact bytes, partial-submit, cancellation, peer-loss, delayed completion
and final drain controls. A wrapper polling delay belongs in OrbitKV; an
upstream transport repair needs its own reproducer and regression evidence.
Use KDA for a measured GPU packing/scatter or codec bottleneck, with device-event
and actual serving controls; it cannot optimize an RPC or connection delay.

## Historical tracked evidence

Formerly tracked collections remain at the immutable
[September 29 snapshot](https://github.com/OrbitInfra/OrbitKV/tree/9fe1441c0d7d4c47b1914c303f837bba9f4a758f/benches/results),
[September 21 snapshot](https://github.com/OrbitInfra/OrbitKV/tree/44c1e5f9a253aa7378c6187b2aeea9bff93df304/benches/results)
and [recovery snapshot](https://github.com/OrbitInfra/OrbitKV/tree/4712f780c900120719f178b2ea36c9e0ac7c135f/benches/results/20260922-recovery-baseline).
Original tables and migration details are in the
[frozen evidence index](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#historical-tracked-evidence).
Git history is unchanged. Website link tests require full history; shallow
checkouts need `git fetch --unshallow` before `npm test`.

## Local archive and verification

The selected S1 archive is `/root/orbitkv-artifacts/s1-evidence-20260929/`.
`archive-manifest.json` records source revisions, paths, sizes and hashes;
`SHA256SUMS` verifies regular files. Its manifest SHA-256 is
`55d5155a24a730c6053db91870e0a9515d35e026e18565989b9f0e0bd2f66db3`.

```bash
cd /root/orbitkv-artifacts/s1-evidence-20260929
sha256sum --check SHA256SUMS
```

The [frozen inventory and relocation record](https://github.com/OrbitInfra/OrbitKV/blob/81bf353a416189b1dd139a18b57c8578184ecda4/docs/benchmark-evidence.md#local-archive-and-verification)
identifies the archived bytes, relocated symlinks and older missing raw logs.
This private archive is not an off-host backup or public download. Publish new
charts from durable versioned artifacts and qualify every new performance claim.
